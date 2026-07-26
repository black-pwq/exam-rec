from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pytest

from app_logging import APP_LOGGER_NAME, configure_logging
from extractor.base_extractor import Problem
from extractor.evaluator import EvaluationReport
from pipeline import PageProcessingResult
from recognition_jobs import (
    JobNotCompleteError,
    JobNotFoundError,
    JobQueueFullError,
    JobStore,
    LlmSettings,
    RecognitionService,
    ServiceAlreadyRunningError,
)


def create_job(store: JobStore, content: bytes = b"input") -> str:
    job_id = store.new_job_id()
    upload = store.prepare_upload(job_id)
    upload.write_bytes(content)
    store.create(job_id, original_filename="input.pdf", page_count=2)
    return job_id


def page_result(page_index: int, number: str) -> PageProcessingResult:
    return PageProcessingResult(
        page_index=page_index,
        problems=(
            Problem(
                number=number,
                question=f"question {number}",
                answer="",
                options={"A": "first", "B": "second"},
                analysis="",
            ),
        ),
        extractor_name="StubExtractor",
        evaluation=EvaluationReport(score=1.0),
    )


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not met before timeout")
        time.sleep(0.005)


@pytest.fixture
def app_caplog(caplog, monkeypatch):
    monkeypatch.setenv("EXAM_REC_LOG_LEVEL", "INFO")
    configure_logging()
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    previous_level = caplog.handler.level
    caplog.handler.setLevel(logging.INFO)
    app_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        app_logger.removeHandler(caplog.handler)
        caplog.handler.setLevel(previous_level)


def test_job_store_persists_cursor_updates_and_result(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job_id = create_job(store)

    store.update_status(job_id, "running")
    store.append_event(job_id, "started")
    store.record_page(
        job_id,
        page_result(3, "1"),
        processed_pages=1,
        problem_count=1,
    )
    store.complete(
        job_id,
        [
            {
                "number": "1",
                "question": "question 1",
                "answer": "",
                "options": {"A": "first", "B": "second"},
                "analysis": "",
            }
        ],
    )

    first = store.get_updates(job_id, after=0, limit=2)
    assert [event["type"] for event in first["events"]] == ["queued", "started"]
    assert first["next_cursor"] == 2
    assert first["has_more"]

    remaining = store.get_updates(job_id, after=first["next_cursor"])
    assert [event["type"] for event in remaining["events"]] == [
        "page",
        "completed",
    ]
    assert not remaining["has_more"]
    assert remaining["events"][0]["page_index"] == 3
    assert store.get_status(job_id)["processed_pages"] == 1
    assert store.get_result(job_id)["problems"][0]["number"] == "1"


def test_job_store_marks_unfinished_jobs_interrupted(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job_id = create_job(store)

    store.mark_unfinished_interrupted()

    assert store.get_status(job_id)["status"] == "interrupted"
    assert store.get_updates(job_id)["events"][-1]["type"] == "interrupted"
    with pytest.raises(JobNotCompleteError):
        store.get_result(job_id)


class ControlledProcessor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def process_iter(self, path: Path):
        label = path.read_bytes().decode()
        self.calls.append(label)
        if label == "first":
            self.first_started.set()
            if not self.release_first.wait(2):
                raise TimeoutError("first job was not released")
        yield page_result(0, label)


def test_service_processes_whole_jobs_in_fifo_order(
    tmp_path: Path,
    app_caplog,
) -> None:
    store = JobStore(tmp_path)
    processor = ControlledProcessor()
    service = RecognitionService(store, lambda: processor)
    service.start()
    first = create_job(store, b"first")
    second = create_job(store, b"second")
    try:
        service.submit(first)
        assert processor.first_started.wait(1)
        service.submit(second)
        time.sleep(0.02)
        assert processor.calls == ["first"]

        processor.release_first.set()
        wait_until(lambda: store.get_status(second)["status"] == "completed")

        assert processor.calls == ["first", "second"]
        assert store.get_result(first)["problems"][0]["number"] == "first"
        assert store.get_result(second)["problems"][0]["number"] == "second"
    finally:
        processor.release_first.set()
        service.stop()

    messages = [record.getMessage() for record in app_caplog.records]
    assert any(f"Recognition job queued job_id={first}" in message for message in messages)
    assert any(f"Recognition job started job_id={first}" in message for message in messages)
    assert any(
        f"Recognition job completed job_id={first}" in message
        and "processed_pages=1" in message
        and "problem_count=1" in message
        and "duration_seconds=" in message
        for message in messages
    )
    assert any("Recognition service stopped job_root=" in message for message in messages)


def test_delete_removes_a_queued_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    processor = ControlledProcessor()
    service = RecognitionService(store, lambda: processor)
    service.start()
    running = create_job(store, b"first")
    queued = create_job(store, b"second")
    try:
        service.submit(running)
        assert processor.first_started.wait(1)
        service.submit(queued)

        assert service.delete(queued)
        processor.release_first.set()
        wait_until(lambda: store.get_status(running)["status"] == "completed")

        with pytest.raises(JobNotFoundError):
            store.get_status(queued)
        assert processor.calls == ["first"]
    finally:
        processor.release_first.set()
        service.stop()


def test_service_logs_running_job_cancellation(
    tmp_path: Path,
    app_caplog,
) -> None:
    store = JobStore(tmp_path)
    processor = ControlledProcessor()
    service = RecognitionService(store, lambda: processor)
    service.start()
    job_id = create_job(store, b"first")
    try:
        service.submit(job_id)
        assert processor.first_started.wait(1)

        assert not service.delete(job_id)
        processor.release_first.set()

        def job_was_deleted() -> bool:
            try:
                store.get_status(job_id)
            except JobNotFoundError:
                return True
            return False

        wait_until(job_was_deleted)
    finally:
        processor.release_first.set()
        service.stop()

    messages = [record.getMessage() for record in app_caplog.records]
    assert any(
        f"Recognition job cancellation requested job_id={job_id}" in message
        for message in messages
    )
    assert any(
        f"Recognition job cancelled job_id={job_id}" in message
        and "duration_seconds=" in message
        for message in messages
    )
    assert any(
        f"Recognition job deleted job_id={job_id}" in message
        for message in messages
    )


class FailingProcessor:
    def process_iter(self, path: Path):
        raise ValueError("processor failed")
        yield


def test_service_logs_job_failure_with_traceback(
    tmp_path: Path,
    app_caplog,
) -> None:
    store = JobStore(tmp_path)
    service = RecognitionService(store, FailingProcessor)
    service.start()
    job_id = create_job(store)
    try:
        service.submit(job_id)
        wait_until(lambda: store.get_status(job_id)["status"] == "failed")
    finally:
        service.stop()

    failure = next(
        record
        for record in app_caplog.records
        if f"Recognition job failed job_id={job_id}" in record.getMessage()
    )
    assert failure.levelno == logging.ERROR
    assert failure.exc_info is not None
    assert "duration_seconds=" in failure.getMessage()


def test_service_rejects_jobs_when_waiting_queue_is_full(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    processor = ControlledProcessor()
    service = RecognitionService(store, lambda: processor, max_queued_jobs=1)
    service.start()
    running = create_job(store, b"first")
    queued = create_job(store, b"second")
    rejected = create_job(store, b"third")
    try:
        service.submit(running)
        assert processor.first_started.wait(1)
        service.submit(queued)

        with pytest.raises(JobQueueFullError):
            service.submit(rejected)
    finally:
        processor.release_first.set()
        service.stop()


def test_service_rejects_a_second_process_for_the_same_job_root(
    tmp_path: Path,
) -> None:
    first = RecognitionService(JobStore(tmp_path), ControlledProcessor)
    second = RecognitionService(JobStore(tmp_path), ControlledProcessor)
    first.start()
    try:
        with pytest.raises(ServiceAlreadyRunningError, match="job root"):
            second.start()
    finally:
        first.stop()

    second.start()
    second.stop()


def test_llm_settings_reads_required_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("EXAM_REC_LLM_API_KEY", " secret-key ")
    monkeypatch.setenv("EXAM_REC_LLM_BASE_URL", "https://llm.internal/v1")
    monkeypatch.setenv("EXAM_REC_LLM_MODEL", "model")

    assert LlmSettings.from_env() == LlmSettings(
        api_key="secret-key",
        base_url="https://llm.internal/v1",
        model="model",
    )


def test_llm_settings_requires_api_key_environment_variable(monkeypatch) -> None:
    monkeypatch.delenv("EXAM_REC_LLM_API_KEY", raising=False)
    monkeypatch.setenv("EXAM_REC_LLM_BASE_URL", "https://llm.internal/v1")
    monkeypatch.setenv("EXAM_REC_LLM_MODEL", "model")

    with pytest.raises(RuntimeError, match="EXAM_REC_LLM_API_KEY"):
        LlmSettings.from_env()
