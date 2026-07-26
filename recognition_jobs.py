from __future__ import annotations

import fcntl
import json
import os
import queue
import re
import shutil
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from extractor.regex_extractor import LlmRegexAnalyzer
from ocr.ocr_factory import OcrFactory, OcrRegistry
from ocr.page_ocr import PdfPageSource
from pipeline import (
    PageProcessingResult,
    ProblemProcessingPipeline,
    ProcessingPolicy,
    ProcessingRequest,
)
from question_range import (
    LlmQuestionStartAnalyzer,
    QuestionRangePolicy,
    QuestionRangeResolver,
)


_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_INTERRUPTIBLE_STATUSES = {"queued", "running", "cancelling"}
_RESULT_STATUSES = {"completed"}


class JobNotFoundError(KeyError):
    pass


class JobQueueFullError(RuntimeError):
    pass


class JobNotCompleteError(RuntimeError):
    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(f"recognition job is not complete: {status}")


class ServiceAlreadyRunningError(RuntimeError):
    pass


class RecognitionProcessor(Protocol):
    def process_iter(self, path: Path) -> Iterator[PageProcessingResult]: ...


@dataclass(frozen=True)
class LlmSettings:
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> LlmSettings:
        names = {
            "api_key": "EXAM_REC_LLM_API_KEY",
            "base_url": "EXAM_REC_LLM_BASE_URL",
            "model": "EXAM_REC_LLM_MODEL",
        }
        values = {name: os.getenv(variable, "").strip() for name, variable in names.items()}
        missing = [names[name] for name, value in values.items() if not value]
        if missing:
            raise RuntimeError(
                "missing required LLM settings: " + ", ".join(sorted(missing))
            )
        return cls(**values)


class DefaultRecognitionProcessor:
    """Use worker-level model reuse and task-level page-result reuse."""

    def __init__(self, settings: LlmSettings) -> None:
        self.registry = OcrRegistry()
        self.ocr_factory = OcrFactory(registry=self.registry)
        self.question_range_policy = QuestionRangePolicy()
        self.processing_policy = ProcessingPolicy()
        self.question_start_analyzer = LlmQuestionStartAnalyzer(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
        )
        self.regex_analyzer = LlmRegexAnalyzer(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
        )

    def process_iter(self, path: Path) -> Iterator[PageProcessingResult]:
        document = PdfPageSource(path)
        scan_indexes = range(
            min(document.page_count, self.question_range_policy.max_scan_pages)
        )
        page_ocr = self.ocr_factory.create_page_ocr(
            path,
            scan_indexes,
            max_cached_pages=(
                self.question_range_policy.max_scan_pages
                + self.processing_policy.sample_page_count
            ),
        )
        resolver = QuestionRangeResolver(
            self.question_start_analyzer,
            page_ocr=page_ocr,
            policy=self.question_range_policy,
        )
        pipeline = ProblemProcessingPipeline(
            page_ocr=page_ocr,
            analyzer=self.regex_analyzer,
            policy=self.processing_policy,
        )
        try:
            questions = resolver.resolve(path)
            yield from pipeline.process_iter(
                ProcessingRequest(path=path, questions=questions)
            )
        finally:
            page_ocr.clear_cache()

    def close(self) -> None:
        self.registry.close()


class JobStore:
    """Persist job metadata and append-only updates under one local directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._next_sequences: dict[str, int] = {}

    def new_job_id(self) -> str:
        return uuid4().hex

    def prepare_upload(self, job_id: str) -> Path:
        with self._lock:
            job_dir = self._job_dir(job_id)
            job_dir.mkdir(parents=False, exist_ok=False)
            return job_dir / "input.upload"

    def create(
        self,
        job_id: str,
        *,
        original_filename: str,
        page_count: int,
    ) -> dict[str, Any]:
        with self._lock:
            job_dir = self._job_dir(job_id)
            upload = job_dir / "input.upload"
            if not upload.is_file():
                raise FileNotFoundError(f"uploaded PDF is missing for job {job_id}")
            os.replace(upload, job_dir / "input.pdf")
            now = self._now()
            status = {
                "job_id": job_id,
                "status": "queued",
                "original_filename": original_filename,
                "page_count": page_count,
                "processed_pages": 0,
                "problem_count": 0,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
            self._write_json(job_dir / "status.json", status)
            self._next_sequences[job_id] = 1
            self._append_event_locked(job_id, "queued")
            return status

    def input_path(self, job_id: str) -> Path:
        path = self._job_dir(job_id) / "input.pdf"
        if not path.is_file():
            raise JobNotFoundError(job_id)
        return path

    def get_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._job_dir(job_id) / "status.json"
            if not path.is_file():
                raise JobNotFoundError(job_id)
            return self._read_json(path)

    def update_status(
        self,
        job_id: str,
        status: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        with self._lock:
            value = self.get_status(job_id)
            if status is not None:
                value["status"] = status
            value.update(fields)
            value["updated_at"] = self._now()
            self._write_json(self._job_dir(job_id) / "status.json", value)
            return value

    def append_event(
        self,
        job_id: str,
        event_type: str,
        **payload: Any,
    ) -> dict[str, Any]:
        with self._lock:
            return self._append_event_locked(job_id, event_type, **payload)

    def record_page(
        self,
        job_id: str,
        result: PageProcessingResult,
        *,
        processed_pages: int,
        problem_count: int,
    ) -> dict[str, Any]:
        with self._lock:
            event = self._append_event_locked(
                job_id,
                "page",
                page_index=result.page_index,
                problems=[asdict(problem) for problem in result.problems],
                extractor_name=result.extractor_name,
                evaluation={
                    "score": result.evaluation.score,
                    "metrics": dict(result.evaluation.metrics),
                    "warnings": list(result.evaluation.warnings),
                },
            )
            self.update_status(
                job_id,
                processed_pages=processed_pages,
                problem_count=problem_count,
            )
            return event

    def complete(self, job_id: str, problems: list[dict[str, Any]]) -> None:
        with self._lock:
            result = {
                "job_id": job_id,
                "status": "completed",
                "problem_count": len(problems),
                "problems": problems,
            }
            self._write_json(self._job_dir(job_id) / "result.json", result)
            self.update_status(
                job_id,
                "completed",
                problem_count=len(problems),
                error=None,
            )
            self._append_event_locked(
                job_id,
                "completed",
                problem_count=len(problems),
            )

    def fail(self, job_id: str, error: BaseException) -> None:
        with self._lock:
            message = str(error) or type(error).__name__
            self.update_status(job_id, "failed", error=message)
            self._append_event_locked(
                job_id,
                "error",
                error_type=type(error).__name__,
                message=message,
            )

    def mark_cancelled(self, job_id: str) -> None:
        with self._lock:
            self.update_status(job_id, "cancelled")
            self._append_event_locked(job_id, "cancelled")

    def get_updates(
        self,
        job_id: str,
        *,
        after: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        with self._lock:
            status = self.get_status(job_id)
            path = self._job_dir(job_id) / "events.jsonl"
            events: list[dict[str, Any]] = []
            has_more = False
            if path.is_file():
                with path.open(encoding="utf-8") as file:
                    for line in file:
                        event = json.loads(line)
                        if event["sequence"] <= after:
                            continue
                        if len(events) == limit:
                            has_more = True
                            break
                        events.append(event)
            next_cursor = events[-1]["sequence"] if events else after
            return {
                "job_id": job_id,
                "status": status["status"],
                "next_cursor": next_cursor,
                "has_more": has_more,
                "events": events,
            }

    def get_result(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            status = self.get_status(job_id)
            if status["status"] not in _RESULT_STATUSES:
                raise JobNotCompleteError(status["status"])
            return self._read_json(self._job_dir(job_id) / "result.json")

    def delete(self, job_id: str) -> None:
        with self._lock:
            job_dir = self._job_dir(job_id)
            if not job_dir.is_dir():
                raise JobNotFoundError(job_id)
            shutil.rmtree(job_dir)
            self._next_sequences.pop(job_id, None)

    def discard_upload(self, job_id: str) -> None:
        with self._lock:
            job_dir = self._job_dir(job_id)
            if job_dir.is_dir():
                shutil.rmtree(job_dir)
            self._next_sequences.pop(job_id, None)

    def mark_unfinished_interrupted(self) -> None:
        with self._lock:
            for status_path in self.root.glob("*/status.json"):
                status = self._read_json(status_path)
                if status.get("status") not in _INTERRUPTIBLE_STATUSES:
                    continue
                job_id = str(status["job_id"])
                self.update_status(
                    job_id,
                    "interrupted",
                    error="service stopped before the job completed",
                )
                self._append_event_locked(
                    job_id,
                    "interrupted",
                    message="service stopped before the job completed",
                )

    def _append_event_locked(
        self,
        job_id: str,
        event_type: str,
        **payload: Any,
    ) -> dict[str, Any]:
        self.get_status(job_id)
        sequence = self._next_sequences.get(job_id)
        if sequence is None:
            sequence = self._read_next_sequence(job_id)
        event = {
            "sequence": sequence,
            "type": event_type,
            "created_at": self._now(),
            **payload,
        }
        path = self._job_dir(job_id) / "events.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
            file.flush()
        self._next_sequences[job_id] = sequence + 1
        return event

    def _read_next_sequence(self, job_id: str) -> int:
        path = self._job_dir(job_id) / "events.jsonl"
        if not path.is_file():
            return 1
        last = 0
        with path.open(encoding="utf-8") as file:
            for line in file:
                last = int(json.loads(line)["sequence"])
        return last + 1

    def _job_dir(self, job_id: str) -> Path:
        if not _JOB_ID_RE.fullmatch(job_id):
            raise JobNotFoundError(job_id)
        return self.root / job_id

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


class _JobCancelled(Exception):
    pass


class ServiceInstanceLock:
    """Hold a process-level exclusive lock for one local job store."""

    def __init__(self, root: Path) -> None:
        self.path = root / ".service.lock"
        self._file: Any | None = None

    @property
    def acquired(self) -> bool:
        return self._file is not None

    def acquire(self) -> None:
        if self._file is not None:
            return
        file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            file.close()
            raise ServiceAlreadyRunningError(
                f"another recognition service is using job root: "
                f"{self.path.parent}"
            ) from error
        file.seek(0)
        file.truncate()
        file.write(f"{os.getpid()}\n")
        file.flush()
        self._file = file

    def release(self) -> None:
        file = self._file
        if file is None:
            return
        self._file = None
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        finally:
            file.close()


class RecognitionService:
    """Run whole-document recognition jobs serially in FIFO order."""

    def __init__(
        self,
        store: JobStore,
        processor_factory: Callable[[], RecognitionProcessor],
        *,
        max_queued_jobs: int = 32,
    ) -> None:
        if max_queued_jobs < 1:
            raise ValueError("max_queued_jobs must be at least 1")
        self.store = store
        self.processor_factory = processor_factory
        self.max_queued_jobs = max_queued_jobs
        self._queue: queue.Queue[str | None] = queue.Queue(max_queued_jobs)
        self._cancellations: dict[str, threading.Event] = {}
        self._delete_requested: set[str] = set()
        self._state_lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._startup_error: BaseException | None = None
        self._current_job_id: str | None = None
        self._thread: threading.Thread | None = None
        self._instance_lock = ServiceInstanceLock(self.store.root)

    @property
    def running(self) -> bool:
        thread = self._thread
        return (
            thread is not None
            and thread.is_alive()
            and self._startup_error is None
            and not self._stop.is_set()
        )

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None:
                return
            self._instance_lock.acquire()
            try:
                self.store.mark_unfinished_interrupted()
                self._thread = threading.Thread(
                    target=self._worker_main,
                    name="recognition-worker",
                    daemon=True,
                )
                self._thread.start()
            except BaseException:
                self._thread = None
                self._instance_lock.release()
                raise
        self._ready.wait()
        if self._startup_error is not None:
            self._instance_lock.release()
            raise RuntimeError(
                f"failed to start recognition worker: {self._startup_error}"
            ) from self._startup_error

    def submit(self, job_id: str) -> None:
        if not self.running:
            raise RuntimeError("recognition service is not running")
        cancellation = threading.Event()
        with self._state_lock:
            self._cancellations[job_id] = cancellation
        try:
            self._queue.put_nowait(job_id)
        except queue.Full:
            with self._state_lock:
                self._cancellations.pop(job_id, None)
            raise JobQueueFullError("recognition queue is full") from None

    def delete(self, job_id: str) -> bool:
        self.store.get_status(job_id)
        with self._state_lock:
            cancellation = self._cancellations.get(job_id)
            is_running = self._current_job_id == job_id
            if cancellation is not None:
                cancellation.set()
            if is_running:
                self._delete_requested.add(job_id)
        if is_running:
            self.store.update_status(job_id, "cancelling")
            self.store.append_event(job_id, "cancellation_requested")
            return False
        self.store.delete(job_id)
        return True

    def stop(self) -> None:
        with self._state_lock:
            thread = self._thread
            if thread is None:
                self._instance_lock.release()
                return
            self._stop.set()
            for cancellation in self._cancellations.values():
                cancellation.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if threading.current_thread() is not thread:
            thread.join()
        self.store.mark_unfinished_interrupted()
        self._instance_lock.release()

    def _worker_main(self) -> None:
        processor: RecognitionProcessor | None = None
        try:
            try:
                processor = self.processor_factory()
            except BaseException as error:
                self._startup_error = error
                return
            finally:
                self._ready.set()

            while not self._stop.is_set():
                job_id = self._queue.get()
                try:
                    if job_id is None:
                        return
                    self._run_job(processor, job_id)
                finally:
                    self._queue.task_done()
        finally:
            self._ready.set()
            if processor is not None:
                close = getattr(processor, "close", None)
                if close is not None:
                    close()

    def _run_job(self, processor: RecognitionProcessor, job_id: str) -> None:
        with self._state_lock:
            cancellation = self._cancellations.get(job_id)
            self._current_job_id = job_id
        if cancellation is None:
            self._finish_job(job_id)
            return

        iterator: Iterator[PageProcessingResult] | None = None
        try:
            try:
                self.store.get_status(job_id)
            except JobNotFoundError:
                return
            if cancellation.is_set():
                raise _JobCancelled

            self.store.update_status(job_id, "running")
            self.store.append_event(job_id, "started")
            iterator = iter(processor.process_iter(self.store.input_path(job_id)))
            processed_pages = 0
            problems: list[dict[str, Any]] = []
            for page in iterator:
                if cancellation.is_set() or self._stop.is_set():
                    raise _JobCancelled
                page_problems = [asdict(problem) for problem in page.problems]
                problems.extend(page_problems)
                processed_pages += 1
                self.store.record_page(
                    job_id,
                    page,
                    processed_pages=processed_pages,
                    problem_count=len(problems),
                )
            if cancellation.is_set() or self._stop.is_set():
                raise _JobCancelled
            self.store.complete(job_id, problems)
        except _JobCancelled:
            if self._job_exists(job_id):
                self.store.mark_cancelled(job_id)
        except Exception as error:
            if self._job_exists(job_id):
                self.store.fail(job_id, error)
        finally:
            if iterator is not None:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            self._finish_job(job_id)

    def _finish_job(self, job_id: str) -> None:
        with self._state_lock:
            delete_requested = job_id in self._delete_requested
            self._delete_requested.discard(job_id)
            self._cancellations.pop(job_id, None)
            if self._current_job_id == job_id:
                self._current_job_id = None
        if delete_requested and self._job_exists(job_id):
            self.store.delete(job_id)

    def _job_exists(self, job_id: str) -> bool:
        try:
            self.store.get_status(job_id)
        except JobNotFoundError:
            return False
        return True
