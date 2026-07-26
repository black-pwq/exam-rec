from __future__ import annotations

import time
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from api import ApiSettings, create_app
from extractor.base_extractor import Problem
from extractor.evaluator import EvaluationReport
from pipeline import PageProcessingResult
from recognition_jobs import JobStore, RecognitionService


def make_pdf(page_count: int = 1) -> bytes:
    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page()
    try:
        return document.tobytes()
    finally:
        document.close()


class StubProcessor:
    def process_iter(self, path: Path):
        assert path.is_file()
        yield PageProcessingResult(
            page_index=0,
            problems=(
                Problem(
                    number="1",
                    question="question",
                    answer="",
                    options={"A": "first", "B": "second"},
                    analysis="",
                ),
            ),
            extractor_name="StubExtractor",
            evaluation=EvaluationReport(score=1.0),
        )


def wait_for_status(
    client: TestClient,
    job_id: str,
    expected: str,
    timeout: float = 2.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        response = client.get(f"/recognitions/{job_id}")
        assert response.status_code == 200
        status = response.json()
        if status["status"] == expected:
            return status
        if time.monotonic() >= deadline:
            raise AssertionError(f"job did not reach {expected}: {status}")
        time.sleep(0.005)


def make_client(tmp_path: Path, *, max_pdf_pages: int = 500) -> TestClient:
    settings = ApiSettings(
        job_root=tmp_path,
        max_upload_bytes=1024 * 1024,
        max_pdf_pages=max_pdf_pages,
        max_queued_jobs=4,
    )
    service = RecognitionService(JobStore(tmp_path), StubProcessor)
    return TestClient(create_app(service=service, settings=settings))


def test_submit_poll_updates_get_result_and_delete(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        submitted = client.post(
            "/recognitions",
            files={"file": ("questions.pdf", make_pdf(), "application/pdf")},
        )
        assert submitted.status_code == 202
        body = submitted.json()
        assert body["status"] == "queued"
        assert body["page_count"] == 1
        job_id = body["job_id"]
        assert set(body) == {"job_id", "status", "file_size", "page_count"}
        assert body["file_size"] > 0
        assert submitted.headers["location"] == f"/recognitions/{job_id}"

        status = wait_for_status(client, job_id, "completed")
        assert status["processed_pages"] == 1
        assert status["problem_count"] == 1

        first = client.get(
            f"/recognitions/{job_id}/updates",
            params={"after": 0, "limit": 2},
        ).json()
        assert [event["type"] for event in first["events"]] == [
            "queued",
            "started",
        ]
        remaining = client.get(
            f"/recognitions/{job_id}/updates",
            params={"after": first["next_cursor"]},
        ).json()
        assert [event["type"] for event in remaining["events"]] == [
            "page",
            "completed",
        ]

        result = client.get(f"/recognitions/{job_id}/result")
        assert result.status_code == 200
        assert result.json()["problems"][0]["question"] == "question"

        deleted = client.delete(f"/recognitions/{job_id}")
        assert deleted.status_code == 202
        assert deleted.json()["status"] == "deleted"
        assert client.get(f"/recognitions/{job_id}").status_code == 404


def test_rejects_invalid_and_oversized_pdfs(tmp_path: Path) -> None:
    with make_client(tmp_path, max_pdf_pages=1) as client:
        invalid = client.post(
            "/recognitions",
            files={"file": ("input.pdf", b"not a pdf", "application/pdf")},
        )
        assert invalid.status_code == 422

        too_many_pages = client.post(
            "/recognitions",
            files={"file": ("input.pdf", make_pdf(2), "application/pdf")},
        )
        assert too_many_pages.status_code == 413

    assert list(tmp_path.glob("[!.]*")) == []


def test_health_endpoints_distinguish_liveness_and_readiness(
    tmp_path: Path,
) -> None:
    with make_client(tmp_path) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/health/live").json() == {"status": "ok"}
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready"}

        client.app.state.recognition_service.stop()

        assert client.get("/health").json() == {"status": "unavailable"}
        assert client.get("/health/live").status_code == 200
        unavailable = client.get("/health/ready")
        assert unavailable.status_code == 503
        assert unavailable.json() == {"status": "unavailable"}
