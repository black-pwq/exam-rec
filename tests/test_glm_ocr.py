from __future__ import annotations

from base64 import b64decode
from typing import Any

import numpy as np
import pymupdf
import pytest
import requests

from exam_rec.ocr.base_ocr import OcrElement, Point
from exam_rec.ocr.glm_ocr import GlmOcr


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> Any:
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def post(self, endpoint: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"endpoint": endpoint, **kwargs})
        return self.response


def result(*pages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "task-1",
        "model": "GLM-OCR",
        "layout_details": list(pages),
    }


def test_posts_pdf_as_json_base64_and_converts_all_pages(tmp_path) -> None:
    pdf = tmp_path / "sample.pdf"
    document = pymupdf.open()
    document.new_page()
    pdf.write_bytes(document.tobytes())
    document.close()
    pdf_content = pdf.read_bytes()
    session = FakeSession(
        FakeResponse(
            result(
                [
                    {
                        "label": "text",
                        "content": "first",
                        "bbox_2d": [0.1, 0.2, 0.5, 0.4],
                        "width": 600,
                        "height": 800,
                    }
                ],
                [
                    {
                        "label": "formula",
                        "content": "x^2",
                        "bbox_2d": [0, 0, 1, 1],
                        "width": 100,
                        "height": 200,
                    }
                ],
            )
        )
    )

    pages = GlmOcr(api_key=" secret ", session=session).predict(pdf)

    assert pages == [
        [
            OcrElement(
                bbox=[
                    Point(60, 160),
                    Point(300, 160),
                    Point(300, 320),
                    Point(60, 320),
                ],
                label="text",
                content="first",
            )
        ],
        [
            OcrElement(
                bbox=[
                    Point(0, 0),
                    Point(100, 0),
                    Point(100, 200),
                    Point(0, 200),
                ],
                label="formula",
                content="x^2",
            )
        ],
    ]
    request = session.requests[0]
    assert request["endpoint"] == GlmOcr.DEFAULT_ENDPOINT
    assert request["headers"]["Authorization"] == "Bearer secret"
    payload = request["json"]
    assert payload["model"] == "glm-ocr"
    assert payload["file"].startswith("data:application/pdf;base64,")
    assert b64decode(payload["file"].split(",", 1)[1]) == pdf_content
    assert payload["return_crop_images"] is False
    assert payload["need_layout_visualization"] is False


def test_url_and_optional_parameters_are_forwarded() -> None:
    session = FakeSession(FakeResponse(result([])))
    ocr = GlmOcr(
        api_key="key",
        session=session,
        start_page_id=2,
        end_page_id=4,
        request_id="request-1",
        user_id="user-123",
        return_crop_images=True,
    )

    assert ocr.predict("https://example.com/exam.pdf") == [[]]
    assert session.requests[0]["json"] == {
        "model": "glm-ocr",
        "file": "https://example.com/exam.pdf",
        "return_crop_images": True,
        "need_layout_visualization": False,
        "start_page_id": 2,
        "end_page_id": 4,
        "request_id": "request-1",
        "user_id": "user-123",
    }


def test_accepts_numpy_image() -> None:
    session = FakeSession(FakeResponse(result([])))
    ocr = GlmOcr(api_key="key", session=session)

    assert ocr.predict(np.zeros((2, 2, 3), dtype=np.uint8)) == [[]]
    assert session.requests[0]["json"]["file"].startswith("data:image/png;base64,")


def test_large_page_count_pdf_is_split_and_page_order_is_preserved(
    monkeypatch,
) -> None:
    document = pymupdf.open()
    document.new_page()
    document.new_page()
    pdf = document.tobytes()
    document.close()
    session = FakeSession(FakeResponse(result([])))
    monkeypatch.setattr(GlmOcr, "PDF_BATCH_PAGES", 1)

    assert GlmOcr(api_key="key", session=session).predict(pdf) == [[], []]
    assert len(session.requests) == 2
    for request in session.requests:
        value = request["json"]["file"]
        chunk = b64decode(value.split(",", 1)[1])
        chunk_document = pymupdf.open(stream=chunk, filetype="pdf")
        try:
            assert chunk_document.page_count == 1
        finally:
            chunk_document.close()


def test_large_pdf_bytes_are_split(monkeypatch) -> None:
    document = pymupdf.open()
    document.new_page()
    document.new_page()
    pdf = document.tobytes()
    one_page = pymupdf.open()
    one_page.insert_pdf(document, from_page=0, to_page=0)
    one_page_size = len(one_page.tobytes(garbage=3, deflate=True))
    one_page.close()
    document.close()
    session = FakeSession(FakeResponse(result([])))
    assert one_page_size < len(pdf)
    monkeypatch.setattr(GlmOcr, "MAX_PDF_BYTES", one_page_size)

    assert GlmOcr(api_key="key", session=session).predict(pdf) == [[], []]
    assert len(session.requests) == 2


def test_pdf_uses_twenty_page_operational_batches() -> None:
    document = pymupdf.open()
    for _ in range(45):
        document.new_page()
    pdf = document.tobytes()
    document.close()

    values = list(GlmOcr._file_values(pdf))
    page_counts = []
    for value in values:
        chunk = b64decode(value.split(",", 1)[1])
        chunk_document = pymupdf.open(stream=chunk, filetype="pdf")
        try:
            page_counts.append(chunk_document.page_count)
        finally:
            chunk_document.close()

    assert page_counts == [20, 20, 5]


def test_uses_environment_api_key(monkeypatch) -> None:
    monkeypatch.setenv("EXAM_REC_GLMOCR_API_KEY", "env-key")
    assert GlmOcr(session=FakeSession(FakeResponse(result()))).api_key == "env-key"


def test_rejects_invalid_page_range() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        GlmOcr(api_key="key", start_page_id=3, end_page_id=2)


def test_rejects_unsupported_input() -> None:
    ocr = GlmOcr(api_key="key", session=FakeSession(FakeResponse(result())))
    with pytest.raises(TypeError, match="image/PDF"):
        ocr.predict(object())


def test_http_error_includes_service_detail() -> None:
    session = FakeSession(
        FakeResponse(
            {"error": {"code": "1214", "message": "invalid parameter"}},
            status_code=400,
        )
    )
    ocr = GlmOcr(api_key="key", session=session)

    with pytest.raises(
        RuntimeError,
        match=r"HTTP 400: 1214: invalid parameter",
    ):
        ocr.predict(b"\x89PNG\r\n\x1a\n")


def test_success_response_error_is_reported() -> None:
    session = FakeSession(
        FakeResponse({"error": {"code": "1001", "message": "bad file"}})
    )
    ocr = GlmOcr(api_key="key", session=session)

    with pytest.raises(RuntimeError, match="1001: bad file"):
        ocr.predict(b"\xff\xd8\xff")
