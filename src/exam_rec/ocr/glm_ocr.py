from __future__ import annotations

from base64 import b64encode
from collections.abc import Iterator, Mapping
from os import PathLike, environ, fspath
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np
import pymupdf
import requests

from exam_rec.app_logging import get_logger
from exam_rec.ocr.base_ocr import BaseOcr, OcrElement, Point


logger = get_logger(__name__)


class GlmOcr(BaseOcr):
    """Parse images and PDFs with the BigModel GLM-OCR model."""

    DEFAULT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
    MAX_PDF_BYTES = 50 * 1024 * 1024
    MAX_PDF_PAGES = 100
    PDF_BATCH_PAGES = 20
    MAX_IMAGE_BYTES = 10 * 1024 * 1024

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "glm-ocr",
        return_crop_images: bool = False,
        need_layout_visualization: bool = False,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        request_id: str | None = None,
        user_id: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 120,
        session: requests.Session | None = None,
    ) -> None:
        resolved_key = (
            api_key or environ.get("EXAM_REC_GLMOCR_API_KEY", "")
        ).strip()
        if not resolved_key:
            raise ValueError(
                "GlmOcr requires api_key or the "
                "EXAM_REC_GLMOCR_API_KEY environment variable"
            )
        if start_page_id is not None and start_page_id < 1:
            raise ValueError("start_page_id must be at least 1")
        if end_page_id is not None and end_page_id < 1:
            raise ValueError("end_page_id must be at least 1")
        if (
            start_page_id is not None
            and end_page_id is not None
            and end_page_id < start_page_id
        ):
            raise ValueError("end_page_id must not precede start_page_id")

        self.api_key = resolved_key
        self.model = model
        self.return_crop_images = return_crop_images
        self.need_layout_visualization = need_layout_visualization
        self.start_page_id = start_page_id
        self.end_page_id = end_page_id
        self.request_id = request_id
        self.user_id = user_id
        self.endpoint = endpoint
        self.timeout = timeout
        self.session = session or requests.Session()
        self._owns_session = session is None

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        for request_index, file in enumerate(self._file_values(input), start=1):
            yield from self._predict_file(file, request_index=request_index)

    def _predict_file(
        self,
        file: str,
        *,
        request_index: int,
    ) -> Iterator[list[OcrElement]]:
        payload = self._request_payload(file)
        logger.info(
            "GLM-OCR request started: request_index=%d model=%s "
            "file_value_chars=%d timeout_seconds=%s",
            request_index,
            self.model,
            len(file),
            self.timeout,
        )
        started = monotonic()
        try:
            response = self.session.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            logger.warning(
                "GLM-OCR request failed: request_index=%d model=%s "
                "duration_seconds=%.3f error_type=%s",
                request_index,
                self.model,
                monotonic() - started,
                type(error).__name__,
            )
            raise
        logger.info(
            "GLM-OCR response received: request_index=%d model=%s "
            "status_code=%d duration_seconds=%.3f",
            request_index,
            self.model,
            response.status_code,
            monotonic() - started,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            detail = self._error_detail(response)
            raise RuntimeError(
                f"BigModel GLM-OCR request failed: "
                f"HTTP {response.status_code}: {detail}"
            ) from error

        result = response.json()
        if not isinstance(result, Mapping):
            raise ValueError("BigModel GLM-OCR returned a non-object response")
        if isinstance(result.get("error"), Mapping):
            raise RuntimeError(
                f"BigModel GLM-OCR failed: {self._payload_error_detail(result)}"
            )

        pages = result.get("layout_details")
        if not isinstance(pages, (list, tuple)):
            raise ValueError("BigModel GLM-OCR returned invalid layout_details")
        for page in pages:
            yield self._page_elements(page)

    def _request_payload(self, file: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "file": file,
            "return_crop_images": self.return_crop_images,
            "need_layout_visualization": self.need_layout_visualization,
        }
        optional = {
            "start_page_id": self.start_page_id,
            "end_page_id": self.end_page_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload

    @classmethod
    def _file_values(cls, input: Any) -> Iterator[str]:
        if isinstance(input, str) and input.startswith(("http://", "https://")):
            yield input
            return
        if isinstance(input, (str, PathLike)):
            path = Path(fspath(input))
            content = path.read_bytes()
            media_type = cls._media_type(path.suffix, content)
            yield from cls._content_values(content, media_type)
            return
        if isinstance(input, (bytes, bytearray, memoryview)):
            content = bytes(input)
            media_type = cls._media_type("", content)
            yield from cls._content_values(content, media_type)
            return
        if isinstance(input, np.ndarray):
            success, encoded = cv2.imencode(".png", input)
            if not success:
                raise ValueError("could not encode numpy.ndarray as PNG")
            content = encoded.tobytes()
            cls._validate_size(content, "image/png")
            yield cls._data_uri(content, "image/png")
            return
        raise TypeError(
            "GlmOcr input must be an image/PDF path or URL, image/PDF bytes, "
            "or numpy.ndarray"
        )

    @classmethod
    def _content_values(cls, content: bytes, media_type: str) -> Iterator[str]:
        if media_type == "application/pdf":
            yield from cls._pdf_values(content)
            return
        cls._validate_size(content, media_type)
        yield cls._data_uri(content, media_type)

    @classmethod
    def _pdf_values(cls, content: bytes) -> Iterator[str]:
        document = pymupdf.open(stream=content, filetype="pdf")
        try:
            if document.needs_pass:
                raise ValueError("GlmOcr input PDF is password protected")
            if document.page_count == 0:
                raise ValueError("GlmOcr input PDF must not be empty")
            page_limit = min(cls.PDF_BATCH_PAGES, cls.MAX_PDF_PAGES)
            if (
                len(content) <= cls.MAX_PDF_BYTES
                and document.page_count <= page_limit
            ):
                yield cls._data_uri(content, "application/pdf")
                return

            start = 0
            while start < document.page_count:
                end, chunk = cls._largest_pdf_chunk(
                    document,
                    start,
                    page_limit=page_limit,
                )
                yield cls._data_uri(chunk, "application/pdf")
                start = end
        finally:
            document.close()

    @classmethod
    def _largest_pdf_chunk(
        cls,
        document: Any,
        start: int,
        *,
        page_limit: int,
    ) -> tuple[int, bytes]:
        chunk = pymupdf.open()
        accepted = b""
        accepted_end = start
        try:
            stop = min(document.page_count, start + page_limit)
            for page_index in range(start, stop):
                chunk.insert_pdf(
                    document,
                    from_page=page_index,
                    to_page=page_index,
                )
                candidate = chunk.tobytes(garbage=3, deflate=True)
                if len(candidate) > cls.MAX_PDF_BYTES:
                    if not accepted:
                        raise ValueError(
                            f"GlmOcr PDF page {page_index + 1} exceeds the "
                            "single-request 50 MB limit"
                        )
                    break
                accepted = candidate
                accepted_end = page_index + 1
        finally:
            chunk.close()
        return accepted_end, accepted

    @staticmethod
    def _media_type(suffix: str, content: bytes) -> str:
        if content.lstrip().startswith(b"%PDF-"):
            return "application/pdf"
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        media_type = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(suffix.lower())
        if media_type is None:
            raise ValueError("GlmOcr supports only PDF, PNG, JPG, and JPEG inputs")
        return media_type

    @staticmethod
    def _validate_size(content: bytes, media_type: str) -> None:
        limit = (
            GlmOcr.MAX_PDF_BYTES
            if media_type == "application/pdf"
            else GlmOcr.MAX_IMAGE_BYTES
        )
        if len(content) > limit:
            kind = "PDF" if media_type == "application/pdf" else "image"
            raise ValueError(f"GlmOcr {kind} input exceeds the API size limit")

    @staticmethod
    def _data_uri(content: bytes, media_type: str) -> str:
        encoded = b64encode(content).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    @classmethod
    def _page_elements(cls, page: Any) -> list[OcrElement]:
        if not isinstance(page, (list, tuple)):
            return []
        elements: list[OcrElement] = []
        for item in page:
            if not isinstance(item, Mapping):
                continue
            elements.append(
                OcrElement(
                    bbox=cls._bbox_points(
                        item.get("bbox_2d"),
                        item.get("width"),
                        item.get("height"),
                    ),
                    label=str(item.get("label", "text")),
                    content=str(item.get("content", "")),
                )
            )
        return elements

    @staticmethod
    def _bbox_points(bbox: Any, width: Any, height: Any) -> list[Point]:
        if isinstance(bbox, (str, bytes)):
            return []
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
            page_width = float(width)
            page_height = float(height)
        except (TypeError, ValueError):
            return []
        return [
            Point(x1 * page_width, y1 * page_height),
            Point(x2 * page_width, y1 * page_height),
            Point(x2 * page_width, y2 * page_height),
            Point(x1 * page_width, y2 * page_height),
        ]

    @staticmethod
    def _payload_error_detail(payload: Mapping[str, Any]) -> str:
        error = payload.get("error")
        if isinstance(error, Mapping):
            code = error.get("code")
            message = error.get("message")
            if code and message:
                return f"{code}: {message}"
            if message or code:
                return str(message or code)
        message = payload.get("message")
        return str(message or payload)

    @classmethod
    def _error_detail(cls, response: requests.Response) -> str:
        try:
            payload = response.json()
        except (requests.JSONDecodeError, ValueError):
            text = response.text.strip()
            return text[:500] if text else "empty response"
        if isinstance(payload, Mapping):
            return cls._payload_error_detail(payload)
        return str(payload)[:500]

    def close(self) -> None:
        if self._owns_session:
            self.session.close()
