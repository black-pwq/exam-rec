from collections.abc import Iterator, Mapping
from os import PathLike, fspath
from typing import Any

import pymupdf

from exam_rec.ocr.base_ocr import BaseOcr, OcrElement, Point


class PymuOcr(BaseOcr):
    """Extract text lines and their positions from a PDF text layer."""

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        document = self._open_document(input)
        try:
            for page in document:
                yield self._page_elements(page.get_text("dict", sort=True))
        finally:
            document.close()

    @staticmethod
    def _open_document(input: Any) -> Any:
        if isinstance(input, PathLike):
            return pymupdf.open(fspath(input))
        if isinstance(input, (bytes, bytearray, memoryview)):
            return pymupdf.open(stream=bytes(input), filetype="pdf")
        return pymupdf.open(input)

    @classmethod
    def _page_elements(cls, page_data: Any) -> list[OcrElement]:
        if not isinstance(page_data, Mapping):
            return []

        elements = []
        for block in page_data.get("blocks", ()):
            if not isinstance(block, Mapping) or block.get("type", 0) != 0:
                continue
            for line in block.get("lines", ()):
                element = cls._line_element(line)
                if element is not None:
                    elements.append(element)
        return elements

    @staticmethod
    def _line_element(line: Any) -> OcrElement | None:
        if not isinstance(line, Mapping):
            return None

        spans = line.get("spans", ())
        if not isinstance(spans, (list, tuple)):
            return None
        content = "".join(
            str(span.get("text", ""))
            for span in spans
            if isinstance(span, Mapping)
        )
        if not content.strip():
            return None

        bbox = PymuOcr._rectangle_points(line.get("bbox"))
        if not bbox:
            span_boxes = [
                span.get("bbox") for span in spans if isinstance(span, Mapping)
            ]
            bbox = PymuOcr._merged_rectangle_points(span_boxes)
        return OcrElement(bbox=bbox, label="text", content=content)

    @staticmethod
    def _rectangle_points(bbox: Any) -> list[Point]:
        if isinstance(bbox, (str, bytes)):
            return []
        try:
            x0, y0, x1, y1 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return []
        return [Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)]

    @staticmethod
    def _merged_rectangle_points(boxes: list[Any]) -> list[Point]:
        rectangles = []
        for bbox in boxes:
            if isinstance(bbox, (str, bytes)):
                continue
            try:
                x0, y0, x1, y1 = (float(value) for value in bbox)
            except (TypeError, ValueError):
                continue
            rectangles.append((x0, y0, x1, y1))
        if not rectangles:
            return []
        return PymuOcr._rectangle_points(
            (
                min(rect[0] for rect in rectangles),
                min(rect[1] for rect in rectangles),
                max(rect[2] for rect in rectangles),
                max(rect[3] for rect in rectangles),
            )
        )


PyMuPDFOcr = PymuOcr
