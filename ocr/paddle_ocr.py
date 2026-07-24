import warnings
warnings.filterwarnings(
    "ignore",
    message=r"No ccache found\..*",
    category=UserWarning,
)

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from os import PathLike, fspath, unlink
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from paddleocr import PaddleOCR
from ocr.base_ocr import BaseOcr, OcrElement, Point


class PaddleInputAdapter:
    """Convert supported application inputs to PaddleOCR input types."""

    @staticmethod
    @contextmanager
    def adapt(input: Any) -> Iterator[str | np.ndarray]:
        if isinstance(input, PathLike):
            yield fspath(input)
            return
        if isinstance(input, (str, np.ndarray)):
            yield input
            return
        if isinstance(input, (bytes, bytearray, memoryview)):
            temporary_path = ""
            try:
                with NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
                    temporary.write(bytes(input))
                    temporary_path = temporary.name
                yield temporary_path
            finally:
                if temporary_path:
                    unlink(temporary_path)
            return
        raise TypeError(
            "PaddleOCR input must be a path, PDF bytes, or numpy.ndarray"
        )


class PaddleOcr(BaseOcr):
    def __init__(self, **kwargs: Any) -> None:
        options = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        options.update(kwargs)
        self.ocr = PaddleOCR(**options)

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        with PaddleInputAdapter.adapt(input) as paddle_input:
            for page in self.ocr.predict_iter(paddle_input):
                result = self._result_data(page)
                texts = result.get("rec_texts", ())
                polygons = result.get("rec_polys", result.get("dt_polys", ()))

                elements = []
                for text, polygon in zip(texts, polygons):
                    elements.append(OcrElement(
                        bbox=self._points(polygon),
                        label="text",
                        content=str(text),
                    ))
                yield elements

    @staticmethod
    def _result_data(page: Any) -> Mapping[str, Any]:
        if isinstance(page, Mapping):
            data: Any = page
        else:
            data = getattr(page, "json", {})

        if callable(data):
            data = data()
        if isinstance(data, Mapping) and isinstance(data.get("res"), Mapping):
            data = data["res"]
        return data if isinstance(data, Mapping) else {}

    @staticmethod
    def _points(polygon: Any) -> list[Point]:
        if isinstance(polygon, (str, bytes)):
            return []
        try:
            points = iter(polygon)
        except TypeError:
            return []

        result = []
        for point in points:
            if isinstance(point, (str, bytes)):
                continue
            try:
                if len(point) >= 2:
                    result.append(Point(x=float(point[0]), y=float(point[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return result
