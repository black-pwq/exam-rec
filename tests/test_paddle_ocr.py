from typing import Any
from pathlib import Path
import os

import numpy as np
import pytest

from exam_rec.ocr.base_ocr import BaseOcr, OcrElement, Point
from exam_rec.ocr.paddle_ocr import PaddleOcr


class FakeEngine:
    def __init__(self, pages: list[Any]) -> None:
        self.pages = pages
        self.received_input: Any = None

    def predict_iter(self, input: Any):
        self.received_input = input
        yield from self.pages


class ResultObject:
    def __init__(self, data: dict[str, Any]) -> None:
        self.json = {"res": data}


def make_ocr(*pages: Any) -> PaddleOcr:
    ocr = PaddleOcr.__new__(PaddleOcr)
    BaseOcr.__init__(ocr)
    ocr.ocr = FakeEngine(list(pages))
    return ocr


def test_init_uses_defaults_and_allows_overrides(monkeypatch) -> None:
    received: dict[str, Any] = {}

    def fake_paddle_ocr(**kwargs: Any) -> object:
        received.update(kwargs)
        return object()

    monkeypatch.setattr(
        "exam_rec.ocr.paddle_ocr.PaddleOCR",
        fake_paddle_ocr,
    )

    ocr = PaddleOcr(lang="en", use_doc_unwarping=True)

    assert ocr.ocr is not None
    assert received == {
        "enable_mkldnn": True,
        "lang": "en",
        "precision": "fp16",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": True,
        "use_textline_orientation": False,
    }


def test_predict_iter_converts_page_result() -> None:
    page = ResultObject(
        {
            "rec_texts": ["first", "second"],
            "rec_polys": np.array(
                [
                    [[1, 2], [3, 2], [3, 4], [1, 4]],
                    [[5, 6], [7, 6], [7, 8], [5, 8]],
                ]
            ),
        }
    )
    ocr = make_ocr(page)

    assert list(ocr.predict_iter("page.png")) == [
        [
            OcrElement(
                bbox=[Point(1, 2), Point(3, 2), Point(3, 4), Point(1, 4)],
                label="text",
                content="first",
            ),
            OcrElement(
                bbox=[Point(5, 6), Point(7, 6), Point(7, 8), Point(5, 8)],
                label="text",
                content="second",
            ),
        ]
    ]
    assert ocr.ocr.received_input == "page.png"


def test_predict_iter_accepts_dict_and_detection_polygon_fallback() -> None:
    ocr = make_ocr(
        {
            "rec_texts": [123],
            "dt_polys": [[[10, 20], [30, 20], [30, 40], [10, 40]]],
        }
    )

    assert ocr.predict("page.png") == [
        [
            OcrElement(
                bbox=[Point(10, 20), Point(30, 20), Point(30, 40), Point(10, 40)],
                label="text",
                content="123",
            )
        ]
    ]


def test_predict_iter_converts_pathlike_input() -> None:
    ocr = make_ocr()

    assert ocr.predict(Path("page.png")) == []
    assert ocr.ocr.received_input == "page.png"


def test_predict_iter_adapts_pdf_bytes_to_temporary_path() -> None:
    class InspectingEngine(FakeEngine):
        received_content: bytes | None = None
        path_existed_during_prediction = False

        def predict_iter(self, input: Any):
            self.received_input = input
            self.path_existed_during_prediction = os.path.exists(input)
            with open(input, "rb") as file:
                self.received_content = file.read()
            yield from self.pages

    ocr = PaddleOcr.__new__(PaddleOcr)
    BaseOcr.__init__(ocr)
    ocr.ocr = InspectingEngine([])

    assert ocr.predict(b"%PDF-test") == []
    assert ocr.ocr.path_existed_during_prediction
    assert ocr.ocr.received_content == b"%PDF-test"
    assert not os.path.exists(ocr.ocr.received_input)


def test_predict_iter_rejects_unsupported_input_type() -> None:
    ocr = make_ocr()

    with pytest.raises(TypeError, match="PaddleOCR input"):
        ocr.predict(object())


def test_temporary_pdf_is_removed_when_iteration_stops_early() -> None:
    ocr = make_ocr({"rec_texts": [], "rec_polys": []})
    iterator = ocr.predict_iter(b"%PDF-test")

    assert next(iterator) == []
    temporary_path = ocr.ocr.received_input
    assert os.path.exists(temporary_path)

    iterator.close()

    assert not os.path.exists(temporary_path)


def test_invalid_result_data_produces_no_results() -> None:
    assert make_ocr(object(), None, {"res": None}).predict("bad") == [[], [], []]


def test_points_ignore_invalid_values() -> None:
    assert PaddleOcr._points([None, "bad", [1], [2, 3], ["x", 4]]) == [
        Point(2, 3)
    ]
