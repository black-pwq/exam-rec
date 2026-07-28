from pathlib import Path
from typing import Any

from exam_rec.ocr.base_ocr import OcrElement, Point
from exam_rec.ocr.pymu_ocr import PymuOcr


class FakePage:
    def __init__(self, data: Any) -> None:
        self.data = data
        self.calls: list[tuple[str, bool]] = []

    def get_text(self, mode: str, *, sort: bool) -> Any:
        self.calls.append((mode, sort))
        return self.data


class FakeDocument:
    def __init__(self, *pages: FakePage) -> None:
        self.pages = pages
        self.closed = False

    def __iter__(self):
        return iter(self.pages)

    def close(self) -> None:
        self.closed = True


def test_predict_iter_extracts_text_lines(monkeypatch) -> None:
    page = FakePage(
        {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "bbox": (1, 2, 30, 12),
                            "spans": [{"text": "hello "}, {"text": "world"}],
                        },
                        {"bbox": (1, 20, 10, 30), "spans": [{"text": "  "}]},
                    ],
                },
                {"type": 1, "lines": [{"spans": [{"text": "image"}]}]},
            ]
        }
    )
    document = FakeDocument(page)
    received: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_open(*args: Any, **kwargs: Any) -> FakeDocument:
        received.append((args, kwargs))
        return document

    monkeypatch.setattr("exam_rec.ocr.pymu_ocr.pymupdf.open", fake_open)

    assert PymuOcr().predict(Path("sample.pdf")) == [
        [
            OcrElement(
                bbox=[Point(1, 2), Point(30, 2), Point(30, 12), Point(1, 12)],
                label="text",
                content="hello world",
            )
        ]
    ]
    assert received == [(('sample.pdf',), {})]
    assert page.calls == [("dict", True)]
    assert document.closed


def test_bytes_input_and_span_bbox_fallback(monkeypatch) -> None:
    page = FakePage(
        {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"text": "a", "bbox": (5, 6, 10, 12)},
                                {"text": "b", "bbox": (12, 4, 20, 14)},
                            ]
                        }
                    ]
                }
            ]
        }
    )
    document = FakeDocument(page)
    received: dict[str, Any] = {}

    def fake_open(**kwargs: Any) -> FakeDocument:
        received.update(kwargs)
        return document

    monkeypatch.setattr("exam_rec.ocr.pymu_ocr.pymupdf.open", fake_open)

    assert PymuOcr().predict(b"pdf") == [
        [
            OcrElement(
                bbox=[Point(5, 4), Point(20, 4), Point(20, 14), Point(5, 14)],
                label="text",
                content="ab",
            )
        ]
    ]
    assert received == {"stream": b"pdf", "filetype": "pdf"}


def test_document_closes_when_iteration_stops_early(monkeypatch) -> None:
    document = FakeDocument(FakePage({"blocks": []}), FakePage({"blocks": []}))
    monkeypatch.setattr(
        "exam_rec.ocr.pymu_ocr.pymupdf.open",
        lambda *args: document,
    )
    iterator = PymuOcr().predict_iter("sample.pdf")

    assert next(iterator) == []
    iterator.close()

    assert document.closed
