from os import PathLike
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from exam_rec.ocr.base_ocr import BaseOcr, OcrElement, Point, TransformedOcr
from exam_rec.ocr.glm_ocr import GlmOcr
from exam_rec.ocr.ocr_factory import (
    OcrFactory,
    OcrRegistry,
    OcrType,
    PdfTextLayerSelector,
)
from exam_rec.ocr.page_ocr import PageCachingOcr
from exam_rec.ocr.paddle_ocr import PaddleOcr
from exam_rec.ocr.pymu_ocr import PyMuPDFOcr


def make_pdf(path: Path, page_texts: list[str | None]) -> None:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text is not None:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_selector_chooses_pymupdf_for_meaningful_text_layer(tmp_path) -> None:
    path = tmp_path / "text.pdf"
    make_pdf(
        path,
        [
            None,
            "This page contains enough searchable text for direct extraction.",
            "Another page contains enough searchable text for direct extraction.",
        ],
    )

    assert PdfTextLayerSelector().select(path) is PyMuPDFOcr


def test_selector_chooses_configured_fallback_for_sparse_pdf(tmp_path) -> None:
    path = tmp_path / "scanned.pdf"
    make_pdf(path, [None, "page 2", None])
    selector = PdfTextLayerSelector(fallback_ocr_type=DummyOcr)

    assert selector.select(path) is DummyOcr


def test_selector_only_checks_requested_pages(tmp_path) -> None:
    path = tmp_path / "mixed.pdf"
    make_pdf(
        path,
        [
            "This front page contains enough searchable text for extraction.",
            None,
            None,
        ],
    )

    selector = PdfTextLayerSelector(fallback_ocr_type=DummyOcr)

    assert selector.select(path, [0]) is PyMuPDFOcr
    assert selector.select(path, [1, 2]) is DummyOcr


def test_selector_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        PdfTextLayerSelector().select(tmp_path / "missing.pdf")


class FixedSelector:
    def __init__(self, ocr_type: OcrType) -> None:
        self.ocr_type = ocr_type
        self.page_indexes = None

    def select(self, path: str | PathLike[str], page_indexes=None) -> OcrType:
        self.page_indexes = page_indexes
        return self.ocr_type


class DummyOcr(BaseOcr):
    def __init__(self, **options: Any) -> None:
        self.options = options

    def predict_iter(self, input: Any):
        yield []


def test_factory_uses_selected_builder_and_options(tmp_path) -> None:
    selector = FixedSelector(DummyOcr)
    registry = OcrRegistry({DummyOcr: lambda: DummyOcr(lang="ch")})
    factory = OcrFactory(
        selector,
        registry=registry,
    )

    indexes = [2, 3, 4]
    ocr = factory.create("input.pdf", indexes)

    assert isinstance(ocr, TransformedOcr)
    assert isinstance(ocr.source, DummyOcr)
    assert ocr.source.options == {"lang": "ch"}
    assert selector.page_indexes is indexes


def test_factory_creates_independent_document_bound_page_caches(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path, [None])
    registry = OcrRegistry({DummyOcr: DummyOcr})
    factory = OcrFactory(FixedSelector(DummyOcr), registry=registry)

    first = factory.create_page_ocr(path, [0], max_cached_pages=1)
    second = factory.create_page_ocr(path, [0], max_cached_pages=1)

    assert isinstance(first, PageCachingOcr)
    assert first is not second
    assert first.ocr is second.ocr
    assert first.document is not second.document
    assert first.predict_pages([0]) == [[]]


def test_registry_lazily_reuses_one_instance_per_type() -> None:
    calls = {DummyOcr: 0, PaddleOcr: 0}

    def build_dummy() -> BaseOcr:
        calls[DummyOcr] += 1
        return DummyOcr(backend="dummy")

    def build_paddle(**options: Any) -> BaseOcr:
        calls[PaddleOcr] += 1
        return DummyOcr(backend="paddle", **options)

    registry = OcrRegistry(
        {
            DummyOcr: build_dummy,
            PaddleOcr: lambda: build_paddle(lang="ch"),
        }
    )

    dummy = registry.get(DummyOcr)
    paddle = registry.get(PaddleOcr)

    assert registry.get(DummyOcr) is dummy
    assert registry.get(PaddleOcr) is paddle
    assert calls == {DummyOcr: 1, PaddleOcr: 1}
    assert isinstance(dummy, TransformedOcr)
    assert isinstance(paddle, TransformedOcr)
    assert isinstance(paddle.source, DummyOcr)
    assert paddle.source.options == {"backend": "paddle", "lang": "ch"}


def test_factory_reuses_registry_instance_across_files() -> None:
    registry = OcrRegistry({DummyOcr: DummyOcr})
    factory = OcrFactory(FixedSelector(DummyOcr), registry=registry)

    assert factory.create("first.pdf") is factory.create("second.pdf")


def test_default_factories_share_global_registry() -> None:
    first = OcrFactory(FixedSelector(PyMuPDFOcr))
    second = OcrFactory(FixedSelector(PyMuPDFOcr))

    assert first.create("first.pdf") is second.create("second.pdf")


def test_registry_wraps_pymupdf_with_collinear_merge() -> None:
    class PymuStub(PyMuPDFOcr):
        def predict_iter(self, input: Any):
            yield [
                OcrElement(
                    bbox=[Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)],
                    label="text",
                    content="left",
                ),
                OcrElement(
                    bbox=[Point(11, 0), Point(20, 0), Point(20, 10), Point(11, 10)],
                    label="text",
                    content=" right",
                ),
            ]

    ocr = OcrRegistry().get(PymuStub)

    assert isinstance(ocr, TransformedOcr)
    assert [item.content for item in ocr.predict(None)[0]] == ["left right"]


def test_registry_wraps_glm_with_multiline_split_and_forwards_close() -> None:
    class GlmStub(GlmOcr):
        def __init__(self) -> None:
            self.closed = False

        def predict_iter(self, input: Any):
            yield [
                OcrElement(
                    bbox=[
                        Point(0, 0),
                        Point(100, 0),
                        Point(100, 30),
                        Point(0, 30),
                    ],
                    label="text",
                    content="question\nA. first\nB. second",
                ),
                OcrElement(
                    bbox=[],
                    label="table",
                    content="<tr>\n<td>value</td>",
                ),
            ]

        def close(self) -> None:
            self.closed = True

    source = GlmStub()
    registry = OcrRegistry({GlmStub: lambda: source})
    ocr = registry.get(GlmStub)

    assert isinstance(ocr, TransformedOcr)
    assert [item.content for item in ocr.predict(None)[0]] == [
        "question",
        "A. first",
        "B. second",
    ]

    registry.close()

    assert source.closed


def test_registry_normalizes_fullwidth_ascii_for_every_ocr_type() -> None:
    class FullwidthOcr(BaseOcr):
        def predict_iter(self, input: Any):
            yield [
                OcrElement(
                    bbox=[],
                    label="text",
                    content="Ａ．答案：１２３　（测试）",
                )
            ]

    ocr = OcrRegistry().get(FullwidthOcr)

    assert ocr.predict(None)[0][0].content == "A.答案:123 (测试)"
