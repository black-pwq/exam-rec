from collections.abc import Iterator
from typing import Any

import pytest

from exam_rec.extractor.base_extractor import (
    ContextualProblemExtractor,
    OcrPage,
    Problem,
    ProblemExtractor,
    RawTextExtractor,
)
from exam_rec.ocr.base_ocr import BaseOcr, OcrElement


class StubOcr(BaseOcr):
    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        yield [
            OcrElement(bbox=[], label="text", content="page 1 line 1"),
            OcrElement(bbox=[], label="text", content="page 1 line 2"),
        ]
        yield []
        yield [OcrElement(bbox=[], label="text", content="page 3")]


def test_extract_preserves_pages() -> None:
    extractor = RawTextExtractor()

    assert extractor.extract(StubOcr().predict_iter("input.pdf")) == [
        "page 1 line 1\npage 1 line 2",
        "",
        "page 3",
    ]


def test_extract_iter_supports_custom_in_page_delimiter() -> None:
    extractor = RawTextExtractor()

    pages = StubOcr().predict_iter("input.pdf")
    assert list(extractor.extract_iter(pages, delimiter=" | ")) == [
        "page 1 line 1 | page 1 line 2",
        "",
        "page 3",
    ]


def test_extract_page_does_not_run_ocr() -> None:
    page = [
        OcrElement(bbox=[], label="text", content="first"),
        OcrElement(bbox=[], label="text", content="second"),
    ]

    assert RawTextExtractor.extract_page(page, " | ") == "first | second"


class BoundaryExtractor(ProblemExtractor):
    def extract_iter(self, pages: Iterator[OcrPage]) -> Iterator[Problem]:
        current: Problem | None = None
        for page in pages:
            for item in page:
                if item.content.startswith("Q"):
                    if current is not None:
                        yield current
                    current = Problem(
                        number=item.content[1:],
                        question=item.content,
                        answer="",
                        options={},
                        analysis="",
                    )
                elif current is not None:
                    current.question += item.content
        if current is not None:
            yield current


def test_contextual_extractor_assigns_cross_page_and_final_problems() -> None:
    pages = [
        [OcrElement(bbox=[], label="text", content="Q1")],
        [
            OcrElement(bbox=[], label="text", content=" continuation"),
            OcrElement(bbox=[], label="text", content="Q2"),
        ],
        [OcrElement(bbox=[], label="text", content=" final")],
    ]

    extracted = ContextualProblemExtractor(
        BoundaryExtractor(), page_offset=10
    ).extract(pages)

    assert [page.page_index for page in extracted] == [10, 11, 12]
    assert [problem.number for problem in extracted[0].problems] == []
    assert [problem.number for problem in extracted[1].problems] == ["1"]
    assert [problem.number for problem in extracted[2].problems] == ["2"]


def test_contextual_extractor_preserves_pages_without_problems() -> None:
    class EmptyExtractor(ProblemExtractor):
        def extract_iter(self, pages: Iterator[OcrPage]) -> Iterator[Problem]:
            for _ in pages:
                pass
            return
            yield  # pragma: no cover

    extracted = ContextualProblemExtractor(EmptyExtractor()).extract([[], [], []])

    assert [(page.page_index, page.problems) for page in extracted] == [
        (0, []),
        (1, []),
        (2, []),
    ]


def test_contextual_extractor_rejects_negative_page_offset() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ContextualProblemExtractor(BoundaryExtractor(), page_offset=-1)
