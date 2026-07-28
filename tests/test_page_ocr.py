from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from exam_rec.ocr.base_ocr import BaseOcr, OcrElement, PersistingOcr, Point
from exam_rec.ocr.page_ocr import (
    CachedPageOcr,
    PageCachingOcr,
    PageOcr,
    PageOcrPageCountError,
)


def element(text: str) -> OcrElement:
    return OcrElement(
        bbox=[Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)],
        label="text",
        content=text,
    )


class MemoryPageSource:
    def __init__(self, pages: Sequence[str | None]) -> None:
        self.pages = tuple(pages)
        self.selections: list[tuple[int, ...]] = []

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def select_pages(self, page_indexes: Sequence[int]) -> list[str | None]:
        indexes = tuple(page_indexes)
        self.selections.append(indexes)
        return [self.pages[index] for index in indexes]


class MemoryOcr(BaseOcr):
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, ...]] = []

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        pages = tuple(input)
        self.calls.append(pages)
        for page in pages:
            yield [] if page is None else [element(page)]


class FixedOutputOcr(BaseOcr):
    def __init__(self, pages: Sequence[list[OcrElement]]) -> None:
        self.pages = pages

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        yield from self.pages


def contents(pages: Sequence[list[OcrElement]]) -> list[list[str]]:
    return [[element.content for element in page] for page in pages]


def test_page_ocr_is_bound_to_an_in_memory_page_source() -> None:
    source = MemoryPageSource(["zero", "one", "two"])
    ocr = PageOcr(source, MemoryOcr())

    pages = ocr.predict_pages([2, 0, 2])

    assert ocr.page_count == 3
    assert contents(pages) == [["two"], ["zero"], ["two"]]
    assert source.selections == [(2, 0, 2)]


@pytest.mark.parametrize("page_indexes", [[], [-1], [2], [True], ["0"]])
def test_page_ocr_validates_page_indexes(page_indexes) -> None:
    ocr = PageOcr(MemoryPageSource(["zero", "one"]), MemoryOcr())

    with pytest.raises(ValueError, match="page_indexes"):
        ocr.predict_pages(page_indexes)


@pytest.mark.parametrize(
    "pages, expected, actual",
    [
        ([[]], 2, 1),
        ([[], [], []], 2, 3),
    ],
)
def test_page_ocr_validates_backend_page_count(
    pages: Sequence[list[OcrElement]], expected: int, actual: int
) -> None:
    ocr = PageOcr(
        MemoryPageSource(["zero", "one"]),
        FixedOutputOcr(pages),
    )

    with pytest.raises(PageOcrPageCountError) as mismatch:
        ocr.predict_pages([0, 1])

    assert mismatch.value.expected == expected
    assert mismatch.value.actual == actual


def test_cached_page_ocr_loads_once_and_returns_selected_defensive_copies(
    tmp_path,
) -> None:
    persistence = tmp_path / "ocr.jsonl"
    PersistingOcr(
        FixedOutputOcr([[element("zero")], [], [element("two")]]),
        persistence,
    ).predict(None)

    ocr = CachedPageOcr(persistence)
    persistence.unlink()
    pages = ocr.predict_pages([2, 0, 2, 1])
    pages[0][0].bbox.append(Point(20, 20))
    reread = ocr.predict_pages([2])

    assert ocr.page_count == 3
    assert contents(pages) == [["two"], ["zero"], ["two"], []]
    assert len(pages[0][0].bbox) == 5
    assert len(pages[2][0].bbox) == 4
    assert len(reread[0][0].bbox) == 4


@pytest.mark.parametrize("page_indexes", [[], [-1], [3], [True], ["0"]])
def test_cached_page_ocr_validates_page_indexes(tmp_path, page_indexes) -> None:
    persistence = tmp_path / "ocr.jsonl"
    PersistingOcr(FixedOutputOcr([[], [], []]), persistence).predict(None)
    ocr = CachedPageOcr(persistence)

    with pytest.raises(ValueError, match="page_indexes"):
        ocr.predict_pages(page_indexes)


@pytest.mark.parametrize(
    "content",
    [
        None,
        '{"type":"header","version":1}\n'
        '{"type":"page","elements":[]}\n',
        '{"type":"header","version":2}\n'
        '{"type":"complete"}\n',
        '{"type":"header","version":1}\n'
        'not-json\n'
        '{"type":"complete"}\n',
    ],
)
def test_cached_page_ocr_preserves_cached_ocr_validation(
    tmp_path, content: str | None
) -> None:
    persistence = tmp_path / "ocr.jsonl"
    if content is not None:
        persistence.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete or invalid"):
        CachedPageOcr(persistence)


def test_page_cache_only_recognizes_unique_misses_and_caches_empty_pages() -> None:
    source = MemoryPageSource(["zero", None, "two"])
    model = MemoryOcr()
    ocr = PageCachingOcr(source, model)

    first = ocr.predict_pages([2, 0, 2, 1])
    second = ocr.predict_pages([1, 0])

    assert contents(first) == [["two"], ["zero"], ["two"], []]
    assert contents(second) == [[], ["zero"]]
    assert source.selections == [(2, 0, 1)]
    assert model.calls == [("two", "zero", None)]
    assert ocr.cached_page_count == 3


def test_page_cache_returns_defensive_copies() -> None:
    ocr = PageCachingOcr(MemoryPageSource(["zero"]), MemoryOcr())

    first = ocr.predict_pages([0])
    first[0][0].bbox.append(Point(20, 20))
    second = ocr.predict_pages([0])

    assert len(first[0][0].bbox) == 5
    assert len(second[0][0].bbox) == 4
    assert first[0] is not second[0]


def test_two_page_caching_ocr_instances_do_not_share_cache() -> None:
    source = MemoryPageSource(["zero"])
    model = MemoryOcr()
    first = PageCachingOcr(source, model)
    second = PageCachingOcr(source, model)

    first.predict_pages([0])
    second.predict_pages([0])

    assert model.calls == [("zero",), ("zero",)]


def test_page_cache_uses_lru_and_supports_explicit_invalidation() -> None:
    model = MemoryOcr()
    ocr = PageCachingOcr(
        MemoryPageSource(["zero", "one", "two"]),
        model,
        max_cached_pages=2,
    )

    ocr.predict_pages([0, 1])
    ocr.predict_pages([0])
    ocr.predict_pages([2])
    ocr.predict_pages([1])
    ocr.invalidate_pages([0])
    ocr.predict_pages([0])
    ocr.clear_cache()

    assert model.calls == [
        ("zero", "one"),
        ("two",),
        ("one",),
        ("zero",),
    ]
    assert ocr.cached_page_count == 0
