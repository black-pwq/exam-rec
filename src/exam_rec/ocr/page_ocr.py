from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from os import PathLike, fspath
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import pymupdf

from exam_rec.app_logging import get_logger
from exam_rec.ocr.base_ocr import BaseOcr, OcrElement
from exam_rec.ocr.cached_ocr import CachedOcr


logger = get_logger(__name__)

OcrPage: TypeAlias = list[OcrElement]
PdfInput: TypeAlias = str | PathLike[str] | bytes | bytearray | memoryview


class PageSource(Protocol):
    """A stable, page-addressable source bound to a PageOcr instance."""

    @property
    def page_count(self) -> int: ...

    def select_pages(self, page_indexes: Sequence[int]) -> Any: ...


class PageOcrPageCountError(RuntimeError):
    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"OCR returned {actual} pages for a request containing {expected} pages"
        )


class PdfPageSource:
    """Expose a PDF path or in-memory PDF snapshot as a page source."""

    def __init__(self, source: PdfInput) -> None:
        if isinstance(source, (bytes, bytearray, memoryview)):
            self._source: Path | bytes = bytes(source)
        elif isinstance(source, (str, PathLike)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"PDF file does not exist: {path}")
            self._source = path
        else:
            raise TypeError("PDF source must be a path or PDF bytes")

        document = self._open()
        try:
            self._validate_document(document)
            self._page_count = document.page_count
        finally:
            document.close()

    @property
    def page_count(self) -> int:
        return self._page_count

    @property
    def source(self) -> Path | bytes:
        return self._source

    def select_pages(self, page_indexes: Sequence[int]) -> bytes:
        indexes = tuple(page_indexes)
        self._validate_page_indexes(indexes)

        document = self._open()
        selected = pymupdf.open()
        try:
            self._validate_document(document)
            if document.page_count != self.page_count:
                raise ValueError("PDF page count changed after the page source was created")
            for page_index in indexes:
                selected.insert_pdf(
                    document,
                    from_page=page_index,
                    to_page=page_index,
                )
            return selected.tobytes()
        finally:
            selected.close()
            document.close()

    def _open(self) -> Any:
        if isinstance(self._source, Path):
            return pymupdf.open(fspath(self._source))
        return pymupdf.open(stream=self._source, filetype="pdf")

    def _validate_page_indexes(self, page_indexes: Sequence[int]) -> None:
        if not page_indexes:
            raise ValueError("page_indexes must not be empty")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= self.page_count
            for index in page_indexes
        ):
            raise ValueError("page_indexes contains an invalid PDF page index")

    @staticmethod
    def _validate_document(document: Any) -> None:
        if document.needs_pass:
            raise ValueError("PDF is password protected")
        if not document.is_pdf or document.page_count == 0:
            raise ValueError("input is not a non-empty PDF")


class PageOcr:
    """Recognize selected pages from one bound document with one OCR model."""

    def __init__(self, document: PageSource, ocr: BaseOcr) -> None:
        page_count = document.page_count
        if (
            isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or page_count < 0
        ):
            raise ValueError("document.page_count must be a non-negative integer")
        self.document = document
        self.ocr = ocr

    @property
    def page_count(self) -> int:
        return self.document.page_count

    def predict_pages(self, page_indexes: Iterable[int]) -> list[OcrPage]:
        return list(self.predict_pages_iter(page_indexes))

    def predict_pages_iter(self, page_indexes: Iterable[int]) -> Iterator[OcrPage]:
        indexes = self._validate_page_indexes(page_indexes)
        selected_input = self.document.select_pages(indexes)
        pages = iter(self.ocr.predict_iter(selected_input))
        actual = 0
        try:
            while actual < len(indexes):
                try:
                    page = next(pages)
                except StopIteration:
                    raise PageOcrPageCountError(len(indexes), actual) from None
                actual += 1
                yield page

            try:
                next(pages)
            except StopIteration:
                return
            raise PageOcrPageCountError(len(indexes), actual + 1)
        finally:
            close = getattr(pages, "close", None)
            if close is not None:
                close()

    def _validate_page_indexes(self, page_indexes: Iterable[int]) -> tuple[int, ...]:
        indexes = tuple(page_indexes)
        if not indexes:
            raise ValueError("page_indexes must not be empty")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= self.page_count
            for index in indexes
        ):
            raise ValueError("page_indexes contains an invalid page index")
        return indexes


_PageSnapshot: TypeAlias = tuple[OcrElement, ...]


class PageCachingOcr(PageOcr):
    """Cache OCR page results within one document-bound instance."""

    DEFAULT_MAX_CACHED_PAGES = 23

    def __init__(
        self,
        document: PageSource,
        ocr: BaseOcr,
        *,
        max_cached_pages: int = DEFAULT_MAX_CACHED_PAGES,
    ) -> None:
        super().__init__(document, ocr)
        if (
            isinstance(max_cached_pages, bool)
            or not isinstance(max_cached_pages, int)
            or max_cached_pages < 1
        ):
            raise ValueError("max_cached_pages must be a positive integer")
        self.max_cached_pages = max_cached_pages
        self._cache: OrderedDict[int, _PageSnapshot] = OrderedDict()

    @property
    def cached_page_count(self) -> int:
        return len(self._cache)

    def predict_pages_iter(self, page_indexes: Iterable[int]) -> Iterator[OcrPage]:
        indexes = self._validate_page_indexes(page_indexes)
        resolved: dict[int, _PageSnapshot] = {}
        missing_indexes: list[int] = []

        for page_index in dict.fromkeys(indexes):
            cached = self._cache.get(page_index)
            if cached is None:
                missing_indexes.append(page_index)
                continue
            self._cache.move_to_end(page_index)
            resolved[page_index] = cached

        logger.info(
            "Page OCR cache request: ocr=%s pages=%d hits=%d misses=%d",
            type(self.ocr).__name__,
            len(indexes),
            len(indexes) - len(missing_indexes),
            len(missing_indexes),
        )

        missing_pages = (
            iter(super().predict_pages_iter(missing_indexes))
            if missing_indexes
            else None
        )
        try:
            missing_position = 0
            for page_index in indexes:
                snapshot = resolved.get(page_index)
                if snapshot is None:
                    if (
                        missing_pages is None
                        or missing_position >= len(missing_indexes)
                        or missing_indexes[missing_position] != page_index
                    ):
                        raise AssertionError("invalid page cache miss ordering")
                    page = next(missing_pages)
                    snapshot = self._snapshot(page)
                    resolved[page_index] = snapshot
                    self._put(page_index, snapshot)
                    missing_position += 1
                yield self._restore(snapshot)

            if missing_pages is not None:
                try:
                    next(missing_pages)
                except StopIteration:
                    pass
        finally:
            if missing_pages is not None:
                close = getattr(missing_pages, "close", None)
                if close is not None:
                    close()

    def clear_cache(self) -> None:
        self._cache.clear()

    def invalidate_pages(self, page_indexes: Iterable[int]) -> None:
        indexes = self._validate_page_indexes(page_indexes)
        for page_index in indexes:
            self._cache.pop(page_index, None)

    def _put(self, page_index: int, snapshot: _PageSnapshot) -> None:
        self._cache[page_index] = snapshot
        self._cache.move_to_end(page_index)
        while len(self._cache) > self.max_cached_pages:
            self._cache.popitem(last=False)

    @staticmethod
    def _snapshot(page: OcrPage) -> _PageSnapshot:
        return tuple(PageCachingOcr._copy_element(element) for element in page)

    @staticmethod
    def _restore(snapshot: _PageSnapshot) -> OcrPage:
        return [PageCachingOcr._copy_element(element) for element in snapshot]

    @staticmethod
    def _copy_element(element: OcrElement) -> OcrElement:
        return OcrElement(
            bbox=list(element.bbox),
            label=element.label,
            content=element.content,
        )


class _CachedOcrPageSource:
    def __init__(self, source: str | PathLike[str]) -> None:
        self._source = Path(source)
        pages = CachedOcr().predict(self._source)
        self._pages = tuple(PageCachingOcr._snapshot(page) for page in pages)

    @property
    def source(self) -> Path:
        return self._source

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def select_pages(self, page_indexes: Sequence[int]) -> list[OcrPage]:
        return [
            PageCachingOcr._restore(self._pages[page_index])
            for page_index in page_indexes
        ]


class _SelectedPageOcr(BaseOcr):
    def predict_iter(self, input: Any) -> Iterator[OcrPage]:
        yield from input


class CachedPageOcr(PageOcr):
    """Expose a complete CachedOcr JSONL file as a page-addressable source."""

    def __init__(self, source: str | PathLike[str]) -> None:
        super().__init__(_CachedOcrPageSource(source), _SelectedPageOcr())
