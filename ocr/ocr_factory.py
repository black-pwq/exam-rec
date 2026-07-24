from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from os import PathLike, fspath
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

import pymupdf

from ocr.base_ocr import BaseOcr, TransformedOcr
from ocr.pymu_ocr import PyMuPDFOcr
from transform import MergeCollinearElements


OcrType = type[BaseOcr]


class OcrSelector(Protocol):
    def select(
        self,
        path: str | PathLike[str],
        page_indexes: Iterable[int] | None = None,
    ) -> OcrType: ...


@dataclass(frozen=True)
class PdfTextLayerSelector:
    """Choose an OCR backend by sampling meaningful PDF text layers."""

    max_sample_pages: int = 5
    min_chars_per_page: int = 20
    min_text_page_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.max_sample_pages < 1:
            raise ValueError("max_sample_pages must be at least 1")
        if self.min_chars_per_page < 1:
            raise ValueError("min_chars_per_page must be at least 1")
        if not 0 <= self.min_text_page_ratio <= 1:
            raise ValueError("min_text_page_ratio must be between 0 and 1")

    def select(
        self,
        path: str | PathLike[str],
        page_indexes: Iterable[int] | None = None,
    ) -> OcrType:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"OCR input file does not exist: {source}")

        document = pymupdf.open(fspath(source))
        try:
            if document.needs_pass:
                raise ValueError(f"OCR input file is password protected: {source}")
            if not document.is_pdf or document.page_count == 0:
                return self._paddle_type()

            if page_indexes is None:
                selected_indexes = self._sample_page_indexes(document.page_count)
            else:
                selected_indexes = list(page_indexes)
                if not selected_indexes:
                    raise ValueError("page_indexes must not be empty")
                if any(
                    not isinstance(index, int)
                    or index < 0
                    or index >= document.page_count
                    for index in selected_indexes
                ):
                    raise ValueError("page_indexes contains an invalid PDF page index")
            text_page_count = sum(
                self._has_meaningful_text(document[index])
                for index in selected_indexes
            )
            ratio = text_page_count / len(selected_indexes)
            return PyMuPDFOcr if ratio >= self.min_text_page_ratio else self._paddle_type()
        finally:
            document.close()

    def _has_meaningful_text(self, page: Any) -> bool:
        text = page.get_text("text")
        character_count = sum(not character.isspace() for character in text)
        return character_count >= self.min_chars_per_page

    def _sample_page_indexes(self, page_count: int) -> list[int]:
        sample_count = min(page_count, self.max_sample_pages)
        if sample_count == 1:
            return [0]
        return sorted(
            {
                round(index * (page_count - 1) / (sample_count - 1))
                for index in range(sample_count)
            }
        )

    @staticmethod
    def _paddle_type() -> OcrType:
        from ocr.paddle_ocr import PaddleOcr

        return PaddleOcr


OcrBuilder = Callable[[], BaseOcr]


class OcrRegistry:
    """Lazily create and reuse one configured instance for each OCR type."""

    def __init__(
        self,
        builders: Mapping[OcrType, OcrBuilder] | None = None,
    ) -> None:
        self.builders = dict(builders or {})
        self._instances: dict[OcrType, BaseOcr] = {}
        self._lock = Lock()

    def get(self, ocr_type: OcrType) -> BaseOcr:
        try:
            return self._instances[ocr_type]
        except KeyError:
            pass

        with self._lock:
            instance = self._instances.get(ocr_type)
            if instance is None:
                instance = self._create(ocr_type)
                self._instances[ocr_type] = instance
            return instance

    def _create(self, ocr_type: OcrType) -> BaseOcr:
        builder = self.builders.get(ocr_type)
        source = builder() if builder is not None else ocr_type()
        if issubclass(ocr_type, PyMuPDFOcr):
            return TransformedOcr(source, MergeCollinearElements())
        return source


_DEFAULT_OCR_REGISTRY = OcrRegistry()


class OcrFactory:
    """Select an OCR type for a file and retrieve its shared instance."""

    def __init__(
        self,
        selector: OcrSelector | None = None,
        *,
        registry: OcrRegistry | None = None,
    ) -> None:
        self.selector = selector or PdfTextLayerSelector()
        self.registry = registry or _DEFAULT_OCR_REGISTRY

    def create(
        self,
        path: str | PathLike[str],
        page_indexes: Iterable[int] | None = None,
    ) -> BaseOcr:
        ocr_type = self.selector.select(path, page_indexes)
        return self.registry.get(ocr_type)


_DEFAULT_OCR_FACTORY = OcrFactory(registry=_DEFAULT_OCR_REGISTRY)


def create_ocr(
    path: str | PathLike[str],
    page_indexes: Iterable[int] | None = None,
) -> BaseOcr:
    """Retrieve a shared OCR implementation using the default policy."""
    return _DEFAULT_OCR_FACTORY.create(path, page_indexes)
