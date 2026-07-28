from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from exam_rec.ocr.base_ocr import OcrElement

OcrPage = list[OcrElement]


@dataclass
class Problem:
    number: str
    question: str
    answer: str
    options: dict[str, str]
    analysis: str


class ProblemExtractor(ABC):
    def extract(self, pages: Iterable[OcrPage]) -> list[Problem]:
        return list(self.extract_iter(pages))

    @abstractmethod
    def extract_iter(self, pages: Iterable[OcrPage]) -> Iterator[Problem]:
        raise NotImplementedError


@dataclass
class ExtractedPage:
    page_index: int
    problems: list[Problem]


class ContextualProblemExtractor:
    """Attach page indexes to problems produced by another extractor."""

    def __init__(self, source: ProblemExtractor, page_offset: int = 0) -> None:
        if page_offset < 0:
            raise ValueError("page_offset must be non-negative")
        self.source = source
        self.page_offset = page_offset

    def extract(self, pages: Iterable[OcrPage]) -> list[ExtractedPage]:
        return list(self.extract_iter(pages))

    def extract_iter(self, pages: Iterable[OcrPage]) -> Iterator[ExtractedPage]:
        tracked = _TrackedPages(pages)
        problems = self.source.extract_iter(tracked)
        grouped: dict[int, list[Problem]] = {}
        emitted = 0

        try:
            while True:
                try:
                    problem = next(problems)
                except StopIteration:
                    break

                while emitted < tracked.page_count - 1:
                    yield ExtractedPage(
                        page_index=self.page_offset + emitted,
                        problems=grouped.pop(emitted, []),
                    )
                    emitted += 1
                if tracked.page_count == 0:
                    raise RuntimeError("extractor yielded before consuming a page")
                grouped.setdefault(tracked.page_count - 1, []).append(problem)

            for _ in tracked:
                pass
            while emitted < tracked.page_count:
                yield ExtractedPage(
                    page_index=self.page_offset + emitted,
                    problems=grouped.pop(emitted, []),
                )
                emitted += 1
        finally:
            close = getattr(problems, "close", None)
            if close is not None:
                close()
            tracked.close()


class _TrackedPages(Iterator[OcrPage]):
    def __init__(self, pages: Iterable[OcrPage]) -> None:
        self._pages = iter(pages)
        self.page_count = 0

    def __iter__(self) -> "_TrackedPages":
        return self

    def __next__(self) -> OcrPage:
        page = next(self._pages)
        self.page_count += 1
        return page

    def close(self) -> None:
        close = getattr(self._pages, "close", None)
        if close is not None:
            close()


class RawTextExtractor:
    def extract(self, pages: Iterable[OcrPage], delimiter: str = "\n") -> list[str]:
        return list(self.extract_iter(pages, delimiter))

    def extract_iter(
        self, pages: Iterable[OcrPage], delimiter: str = "\n"
    ) -> Iterator[str]:
        for page in pages:
            yield self.extract_page(page, delimiter)

    @staticmethod
    def extract_page(elements: OcrPage, delimiter: str = "\n") -> str:
        """Convert an already-produced OCR page to text without running OCR."""
        return delimiter.join(element.content for element in elements)
