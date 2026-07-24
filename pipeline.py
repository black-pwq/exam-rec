from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pymupdf

from extractor.base_extractor import (
    ContextualProblemExtractor,
    OcrPage,
    Problem,
    ProblemExtractor,
    RawTextExtractor,
)
from extractor.evaluator import (
    BestExtractorSelector,
    EvaluationReport,
    ExtractionCandidate,
    ProblemEvaluator,
    StructuralProblemEvaluator,
)
from extractor.regex_extractor import (
    GeneralRegexExtractor,
    HuaShengRegexExtractor,
    HuaShengYanyu700RegexExtractor,
    RegexPatterns,
)
from ocr.ocr_factory import OcrFactory
from utils.pdf import select_pdf_pages


class ProcessingError(RuntimeError):
    pass


class InvalidProcessingRequestError(ProcessingError, ValueError):
    pass


class InvalidPdfError(ProcessingError, ValueError):
    pass


class LlmFallbackUnavailableError(ProcessingError):
    pass


class LlmFallbackError(ProcessingError):
    pass


class LowConfidenceExtractionError(ProcessingError):
    def __init__(self, report: EvaluationReport) -> None:
        self.report = report
        super().__init__(
            f"best extractor score {report.score:.6f} is below the minimum"
        )


@dataclass(frozen=True)
class ProcessingRequest:
    path: Path
    questions: range
    front_matter: range | None = None
    contents: range | None = None
    answers: range | None = None


@dataclass(frozen=True)
class ProcessingPolicy:
    sample_page_count: int = 3
    llm_fallback_threshold: float = 0.90
    minimum_acceptable_score: float = 0.70

    def __post_init__(self) -> None:
        if self.sample_page_count < 1:
            raise ValueError("sample_page_count must be positive")
        if not 0 <= self.minimum_acceptable_score <= 1:
            raise ValueError("minimum_acceptable_score must be between 0 and 1")
        if not 0 <= self.llm_fallback_threshold <= 1:
            raise ValueError("llm_fallback_threshold must be between 0 and 1")
        if self.minimum_acceptable_score > self.llm_fallback_threshold:
            raise ValueError(
                "minimum_acceptable_score must not exceed llm_fallback_threshold"
            )


@dataclass(frozen=True)
class PageProcessingResult:
    page_index: int
    problems: tuple[Problem, ...]
    extractor_name: str
    evaluation: EvaluationReport


@dataclass(frozen=True)
class ProcessingResult:
    pages: tuple[PageProcessingResult, ...]
    problems: tuple[Problem, ...]
    extractor_name: str
    evaluation: EvaluationReport


class RegexAnalyzer(Protocol):
    def analyze_regex_pattern(self, sample_text: str) -> RegexPatterns: ...


ExtractorFactory = Callable[[], ProblemExtractor]


class ProblemProcessingPipeline:
    def __init__(
        self,
        *,
        ocr_factory: OcrFactory | None = None,
        evaluator: ProblemEvaluator | None = None,
        analyzer: RegexAnalyzer | None = None,
        policy: ProcessingPolicy | None = None,
        fixed_extractor_factories: Sequence[ExtractorFactory] | None = None,
    ) -> None:
        self.ocr_factory = ocr_factory or OcrFactory()
        self.evaluator = evaluator or StructuralProblemEvaluator()
        self.analyzer = analyzer
        self.policy = policy or ProcessingPolicy()
        self.fixed_extractor_factories = tuple(
            fixed_extractor_factories
            or (HuaShengRegexExtractor, HuaShengYanyu700RegexExtractor)
        )
        if not self.fixed_extractor_factories:
            raise ValueError("at least one fixed extractor factory is required")

    def process(self, request: ProcessingRequest) -> ProcessingResult:
        pages = tuple(self.process_iter(request))
        if not pages:
            raise ProcessingError("processing produced no page results")
        first = pages[0]
        return ProcessingResult(
            pages=pages,
            problems=tuple(problem for page in pages for problem in page.problems),
            extractor_name=first.extractor_name,
            evaluation=first.evaluation,
        )

    def process_iter(
        self, request: ProcessingRequest
    ) -> Iterator[PageProcessingResult]:
        path = Path(request.path)
        self._validate_request(request, path)
        question_indexes = list(request.questions)
        sample_indexes = question_indexes[: self.policy.sample_page_count]
        ocr = self.ocr_factory.create(path, sample_indexes)

        sample_pdf_bytes = select_pdf_pages(path, sample_indexes)
        sample_pages = ocr.predict(sample_pdf_bytes)
        if len(sample_pages) != len(sample_indexes):
            raise InvalidPdfError("OCR returned fewer pages than the sample page range")
        candidate = self._select_extractor(sample_pages)
        if candidate.report.score < self.policy.minimum_acceptable_score:
            raise LowConfidenceExtractionError(candidate.report)

        questions_pdf_bytes = select_pdf_pages(path, question_indexes)
        contextual = ContextualProblemExtractor(
            candidate.extractor,
            page_offset=request.questions.start,
        )
        for extracted_page in contextual.extract_iter(
            ocr.predict_iter(questions_pdf_bytes)
        ):
            yield PageProcessingResult(
                page_index=extracted_page.page_index,
                problems=tuple(extracted_page.problems),
                extractor_name=type(candidate.extractor).__name__,
                evaluation=candidate.report,
            )

    def _select_extractor(self, sample_pages: Sequence[OcrPage]) -> ExtractionCandidate:
        selector = BestExtractorSelector(self.evaluator)
        fixed = selector.rank(
            (factory() for factory in self.fixed_extractor_factories), sample_pages
        )[0]
        if fixed.report.score >= self.policy.llm_fallback_threshold:
            return fixed
        if self.analyzer is None:
            raise LlmFallbackUnavailableError(
                "analyzer is required when fixed extractors score below the threshold"
            )

        sample_text = "\n".join(
            RawTextExtractor.extract_page(page) for page in sample_pages
        )
        try:
            patterns = self.analyzer.analyze_regex_pattern(sample_text)
            general = selector.evaluate(GeneralRegexExtractor(patterns), sample_pages)
        except Exception as error:
            raise LlmFallbackError(
                f"failed to build the LLM regex extractor: {error}"
            ) from error
        return max((fixed, general), key=lambda item: item.report.score)

    def _validate_request(self, request: ProcessingRequest, path: Path) -> int:
        if not path.is_file():
            raise InvalidPdfError(f"PDF file does not exist: {path}")
        try:
            document = pymupdf.open(path)
        except Exception as error:
            raise InvalidPdfError(f"cannot open PDF: {path}") from error
        try:
            if not document.is_pdf or document.page_count == 0:
                raise InvalidPdfError(f"input is not a non-empty PDF: {path}")
            if document.needs_pass:
                raise InvalidPdfError(f"PDF is password protected: {path}")
            page_count = document.page_count
        finally:
            document.close()

        named_ranges = {
            "questions": request.questions,
            "front_matter": request.front_matter,
            "contents": request.contents,
            "answers": request.answers,
        }
        if request.questions is None:
            raise InvalidProcessingRequestError("questions range is required")
        present = []
        for name, pages in named_ranges.items():
            if pages is None:
                continue
            if not isinstance(pages, range):
                raise InvalidProcessingRequestError(f"{name} must be a range")
            if pages.step != 1 or len(pages) == 0:
                raise InvalidProcessingRequestError(
                    f"{name} must be a non-empty range with step 1"
                )
            if pages.start < 0 or pages.stop > page_count:
                raise InvalidProcessingRequestError(
                    f"{name} range exceeds the PDF page bounds"
                )
            present.append((name, pages))

        for index, (left_name, left) in enumerate(present):
            for right_name, right in present[index + 1 :]:
                if left.start < right.stop and right.start < left.stop:
                    raise InvalidProcessingRequestError(
                        f"{left_name} and {right_name} ranges overlap"
                    )
        return page_count
