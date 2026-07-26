from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from extractor.base_extractor import OcrPage, Problem, ProblemExtractor
from extractor.evaluator import EvaluationReport, ProblemEvaluator
from extractor.regex_extractor import GeneralRegexExtractor, RegexPatterns
from ocr.base_ocr import BaseOcr, OcrElement, Point
from ocr.page_ocr import PageCachingOcr, PdfPageSource
from pipeline import (
    InvalidProcessingRequestError,
    LowConfidenceExtractionError,
    ProblemProcessingPipeline,
    ProcessingPolicy,
    ProcessingRequest,
)
from question_range import (
    QuestionRangeResolver,
    QuestionStartDecision,
)


def make_pdf(path: Path, page_count: int = 3) -> None:
    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page()
    document.save(path)
    document.close()


def element(text: str, x: float = 0, y: float = 10) -> OcrElement:
    return OcrElement(
        bbox=[
            Point(x, y),
            Point(x + 10, y),
            Point(x + 10, y + 10),
            Point(x, y + 10),
        ],
        label="text",
        content=text,
    )


class StubOcr(BaseOcr):
    def __init__(self, pages: Iterable[OcrPage]) -> None:
        self.pages = tuple(pages)
        self.calls = 0
        self.input_page_counts: list[int] = []

    def predict_iter(self, input: Any) -> Iterator[OcrPage]:
        self.calls += 1
        document = pymupdf.open(stream=input, filetype="pdf")
        try:
            page_count = document.page_count
            self.input_page_counts.append(page_count)
        finally:
            document.close()
        yield from self.pages[:page_count]


class FixedScoreEvaluator(ProblemEvaluator):
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}

    def evaluate(self, pages, problems) -> EvaluationReport:
        problem_list = list(problems)
        name = problem_list[0].analysis if problem_list else ""
        return EvaluationReport(score=self.scores.get(name, 1.0))


class NamedExtractor(ProblemExtractor):
    def __init__(self, name: str, question: str = "完整题目") -> None:
        self.name = name
        self.question = question
        self.calls = 0

    def extract_iter(self, pages: Iterable[OcrPage]) -> Iterator[Problem]:
        self.calls += 1
        list(pages)
        yield Problem("1", self.question, "", {"A": "甲", "B": "乙"}, self.name)


class StubAnalyzer:
    def __init__(self) -> None:
        self.samples: list[str] = []

    def analyze_regex_pattern(self, sample_text: str) -> RegexPatterns:
        self.samples.append(sample_text)
        return RegexPatterns(
            question=r"^第(?P<number>\d+)题\s*(?P<question>.*)$",
            options=r"(?P<label>[A-D])[.．](?P<text>.*)$",
        )


class FixedQuestionStartAnalyzer:
    def analyze(self, page_texts) -> QuestionStartDecision:
        return QuestionStartDecision(0, 0.95)


def cached_ocr(
    path: Path,
    ocr: BaseOcr,
    *,
    max_cached_pages: int = PageCachingOcr.DEFAULT_MAX_CACHED_PAGES,
) -> PageCachingOcr:
    return PageCachingOcr(
        PdfPageSource(path),
        ocr,
        max_cached_pages=max_cached_pages,
    )


def request(path: Path, questions: range | None = None) -> ProcessingRequest:
    return ProcessingRequest(path=path, questions=questions or range(0, 3))


def test_request_rejects_invalid_and_overlapping_ranges(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path)
    pipeline = ProblemProcessingPipeline(
        page_ocr=cached_ocr(path, StubOcr([[]] * 3))
    )

    with pytest.raises(InvalidProcessingRequestError, match="step 1"):
        list(pipeline.process_iter(request(path, range(0, 3, 2))))
    with pytest.raises(InvalidProcessingRequestError, match="non-empty"):
        list(
            pipeline.process_iter(
                ProcessingRequest(path=path, questions=range(1, 1))
            )
        )
    with pytest.raises(InvalidProcessingRequestError, match="required"):
        list(
            pipeline.process_iter(
                ProcessingRequest(path=path, questions=None)  # type: ignore[arg-type]
            )
        )
    with pytest.raises(InvalidProcessingRequestError, match="bounds"):
        list(
            pipeline.process_iter(
                ProcessingRequest(path=path, questions=range(0, 4))
            )
        )
    with pytest.raises(InvalidProcessingRequestError, match="overlap"):
        list(
            pipeline.process_iter(
                ProcessingRequest(
                    path=path,
                    questions=range(1, 3),
                    contents=range(0, 2),
                )
            )
        )


def test_uses_first_three_question_pages_and_only_ocrs_remaining_misses(
    tmp_path,
) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path, 6)
    ocr = StubOcr([[element("page")] for _ in range(4)])
    pipeline = ProblemProcessingPipeline(
        page_ocr=cached_ocr(path, ocr),
        evaluator=FixedScoreEvaluator(),
        fixed_extractor_factories=(lambda: NamedExtractor("fixed"),),
    )

    pipeline.process(ProcessingRequest(path=path, questions=range(2, 6)))

    assert ocr.calls == 2
    assert ocr.input_page_counts == [3, 1]


def test_llm_fallback_uses_all_three_sample_pages_once(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path)
    ocr = StubOcr([[element("第1题 一")], [element("")], [element("第2题 二")]])
    analyzer = StubAnalyzer()
    pipeline = ProblemProcessingPipeline(
        page_ocr=cached_ocr(path, ocr),
        evaluator=FixedScoreEvaluator({"fixed": 0.8, "": 0.95}),
        analyzer=analyzer,
        fixed_extractor_factories=(lambda: NamedExtractor("fixed"),),
    )

    pipeline.process(request(path))

    assert len(analyzer.samples) == 1
    assert analyzer.samples[0] == "第1题 一\n\n第2题 二"


def test_low_confidence_fails_before_page_results(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path)
    ocr = StubOcr([[element("text")]] * 3)
    pipeline = ProblemProcessingPipeline(
        page_ocr=cached_ocr(path, ocr),
        evaluator=FixedScoreEvaluator({"fixed": 0.6, "": 0.6}),
        analyzer=StubAnalyzer(),
        fixed_extractor_factories=(lambda: NamedExtractor("fixed"),),
    )

    iterator = pipeline.process_iter(request(path))
    with pytest.raises(LowConfidenceExtractionError):
        next(iterator)
    assert ocr.calls == 1


def test_cross_page_problem_is_assigned_to_page_where_completed(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path)
    pages = [
        [element("第1题 第一页")],
        [element("题干续行"), element("第2题 第二题")],
        [element("第二题续行")],
    ]
    extractor = GeneralRegexExtractor(
        RegexPatterns(
            question=r"^第(?P<number>\d+)题\s*(?P<question>.*)$",
            options=r"(?P<label>[A-D])[.．](?P<text>.*)$",
        )
    )
    pipeline = ProblemProcessingPipeline(
        page_ocr=cached_ocr(path, StubOcr(pages)),
        evaluator=FixedScoreEvaluator(),
        fixed_extractor_factories=(lambda: extractor,),
    )

    result = pipeline.process(request(path))

    assert [page.page_index for page in result.pages] == [0, 1, 2]
    assert [problem.number for problem in result.pages[0].problems] == []
    assert [problem.number for problem in result.pages[1].problems] == ["1"]
    assert [problem.number for problem in result.pages[2].problems] == ["2"]
    assert [problem.number for problem in result.problems] == ["1", "2"]


def test_shared_page_ocr_recognizes_each_page_only_once(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path, 25)
    ocr = StubOcr([[element(f"page {index}")] for index in range(25)])
    page_ocr = cached_ocr(
        path,
        ocr,
        max_cached_pages=23,
    )
    resolver = QuestionRangeResolver(
        FixedQuestionStartAnalyzer(),  # type: ignore[arg-type]
        page_ocr=page_ocr,
    )
    pipeline = ProblemProcessingPipeline(
        page_ocr=page_ocr,
        evaluator=FixedScoreEvaluator(),
        fixed_extractor_factories=(lambda: NamedExtractor("fixed"),),
    )

    try:
        questions = resolver.resolve(path)
        result = pipeline.process(
            ProcessingRequest(path=path, questions=questions),
        )
    finally:
        page_ocr.clear_cache()

    assert questions == range(0, 25)
    assert len(result.pages) == 25
    assert ocr.input_page_counts == [20, 5]
