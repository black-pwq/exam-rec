from collections.abc import Iterable, Iterator

import pytest

from extractor.base_extractor import OcrPage, Problem, ProblemExtractor
from extractor.evaluator import (
    BestExtractorSelector,
    EvaluationReport,
    StructuralProblemEvaluator,
)
from ocr.base_ocr import OcrElement


def page(*texts: str) -> OcrPage:
    return [OcrElement(bbox=[], label="text", content=text) for text in texts]


def problem(
    number: str = "1",
    question: str = "这是一段完整的题目正文",
    options: dict[str, str] | None = None,
) -> Problem:
    return Problem(
        number=number,
        question=question,
        options=(
            options
            if options is not None
            else {"A": "选项甲", "B": "选项乙", "C": "选项丙", "D": "选项丁"}
        ),
        answer="",
        analysis="",
    )


def test_structural_evaluator_returns_detailed_report() -> None:
    evaluator = StructuralProblemEvaluator()
    pages = [page("1. 这是一段完整的题目正文", "A. 选项甲 B. 选项乙 C. 选项丙 D. 选项丁")]

    report = evaluator.evaluate(pages, [problem()])

    assert 0 < report.score <= 1
    assert set(report.metrics) == {
        "question_completeness",
        "option_completeness",
        "text_coverage",
        "structural_consistency",
        "extraction_validity",
    }


def test_empty_extraction_scores_zero() -> None:
    report = StructuralProblemEvaluator().evaluate([page("题目正文")], [])

    assert report.score == 0
    assert report.metrics["extraction_validity"] == 0
    assert report.warnings == ("extractor produced no problems",)


def test_complete_result_scores_higher_than_incomplete_result() -> None:
    evaluator = StructuralProblemEvaluator()
    pages = [page("1. 这是一段完整的题目正文", "A. 甲 B. 乙 C. 丙 D. 丁")]

    complete = evaluator.evaluate(pages, [problem()])
    incomplete = evaluator.evaluate(
        pages,
        [problem(number="", question="题", options={"C": ""})],
    )

    assert complete.score > incomplete.score


class FixedExtractor(ProblemExtractor):
    def __init__(self, problems: Iterable[Problem], *, fail: bool = False) -> None:
        self.problems = tuple(problems)
        self.fail = fail
        self.calls = 0

    def extract_iter(self, pages: Iterable[OcrPage]) -> Iterator[Problem]:
        self.calls += 1
        list(pages)
        if self.fail:
            raise RuntimeError("broken extractor")
        yield from self.problems


def test_selector_returns_best_candidate_without_rerunning_it() -> None:
    weak = FixedExtractor([problem(number="", question="题", options={"C": ""})])
    strong = FixedExtractor([problem()])
    selector = BestExtractorSelector(StructuralProblemEvaluator())

    selected = selector.select(
        [weak, strong],
        iter([page("1. 这是一段完整的题目正文", "A. 甲 B. 乙 C. 丙 D. 丁")]),
    )

    assert selected.extractor is strong
    assert selected.problems == (problem(),)
    assert weak.calls == 1
    assert strong.calls == 1


def test_selector_isolates_failed_extractor() -> None:
    failed = FixedExtractor([], fail=True)
    working = FixedExtractor([problem()])

    selected = BestExtractorSelector(StructuralProblemEvaluator()).select(
        [failed, working],
        [page("1. 这是一段完整的题目正文", "A. 甲 B. 乙 C. 丙 D. 丁")],
    )

    assert selected.extractor is working
    assert failed.calls == 1


def test_evaluation_report_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        EvaluationReport(score=1.1)
