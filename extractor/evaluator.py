from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from extractor.base_extractor import OcrPage, Problem, ProblemExtractor
from ocr.base_ocr import OcrElement


@dataclass(frozen=True)
class EvaluationReport:
    score: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("evaluation score must be between 0 and 1")
        if any(not 0 <= value <= 1 for value in self.metrics.values()):
            raise ValueError("evaluation metrics must be between 0 and 1")


class ProblemEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        pages: Iterable[OcrPage],
        problems: Iterable[Problem],
    ) -> EvaluationReport:
        raise NotImplementedError


@dataclass(frozen=True)
class StructuralEvaluationConfig:
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "question_completeness": 0.30,
            "option_completeness": 0.30,
            "text_coverage": 0.20,
            "structural_consistency": 0.15,
            "extraction_validity": 0.05,
        }
    )
    minimum_question_chars: int = 8
    minimum_options: int = 2
    maximum_options: int = 6

    def __post_init__(self) -> None:
        expected = {
            "question_completeness",
            "option_completeness",
            "text_coverage",
            "structural_consistency",
            "extraction_validity",
        }
        if set(self.weights) != expected:
            raise ValueError(f"weights must contain exactly: {sorted(expected)}")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("evaluation weights must be non-negative")
        if not math.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("evaluation weights must sum to 1")
        if self.minimum_question_chars < 1:
            raise ValueError("minimum_question_chars must be positive")
        if not 1 <= self.minimum_options <= self.maximum_options:
            raise ValueError("option limits are invalid")


class StructuralProblemEvaluator(ProblemEvaluator):
    """Estimate extraction quality without requiring ground-truth problems."""

    _OPTION_MARKER = re.compile(r"(?:^|\s)[A-Z][.．、:]", re.IGNORECASE)

    def __init__(self, config: StructuralEvaluationConfig | None = None) -> None:
        self.config = config or StructuralEvaluationConfig()

    def evaluate(
        self,
        pages: Iterable[OcrPage],
        problems: Iterable[Problem],
    ) -> EvaluationReport:
        page_list = list(pages)
        problem_list = list(problems)
        if not problem_list:
            return EvaluationReport(
                score=0.0,
                metrics=self._zero_metrics(),
                warnings=("extractor produced no problems",),
            )

        metrics = {
            "question_completeness": self._question_completeness(problem_list),
            "option_completeness": self._option_completeness(problem_list),
            "text_coverage": self._text_coverage(page_list, problem_list),
            "structural_consistency": self._structural_consistency(problem_list),
            "extraction_validity": self._extraction_validity(problem_list),
        }
        score = math.prod(
            max(metrics[name], 1e-6) ** weight
            for name, weight in self.config.weights.items()
        )
        return EvaluationReport(
            score=min(max(score, 0.0), 1.0),
            metrics=metrics,
            warnings=self._warnings(metrics, problem_list),
        )

    def _question_completeness(self, problems: Sequence[Problem]) -> float:
        scores = []
        for problem in problems:
            text = self._normalized_text(problem.question)
            length_score = min(len(text) / self.config.minimum_question_chars, 1.0)
            marker_penalty = 0.5 if self._OPTION_MARKER.search(problem.question) else 1.0
            scores.append(length_score * marker_penalty)
        return self._mean(scores)

    def _option_completeness(self, problems: Sequence[Problem]) -> float:
        scores = []
        for problem in problems:
            labels = [str(label).strip().upper() for label in problem.options]
            count = len(labels)
            if count == 0:
                scores.append(0.0)
                continue

            count_score = min(count / self.config.minimum_options, 1.0)
            if count > self.config.maximum_options:
                count_score *= self.config.maximum_options / count
            expected_labels = [chr(ord("A") + index) for index in range(count)]
            label_score = 1.0 if labels == expected_labels else 0.5
            text_score = self._mean(
                [bool(self._normalized_text(text)) for text in problem.options.values()]
            )
            scores.append((count_score + label_score + text_score) / 3)
        return self._mean(scores)

    def _text_coverage(
        self, pages: Sequence[OcrPage], problems: Sequence[Problem]
    ) -> float:
        source = "".join(
            self._normalized_text(element.content)
            for page in pages
            for element in page
        )
        if not source:
            return 0.0
        extracted = "".join(
            self._normalized_text(text)
            for problem in problems
            for text in self._problem_texts(problem)
        )
        overlap = Counter(source) & Counter(extracted)
        return sum(overlap.values()) / len(source)

    @classmethod
    def _structural_consistency(cls, problems: Sequence[Problem]) -> float:
        numbers = [cls._normalized_text(problem.number) for problem in problems]
        nonempty = [number for number in numbers if number]
        presence = len(nonempty) / len(numbers)
        uniqueness = len(set(nonempty)) / len(nonempty) if nonempty else 0.0

        numeric = [int(number) for number in nonempty if number.isdigit()]
        if len(numeric) <= 1:
            ordering = 1.0 if numeric else 0.0
        else:
            ordering = cls._mean(
                [right > left for left, right in zip(numeric, numeric[1:])]
            )
        return (presence + uniqueness + ordering) / 3

    @classmethod
    def _extraction_validity(cls, problems: Sequence[Problem]) -> float:
        valid = [
            bool(
                cls._normalized_text(problem.question)
                or any(cls._normalized_text(text) for text in problem.options.values())
            )
            for problem in problems
        ]
        return cls._mean(valid)

    @staticmethod
    def _problem_texts(problem: Problem) -> tuple[str, ...]:
        return (
            problem.number,
            problem.question,
            *problem.options.keys(),
            *problem.options.values(),
            problem.answer,
            problem.analysis,
        )

    @staticmethod
    def _normalized_text(value: str) -> str:
        return "".join(character.casefold() for character in value if character.isalnum())

    @staticmethod
    def _mean(values: Sequence[float | bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _zero_metrics() -> dict[str, float]:
        return {
            "question_completeness": 0.0,
            "option_completeness": 0.0,
            "text_coverage": 0.0,
            "structural_consistency": 0.0,
            "extraction_validity": 0.0,
        }

    @staticmethod
    def _warnings(
        metrics: Mapping[str, float], problems: Sequence[Problem]
    ) -> tuple[str, ...]:
        warnings = []
        if metrics["question_completeness"] < 0.6:
            warnings.append("many questions appear incomplete")
        if metrics["option_completeness"] < 0.6:
            warnings.append("many option sets appear incomplete")
        if metrics["text_coverage"] < 0.5:
            warnings.append("extracted problems cover little OCR text")
        numbers = [problem.number for problem in problems if problem.number]
        if len(numbers) != len(set(numbers)):
            warnings.append("duplicate problem numbers detected")
        return tuple(warnings)


@dataclass(frozen=True)
class ExtractionCandidate:
    extractor: ProblemExtractor
    problems: tuple[Problem, ...]
    report: EvaluationReport


class BestExtractorSelector:
    def __init__(self, evaluator: ProblemEvaluator) -> None:
        self.evaluator = evaluator

    def select(
        self,
        extractors: Iterable[ProblemExtractor],
        pages: Iterable[OcrPage],
    ) -> ExtractionCandidate:
        return self.rank(extractors, pages)[0]

    def rank(
        self,
        extractors: Iterable[ProblemExtractor],
        pages: Iterable[OcrPage],
    ) -> tuple[ExtractionCandidate, ...]:
        extractor_list = list(extractors)
        if not extractor_list:
            raise ValueError("at least one extractor is required")
        buffered_pages = tuple(tuple(page) for page in pages)
        candidates = (
            self._evaluate_candidate(extractor, buffered_pages)
            for extractor in extractor_list
        )
        return tuple(
            sorted(candidates, key=lambda candidate: candidate.report.score, reverse=True)
        )

    def evaluate(
        self,
        extractor: ProblemExtractor,
        pages: Iterable[OcrPage],
    ) -> ExtractionCandidate:
        buffered_pages = tuple(tuple(page) for page in pages)
        return self._evaluate_candidate(extractor, buffered_pages)

    def _evaluate_candidate(
        self,
        extractor: ProblemExtractor,
        pages: Sequence[Sequence[OcrElement]],
    ) -> ExtractionCandidate:
        try:
            problems = tuple(extractor.extract(self._replay(pages)))
            report = self.evaluator.evaluate(self._replay(pages), problems)
        except Exception as error:
            problems = ()
            report = EvaluationReport(
                score=0.0,
                warnings=(f"extractor failed: {type(error).__name__}: {error}",),
            )
        return ExtractionCandidate(extractor, problems, report)

    @staticmethod
    def _replay(pages: Sequence[Sequence[OcrElement]]) -> Iterable[OcrPage]:
        return (list(page) for page in pages)
