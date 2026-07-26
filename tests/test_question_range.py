import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pymupdf
import pytest

from ocr.base_ocr import BaseOcr, OcrElement
from ocr.page_ocr import PageCachingOcr, PdfPageSource
from question_range import (
    LlmQuestionStartAnalyzer,
    LowConfidenceQuestionRangeError,
    QuestionRangePolicy,
    QuestionRangeResolutionError,
    QuestionRangeResolver,
    QuestionStartDecision,
    QuestionStartOutOfRangeError,
)


def make_pdf(path: Path, page_count: int) -> None:
    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page()
    document.save(path)
    document.close()


class StubOcr(BaseOcr):
    def __init__(self, pages: Sequence[list[OcrElement]]) -> None:
        self.pages = pages
        self.input_page_count = 0

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        document = pymupdf.open(stream=input, filetype="pdf")
        try:
            self.input_page_count = document.page_count
        finally:
            document.close()
        yield from self.pages[: self.input_page_count]


class StubAnalyzer:
    def __init__(self, decision: QuestionStartDecision) -> None:
        self.decision = decision
        self.page_texts: Sequence[str] | None = None

    def analyze(self, page_texts: Sequence[str]) -> QuestionStartDecision:
        self.page_texts = page_texts
        return self.decision


def element(text: str) -> OcrElement:
    return OcrElement(bbox=[], label="text", content=text)


def cached_ocr(path: Path, ocr: BaseOcr) -> PageCachingOcr:
    return PageCachingOcr(PdfPageSource(path), ocr)


@pytest.mark.parametrize("page_count, expected_scan", [(3, 3), (25, 20)])
def test_resolver_scans_up_to_configured_limit(
    tmp_path, page_count: int, expected_scan: int
) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path, page_count)
    ocr = StubOcr([[element(str(index))] for index in range(expected_scan)])
    analyzer = StubAnalyzer(QuestionStartDecision(expected_scan - 1, 0.9))

    result = QuestionRangeResolver(
        analyzer,
        page_ocr=cached_ocr(path, ocr),
    ).resolve(path)

    assert ocr.input_page_count == expected_scan
    assert result == range(expected_scan - 1, page_count)


def test_resolver_preserves_empty_pages_and_truncates_each_page(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path, 3)
    ocr = StubOcr([[element("abcdef")], [], [element("third")]])
    analyzer = StubAnalyzer(QuestionStartDecision(0, 0.9))

    QuestionRangeResolver(
        analyzer,
        page_ocr=cached_ocr(path, ocr),
        policy=QuestionRangePolicy(max_chars_per_page=3),
    ).resolve(path)

    assert analyzer.page_texts == ["abc", "", "thi"]


def test_resolver_rejects_low_confidence_and_out_of_range(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path, 2)
    page_ocr = cached_ocr(path, StubOcr([[], []]))

    with pytest.raises(LowConfidenceQuestionRangeError) as low:
        QuestionRangeResolver(
            StubAnalyzer(QuestionStartDecision(0, 0.69)),
            page_ocr=page_ocr,
        ).resolve(path)
    assert low.value.decision.confidence == 0.69

    with pytest.raises(QuestionStartOutOfRangeError):
        QuestionRangeResolver(
            StubAnalyzer(QuestionStartDecision(2, 0.9)),
            page_ocr=page_ocr,
        ).resolve(path)


def test_resolver_rejects_missing_and_password_protected_pdf(tmp_path) -> None:
    valid = tmp_path / "valid.pdf"
    make_pdf(valid, 1)
    resolver = QuestionRangeResolver(
        StubAnalyzer(QuestionStartDecision(0, 1)),
        page_ocr=cached_ocr(valid, StubOcr([[]])),
    )
    with pytest.raises(QuestionRangeResolutionError, match="does not exist"):
        resolver.resolve(tmp_path / "missing.pdf")

    document = pymupdf.open()
    document.new_page()
    protected = tmp_path / "protected.pdf"
    document.save(
        protected,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    document.close()
    with pytest.raises(QuestionRangeResolutionError, match="password protected"):
        resolver.resolve(protected)


class FakeCompletions:
    def __init__(self, contents: Sequence[str]) -> None:
        self.contents = iter(contents)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        content = next(self.contents)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=2, total_tokens=12
            ),
        )


def fake_client(contents: Sequence[str]) -> tuple[Any, FakeCompletions]:
    completions = FakeCompletions(contents)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    ), completions


@pytest.mark.parametrize(
    "invalid",
    [
        "not-json",
        json.dumps({"confidence": 0.9}),
        json.dumps({"start_page_index": "1", "confidence": 0.9}),
        json.dumps({"start_page_index": 1, "confidence": 2}),
        json.dumps({"start_page_index": 3, "confidence": 0.9}),
    ],
)
def test_llm_analyzer_retries_invalid_decisions(invalid: str) -> None:
    valid = json.dumps({"start_page_index": 1, "confidence": 0.91})
    client, completions = fake_client([invalid, valid])
    analyzer = LlmQuestionStartAnalyzer(
        "base", "key", "model", client=client, max_attempts=2
    )

    assert analyzer.analyze(["cover", "1. question", "other"]) == (
        QuestionStartDecision(1, 0.91)
    )
    assert len(completions.calls) == 2
    messages = completions.calls[0]["messages"]
    assert '<page index="0">' in messages[1]["content"]
    assert '<page index="2">' in messages[1]["content"]


def test_llm_analyzer_raises_last_validation_error() -> None:
    client, _ = fake_client(["{}", "{}"])
    analyzer = LlmQuestionStartAnalyzer(
        "base", "key", "model", client=client, max_attempts=2
    )

    with pytest.raises(QuestionRangeResolutionError):
        analyzer.analyze(["page"])


def test_llm_analyzer_uses_policy_attempt_limit() -> None:
    client, completions = fake_client(["{}"])
    analyzer = LlmQuestionStartAnalyzer(
        "base",
        "key",
        "model",
        client=client,
        policy=QuestionRangePolicy(max_attempts=1),
    )

    with pytest.raises(QuestionRangeResolutionError):
        analyzer.analyze(["page"])
    assert len(completions.calls) == 1
