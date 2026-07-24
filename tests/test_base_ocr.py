from collections.abc import Iterator
import json
from typing import Any

import pytest

from ocr.base_ocr import (
    BaseOcr,
    OcrElement,
    PersistingOcr,
    Point,
    TransformedOcr,
)
from ocr.cached_ocr import CachedOcr
from transform import RemoveElementsInRegions, PageRegion, ReplaceText


class StubOcr(BaseOcr):
    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        yield [OcrElement(bbox=[], label="text", content=str(input))]


def test_predict_collects_iterator_results() -> None:
    assert StubOcr().predict("hello") == [
        [OcrElement(bbox=[], label="text", content="hello")]
    ]


class CountingOcr(BaseOcr):
    def __init__(self) -> None:
        self.calls = 0

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        self.calls += 1
        yield [OcrElement(bbox=[Point(1, 2)], label="text", content="first")]
        yield [OcrElement(bbox=[], label="text", content="second")]


def test_predict_persists_incrementally_and_always_runs_backend(tmp_path) -> None:
    persistence = tmp_path / "ocr.jsonl"
    source = CountingOcr()
    ocr = PersistingOcr(source, persistence)
    iterator = ocr.predict_iter("input.pdf")

    first_page = next(iterator)

    records = [json.loads(line) for line in persistence.read_text().splitlines()]
    assert first_page[0].content == "first"
    assert [record["type"] for record in records] == ["header", "page"]

    assert list(iterator)[0][0].content == "second"
    assert json.loads(persistence.read_text().splitlines()[-1]) == {
        "type": "complete"
    }

    assert ocr.predict("different input") == [
        [OcrElement(bbox=[Point(1, 2)], label="text", content="first")],
        [OcrElement(bbox=[], label="text", content="second")],
    ]
    assert source.calls == 2


def test_cached_ocr_reads_complete_persistence_without_backend(tmp_path) -> None:
    persistence = tmp_path / "ocr.jsonl"
    source = CountingOcr()
    expected = PersistingOcr(source, persistence).predict("input.pdf")

    assert CachedOcr().predict(persistence) == expected


def test_cached_ocr_rejects_incomplete_persistence(tmp_path) -> None:
    persistence = tmp_path / "ocr.jsonl"
    persistence.write_text(
        '{"type":"header","version":1}\n'
        '{"type":"page","elements":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete or invalid"):
        CachedOcr().predict(persistence)


def test_cached_ocr_rejects_non_path_input() -> None:
    with pytest.raises(TypeError, match="persistence file path"):
        CachedOcr().predict(b"not a path")


def test_incomplete_persistence_is_recomputed(tmp_path) -> None:
    persistence = tmp_path / "ocr.jsonl"
    persistence.write_text(
        '{"type":"header","version":1}\n'
        '{"type":"page","elements":[]}\n',
        encoding="utf-8",
    )
    source = CountingOcr()
    ocr = PersistingOcr(source, persistence)

    assert len(ocr.predict("input.pdf")) == 2
    assert source.calls == 1
    assert json.loads(persistence.read_text().splitlines()[-1]) == {
        "type": "complete"
    }


def test_persisting_ocr_does_not_mutate_shared_source(tmp_path) -> None:
    source = StubOcr()
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    first = PersistingOcr(source, first_path).predict("first input")
    second = PersistingOcr(source, second_path).predict("second input")

    assert CachedOcr().predict(first_path) == first
    assert CachedOcr().predict(second_path) == second
    assert first[0][0].content == "first input"
    assert second[0][0].content == "second input"


class TwoPageOcr(BaseOcr):
    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        yield [
            OcrElement(
                bbox=[Point(0, 0), Point(20, 0), Point(20, 10), Point(0, 10)],
                label="text",
                content="A．first",
            )
        ]
        yield [
            OcrElement(
                bbox=[Point(0, 50), Point(20, 50), Point(20, 60), Point(0, 60)],
                label="text",
                content="B．second",
            )
        ]


def test_transformed_ocr_applies_transform_lazily() -> None:
    ocr = TransformedOcr(TwoPageOcr(), ReplaceText({"．": ". "}))
    iterator = ocr.predict_iter("input.pdf")

    assert next(iterator)[0].content == "A. first"
    assert next(iterator)[0].content == "B. second"


def test_transformed_ocr_applies_absolute_region_filter() -> None:
    ocr = TransformedOcr(
        TwoPageOcr(),
        RemoveElementsInRegions([PageRegion(0, 0, 100, 20)]),
    )

    assert ocr.predict("input.pdf") == [[], TwoPageOcr().predict("input.pdf")[1]]


def test_transformed_ocr_persists_transformed_pages(tmp_path) -> None:
    persistence = tmp_path / "transformed.jsonl"
    ocr = PersistingOcr(
        TransformedOcr(TwoPageOcr(), ReplaceText({"．": ". "})), persistence
    )

    expected = ocr.predict("input.pdf")

    assert CachedOcr().predict(persistence) == expected
    assert expected[0][0].content == "A. first"
