from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exam_rec.transform import PageTransform


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class OcrElement:
    bbox: list[Point]
    label: str
    content: str


class BaseOcr(ABC):
    def predict(self, input: Any) -> list[list[OcrElement]]:
        return list(self.predict_iter(input))

    @abstractmethod
    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        raise NotImplementedError


PERSISTENCE_VERSION = 1


class PersistingOcr(BaseOcr):
    """Persist results from an OCR source without mutating the source."""

    def __init__(
        self,
        source: BaseOcr,
        persist_to: str | PathLike[str],
    ) -> None:
        self.source = source
        self.persist_to = Path(persist_to)

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        destination = self.persist_to
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(f"OCR persistence path is a directory: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("w", encoding="utf-8") as file:
            self._write_record(
                file,
                {"type": "header", "version": PERSISTENCE_VERSION},
            )
            for page in self.source.predict_iter(input):
                self._write_record(
                    file,
                    {
                        "type": "page",
                        "elements": [asdict(element) for element in page],
                    },
                )
                yield page
            self._write_record(file, {"type": "complete"})

    @staticmethod
    def _write_record(file: Any, record: dict[str, Any]) -> None:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


class TransformedOcr(BaseOcr):
    """Apply a page transform lazily to another OCR implementation."""

    def __init__(
        self,
        source: BaseOcr,
        transform: PageTransform,
    ) -> None:
        self.source = source
        self.transform = transform

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        for elements in self.source.predict_iter(input):
            yield self.transform.transform(elements)
