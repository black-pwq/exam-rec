import json
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import Any

from app_logging import get_logger
from ocr.base_ocr import PERSISTENCE_VERSION, BaseOcr, OcrElement, Point


logger = get_logger(__name__)


class CachedOcr(BaseOcr):
    """Read page-level OCR results from a BaseOcr persistence file."""

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        if not isinstance(input, (str, PathLike)):
            raise TypeError("OcrCached input must be a persistence file path")
        persist_from = Path(input)
        pages = self._load_persisted(persist_from)
        if pages is None:
            raise ValueError(
                "OCR persistence file is incomplete or invalid: "
                f"{persist_from}"
            )
        logger.info("Reading persisted OCR results from %s", persist_from)
        yield from pages

    @classmethod
    def _load_persisted(cls, path: Path) -> list[list[OcrElement]] | None:
        if not path.is_file():
            return None

        try:
            records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
            if (
                len(records) < 2
                or records[0]
                != {"type": "header", "version": PERSISTENCE_VERSION}
                or records[-1] != {"type": "complete"}
            ):
                return None
            page_records = records[1:-1]
            if not all(
                isinstance(record, dict) and record.get("type") == "page"
                for record in page_records
            ):
                return None
            return [cls._deserialize_page(record) for record in page_records]
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring invalid OCR persistence file %s", path)
            return None

    @staticmethod
    def _deserialize_page(record: dict[str, Any]) -> list[OcrElement]:
        if record.get("type") != "page" or not isinstance(record.get("elements"), list):
            raise ValueError("invalid OCR page record")
        return [
            OcrElement(
                bbox=[Point(x=point["x"], y=point["y"]) for point in item["bbox"]],
                label=item["label"],
                content=item["content"],
            )
            for item in record["elements"]
        ]
