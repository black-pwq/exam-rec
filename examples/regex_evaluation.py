import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from exam_rec.extractor.evaluator import StructuralProblemEvaluator
from exam_rec.extractor.regex_extractor import (
    GeneralRegexExtractor,
    HuaShengRegexExtractor,
    HuaShengYanyu700RegexExtractor,
    LlmRegexAnalyzer,
    LlmRegexExtractor,
    RegexPatterns,
)
from exam_rec.ocr.base_ocr import BaseOcr, PersistingOcr, TransformedOcr
from exam_rec.ocr.cached_ocr import CachedOcr
from exam_rec.ocr.ocr_factory import OcrRegistry
from exam_rec.ocr.paddle_ocr import PaddleOcr
from exam_rec.ocr.pymu_ocr import PyMuPDFOcr
from exam_rec.transform import MergeCollinearElements
from exam_rec.utils.pdf import select_pdf_pages


OUTPUT_DIR = Path("output") / "llm_regex_extractor" / "thinking"
OCR_CACHE_DIR = Path("output") / "llm_regex_extractor" / "ocr_cache"

base_path = Path("/mnt/d/Temp/正在使用的题库")
base_url = "https://llm-bihlcy2wklxn0jq7.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
api_key = os.getenv("ANTHROPIC_API_KEY")
model = "qwen3.7-plus"

# Page indexes are zero-based; all ranges are [start, end).
l: list[tuple[Path, tuple[int, int], tuple[int, int], type[BaseOcr]]] = [
    (base_path / "1常识题库/26最新版 常识上.pdf", (13, 18), (0, 1), PaddleOcr),
    (base_path / "3言语题库/700词涉及真题/25花生700词选词例题.pdf", (0, 5), (0, 3), PyMuPDFOcr),
    (base_path / "3言语题库/read2025片段阅读——花生600题/2025年花生片段阅读600题-题本【上】.pdf", (2, 7), (0, 1), PaddleOcr),
    (base_path / "3言语题库/read2025片段阅读——花生600题/2025年花生片段阅读600题-解析【上】.pdf", (2, 7), (0, 3), PaddleOcr),
    (base_path / "3言语题库/粉笔5000题言语理解部分/26最新版 言语理解与表达上.pdf", (13, 18), (0, 3), PaddleOcr),
    (base_path / "5数量关系/26最新版 数量关系上.pdf", (8, 13), (0, 3), PaddleOcr),
    (base_path / "6资料分析/26最新版 资料分析上.pdf", (9, 14), (0, 3), PaddleOcr),
    (base_path / "4推理题库/逻辑判断——花生600词/逻辑判断600题题本篇.pdf", (2, 7), (0, 1), PaddleOcr)
]


def output_path(path: Path, start: int, end: int, ocr_type: type[BaseOcr]) -> Path:
    ocr_name = ocr_type.__name__
    return OUTPUT_DIR / f"{path.stem}_{start}_{end}_{ocr_name}.json"


def ocr_cache_path(
    path: Path, start: int, end: int, ocr_type: type[BaseOcr]
) -> Path:
    ocr_name = ocr_type.__name__
    return OCR_CACHE_DIR / f"{path.stem}_{start}_{end}_{ocr_name}.ocr.jsonl"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def persist_ocr_results(overwrite: bool = False) -> list[dict[str, Any]]:
    """Run OCR for missing test slices and persist page-level OcrElement data."""
    configure_logging()
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    registry = OcrRegistry()

    for path, (start, end), _, ocr_type in l:
        cache = ocr_cache_path(path, start, end, ocr_type)
        record: dict[str, Any] = {
            "source": str(path),
            "pages": {"start": start, "end": end, "end_exclusive": True},
            "ocr": ocr_type.__name__,
            "cache": str(cache),
        }
        expected_page_count = end - start
        if cache.exists() and not overwrite:
            try:
                pages = CachedOcr().predict(cache)
            except ValueError:
                print(f"Rebuilding incomplete or legacy OCR cache {cache}")
            else:
                if len(pages) == expected_page_count:
                    record.update(status="cached", page_count=len(pages))
                    summary.append(record)
                    print(f"Using existing OCR cache {cache}")
                    continue
                print(
                    f"Rebuilding OCR cache with {len(pages)} pages; "
                    f"expected {expected_page_count}: {cache}"
                )

        print(f"OCR {path.name} pages [{start}, {end}) with {ocr_type.__name__}")
        try:
            pdf_bytes = select_pdf_pages(path, range(start, end))
            ocr = registry.get(ocr_type)
            pages = PersistingOcr(ocr, cache).predict(pdf_bytes)
            if len(pages) != expected_page_count:
                cache.unlink(missing_ok=True)
                raise ValueError(
                    f"OCR returned {len(pages)} pages; expected {expected_page_count}"
                )
            record.update(status="success", page_count=len(pages))
            print(f"Saved {len(pages)} OCR pages to {cache}")
        except Exception as error:
            record.update(
                status="error",
                error_type=type(error).__name__,
                error=str(error),
            )
            print(f"OCR failed: {type(error).__name__}: {error}")
        summary.append(record)

    (OCR_CACHE_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def analyze_persisted_ocr_results() -> list[dict[str, Any]]:
    """Run LLM regex analysis and extraction using only persisted OCR pages."""
    configure_logging()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    analyzer = LlmRegexAnalyzer(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []

    for path, (start, end), sample_range, ocr_type in l:
        destination = output_path(path, start, end, ocr_type)
        cache = ocr_cache_path(path, start, end, ocr_type)
        record: dict[str, Any] = {
            "source": str(path),
            "pages": {"start": start, "end": end, "end_exclusive": True},
            "sample_range": {
                "start": sample_range[0],
                "end": sample_range[1],
                "end_exclusive": True,
            },
            "ocr": ocr_type.__name__,
            "ocr_cache": str(cache),
            "output": str(destination),
        }
        print(f"Analyzing cached OCR for {path.name} pages [{start}, {end})")

        extractor: LlmRegexExtractor | None = None
        try:
            extractor = LlmRegexExtractor(
                range(*sample_range),
                analyzer,
            )
            ocr = TransformedOcr(
                source=CachedOcr(), transform=MergeCollinearElements()
            )
            problems = extractor.extract(ocr.predict_iter(cache))
            destination.write_text(
                json.dumps(
                    [asdict(problem) for problem in problems],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            record.update(
                status="success",
                problem_count=len(problems),
                patterns=asdict(extractor.patterns) if extractor.patterns else None,
            )
            print(f"Saved {len(problems)} problems to {destination}")
        except Exception as error:
            record.update(
                status="error",
                error_type=type(error).__name__,
                error=str(error),
            )
            if extractor is not None and extractor.patterns is not None:
                record["patterns"] = asdict(extractor.patterns)
            print(f"Failed: {type(error).__name__}: {error}")

        summary.append(record)
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def evaluate_persisted_ocr_results() -> list[dict[str, Any]]:
    """Evaluate regex extractors using persisted OCR and saved LLM patterns."""
    summary_paths = [
        OUTPUT_DIR / "summary.json",
        OUTPUT_DIR.parent / "summary.json",
    ]
    analysis_records: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        if not summary_path.is_file():
            continue
        records = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(
                f"analysis summary must contain a JSON array: {summary_path}"
            )
        analysis_records.extend(
            record for record in records if isinstance(record, dict)
        )
    if not analysis_records:
        raise FileNotFoundError(
            "no analysis summary exists: "
            + ", ".join(str(path) for path in summary_paths)
        )
    records_by_cache = {
        Path(record["ocr_cache"]).resolve(): record
        for record in analysis_records
        if isinstance(record, dict) and isinstance(record.get("ocr_cache"), str)
    }

    evaluator = StructuralProblemEvaluator()
    results: list[dict[str, Any]] = []

    for path, (start, end), _, ocr_type in l:
        cache = ocr_cache_path(path, start, end, ocr_type)
        print(f"Evaluating {path.name} pages [{start}, {end})")
        try:
            pages = TransformedOcr(
                source=CachedOcr(), transform=MergeCollinearElements()
            ).predict(cache)
        except Exception as error:
            print(f"  OCR cache failed: {type(error).__name__}: {error}")
            results.append(
                {
                    "source": str(path),
                    "ocr_cache": str(cache),
                    "status": "cache_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue

        extractors = [
            ("HuaShengRegexExtractor", HuaShengRegexExtractor()),
            ("HuaShengYanyu700RegexExtractor", HuaShengYanyu700RegexExtractor()),
        ]
        analysis_record = records_by_cache.get(cache.resolve())
        if analysis_record is None:
            analysis_record = next(
                (
                    record
                    for record in analysis_records
                    if record.get("source") == str(path)
                    and isinstance(record.get("pages"), dict)
                    and record.get("pages", {}).get("start") == start
                    and record.get("pages", {}).get("end") == end
                    and isinstance(record.get("patterns"), dict)
                ),
                None,
            )
        patterns_data = analysis_record.get("patterns") if analysis_record else None
        if isinstance(patterns_data, dict):
            try:
                patterns = RegexPatterns.from_dict(patterns_data)
                extractors.append(
                    ("GeneralRegexExtractor", GeneralRegexExtractor(patterns))
                )
            except Exception as error:
                print(
                    "  GeneralRegexExtractor: skipped: invalid saved patterns: "
                    f"{type(error).__name__}: {error}"
                )
        else:
            print("  GeneralRegexExtractor: skipped: saved patterns not found")

        for extractor_name, extractor in extractors:
            record: dict[str, Any] = {
                "source": str(path),
                "ocr_cache": str(cache),
                "extractor": extractor_name,
            }
            try:
                problems = extractor.extract(iter(pages))
                report = evaluator.evaluate(iter(pages), problems)
            except Exception as error:
                record.update(
                    status="error",
                    error_type=type(error).__name__,
                    error=str(error),
                )
                print(
                    f"  {extractor_name}: failed: "
                    f"{type(error).__name__}: {error}"
                )
            else:
                record.update(
                    status="success",
                    problem_count=len(problems),
                    score=report.score,
                    metrics=dict(report.metrics),
                    warnings=list(report.warnings),
                )
                print(
                    f"  {extractor_name}: {report.score:.6f} "
                    f"({len(problems)} problems)"
                )
                for metric, score in report.metrics.items():
                    print(f"    {metric}: {score:.6f}")
                for warning in report.warnings:
                    print(f"    warning: {warning}")
            results.append(record)

    return results


def run(overwrite_ocr: bool = False) -> list[dict[str, Any]]:
    from pprint import pprint
    persist_ocr_results(overwrite=overwrite_ocr)
    pprint(evaluate_persisted_ocr_results())
    # return analyze_persisted_ocr_results()


if __name__ == "__main__":
    run()
