from dataclasses import replace

import pytest

from exam_rec.ocr.base_ocr import OcrElement, Point
from exam_rec.transform import (
    FilterElements,
    MapElements,
    MergeCollinearElements,
    PageRegion,
    RemoveElementsInRegions,
    RemoveText,
    ReplaceText,
    TransformPipeline,
    TwoColumnLayout,
    cluster_ocr_elements_by_x,
)


def element(
    content: str,
    x: float,
    y: float = 10,
    width: float = 20,
    height: float = 10,
    label: str = "text",
) -> OcrElement:
    return OcrElement(
        bbox=[
            Point(x=x, y=y),
            Point(x=x + width, y=y),
            Point(x=x + width, y=y + height),
            Point(x=x, y=y + height),
        ],
        label=label,
        content=content,
    )


def test_pipeline_replaces_removes_and_filters_text() -> None:
    elements = [
        element("A．答案 禁止商用", 10),
        element("禁止商用", 10),
        element("keep", 10),
    ]
    pipeline = TransformPipeline(
        [
            ReplaceText({"．": ". "}),
            RemoveText(["禁止商用"]),
            FilterElements(
                lambda item: bool(item.content.strip()) and item.content != "keep"
            ),
        ]
    )

    result = pipeline.transform(elements)

    assert [item.content for item in result] == ["A. 答案 "]
    assert elements[0].content == "A．答案 禁止商用"


def test_map_elements_preserves_count_and_order() -> None:
    elements = [element("first", 10), element("second", 40)]
    transform = MapElements(lambda item: replace(item, content=item.content.upper()))

    result = transform.transform(elements)

    assert [item.content for item in result] == ["FIRST", "SECOND"]
    assert len(result) == len(elements)


def test_remove_text_preserves_elements_left_empty() -> None:
    elements = [element("禁止商用", 10), element("正文", 40)]

    result = RemoveText(["禁止商用"]).transform(elements)

    assert [item.content for item in result] == ["", "正文"]


def test_removes_elements_in_absolute_page_regions() -> None:
    elements = [
        element("header", 10, y=1),
        element("body", 10, y=40),
        element("footer", 10, y=91),
    ]
    transform = RemoveElementsInRegions(
        [PageRegion(0, 0, 200, 15), PageRegion(0, 85, 200, 100)]
    )

    assert transform.transform(elements) == [elements[1]]


def test_rejects_invalid_region_match_mode() -> None:
    with pytest.raises(ValueError, match="unsupported region match mode"):
        RemoveElementsInRegions([], match="invalid")  # type: ignore[arg-type]


def test_merges_adjacent_collinear_elements_and_bbox() -> None:
    elements = [
        element("A．一脉相承 擘画 ", 54, y=317, width=94, height=14),
        element(" B．薪火相传 描摹", 159, y=317, width=188, height=14),
        element("next line", 54, y=337, width=94, height=14),
    ]

    result = MergeCollinearElements().transform(elements)

    assert [item.content for item in result] == [
        "A．一脉相承 擘画  B．薪火相传 描摹",
        "next line",
    ]
    assert result[0].bbox == [
        Point(54, 317),
        Point(347, 317),
        Point(347, 331),
        Point(54, 331),
    ]


def test_merge_rejects_large_gap_different_label_and_missing_bbox() -> None:
    invalid = OcrElement(bbox=[], label="text", content="invalid")
    elements = [
        element("left", 0),
        element("far right", 100),
        element("figure", 125, label="figure"),
        invalid,
    ]

    assert MergeCollinearElements().transform(elements) == elements


def test_clusters_by_top_left_x_and_preserves_input_order() -> None:
    elements = [
        element("right-1", 810),
        element("left-1", 100),
        element("right-2", 850),
        element("left-2", 180),
    ]

    layout = cluster_ocr_elements_by_x(elements)

    assert [item.content for item in layout.left] == ["left-1", "left-2"]
    assert [item.content for item in layout.right] == ["right-1", "right-2"]


def test_empty_and_identical_x_inputs() -> None:
    assert cluster_ocr_elements_by_x([]) == TwoColumnLayout(left=[], right=[])

    elements = [element("a", 100), element("b", 100)]
    assert cluster_ocr_elements_by_x(elements) == TwoColumnLayout(
        left=elements, right=[]
    )


def test_rejects_element_without_bbox() -> None:
    invalid = OcrElement(bbox=[], label="text", content="invalid")

    with pytest.raises(ValueError, match="index 0 has no bbox"):
        cluster_ocr_elements_by_x([invalid])
