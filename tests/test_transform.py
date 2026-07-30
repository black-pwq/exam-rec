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
    SplitMultilineElements,
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


def test_split_multiline_elements_creates_line_level_boxes() -> None:
    source = element("question\nA. first\nB. second", 10, y=100, width=490, height=90)

    result = SplitMultilineElements().transform([source])

    assert [item.content for item in result] == [
        "question",
        "A. first",
        "B. second",
    ]
    assert [item.bbox for item in result] == [
        [Point(10, 100), Point(500, 100), Point(500, 130), Point(10, 130)],
        [Point(10, 130), Point(500, 130), Point(500, 160), Point(10, 160)],
        [Point(10, 160), Point(500, 160), Point(500, 190), Point(10, 190)],
    ]


def test_split_multiline_elements_keeps_empty_line_position() -> None:
    source = element("first\n\nthird", 0, y=0, width=40, height=30)

    result = SplitMultilineElements().transform([source])

    assert [item.content for item in result] == ["first", "third"]
    assert [item.bbox for item in result] == [
        [Point(0, 0), Point(40, 0), Point(40, 10), Point(0, 10)],
        [Point(0, 20), Point(40, 20), Point(40, 30), Point(0, 30)],
    ]


def test_split_multiline_elements_can_keep_empty_lines() -> None:
    source = OcrElement(
        bbox=[],
        label="text",
        content="first\r\n\r\nthird",
    )

    result = SplitMultilineElements(keep_empty=True).transform([source])

    assert [item.content for item in result] == ["first", "", "third"]
    assert all(item.bbox == [] for item in result)


def test_split_multiline_elements_only_processes_configured_labels() -> None:
    table = OcrElement(bbox=[], label="table", content="<tr>\n<td>value</td>")
    formula = OcrElement(bbox=[], label="formula", content="a\nb")

    result = SplitMultilineElements(labels=("formula",)).transform([table, formula])

    assert result == [
        table,
        OcrElement(bbox=[], label="formula", content="a"),
        OcrElement(bbox=[], label="formula", content="b"),
    ]


def test_split_multiline_elements_preserves_single_line_identity() -> None:
    source = element("single line", 10)

    result = SplitMultilineElements().transform([source])

    assert result == [source]
    assert result[0] is source


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
