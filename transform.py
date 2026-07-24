"""Composable transformations for page-level OCR results."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol, TypeAlias, TypeVar

from ocr.base_ocr import OcrElement, Point

OcrPage: TypeAlias = list[OcrElement]


class PageTransform(Protocol):
    def transform(self, elements: OcrPage) -> OcrPage: ...


PartitionT = TypeVar("PartitionT", covariant=True)


class PagePartition(Protocol[PartitionT]):
    def partition(self, elements: OcrPage) -> PartitionT: ...


class TransformPipeline:
    def __init__(self, transforms: Sequence[PageTransform]) -> None:
        self.transforms = tuple(transforms)

    def transform(self, elements: OcrPage) -> OcrPage:
        result = elements
        for transform in self.transforms:
            result = transform.transform(result)
        return result


class MapElements:
    """Map every element while preserving page length and element order."""

    def __init__(
        self,
        mapper: Callable[[OcrElement], OcrElement],
    ) -> None:
        self.mapper = mapper

    def transform(self, elements: OcrPage) -> OcrPage:
        return [self.mapper(element) for element in elements]


class ReplaceText(MapElements):
    """Apply literal replacements to every element's content in insertion order."""

    def __init__(self, replacements: Mapping[str, str]) -> None:
        self.replacements = tuple(replacements.items())
        super().__init__(self._replace_text)

    def _replace_text(self, element: OcrElement) -> OcrElement:
        content = element.content
        for old, new in self.replacements:
            content = content.replace(old, new)
        return replace(element, content=content)


class RemoveText(MapElements):
    """Remove literal strings while preserving every OCR element."""

    def __init__(self, strings: Sequence[str]) -> None:
        self.strings = tuple(strings)
        super().__init__(self._remove_text)

    def _remove_text(self, element: OcrElement) -> OcrElement:
        content = element.content
        for value in self.strings:
            content = content.replace(value, "")
        return replace(element, content=content)


class FilterElements:
    """Keep elements for which the predicate returns true."""

    def __init__(self, predicate: Callable[[OcrElement], bool]) -> None:
        self.predicate = predicate

    def transform(self, elements: OcrPage) -> OcrPage:
        return [element for element in elements if self.predicate(element)]


@dataclass(frozen=True)
class PageRegion:
    x0: float
    y0: float
    x1: float
    y1: float


class RemoveElementsInRegions:
    """Remove positioned elements matching one of the supplied page regions."""

    def __init__(
        self,
        regions: Sequence[PageRegion],
        *,
        match: Literal["center", "contained", "overlap"] = "center",
        min_overlap: float = 0.5,
        remove_without_bbox: bool = False,
    ) -> None:
        if not 0 <= min_overlap <= 1:
            raise ValueError("min_overlap must be between 0 and 1")
        if match not in ("center", "contained", "overlap"):
            raise ValueError(f"unsupported region match mode: {match}")
        self.regions = tuple(regions)
        self.match = match
        self.min_overlap = min_overlap
        self.remove_without_bbox = remove_without_bbox

    def transform(self, elements: OcrPage) -> OcrPage:
        regions = [
            (region.x0, region.y0, region.x1, region.y1)
            for region in self.regions
        ]
        return [
            element
            for element in elements
            if not self._should_remove(element, regions)
        ]

    def _should_remove(
        self,
        element: OcrElement,
        regions: Sequence[tuple[float, float, float, float]],
    ) -> bool:
        bounds = _element_bounds(element)
        if bounds is None:
            return self.remove_without_bbox
        x0, y0, x1, y1 = bounds
        for rx0, ry0, rx1, ry1 in regions:
            if self.match == "center":
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                if rx0 <= center_x <= rx1 and ry0 <= center_y <= ry1:
                    return True
            elif self.match == "contained":
                if rx0 <= x0 and ry0 <= y0 and x1 <= rx1 and y1 <= ry1:
                    return True
            elif _overlap_ratio(bounds, (rx0, ry0, rx1, ry1)) >= self.min_overlap:
                return True
        return False


class MergeCollinearElements:
    """Merge adjacent elements that occupy the same visual text line."""

    def __init__(
        self,
        *,
        min_vertical_overlap: float = 0.8,
        max_horizontal_gap_ratio: float = 2.0,
        separator: str = "",
    ) -> None:
        if not 0 <= min_vertical_overlap <= 1:
            raise ValueError("min_vertical_overlap must be between 0 and 1")
        if max_horizontal_gap_ratio < 0:
            raise ValueError("max_horizontal_gap_ratio must not be negative")
        self.min_vertical_overlap = min_vertical_overlap
        self.max_horizontal_gap_ratio = max_horizontal_gap_ratio
        self.separator = separator

    def transform(self, elements: OcrPage) -> OcrPage:
        result: OcrPage = []
        for element in elements:
            if result and self._can_merge(result[-1], element):
                result[-1] = self._merge(result[-1], element)
            else:
                result.append(element)
        return result

    def _can_merge(self, left: OcrElement, right: OcrElement) -> bool:
        if left.label != right.label:
            return False
        left_bounds = _element_bounds(left)
        right_bounds = _element_bounds(right)
        if left_bounds is None or right_bounds is None:
            return False
        lx0, ly0, lx1, ly1 = left_bounds
        rx0, ry0, rx1, ry1 = right_bounds
        left_height = ly1 - ly0
        right_height = ry1 - ry0
        if left_height <= 0 or right_height <= 0 or rx0 < lx1:
            return False
        vertical_overlap = max(0.0, min(ly1, ry1) - max(ly0, ry0))
        overlap_ratio = vertical_overlap / min(left_height, right_height)
        gap = rx0 - lx1
        return (
            overlap_ratio >= self.min_vertical_overlap
            and gap <= max(left_height, right_height) * self.max_horizontal_gap_ratio
        )

    def _merge(self, left: OcrElement, right: OcrElement) -> OcrElement:
        left_bounds = _element_bounds(left)
        right_bounds = _element_bounds(right)
        assert left_bounds is not None and right_bounds is not None
        x0 = min(left_bounds[0], right_bounds[0])
        y0 = min(left_bounds[1], right_bounds[1])
        x1 = max(left_bounds[2], right_bounds[2])
        y1 = max(left_bounds[3], right_bounds[3])
        return OcrElement(
            bbox=_rectangle_points(x0, y0, x1, y1),
            label=left.label,
            content=left.content + self.separator + right.content,
        )


@dataclass(frozen=True)
class TwoColumnLayout:
    left: OcrPage
    right: OcrPage


class TwoColumnPartition:
    """Partition elements by the x coordinate of their top-left bbox point."""

    def partition(self, elements: OcrPage) -> TwoColumnLayout:
        if not elements:
            return TwoColumnLayout(left=[], right=[])

        xs: list[float] = []
        for index, element in enumerate(elements):
            if not element.bbox:
                raise ValueError(f"OCR element at index {index} has no bbox")
            xs.append(element.bbox[0].x)

        left_center = min(xs)
        right_center = max(xs)
        if left_center == right_center:
            return TwoColumnLayout(left=list(elements), right=[])

        assignments = [0] * len(elements)
        for _ in range(100):
            new_assignments = [
                0 if abs(x - left_center) <= abs(x - right_center) else 1
                for x in xs
            ]
            if new_assignments == assignments:
                break
            assignments = new_assignments

            left_xs = [x for x, cluster in zip(xs, assignments) if cluster == 0]
            right_xs = [x for x, cluster in zip(xs, assignments) if cluster == 1]
            left_center = sum(left_xs) / len(left_xs)
            right_center = sum(right_xs) / len(right_xs)

        left = [
            element for element, cluster in zip(elements, assignments) if cluster == 0
        ]
        right = [
            element for element, cluster in zip(elements, assignments) if cluster == 1
        ]
        if left_center <= right_center:
            return TwoColumnLayout(left=left, right=right)
        return TwoColumnLayout(left=right, right=left)


def cluster_ocr_elements_by_x(elements: OcrPage) -> TwoColumnLayout:
    """Partition a page into explicitly named left and right columns."""
    return TwoColumnPartition().partition(elements)


def _element_bounds(
    element: OcrElement,
) -> tuple[float, float, float, float] | None:
    if not element.bbox:
        return None
    xs = [point.x for point in element.bbox]
    ys = [point.y for point in element.bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _rectangle_points(x0: float, y0: float, x1: float, y1: float) -> list[Point]:
    return [Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)]


def _overlap_ratio(
    bounds: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    x0, y0, x1, y1 = bounds
    rx0, ry0, rx1, ry1 = region
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    area = width * height
    if area == 0:
        return 0.0
    overlap_width = max(0.0, min(x1, rx1) - max(x0, rx0))
    overlap_height = max(0.0, min(y1, ry1) - max(y0, ry0))
    return overlap_width * overlap_height / area
