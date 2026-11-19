"""Spatial reading-order sorter for detected manga panels (RTL, LTR, and 4-koma)."""

from __future__ import annotations

from manga2kindle.cv.boxes import CutAxis, axis_interval, split_by_gutter
from manga2kindle.models import PanelBox, ReadingDirection


class PanelSorter:
    """Orders detected panels by recursive XY-cut gutter partitioning with row-clustering fallback."""

    def __init__(
        self,
        direction: ReadingDirection = ReadingDirection.RTL,
        row_tolerance_px: int = 50,
    ) -> None:
        self.direction = direction
        self.row_tolerance_px = row_tolerance_px

    def sort(
        self,
        boxes: list[PanelBox],
        *,
        page_width: int | None = None,
        is_spread: bool = False,
    ) -> list[PanelBox]:
        """Sort panels into reading order and assign 1-based ordinals."""
        if not boxes:
            return []
        if len(boxes) == 1:
            return [boxes[0].model_copy(update={"ordinal": 1})]

        if is_spread and page_width is not None and page_width > 0:
            ordered = self._sort_spread_halves(boxes, page_width)
        else:
            ordered = self._sort_group(boxes)

        return [
            box.model_copy(update={"ordinal": idx})
            for idx, box in enumerate(ordered, start=1)
        ]

    def _sort_group(self, boxes: list[PanelBox]) -> list[PanelBox]:
        if len(boxes) <= 1:
            return list(boxes)
        sorted_by_y = sorted(boxes, key=lambda b: (b.y, b.x))
        if self._is_four_koma_stack(sorted_by_y):
            return sorted_by_y
        return self._xy_cut_sort(boxes)

    def _xy_cut_sort(self, boxes: list[PanelBox]) -> list[PanelBox]:
        """Recursively partition panels by strict then inset XY gutters."""
        if len(boxes) <= 1:
            return list(boxes)

        for inset in (False, True):
            for axis in (CutAxis.HORIZONTAL, CutAxis.VERTICAL):
                groups = self._split_by_axis(boxes, axis, inset=inset)
                if len(groups) >= 2:
                    ordered: list[PanelBox] = []
                    for group in groups:
                        ordered.extend(self._xy_cut_sort(group))
                    return ordered

        sorted_by_y = sorted(boxes, key=lambda b: (b.y, b.x))
        return self._sort_by_rows(sorted_by_y)

    def _split_by_axis(
        self,
        boxes: list[PanelBox],
        axis: CutAxis,
        *,
        inset: bool,
    ) -> list[list[PanelBox]]:
        """Sweep 1D panel intervals to split into non-overlapping bands or columns."""
        return split_by_gutter(
            boxes,
            axis,
            inset=inset,
            reverse=(axis is CutAxis.VERTICAL and self.direction == ReadingDirection.RTL),
        )

    @staticmethod
    def _axis_interval(box: PanelBox, axis: CutAxis, *, inset: bool) -> tuple[int, int]:
        """Return [start, end) along the cut axis, optionally inset to clear minor padding overlap."""
        return axis_interval(box, axis, inset=inset)

    def _sort_spread_halves(self, boxes: list[PanelBox], page_width: int) -> list[PanelBox]:
        """Partition an unsplit two-page spread across the center gutter before sorting."""
        midpoint = page_width / 2.0
        left_half = [box for box in boxes if (box.x + box.w / 2.0) < midpoint]
        right_half = [box for box in boxes if (box.x + box.w / 2.0) >= midpoint]
        halves = (
            (right_half, left_half)
            if self.direction == ReadingDirection.RTL
            else (left_half, right_half)
        )
        ordered: list[PanelBox] = []
        for half in halves:
            ordered.extend(self._sort_group(half))
        return ordered

    @staticmethod
    def _is_four_koma_stack(sorted_by_y: list[PanelBox]) -> bool:
        """Return True when panels form a single-column vertical 4-koma progression."""
        centers_x = [b.x + b.w / 2.0 for b in sorted_by_y]
        min_w = min(b.w for b in sorted_by_y)
        if max(centers_x) - min(centers_x) > min_w * 0.5:
            return False

        return all(
            curr.y >= prev.y + prev.h * 0.5
            for prev, curr in zip(sorted_by_y, sorted_by_y[1:])
        )

    def _sort_by_rows(self, sorted_by_y: list[PanelBox]) -> list[PanelBox]:
        """Cluster panels into horizontal rows and order each row by reading direction."""
        rows: list[list[PanelBox]] = [[sorted_by_y[0]]]
        row_anchor = sorted_by_y[0]

        for box in sorted_by_y[1:]:
            overlap_h = min(box.y + box.h, row_anchor.y + row_anchor.h) - max(
                box.y, row_anchor.y
            )
            min_h = min(box.h, row_anchor.h)
            same_row = (
                box.y - row_anchor.y <= self.row_tolerance_px
                or (min_h > 0 and overlap_h >= 0.5 * min_h)
            )
            if same_row:
                rows[-1].append(box)
            else:
                rows.append([box])
                row_anchor = box

        ordered: list[PanelBox] = []
        reverse_x = self.direction == ReadingDirection.RTL
        for row in rows:
            row_sorted = sorted(
                row,
                key=lambda b: (-b.x if reverse_x else b.x, b.y),
            )
            ordered.extend(row_sorted)
        return ordered
