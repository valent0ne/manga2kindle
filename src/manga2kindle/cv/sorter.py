"""Spatial reading-order sorter for detected manga panels (RTL, LTR, and 4-koma)."""

from __future__ import annotations

from manga2kindle.models import PanelBox, ReadingDirection


class PanelSorter:
    """Orders detected panels by row-clustering (RTL/LTR) or single-column 4-koma stack."""

    def __init__(
        self,
        direction: ReadingDirection = ReadingDirection.RTL,
        row_tolerance_px: int = 50,
    ) -> None:
        self.direction = direction
        self.row_tolerance_px = row_tolerance_px

    def sort(self, boxes: list[PanelBox]) -> list[PanelBox]:
        """Sort panels into reading order and assign 1-based ordinals."""
        if not boxes:
            return []
        if len(boxes) == 1:
            return [boxes[0].model_copy(update={"ordinal": 1})]

        sorted_by_y = sorted(boxes, key=lambda b: (b.y, b.x))
        if self._is_four_koma_stack(sorted_by_y):
            ordered = sorted_by_y
        else:
            ordered = self._sort_by_rows(sorted_by_y)

        return [
            box.model_copy(update={"ordinal": idx})
            for idx, box in enumerate(ordered, start=1)
        ]

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
        row_anchor_y = sorted_by_y[0].y

        for box in sorted_by_y[1:]:
            if box.y - row_anchor_y <= self.row_tolerance_px:
                rows[-1].append(box)
            else:
                rows.append([box])
                row_anchor_y = box.y

        ordered: list[PanelBox] = []
        reverse_x = self.direction == ReadingDirection.RTL
        for row in rows:
            row_sorted = sorted(
                row,
                key=lambda b: (-b.x if reverse_x else b.x, b.y),
            )
            ordered.extend(row_sorted)
        return ordered
