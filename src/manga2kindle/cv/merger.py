"""Profile-calibrated post-sort mini-panel merging for Kindle Guided View."""

from __future__ import annotations

from collections.abc import Sequence

from manga2kindle.cv.boxes import (
    GUTTER_INSET_MAX_PX,
    GUTTER_INSET_RATIO,
    CutAxis,
    Rect,
    compute_containment,
    panel_to_rect,
    split_by_gutter,
    union_rect,
    unpad_rect,
)
from manga2kindle.models import PanelBox, PanelMergeConfig

_SPREAD_STRADDLE_RATIO: float = 0.25
_PADDING_INSET_BUFFER: float = 0.015


def _renumber_ordinals(panels: Sequence[PanelBox]) -> list[PanelBox]:
    """Return copied PanelBoxes with sequential 1-based ordinals."""
    return [panel.model_copy(update={"ordinal": idx}) for idx, panel in enumerate(panels, start=1)]


def _rect_union(rects: Sequence[Rect]) -> Rect:
    """Return the enclosing (x, y, w, h) rectangle for non-empty Rects."""
    x1 = min(rect[0] for rect in rects)
    y1 = min(rect[1] for rect in rects)
    x2 = max(rect[0] + rect[2] for rect in rects)
    y2 = max(rect[1] + rect[3] for rect in rects)
    return (x1, y1, x2 - x1, y2 - y1)


class PanelMerger:
    """Merges compact mini-panel horizontal tiers and stacked sub-columns after sorting."""

    def __init__(self, config: PanelMergeConfig | None = None) -> None:
        self.config = config or PanelMergeConfig()

    def merge(
        self,
        panels: Sequence[PanelBox],
        *,
        page_width: int,
        page_height: int,
        is_spread: bool = False,
        padding: float = 0.0,
    ) -> list[PanelBox]:
        """Merge eligible mini-panel groups while preserving reading order and bounds."""
        if not panels:
            return []
        if not self.config.enabled or len(panels) < 2 or page_width <= 0 or page_height <= 0:
            return _renumber_ordinals(panels)

        # Copy panels so duplicate PanelBox instances in the input keep unique object IDs.
        items = [panel.model_copy() for panel in panels]
        index_by_id = {id(panel): idx for idx, panel in enumerate(items)}
        raw_rects = [unpad_rect(panel_to_rect(panel), padding) for panel in items]

        if padding > 0.0:
            inset_ratio = max(
                GUTTER_INSET_RATIO,
                (padding / (1.0 + 2.0 * padding)) + _PADDING_INSET_BUFFER,
            )
            inset_max_px: int | None = None
        else:
            inset_ratio = GUTTER_INSET_RATIO
            inset_max_px = GUTTER_INSET_MAX_PX

        merges: list[tuple[int, int, PanelBox]] = []
        for half_panels, eff_w in self._partition_halves(items, page_width, is_spread=is_spread):
            bands = split_by_gutter(
                half_panels,
                CutAxis.HORIZONTAL,
                inset=True,
                inset_ratio=inset_ratio,
                inset_max_px=inset_max_px,
            )
            for band in bands:
                tier_merge = self._try_merge_group(
                    band,
                    eff_w=eff_w,
                    page_height=page_height,
                    index_by_id=index_by_id,
                    raw_rects=raw_rects,
                    min_panels=self.config.min_tier_panels,
                    max_group_w=None,
                    max_group_h=self.config.max_tier_height_ratio,
                    max_panel_w=self.config.max_tier_panel_width_ratio,
                    max_panel_area=self.config.max_tier_panel_area_ratio,
                )
                if tier_merge is not None:
                    merges.append(tier_merge)
                    continue

                cols = split_by_gutter(
                    band,
                    CutAxis.VERTICAL,
                    inset=True,
                    inset_ratio=inset_ratio,
                    inset_max_px=inset_max_px,
                )
                if len(cols) < 2:
                    continue
                for col in cols:
                    rows_in_col = split_by_gutter(
                        col,
                        CutAxis.HORIZONTAL,
                        inset=True,
                        inset_ratio=inset_ratio,
                        inset_max_px=inset_max_px,
                    )
                    if len(rows_in_col) < 2:
                        continue
                    subcol_merge = self._try_merge_group(
                        col,
                        eff_w=eff_w,
                        page_height=page_height,
                        index_by_id=index_by_id,
                        raw_rects=raw_rects,
                        min_panels=self.config.min_subcol_panels,
                        max_group_w=self.config.max_subcol_width_ratio,
                        max_group_h=self.config.max_subcol_height_ratio,
                        max_panel_w=None,
                        max_panel_area=self.config.max_subcol_panel_area_ratio,
                    )
                    if subcol_merge is not None:
                        merges.append(subcol_merge)

        if not merges:
            return _renumber_ordinals(items)

        merge_by_start = {lo: (hi, box) for lo, hi, box in merges}
        merged: list[PanelBox] = []
        idx = 0
        while idx < len(items):
            replacement = merge_by_start.get(idx)
            if replacement is not None:
                hi, box = replacement
                merged.append(box)
                idx = hi + 1
            else:
                merged.append(items[idx])
                idx += 1
        return _renumber_ordinals(merged)

    @staticmethod
    def _partition_halves(
        items: list[PanelBox],
        page_width: int,
        *,
        is_spread: bool,
    ) -> list[tuple[list[PanelBox], float]]:
        """Return candidate (panels, effective_width) groups per page or unsplit spread half."""
        if not is_spread:
            return [(items, float(page_width))]

        midpoint = page_width / 2.0
        eff_w = page_width / 2.0
        right_half: list[PanelBox] = []
        left_half: list[PanelBox] = []

        for box in items:
            if box.x < midpoint < (box.x + box.w):
                straddle_px = min(midpoint - box.x, (box.x + box.w) - midpoint)
                if (straddle_px / box.w) > _SPREAD_STRADDLE_RATIO:
                    continue
            if (box.x + box.w / 2.0) >= midpoint:
                right_half.append(box)
            else:
                left_half.append(box)

        return [(half, eff_w) for half in (right_half, left_half) if len(half) >= 2]

    def _try_merge_group(
        self,
        group: Sequence[PanelBox],
        *,
        eff_w: float,
        page_height: int,
        index_by_id: dict[int, int],
        raw_rects: Sequence[Rect],
        min_panels: int,
        max_group_w: float | None,
        max_group_h: float,
        max_panel_w: float | None,
        max_panel_area: float,
    ) -> tuple[int, int, PanelBox] | None:
        """Validate geometry, reading-order contiguity, and overlap guards for one candidate group."""
        if len(group) < min_panels:
            return None

        indices = sorted(index_by_id[id(panel)] for panel in group)
        lo, hi = indices[0], indices[-1]
        if (hi - lo + 1) != len(indices):
            return None

        raw_union = _rect_union([raw_rects[i] for i in indices])
        _, _, ruw, ruh = raw_union
        if max_group_w is not None and (ruw / eff_w) > max_group_w:
            return None
        if (ruh / page_height) > max_group_h:
            return None

        eff_page_area = eff_w * page_height
        if max_panel_w is not None and any((raw_rects[i][2] / eff_w) > max_panel_w for i in indices):
            return None
        if any(
            ((raw_rects[i][2] * raw_rects[i][3]) / eff_page_area) > max_panel_area
            for i in indices
        ):
            return None

        member_indices = set(indices)
        max_overlap = self.config.max_external_overlap_ratio
        for idx, rect in enumerate(raw_rects):
            if idx in member_indices:
                continue
            if compute_containment(raw_union, rect) > max_overlap:
                return None

        # Emit the exact union of already-padded member boxes to preserve bubble clearance.
        ux, uy, uw, uh = union_rect(group)
        return (lo, hi, PanelBox(x=ux, y=uy, w=uw, h=uh))
