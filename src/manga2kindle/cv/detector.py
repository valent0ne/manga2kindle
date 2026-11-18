"""Backward-compatible OpenCV panel detector wrapper delegating to ContourBackend."""

from __future__ import annotations

from pathlib import Path

from manga2kindle.cv.backends.contour import ContourBackend
from manga2kindle.cv.boxes import (
    DEFAULT_BLEED_MARGIN,
    Rect,
    compute_iou,
    nms_and_prune,
    pad_and_clamp,
)
from manga2kindle.models import PanelBox

_BLEED_SAFETY_MARGIN = DEFAULT_BLEED_MARGIN


class PanelDetector:
    """Detects comic/manga panel bounding boxes using morphological contour extraction."""

    def __init__(
        self,
        min_panel_area: float = 0.03,
        padding: float = 0.04,
        iou_threshold: float = 0.4,
        min_ink_ratio: float = 0.02,
        max_ink_ratio: float = 0.98,
    ) -> None:
        self.min_panel_area = min_panel_area
        self.padding = padding
        self.iou_threshold = iou_threshold
        self.min_ink_ratio = min_ink_ratio
        self.max_ink_ratio = max_ink_ratio
        self._backend = ContourBackend(
            min_panel_area=min_panel_area,
            padding=padding,
            iou_threshold=iou_threshold,
            min_ink_ratio=min_ink_ratio,
            max_ink_ratio=max_ink_ratio,
        )

    def detect(self, image_path: Path) -> list[PanelBox]:
        """Detect manga panels on a page image, falling back to full-page when none survive."""
        result = self._backend.detect(image_path)
        if not result.panels:
            return [PanelBox(x=0, y=0, w=result.page_width, h=result.page_height)]
        return result.panels

    @staticmethod
    def non_max_suppression(boxes: list[Rect], iou_threshold: float) -> list[Rect]:
        """Prune smaller overlapping or nested boxes via containment and IoU NMS."""
        return nms_and_prune(boxes, iou_threshold=iou_threshold)

    def _pad_and_clamp(self, rect: Rect, page_w: int, page_h: int) -> PanelBox:
        """Apply single-pass padding expansion and clamp within bleed-safe page bounds."""
        return pad_and_clamp(
            rect,
            page_w,
            page_h,
            padding=self.padding,
            bleed_margin=_BLEED_SAFETY_MARGIN,
        )
