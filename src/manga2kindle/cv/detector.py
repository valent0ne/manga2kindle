"""OpenCV-based manga panel detector with contour filtering, IoU NMS, and bleed clamping."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga2kindle.cv.boxes import (
    DEFAULT_BLEED_MARGIN,
    Rect,
    compute_iou,
    nms_and_prune,
    pad_and_clamp,
)
from manga2kindle.cv.io import read_image_bgr
from manga2kindle.models import PanelBox

_MAX_PANEL_AREA_RATIO = 0.98
_MAX_ASPECT_RATIO = 15.0
_BLEED_SAFETY_MARGIN = DEFAULT_BLEED_MARGIN


class PanelDetector:
    """Detects comic/manga panel bounding boxes using morphological contour extraction."""

    def __init__(
        self,
        min_panel_area: float = 0.03,
        padding: float = 0.02,
        iou_threshold: float = 0.4,
        min_ink_ratio: float = 0.02,
        max_ink_ratio: float = 0.98,
    ) -> None:
        self.min_panel_area = min_panel_area
        self.padding = padding
        self.iou_threshold = iou_threshold
        self.min_ink_ratio = min_ink_ratio
        self.max_ink_ratio = max_ink_ratio

    def detect(self, image_path: Path) -> list[PanelBox]:
        """Detect manga panels on a page image, falling back to full-page when none survive."""
        image = read_image_bgr(image_path)
        page_h, page_w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = self._filter_contours(contours, binary, page_w, page_h)
        surviving = self.non_max_suppression(candidates, self.iou_threshold)

        if not surviving:
            return [PanelBox(x=0, y=0, w=page_w, h=page_h)]

        return [
            self._pad_and_clamp(rect, page_w, page_h)
            for rect in surviving
        ]

    @staticmethod
    def non_max_suppression(boxes: list[Rect], iou_threshold: float) -> list[Rect]:
        """Prune smaller overlapping or nested boxes via containment and IoU NMS."""
        return nms_and_prune(boxes, iou_threshold=iou_threshold)

    def _filter_contours(
        self,
        contours: tuple[np.ndarray, ...] | list[np.ndarray],
        binary: np.ndarray,
        page_w: int,
        page_h: int,
    ) -> list[Rect]:
        """Filter raw OpenCV contours by area ratio, aspect ratio, and ink density."""
        page_area = page_w * page_h
        candidates: list[Rect] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            area_ratio = (w * h) / page_area
            if not (self.min_panel_area <= area_ratio <= _MAX_PANEL_AREA_RATIO):
                continue

            aspect_ratio = max(w / h, h / w)
            if aspect_ratio > _MAX_ASPECT_RATIO:
                continue

            roi = binary[y : y + h, x : x + w]
            ink_ratio = cv2.countNonZero(roi) / (w * h)
            if not (self.min_ink_ratio <= ink_ratio <= self.max_ink_ratio):
                continue

            candidates.append((x, y, w, h))
        return candidates

    def _pad_and_clamp(self, rect: Rect, page_w: int, page_h: int) -> PanelBox:
        """Apply single-pass padding expansion and clamp within bleed-safe page bounds."""
        return pad_and_clamp(
            rect,
            page_w,
            page_h,
            padding=self.padding,
            bleed_margin=_BLEED_SAFETY_MARGIN,
        )
