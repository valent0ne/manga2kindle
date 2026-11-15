"""OpenCV-based manga panel detector with contour filtering, IoU NMS, and bleed clamping."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga2kindle.cv.io import read_image_bgr
from manga2kindle.models import PanelBox

_MAX_PANEL_AREA_RATIO = 0.98
_MAX_ASPECT_RATIO = 15.0
_BLEED_SAFETY_MARGIN = 2

Rect = tuple[int, int, int, int]


def compute_iou(box_a: Rect, box_b: Rect) -> float:
    """Compute Intersection-over-Union (IoU) for two (x, y, w, h) rectangles."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


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
        """Prune smaller overlapping boxes whose IoU with a larger kept box exceeds threshold."""
        sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        kept: list[Rect] = []
        for candidate in sorted_boxes:
            if all(compute_iou(candidate, existing) <= iou_threshold for existing in kept):
                kept.append(candidate)
        return kept

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
        x, y, w, h = rect
        pad_x = int(round(w * self.padding))
        pad_y = int(round(h * self.padding))

        margin_x = _BLEED_SAFETY_MARGIN if page_w > 2 * _BLEED_SAFETY_MARGIN else 0
        margin_y = _BLEED_SAFETY_MARGIN if page_h > 2 * _BLEED_SAFETY_MARGIN else 0

        x1 = max(margin_x, x - pad_x)
        y1 = max(margin_y, y - pad_y)
        x2 = min(page_w - margin_x, x + w + pad_x)
        y2 = min(page_h - margin_y, y + h + pad_y)

        return PanelBox(
            x=x1,
            y=y1,
            w=max(1, x2 - x1),
            h=max(1, y2 - y1),
        )
