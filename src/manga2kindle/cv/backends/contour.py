"""OpenCV morphological contour backend for manga panel detection."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from manga2kindle.cv.backends.base import DetectionResult
from manga2kindle.cv.boxes import (
    DEFAULT_BLEED_MARGIN,
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_PADDING,
    Rect,
    nms_and_prune,
    pad_and_clamp,
)
from manga2kindle.cv.io import read_image_bgr
from manga2kindle.models import DetectorKind

_MAX_PANEL_AREA_RATIO = 0.98
_MAX_ASPECT_RATIO = 15.0
_MIN_INK_RATIO = 0.02
_MAX_INK_RATIO = 0.98


class ContourBackend:
    """Detects comic/manga panels via bilateral filtering, Otsu thresholding, and contours."""

    kind: DetectorKind = DetectorKind.CONTOUR

    def __init__(
        self,
        min_panel_area: float = 0.03,
        padding: float = DEFAULT_PADDING,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
        min_ink_ratio: float = _MIN_INK_RATIO,
        max_ink_ratio: float = _MAX_INK_RATIO,
    ) -> None:
        self.min_panel_area = min_panel_area
        self.padding = padding
        self.iou_threshold = iou_threshold
        self.containment_threshold = containment_threshold
        self.min_ink_ratio = min_ink_ratio
        self.max_ink_ratio = max_ink_ratio

    def detect(self, image_path: Path) -> DetectionResult:
        """Detect panels on a single page; returns panels=[] when no contours survive."""
        image = read_image_bgr(image_path)
        page_h, page_w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = self._filter_contours(contours, binary, page_w, page_h)
        surviving = nms_and_prune(
            candidates,
            iou_threshold=self.iou_threshold,
            containment_threshold=self.containment_threshold,
        )

        panels = [
            pad_and_clamp(
                rect,
                page_w,
                page_h,
                padding=self.padding,
                bleed_margin=DEFAULT_BLEED_MARGIN,
            )
            for rect in surviving
        ]
        return DetectionResult(
            panels=panels,
            ordered=False,
            backend=self.kind,
            page_width=page_w,
            page_height=page_h,
        )

    def detect_batch(
        self,
        image_paths: Sequence[Path],
        workers: int = 1,
    ) -> list[DetectionResult]:
        """Detect panels across multiple pages, optionally using a thread pool."""
        if not image_paths:
            return []
        if workers <= 1 or len(image_paths) <= 1:
            return [self.detect(path) for path in image_paths]

        max_workers = min(workers, len(image_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.detect, image_paths))

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
