"""Pluggable panel detection backends."""

from manga2kindle.cv.backends.base import DetectionResult, PanelDetectorBackend
from manga2kindle.cv.backends.contour import ContourBackend

__all__ = [
    "ContourBackend",
    "DetectionResult",
    "PanelDetectorBackend",
]
