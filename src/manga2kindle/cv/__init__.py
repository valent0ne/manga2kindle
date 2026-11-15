"""OpenCV panel detection, spatial sorting, and Unicode-safe image I/O."""

from manga2kindle.cv.detector import PanelDetector, compute_iou
from manga2kindle.cv.io import read_image_bgr, write_image
from manga2kindle.cv.preview import PanelPreviewRenderer
from manga2kindle.cv.sorter import PanelSorter

__all__ = [
    "PanelDetector",
    "PanelPreviewRenderer",
    "PanelSorter",
    "compute_iou",
    "read_image_bgr",
    "write_image",
]
