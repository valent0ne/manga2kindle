"""Unicode-safe image reading and writing utilities built on OpenCV and pathlib."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_bgr(path: Path) -> np.ndarray:
    """Read an image from a Unicode/cross-platform Path as a BGR uint8 ndarray."""
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    raw_bytes = np.fromfile(str(path), dtype=np.uint8)
    if raw_bytes.size == 0:
        raise ValueError(f"Image file is empty: {path}")

    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image file: {path}")

    return image


def write_image(path: Path, image: np.ndarray) -> Path:
    """Write an ndarray image to a Unicode/cross-platform Path via imencode."""
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Failed to encode image with extension '{suffix}': {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return path
