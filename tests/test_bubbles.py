"""Tests for OpenCV bubble-contour expansion around detected text boxes."""

from __future__ import annotations

import cv2
import numpy as np

from manga2kindle.cv.bubbles import expand_text_boxes_to_bubbles


def test_expand_text_boxes_to_bubbles_grows_bounded_bubble() -> None:
    # 1000x1000 gray/dark page with a white speech bubble (200x160) drawn in black outline.
    img = np.full((1000, 1000, 3), 120, dtype=np.uint8)
    cv2.rectangle(img, (300, 200), (500, 360), (255, 255, 255), thickness=-1)
    cv2.rectangle(img, (300, 200), (500, 360), (0, 0, 0), thickness=4)
    # Tight text box inside the bubble: 80x60
    tight_text = (360, 250, 80, 60)

    expanded = expand_text_boxes_to_bubbles(img, [tight_text])
    assert len(expanded) == 1
    ex, ey, ew, eh = expanded[0]
    assert ex <= 304 and ey <= 204
    assert ex + ew >= 496 and ey + eh >= 356


def test_expand_text_boxes_to_bubbles_ignores_unbounded_white_background() -> None:
    # Unbounded white page background (area > 15% of page) must NOT explode.
    img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    tight_text = (100, 100, 80, 60)

    expanded = expand_text_boxes_to_bubbles(img, [tight_text])
    assert expanded == [tight_text]


def test_expand_text_boxes_to_bubbles_handles_empty_and_degenerate() -> None:
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert expand_text_boxes_to_bubbles(img, []) == []
    assert expand_text_boxes_to_bubbles(img, [(10, 10, 0, 20)]) == [(10, 10, 0, 20)]
