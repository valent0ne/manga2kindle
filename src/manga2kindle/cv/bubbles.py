"""OpenCV bubble-contour expansion around detected text boxes."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from manga2kindle.cv.boxes import Rect

_WHITE_THRESHOLD = 210
_MAX_BUBBLE_PAGE_AREA_RATIO = 0.15
_MAX_BUBBLE_DIM_SCALE = 6.0
_MAX_BUBBLE_AREA_SCALE = 12.0
_BORDER_PAD_PX = 2


def expand_text_boxes_to_bubbles(
    image: np.ndarray,
    text_boxes: Sequence[Rect],
    *,
    is_rgb: bool = False,
) -> list[Rect]:
    """Expand tight text bounding boxes to their enclosing white bubble/narration box.

    Uses a single connected-components pass over bright pixels; falls back to the
    original text box whenever the enclosing white region looks like open page
    background/gutter rather than a bounded speech bubble or narration box.
    """
    if not text_boxes:
        return []
    if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        return list(text_boxes)

    page_h, page_w = image.shape[:2]
    page_area = page_w * page_h
    if image.ndim == 3:
        code = cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        gray = image
    white_mask = (gray >= _WHITE_THRESHOLD).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=4)
    if num_labels <= 1:
        return list(text_boxes)

    expanded: list[Rect] = []
    for tx, ty, tw, th in text_boxes:
        if tw <= 0 or th <= 0:
            expanded.append((tx, ty, tw, th))
            continue

        x1 = max(0, tx)
        y1 = max(0, ty)
        x2 = min(page_w, tx + tw)
        y2 = min(page_h, ty + th)
        if x2 <= x1 or y2 <= y1:
            expanded.append((tx, ty, tw, th))
            continue

        roi_labels = labels[y1:y2, x1:x2]
        non_zero = roi_labels[roi_labels > 0]
        if non_zero.size == 0:
            expanded.append((tx, ty, tw, th))
            continue

        unique_lbls, counts = np.unique(non_zero, return_counts=True)
        best_lbl = int(unique_lbls[int(np.argmax(counts))])

        bx = int(stats[best_lbl, cv2.CC_STAT_LEFT])
        by = int(stats[best_lbl, cv2.CC_STAT_TOP])
        bw = int(stats[best_lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[best_lbl, cv2.CC_STAT_HEIGHT])
        b_area = bw * bh
        t_area = tw * th

        if (
            (b_area / page_area) > _MAX_BUBBLE_PAGE_AREA_RATIO
            or bw > _MAX_BUBBLE_DIM_SCALE * tw
            or bh > _MAX_BUBBLE_DIM_SCALE * th
            or b_area > _MAX_BUBBLE_AREA_SCALE * t_area
        ):
            expanded.append((tx, ty, tw, th))
            continue

        ux1 = max(0, min(tx, bx - _BORDER_PAD_PX))
        uy1 = max(0, min(ty, by - _BORDER_PAD_PX))
        ux2 = min(page_w, max(tx + tw, bx + bw + _BORDER_PAD_PX))
        uy2 = min(page_h, max(ty + th, by + bh + _BORDER_PAD_PX))
        expanded.append((ux1, uy1, ux2 - ux1, uy2 - uy1))

    return expanded
