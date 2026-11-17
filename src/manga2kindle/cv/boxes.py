"""Shared bounding-box geometry, containment suppression, IoU NMS, and bleed clamping."""

from __future__ import annotations

from collections.abc import Sequence

from manga2kindle.models import PanelBox

Rect = tuple[int, int, int, int]

DEFAULT_IOU_THRESHOLD = 0.4
DEFAULT_CONTAINMENT_THRESHOLD = 0.85
DEFAULT_PADDING = 0.02
DEFAULT_BLEED_MARGIN = 2


def _box_area(box: Rect) -> int:
    """Return non-negative pixel area of an (x, y, w, h) rectangle."""
    return max(0, box[2]) * max(0, box[3])


def _intersection_area(box_a: Rect, box_b: Rect) -> int:
    """Return pixel area of the overlap between two (x, y, w, h) rectangles."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    return inter_w * inter_h


def compute_iou(box_a: Rect, box_b: Rect) -> float:
    """Compute Intersection-over-Union (IoU) for two (x, y, w, h) rectangles."""
    inter_area = _intersection_area(box_a, box_b)
    if inter_area == 0:
        return 0.0

    union_area = _box_area(box_a) + _box_area(box_b) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_containment(box_a: Rect, box_b: Rect) -> float:
    """Compute intersection area divided by the smaller box's area."""
    min_area = min(_box_area(box_a), _box_area(box_b))
    if min_area <= 0:
        return 0.0
    return _intersection_area(box_a, box_b) / min_area


def nms_and_prune_indices(
    boxes: Sequence[Rect],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    scores: Sequence[float] | None = None,
    preserve_order: bool = False,
) -> list[int]:
    """Return surviving indices after containment pruning and IoU NMS.

    Stage 1 (containment pass): sorted by area descending (ties by score descending),
    drops smaller candidates mostly contained inside a larger kept box.
    Stage 2 (IoU NMS pass): sorted by (score, area) descending when scores are provided
    else area descending, drops overlapping candidates exceeding ``iou_threshold``.
    """
    if scores is not None and len(scores) != len(boxes):
        raise ValueError(
            f"scores length ({len(scores)}) must match boxes length ({len(boxes)})"
        )
    if not boxes:
        return []

    areas = [_box_area(b) for b in boxes]
    score_vals = [float(s) for s in scores] if scores is not None else [0.0] * len(boxes)

    # Stage 1: Containment suppression (larger boxes always take precedence).
    by_area = sorted(
        range(len(boxes)),
        key=lambda i: (areas[i], score_vals[i]),
        reverse=True,
    )
    containment_kept: list[int] = []
    for idx in by_area:
        cand = boxes[idx]
        cand_area = areas[idx]
        if any(
            cand_area < areas[kept_idx]
            and compute_containment(cand, boxes[kept_idx]) > containment_threshold
            for kept_idx in containment_kept
        ):
            continue
        containment_kept.append(idx)

    # Stage 2: IoU NMS.
    if scores is not None:
        nms_order = sorted(
            containment_kept,
            key=lambda i: (score_vals[i], areas[i]),
            reverse=True,
        )
    else:
        nms_order = containment_kept

    surviving: list[int] = []
    for idx in nms_order:
        cand = boxes[idx]
        if all(compute_iou(cand, boxes[kept_idx]) <= iou_threshold for kept_idx in surviving):
            surviving.append(idx)

    if preserve_order:
        surviving.sort()
    return surviving


def nms_and_prune(
    boxes: Sequence[Rect],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    scores: Sequence[float] | None = None,
    preserve_order: bool = False,
) -> list[Rect]:
    """Return surviving boxes after containment pruning and IoU NMS."""
    indices = nms_and_prune_indices(
        boxes,
        iou_threshold=iou_threshold,
        containment_threshold=containment_threshold,
        scores=scores,
        preserve_order=preserve_order,
    )
    return [boxes[i] for i in indices]


def pad_and_clamp(
    rect: Rect,
    page_w: int,
    page_h: int,
    padding: float = DEFAULT_PADDING,
    bleed_margin: int = DEFAULT_BLEED_MARGIN,
) -> PanelBox:
    """Apply single-pass padding expansion and clamp within bleed-safe page bounds."""
    if page_w <= 0 or page_h <= 0:
        raise ValueError(f"Page dimensions must be positive, got ({page_w}, {page_h})")

    x, y, w, h = rect
    pad_x = int(round(w * padding))
    pad_y = int(round(h * padding))

    margin_x = bleed_margin if page_w > 2 * bleed_margin else 0
    margin_y = bleed_margin if page_h > 2 * bleed_margin else 0

    x1 = max(margin_x, x - pad_x)
    y1 = max(margin_y, y - pad_y)
    x2 = min(page_w - margin_x, x + w + pad_x)
    y2 = min(page_h - margin_y, y + h + pad_y)

    if margin_x > 0:
        x1 = min(x1, page_w - margin_x - 1)
    if margin_y > 0:
        y1 = min(y1, page_h - margin_y - 1)

    return PanelBox(
        x=x1,
        y=y1,
        w=max(1, x2 - x1),
        h=max(1, y2 - y1),
    )
