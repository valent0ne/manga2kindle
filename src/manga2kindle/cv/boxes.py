"""Shared bounding-box geometry, containment suppression, IoU NMS, and bleed clamping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from manga2kindle.models import (
    DEFAULT_MAX_MERGED_AREA_RATIO,
    DEFAULT_MIN_FILL_RATIO,
    DEFAULT_MIN_TEXT_AREA_RATIO,
    DEFAULT_PANEL_OVERLAP_THRESHOLD,
    DEFAULT_TEXT_BRIDGE_OVERLAP,
    PanelBox,
)

Rect = tuple[int, int, int, int]

DEFAULT_IOU_THRESHOLD = 0.4
DEFAULT_CONTAINMENT_THRESHOLD = 0.85
DEFAULT_PADDING = 0.02
DEFAULT_BLEED_MARGIN = 2


@dataclass(frozen=True)
class MergedCluster:
    """A merged panel bounding box and the sorted indices of its member panels."""

    rect: Rect
    member_indices: tuple[int, ...]


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


def _union_rect(box_a: Rect, box_b: Rect) -> Rect:
    """Return the axis-aligned bounding rectangle enclosing both boxes."""
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y2 = max(box_a[1] + box_a[3], box_b[1] + box_b[3])
    return (x1, y1, x2 - x1, y2 - y1)


def merge_bridged_panel_clusters(
    boxes: Sequence[Rect],
    page_width: int,
    page_height: int,
    *,
    text_boxes: Sequence[Rect] = (),
    text_bridge_overlap: float = DEFAULT_TEXT_BRIDGE_OVERLAP,
    panel_overlap_threshold: float = DEFAULT_PANEL_OVERLAP_THRESHOLD,
    max_merged_area_ratio: float = DEFAULT_MAX_MERGED_AREA_RATIO,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
    min_text_area_ratio: float = DEFAULT_MIN_TEXT_AREA_RATIO,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> list[MergedCluster]:
    """Merge panel boxes bridged by shared text/dialogue boxes or substantial overlap.

    Enforces readability guardrails (``max_merged_area_ratio``, ``min_fill_ratio``,
    and third-panel engulfment) so merges never collapse a page into an unreadable crop.
    Returned clusters are ordered by each cluster's earliest member index.
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError(
            f"Page dimensions must be positive, got ({page_width}, {page_height})"
        )

    valid_indices = [i for i, b in enumerate(boxes) if b[2] > 0 and b[3] > 0]
    if not valid_indices:
        return []

    page_area = page_width * page_height
    edge_weights: dict[tuple[int, int], float] = {}

    def _record_edge(a: int, b: int, weight: float) -> None:
        key = (a, b) if a < b else (b, a)
        prev = edge_weights.get(key)
        if prev is None or weight > prev:
            edge_weights[key] = weight

    # 1a. Text-bridge candidate edges.
    for t_box in text_boxes:
        t_area = _box_area(t_box)
        if t_area <= 0 or (t_area / page_area) < min_text_area_ratio:
            continue
        matched: list[tuple[int, float]] = []
        for idx in valid_indices:
            frac = _intersection_area(t_box, boxes[idx]) / t_area
            if frac >= text_bridge_overlap:
                matched.append((idx, frac))
        if len(matched) >= 2:
            for p_pos in range(len(matched)):
                idx_a, frac_a = matched[p_pos]
                for q_pos in range(p_pos + 1, len(matched)):
                    idx_b, frac_b = matched[q_pos]
                    _record_edge(idx_a, idx_b, min(frac_a, frac_b))

    # 1b. Direct panel-overlap candidate edges.
    for pos_a in range(len(valid_indices)):
        idx_a = valid_indices[pos_a]
        for pos_b in range(pos_a + 1, len(valid_indices)):
            idx_b = valid_indices[pos_b]
            overlap = compute_containment(boxes[idx_a], boxes[idx_b])
            if overlap >= panel_overlap_threshold:
                _record_edge(idx_a, idx_b, overlap)

    sorted_edges = sorted(
        edge_weights.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )

    # 2. Guarded Union-Find.
    parent = {i: i for i in valid_indices}
    cluster_members = {i: [i] for i in valid_indices}
    cluster_rects = {i: boxes[i] for i in valid_indices}
    cluster_member_areas = {i: _box_area(boxes[i]) for i in valid_indices}

    def _find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for (idx_a, idx_b), _weight in sorted_edges:
        root_a = _find(idx_a)
        root_b = _find(idx_b)
        if root_a == root_b:
            continue

        roots_to_merge = {root_a, root_b}
        union_rect = _union_rect(cluster_rects[root_a], cluster_rects[root_b])

        # Absorb any external cluster engulfed by the merged bounding box so we
        # preserve bridged layouts without creating nested duplicate panels.
        while True:
            engulfed_roots = {
                _find(k)
                for k in valid_indices
                if _find(k) not in roots_to_merge
                and compute_containment(union_rect, boxes[k]) >= containment_threshold
            }
            if not engulfed_roots:
                break
            roots_to_merge.update(engulfed_roots)
            for eroot in engulfed_roots:
                union_rect = _union_rect(union_rect, cluster_rects[eroot])

        union_area = _box_area(union_rect)
        if (union_area / page_area) > max_merged_area_ratio:
            continue

        combined_area = sum(cluster_member_areas[r] for r in roots_to_merge)
        if union_area <= 0 or (combined_area / union_area) < min_fill_ratio:
            continue

        for other_root in roots_to_merge:
            if other_root == root_a:
                continue
            parent[other_root] = root_a
            cluster_members[root_a].extend(cluster_members.pop(other_root))
            cluster_rects.pop(other_root)
            cluster_member_areas.pop(other_root)

        cluster_rects[root_a] = union_rect
        cluster_member_areas[root_a] = combined_area

    # 3. Expand clusters to enclose attached gutter/margin text boxes that protrude
    # outside a single panel (or sit immediately adjacent in the white margin).
    margin_px = max(8, int(round(0.02 * min(page_width, page_height))))
    active_roots = list(cluster_rects.keys())
    for t_box in text_boxes:
        t_area = _box_area(t_box)
        if t_area <= 0 or (t_area / page_area) < min_text_area_ratio:
            continue

        direct_matches = [
            (r, _intersection_area(t_box, cluster_rects[r]) / t_area)
            for r in active_roots
            if _intersection_area(t_box, cluster_rects[r]) / t_area >= text_bridge_overlap
        ]
        target_root: int | None = None
        if len(direct_matches) == 1:
            only_root, frac = direct_matches[0]
            if frac < 1.0:
                target_root = only_root
        elif not direct_matches:
            prox_matches: list[int] = []
            for r in active_roots:
                rx, ry, rw, rh = cluster_rects[r]
                expanded_r = (
                    max(0, rx - margin_px),
                    max(0, ry - margin_px),
                    min(page_width, rx + rw + margin_px) - max(0, rx - margin_px),
                    min(page_height, ry + rh + margin_px) - max(0, ry - margin_px),
                )
                if _intersection_area(t_box, expanded_r) > 0:
                    prox_matches.append(r)
            if len(prox_matches) == 1:
                target_root = prox_matches[0]

        if target_root is None:
            continue

        candidate_rect = _union_rect(cluster_rects[target_root], t_box)
        if any(
            other != target_root
            and compute_containment(candidate_rect, cluster_rects[other]) >= containment_threshold
            for other in active_roots
        ):
            continue
        cluster_rects[target_root] = candidate_rect

    clusters = [
        MergedCluster(
            rect=cluster_rects[root],
            member_indices=tuple(sorted(members)),
        )
        for root, members in cluster_members.items()
    ]
    clusters.sort(key=lambda c: c.member_indices[0])
    return clusters


def merge_bridged_panels(
    boxes: Sequence[Rect],
    page_width: int,
    page_height: int,
    *,
    text_boxes: Sequence[Rect] = (),
    text_bridge_overlap: float = DEFAULT_TEXT_BRIDGE_OVERLAP,
    panel_overlap_threshold: float = DEFAULT_PANEL_OVERLAP_THRESHOLD,
    max_merged_area_ratio: float = DEFAULT_MAX_MERGED_AREA_RATIO,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
    min_text_area_ratio: float = DEFAULT_MIN_TEXT_AREA_RATIO,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> list[Rect]:
    """Convenience wrapper returning only merged ``Rect``s in cluster order."""
    clusters = merge_bridged_panel_clusters(
        boxes,
        page_width,
        page_height,
        text_boxes=text_boxes,
        text_bridge_overlap=text_bridge_overlap,
        panel_overlap_threshold=panel_overlap_threshold,
        max_merged_area_ratio=max_merged_area_ratio,
        min_fill_ratio=min_fill_ratio,
        min_text_area_ratio=min_text_area_ratio,
        containment_threshold=containment_threshold,
    )
    return [cluster.rect for cluster in clusters]


def finalize_panel_boxes(
    boxes: Sequence[Rect],
    page_w: int,
    page_h: int,
    *,
    text_boxes: Sequence[Rect] = (),
    scores: Sequence[float] | None = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    padding: float = DEFAULT_PADDING,
    bleed_margin: int = DEFAULT_BLEED_MARGIN,
    preserve_order: bool = False,
    merge_bridged_panels: bool = True,
    text_bridge_overlap: float = DEFAULT_TEXT_BRIDGE_OVERLAP,
    panel_overlap_threshold: float = DEFAULT_PANEL_OVERLAP_THRESHOLD,
    max_merged_area_ratio: float = DEFAULT_MAX_MERGED_AREA_RATIO,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
) -> tuple[list[PanelBox], list[float] | None]:
    """Run NMS/containment pruning, optional bridged-panel merging, and pad/clamp."""
    if page_w <= 0 or page_h <= 0:
        raise ValueError(f"Page dimensions must be positive, got ({page_w}, {page_h})")

    surviving = nms_and_prune_indices(
        boxes,
        iou_threshold=iou_threshold,
        containment_threshold=containment_threshold,
        scores=scores,
        preserve_order=preserve_order,
    )
    kept_rects = [boxes[i] for i in surviving]
    kept_scores = [float(scores[i]) for i in surviving] if scores is not None else None

    if not kept_rects:
        return [], ([] if kept_scores is not None else None)

    if merge_bridged_panels:
        clusters = merge_bridged_panel_clusters(
            kept_rects,
            page_w,
            page_h,
            text_boxes=text_boxes,
            text_bridge_overlap=text_bridge_overlap,
            panel_overlap_threshold=panel_overlap_threshold,
            max_merged_area_ratio=max_merged_area_ratio,
            min_fill_ratio=min_fill_ratio,
            containment_threshold=containment_threshold,
        )
        final_rects = [cluster.rect for cluster in clusters]
        out_scores = (
            [max(kept_scores[m] for m in cluster.member_indices) for cluster in clusters]
            if kept_scores is not None
            else None
        )
    else:
        final_rects = kept_rects
        out_scores = kept_scores

    panels: list[PanelBox] = []
    for idx, rect in enumerate(final_rects, start=1):
        panel = pad_and_clamp(
            rect,
            page_w,
            page_h,
            padding=padding,
            bleed_margin=bleed_margin,
        )
        if preserve_order:
            panel = panel.model_copy(update={"ordinal": idx})
        panels.append(panel)

    return panels, out_scores
