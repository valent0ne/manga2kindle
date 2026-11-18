"""Tests for shared bounding-box geometry, containment suppression, and clamping."""

from __future__ import annotations

import pytest

from manga2kindle.cv.boxes import (
    DEFAULT_MAX_MERGED_AREA_RATIO,
    DEFAULT_MIN_FILL_RATIO,
    DEFAULT_MIN_TEXT_AREA_RATIO,
    DEFAULT_PANEL_OVERLAP_THRESHOLD,
    DEFAULT_TEXT_BRIDGE_OVERLAP,
    MergedCluster,
    compute_containment,
    compute_iou,
    finalize_panel_boxes,
    merge_bridged_panel_clusters,
    merge_bridged_panels,
    nms_and_prune,
    nms_and_prune_indices,
    pad_and_clamp,
)
from manga2kindle.models import DetectorConfig, PanelBox


def test_compute_iou_basic_cases() -> None:
    assert compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0
    assert compute_iou((0, 0, 100, 100), (200, 200, 100, 100)) == 0.0
    assert compute_iou((0, 0, 100, 100), (50, 0, 100, 100)) == pytest.approx(5000 / 15000)
    assert compute_iou((0, 0, 0, 100), (0, 0, 100, 100)) == 0.0


def test_compute_containment_nested_box_is_one() -> None:
    outer = (0, 0, 400, 600)
    inner = (50, 50, 100, 100)
    assert compute_iou(outer, inner) < 0.1
    assert compute_containment(outer, inner) == 1.0
    assert compute_containment(inner, outer) == 1.0
    assert compute_containment(outer, (0, 0, 0, 10)) == 0.0
    assert compute_containment(outer, (500, 700, 100, 100)) == 0.0


def test_nms_and_prune_removes_nested_small_panel_even_with_higher_score() -> None:
    outer = (10, 10, 500, 700)
    inner = (60, 80, 120, 140)
    kept = nms_and_prune(
        [inner, outer],
        scores=[0.95, 0.78],
        iou_threshold=0.4,
        containment_threshold=0.85,
    )
    assert kept == [outer]


def test_nms_and_prune_indices_preserve_order() -> None:
    small_first = (20, 20, 100, 100)
    outer_second = (150, 20, 400, 500)
    inner_third = (200, 80, 80, 80)
    indices = nms_and_prune_indices(
        [small_first, outer_second, inner_third],
        preserve_order=True,
    )
    assert indices == [0, 1]


def test_nms_and_prune_iou_prefers_higher_score_over_larger_area() -> None:
    larger_low_score = (0, 0, 200, 200)  # area 40_000
    smaller_high_score = (10, 10, 180, 180)  # area 32_400, IoU = 0.81 > 0.4, containment = 1.0
    # Use containment_threshold=1.0 so containment pass does not drop smaller_high_score
    # (containment == 1.0 is not > 1.0), leaving IoU pass to decide by score.
    kept = nms_and_prune(
        [larger_low_score, smaller_high_score],
        scores=[0.5, 0.9],
        iou_threshold=0.4,
        containment_threshold=1.0,
    )
    assert kept == [smaller_high_score]


def test_nms_and_prune_empty_and_score_length_mismatch() -> None:
    assert nms_and_prune([]) == []
    assert nms_and_prune_indices([]) == []
    with pytest.raises(ValueError, match="scores"):
        nms_and_prune_indices([(0, 0, 10, 10)], scores=[0.5, 0.6])


def test_pad_and_clamp_applies_padding_bleed_margin_and_guards_page_dims() -> None:
    box = pad_and_clamp((100, 100, 200, 300), page_w=1000, page_h=1400, padding=0.02, bleed_margin=2)
    assert box == PanelBox(x=96, y=94, w=208, h=312)

    edge_box = pad_and_clamp((0, 0, 1000, 1400), page_w=1000, page_h=1400, padding=0.02, bleed_margin=2)
    assert edge_box == PanelBox(x=2, y=2, w=996, h=1396)

    with pytest.raises(ValueError):
        pad_and_clamp((0, 0, 10, 10), page_w=0, page_h=100)
    with pytest.raises(ValueError):
        pad_and_clamp((0, 0, 10, 10), page_w=100, page_h=-5)


def test_merge_bridged_panel_clusters_merges_text_bridged_side_by_side_panels() -> None:
    # Page 1000x1000
    left = (50, 50, 400, 400)
    right = (470, 50, 400, 400)
    # Text box 200x100 = 20_000 px (2% of page), spanning x=350..550 -> 100px in left (50%), 80px in right (40%)
    bridge_text = (350, 150, 200, 100)

    clusters = merge_bridged_panel_clusters(
        [right, left],
        page_width=1000,
        page_height=1000,
        text_boxes=[bridge_text],
    )
    assert clusters == [
        MergedCluster(rect=(50, 50, 820, 400), member_indices=(0, 1)),
    ]


def test_merge_bridged_panel_clusters_ignores_non_bridging_or_tiny_text() -> None:
    left = (50, 50, 400, 400)
    right = (470, 50, 400, 400)
    inside_left = (100, 100, 100, 100)
    # Mostly in left (180px of 200px = 90%), barely in right (10px of 200px = 5% < 15%)
    weak_bridge = (270, 150, 200, 100)
    # Tiny noise straddling seam: 10x10 = 100 px < 0.0005 * 1_000_000 (= 500 px)
    tiny_noise = (455, 150, 10, 10)

    clusters = merge_bridged_panel_clusters(
        [left, right],
        page_width=1000,
        page_height=1000,
        text_boxes=[inside_left, weak_bridge, tiny_noise],
    )
    assert clusters == [
        MergedCluster(rect=left, member_indices=(0,)),
        MergedCluster(rect=right, member_indices=(1,)),
    ]


def test_merge_bridged_panel_clusters_ignores_degenerate_boxes() -> None:
    p1 = (50, 50, 300, 300)
    degenerate_panel = (100, 100, 0, 50)
    degenerate_panel_neg = (100, 100, 50, -10)
    degenerate_text = (200, 200, 0, 100)

    clusters = merge_bridged_panel_clusters(
        [degenerate_panel, p1, degenerate_panel_neg],
        page_width=1000,
        page_height=1000,
        text_boxes=[degenerate_text],
    )
    assert clusters == [MergedCluster(rect=p1, member_indices=(1,))]


def test_merge_bridged_panel_clusters_three_panels_strongest_first_and_area_cap() -> None:
    # Page 1000x1000 -> max_merged_area_ratio=0.65 -> max 650_000 px
    p0 = (50, 50, 450, 400)    # 180_000
    p1 = (520, 50, 430, 400)   # 172_000; union(p0, p1) = 900x400 = 360_000 (<= 0.65)
    p2 = (50, 470, 900, 480)   # union(p0,p1,p2) = 900x900 = 810_000 (> 0.65)
    # Text box bridges p0, p1, p2:
    # x=300..700 (w=400), y=400..520 (h=120) -> area=48_000
    # In p0 (y=400..450, x=300..500): 200x50 = 10_000 (20.8%)
    # In p1 (y=400..450, x=520..700): 180x50 = 9_000 (18.75%)
    # In p2 (y=470..520, x=300..700): 400x50 = 20_000 (41.7%)
    # Pair (0, 2) min=20.8%, (1, 2) min=18.75%, (0, 1) min=18.75%
    # Actually let's make (0, 1) strongest so p0+p1 merge, and adding p2 exceeds area cap.
    bridge = (300, 380, 400, 120)
    # In p0 (x=300..500, y=380..450): 200x70 = 14_000 (29.2%)
    # In p1 (x=520..700, y=380..450): 180x70 = 12_600 (26.25%)
    # In p2 (x=300..700, y=470..500): 400x30 = 12_000 (25.0%)
    # Weights: (0,1)=0.2625, (0,2)=0.25, (1,2)=0.25 -> (0,1) merges first (360_000 <= 650_000),
    # then merging with p2 would be 900x900=810_000 > 650_000 so rejected!
    clusters = merge_bridged_panel_clusters(
        [p0, p1, p2],
        page_width=1000,
        page_height=1000,
        text_boxes=[bridge],
    )
    assert clusters == [
        MergedCluster(rect=(50, 50, 900, 400), member_indices=(0, 1)),
        MergedCluster(rect=p2, member_indices=(2,)),
    ]


def test_merge_bridged_panel_clusters_direct_panel_overlap() -> None:
    p0 = (100, 100, 300, 300)  # area 90_000
    # Overlaps p0 by 150x200 = 30_000; smaller area 200x200 = 40_000 -> containment = 0.75 >= 0.25
    p1_high = (250, 100, 200, 200)
    # Overlaps p0 by 20x200 = 4_000; smaller area 40_000 -> containment = 0.10 < 0.25
    p1_low = (380, 100, 200, 200)

    merged = merge_bridged_panel_clusters([p0, p1_high], page_width=1000, page_height=1000)
    assert merged == [MergedCluster(rect=(100, 100, 350, 300), member_indices=(0, 1))]

    unmerged = merge_bridged_panel_clusters([p0, p1_low], page_width=1000, page_height=1000)
    assert unmerged == [
        MergedCluster(rect=p0, member_indices=(0,)),
        MergedCluster(rect=p1_low, member_indices=(1,)),
    ]


def test_merge_bridged_panel_clusters_rejects_diagonal_dead_space_and_engulfment() -> None:
    # Diagonal panels: p0 top-left (200x200), p1 bottom-right (200x200)
    # Union = 600x600 = 360_000, sum_member_areas = 80_000 -> fill_ratio = 0.22 < 0.55
    p0 = (50, 50, 200, 200)
    p1 = (450, 450, 200, 200)
    diag_bridge = (200, 200, 300, 300)  # overlaps both strongly
    clusters = merge_bridged_panel_clusters(
        [p0, p1],
        page_width=1000,
        page_height=1000,
        text_boxes=[diag_bridge],
    )
    assert len(clusters) == 2

    # Engulfment: two panels whose union fills >= 0.55 and <= 0.65 of page,
    # but swallows an unrelated third panel inside the gap/union.
    left = (50, 50, 300, 500)     # 150_000
    right = (450, 50, 300, 500)   # 150_000
    # Union = (50, 50, 700, 500) = 350_000 (35% of 1000x1000); fill = 300_000 / 350_000 = 0.857 >= 0.55
    # Unrelated panel sits in the gap (360, 100, 80, 300) -> 100% contained in union!
    unrelated = (360, 100, 80, 300)
    bridge = (300, 200, 200, 100)  # bridges left and right (50px/200px=25% each), misses unrelated? Wait:
    # Put unrelated at y=400..500 so bridge at y=100..200 doesn't touch unrelated!
    unrelated = (360, 350, 80, 150)
    bridge = (300, 100, 200, 100)
    clusters_engulf = merge_bridged_panel_clusters(
        [left, right, unrelated],
        page_width=1000,
        page_height=1000,
        text_boxes=[bridge],
    )
    assert len(clusters_engulf) == 3


def test_merge_bridged_panel_clusters_ordering_and_validation() -> None:
    p0 = (500, 50, 400, 400)
    p1 = (50, 50, 300, 300)
    p2 = (370, 50, 100, 300)
    bridge = (300, 150, 120, 80)  # bridges p1 (idx 1) and p2 (idx 2)
    clusters = merge_bridged_panel_clusters(
        [p0, p1, p2],
        page_width=1000,
        page_height=1000,
        text_boxes=[bridge],
    )
    assert [c.member_indices for c in clusters] == [(0,), (1, 2)]

    with pytest.raises(ValueError):
        merge_bridged_panel_clusters([], page_width=0, page_height=1000)
    with pytest.raises(ValueError):
        merge_bridged_panel_clusters([], page_width=1000, page_height=-1)


def test_merge_bridged_panels_wrapper() -> None:
    left = (50, 50, 400, 400)
    right = (470, 50, 400, 400)
    bridge = (350, 150, 200, 100)
    rects = merge_bridged_panels([left, right], 1000, 1000, text_boxes=[bridge])
    assert rects == [(50, 50, 820, 400)]


def test_finalize_panel_boxes_nms_merge_pad_order_and_scores() -> None:
    left = (50, 50, 400, 400)
    nested_in_left = (100, 100, 100, 100)
    right = (470, 50, 400, 400)
    bridge = (350, 150, 200, 100)

    # preserve_order=True -> ordinal 1..N
    panels, out_scores = finalize_panel_boxes(
        [left, nested_in_left, right],
        page_w=1000,
        page_h=1000,
        text_boxes=[bridge],
        scores=[0.6, 0.99, 0.85],
        preserve_order=True,
        merge_bridged_panels=True,
    )
    assert len(panels) == 1
    assert panels[0].ordinal == 1
    # nested_in_left (0.99) was pruned by NMS before merge, so max score is max(0.6, 0.85) = 0.85
    assert out_scores == [pytest.approx(0.85)]

    # preserve_order=False -> ordinal=0
    panels_unordered, out_scores_unordered = finalize_panel_boxes(
        [left, right],
        page_w=1000,
        page_h=1000,
        text_boxes=[bridge],
        scores=[0.6, 0.85],
        preserve_order=False,
        merge_bridged_panels=False,
    )
    assert [p.ordinal for p in panels_unordered] == [0, 0]
    assert len(out_scores_unordered or []) == 2

    # Empty inputs
    assert finalize_panel_boxes([], page_w=1000, page_h=1000, scores=[]) == ([], [])
    assert finalize_panel_boxes([], page_w=1000, page_h=1000, scores=None) == ([], None)


def test_detector_config_merge_fields_validation() -> None:
    cfg = DetectorConfig()
    assert cfg.merge_bridged_panels is True
    assert cfg.text_bridge_overlap == DEFAULT_TEXT_BRIDGE_OVERLAP
    assert cfg.panel_overlap_threshold == DEFAULT_PANEL_OVERLAP_THRESHOLD
    assert cfg.max_merged_area_ratio == DEFAULT_MAX_MERGED_AREA_RATIO
    assert cfg.min_fill_ratio == DEFAULT_MIN_FILL_RATIO
    assert DEFAULT_MIN_TEXT_AREA_RATIO == 0.0005

    with pytest.raises(ValueError):
        DetectorConfig(text_bridge_overlap=0.0)
    with pytest.raises(ValueError):
        DetectorConfig(panel_overlap_threshold=1.5)
    with pytest.raises(ValueError):
        DetectorConfig(max_merged_area_ratio=0.0)
    with pytest.raises(ValueError):
        DetectorConfig(min_fill_ratio=1.1)
