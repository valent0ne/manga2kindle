"""Tests for shared bounding-box geometry, containment suppression, and clamping."""

from __future__ import annotations

import pytest

from manga2kindle.cv.boxes import (
    DEFAULT_MAX_MERGED_AREA_RATIO,
    DEFAULT_MIN_FILL_RATIO,
    DEFAULT_MIN_TEXT_AREA_RATIO,
    DEFAULT_PANEL_OVERLAP_THRESHOLD,
    DEFAULT_PADDING,
    DEFAULT_TEXT_BRIDGE_OVERLAP,
    GUTTER_INSET_MAX_PX,
    CutAxis,
    MergedCluster,
    axis_interval,
    compute_containment,
    compute_iou,
    finalize_panel_boxes,
    merge_bridged_panel_clusters,
    merge_bridged_panels,
    nms_and_prune,
    nms_and_prune_indices,
    pad_and_clamp,
    panel_to_rect,
    split_by_gutter,
    union_rect,
    unpad_rect,
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
        text_bridge_overlap=0.15,
    )
    assert len(clusters) == 2
    assert [c.member_indices for c in clusters] == [(0,), (1,)]


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
    # Weights: (0,1)=0.2625, (0,2)=0.25, (1,2)=0.25 -> (0,1) merges first (360_000 <= 550_000),
    # then merging with p2 would be 900x900=810_000 > 550_000 so rejected!
    # Merged cluster (0, 1) holds 400x70 = 28_000 / 48_000 (58.3% >= _DOMINANT_CLUSTER_MIN_TEXT_FRAC)
    # of `bridge`, so it expands downward to y=500 to keep `bridge` intact while keeping member_indices split.
    clusters = merge_bridged_panel_clusters(
        [p0, p1, p2],
        page_width=1000,
        page_height=1000,
        text_boxes=[bridge],
    )
    assert clusters == [
        MergedCluster(rect=(50, 50, 900, 450), member_indices=(0, 1)),
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

    # Engulfment: two panels whose union fills >= 0.55 and <= max_merged_area_ratio,
    # absorbing an unrelated third panel inside the gap/union.
    left = (50, 50, 300, 500)     # 150_000
    right = (450, 50, 300, 500)   # 150_000
    unrelated = (360, 350, 80, 150)
    bridge = (300, 100, 200, 100)
    clusters_engulf = merge_bridged_panel_clusters(
        [left, right, unrelated],
        page_width=1000,
        page_height=1000,
        text_boxes=[bridge],
    )
    assert len(clusters_engulf) == 1
    assert clusters_engulf[0].member_indices == (0, 1, 2)

    # Absorbing an engulfed panel is rejected if the total union exceeds max_merged_area_ratio.
    clusters_engulf_capped = merge_bridged_panel_clusters(
        [left, right, unrelated],
        page_width=1000,
        page_height=1000,
        text_boxes=[bridge],
        max_merged_area_ratio=0.30,
    )
    assert len(clusters_engulf_capped) == 3


def test_merge_bridged_panel_clusters_expands_for_gutter_and_margin_text() -> None:
    p0 = (50, 50, 400, 300)   # x=50..450, y=50..350
    p1 = (50, 420, 400, 300)  # well below p0
    # Text box protruding below p0 into the gutter (75% inside p0, 25% below)
    gutter_tail = (200, 290, 100, 80)  # y=290..370
    # Margin narration box sitting 10px to the right of p0 (0% overlap, within 2% margin=20px)
    margin_note = (460, 100, 80, 150)  # x=460..540

    clusters = merge_bridged_panel_clusters(
        [p0, p1],
        page_width=1000,
        page_height=1000,
        text_boxes=[gutter_tail, margin_note],
    )
    assert len(clusters) == 2
    assert clusters[0].rect == (50, 50, 490, 320)  # x=50..540, y=50..370
    assert clusters[1].rect == p1


def test_merge_bridged_panel_clusters_expands_dominant_panel_when_merge_capped() -> None:
    # Two large panels whose merge is blocked by max_merged_area_ratio,
    # with a text box 80% in top panel and 20% in bottom panel.
    top = (50, 50, 900, 600)      # 540_000
    bottom = (50, 660, 900, 290)  # 261_000; union = 900x900 = 810_000 (> 0.70)
    protruding = (700, 570, 100, 100)  # y=570..670 -> 80% in top, 10% in bottom

    clusters = merge_bridged_panel_clusters(
        [top, bottom],
        page_width=1000,
        page_height=1000,
        text_boxes=[protruding],
    )
    assert len(clusters) == 2
    assert clusters[0].rect == (50, 50, 900, 620)  # top expanded to y=670
    assert clusters[1].rect == bottom


def test_merge_bridged_panel_clusters_merges_same_row_tiny_adjacent_panel() -> None:
    # Wide strip (650x220 = 143_000) + tiny adjacent box (120x220 = 26_400 < 0.04 * 1_000_000)
    wide = (50, 720, 650, 220)
    tiny = (715, 720, 120, 220)  # 15px gap <= 0.02 * 1000 (= 20px), >= 3x width ratio

    clusters = merge_bridged_panel_clusters(
        [wide, tiny],
        page_width=1000,
        page_height=1000,
    )
    assert len(clusters) == 1
    assert clusters[0].rect == (50, 720, 785, 220)
    assert clusters[0].member_indices == (0, 1)


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


def test_panel_to_rect_and_union_rect() -> None:
    b1 = PanelBox(x=100, y=50, w=200, h=300)
    b2 = PanelBox(x=250, y=100, w=150, h=350)
    assert panel_to_rect(b1) == (100, 50, 200, 300)
    assert union_rect([b1, b2]) == (100, 50, 300, 400)
    with pytest.raises(ValueError):
        union_rect([])


def test_unpad_rect_inverts_padding_and_preserves_zero() -> None:
    raw = (200, 300, 500, 800)
    assert unpad_rect(raw, 0.0) == raw
    padded = pad_and_clamp(raw, page_w=2000, page_h=2600, padding=DEFAULT_PADDING, bleed_margin=0)
    unpadded = unpad_rect(panel_to_rect(padded), DEFAULT_PADDING)
    assert unpadded == raw
    # Tiny box still preserves at least 1x1 dimensions
    assert unpad_rect((10, 10, 2, 2), 0.1) == (10, 10, 2, 2)


def test_axis_interval_and_split_by_gutter() -> None:
    # Wide box exceeds GUTTER_INSET_MAX_PX when uncapped
    wide = PanelBox(x=100, y=200, w=1000, h=400)
    assert axis_interval(wide, CutAxis.VERTICAL, inset=False) == (100, 1100)
    assert axis_interval(wide, CutAxis.VERTICAL, inset=True) == (
        100 + GUTTER_INSET_MAX_PX,
        1100 - GUTTER_INSET_MAX_PX,
    )
    assert axis_interval(wide, CutAxis.VERTICAL, inset=True, inset_max_px=None) == (150, 1050)

    # High-res panels padded with 0.04 overlap > 24px across a 10px vertical gutter
    raw_hero = (700, 100, 1100, 800)
    raw_sub = (100, 100, 590, 380)
    p_hero = pad_and_clamp(raw_hero, page_w=1908, page_h=2544, padding=0.04, bleed_margin=0)
    p_sub = pad_and_clamp(raw_sub, page_w=1908, page_h=2544, padding=0.04, bleed_margin=0)
    # Capped inset (24px) fails to split across 10px gutter, uncapped 6% splits cleanly
    assert len(split_by_gutter([p_hero, p_sub], CutAxis.VERTICAL, inset=True)) == 1
    cols = split_by_gutter(
        [p_hero, p_sub],
        CutAxis.VERTICAL,
        inset=True,
        reverse=True,
        inset_ratio=0.06,
        inset_max_px=None,
    )
    assert len(cols) == 2
    assert cols[0] == [p_hero]
    assert cols[1] == [p_sub]

    # Horizontal strict gutter split
    top = PanelBox(x=50, y=40, w=900, h=300)
    bot = PanelBox(x=50, y=360, w=900, h=400)
    assert split_by_gutter([bot, top], CutAxis.HORIZONTAL) == [[top], [bot]]


def test_merge_bridged_panel_clusters_rejects_partial_slice_bridge_cascade() -> None:
    """L-shaped bridge slicing >=25% across unrelated panels is rejected; clean sub-merge & text expansion succeed."""
    # Models Ch.69 p.0013 (1000x1000 normalized): middle wide P2, bottom-right P3/P4, P5, P6
    p2 = (50, 330, 900, 320)    # middle full-width
    p3 = (680, 665, 270, 155)   # bottom-right top
    p4 = (675, 830, 275, 165)   # bottom-right bottom
    p5 = (440, 665, 230, 330)   # bottom middle (~46% sliced if (P2, P3) merged!)
    p6 = (60, 670, 370, 325)    # bottom left (~45% sliced if (P2, P3) merged!)
    # Bubble bridges P2 (28%) and P3 (69%): must NOT merge (P2, P3), instead P3 expands up after (P3, P4, P5) merges
    bubble_p2_p3 = (690, 580, 210, 230)
    # Bubble bridges P4 (72%) and P5 (23%): clean merge of (P4, P5) which engulfs P3 (~100%)
    bubble_p4_p5 = (615, 815, 225, 155)

    clusters = merge_bridged_panel_clusters(
        [p2, p3, p4, p5, p6],
        page_width=1000,
        page_height=1000,
        text_boxes=[bubble_p2_p3, bubble_p4_p5],
        max_merged_area_ratio=0.55,
    )
    assert [c.member_indices for c in clusters] == [(0,), (1, 2, 3), (4,)]
    assert clusters[0].rect == p2
    # Bottom-right+mid cluster expanded upward to y=580 to enclose bubble_p2_p3 completely
    assert clusters[1].rect == (440, 580, 510, 415)
    assert clusters[2].rect == p6


def test_merge_bridged_panel_clusters_encloses_stranded_and_floating_margin_text() -> None:
    """Stranded 2-neighbor gutter text and wide floating margin text attach to zero-conflict panels without bridging."""
    # Case 1 (Ch.70 p.0009): text overlaps right panel by 3.6% (<5%) and sits 4px from left panel
    p_right = (504, 10, 450, 440)
    p_left = (50, 10, 344, 440)
    p_mid = (50, 500, 904, 300)
    stranded = (398, 150, 110, 140)  # touches p_right 4px, gap 4px from p_left
    c_stranded = merge_bridged_panel_clusters(
        [p_right, p_left, p_mid],
        page_width=1000,
        page_height=1000,
        text_boxes=[stranded],
    )
    assert [c.member_indices for c in c_stranded] == [(0,), (1,), (2,)]
    assert c_stranded[0].rect == (398, 10, 556, 440)  # p_right enclosed stranded without touching p_left
    assert c_stranded[1].rect == p_left

    # Case 2 (Ch.69 p.0018 + Ch.70 p.0015): margin_text_boxes never merge, pick zero-conflict panel, and allow <=0.85 expansion
    top_wide = (80, 10, 840, 260)
    mid_right = (510, 330, 410, 250)
    mid_left = (80, 330, 410, 250)
    caption_below_top = (200, 290, 600, 26)  # 20px below top_wide, 14px above split mid_right/mid_left -> only top_wide has 0 overlap conflict!

    c_caption = merge_bridged_panel_clusters(
        [top_wide, mid_right, mid_left],
        page_width=1000,
        page_height=1000,
        margin_text_boxes=[caption_below_top],
        max_merged_area_ratio=0.55,
    )
    assert len(c_caption) == 3
    assert c_caption[0].rect == (80, 10, 840, 306)  # expanded down to y=316, leaving mid_right/mid_left untouched
    assert c_caption[1].rect == mid_right
    assert c_caption[2].rect == mid_left

    # Case 3 (Ch.70 p.0015 area > 0.55): single panel at 0.55 area expands up to <= 0.85 for floating caption
    upper_wide = (60, 100, 880, 200)
    lower_large = (60, 380, 880, 610)  # area = 0.537; expanded up to y=330 -> area = 0.581 (> 0.55, <= 0.85)
    lower_caption = (200, 330, 600, 26)  # 24px above lower_large vs 30px below upper_wide -> closer to lower_large
    c_large = merge_bridged_panel_clusters(
        [upper_wide, lower_large],
        page_width=1000,
        page_height=1000,
        margin_text_boxes=[lower_caption],
        max_merged_area_ratio=0.55,
    )
    assert len(c_large) == 2
    assert c_large[0].rect == upper_wide
    assert c_large[1].rect == (60, 330, 880, 660)


def test_merge_bridged_panel_clusters_alcove_fallback_essential_only_and_bounded() -> None:
    """Essential text in an open alcove beyond the 4% halo expands the unique zero-conflict panel."""
    # Models Ch.71 p.0015 (1000x1000): 4% margin_px=40, 12% alcove_px=120
    p_right = (440, 10, 480, 440)   # gap 60px from alcove_text (collides with p_above if expanded left)
    p_above = (80, 10, 340, 260)    # y=10..270 -> gap 70px (>40, <=120); added area = 340*110 = 37_400 (3.7% <= 8%)
    p_below = (80, 470, 840, 480)   # gap 90px below alcove_text (collides with p_right if expanded up)
    alcove_text = (150, 340, 220, 40)  # y=340..380

    # Essential text -> expands p_above down to y=380 with zero overlap on p_right or p_below
    c_ess = merge_bridged_panel_clusters(
        [p_right, p_above, p_below],
        page_width=1000,
        page_height=1000,
        text_boxes=[alcove_text],
    )
    assert [c.rect for c in c_ess] == [p_right, (80, 10, 340, 370), p_below]

    # Non-essential margin_text_boxes at the exact same distance must NOT trigger alcove expansion
    c_margin = merge_bridged_panel_clusters(
        [p_right, p_above, p_below],
        page_width=1000,
        page_height=1000,
        margin_text_boxes=[alcove_text],
    )
    assert [c.rect for c in c_margin] == [p_right, p_above, p_below]

    # Essential alcove text requiring > 8% added area is rejected by the bounded bloat guard
    wide_top = (80, 10, 840, 200)
    far_text = (200, 315, 400, 30)  # 105px below wide_top -> added area = 840*135 = 113_400 (11.3% > 8%)
    c_bloat = merge_bridged_panel_clusters(
        [wide_top],
        page_width=1000,
        page_height=1000,
        text_boxes=[far_text],
    )
    assert c_bloat[0].rect == wide_top
