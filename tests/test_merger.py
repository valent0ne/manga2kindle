"""Tests for padding-aware, profile-calibrated Guided View mini-panel merging."""

from __future__ import annotations

import pytest

from manga2kindle.cv.boxes import pad_and_clamp
from manga2kindle.cv.merger import PanelMerger
from manga2kindle.models import DeviceProfile, PanelBox, PanelMergeConfig


def test_merger_merges_compact_flat_four_panel_row_and_assigns_ordinals() -> None:
    """Compact bottom row of 4 mini panels merges into exact union without double-padding."""
    page_w, page_h = 1000, 2000
    top = PanelBox(x=20, y=40, w=960, h=1360, ordinal=1)
    # 4 mini panels in bottom 24% band (RTL order)
    r1 = PanelBox(x=740, y=1480, w=230, h=460, ordinal=2)
    r2 = PanelBox(x=540, y=1490, w=180, h=450, ordinal=3)
    r3 = PanelBox(x=280, y=1485, w=240, h=465, ordinal=4)
    r4 = PanelBox(x=30, y=1495, w=230, h=455, ordinal=5)

    merger = PanelMerger(PanelMergeConfig.for_profile(DeviceProfile.from_name("KPW6")))
    merged = merger.merge([top, r1, r2, r3, r4], page_width=page_w, page_height=page_h)

    assert len(merged) == 2
    assert merged[0] == PanelBox(x=20, y=40, w=960, h=1360, ordinal=1)
    # Exact union of [r1, r2, r3, r4]: x=30..970 (w=940), y=1480..1950 (h=470)
    assert merged[1] == PanelBox(x=30, y=1480, w=940, h=470, ordinal=2)


def test_merger_merges_compact_tier_with_stacked_subpanels_and_padding_overlap() -> None:
    """Compact tier containing stacked pairs merges cleanly even with 15px padding overlap."""
    page_w, page_h = 1000, 2000
    # Top band (p.0004-like): 4 panels across 3 columns, height=580 (0.29h), max_pw=0.40
    p1 = PanelBox(x=580, y=40, w=390, h=580, ordinal=1)
    p2 = PanelBox(x=200, y=40, w=350, h=280, ordinal=2)
    p3 = PanelBox(x=200, y=335, w=350, h=285, ordinal=3)
    p4 = PanelBox(x=30, y=40, w=150, h=580, ordinal=4)
    # Lower panel overlaps top band by 15px from pad_and_clamp
    p5 = PanelBox(x=30, y=605, w=940, h=600, ordinal=5)

    merger = PanelMerger(PanelMergeConfig.for_profile(DeviceProfile.from_name("KPW6")))
    merged = merger.merge([p1, p2, p3, p4, p5], page_width=page_w, page_height=page_h)

    assert len(merged) == 2
    assert merged[0] == PanelBox(x=30, y=40, w=940, h=580, ordinal=1)
    assert merged[1] == PanelBox(x=30, y=605, w=940, h=600, ordinal=2)


def test_merger_merges_stacked_mini_subcolumn_beside_hero_panel() -> None:
    """Mixed band keeps dominant hero panel solo and merges stacked mini sub-column."""
    page_w, page_h = 1000, 2000
    top = PanelBox(x=30, y=40, w=940, h=860, ordinal=1)
    # Bottom band (p.0002 / p.0017-like): large right panel + 3 stacked mini left panels
    hero = PanelBox(x=460, y=940, w=510, h=960, ordinal=2)  # 48%h > 0.37 -> Rule 1 skips
    m1 = PanelBox(x=40, y=940, w=380, h=260, ordinal=3)
    m2 = PanelBox(x=40, y=1220, w=380, h=320, ordinal=4)
    m3 = PanelBox(x=40, y=1560, w=380, h=340, ordinal=5)

    merger = PanelMerger(PanelMergeConfig.for_profile(DeviceProfile.from_name("KPW6")))
    merged = merger.merge([top, hero, m1, m2, m3], page_width=page_w, page_height=page_h)

    assert len(merged) == 3
    assert merged[0] == PanelBox(x=30, y=40, w=940, h=860, ordinal=1)
    assert merged[1] == PanelBox(x=460, y=940, w=510, h=960, ordinal=2)
    assert merged[2] == PanelBox(x=40, y=940, w=380, h=960, ordinal=3)


def test_merger_calibrates_tall_and_medium_bands_across_device_tiers() -> None:
    """Profile tiers control tall 3-col (0.45h) and 5-panel medium (0.36h) tier decisions."""
    page_w, page_h = 1000, 2000

    # Tall 3-col band (0.45h, p.0009 top): split on STANDARD (KPW6), merged on LARGE (KS)
    tall_cols = [
        PanelBox(x=690, y=40, w=270, h=900, ordinal=1),
        PanelBox(x=440, y=40, w=230, h=890, ordinal=2),
        PanelBox(x=30, y=40, w=390, h=900, ordinal=3),
    ]
    std_merger = PanelMerger(PanelMergeConfig.for_profile(DeviceProfile.from_name("KPW6")))
    large_merger = PanelMerger(PanelMergeConfig.for_profile(DeviceProfile.from_name("KS")))
    assert len(std_merger.merge(tall_cols, page_width=page_w, page_height=page_h)) == 3
    large_out = large_merger.merge(tall_cols, page_width=page_w, page_height=page_h)
    assert large_out == [PanelBox(x=30, y=40, w=930, h=900, ordinal=1)]

    # 5-panel 0.36h band (p.0006 top): whole tier on STANDARD, sub-columns on COMPACT
    med_five = [
        PanelBox(x=610, y=40, w=350, h=340, ordinal=1),
        PanelBox(x=610, y=400, w=350, h=360, ordinal=2),
        PanelBox(x=420, y=40, w=170, h=720, ordinal=3),
        PanelBox(x=40, y=40, w=360, h=340, ordinal=4),
        PanelBox(x=40, y=400, w=360, h=360, ordinal=5),
    ]
    assert len(std_merger.merge(med_five, page_width=page_w, page_height=page_h)) == 1

    compact_profile = DeviceProfile.from_name("CUSTOM", custom_size=(1072, 1448))
    compact_merger = PanelMerger(PanelMergeConfig.for_profile(compact_profile))
    compact_out = compact_merger.merge(med_five, page_width=page_w, page_height=page_h)
    assert compact_out == [
        PanelBox(x=610, y=40, w=350, h=720, ordinal=1),
        PanelBox(x=420, y=40, w=170, h=720, ordinal=2),
        PanelBox(x=40, y=40, w=360, h=720, ordinal=3),
    ]


def test_merger_rejects_non_contiguous_reading_order() -> None:
    """Interleaved reading order must never merge non-contiguous panels."""
    page_w, page_h = 1000, 2000
    t_right = PanelBox(x=680, y=40, w=280, h=400, ordinal=1)
    bottom = PanelBox(x=40, y=500, w=920, h=1400, ordinal=2)
    t_mid = PanelBox(x=360, y=40, w=280, h=400, ordinal=3)
    t_left = PanelBox(x=40, y=40, w=280, h=400, ordinal=4)

    # Also test non-contiguous sub-column in a mixed band
    hero = PanelBox(x=500, y=500, w=460, h=900, ordinal=1)
    sub_top = PanelBox(x=40, y=500, w=360, h=300, ordinal=2)
    interleaved_other = PanelBox(x=40, y=1500, w=920, h=400, ordinal=3)
    sub_bot = PanelBox(x=40, y=840, w=360, h=300, ordinal=4)

    merger = PanelMerger()
    out_tier = merger.merge(
        [t_right, bottom, t_mid, t_left], page_width=page_w, page_height=page_h
    )
    assert len(out_tier) == 4
    assert [p.x for p in out_tier] == [680, 40, 360, 40]

    out_sub = merger.merge(
        [hero, sub_top, interleaved_other, sub_bot], page_width=page_w, page_height=page_h
    )
    assert len(out_sub) == 4


def test_merger_rejects_merge_that_overlaps_external_panel() -> None:
    """Sub-column whose union overlaps an external panel beyond threshold stays unmerged."""
    page_w, page_h = 1000, 2000
    hero = PanelBox(x=460, y=100, w=500, h=800, ordinal=1)
    p_ext = PanelBox(x=430, y=320, w=25, h=240, ordinal=2)
    col_top = PanelBox(x=10, y=100, w=200, h=200, ordinal=3)
    col_bot = PanelBox(x=10, y=700, w=440, h=200, ordinal=4)

    merger = PanelMerger(PanelMergeConfig.for_profile(DeviceProfile.from_name("KPW6")))
    out = merger.merge(
        [hero, p_ext, col_top, col_bot],
        page_width=page_w,
        page_height=page_h,
    )
    assert len(out) == 4

    permissive = PanelMerger(
        PanelMergeConfig.for_profile(DeviceProfile.from_name("KPW6")).model_copy(
            update={"max_external_overlap_ratio": 0.90}
        )
    )
    assert (
        len(permissive.merge([hero, p_ext, col_top, col_bot], page_width=page_w, page_height=page_h))
        == 3
    )


def test_merger_protects_four_koma_and_non_stacked_columns() -> None:
    """4-koma single-column stack and non-stacked side-by-side column panels stay separate."""
    page_w, page_h = 1000, 2000
    four_koma = [
        PanelBox(x=300, y=40, w=360, h=420, ordinal=1),
        PanelBox(x=305, y=490, w=355, h=420, ordinal=2),
        PanelBox(x=298, y=940, w=362, h=420, ordinal=3),
        PanelBox(x=302, y=1390, w=358, h=420, ordinal=4),
    ]
    merger = PanelMerger()
    out_4k = merger.merge(four_koma, page_width=page_w, page_height=page_h)
    assert len(out_4k) == 4

    hero = PanelBox(x=520, y=100, w=440, h=1000, ordinal=1)
    left_r = PanelBox(x=230, y=100, w=200, h=240, ordinal=2)
    left_l = PanelBox(x=40, y=100, w=200, h=240, ordinal=3)
    out_non_stacked = merger.merge(
        [hero, left_r, left_l],
        page_width=page_w,
        page_height=page_h,
    )
    assert len(out_non_stacked) == 3


def test_merger_handles_duplicate_panel_instances_safely() -> None:
    """Duplicate PanelBox object references in input do not collide during indexing."""
    dup = PanelBox(x=680, y=40, w=240, h=300, ordinal=1)
    p3 = PanelBox(x=400, y=40, w=240, h=300, ordinal=3)
    p4 = PanelBox(x=120, y=40, w=240, h=300, ordinal=4)

    merged = PanelMerger().merge([dup, dup, p3, p4], page_width=1000, page_height=2000)
    assert merged == [PanelBox(x=120, y=40, w=800, h=300, ordinal=1)]


def test_merger_spread_halves_and_center_straddling_guard() -> None:
    """Unsplit spreads evaluate halves against page_width/2 and keep straddling panels solo."""
    page_w, page_h = 2000, 1500  # eff_w = 1000 per half
    r1 = PanelBox(x=1660, y=40, w=300, h=360, ordinal=1)
    r2 = PanelBox(x=1340, y=40, w=300, h=360, ordinal=2)
    r3 = PanelBox(x=1030, y=40, w=290, h=360, ordinal=3)
    straddle = PanelBox(x=700, y=440, w=600, h=400, ordinal=4)
    l1 = PanelBox(x=520, y=880, w=440, h=560, ordinal=5)
    l2 = PanelBox(x=40, y=880, w=440, h=560, ordinal=6)

    merged = PanelMerger().merge(
        [r1, r2, r3, straddle, l1, l2],
        page_width=page_w,
        page_height=page_h,
        is_spread=True,
    )
    assert merged == [
        PanelBox(x=1030, y=40, w=930, h=360, ordinal=1),
        PanelBox(x=700, y=440, w=600, h=400, ordinal=2),
        PanelBox(x=520, y=880, w=440, h=560, ordinal=3),
        PanelBox(x=40, y=880, w=440, h=560, ordinal=4),
    ]


def test_merger_noop_guards() -> None:
    """Disabled config, <2 panels, and invalid page dims preserve panels and normalize ordinals."""
    p1 = PanelBox(x=680, y=40, w=240, h=300, ordinal=0)
    p2 = PanelBox(x=400, y=40, w=240, h=300, ordinal=0)
    p3 = PanelBox(x=120, y=40, w=240, h=300, ordinal=0)

    disabled = PanelMerger(PanelMergeConfig(enabled=False))
    out_disabled = disabled.merge([p1, p2, p3], page_width=1000, page_height=2000)
    assert [p.ordinal for p in out_disabled] == [1, 2, 3]
    assert len(out_disabled) == 3

    enabled = PanelMerger()
    assert enabled.merge([], page_width=1000, page_height=2000) == []
    assert enabled.merge([p1], page_width=1000, page_height=2000) == [
        p1.model_copy(update={"ordinal": 1})
    ]
    assert enabled.merge([p1, p2, p3], page_width=0, page_height=2000) == [
        p1.model_copy(update={"ordinal": 1}),
        p2.model_copy(update={"ordinal": 2}),
        p3.model_copy(update={"ordinal": 3}),
    ]
    assert enabled.merge([p1, p2, p3], page_width=1000, page_height=-10) == [
        p1.model_copy(update={"ordinal": 1}),
        p2.model_copy(update={"ordinal": 2}),
        p3.model_copy(update={"ordinal": 3}),
    ]


def test_merger_high_res_padded_subcolumn_beside_hero_panel() -> None:
    """padding=0.04 on 1908x2544 page splits wide hero (>24px pad) cleanly and avoids false overlap."""
    page_w, page_h = 1908, 2544
    # Hero x=[620..1820] (w=1200 -> 4% pad=48px > 24px cap!), beside 2 stacked minis x=[130..580] (w=450)
    # 40px vertical gutter between sub-column (580) and hero (620): padded edges overlap (598 vs 572)!
    raw_hero = (620, 1200, 1200, 1200)  # 47%h -> Rule 1 skips
    raw_sub1 = (130, 1200, 450, 560)
    raw_sub2 = (130, 1800, 450, 600)

    hero = pad_and_clamp(raw_hero, page_w, page_h, padding=0.04, bleed_margin=2).model_copy(
        update={"ordinal": 1}
    )
    sub1 = pad_and_clamp(raw_sub1, page_w, page_h, padding=0.04, bleed_margin=2).model_copy(
        update={"ordinal": 2}
    )
    sub2 = pad_and_clamp(raw_sub2, page_w, page_h, padding=0.04, bleed_margin=2).model_copy(
        update={"ordinal": 3}
    )

    merger = PanelMerger(PanelMergeConfig.for_profile(DeviceProfile.from_name("KPW6")))
    merged = merger.merge([hero, sub1, sub2], page_width=page_w, page_height=page_h, padding=0.04)

    assert len(merged) == 2
    assert merged[0] == hero
    assert merged[1] == PanelBox(
        x=min(sub1.x, sub2.x),
        y=sub1.y,
        w=max(sub1.x + sub1.w, sub2.x + sub2.w) - min(sub1.x, sub2.x),
        h=(sub2.y + sub2.h) - sub1.y,
        ordinal=2,
    )


def test_merger_high_res_padded_compact_tier_stays_below_height_threshold() -> None:
    """Compact 0.35h 5-panel tier padded with 0.04 (0.378h padded) still merges on STANDARD."""
    page_w, page_h = 1908, 2544
    # Raw tier height = 890px (0.350h <= 0.37), but padded with 0.04 becomes ~0.378h > 0.37!
    # Unpadding recovers 0.350h so Rule 1 merges the entire 5-panel tier and preserves padded outer union.
    raw_boxes = [
        (1180, 100, 640, 430),
        (1180, 560, 640, 430),
        (800, 100, 340, 890),
        (100, 100, 660, 430),
        (100, 560, 660, 430),
    ]
    padded = [
        pad_and_clamp(rect, page_w, page_h, padding=0.04, bleed_margin=2).model_copy(
            update={"ordinal": idx}
        )
        for idx, rect in enumerate(raw_boxes, start=1)
    ]

    merger = PanelMerger(PanelMergeConfig.for_profile(DeviceProfile.from_name("KPW6")))
    merged = merger.merge(padded, page_width=page_w, page_height=page_h, padding=0.04)

    assert len(merged) == 1
    assert merged[0] == PanelBox(
        x=min(p.x for p in padded),
        y=min(p.y for p in padded),
        w=max(p.x + p.w for p in padded) - min(p.x for p in padded),
        h=max(p.y + p.h for p in padded) - min(p.y for p in padded),
        ordinal=1,
    )
