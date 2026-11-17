"""Tests for shared bounding-box geometry, containment suppression, and clamping."""

from __future__ import annotations

import pytest

from manga2kindle.cv.boxes import (
    compute_containment,
    compute_iou,
    nms_and_prune,
    nms_and_prune_indices,
    pad_and_clamp,
)
from manga2kindle.models import PanelBox


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
