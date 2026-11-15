"""Tests for OpenCV Unicode I/O, PanelDetector, and PanelSorter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from manga2kindle.cv import PanelDetector, PanelSorter, read_image_bgr, write_image
from manga2kindle.models import PanelBox, ReadingDirection


def _draw_panel_with_artwork(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    border_thickness: int = 4,
    gap_bottom: int = 0,
) -> None:
    """Draw a black-framed manga panel with internal ink lines onto a white canvas."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), border_thickness)
    if gap_bottom > 0:
        cx = x + w // 2
        half = gap_bottom // 2
        cv2.rectangle(
            canvas,
            (cx - half, y + h - border_thickness - 2),
            (cx + half, y + h + border_thickness + 2),
            (255, 255, 255),
            -1,
        )
    # Add internal artwork lines so ink density sits cleanly within [0.02, 0.98]
    step = max(12, min(w, h) // 10)
    for offset in range(step, min(w, h) - step, step):
        cv2.line(
            canvas,
            (x + 10, y + offset),
            (x + w - 10, y + offset),
            (0, 0, 0),
            2,
        )
    cv2.circle(
        canvas,
        (x + w // 2, y + h // 2),
        max(8, min(w, h) // 6),
        (0, 0, 0),
        -1,
    )


def test_unicode_path_read_and_write(tmp_path: Path) -> None:
    """read_image_bgr and write_image handle Japanese and accented Unicode paths."""
    unicode_dir = tmp_path / "葬送のフリーレン_café"
    image_path = unicode_dir / "第01話_résumé.png"

    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (100, 90), (15, 30, 45), -1)

    returned = write_image(image_path, image)
    assert returned == image_path
    assert image_path.is_file()

    loaded = read_image_bgr(image_path)
    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)

    with pytest.raises(FileNotFoundError):
        read_image_bgr(unicode_dir / "missing_画像.png")

    corrupt_path = unicode_dir / "corrupt_画像.png"
    corrupt_path.write_bytes(b"not-a-valid-png")
    with pytest.raises(ValueError):
        read_image_bgr(corrupt_path)


def test_panel_detector_detects_three_panels_filters_noise_and_applies_nms_and_padding(
    tmp_path: Path,
) -> None:
    """Detects 3 panels, filters tiny noise/bubbles, prunes nested duplicates, and pads once."""
    page_w, page_h = 1000, 1400
    canvas = np.full((page_h, page_w, 3), 255, dtype=np.uint8)

    # Panel 1 (top-right): broken outer frame + closed inner nested frame -> NMS prunes inner
    _draw_panel_with_artwork(canvas, 520, 60, 400, 380, border_thickness=4, gap_bottom=20)
    cv2.rectangle(canvas, (545, 85), (895, 415), (0, 0, 0), 4)

    # Panel 2 (top-left)
    _draw_panel_with_artwork(canvas, 80, 60, 380, 380, border_thickness=4)

    # Panel 3 (bottom full-width)
    _draw_panel_with_artwork(canvas, 80, 500, 840, 800, border_thickness=4)

    # Tiny noise / small speech bubbles outside panels (< 3% of page area)
    cv2.circle(canvas, (490, 250), 12, (0, 0, 0), 2)
    cv2.rectangle(canvas, (460, 460), (520, 485), (0, 0, 0), 2)
    # Ultra-thin horizontal divider line (aspect ratio > 15)
    cv2.rectangle(canvas, (80, 465), (920, 472), (0, 0, 0), -1)

    image_path = tmp_path / "three_panels.png"
    write_image(image_path, canvas)

    detector = PanelDetector(min_panel_area=0.03, padding=0.02, iou_threshold=0.4)
    boxes = detector.detect(image_path)

    assert len(boxes) == 3

    # Verify each detected box matches one of the 3 panels with 1x padding (~2%)
    # and stays clamped within page bounds.
    expected_panels = [
        (520, 60, 400, 380),
        (80, 60, 380, 380),
        (80, 500, 840, 800),
    ]
    for ex, ey, ew, eh in expected_panels:
        matches = [
            b
            for b in boxes
            if abs((b.x + b.w / 2) - (ex + ew / 2)) < 25
            and abs((b.y + b.h / 2) - (ey + eh / 2)) < 25
        ]
        assert len(matches) == 1, f"Expected single match for panel {(ex, ey, ew, eh)}, got {boxes}"
        box = matches[0]
        # Applied padding once (~4% total width/height growth, definitely not 2x padding)
        assert ew <= box.w <= int(ew * 1.07)
        assert eh <= box.h <= int(eh * 1.07)
        assert 0 <= box.x
        assert 0 <= box.y
        assert box.x + box.w <= page_w
        assert box.y + box.h <= page_h


def test_panel_detector_nms_suppresses_overlapping_box() -> None:
    """NMS suppresses overlapping smaller boxes above iou_threshold while keeping distinct ones."""
    outer = (100, 100, 400, 400)  # area 160,000
    inner = (130, 130, 340, 340)  # area 115,600 -> IoU = 0.7225 > 0.4
    separate = (550, 100, 300, 400)  # IoU = 0.0

    kept = PanelDetector.non_max_suppression([inner, outer, separate], iou_threshold=0.4)
    assert outer in kept
    assert separate in kept
    assert inner not in kept
    assert len(kept) == 2


def test_panel_detector_blank_splash_page_falls_back_to_full_page(tmp_path: Path) -> None:
    """Blank page or page with only sub-threshold speckles falls back to full-page PanelBox."""
    page_w, page_h = 800, 1200

    blank = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    blank_path = tmp_path / "blank.png"
    write_image(blank_path, blank)

    detector = PanelDetector()
    assert detector.detect(blank_path) == [PanelBox(x=0, y=0, w=page_w, h=page_h)]

    speckles = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    cv2.circle(speckles, (200, 200), 15, (0, 0, 0), -1)
    cv2.rectangle(speckles, (500, 600), (540, 640), (0, 0, 0), 2)
    speckles_path = tmp_path / "speckles.png"
    write_image(speckles_path, speckles)

    assert detector.detect(speckles_path) == [PanelBox(x=0, y=0, w=page_w, h=page_h)]


def test_panel_detector_filters_solid_black_and_empty_white_artifacts(tmp_path: Path) -> None:
    """Regions outside [min_ink_ratio, max_ink_ratio] (0.02..0.98) are filtered out."""
    page_w, page_h = 1000, 1200
    canvas = np.full((page_h, page_w, 3), 255, dtype=np.uint8)

    # Valid panel (top)
    _draw_panel_with_artwork(canvas, 100, 80, 800, 320, border_thickness=4)

    # Solid-black artifact block (ink_ratio == 1.0 > 0.98)
    cv2.rectangle(canvas, (100, 450), (450, 780), (0, 0, 0), -1)

    # Nearly empty white box with 1px border (ink_ratio ~ 0.01 < 0.02)
    cv2.rectangle(canvas, (520, 450), (900, 850), (0, 0, 0), 1)

    mixed_path = tmp_path / "ink_filter.png"
    write_image(mixed_path, canvas)

    detector = PanelDetector(min_ink_ratio=0.02, max_ink_ratio=0.98)
    boxes = detector.detect(mixed_path)

    assert len(boxes) == 1
    assert boxes[0].y < 400

    # Pure solid-black page also falls back to full-page box after artifact filtering
    solid_black = np.zeros((page_h, page_w, 3), dtype=np.uint8)
    black_path = tmp_path / "solid_black.png"
    write_image(black_path, solid_black)
    assert detector.detect(black_path) == [PanelBox(x=0, y=0, w=page_w, h=page_h)]


def test_panel_detector_bleed_to_edge_clamped_with_safety_margin(tmp_path: Path) -> None:
    """Bleed-to-edge panels touching borders are clamped with a 2px inner safety margin."""
    page_w, page_h = 1000, 1200
    canvas = np.full((page_h, page_w, 3), 255, dtype=np.uint8)

    # Top-left bleed panel touching x=0 and y=0
    _draw_panel_with_artwork(canvas, 0, 0, 450, 500, border_thickness=4)

    # Bottom-right bleed panel touching x+w=page_w and y+h=page_h
    _draw_panel_with_artwork(canvas, 500, 600, 500, 600, border_thickness=4)

    bleed_path = tmp_path / "bleed.png"
    write_image(bleed_path, canvas)

    detector = PanelDetector(padding=0.02)
    boxes = detector.detect(bleed_path)

    assert len(boxes) == 2
    for box in boxes:
        assert box.x >= 2
        assert box.y >= 2
        assert box.x + box.w <= page_w - 2
        assert box.y + box.h <= page_h - 2


def test_panel_sorter_rtl_and_ltr_ordering() -> None:
    """PanelSorter orders rows top-to-bottom and within rows RTL (x desc) or LTR (x asc)."""
    top_left = PanelBox(x=50, y=60, w=350, h=300)
    top_right = PanelBox(x=460, y=50, w=350, h=310)  # 10px y offset within row_tolerance_px=50
    bottom_left = PanelBox(x=50, y=420, w=350, h=320)
    bottom_right = PanelBox(x=460, y=410, w=350, h=330)

    shuffled = [bottom_left, top_left, bottom_right, top_right]

    rtl_sorted = PanelSorter(direction=ReadingDirection.RTL, row_tolerance_px=50).sort(shuffled)
    assert [(b.x, b.y, b.ordinal) for b in rtl_sorted] == [
        (460, 50, 1),
        (50, 60, 2),
        (460, 410, 3),
        (50, 420, 4),
    ]

    ltr_sorted = PanelSorter(direction=ReadingDirection.LTR, row_tolerance_px=50).sort(shuffled)
    assert [(b.x, b.y, b.ordinal) for b in ltr_sorted] == [
        (50, 60, 1),
        (460, 50, 2),
        (50, 420, 3),
        (460, 410, 4),
    ]

    # Empty & single-element edge cases
    assert PanelSorter().sort([]) == []
    single = PanelSorter().sort([top_left])
    assert len(single) == 1
    assert single[0].ordinal == 1


def test_panel_sorter_four_koma_vertical_stack() -> None:
    """4-koma single-column vertical stack preserves y0 < y1 < y2 < y3 even under RTL."""
    p0 = PanelBox(x=100, y=20, w=300, h=36)
    p1 = PanelBox(x=105, y=62, w=295, h=36)  # y1 - y0 = 42 <= row_tolerance_px=50, higher x
    p2 = PanelBox(x=98, y=104, w=302, h=36)
    p3 = PanelBox(x=103, y=146, w=297, h=36)

    shuffled = [p3, p1, p0, p2]
    sorted_panels = PanelSorter(direction=ReadingDirection.RTL, row_tolerance_px=50).sort(shuffled)

    assert [b.y for b in sorted_panels] == [20, 62, 104, 146]
    assert [b.ordinal for b in sorted_panels] == [1, 2, 3, 4]


def test_panel_sorter_overlapping_staggered_row_and_spread_halves() -> None:
    """Overlapping staggered panels stay in the same row; spreads sort right half before left half in RTL."""
    # Staggered vertical top edges (y=50 vs y=120 -> delta 70 > 50) yet high overlap -> single row RTL
    staggered_right = PanelBox(x=520, y=50, w=300, h=300)
    staggered_left = PanelBox(x=80, y=120, w=300, h=220)
    row_sorted = PanelSorter(direction=ReadingDirection.RTL, row_tolerance_px=50).sort(
        [staggered_left, staggered_right]
    )
    assert [(b.x, b.ordinal) for b in row_sorted] == [(520, 1), (80, 2)]

    # Spread 2000x1000: top-left (x=100,y=50), bottom-right (x=1200,y=500), top-right (x=1200,y=50)
    spread_panels = [
        PanelBox(x=100, y=50, w=700, h=400),
        PanelBox(x=1200, y=500, w=700, h=400),
        PanelBox(x=1200, y=50, w=700, h=400),
    ]
    rtl_spread = PanelSorter(direction=ReadingDirection.RTL).sort(
        spread_panels, page_width=2000, is_spread=True
    )
    assert [(b.x, b.y, b.ordinal) for b in rtl_spread] == [
        (1200, 50, 1),
        (1200, 500, 2),
        (100, 50, 3),
    ]
    ltr_spread = PanelSorter(direction=ReadingDirection.LTR).sort(
        spread_panels, page_width=2000, is_spread=True
    )
    assert [(b.x, b.y, b.ordinal) for b in ltr_spread] == [
        (100, 50, 1),
        (1200, 50, 2),
        (1200, 500, 3),
    ]


def test_panel_detector_pad_and_clamp_edge_safety_margin() -> None:
    """_pad_and_clamp clamps x1/y1 so even extreme right/bottom contours stay inside bleed margin."""
    detector = PanelDetector(padding=0.02)
    box = detector._pad_and_clamp((999, 1199, 1, 1), page_w=1000, page_h=1200)
    assert box.x + box.w <= 998
    assert box.y + box.h <= 1198
