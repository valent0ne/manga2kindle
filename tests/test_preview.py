"""Tests for visual debug panel overlay rendering (PanelPreviewRenderer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from manga2kindle.cv import read_image_bgr, write_image
from manga2kindle.cv.preview import BACKEND_COLORS_BGR, PanelPreviewRenderer
from manga2kindle.models import DetectorKind, PanelBox


def _create_white_canvas(path: Path, width: int = 600, height: int = 800) -> Path:
    """Create and write a blank white BGR image."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    return write_image(path, canvas)


def test_render_two_panels_debug_overlay(tmp_path: Path) -> None:
    """Render debug overlay on a 600x800 image with 2 panels, verifying green boxes and red arrow."""
    input_path = _create_white_canvas(tmp_path / "page_01.png", width=600, height=800)
    output_path = tmp_path / "debug_page_01.png"

    panels = [
        PanelBox(x=50, y=50, w=500, h=300, ordinal=1),
        PanelBox(x=50, y=400, w=500, h=300, ordinal=2),
    ]

    renderer = PanelPreviewRenderer()
    result_path = renderer.render(input_path, panels, output_path)

    assert result_path == output_path
    assert output_path.exists()

    rendered = read_image_bgr(output_path)
    assert rendered.shape == (800, 600, 3)

    # Green (0, 200, 0) box/badge pixels present
    has_green = np.any((rendered[:, :, 0] == 0) & (rendered[:, :, 1] == 200) & (rendered[:, :, 2] == 0))
    assert has_green, "Expected green (0, 200, 0) bounding box/badge pixels in overlay"

    # Red (0, 0, 255) arrow pixels present
    has_red = np.any((rendered[:, :, 0] == 0) & (rendered[:, :, 1] == 0) & (rendered[:, :, 2] == 255))
    assert has_red, "Expected red (0, 0, 255) arrow pixels connecting panel 1 to panel 2"


def test_render_single_panel_has_no_arrow(tmp_path: Path) -> None:
    """A single panel overlay renders the box and badge but no inter-panel arrows."""
    input_path = _create_white_canvas(tmp_path / "page_single.png", width=600, height=800)
    output_path = tmp_path / "debug_single.png"

    panels = [PanelBox(x=80, y=80, w=440, h=640, ordinal=1)]

    renderer = PanelPreviewRenderer()
    renderer.render(input_path, panels, output_path)

    assert output_path.exists()
    rendered = read_image_bgr(output_path)
    assert rendered.shape == (800, 600, 3)

    has_green = np.any((rendered[:, :, 0] == 0) & (rendered[:, :, 1] == 200) & (rendered[:, :, 2] == 0))
    assert has_green, "Expected green box/badge for the single panel"

    has_red = np.any((rendered[:, :, 0] == 0) & (rendered[:, :, 1] == 0) & (rendered[:, :, 2] == 255))
    assert not has_red, "Single panel should not produce any red connecting arrows"


def test_render_empty_panels(tmp_path: Path) -> None:
    """Empty panels list writes an unmodified copy without boxes or arrows."""
    input_path = _create_white_canvas(tmp_path / "page_empty.png", width=600, height=800)
    output_path = tmp_path / "debug_empty.png"

    renderer = PanelPreviewRenderer()
    renderer.render(input_path, [], output_path)

    assert output_path.exists()
    rendered = read_image_bgr(output_path)
    original = read_image_bgr(input_path)
    assert rendered.shape == (800, 600, 3)
    assert np.array_equal(rendered, original)


def test_render_unicode_paths(tmp_path: Path) -> None:
    """Renderer cleanly reads and writes images residing on Unicode paths."""
    unicode_dir = tmp_path / "葬送のフリーレン_café"
    input_path = _create_white_canvas(unicode_dir / "第01話_raw.png", width=600, height=800)
    output_path = unicode_dir / "プレビュー_preview" / "第01話_overlay.png"

    panels = [
        PanelBox(x=40, y=40, w=520, h=320, ordinal=1),
        PanelBox(x=40, y=420, w=520, h=320, ordinal=2),
    ]

    renderer = PanelPreviewRenderer()
    result_path = renderer.render(input_path, panels, output_path)

    assert result_path == output_path
    assert output_path.is_file()

    rendered = read_image_bgr(output_path)
    assert rendered.shape == (800, 600, 3)
    has_green = np.any((rendered[:, :, 0] == 0) & (rendered[:, :, 1] == 200) & (rendered[:, :, 2] == 0))
    has_red = np.any((rendered[:, :, 0] == 0) & (rendered[:, :, 1] == 0) & (rendered[:, :, 2] == 255))
    assert has_green and has_red


def test_render_panels_without_ordinals(tmp_path: Path) -> None:
    """Panels with ordinal=0 default to 1-based index numbering in badge."""
    input_path = _create_white_canvas(tmp_path / "page_no_ord.png", width=600, height=800)
    output_path = tmp_path / "debug_no_ord.png"

    panels = [
        PanelBox(x=50, y=50, w=200, h=200, ordinal=0),
        PanelBox(x=50, y=300, w=200, h=200, ordinal=0),
    ]

    renderer = PanelPreviewRenderer()
    renderer.render(input_path, panels, output_path)

    assert output_path.exists()
    rendered = read_image_bgr(output_path)
    has_green = np.any((rendered[:, :, 0] == 0) & (rendered[:, :, 1] == 200) & (rendered[:, :, 2] == 0))
    has_red = np.any((rendered[:, :, 0] == 0) & (rendered[:, :, 1] == 0) & (rendered[:, :, 2] == 255))
    assert has_green and has_red


def test_render_comparison_tiles_horizontally_with_headers_and_colors(tmp_path: Path) -> None:
    """render_comparison tiles backends side by side with distinct colors and header banners."""
    input_path = _create_white_canvas(tmp_path / "page_cmp.png", width=600, height=800)
    output_path = tmp_path / "comparison.png"

    results = {
        DetectorKind.CONTOUR: [
            PanelBox(x=50, y=50, w=500, h=300, ordinal=1),
            PanelBox(x=50, y=400, w=500, h=300, ordinal=2),
        ],
        DetectorKind.ONNX_YOLO: [
            PanelBox(x=60, y=60, w=480, h=640, ordinal=1),
        ],
    }

    renderer = PanelPreviewRenderer()
    result_path = renderer.render_comparison(input_path, results, output_path)

    assert result_path == output_path
    assert output_path.is_file()

    rendered = read_image_bgr(output_path)
    assert rendered.shape[1] == 600 * 2
    assert rendered.shape[0] > 800

    left_tile = rendered[:, :600]
    right_tile = rendered[:, 600:]

    contour_bgr = BACKEND_COLORS_BGR[DetectorKind.CONTOUR]
    onnx_bgr = BACKEND_COLORS_BGR[DetectorKind.ONNX_YOLO]
    assert contour_bgr != onnx_bgr

    assert np.any(
        (left_tile[:, :, 0] == contour_bgr[0])
        & (left_tile[:, :, 1] == contour_bgr[1])
        & (left_tile[:, :, 2] == contour_bgr[2])
    )
    assert np.any(
        (right_tile[:, :, 0] == onnx_bgr[0])
        & (right_tile[:, :, 1] == onnx_bgr[1])
        & (right_tile[:, :, 2] == onnx_bgr[2])
    )
    # Header strip at top should be rendered (non-white pixels)
    banner_strip = rendered[: rendered.shape[0] - 800, :]
    assert np.any(banner_strip < 255)


def test_render_comparison_handles_zero_panels_for_one_backend(tmp_path: Path) -> None:
    """render_comparison renders all tiles even if one backend detects zero panels."""
    input_path = _create_white_canvas(tmp_path / "page_zero.png", width=600, height=800)
    output_path = tmp_path / "comparison_zero.png"

    results = {
        DetectorKind.CONTOUR: [PanelBox(x=40, y=40, w=520, h=720, ordinal=1)],
        DetectorKind.MAGI: [],
    }

    renderer = PanelPreviewRenderer()
    renderer.render_comparison(input_path, results, output_path)

    rendered = read_image_bgr(output_path)
    assert rendered.shape[1] == 600 * 2


def test_render_comparison_empty_results_raises(tmp_path: Path) -> None:
    """render_comparison raises ValueError when given an empty results mapping."""
    input_path = _create_white_canvas(tmp_path / "page_empty_cmp.png", width=600, height=800)
    renderer = PanelPreviewRenderer()

    with pytest.raises(ValueError, match="at least one"):
        renderer.render_comparison(input_path, {}, tmp_path / "empty_cmp.png")


def test_render_comparison_banner_pluralization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """render_comparison uses '1 panel' singular and '0 panels' / '2 panels' plural."""
    import cv2

    input_path = _create_white_canvas(tmp_path / "page_plural.png", width=600, height=800)
    output_path = tmp_path / "comparison_plural.png"

    labels: list[str] = []
    real_put_text = cv2.putText

    def spy_put_text(img, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_8):
        labels.append(text)
        return real_put_text(img, text, org, fontFace, fontScale, color, thickness, lineType)

    monkeypatch.setattr("manga2kindle.cv.preview.cv2.putText", spy_put_text)

    results = {
        DetectorKind.CONTOUR: [PanelBox(x=40, y=40, w=520, h=720, ordinal=1)],
        DetectorKind.ONNX_YOLO: [
            PanelBox(x=40, y=40, w=250, h=720, ordinal=1),
            PanelBox(x=310, y=40, w=250, h=720, ordinal=2),
        ],
        DetectorKind.MAGI: [],
    }

    renderer = PanelPreviewRenderer()
    renderer.render_comparison(input_path, results, output_path)

    assert "CONTOUR (1 panel)" in labels
    assert "ONNX (2 panels)" in labels
    assert "MAGI (0 panels)" in labels
