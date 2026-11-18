"""Tests for PanelDetectorBackend protocol, domain types, and ContourBackend."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from manga2kindle.cv import (
    ContourBackend,
    DetectionResult,
    PanelDetector,
    PanelDetectorBackend,
    write_image,
)
from manga2kindle.models import (
    ConversionConfig,
    DetectorConfig,
    DetectorKind,
    DevicePreference,
    DeviceProfile,
    PanelBox,
)


def _draw_panel_with_artwork(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    border_thickness: int = 4,
) -> None:
    """Draw a black-framed manga panel with internal ink lines onto a white canvas."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), border_thickness)
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


def test_detector_enums_and_config_defaults() -> None:
    """DetectorKind, DevicePreference, and DetectorConfig expose expected defaults."""
    assert DetectorKind.AUTO.value == "auto"
    assert DetectorKind.CONTOUR.value == "contour"
    assert DetectorKind.ONNX_YOLO.value == "onnx"
    assert DetectorKind.MAGI.value == "magi"

    assert DevicePreference.AUTO.value == "auto"
    assert DevicePreference.CPU.value == "cpu"
    assert DevicePreference.CUDA.value == "cuda"
    assert DevicePreference.MPS.value == "mps"

    cfg = DetectorConfig()
    assert cfg.kind == DetectorKind.AUTO
    assert cfg.model_path is None
    assert cfg.model_id is None
    assert cfg.device == DevicePreference.AUTO
    assert cfg.score_threshold == 0.25
    assert cfg.iou_threshold == 0.4
    assert cfg.containment_threshold == 0.85
    assert cfg.padding == 0.02
    assert cfg.min_panel_area == 0.03
    assert cfg.allow_remote_code is False


def test_conversion_config_detector_sync(tmp_path: Path) -> None:
    """ConversionConfig syncs top-level padding/min_panel_area with detector when defaulted."""
    profile = DeviceProfile.from_name("KPW5")

    legacy_cfg = ConversionConfig(
        input_path=tmp_path / "in.cbz",
        output_path=tmp_path / "out.kfx",
        profile=profile,
        padding=0.05,
        min_panel_area=0.04,
    )
    assert legacy_cfg.detector.padding == 0.05
    assert legacy_cfg.detector.min_panel_area == 0.04

    explicit_detector_cfg = ConversionConfig(
        input_path=tmp_path / "in.cbz",
        output_path=tmp_path / "out.kfx",
        profile=profile,
        detector=DetectorConfig(padding=0.08, min_panel_area=0.06),
    )
    assert explicit_detector_cfg.detector.padding == 0.08
    assert explicit_detector_cfg.detector.min_panel_area == 0.06
    assert explicit_detector_cfg.padding == 0.08
    assert explicit_detector_cfg.min_panel_area == 0.06


def test_contour_backend_detects_panels_and_returns_empty_on_blank_page(
    tmp_path: Path,
) -> None:
    """ContourBackend returns DetectionResult with panels=[] on blank pages."""
    backend = ContourBackend(min_panel_area=0.03, padding=0.02, iou_threshold=0.4)
    assert isinstance(backend, PanelDetectorBackend)
    assert backend.kind == DetectorKind.CONTOUR

    page_w, page_h = 1000, 1400
    canvas = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    _draw_panel_with_artwork(canvas, 520, 60, 400, 380)
    _draw_panel_with_artwork(canvas, 80, 60, 380, 380)
    _draw_panel_with_artwork(canvas, 80, 500, 840, 800)

    page_path = tmp_path / "three_panels.png"
    write_image(page_path, canvas)

    result = backend.detect(page_path)
    assert isinstance(result, DetectionResult)
    assert result.ordered is False
    assert result.backend == DetectorKind.CONTOUR
    assert result.page_width == page_w
    assert result.page_height == page_h
    assert len(result.panels) == 3
    assert all(isinstance(p, PanelBox) for p in result.panels)

    blank_path = tmp_path / "blank.png"
    write_image(blank_path, np.full((800, 600, 3), 255, dtype=np.uint8))

    blank_result = backend.detect(blank_path)
    assert blank_result.ordered is False
    assert blank_result.backend == DetectorKind.CONTOUR
    assert blank_result.page_width == 600
    assert blank_result.page_height == 800
    assert blank_result.panels == []


def test_contour_backend_detect_batch_preserves_order(tmp_path: Path) -> None:
    """detect_batch supports workers=1 and workers=2 while preserving page order."""
    backend = ContourBackend()

    p1 = tmp_path / "p1.png"
    c1 = np.full((1000, 800, 3), 255, dtype=np.uint8)
    _draw_panel_with_artwork(c1, 80, 80, 640, 840)
    write_image(p1, c1)

    p2 = tmp_path / "p2.png"
    write_image(p2, np.full((900, 700, 3), 255, dtype=np.uint8))

    seq_results = backend.detect_batch([p1, p2], workers=1)
    par_results = backend.detect_batch([p1, p2], workers=2)

    assert len(seq_results) == 2
    assert len(par_results) == 2
    assert seq_results[0].page_width == 800
    assert len(seq_results[0].panels) == 1
    assert seq_results[1].page_width == 700
    assert seq_results[1].panels == []
    assert par_results == seq_results


def test_panel_detector_wrapper_preserves_full_page_fallback(tmp_path: Path) -> None:
    """PanelDetector delegates to ContourBackend and falls back to full-page on blank."""
    blank_path = tmp_path / "blank.png"
    write_image(blank_path, np.full((800, 600, 3), 255, dtype=np.uint8))

    detector = PanelDetector()
    boxes = detector.detect(blank_path)
    assert boxes == [PanelBox(x=0, y=0, w=600, h=800)]


def test_contour_backend_merges_overlapping_contours_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlapping contours below iou_threshold merge when merge_bridged_panels=True."""
    page_path = tmp_path / "overlap.png"
    write_image(page_path, np.full((1000, 1000, 3), 255, dtype=np.uint8))

    # Two overlapping boxes:
    # (100, 100, 300, 300) area=90_000 and (250, 100, 200, 200) area=40_000
    # intersection=150x200=30_000 -> IoU=30_000/100_000=0.30 <= 0.40, containment=30_000/40_000=0.75 >= 0.25
    overlapping_candidates = [(100, 100, 300, 300), (250, 100, 200, 200)]

    backend_merged = ContourBackend(
        padding=0.0,
        merge_bridged_panels=True,
        panel_overlap_threshold=0.25,
        max_merged_area_ratio=0.65,
        min_fill_ratio=0.55,
    )
    monkeypatch.setattr(backend_merged, "_filter_contours", lambda *a, **kw: overlapping_candidates)
    res_merged = backend_merged.detect(page_path)
    assert len(res_merged.panels) == 1
    assert res_merged.panels[0] == PanelBox(x=100, y=100, w=350, h=300)

    backend_unmerged = ContourBackend(padding=0.0, merge_bridged_panels=False)
    monkeypatch.setattr(backend_unmerged, "_filter_contours", lambda *a, **kw: overlapping_candidates)
    res_unmerged = backend_unmerged.detect(page_path)
    assert len(res_unmerged.panels) == 2
