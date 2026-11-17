"""Unit tests for Layer 2 OnnxYoloBackend and models_hub (no real .onnx or network)."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from manga2kindle.cv.backends.base import DetectionResult, PanelDetectorBackend
from manga2kindle.cv.backends.onnx_yolo import (
    PANEL_CLASS_ID,
    OnnxYoloBackend,
    decode_yolo_output,
    letterbox_image,
    select_ort_providers,
)
from manga2kindle.cv.models_hub import (
    DEFAULT_ONNX_MODEL_KEY,
    ModelSpec,
    fetch_model,
    is_model_cached,
    list_registered_models,
    resolve_model_spec,
)
from manga2kindle.models import DetectorConfig, DetectorKind, DevicePreference


def _write_blank_page(path: Path, width: int, height: int) -> Path:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def _to_letterbox_coords(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float,
    pad: tuple[int, int],
) -> tuple[float, float, float, float]:
    pad_x, pad_y = pad
    return (
        x1 * scale + pad_x,
        y1 * scale + pad_y,
        x2 * scale + pad_x,
        y2 * scale + pad_y,
    )


class TestLetterboxImage:
    def test_non_square_kpw5_page_round_trip(self) -> None:
        orig_w, orig_h = 1264, 1680
        image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        tensor, scale, pad = letterbox_image(image, target_size=640)

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
        assert scale == pytest.approx(640.0 / orig_h)
        pad_x, pad_y = pad
        expected_new_w = int(round(orig_w * scale))
        assert pad_x == (640 - expected_new_w) // 2
        assert pad_y == 0


class TestDecodeYoloOutput:
    def test_decode_raw_yolov8_v11_format_filters_and_clamps(self) -> None:
        orig_w, orig_h = 1264, 1680
        image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _, scale, pad = letterbox_image(image, target_size=640)

        # Box 0: valid frame (class 2), score 0.90
        b0 = _to_letterbox_coords(100, 120, 600, 720, scale, pad)
        # Box 1: small valid frame (< 3% area: 120x120 = 14400 / 2123520 = 0.68%), score 0.80
        b1 = _to_letterbox_coords(700, 100, 820, 220, scale, pad)
        # Box 2: non-frame (face, class 1) with high score -> must be ignored
        b2 = _to_letterbox_coords(200, 200, 400, 400, scale, pad)
        # Box 3: below score_threshold (0.20 < 0.25) -> must be ignored
        b3 = _to_letterbox_coords(100, 900, 500, 1200, scale, pad)
        # Box 4: extends outside page bounds -> must be clamped to [0, orig_w] x [0, orig_h]
        b4 = _to_letterbox_coords(-50, -30, 1350, 1720, scale, pad)

        raw = np.zeros((1, 8, 5), dtype=np.float32)
        entries = [
            (b0, PANEL_CLASS_ID, 0.90),
            (b1, PANEL_CLASS_ID, 0.80),
            (b2, 1, 0.95),
            (b3, PANEL_CLASS_ID, 0.20),
            (b4, PANEL_CLASS_ID, 0.70),
        ]
        for idx, ((x1, y1, x2, y2), cls_id, score) in enumerate(entries):
            raw[0, 0, idx] = (x1 + x2) / 2.0
            raw[0, 1, idx] = (y1 + y2) / 2.0
            raw[0, 2, idx] = x2 - x1
            raw[0, 3, idx] = y2 - y1
            raw[0, 4 + cls_id, idx] = score

        rects, scores = decode_yolo_output(
            raw,
            orig_w=orig_w,
            orig_h=orig_h,
            scale=scale,
            pad=pad,
            score_threshold=0.25,
            panel_class_id=PANEL_CLASS_ID,
        )

        assert len(rects) == 3
        assert len(scores) == 3
        assert scores == pytest.approx([0.90, 0.80, 0.70], abs=1e-5)

        # Box 0 un-letterboxed accurately
        assert rects[0][0] == pytest.approx(100, abs=3)
        assert rects[0][1] == pytest.approx(120, abs=3)
        assert rects[0][2] == pytest.approx(500, abs=4)
        assert rects[0][3] == pytest.approx(600, abs=4)

        # Small valid panel (< 3% area) preserved
        assert rects[1][2] * rects[1][3] < 0.03 * orig_w * orig_h
        assert rects[1][2] == pytest.approx(120, abs=3)

        # Box 4 clamped to source page dimensions
        x, y, w, h = rects[2]
        assert x == 0
        assert y == 0
        assert x + w == orig_w
        assert y + h == orig_h

    def test_decode_nms_exported_n6_format(self) -> None:
        orig_w, orig_h = 1264, 1680
        image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _, scale, pad = letterbox_image(image, target_size=640)

        b0 = _to_letterbox_coords(50, 60, 550, 760, scale, pad)
        b1 = _to_letterbox_coords(600, 800, 1200, 1600, scale, pad)
        b2 = _to_letterbox_coords(100, 100, 300, 300, scale, pad)

        raw = np.array(
            [
                [
                    [*b0, 0.88, float(PANEL_CLASS_ID)],
                    [*b1, 0.76, float(PANEL_CLASS_ID)],
                    [*b2, 0.92, 3.0],  # text class -> ignored
                ]
            ],
            dtype=np.float32,
        )

        rects, scores = decode_yolo_output(
            raw,
            orig_w=orig_w,
            orig_h=orig_h,
            scale=scale,
            pad=pad,
            score_threshold=0.25,
            panel_class_id=PANEL_CLASS_ID,
        )

        assert len(rects) == 2
        assert scores == pytest.approx([0.88, 0.76], abs=1e-5)
        assert rects[0][0] == pytest.approx(50, abs=3)
        assert rects[1][0] == pytest.approx(600, abs=3)


class TestSelectOrtProviders:
    def test_auto_priority_order(self) -> None:
        available = [
            "CPUExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
        ]
        selected = select_ort_providers(DevicePreference.AUTO, available)
        assert selected == [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_auto_falls_back_through_priority(self) -> None:
        selected = select_ort_providers(
            DevicePreference.AUTO,
            ["CPUExecutionProvider", "DmlExecutionProvider"],
        )
        assert selected == ["DmlExecutionProvider", "CPUExecutionProvider"]

    def test_mps_maps_to_coreml(self) -> None:
        selected = select_ort_providers(
            DevicePreference.MPS,
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        assert selected[0] == "CoreMLExecutionProvider"

    def test_explicit_unavailable_provider_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
            select_ort_providers(DevicePreference.CUDA, ["CPUExecutionProvider"])

        with pytest.raises(RuntimeError, match="CoreMLExecutionProvider"):
            select_ort_providers(DevicePreference.MPS, ["CPUExecutionProvider"])

        with pytest.raises(RuntimeError, match="CPUExecutionProvider"):
            select_ort_providers(DevicePreference.CPU, [])


class TestOnnxYoloBackend:
    def test_backend_prunes_nested_boxes_aligns_scores_and_preserves_small_panels(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        orig_w, orig_h = 1264, 1680
        page_path = _write_blank_page(tmp_path / "page_001.png", orig_w, orig_h)
        model_path = tmp_path / "fake.onnx"
        model_path.write_bytes(b"fake-onnx-model")

        _, scale, pad = letterbox_image(
            np.zeros((orig_h, orig_w, 3), dtype=np.uint8),
            target_size=640,
        )

        # Outer panel + nested inner panel (>85% contained) + small valid panel (<3% area)
        outer = _to_letterbox_coords(50, 50, 650, 850, scale, pad)
        nested = _to_letterbox_coords(80, 80, 480, 680, scale, pad)
        small = _to_letterbox_coords(800, 100, 920, 220, scale, pad)

        raw = np.zeros((1, 8, 3), dtype=np.float32)
        for idx, (box, score) in enumerate([(outer, 0.85), (nested, 0.95), (small, 0.72)]):
            x1, y1, x2, y2 = box
            raw[0, 0, idx] = (x1 + x2) / 2.0
            raw[0, 1, idx] = (y1 + y2) / 2.0
            raw[0, 2, idx] = x2 - x1
            raw[0, 3, idx] = y2 - y1
            raw[0, 4 + PANEL_CLASS_ID, idx] = score

        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [SimpleNamespace(name="images", shape=[1, 3, 640, 640])]
        fake_session.run.return_value = [raw]

        fake_ort = SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=MagicMock(return_value=fake_session),
        )
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

        config = DetectorConfig(
            kind=DetectorKind.ONNX_YOLO,
            model_path=model_path,
            device=DevicePreference.CPU,
        )
        backend = OnnxYoloBackend(config)
        assert isinstance(backend, PanelDetectorBackend)

        result = backend.detect(page_path)
        assert isinstance(result, DetectionResult)
        assert result.ordered is False
        assert result.backend == DetectorKind.ONNX_YOLO
        assert result.page_width == orig_w
        assert result.page_height == orig_h

        # Nested inner box was pruned despite higher score; outer + small kept
        assert len(result.panels) == 2
        assert result.scores is not None
        assert len(result.scores) == 2
        assert result.scores == pytest.approx([0.85, 0.72], abs=1e-5)

        # Small panel (< 3% area) was NOT clipped out
        areas = [p.w * p.h for p in result.panels]
        assert min(areas) < 0.03 * orig_w * orig_h

        # detect_batch preserves order
        batch_results = backend.detect_batch([page_path, page_path])
        assert len(batch_results) == 2

    def test_missing_local_model_raises_without_downloading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_ort = SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

        config = DetectorConfig(
            kind=DetectorKind.ONNX_YOLO,
            model_path=tmp_path / "does_not_exist.onnx",
        )
        with pytest.raises(FileNotFoundError):
            OnnxYoloBackend(config)


class TestModelsHub:
    def test_registered_models_include_license_and_pinned_revision(self) -> None:
        models = list_registered_models()
        assert len(models) >= 1
        assert all(isinstance(m, ModelSpec) for m in models)
        default_spec = next(m for m in models if m.key == DEFAULT_ONNX_MODEL_KEY)
        assert default_spec.repo_id == "deepghs/manga109_yolo"
        assert len(default_spec.revision) == 40
        assert default_spec.panel_class_id == PANEL_CLASS_ID
        assert "Manga109" in default_spec.license_notice
        assert default_spec.source_url.startswith("https://huggingface.co/")

    def test_resolve_model_spec_distinguishes_paths_and_repo_ids(self, tmp_path: Path) -> None:
        onnx_file = tmp_path / "custom.onnx"
        assert resolve_model_spec(str(onnx_file)) == (onnx_file, None)
        assert resolve_model_spec("./relative/model.onnx") == (Path("./relative/model.onnx"), None)
        assert resolve_model_spec("C:\\weights\\detector.onnx") == (
            Path("C:\\weights\\detector.onnx"),
            None,
        )

        path_part, key_part = resolve_model_spec("deepghs/manga109_yolo")
        assert path_part is None
        assert key_part == DEFAULT_ONNX_MODEL_KEY

        path_part2, key_part2 = resolve_model_spec(DEFAULT_ONNX_MODEL_KEY)
        assert path_part2 is None
        assert key_part2 == DEFAULT_ONNX_MODEL_KEY

    def test_fetch_model_verifies_sha256_and_caches(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = b"dummy-onnx-weights-payload"
        good_sha = hashlib.sha256(payload).hexdigest()

        spec = ModelSpec(
            key="test-model",
            repo_id="deepghs/manga109_yolo",
            revision="6b5730a9f1f34581b87468f2c609e3269a9faccf",
            filename="v2023.12.07_s_yv11/model.onnx",
            sha256=good_sha,
            panel_class_id=2,
            source_url="https://example.com/model.onnx",
            license_notice="Manga109-s attribution",
        )
        monkeypatch.setattr(
            "manga2kindle.cv.models_hub._MODEL_REGISTRY",
            {"test-model": spec},
        )

        calls = 0

        def fake_urlopen(url: str, timeout: float = 30.0) -> io.BytesIO:
            nonlocal calls
            calls += 1
            return io.BytesIO(payload)

        monkeypatch.setattr("manga2kindle.cv.models_hub.urllib.request.urlopen", fake_urlopen)

        assert is_model_cached("test-model", cache_dir=tmp_path) is False
        cached_path = fetch_model("test-model", cache_dir=tmp_path)
        assert cached_path.is_file()
        assert cached_path.read_bytes() == payload
        assert is_model_cached("test-model", cache_dir=tmp_path) is True
        assert calls == 1

        # Second fetch uses cache without network call
        cached_path_2 = fetch_model("test-model", cache_dir=tmp_path)
        assert cached_path_2 == cached_path
        assert calls == 1

    def test_fetch_model_sha256_mismatch_raises_and_cleans_up(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spec = ModelSpec(
            key="bad-sha-model",
            repo_id="deepghs/manga109_yolo",
            revision="6b5730a9f1f34581b87468f2c609e3269a9faccf",
            filename="v2023.12.07_s_yv11/model.onnx",
            sha256="0" * 64,
            panel_class_id=2,
            source_url="https://example.com/model.onnx",
            license_notice="Manga109-s attribution",
        )
        monkeypatch.setattr(
            "manga2kindle.cv.models_hub._MODEL_REGISTRY",
            {"bad-sha-model": spec},
        )
        monkeypatch.setattr(
            "manga2kindle.cv.models_hub.urllib.request.urlopen",
            lambda url, timeout=30.0: io.BytesIO(b"corrupted-bytes"),
        )

        with pytest.raises(ValueError, match="SHA-256"):
            fetch_model("bad-sha-model", cache_dir=tmp_path)

        assert is_model_cached("bad-sha-model", cache_dir=tmp_path) is False
