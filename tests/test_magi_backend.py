"""Unit tests for Layer 3 MagiBackend and Magi models_hub support (mocked torch/transformers)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from manga2kindle.cv.backends.base import DetectionResult, PanelDetectorBackend
from manga2kindle.cv.backends.magi import (
    MagiBackend,
    select_torch_device,
    to_rgb_array,
)
from manga2kindle.cv.models_hub import (
    DEFAULT_MAGI_MODEL_KEY,
    MAGIV2_REVISION,
    MAGIV3_REVISION,
    fetch_magi_model,
    get_cached_magi_dir,
    get_magi_model_spec,
    is_magi_cached,
    list_registered_magi_models,
    resolve_magi_model_spec,
)
from manga2kindle.models import DetectorConfig, DetectorKind, DevicePreference


def _write_bgr_page(path: Path, width: int, height: int, bgr: tuple[int, int, int] = (10, 20, 30)) -> Path:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = bgr
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def _make_fake_magi_dir(root: Path, name: str = "magiv3") -> Path:
    model_dir = root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text('{"model_type": "florence2"}', encoding="utf-8")
    return model_dir


def _build_mock_torch_and_transformers(
    *,
    cuda_available: bool = True,
    mps_available: bool = False,
    panels_per_call: list[list[list[float]]] | None = None,
) -> tuple[SimpleNamespace, SimpleNamespace, MagicMock, MagicMock, list[str]]:
    events: list[str] = []

    @contextmanager
    def _inference_mode():
        events.append("inference_mode_enter")
        try:
            yield
        finally:
            events.append("inference_mode_exit")

    fake_torch = SimpleNamespace(
        float16="float16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
        inference_mode=_inference_mode,
    )

    call_idx = {"i": 0}
    captured_images: list[np.ndarray] = []

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    def _predict(images: list[np.ndarray], processor: object = None) -> list[dict[str, object]]:
        events.append("predict")
        captured_images.extend(images)
        idx = call_idx["i"]
        call_idx["i"] += 1
        panels = panels_per_call[idx] if panels_per_call and idx < len(panels_per_call) else []
        return [{"panels": panels, "texts": [], "characters": [], "tails": []}]

    fake_model.predict_detections_and_associations.side_effect = _predict
    fake_model._captured_images = captured_images  # type: ignore[attr-defined]

    fake_processor = MagicMock()
    mock_auto_causal_lm = MagicMock()
    mock_auto_causal_lm.from_pretrained.return_value = fake_model
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = fake_model
    mock_auto_processor = MagicMock()
    mock_auto_processor.from_pretrained.return_value = fake_processor

    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=mock_auto_causal_lm,
        AutoModel=mock_auto_model,
        AutoProcessor=mock_auto_processor,
    )
    return fake_torch, fake_transformers, mock_auto_causal_lm, mock_auto_processor, events


class TestRemoteCodeGate:
    def test_raises_actionable_runtime_error_when_allow_remote_code_is_false(self, tmp_path: Path) -> None:
        model_dir = _make_fake_magi_dir(tmp_path)
        config = DetectorConfig(
            kind=DetectorKind.MAGI,
            model_path=model_dir,
            allow_remote_code=False,
        )

        with pytest.raises(RuntimeError, match="--allow-remote-code"):
            MagiBackend(config)

    def test_raises_actionable_runtime_error_when_torch_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model_dir = _make_fake_magi_dir(tmp_path)
        monkeypatch.setitem(sys.modules, "torch", None)
        config = DetectorConfig(
            kind=DetectorKind.MAGI,
            model_path=model_dir,
            allow_remote_code=True,
        )

        with pytest.raises(RuntimeError, match=r"manga2kindle\[magi\]"):
            MagiBackend(config)


class TestDeviceAndColorConversion:
    def test_select_torch_device_auto_prefers_cuda_then_mps_then_cpu(self) -> None:
        t_cuda = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
        assert select_torch_device(DevicePreference.AUTO, t_cuda) == "cuda"

        t_mps = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
        assert select_torch_device(DevicePreference.AUTO, t_mps) == "mps"

        t_cpu = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        assert select_torch_device(DevicePreference.AUTO, t_cpu) == "cpu"

    def test_select_torch_device_explicit_unavailable_raises(self) -> None:
        t_cpu = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        with pytest.raises(RuntimeError, match="cuda"):
            select_torch_device(DevicePreference.CUDA, t_cpu)
        with pytest.raises(RuntimeError, match="mps"):
            select_torch_device(DevicePreference.MPS, t_cpu)

    def test_to_rgb_array_converts_bgr_and_grayscale(self) -> None:
        bgr = np.zeros((4, 5, 3), dtype=np.uint8)
        bgr[:, :] = (10, 20, 30)  # B=10, G=20, R=30
        rgb_from_bgr = to_rgb_array(bgr)
        assert rgb_from_bgr.shape == (4, 5, 3)
        assert tuple(int(c) for c in rgb_from_bgr[0, 0]) == (30, 20, 10)

        gray = np.full((4, 5), 128, dtype=np.uint8)
        rgb_from_gray = to_rgb_array(gray)
        assert rgb_from_gray.shape == (4, 5, 3)
        assert tuple(int(c) for c in rgb_from_gray[0, 0]) == (128, 128, 128)


class TestMagiBackendInference:
    def test_detect_preserves_native_reading_order_and_prunes_containment(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model_dir = _make_fake_magi_dir(tmp_path)
        page_path = _write_bgr_page(tmp_path / "page1.png", 1000, 1400, bgr=(10, 20, 30))

        # Deliberately non-XY-cut order (bottom-left first, then top-right) to verify
        # Magi's native reading order is preserved (preserve_order=True).
        # Box 1 is nested inside Box 0 (>85% containment) and must be pruned.
        mock_panels = [
            [50.0, 750.0, 450.0, 1300.0],   # Panel 1 (bottom-left)
            [100.0, 800.0, 400.0, 1200.0],  # Nested inside Panel 1 -> pruned
            [520.0, 40.0, 950.0, 600.0],    # Panel 2 (top-right)
        ]
        fake_torch, fake_transformers, mock_lm, mock_proc, events = _build_mock_torch_and_transformers(
            cuda_available=True,
            panels_per_call=[mock_panels],
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        config = DetectorConfig(
            kind=DetectorKind.MAGI,
            model_path=model_dir,
            device=DevicePreference.AUTO,
            padding=0.0,
            allow_remote_code=True,
        )
        backend = MagiBackend(config)
        assert isinstance(backend, PanelDetectorBackend)

        mock_lm.from_pretrained.assert_called_once()
        lm_kwargs = mock_lm.from_pretrained.call_args.kwargs
        assert lm_kwargs["trust_remote_code"] is True
        assert lm_kwargs["local_files_only"] is True

        mock_proc.from_pretrained.assert_called_once()
        proc_kwargs = mock_proc.from_pretrained.call_args.kwargs
        assert proc_kwargs["trust_remote_code"] is True
        assert proc_kwargs["local_files_only"] is True

        result = backend.detect(page_path)

        assert isinstance(result, DetectionResult)
        assert result.ordered is True
        assert result.backend == DetectorKind.MAGI
        assert result.page_width == 1000
        assert result.page_height == 1400
        assert result.scores is None

        # Verify RGB conversion occurred before passing image to Magi
        passed_img = mock_lm.from_pretrained.return_value._captured_images[0]
        assert tuple(int(c) for c in passed_img[0, 0]) == (30, 20, 10)

        # Verify inference_mode wrapped predict
        assert events == ["inference_mode_enter", "predict", "inference_mode_exit"]

        # Verify native order preserved & 1-based ordinals assigned
        assert len(result.panels) == 2
        assert result.panels[0].ordinal == 1
        assert result.panels[1].ordinal == 2
        assert result.panels[0].y > result.panels[1].y  # bottom-left stayed before top-right

    def test_detect_returns_empty_panels_when_magi_returns_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model_dir = _make_fake_magi_dir(tmp_path)
        page_path = _write_bgr_page(tmp_path / "blank.png", 800, 1200)

        fake_torch, fake_transformers, _, _, _ = _build_mock_torch_and_transformers(
            cuda_available=False,
            panels_per_call=[[]],
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        config = DetectorConfig(
            kind=DetectorKind.MAGI,
            model_path=model_dir,
            device=DevicePreference.CPU,
            allow_remote_code=True,
        )
        backend = MagiBackend(config)
        result = backend.detect(page_path)

        assert result.panels == []
        assert result.ordered is True
        assert result.backend == DetectorKind.MAGI
        assert result.page_width == 800
        assert result.page_height == 1200

    def test_detect_batch_preserves_page_order(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model_dir = _make_fake_magi_dir(tmp_path)
        p1 = _write_bgr_page(tmp_path / "p1.png", 800, 1000)
        p2 = _write_bgr_page(tmp_path / "p2.png", 800, 1000)

        fake_torch, fake_transformers, _, _, _ = _build_mock_torch_and_transformers(
            cuda_available=False,
            panels_per_call=[[[50.0, 50.0, 700.0, 900.0]], []],
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        backend = MagiBackend(
            DetectorConfig(
                kind=DetectorKind.MAGI,
                model_path=model_dir,
                device=DevicePreference.CPU,
                allow_remote_code=True,
            )
        )
        results = backend.detect_batch([p1, p2])
        assert len(results) == 2
        assert len(results[0].panels) == 1
        assert results[1].panels == []


class TestMagiModelsHub:
    def test_registered_magi_models_have_pinned_revisions_and_license(self) -> None:
        specs = list_registered_magi_models()
        keys = {s.key for s in specs}
        assert {"magiv3", "magiv2"}.issubset(keys)

        v3 = get_magi_model_spec(DEFAULT_MAGI_MODEL_KEY)
        assert v3.repo_id == "ragavsachdeva/magiv3"
        assert v3.revision == MAGIV3_REVISION
        assert len(v3.revision) == 40
        assert "non-commercial" in v3.license_notice.lower()

        v2 = get_magi_model_spec("ragavsachdeva/magiv2")
        assert v2.repo_id == "ragavsachdeva/magiv2"
        assert v2.revision == MAGIV2_REVISION
        assert len(v2.revision) == 40

        assert resolve_magi_model_spec("magi") == (None, "magiv3")

    def test_fetch_magi_model_uses_snapshot_download_with_pinned_revision(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, object]] = []

        def _fake_snapshot_download(**kwargs: object) -> str:
            calls.append(kwargs)
            local_dir = Path(str(kwargs["local_dir"]))
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "config.json").write_text("{}", encoding="utf-8")
            return str(local_dir)

        fake_hub = SimpleNamespace(snapshot_download=_fake_snapshot_download)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        dest = fetch_magi_model("magiv3", cache_dir=tmp_path)
        assert dest == get_cached_magi_dir("magiv3", cache_dir=tmp_path)
        assert is_magi_cached("magiv3", cache_dir=tmp_path) is True
        assert len(calls) == 1
        assert calls[0]["repo_id"] == "ragavsachdeva/magiv3"
        assert calls[0]["revision"] == MAGIV3_REVISION

        # Subsequent call without force=True skips snapshot_download
        fetch_magi_model("magiv3", cache_dir=tmp_path)
        assert len(calls) == 1

    def test_resolve_magi_batch_size_by_vram_and_override(self) -> None:
        from manga2kindle.cv.backends.magi import resolve_magi_batch_size

        fake_cuda_16g = SimpleNamespace(
            get_device_properties=lambda idx: SimpleNamespace(total_memory=16 * (1024**3))
        )
        fake_cuda_8g = SimpleNamespace(
            get_device_properties=lambda idx: SimpleNamespace(total_memory=8 * (1024**3))
        )
        assert resolve_magi_batch_size("cuda", SimpleNamespace(cuda=fake_cuda_16g), None) == 8
        assert resolve_magi_batch_size("cuda", SimpleNamespace(cuda=fake_cuda_8g), None) == 4
        assert resolve_magi_batch_size("mps", SimpleNamespace(), None) == 2
        assert resolve_magi_batch_size("cpu", SimpleNamespace(), None) == 1
        assert resolve_magi_batch_size("cuda", SimpleNamespace(cuda=fake_cuda_16g), 12) == 12
