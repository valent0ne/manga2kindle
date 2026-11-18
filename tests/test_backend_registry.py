"""Tests for BackendRegistry, AUTO fallback chain, pipeline reuse, models CLI, and doctor."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from manga2kindle.cli import app
from manga2kindle.cv.backends import (
    BackendRegistry,
    detect_with_fallback,
    resolve_auto_chain,
)
from manga2kindle.cv.backends.base import DetectionResult
from manga2kindle.cv.sorter import PanelSorter
from manga2kindle.models import (
    ConversionConfig,
    DetectorConfig,
    DetectorKind,
    DevicePreference,
    DeviceProfile,
    PanelBox,
    ReadingDirection,
)
from manga2kindle.pipeline import ConversionPipeline
from manga2kindle.platform import PlatformDiscovery

runner = CliRunner()


def _make_blank_png(path: Path, width: int = 800, height: int = 1200) -> Path:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    img.save(path, format="PNG")
    return path


def _make_cbz(cbz_path: Path, num_pages: int = 2) -> Path:
    cbz_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cbz_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx in range(num_pages):
            img_path = cbz_path.parent / f"_tmp_page_{idx + 1}.png"
            img = Image.new("RGB", (800, 1200), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([420, 60, 740, 560], fill=(220, 220, 220), outline=(0, 0, 0), width=8)
            draw.rectangle([60, 620, 740, 1140], fill=(200, 200, 200), outline=(0, 0, 0), width=8)
            img.save(img_path, format="PNG")
            zf.write(img_path, arcname=f"page_{idx + 1:03d}.png")
            img_path.unlink()
    return cbz_path


class _StubBackend:
    def __init__(
        self,
        kind: DetectorKind,
        *,
        results: list[DetectionResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.kind = kind
        self._results = list(results or [])
        self._error = error
        self.calls: list[Path] = []

    def detect(self, image_path: Path) -> DetectionResult:
        self.calls.append(image_path)
        if self._error is not None:
            raise self._error
        if self._results:
            return self._results.pop(0)
        return DetectionResult(
            panels=[],
            ordered=False,
            backend=self.kind,
            page_width=800,
            page_height=1200,
        )

    def detect_batch(self, image_paths):
        return [self.detect(p) for p in image_paths]


class TestAutoChainAndFallback:
    """Verify AUTO chain resolution, per-session disablement, and zero-panel fallback."""

    def test_auto_chain_orders_magi_onnx_contour_and_logs_active(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        magi_stub = _StubBackend(DetectorKind.MAGI)
        onnx_stub = _StubBackend(DetectorKind.ONNX_YOLO)
        contour_stub = _StubBackend(DetectorKind.CONTOUR)

        monkeypatch.setattr(
            "manga2kindle.cv.backends._probe_magi_readiness",
            lambda _cfg: (True, "ready"),
        )
        monkeypatch.setattr("manga2kindle.cv.backends.MagiBackend", lambda _cfg: magi_stub)
        monkeypatch.setattr("manga2kindle.cv.backends.OnnxYoloBackend", lambda _cfg: onnx_stub)
        monkeypatch.setattr("manga2kindle.cv.backends.ContourBackend", lambda **_kw: contour_stub)

        cfg = DetectorConfig(kind=DetectorKind.AUTO, allow_remote_code=True)
        with caplog.at_level(logging.INFO):
            chain = resolve_auto_chain(cfg)

        assert [b.kind for b in chain] == [
            DetectorKind.MAGI,
            DetectorKind.ONNX_YOLO,
            DetectorKind.CONTOUR,
        ]
        assert any("magi" in rec.message.lower() for rec in caplog.records)

    def test_auto_skips_magi_with_single_warning_when_allow_remote_code_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        magi_ctor = MagicMock()
        onnx_stub = _StubBackend(DetectorKind.ONNX_YOLO)
        contour_stub = _StubBackend(DetectorKind.CONTOUR)

        monkeypatch.setattr(
            "manga2kindle.cv.backends._probe_magi_readiness",
            lambda _cfg: (True, "ready"),
        )
        monkeypatch.setattr("manga2kindle.cv.backends.MagiBackend", magi_ctor)
        monkeypatch.setattr("manga2kindle.cv.backends.OnnxYoloBackend", lambda _cfg: onnx_stub)
        monkeypatch.setattr("manga2kindle.cv.backends.ContourBackend", lambda **_kw: contour_stub)

        cfg = DetectorConfig(kind=DetectorKind.AUTO, allow_remote_code=False)
        with caplog.at_level(logging.WARNING):
            chain = resolve_auto_chain(cfg)

        magi_ctor.assert_not_called()
        assert [b.kind for b in chain] == [DetectorKind.ONNX_YOLO, DetectorKind.CONTOUR]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "allow_remote_code" in warnings[0].message

    def test_load_failed_backend_disabled_once_per_session_and_zero_panel_falls_through(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        page1 = _make_blank_png(tmp_path / "p1.png", 800, 1200)
        page2 = _make_blank_png(tmp_path / "p2.png", 800, 1200)

        magi_broken = _StubBackend(DetectorKind.MAGI, error=RuntimeError("CUDA OOM"))
        onnx_flaky = _StubBackend(
            DetectorKind.ONNX_YOLO,
            results=[
                # Page 1: 0 panels -> fall through to contour
                DetectionResult(
                    panels=[],
                    ordered=False,
                    backend=DetectorKind.ONNX_YOLO,
                    page_width=800,
                    page_height=1200,
                ),
                # Page 2: 0 panels -> fall through to contour
                DetectionResult(
                    panels=[],
                    ordered=False,
                    backend=DetectorKind.ONNX_YOLO,
                    page_width=800,
                    page_height=1200,
                ),
            ],
        )
        contour_ok_then_empty = _StubBackend(
            DetectorKind.CONTOUR,
            results=[
                # Page 1: contour finds 1 panel
                DetectionResult(
                    panels=[PanelBox(x=10, y=10, w=300, h=400)],
                    ordered=False,
                    backend=DetectorKind.CONTOUR,
                    page_width=800,
                    page_height=1200,
                ),
                # Page 2: contour also finds 0 panels -> full-page fallback
                DetectionResult(
                    panels=[],
                    ordered=False,
                    backend=DetectorKind.CONTOUR,
                    page_width=800,
                    page_height=1200,
                ),
            ],
        )

        disabled: set[DetectorKind] = set()
        with caplog.at_level(logging.WARNING):
            res1 = detect_with_fallback(
                page1,
                [magi_broken, onnx_flaky, contour_ok_then_empty],
                disabled_backends=disabled,
            )
            res2 = detect_with_fallback(
                page2,
                [magi_broken, onnx_flaky, contour_ok_then_empty],
                disabled_backends=disabled,
            )

        # Magi failed on page 1, so it was disabled and NOT called on page 2
        assert len(magi_broken.calls) == 1
        assert DetectorKind.MAGI in disabled
        # ONNX returned 0 panels on page 1, so it was still tried on page 2
        assert len(onnx_flaky.calls) == 2

        assert res1.backend == DetectorKind.CONTOUR
        assert len(res1.panels) == 1
        assert res1.panels[0].w == 300

        # Page 2: all backends returned 0 panels -> full-page [PanelBox(0, 0, 800, 1200, ordinal=1)]
        assert len(res2.panels) == 1
        assert res2.panels[0] == PanelBox(x=0, y=0, w=800, h=1200, ordinal=1)

    def test_explicit_onnx_raises_on_missing_weights_instead_of_falling_back(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        missing_model = tmp_path / "nonexistent.onnx"
        monkeypatch.setattr(
            "manga2kindle.cv.backends.onnx_yolo._import_onnxruntime",
            lambda: MagicMock(get_available_providers=lambda: ["CPUExecutionProvider"]),
        )
        cfg = DetectorConfig(kind=DetectorKind.ONNX_YOLO, model_path=missing_model)
        with pytest.raises(FileNotFoundError):
            resolve_auto_chain(cfg)


class TestOrderedResultAndSorterIntegration:
    """Verify PanelSorter is skipped when DetectionResult.ordered is True."""

    def test_ordered_true_skips_panel_sorter_and_preserves_ordinals(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        page = _make_blank_png(tmp_path / "p1.png", 800, 1200)
        ordered_panels = [
            PanelBox(x=50, y=600, w=300, h=400, ordinal=1),
            PanelBox(x=450, y=50, w=300, h=400, ordinal=2),
        ]
        magi_stub = _StubBackend(
            DetectorKind.MAGI,
            results=[
                DetectionResult(
                    panels=ordered_panels,
                    ordered=True,
                    backend=DetectorKind.MAGI,
                    page_width=800,
                    page_height=1200,
                )
            ],
        )
        registry = BackendRegistry(
            DetectorConfig(kind=DetectorKind.MAGI, allow_remote_code=True),
            backends=[magi_stub],
        )
        sorter = PanelSorter(direction=ReadingDirection.RTL)
        sort_spy = MagicMock(side_effect=AssertionError("PanelSorter.sort should not be called"))
        monkeypatch.setattr(sorter, "sort", sort_spy)

        result_panels = registry.detect_and_sort(page, sorter, page_width=800, is_spread=False)
        assert [p.ordinal for p in result_panels] == [1, 2]
        assert result_panels == ordered_panels

    def test_ordered_false_runs_panel_sorter(self, tmp_path: Path) -> None:
        page = _make_blank_png(tmp_path / "p1.png", 800, 1200)
        unordered_panels = [
            PanelBox(x=50, y=50, w=300, h=400, ordinal=0),
            PanelBox(x=450, y=50, w=300, h=400, ordinal=0),
        ]
        onnx_stub = _StubBackend(
            DetectorKind.ONNX_YOLO,
            results=[
                DetectionResult(
                    panels=unordered_panels,
                    ordered=False,
                    backend=DetectorKind.ONNX_YOLO,
                    page_width=800,
                    page_height=1200,
                )
            ],
        )
        registry = BackendRegistry(
            DetectorConfig(kind=DetectorKind.ONNX_YOLO),
            backends=[onnx_stub],
        )
        sorter = PanelSorter(direction=ReadingDirection.RTL)
        result_panels = registry.detect_and_sort(page, sorter, page_width=800, is_spread=False)

        # RTL: right panel (x=450) should be ordinal 1, left panel (x=50) ordinal 2
        assert [(p.x, p.ordinal) for p in result_panels] == [(450, 1), (50, 2)]


class TestPipelineBackendReuse:
    """Verify ConversionPipeline resolves and reuses one backend/session across pages and batch."""

    def test_pipeline_reuses_single_backend_across_pages_and_batch_books(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch_dir = tmp_path / "batch"
        _make_cbz(batch_dir / "vol1.cbz", num_pages=3)
        _make_cbz(batch_dir / "vol2.cbz", num_pages=2)

        resolve_calls = 0
        stub_backend = _StubBackend(DetectorKind.CONTOUR)

        def fake_resolve(config: DetectorConfig):
            nonlocal resolve_calls
            resolve_calls += 1
            return [stub_backend]

        monkeypatch.setattr("manga2kindle.cv.backends.resolve_auto_chain", fake_resolve)

        fake_calibre = tmp_path / "calibre-debug"
        fake_calibre.write_text("#!/bin/sh\n")
        monkeypatch.setattr(PlatformDiscovery, "find_calibre", lambda self, override=None: fake_calibre)
        monkeypatch.setattr(
            "manga2kindle.pipeline.KfxCompiler.compile",
            lambda self, kpf_path, output_kfx: (output_kfx.write_bytes(b"KFX"), output_kfx)[1],
        )
        monkeypatch.setattr("manga2kindle.pipeline.KfxCompiler.verify_pdoc", lambda self, _p: True)

        cfg = ConversionConfig(
            input_path=batch_dir,
            output_path=tmp_path / "out",
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
            workers=2,
        )
        pipeline = ConversionPipeline()
        outputs = pipeline.run_batch(cfg)

        assert len(outputs) == 2
        assert resolve_calls == 1
        assert len(stub_backend.calls) == 5

    def test_preview_batch_reuses_single_backend_across_books(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch_dir = tmp_path / "batch_preview"
        _make_cbz(batch_dir / "vol1.cbz", num_pages=3)
        _make_cbz(batch_dir / "vol2.cbz", num_pages=2)

        resolve_calls = 0
        stub_backend = _StubBackend(DetectorKind.CONTOUR)

        def fake_resolve(config: DetectorConfig):
            nonlocal resolve_calls
            resolve_calls += 1
            return [stub_backend]

        monkeypatch.setattr("manga2kindle.cv.backends.resolve_auto_chain", fake_resolve)

        cfg = ConversionConfig(
            input_path=batch_dir,
            output_path=tmp_path / "preview_out",
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
            workers=2,
        )
        pipeline = ConversionPipeline()
        preview_dirs = pipeline.preview_batch(cfg)

        assert len(preview_dirs) == 2
        assert resolve_calls == 1
        assert len(stub_backend.calls) == 5


class TestModelsAndDoctorCli:
    """Verify `models list`, `models fetch`, and `doctor` optional diagnostics."""

    def test_models_list_shows_registered_models_and_licenses(self) -> None:
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0, result.output
        assert "manga109-yolo-s" in result.output
        assert "manga109-yolo-n" in result.output
        assert "magiv3" in result.output
        assert "Manga109-s" in result.output or "non-commercial" in result.output.lower()

    def test_models_fetch_downloads_and_prints_license(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_path = tmp_path / "model.onnx"
        fake_path.write_bytes(b"onnx")
        monkeypatch.setattr(
            "manga2kindle.cv.models_hub.fetch_model",
            lambda model_key, force=False: fake_path,
        )
        result = runner.invoke(app, ["models", "fetch", "manga109-yolo-s"])
        assert result.exit_code == 0, result.output
        collapsed = result.output.replace("\n", "")
        assert str(fake_path) in collapsed
        assert "Manga109-s" in collapsed

    def test_preview_panels_explicit_onnx_missing_weights_fails(self, tmp_path: Path) -> None:
        cbz_path = _make_cbz(tmp_path / "ch.cbz", num_pages=1)
        result = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(tmp_path / "prev"),
                "--no-kcc",
                "--detector",
                "onnx",
                "--detector-model",
                str(tmp_path / "missing.onnx"),
            ],
        )
        assert result.exit_code == 1

    def test_run_doctor_includes_optional_detector_diagnostics(self) -> None:
        statuses = PlatformDiscovery().run_doctor()
        names = {s.name for s in statuses}
        assert {"onnxruntime", "torch", "Cached Models", "AUTO Detector"} <= names
        auto_status = next(s for s in statuses if s.name == "AUTO Detector")
        assert auto_status.found is True
        assert "contour" in auto_status.details

    def test_resolve_batch_workers_auto_and_tunable(self) -> None:
        assert ConversionPipeline._resolve_batch_workers(
            16, 10, no_kcc=False, detector_kind=DetectorKind.CONTOUR, available_ram_gib=16.0
        ) == (2, 8)
        assert ConversionPipeline._resolve_batch_workers(
            16, 10, no_kcc=False, detector_kind=DetectorKind.MAGI, available_ram_gib=4.0
        ) == (1, 16)
        assert ConversionPipeline._resolve_batch_workers(
            16, 10, no_kcc=False, detector_kind=DetectorKind.MAGI, batch_workers_override=3
        ) == (3, 5)
        assert ConversionPipeline._resolve_batch_workers(
            4, 10, no_kcc=True, detector_kind=DetectorKind.CONTOUR
        ) == (4, 1)

    def test_build_contour_backend_and_registry_forward_merge_config(self) -> None:
        from manga2kindle.cv.backends import ContourBackend, _build_contour_backend

        cfg = DetectorConfig(
            kind=DetectorKind.CONTOUR,
            merge_bridged_panels=False,
            panel_overlap_threshold=0.40,
            max_merged_area_ratio=0.50,
            min_fill_ratio=0.70,
        )
        built = _build_contour_backend(cfg)
        assert isinstance(built, ContourBackend)
        assert built.merge_bridged_panels is False
        assert built.panel_overlap_threshold == 0.40
        assert built.max_merged_area_ratio == 0.50
        assert built.min_fill_ratio == 0.70

        resolved_chain = BackendRegistry(cfg).resolve()
        assert len(resolved_chain) == 1
        resolved = resolved_chain[0]
        assert isinstance(resolved, ContourBackend)
        assert resolved.merge_bridged_panels is False
        assert resolved.panel_overlap_threshold == 0.40
        assert resolved.max_merged_area_ratio == 0.50
        assert resolved.min_fill_ratio == 0.70
