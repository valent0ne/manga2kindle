"""Tests for built-in CLI progress reporting and pipeline streaming."""

from __future__ import annotations

import logging
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageDraw
from rich.console import Console

from manga2kindle.cv.backends import BackendRegistry
from manga2kindle.cv.backends.base import DetectionResult
from manga2kindle.cv.backends.magi import MagiBackend
from manga2kindle.models import (
    ConversionConfig,
    DetectorConfig,
    DetectorKind,
    DevicePreference,
    DeviceProfile,
    PanelBox,
)
from manga2kindle.pipeline import ConversionPipeline
from manga2kindle.platform import PlatformDiscovery
from manga2kindle.progress import (
    STAGE_LABELS,
    ConversionStage,
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    RichProgressReporter,
)


class TestProgressEvent:
    """Unit tests for ConversionStage and ProgressEvent formatting."""

    def test_format_detail_with_completed_and_total(self) -> None:
        event = ProgressEvent(
            book_id="b1",
            book_name="Vol 01",
            stage=ConversionStage.DETECTING,
            completed=12,
            total=180,
        )
        assert event.format_detail() == f"{STAGE_LABELS[ConversionStage.DETECTING]} (12/180 pages)"

    def test_format_detail_with_total_only(self) -> None:
        event = ProgressEvent(
            book_id="b1",
            book_name="Vol 01",
            stage=ConversionStage.PREPROCESSING,
            total=42,
        )
        assert event.format_detail() == f"{STAGE_LABELS[ConversionStage.PREPROCESSING]} (42 pages)"

    def test_format_detail_without_counts_or_zero_total(self) -> None:
        event_none = ProgressEvent(
            book_id="b1",
            book_name="Vol 01",
            stage=ConversionStage.EXTRACTING,
        )
        event_zero = ProgressEvent(
            book_id="b1",
            book_name="Vol 01",
            stage=ConversionStage.COMPILING,
            completed=0,
            total=0,
        )
        assert event_none.format_detail() == STAGE_LABELS[ConversionStage.EXTRACTING]
        assert event_zero.format_detail() == STAGE_LABELS[ConversionStage.COMPILING]


class TestNullProgressReporter:
    """Ensure NullProgressReporter satisfies ProgressReporter protocol."""

    def test_protocol_and_no_op_methods(self, tmp_path: Path) -> None:
        reporter = NullProgressReporter()
        assert isinstance(reporter, ProgressReporter)
        reporter.on_batch_start(2)
        reporter.on_book_start("id", "name")
        reporter.on_book_stage(ProgressEvent("id", "name", ConversionStage.EXTRACTING))
        reporter.on_book_complete("id", "name", tmp_path / "out.kfx")
        reporter.on_book_skip("id", "name", tmp_path / "out.kfx")
        reporter.on_book_error("id", "src.cbz", RuntimeError("boom"))
        reporter.on_batch_finish()


class TestRichProgressReporter:
    """Tests for RichProgressReporter in non-TTY, TTY live, markup-escape, and threaded modes."""

    def test_non_tty_batch_streams_outcomes_without_live_bars(self, tmp_path: Path) -> None:
        console = Console(force_terminal=False, record=True, width=160)
        with RichProgressReporter(console, is_batch=True, show_live=False) as reporter:
            assert isinstance(reporter, ProgressReporter)
            reporter.on_batch_start(3, label="Converting")
            reporter.on_book_skip("1", "Vol 01", tmp_path / "Vol 01.kfx")
            reporter.on_book_start("2", "Vol 02")
            reporter.on_book_stage(
                ProgressEvent("2", "Vol 02", ConversionStage.DETECTING, completed=5, total=10)
            )
            reporter.on_book_complete("2", "Vol 02", tmp_path / "Vol 02.kfx")
            reporter.on_book_start("3", "Vol 03")
            reporter.on_book_error("3", "Vol 03.cbz", RuntimeError("corrupt archive"))
            reporter.on_batch_finish()

        output = console.export_text()
        assert f"Skipped (already exists): {tmp_path / 'Vol 01.kfx'}" in output
        assert f"Converted: {tmp_path / 'Vol 02.kfx'}" in output
        assert "Failed: Vol 03.cbz - corrupt archive" in output
        assert "ETA" not in output

    def test_non_tty_single_mode_does_not_print_completion(self, tmp_path: Path) -> None:
        console = Console(force_terminal=False, record=True, width=160)
        with RichProgressReporter(console, is_batch=False, show_live=False) as reporter:
            reporter.on_book_start("1", "One Shot")
            reporter.on_book_stage(ProgressEvent("1", "One Shot", ConversionStage.COMPILING))
            reporter.on_book_complete("1", "One Shot", tmp_path / "One Shot.kfx")
            reporter.on_book_skip("1", "One Shot", tmp_path / "One Shot.kfx")
            reporter.on_book_error("1", "One Shot.cbz", RuntimeError("fail"))

        assert console.export_text().strip() == ""

    def test_tty_live_mode_updates_tasks_and_cleans_workers(self, tmp_path: Path) -> None:
        console = Console(force_terminal=True, record=True, width=140)
        with RichProgressReporter(console, is_batch=True, show_live=True) as reporter:
            reporter.on_batch_start(3, label="Converting")
            assert reporter._overall_task_id is not None
            assert reporter._overall_progress is not None
            assert reporter._worker_progress is not None

            reporter.on_book_start("b1", "Book 01")
            assert "b1" in reporter._active_tasks

            reporter.on_book_stage(
                ProgressEvent("b1", "Book 01", ConversionStage.DETECTING, completed=4, total=10)
            )
            worker_task = next(
                t for t in reporter._worker_progress.tasks if t.id == reporter._active_tasks["b1"]
            )
            assert "Detecting panels (4/10 pages)" in worker_task.fields["detail"]

            reporter.on_book_complete("b1", "Book 01", tmp_path / "Book 01.kfx")
            assert "b1" not in reporter._active_tasks
            assert len(reporter._worker_progress.tasks) == 0

            reporter.on_book_skip("b3", "Book 03", tmp_path / "Book 03.kfx")

            reporter.on_book_start("b2", "Book 02")
            reporter.on_book_error("b2", "Book 02.cbz", RuntimeError("broken"))
            assert "b2" not in reporter._active_tasks

            overall_task = reporter._overall_progress.tasks[0]
            assert overall_task.completed == 3
            assert "3/3 books (1 ok, 1 skip, 1 fail)" in overall_task.fields["counts"]
            reporter.on_batch_finish()

        text = console.export_text()
        assert "Converted:" not in text
        assert "Skipped (already exists):" not in text
        assert "Failed: Book 02.cbz - broken" in text

    def test_tty_live_mode_sets_transient_on_exception(self) -> None:
        console = Console(force_terminal=True, record=True, width=140)
        reporter = RichProgressReporter(console, is_batch=True, show_live=True)
        try:
            with reporter:
                reporter.on_batch_start(3, label="Converting")
                reporter.on_book_start("b1", "Book 01")
                raise RuntimeError("interrupted")
        except RuntimeError:
            pass

        assert reporter._live is not None
        assert reporter._live.transient is True
        assert reporter._active_tasks == {}

    def test_tty_single_mode_sets_transient_true_on_exit(self, tmp_path: Path) -> None:
        console = Console(force_terminal=True, record=True, width=140)
        reporter = RichProgressReporter(console, is_batch=False, show_live=True)
        with reporter:
            reporter.on_book_start("b1", "Single")
            reporter.on_book_stage(ProgressEvent("b1", "Single", ConversionStage.EXTRACTING))
            reporter.on_book_complete("b1", "Single", tmp_path / "Single.kfx")

        assert reporter._live is not None
        assert reporter._live.transient is True

    def test_markup_characters_in_names_and_errors_are_escaped(self, tmp_path: Path) -> None:
        tricky_name = "[scans] Manga [b] [/tag] [/]"
        tricky_source = f"{tricky_name}.cbz"
        tricky_out = tmp_path / f"{tricky_name}.kfx"
        tricky_err = RuntimeError("bad xml [/tag] and [/] [red]")

        for show_live in (False, True):
            console = Console(force_terminal=show_live, record=True, width=220)
            with RichProgressReporter(console, is_batch=True, show_live=show_live) as reporter:
                reporter.on_batch_start(3, label="Converting [all]")
                reporter.on_book_start("b1", tricky_name)
                reporter.on_book_stage(
                    ProgressEvent("b1", tricky_name, ConversionStage.DETECTING, completed=1, total=2)
                )
                reporter.on_book_complete("b1", tricky_name, tricky_out)
                reporter.on_book_skip("b2", tricky_name, tricky_out)
                reporter.on_book_start("b3", tricky_name)
                reporter.on_book_error("b3", tricky_source, tricky_err)
                reporter.on_batch_finish()

            exported = console.export_text()
            if show_live:
                assert "Converted:" not in exported
                assert "Skipped (already exists):" not in exported
            else:
                assert str(tricky_out) in exported
            assert "[scans] Manga [b] [/tag] [/].cbz" in exported
            assert "bad xml [/tag] and [/] [red]" in exported

    def test_concurrent_thread_updates_keep_exact_counts(self, tmp_path: Path) -> None:
        console = Console(force_terminal=True, record=True, width=160)
        total = 24
        with RichProgressReporter(console, is_batch=True, show_live=True) as reporter:
            reporter.on_batch_start(total, label="Converting")

            def _worker(idx: int) -> None:
                bid = f"book-{idx}"
                bname = f"Book {idx:02d}"
                if idx % 5 == 0:
                    reporter.on_book_skip(bid, bname, tmp_path / f"{bname}.kfx")
                    return
                reporter.on_book_start(bid, bname)
                for stage in (
                    ConversionStage.EXTRACTING,
                    ConversionStage.PREPROCESSING,
                    ConversionStage.DETECTING,
                    ConversionStage.COMPILING,
                ):
                    reporter.on_book_stage(
                        ProgressEvent(bid, bname, stage, completed=1, total=2)
                    )
                if idx % 4 == 0:
                    reporter.on_book_error(bid, f"{bname}.cbz", RuntimeError("err"))
                else:
                    reporter.on_book_complete(bid, bname, tmp_path / f"{bname}.kfx")

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(_worker, range(total)))

            reporter.on_batch_finish()
            assert reporter._active_tasks == {}
            assert reporter._overall_progress is not None
            assert reporter._overall_progress.tasks[0].completed == total
            assert reporter._succeeded + reporter._skipped + reporter._failed == total


class TestBackendBatchProgress:
    """Tests for chunk-level progress callbacks on MagiBackend and BackendRegistry."""

    def test_magi_detect_batch_reports_chunk_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_dir = tmp_path / "magiv3"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "florence2"}', encoding="utf-8")

        pages: list[Path] = []
        for idx in range(5):
            page = tmp_path / f"p{idx}.png"
            Image.new("RGB", (400, 600), (255, 255, 255)).save(page, format="PNG")
            pages.append(page)

        @contextmanager
        def _inference_mode():
            yield

        fake_model = MagicMock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model
        fake_model.predict_detections_and_associations.side_effect = lambda imgs, _proc=None: [
            {"panels": [[20.0, 20.0, 380.0, 580.0]], "texts": []} for _ in imgs
        ]
        auto_lm = MagicMock()
        auto_lm.from_pretrained.return_value = fake_model
        auto_proc = MagicMock()
        auto_proc.from_pretrained.return_value = MagicMock()

        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                float16="float16",
                float32="float32",
                cuda=SimpleNamespace(is_available=lambda: False),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
                inference_mode=_inference_mode,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoModelForCausalLM=auto_lm,
                AutoModel=auto_lm,
                AutoProcessor=auto_proc,
            ),
        )

        backend = MagiBackend(
            DetectorConfig(
                kind=DetectorKind.MAGI,
                model_path=model_dir,
                device=DevicePreference.CPU,
                batch_size=2,
                allow_remote_code=True,
            )
        )
        seen: list[tuple[int, int]] = []
        results = backend.detect_batch(pages, on_progress=lambda done, total: seen.append((done, total)))

        assert len(results) == 5
        assert seen == [(2, 5), (4, 5), (5, 5)]

    def test_backend_registry_supports_progress_aware_and_legacy_backends(self, tmp_path: Path) -> None:
        pages = [tmp_path / "p1.png", tmp_path / "p2.png", tmp_path / "p3.png"]

        class _ProgressAwareBackend:
            kind = DetectorKind.MAGI

            def detect(self, image_path: Path) -> DetectionResult:
                raise NotImplementedError

            def detect_batch(self, image_paths, *, on_progress=None):
                out = []
                for idx, _p in enumerate(image_paths, start=1):
                    out.append(
                        DetectionResult(
                            panels=[PanelBox(x=10, y=10, w=100, h=100, ordinal=1)],
                            ordered=True,
                            backend=self.kind,
                            page_width=400,
                            page_height=600,
                        )
                    )
                    if on_progress is not None:
                        on_progress(idx, len(image_paths))
                return out

        class _LegacyStubBackend:
            kind = DetectorKind.CONTOUR

            def detect(self, image_path: Path) -> DetectionResult:
                raise NotImplementedError

            def detect_batch(self, image_paths):
                return [
                    DetectionResult(
                        panels=[PanelBox(x=10, y=10, w=100, h=100, ordinal=1)],
                        ordered=True,
                        backend=self.kind,
                        page_width=400,
                        page_height=600,
                    )
                    for _ in image_paths
                ]

        aware_calls: list[tuple[int, int]] = []
        reg_aware = BackendRegistry(backends=[_ProgressAwareBackend()])
        res_aware = reg_aware.detect_batch(pages, on_progress=lambda d, t: aware_calls.append((d, t)))
        assert len(res_aware) == 3
        assert aware_calls == [(1, 3), (2, 3), (3, 3)]

        legacy_calls: list[tuple[int, int]] = []
        reg_legacy = BackendRegistry(backends=[_LegacyStubBackend()])
        res_legacy = reg_legacy.detect_batch(pages, on_progress=lambda d, t: legacy_calls.append((d, t)))
        assert len(res_legacy) == 3
        assert legacy_calls == [(3, 3)]


class _RecordingReporter:
    """In-memory ProgressReporter for verifying pipeline event emissions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def on_batch_start(self, total: int, *, label: str = "Converting") -> None:
        self.events.append(("batch_start", (total, label)))

    def on_book_start(self, book_id: str, book_name: str) -> None:
        self.events.append(("book_start", (book_id, book_name)))

    def on_book_stage(self, event: ProgressEvent) -> None:
        self.events.append(("stage", event))

    def on_book_complete(self, book_id: str, book_name: str, output_path: Path) -> None:
        self.events.append(("book_complete", (book_id, book_name, output_path)))

    def on_book_skip(self, book_id: str, book_name: str, output_path: Path) -> None:
        self.events.append(("book_skip", (book_id, book_name, output_path)))

    def on_book_error(self, book_id: str, source_name: str, error: Exception) -> None:
        self.events.append(("book_error", (book_id, source_name, str(error))))

    def on_batch_finish(self) -> None:
        self.events.append(("batch_finish", None))


def _make_sample_cbz(cbz_path: Path, num_pages: int = 2) -> Path:
    cbz_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cbz_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx in range(num_pages):
            img_path = cbz_path.parent / f"__tmp_{cbz_path.stem}_{idx}.png"
            img = Image.new("RGB", (800, 1200), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([420, 60, 740, 560], fill=(220, 220, 220), outline=(0, 0, 0), width=8)
            draw.rectangle([60, 620, 740, 1140], fill=(200, 200, 200), outline=(0, 0, 0), width=8)
            img.save(img_path, format="PNG")
            zf.write(img_path, arcname=f"page_{idx + 1:03d}.png")
            img_path.unlink()
    return cbz_path


def _mock_kfx_compiler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_calibre = tmp_path / "calibre-debug"
    fake_calibre.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(PlatformDiscovery, "find_calibre", lambda self, override=None: fake_calibre)
    monkeypatch.setattr(
        "manga2kindle.pipeline.KfxCompiler.compile",
        lambda self, kpf_path, output_kfx: (
            output_kfx.parent.mkdir(parents=True, exist_ok=True),
            output_kfx.write_bytes(b"FAKE_KFX"),
            output_kfx,
        )[2],
    )
    monkeypatch.setattr("manga2kindle.pipeline.KfxCompiler.verify_pdoc", lambda self, _p: True)


class TestPipelineProgressAndStreaming:
    """Verify stage emissions and real-time as_completed streaming in ConversionPipeline."""

    def test_run_single_emits_all_stages_and_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_kfx_compiler(tmp_path, monkeypatch)
        cbz_path = _make_sample_cbz(tmp_path / "single.cbz", num_pages=2)
        out_kfx = tmp_path / "single.kfx"
        reporter = _RecordingReporter()
        pipeline = ConversionPipeline(progress=reporter)
        cfg = ConversionConfig(
            input_path=cbz_path,
            output_path=out_kfx,
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
            workers=1,
        )

        result = pipeline.run_single(cfg)

        assert result == out_kfx
        kinds = [kind for kind, _ in reporter.events]
        assert kinds[0] == "book_start"
        assert kinds[-1] == "book_complete"

        stage_events = [payload for kind, payload in reporter.events if kind == "stage"]
        assert isinstance(stage_events[0], ProgressEvent)
        stages_seen = [ev.stage for ev in stage_events]
        for expected_stage in (
            ConversionStage.EXTRACTING,
            ConversionStage.PREPROCESSING,
            ConversionStage.DETECTING,
            ConversionStage.PACKAGING,
            ConversionStage.COMPILING,
        ):
            assert expected_stage in stages_seen

        detect_events = [ev for ev in stage_events if ev.stage == ConversionStage.DETECTING]
        assert detect_events[-1].completed == 2
        assert detect_events[-1].total == 2

    def test_run_batch_streams_completions_in_finish_order_but_returns_deterministic_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_kfx_compiler(tmp_path, monkeypatch)
        batch_dir = tmp_path / "batch"
        _make_sample_cbz(batch_dir / "01_slow.cbz", num_pages=1)
        _make_sample_cbz(batch_dir / "02_fast.cbz", num_pages=1)
        out_dir = tmp_path / "kfx_out"

        reporter = _RecordingReporter()
        pipeline = ConversionPipeline(progress=reporter)
        orig_convert_single = ConversionPipeline._convert_single

        def _staggered_convert(self, book_config: ConversionConfig, **kwargs) -> Path:
            if book_config.input_path.name == "01_slow.cbz":
                time.sleep(0.18)
            return orig_convert_single(self, book_config, **kwargs)

        monkeypatch.setattr(ConversionPipeline, "_convert_single", _staggered_convert)

        streamed_names: list[str] = []
        cfg = ConversionConfig(
            input_path=batch_dir,
            output_path=out_dir,
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
            workers=2,
            batch_workers=2,
        )
        outputs = pipeline.run_batch(
            cfg,
            on_success=lambda src, _dst: streamed_names.append(src.name),
        )

        # Streaming hooks receive fast book first...
        assert streamed_names == ["02_fast.cbz", "01_slow.cbz"]
        reporter_completed_names = [
            payload[1] for kind, payload in reporter.events if kind == "book_complete"
        ]
        assert reporter_completed_names == ["02_fast", "01_slow"]
        assert reporter.events[0] == ("batch_start", (2, "Converting"))
        assert reporter.events[-1] == ("batch_finish", None)

        # ...while the returned list preserves archive natural sort order.
        assert outputs == [out_dir / "01_slow.kfx", out_dir / "02_fast.kfx"]

    def test_run_batch_error_reports_source_filename_with_extension(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_kfx_compiler(tmp_path, monkeypatch)
        batch_dir = tmp_path / "batch_err"
        _make_sample_cbz(batch_dir / "01_ok.cbz", num_pages=1)
        (batch_dir / "02_bad.cbz").write_bytes(b"corrupt")
        out_dir = tmp_path / "out_err"

        reporter = _RecordingReporter()
        pipeline = ConversionPipeline(progress=reporter)
        cfg = ConversionConfig(
            input_path=batch_dir,
            output_path=out_dir,
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
            workers=2,
            batch_workers=2,
        )
        outputs = pipeline.run_batch(cfg)

        assert outputs == [out_dir / "01_ok.kfx"]
        errors = [payload for kind, payload in reporter.events if kind == "book_error"]
        assert len(errors) == 1
        assert errors[0][1] == "02_bad.cbz"

    def test_preview_panels_and_preview_batch_emit_rendering_stage_and_stream(
        self, tmp_path: Path
    ) -> None:
        batch_dir = tmp_path / "preview_batch"
        _make_sample_cbz(batch_dir / "a.cbz", num_pages=2)
        _make_sample_cbz(batch_dir / "b.cbz", num_pages=1)
        out_dir = tmp_path / "preview_out"

        single_reporter = _RecordingReporter()
        pipeline = ConversionPipeline(progress=single_reporter)
        rendered = pipeline.preview_panels(
            input_path=batch_dir / "a.cbz",
            output_dir=out_dir / "single_prev",
            no_kcc=True,
            workers=2,
        )
        assert len(rendered) == 2
        single_stages = [
            payload.stage for kind, payload in single_reporter.events if kind == "stage"
        ]
        assert ConversionStage.RENDERING in single_stages
        render_events = [
            payload
            for kind, payload in single_reporter.events
            if kind == "stage" and payload.stage == ConversionStage.RENDERING
        ]
        assert render_events[-1].completed == 2
        assert render_events[-1].total == 2

        batch_reporter = _RecordingReporter()
        batch_pipeline = ConversionPipeline(progress=batch_reporter)
        cfg = ConversionConfig(
            input_path=batch_dir,
            output_path=out_dir / "batch_prev",
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
            workers=2,
            batch_workers=2,
        )
        dirs = batch_pipeline.preview_batch(cfg)
        assert dirs == [out_dir / "batch_prev" / "a_preview", out_dir / "batch_prev" / "b_preview"]
        assert batch_reporter.events[0] == ("batch_start", (2, "Rendering previews"))
        assert batch_reporter.events[-1] == ("batch_finish", None)
        assert len([1 for kind, _ in batch_reporter.events if kind == "book_complete"]) == 2

    def test_run_batch_avoids_duplicate_error_logging_when_ui_reporter_active(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _mock_kfx_compiler(tmp_path, monkeypatch)
        batch_dir = tmp_path / "batch_log_dedup"
        batch_dir.mkdir()
        (batch_dir / "bad.cbz").write_bytes(b"corrupt")
        out_dir = tmp_path / "out_log_dedup"

        cfg = ConversionConfig(
            input_path=batch_dir,
            output_path=out_dir,
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
            workers=1,
            batch_workers=1,
        )

        console = Console(force_terminal=False, record=True, width=160)
        with caplog.at_level(logging.ERROR, logger="manga2kindle.pipeline"):
            with RichProgressReporter(console, is_batch=True, show_live=False) as reporter:
                ConversionPipeline(progress=reporter).run_batch(cfg)

        rich_errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert rich_errors == []
        assert "Failed: bad.cbz" in console.export_text()

        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="manga2kindle.pipeline"):
            ConversionPipeline(progress=NullProgressReporter()).run_batch(cfg)

        null_errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(null_errors) == 1
        assert "Failed to convert" in null_errors[0].getMessage()

    def test_execute_batch_workers_cancels_queued_tasks_on_interrupt(
        self, tmp_path: Path
    ) -> None:
        targets = [
            (tmp_path / f"b{i}.cbz", f"b{i}", tmp_path / f"b{i}.kfx")
            for i in range(6)
        ]
        lock = threading.Lock()
        started: list[int] = []
        captured_events: list[threading.Event] = []
        gate = threading.Event()
        blocked_done = threading.Event()

        def _worker(
            idx: int,
            arch: Path,
            stem: str,
            tgt: Path,
            cancel_event: threading.Event,
        ):
            with lock:
                started.append(idx)
                captured_events.append(cancel_event)
            if idx >= 1:
                gate.wait(timeout=2.0)
                blocked_done.set()
            return idx, arch, stem, tgt, None

        def _handle(_outcome) -> None:
            raise KeyboardInterrupt

        try:
            with pytest.raises(KeyboardInterrupt):
                ConversionPipeline._execute_batch_workers(2, targets, _worker, _handle)
        finally:
            gate.set()

        if any(idx >= 1 for idx in started):
            blocked_done.wait(timeout=1.0)
        assert captured_events and captured_events[0].is_set()
        assert len(started) < len(targets)
