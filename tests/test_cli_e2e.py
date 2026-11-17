"""End-to-end CLI and pipeline tests for manga2kindle."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from manga2kindle.cli import app
from manga2kindle.compiler.kfx import KfxCompiler
from manga2kindle.cv.backends.base import DetectionResult
from manga2kindle.models import (
    ConversionConfig,
    DetectorConfig,
    DetectorKind,
    DeviceProfile,
    PanelBox,
    ReadingDirection,
    SpreadMode,
)
from manga2kindle.pipeline import ConversionPipeline
from manga2kindle.platform import PlatformDiscovery, ToolStatus

runner = CliRunner()


def _make_panel_page(path: Path, width: int = 800, height: int = 1200) -> Path:
    """Create a synthetic manga page PNG with two dark-bordered panels."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Top-right and bottom-left panels
    draw.rectangle([420, 60, 740, 560], fill=(230, 230, 230), outline=(0, 0, 0), width=8)
    draw.rectangle([60, 620, 740, 1140], fill=(210, 210, 210), outline=(0, 0, 0), width=8)
    img.save(path, format="PNG")
    return path


def _make_cbz(cbz_path: Path, num_pages: int = 2) -> Path:
    """Create a synthetic CBZ archive containing synthetic manga pages."""
    cbz_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(cbz_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx in range(num_pages):
                img_path = _make_panel_page(tmp_dir / f"page_{idx + 1:03d}.png")
                zf.write(img_path, arcname=img_path.name)
    return cbz_path


def _has_calibre_and_kfx_output() -> Path | None:
    discovery = PlatformDiscovery()
    statuses = {s.name: s for s in discovery.run_doctor()}
    calibre = statuses.get("calibre-debug")
    kfx_out = statuses.get("KFX Output")
    if calibre and calibre.found and kfx_out and kfx_out.found and calibre.path:
        return calibre.path
    return None


def _all_tools_status() -> list[ToolStatus]:
    return [
        ToolStatus(name="kcc-c2e", found=True, path=Path("/usr/bin/kcc-c2e"), version="13.0.1"),
        ToolStatus(
            name="calibre-debug",
            found=True,
            path=Path("/usr/bin/calibre-debug"),
            version="calibre 9.11",
        ),
        ToolStatus(
            name="KFX Output",
            found=True,
            path=Path("/usr/bin/calibre-debug"),
            version="KFX Output 2.19.0",
        ),
        ToolStatus(
            name="Kindle Previewer 4",
            found=True,
            path=Path("/usr/bin/KindlePreviewer4CLI"),
        ),
        ToolStatus(name="unrar/unar/7z", found=True, path=Path("/usr/bin/unrar")),
    ]


class TestDoctorCli:
    """Tests for `manga2kindle doctor`."""

    def test_doctor_renders_dependency_table_all_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(PlatformDiscovery, "run_doctor", lambda self, **_: _all_tools_status())

        result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 0, result.output
        for expected in (
            "Python",
            "OpenCV",
            "kcc-c2e",
            "calibre-debug",
            "KFX Output",
            "Kindle Previewer 4",
            "unrar/unar/7z",
        ):
            assert expected in result.output

    def test_doctor_exits_nonzero_when_required_tool_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        statuses = _all_tools_status()
        statuses[1] = ToolStatus(name="calibre-debug", found=False, details="Not found")
        statuses[2] = ToolStatus(
            name="KFX Output", found=False, details="calibre-debug not found"
        )
        monkeypatch.setattr(PlatformDiscovery, "run_doctor", lambda self, **_: statuses)

        result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 1
        assert "calibre-debug" in result.output
        assert "KFX Output" in result.output
        assert "Missing" in result.output

    def test_doctor_exits_zero_when_only_optional_tools_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        statuses = _all_tools_status()
        statuses[3] = ToolStatus(name="Kindle Previewer 4", found=False, details="Not found")
        statuses[4] = ToolStatus(name="unrar/unar/7z", found=False, details="Not found")
        monkeypatch.setattr(PlatformDiscovery, "run_doctor", lambda self, **_: statuses)

        result = runner.invoke(app, ["doctor"])

        assert result.exit_code == 0, result.output


class TestPreviewPanelsCli:
    """Tests for `manga2kindle preview-panels`."""

    def test_preview_panels_creates_numbered_overlays_and_cleans_temp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "chapter_01.cbz", num_pages=3)
        out_dir = tmp_path / "preview_out"

        created_temps: list[Path] = []
        real_mkdtemp = tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs) -> str:
            created = real_mkdtemp(*args, **kwargs)
            created_temps.append(Path(created))
            return created

        monkeypatch.setattr(tempfile, "mkdtemp", spy_mkdtemp)

        result = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(out_dir),
                "--no-kcc",
                "-p",
                "KPW5",
                "-d",
                "rtl",
                "-s",
                "split",
                "-w",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        overlays = sorted(out_dir.glob("*.png"))
        assert len(overlays) == 3
        expected_sizes = [(1600, 2560), (1264, 1680), (1264, 1680)]
        for overlay, expected_size in zip(overlays, expected_sizes):
            with Image.open(overlay) as img:
                assert img.size == expected_size

        assert created_temps, "Expected pipeline to create a temporary staging directory"
        for temp_dir in created_temps:
            assert not temp_dir.exists()

    def test_preview_panels_forwards_kcc_and_preprocessing_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "chapter_02.cbz", num_pages=1)
        out_dir = tmp_path / "preview_kcc"
        fake_kcc = tmp_path / "kcc-c2e"
        fake_kcc.write_text("#!/bin/sh\n")

        captured: dict[str, ConversionConfig] = {}

        def fake_process(self, pages, work_dir, config: ConversionConfig):
            captured["config"] = config
            processed_dir = work_dir / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            out_img = processed_dir / "0000.png"
            shutil.copy2(pages[0].source_path, out_img)
            return [pages[0].model_copy(update={"processed_path": out_img})]

        monkeypatch.setattr("manga2kindle.preprocessor.ImagePreprocessor.process", fake_process)

        result = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(out_dir),
                "-p",
                "KS",
                "-d",
                "ltr",
                "-s",
                "keep",
                "--kcc-path",
                str(fake_kcc),
            ],
        )

        assert result.exit_code == 0, result.output
        cfg = captured["config"]
        assert cfg.profile == DeviceProfile.from_name("KS")
        assert cfg.direction == ReadingDirection.LTR
        assert cfg.spread_mode == SpreadMode.KEEP
        assert cfg.no_kcc is False
        assert cfg.kcc_path == fake_kcc

    def test_preview_panels_compare_contour_and_onnx_with_stub_backends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "compare_ch.cbz", num_pages=2)
        out_dir = tmp_path / "compare_out"

        class _StubBackend:
            def __init__(self, kind: DetectorKind) -> None:
                self.kind = kind

            def detect(self, image_path: Path) -> DetectionResult:
                with Image.open(image_path) as img:
                    w, h = img.size
                return DetectionResult(
                    panels=[PanelBox(x=40, y=40, w=w - 80, h=h - 80, ordinal=1)],
                    ordered=True,
                    backend=self.kind,
                    page_width=w,
                    page_height=h,
                )

        def fake_resolve_auto_chain(config: DetectorConfig):
            return [_StubBackend(config.kind)]

        monkeypatch.setattr("manga2kindle.cv.backends.resolve_auto_chain", fake_resolve_auto_chain)

        result = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(out_dir),
                "--no-kcc",
                "--compare",
                "contour,onnx",
            ],
        )

        assert result.exit_code == 0, result.output
        overlays = sorted(out_dir.glob("*.png"))
        assert len(overlays) == 2
        expected_widths = [1600 * 2, 1264 * 2]
        for overlay, expected_w in zip(overlays, expected_widths):
            with Image.open(overlay) as img:
                assert img.size[0] == expected_w

    def test_preview_panels_compare_rejects_auto(self, tmp_path: Path) -> None:
        cbz_path = _make_cbz(tmp_path / "compare_auto.cbz", num_pages=1)
        out_dir = tmp_path / "compare_auto_out"

        result = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(out_dir),
                "--no-kcc",
                "--compare",
                "contour,auto",
            ],
        )

        assert result.exit_code != 0
        assert "auto" in result.output.lower()

    def test_preview_panels_compare_magi_requires_allow_remote_code(self, tmp_path: Path) -> None:
        cbz_path = _make_cbz(tmp_path / "compare_magi.cbz", num_pages=1)
        out_dir = tmp_path / "compare_magi_out"

        result = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(out_dir),
                "--no-kcc",
                "--compare",
                "contour,magi",
            ],
        )

        assert result.exit_code != 0
        assert "allow-remote-code" in result.output.lower()


class TestConvertCliMocked:
    """Fast unit/E2E tests for `manga2kindle convert` with mocked KfxCompiler."""

    @staticmethod
    def _mock_calibre(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        pdoc_valid: bool = True,
        captured_kpfs: list[Path] | None = None,
    ) -> Path:
        fake_calibre = tmp_path / "calibre-debug"
        fake_calibre.write_text("#!/bin/sh\n")
        monkeypatch.setattr(
            PlatformDiscovery,
            "find_calibre",
            lambda self, override=None: override if override else fake_calibre,
        )

        def fake_compile(self, kpf_path: Path, output_kfx: Path) -> Path:
            assert zipfile.is_zipfile(kpf_path)
            if captured_kpfs is not None:
                snap = tmp_path / f"captured_{len(captured_kpfs)}.kpf"
                shutil.copy2(kpf_path, snap)
                captured_kpfs.append(snap)
            output_kfx.parent.mkdir(parents=True, exist_ok=True)
            output_kfx.write_bytes(b"FAKE_KFX_DATA")
            return output_kfx

        monkeypatch.setattr(KfxCompiler, "compile", fake_compile)
        monkeypatch.setattr(KfxCompiler, "verify_pdoc", lambda self, kfx_path: pdoc_valid)
        return fake_calibre

    def test_convert_single_cbz_produces_kfx_enforces_pdoc_and_cleans_temp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "Sample_01.cbz", num_pages=2)
        out_kfx = tmp_path / "out" / "sample.kfx"
        captured_kpfs: list[Path] = []
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True, captured_kpfs=captured_kpfs)

        created_temps: list[Path] = []
        real_mkdtemp = tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs) -> str:
            created = real_mkdtemp(*args, **kwargs)
            created_temps.append(Path(created))
            return created

        monkeypatch.setattr(tempfile, "mkdtemp", spy_mkdtemp)

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "-o",
                str(out_kfx),
                "--no-kcc",
                "--title",
                "Sample Manga Vol 1",
                "--author",
                "Sample Author",
                "-z",
                "1.6",
                "--padding",
                "0.03",
                "--min-panel-area",
                "0.04",
                "-w",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert out_kfx.is_file()
        assert len(captured_kpfs) == 1
        with zipfile.ZipFile(captured_kpfs[0], "r") as zf:
            names = set(zf.namelist())
            assert "book.kdf" in names
            assert "book.kcb" in names
            raw_kdf = zf.read("book.kdf")
            assert b"Sample Manga Vol 1" in raw_kdf
            assert b"Sample Author" in raw_kdf

        assert created_temps
        for temp_dir in created_temps:
            assert not temp_dir.exists()

    def test_convert_keep_temp_retains_temp_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "keep_me.cbz", num_pages=1)
        out_kfx = tmp_path / "keep_me.kfx"
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)

        created_temps: list[Path] = []
        real_mkdtemp = tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs) -> str:
            created = real_mkdtemp(*args, **kwargs)
            created_temps.append(Path(created))
            return created

        monkeypatch.setattr(tempfile, "mkdtemp", spy_mkdtemp)

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "-o",
                str(out_kfx),
                "--no-kcc",
                "--keep-temp",
            ],
        )

        try:
            assert result.exit_code == 0, result.output
            assert out_kfx.is_file()
            assert created_temps
            assert any(temp_dir.exists() for temp_dir in created_temps)
        finally:
            for temp_dir in created_temps:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_convert_fails_when_verify_pdoc_is_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "bad_pdoc.cbz", num_pages=1)
        out_kfx = tmp_path / "bad_pdoc.kfx"
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=False)

        result = runner.invoke(
            app,
            ["convert", "-i", str(cbz_path), "-o", str(out_kfx), "--no-kcc"],
        )

        assert result.exit_code != 0
        assert "PDOC" in result.output

    def test_convert_directory_of_images_disambiguated_as_single(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        img_dir = tmp_path / "Volume_01"
        img_dir.mkdir()
        _make_panel_page(img_dir / "01.png")
        _make_panel_page(img_dir / "02.png")
        out_dir = tmp_path / "output_books"
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)

        result = runner.invoke(
            app,
            ["convert", "-i", str(img_dir), "-o", str(out_dir), "--no-kcc"],
        )

        assert result.exit_code == 0, result.output
        assert (out_dir / "Volume_01.kfx").is_file()

    def test_convert_batch_partial_failure_converts_valid_logs_corrupt_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        batch_dir = tmp_path / "chapters"
        batch_dir.mkdir()
        _make_cbz(batch_dir / "01_valid.cbz", num_pages=2)
        (batch_dir / "02_corrupt.cbz").write_bytes(b"not-a-valid-zip-archive")
        out_dir = tmp_path / "kfx_out"
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)

        created_temps: list[Path] = []
        real_mkdtemp = tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs) -> str:
            created = real_mkdtemp(*args, **kwargs)
            created_temps.append(Path(created))
            return created

        monkeypatch.setattr(tempfile, "mkdtemp", spy_mkdtemp)

        result = runner.invoke(
            app,
            ["convert", "-i", str(batch_dir), "-o", str(out_dir), "--no-kcc", "-w", "2"],
        )

        assert result.exit_code == 1
        assert (out_dir / "01_valid.kfx").is_file()
        assert not (out_dir / "02_corrupt.kfx").exists()
        assert "02_corrupt.cbz" in result.output

        assert created_temps
        for temp_dir in created_temps:
            assert not temp_dir.exists()

    def test_convert_empty_directory_fails_cleanly(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = runner.invoke(
            app,
            ["convert", "-i", str(empty_dir), "-o", str(tmp_path / "out"), "--no-kcc"],
        )

        assert result.exit_code != 0

    def test_convert_batch_all_valid_exits_zero_and_omitted_output_defaults_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        batch_dir = tmp_path / "all_valid"
        batch_dir.mkdir()
        _make_cbz(batch_dir / "ch_01.cbz", num_pages=1)
        _make_cbz(batch_dir / "ch_02.cbz", num_pages=1)
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(batch_dir),
                "--no-kcc",
                "-p",
                "CUSTOM",
                "--custom-size",
                "1200x1600",
                "-w",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (batch_dir / "ch_01.kfx").is_file()
        assert (batch_dir / "ch_02.kfx").is_file()

    def test_convert_single_omitted_output_defaults_adjacent_kfx(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "one_shot.cbz", num_pages=1)
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)

        result = runner.invoke(
            app,
            ["convert", "-i", str(cbz_path), "--no-kcc"],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "one_shot.kfx").is_file()

    def test_pipeline_run_single_and_batch_disambiguation_api(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)
        pipeline = ConversionPipeline()

        img_dir = tmp_path / "single_folder"
        img_dir.mkdir()
        _make_panel_page(img_dir / "p1.png")
        single_cfg = ConversionConfig(
            input_path=img_dir,
            output_path=tmp_path / "single_out",
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
        )
        single_out = pipeline.run(single_cfg)
        assert single_out == [tmp_path / "single_out" / "single_folder.kfx"]
        assert single_out[0].is_file()

        batch_dir = tmp_path / "batch_via_run"
        batch_dir.mkdir()
        _make_cbz(batch_dir / "v01.cbz", num_pages=1)
        _make_cbz(batch_dir / "v02.cbz", num_pages=1)
        batch_cfg = ConversionConfig(
            input_path=batch_dir,
            output_path=tmp_path / "batch_out",
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
        )
        batch_out = pipeline.run(batch_cfg)
        assert sorted(p.name for p in batch_out) == ["v01.kfx", "v02.kfx"]
        assert all(p.is_file() for p in batch_out)

    def test_convert_dotted_directory_name_preserves_full_folder_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dotted_dir = tmp_path / "Manga Vol. 1"
        dotted_dir.mkdir()
        _make_panel_page(dotted_dir / "001.png")
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)

        result = runner.invoke(app, ["convert", "-i", str(dotted_dir), "--no-kcc"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "Manga Vol. 1.kfx").is_file()

    def test_convert_rejects_batch_stem_collision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        batch_dir = tmp_path / "colliding_batch"
        batch_dir.mkdir()
        _make_cbz(batch_dir / "chapter_01.cbz", num_pages=1)
        (batch_dir / "chapter_01.cbr").write_bytes(b"Rar!stub")
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)

        result = runner.invoke(app, ["convert", "-i", str(batch_dir), "--no-kcc"])
        assert result.exit_code != 0
        assert "collision" in result.output.lower()

    def test_cli_validation_and_version_errors(self, tmp_path: Path) -> None:
        ver_res = runner.invoke(app, ["--version"])
        assert ver_res.exit_code == 0
        assert "manga2kindle" in ver_res.output

        cbz_path = _make_cbz(tmp_path / "single.cbz", num_pages=1)

        bad_custom = runner.invoke(
            app,
            ["convert", "-i", str(cbz_path), "-p", "KPW5", "--custom-size", "1000x1400"],
        )
        assert bad_custom.exit_code != 0

        bad_dims = runner.invoke(
            app,
            ["convert", "-i", str(cbz_path), "-p", "CUSTOM", "--custom-size", "not-a-size"],
        )
        assert bad_dims.exit_code != 0

        bad_zoom = runner.invoke(app, ["convert", "-i", str(cbz_path), "-z", "0.5"])
        assert bad_zoom.exit_code != 0

        bad_workers = runner.invoke(app, ["convert", "-i", str(cbz_path), "-w", "0"])
        assert bad_workers.exit_code != 0

        bad_cover_fmt = runner.invoke(
            app, ["convert", "-i", str(cbz_path), "--cover-size", "not-a-size"]
        )
        assert bad_cover_fmt.exit_code != 0

        bad_cover_zero = runner.invoke(
            app, ["convert", "-i", str(cbz_path), "--cover-size", "0x2560"]
        )
        assert bad_cover_zero.exit_code != 0

        batch_dir = tmp_path / "batch_val"
        batch_dir.mkdir()
        _make_cbz(batch_dir / "c1.cbz", num_pages=1)
        bad_batch_out = runner.invoke(
            app, ["convert", "-i", str(batch_dir), "-o", str(tmp_path / "file.kfx")]
        )
        assert bad_batch_out.exit_code != 0

        existing_file = tmp_path / "already_a_file"
        existing_file.write_text("occupied")
        bad_existing = runner.invoke(
            app, ["convert", "-i", str(batch_dir), "-o", str(existing_file)]
        )
        assert bad_existing.exit_code != 0

        bad_batch_title = runner.invoke(
            app, ["convert", "-i", str(batch_dir), "--title", "Series"]
        )
        assert bad_batch_title.exit_code != 0

    def test_convert_forwards_custom_cover_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "cover_flag.cbz", num_pages=1)
        out_kfx = tmp_path / "cover_flag.kfx"
        captured: list[ConversionConfig] = []

        def fake_run_single(self, config: ConversionConfig, **kwargs) -> Path:
            captured.append(config)
            config.output_path.write_bytes(b"kfx")
            return config.output_path

        monkeypatch.setattr(ConversionPipeline, "run_single", fake_run_single)

        res = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "-o",
                str(out_kfx),
                "--no-kcc",
                "--cover-size",
                "1375x2200",
            ],
        )
        assert res.exit_code == 0, res.output
        assert len(captured) == 1
        assert captured[0].cover_size == (1375, 2200)

    def test_convert_preview_only_single_without_calibre(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "preview_single.cbz", num_pages=2)
        preview_dir = tmp_path / "preview_single_out"
        monkeypatch.setattr(PlatformDiscovery, "find_calibre", lambda self, override=None: None)

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "-o",
                str(preview_dir),
                "--no-kcc",
                "--preview-only",
            ],
        )

        assert result.exit_code == 0, result.output
        overlays = sorted(preview_dir.glob("*.png"))
        assert len(overlays) == 2
        assert list(tmp_path.rglob("*.kfx")) == []

        # Also works when -o is omitted (defaults to <name>_preview)
        default_result = runner.invoke(
            app,
            ["convert", "-i", str(cbz_path), "--no-kcc", "--preview-only"],
        )
        assert default_result.exit_code == 0, default_result.output
        default_dir = tmp_path / "preview_single_preview"
        assert len(list(default_dir.glob("*.png"))) == 2
        assert list(tmp_path.rglob("*.kfx")) == []

    def test_convert_preview_only_batch_without_calibre(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        batch_dir = tmp_path / "batch_preview"
        batch_dir.mkdir()
        _make_cbz(batch_dir / "ch_01.cbz", num_pages=1)
        _make_cbz(batch_dir / "ch_02.cbz", num_pages=2)
        out_dir = tmp_path / "batch_preview_out"
        monkeypatch.setattr(PlatformDiscovery, "find_calibre", lambda self, override=None: None)

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(batch_dir),
                "-o",
                str(out_dir),
                "--no-kcc",
                "--preview-only",
                "-w",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(list((out_dir / "ch_01_preview").glob("*.png"))) == 1
        assert len(list((out_dir / "ch_02_preview").glob("*.png"))) == 2
        assert list(tmp_path.rglob("*.kfx")) == []


class TestConvertCliIntegration:
    """Live E2E test with real calibre-debug + KFX Output when installed."""

    def test_convert_single_cbz_real_calibre_integration(self, tmp_path: Path) -> None:
        calibre_debug = _has_calibre_and_kfx_output()
        if calibre_debug is None:
            pytest.skip("calibre-debug or KFX Output plugin not installed")

        cbz_path = _make_cbz(tmp_path / "real_e2e.cbz", num_pages=2)
        out_kfx = tmp_path / "real_e2e.kfx"

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "-o",
                str(out_kfx),
                "--no-kcc",
                "--title",
                "E2E Integration Manga",
                "--author",
                "Test Mangaka",
            ],
        )

        assert result.exit_code == 0, result.output
        assert out_kfx.is_file()
        assert out_kfx.stat().st_size > 0
        assert KfxCompiler(calibre_debug).verify_pdoc(out_kfx) is True

        # Verify compiled KFX cover_image resource has 1600x2560 dimensions
        import subprocess
        inspect = subprocess.run(
            [
                str(calibre_debug),
                "-c",
                (
                    "import os, sys; from calibre.utils.config import config_dir; "
                    "sys.path.insert(0, os.path.join(config_dir, 'plugins', 'KFX Output.zip')); "
                    "from kfxlib.yj_book import YJ_Book; "
                    f"b = YJ_Book({str(out_kfx)!r}); b.decode_book(); "
                    "r = [f for f in b.fragments if f.ftype == '$164' and str(f.fid) == 'e0'][0]; "
                    "assert (r.value['$422'], r.value['$423']) == (1600, 2560), r.value"
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
