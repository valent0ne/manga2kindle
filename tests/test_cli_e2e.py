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
    PanelMergeConfig,
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
        statuses[3] = ToolStatus(name="unrar/unar/7z", found=False, details="Not found")
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

    def test_preview_panels_defaults_profile_to_kpw6(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "chapter_default_profile.cbz", num_pages=1)
        out_dir = tmp_path / "preview_default"
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
                "--no-kcc",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["config"].profile == DeviceProfile.from_name("KPW6")


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

        bad_zoom = runner.invoke(app, ["convert", "-i", str(cbz_path), "-z", "1.5"])
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

    def test_convert_skip_existing_single_avoids_detector_and_calibre(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "vol_01.cbz", num_pages=1)
        out_kfx = tmp_path / "vol_01.kfx"
        out_kfx.write_bytes(b"EXISTING_KFX")

        def fail_if_called(*args, **kwargs):
            raise AssertionError("Should not be called when output is skipped")

        monkeypatch.setattr(PlatformDiscovery, "find_calibre", fail_if_called)
        monkeypatch.setattr(ConversionPipeline, "_resolve_registry", fail_if_called)

        result = runner.invoke(
            app,
            ["convert", "-i", str(cbz_path), "--no-kcc", "--skip-existing"],
        )

        assert result.exit_code == 0, result.output
        assert "Skipped (already exists):" in result.output
        assert out_kfx.read_bytes() == b"EXISTING_KFX"

    def test_convert_skip_existing_batch_mixed_and_all_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        batch_dir = tmp_path / "batch_skip"
        batch_dir.mkdir()
        _make_cbz(batch_dir / "ch_01.cbz", num_pages=1)
        _make_cbz(batch_dir / "ch_02.cbz", num_pages=1)
        out_dir = tmp_path / "batch_skip_out"
        out_dir.mkdir()
        existing_kfx = out_dir / "ch_01.kfx"
        existing_kfx.write_bytes(b"EXISTING_CH_01")

        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(batch_dir),
                "-o",
                str(out_dir),
                "--no-kcc",
                "--skip-existing",
                "-w",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert existing_kfx.read_bytes() == b"EXISTING_CH_01"
        assert (out_dir / "ch_02.kfx").is_file()
        assert "Skipped (already exists):" in result.output
        assert "1/2 succeeded, 1 skipped, 0 failed." in result.output

        # Re-running when all outputs exist should not initialize detector or calibre
        def fail_if_called(*args, **kwargs):
            raise AssertionError("Should not be called when all outputs are skipped")

        monkeypatch.setattr(PlatformDiscovery, "find_calibre", fail_if_called)
        monkeypatch.setattr(ConversionPipeline, "_resolve_registry", fail_if_called)

        all_skipped = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(batch_dir),
                "-o",
                str(out_dir),
                "--no-kcc",
                "--skip-existing",
            ],
        )

        assert all_skipped.exit_code == 0, all_skipped.output
        assert "0/2 succeeded, 2 skipped, 0 failed." in all_skipped.output

    def test_pipeline_run_skip_existing_invokes_on_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)
        pipeline = ConversionPipeline()

        cbz_path = _make_cbz(tmp_path / "single_skip.cbz", num_pages=1)
        out_kfx = tmp_path / "single_skip.kfx"
        out_kfx.write_bytes(b"EXISTING")

        skipped_calls: list[tuple[Path, Path]] = []
        single_cfg = ConversionConfig(
            input_path=cbz_path,
            output_path=out_kfx,
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
            skip_existing=True,
        )
        assert (
            pipeline.run(
                single_cfg,
                on_skip=lambda src, dst: skipped_calls.append((src, dst)),
            )
            == []
        )
        assert skipped_calls == [(cbz_path, out_kfx)]

        batch_dir = tmp_path / "batch_api_skip"
        batch_dir.mkdir()
        c1 = _make_cbz(batch_dir / "c1.cbz", num_pages=1)
        _make_cbz(batch_dir / "c2.cbz", num_pages=1)
        batch_out = tmp_path / "batch_api_skip_out"
        batch_out.mkdir()
        c1_kfx = batch_out / "c1.kfx"
        c1_kfx.write_bytes(b"EXISTING_C1")

        batch_skipped: list[tuple[Path, Path]] = []
        batch_cfg = ConversionConfig(
            input_path=batch_dir,
            output_path=batch_out,
            profile=DeviceProfile.from_name("KPW5"),
            no_kcc=True,
            skip_existing=True,
        )
        converted = pipeline.run(
            batch_cfg,
            on_skip=lambda src, dst: batch_skipped.append((src, dst)),
        )
        assert converted == [batch_out / "c2.kfx"]
        assert batch_skipped == [(c1, c1_kfx)]

    def test_convert_cli_beautify_and_beautify_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)
        cbz_path = _make_cbz(
            tmp_path / "Witch Hat Atelier v12 (2024) (Digital) (LuCaZ) (F).cbz",
            num_pages=1,
        )

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "--no-kcc",
                "--beautify",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "Witch Hat Atelier v12 (2024).kfx").is_file()
        assert cbz_path.is_file()

        result_src = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "--no-kcc",
                "--beautify-source",
            ],
        )
        assert result_src.exit_code == 0, result_src.output
        assert (tmp_path / "Witch Hat Atelier v12 (2024).cbz").is_file()
        assert not cbz_path.exists()

    def test_convert_cli_preview_only_beautify(self, tmp_path: Path) -> None:
        cbz_path = _make_cbz(
            tmp_path / "Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f).cbz",
            num_pages=1,
        )

        result = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "--preview-only",
                "--no-kcc",
                "--beautify",
            ],
        )
        assert result.exit_code == 0, result.output
        expected_preview_dir = tmp_path / "Frieren - Beyond Journey's End v01 (2021)_preview"
        assert expected_preview_dir.is_dir()
        assert list(expected_preview_dir.glob("*.png"))


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
                    "import calibre.customize.ui; "
                    "from calibre_plugins.kfx_output.kfxlib.yj_book import YJ_Book; "
                    f"b = YJ_Book({str(out_kfx)!r}); b.decode_book(); "
                    "r = [f for f in b.fragments if f.ftype == '$164' and str(f.fid) == 'e0'][0]; "
                    "assert (r.value['$422'], r.value['$423']) == (1600, 2560), r.value"
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
        )


def test_cli_convert_and_preview_panels_merge_panels_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cbz_path = _make_cbz(tmp_path / "merge_flag.cbz", num_pages=1)
    captured_convert: list[ConversionConfig] = []
    captured_preview: list[DetectorConfig] = []

    def fake_run_single(self, config: ConversionConfig, **_kwargs) -> Path:
        captured_convert.append(config)
        return config.output_path

    def fake_preview(self, **kwargs):
        captured_preview.append(kwargs["detector_config"])
        return []

    monkeypatch.setattr(ConversionPipeline, "run_single", fake_run_single)
    monkeypatch.setattr(ConversionPipeline, "preview_panels", fake_preview)

    # convert defaults to merge_bridged_panels=True
    res1 = runner.invoke(
        app,
        ["convert", "-i", str(cbz_path), "-o", str(tmp_path / "out1.kfx"), "--no-kcc"],
    )
    assert res1.exit_code == 0, res1.output
    assert captured_convert[-1].detector.merge_bridged_panels is True

    # convert --preview-only --no-merge-panels sets merge_bridged_panels=False
    res2 = runner.invoke(
        app,
        [
            "convert",
            "-i",
            str(cbz_path),
            "--preview-only",
            "--no-kcc",
            "--no-merge-panels",
        ],
    )
    assert res2.exit_code == 0, res2.output
    assert captured_preview[-1].merge_bridged_panels is False

    # preview-panels defaults to True and accepts --no-merge-panels with --compare
    res3 = runner.invoke(
        app,
        [
            "preview-panels",
            "-i",
            str(cbz_path),
            "--output-dir",
            str(tmp_path / "p1"),
            "--no-kcc",
        ],
    )
    assert res3.exit_code == 0, res3.output
    assert captured_preview[-1].merge_bridged_panels is True

    res4 = runner.invoke(
        app,
        [
            "preview-panels",
            "-i",
            str(cbz_path),
            "--output-dir",
            str(tmp_path / "p2"),
            "--no-kcc",
            "--compare",
            "contour,onnx",
            "--no-merge-panels",
        ],
    )
    assert res4.exit_code == 0, res4.output
    assert captured_preview[-1].merge_bridged_panels is False


class TestCliReviewRegressionFixes:
    """Regression tests for CLI markup escaping, validation, preview-only, and detector output."""

    def test_doctor_preserves_extras_brackets_without_rich_stripping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        statuses = [
            ToolStatus(name="kcc-c2e", found=True, path=Path("/usr/bin/kcc-c2e")),
            ToolStatus(name="calibre-debug", found=True, path=Path("/usr/bin/calibre-debug")),
            ToolStatus(name="KFX Output", found=True, path=Path("/usr/bin/calibre-debug")),
            ToolStatus(
                name="onnxruntime",
                found=False,
                details="Optional (pip install manga2kindle[onnx])",
            ),
            ToolStatus(
                name="torch",
                found=False,
                details="Optional (pip install manga2kindle[magi])",
            ),
        ]
        monkeypatch.setattr(PlatformDiscovery, "run_doctor", lambda self, **_: statuses)

        result = runner.invoke(app, ["doctor"], env={"COLUMNS": "200"})

        assert result.exit_code == 0, result.output
        assert "pip install manga2kindle[onnx]" in result.output
        assert "pip install manga2kindle[magi]" in result.output

    def test_convert_preserves_lowercase_brackets_and_survives_closing_tag_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "[scans] Manga [b].cbz", num_pages=1)
        out_kfx = tmp_path / "out" / "[scans] Manga [b].kfx"
        TestConvertCliMocked._mock_calibre(tmp_path, monkeypatch, pdoc_valid=True)

        ok_res = runner.invoke(
            app,
            ["convert", "-i", str(cbz_path), "-o", str(out_kfx), "--no-kcc"],
            env={"COLUMNS": "240"},
        )
        assert ok_res.exit_code == 0, ok_res.output
        assert "[scans] Manga [b].kfx" in ok_res.output
        assert "Panel detector:" in ok_res.output

        def boom(*_args, **_kwargs) -> Path:
            raise RuntimeError("bad xml [/tag] and [/]")

        monkeypatch.setattr(ConversionPipeline, "run_single", boom)
        err_res = runner.invoke(
            app,
            ["convert", "-i", str(cbz_path), "-o", str(out_kfx), "--no-kcc"],
            env={"COLUMNS": "240"},
        )
        assert err_res.exit_code == 1
        assert "[/tag]" in err_res.output
        assert "[/]" in err_res.output

    def test_convert_rejects_removed_zoom_factor_long_flag(self, tmp_path: Path) -> None:
        cbz_path = _make_cbz(tmp_path / "no_zoom.cbz", num_pages=1)
        res = runner.invoke(app, ["convert", "-i", str(cbz_path), "--zoom-factor", "1.5"])
        assert res.exit_code != 0

    def test_convert_auto_fails_fast_on_missing_or_unknown_detector_model(
        self, tmp_path: Path
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "bad_model.cbz", num_pages=1)

        missing_onnx = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "--no-kcc",
                "--detector-model",
                str(tmp_path / "missing.onnx"),
            ],
        )
        assert missing_onnx.exit_code == 1
        assert "not found" in missing_onnx.output.lower()

        unknown_key = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "--no-kcc",
                "--detector-model",
                "totally-unknown-model",
            ],
        )
        assert unknown_key.exit_code == 1
        assert "unknown detector model" in unknown_key.output.lower()

        contour_mismatch = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "--no-kcc",
                "--detector",
                "contour",
                "--detector-model",
                "manga109-yolo-s",
            ],
        )
        assert contour_mismatch.exit_code == 1
        assert "contour" in contour_mismatch.output.lower()

    def test_preview_panels_compare_scopes_detector_model_and_announces_single_detector(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "compare_scoped.cbz", num_pages=1)

        # Single mode announces active panel detector
        single_res = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(tmp_path / "single_prev"),
                "--no-kcc",
                "--detector",
                "contour",
            ],
        )
        assert single_res.exit_code == 0, single_res.output
        assert "Panel detector: contour" in single_res.output

        # --detector-model targeting a backend outside --compare fails fast
        mismatch = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(tmp_path / "mismatch_prev"),
                "--no-kcc",
                "--compare",
                "contour,onnx",
                "--detector-model",
                "magiv2",
            ],
            env={"COLUMNS": "200"},
        )
        assert mismatch.exit_code == 1
        assert "not in --compare" in " ".join(mismatch.output.split())

        seen_configs: dict[DetectorKind, DetectorConfig] = {}

        class _StubBackend:
            def __init__(self, kind: DetectorKind) -> None:
                self.kind = kind

            def detect(self, image_path: Path) -> DetectionResult:
                with Image.open(image_path) as img:
                    w, h = img.size
                return DetectionResult(
                    panels=[PanelBox(x=20, y=20, w=w - 40, h=h - 40, ordinal=1)],
                    ordered=True,
                    backend=self.kind,
                    page_width=w,
                    page_height=h,
                )

            def detect_batch(self, image_paths):
                return [self.detect(p) for p in image_paths]

        def fake_resolve_auto_chain(cfg: DetectorConfig):
            seen_configs[cfg.kind] = cfg
            return [_StubBackend(cfg.kind)]

        monkeypatch.setattr("manga2kindle.cv.backends.resolve_auto_chain", fake_resolve_auto_chain)

        scoped_res = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(tmp_path / "scoped_prev"),
                "--no-kcc",
                "--compare",
                "contour,onnx,magi",
                "--allow-remote-code",
                "--detector-model",
                "magiv2",
            ],
        )
        assert scoped_res.exit_code == 0, scoped_res.output
        assert seen_configs[DetectorKind.MAGI].model_id == "magiv2"
        assert seen_configs[DetectorKind.ONNX_YOLO].model_id is None
        assert seen_configs[DetectorKind.ONNX_YOLO].model_path is None

    def test_convert_preview_only_rejects_beautify_source_and_honors_cover_size(
        self, tmp_path: Path
    ) -> None:
        cbz_path = _make_cbz(tmp_path / "Series v01 (Digital).cbz", num_pages=2)

        bad_combo = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "--preview-only",
                "--beautify-source",
                "--no-kcc",
            ],
        )
        assert bad_combo.exit_code == 1
        assert "--beautify-source" in bad_combo.output
        assert cbz_path.is_file()

        single_out = tmp_path / "single_cover_preview"
        single_ok = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "-o",
                str(single_out),
                "--preview-only",
                "--no-kcc",
                "--cover-size",
                "800x1200",
            ],
        )
        assert single_ok.exit_code == 0, single_ok.output
        with Image.open(single_out / "page_0001.png") as cover_img:
            assert cover_img.size == (800, 1200)

        batch_dir = tmp_path / "batch_cover_preview"
        batch_dir.mkdir()
        _make_cbz(batch_dir / "ch_01.cbz", num_pages=2)
        batch_out = tmp_path / "batch_cover_preview_out"
        batch_ok = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(batch_dir),
                "-o",
                str(batch_out),
                "--preview-only",
                "--no-kcc",
                "--cover-size",
                "800x1200",
            ],
        )
        assert batch_ok.exit_code == 0, batch_ok.output
        with Image.open(batch_out / "ch_01_preview" / "page_0001.png") as batch_cover:
            assert batch_cover.size == (800, 1200)


def test_cli_panel_merge_profile_calibration_and_e2e_post_sort_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cbz_path = _make_cbz(tmp_path / "mini_row.cbz", num_pages=1)
    captured_configs: list[ConversionConfig] = []
    captured_preview_merges: list[PanelMergeConfig] = []

    def fake_run_single(self, config: ConversionConfig, **_kwargs) -> Path:
        captured_configs.append(config)
        return config.output_path

    def fake_preview(self, **kwargs):
        captured_preview_merges.append(kwargs["panel_merge"])
        return []

    with monkeypatch.context() as m:
        m.setattr(ConversionPipeline, "run_single", fake_run_single)
        m.setattr(ConversionPipeline, "preview_panels", fake_preview)

        res_kpw6 = runner.invoke(
            app,
            ["convert", "-i", str(cbz_path), "-o", str(tmp_path / "kpw6.kfx"), "-p", "KPW6", "--no-kcc"],
        )
        assert res_kpw6.exit_code == 0, res_kpw6.output
        assert captured_configs[-1].panel_merge == PanelMergeConfig.for_profile(
            DeviceProfile.from_name("KPW6")
        )

        res_ks = runner.invoke(
            app,
            ["convert", "-i", str(cbz_path), "-o", str(tmp_path / "ks.kfx"), "-p", "KS", "--no-kcc"],
        )
        assert res_ks.exit_code == 0, res_ks.output
        assert captured_configs[-1].panel_merge == PanelMergeConfig.for_profile(
            DeviceProfile.from_name("KS")
        )

        res_disabled = runner.invoke(
            app,
            [
                "convert",
                "-i",
                str(cbz_path),
                "-o",
                str(tmp_path / "off.kfx"),
                "--no-kcc",
                "--no-merge-panels",
            ],
        )
        assert res_disabled.exit_code == 0, res_disabled.output
        assert captured_configs[-1].detector.merge_bridged_panels is False
        assert captured_configs[-1].panel_merge.enabled is False

        res_prev_ks = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(tmp_path / "prev_ks"),
                "-p",
                "KS",
                "--no-kcc",
            ],
        )
        assert res_prev_ks.exit_code == 0, res_prev_ks.output
        assert captured_preview_merges[-1] == PanelMergeConfig.for_profile(
            DeviceProfile.from_name("KS"), enabled=True
        )

        res_prev_off = runner.invoke(
            app,
            [
                "preview-panels",
                "-i",
                str(cbz_path),
                "--output-dir",
                str(tmp_path / "prev_off"),
                "--no-kcc",
                "--no-merge-panels",
            ],
        )
        assert res_prev_off.exit_code == 0, res_prev_off.output
        assert captured_preview_merges[-1].enabled is False

    # End-to-end pipeline verification: PanelMerger runs post-sort in preview_panels
    rendered_panels_seen: list[list[PanelBox]] = []

    class _MiniRowStubBackend:
        kind = DetectorKind.ONNX_YOLO

        def detect(self, image_path: Path) -> DetectionResult:
            with Image.open(image_path) as img:
                w, h = img.size
            # Top hero + compact 4-panel bottom row (24% height)
            hero = PanelBox(x=30, y=40, w=w - 60, h=int(h * 0.65), ordinal=0)
            row_y = int(h * 0.72)
            row_h = int(h * 0.24)
            col_w = (w - 150) // 4
            row = [
                PanelBox(x=30 + i * (col_w + 30), y=row_y, w=col_w, h=row_h, ordinal=0)
                for i in range(4)
            ]
            return DetectionResult(
                panels=[hero, *row],
                ordered=False,
                backend=self.kind,
                page_width=w,
                page_height=h,
            )

        def detect_batch(self, image_paths):
            return [self.detect(p) for p in image_paths]

    class _FakeRegistry:
        active_backend = _MiniRowStubBackend()

        def resolve(self):
            return [self.active_backend]

        def detect_and_sort(self, image_path, sorter, *, page_width=None, is_spread=False):
            res = self.active_backend.detect(image_path)
            return sorter.sort(res.panels, page_width=page_width, is_spread=is_spread)

    monkeypatch.setattr(
        "manga2kindle.pipeline.PanelPreviewRenderer.render",
        lambda self, image_path, panels, out_path: rendered_panels_seen.append(panels) or out_path,
    )

    pipeline = ConversionPipeline()
    pipeline.preview_panels(
        input_path=cbz_path,
        output_dir=tmp_path / "e2e_merged",
        no_kcc=True,
        detector_config=DetectorConfig(kind=DetectorKind.ONNX_YOLO, merge_bridged_panels=True, padding=0.0),
        registry=_FakeRegistry(),  # type: ignore[arg-type]
    )
    assert len(rendered_panels_seen[-1]) == 2

    pipeline.preview_panels(
        input_path=cbz_path,
        output_dir=tmp_path / "e2e_unmerged",
        no_kcc=True,
        detector_config=DetectorConfig(kind=DetectorKind.ONNX_YOLO, merge_bridged_panels=False, padding=0.0),
        registry=_FakeRegistry(),  # type: ignore[arg-type]
    )
    assert len(rendered_panels_seen[-1]) == 5
