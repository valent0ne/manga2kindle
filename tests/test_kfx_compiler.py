"""Tests for headless KFX compilation and structural PDOC verification."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from PIL import Image

import pytest

from manga2kindle.compiler import KfxCompilationError, KfxCompiler
from manga2kindle.models import (
    ConversionConfig,
    DeviceProfile,
    PageAsset,
    PanelBox,
    ReadingDirection,
)
from manga2kindle.packaging import KpfBuilder
from manga2kindle.platform import PlatformDiscovery


def _has_calibre_and_kfx_output() -> Path | None:
    """Return calibre-debug Path when both calibre-debug and KFX Output are available."""
    discovery = PlatformDiscovery()
    calibre_debug = discovery.find_calibre()
    if calibre_debug is None:
        return None
    for status in discovery.run_doctor(calibre_override=calibre_debug):
        if status.name == "KFX Output" and status.found:
            return calibre_debug
    return None


class TestKfxCompilerUnit:
    """Unit tests for KfxCompiler with mocked subprocess calls."""

    def test_kfx_compilation_error_is_runtime_error(self) -> None:
        assert issubclass(KfxCompilationError, RuntimeError)

    def test_compile_invokes_calibre_debug_with_pdoc_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calibre_debug = tmp_path / "calibre-debug"
        kpf_path = tmp_path / "まんが vol 1" / "book.kpf"
        kpf_path.parent.mkdir(parents=True)
        kpf_path.write_bytes(b"kpf-stub")
        output_kfx = tmp_path / "out dir" / "book.kfx"

        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            output_kfx.write_bytes(b"\xeaDRMION\x00compiled-kfx-data")
            return subprocess.CompletedProcess(cmd, 0, stdout="Converted", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        compiler = KfxCompiler(calibre_debug)
        result = compiler.compile(kpf_path, output_kfx)

        assert result == output_kfx
        assert output_kfx.is_file()
        assert captured["cmd"] == [
            str(calibre_debug),
            "-r",
            "KFX Output",
            "--",
            "-d",
            str(kpf_path),
            str(output_kfx),
        ]
        assert captured["kwargs"]["capture_output"] is True
        assert captured["kwargs"]["text"] is True
        assert captured["kwargs"]["encoding"] == "utf-8"
        assert captured["kwargs"]["check"] is False
        assert captured["kwargs"]["env"]["PYTHONIOENCODING"] == "utf-8"
        assert captured["kwargs"]["env"]["PYTHONUTF8"] == "1"

    def test_compile_rejects_stale_output_when_calibre_writes_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calibre_debug = tmp_path / "calibre-debug"
        kpf_path = tmp_path / "book.kpf"
        kpf_path.write_bytes(b"kpf")
        output_kfx = tmp_path / "stale.kfx"
        output_kfx.write_bytes(b"OLD_STALE_KFX")

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr=""),
        )

        compiler = KfxCompiler(calibre_debug)
        with pytest.raises(KfxCompilationError, match="missing"):
            compiler.compile(kpf_path, output_kfx)

    def test_compile_raises_on_nonzero_exit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calibre_debug = tmp_path / "calibre-debug"
        kpf_path = tmp_path / "book.kpf"
        kpf_path.write_bytes(b"kpf")
        output_kfx = tmp_path / "book.kfx"

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="KFX conversion failed")

        monkeypatch.setattr(subprocess, "run", fake_run)

        compiler = KfxCompiler(calibre_debug)
        with pytest.raises(KfxCompilationError, match="KFX conversion failed"):
            compiler.compile(kpf_path, output_kfx)

    def test_compile_raises_on_missing_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calibre_debug = tmp_path / "calibre-debug"
        kpf_path = tmp_path / "book.kpf"
        kpf_path.write_bytes(b"kpf")
        output_kfx = tmp_path / "missing.kfx"

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        compiler = KfxCompiler(calibre_debug)
        with pytest.raises(KfxCompilationError, match="missing|not created|empty"):
            compiler.compile(kpf_path, output_kfx)

    def test_compile_raises_on_empty_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calibre_debug = tmp_path / "calibre-debug"
        kpf_path = tmp_path / "book.kpf"
        kpf_path.write_bytes(b"kpf")
        output_kfx = tmp_path / "empty.kfx"

        def fake_run(cmd, **kwargs):
            output_kfx.write_bytes(b"")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        compiler = KfxCompiler(calibre_debug)
        with pytest.raises(KfxCompilationError, match="empty|0 bytes"):
            compiler.compile(kpf_path, output_kfx)

    def test_compile_raises_on_subprocess_oserror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calibre_debug = tmp_path / "calibre-debug"
        kpf_path = tmp_path / "book.kpf"
        kpf_path.write_bytes(b"kpf")
        output_kfx = tmp_path / "book.kfx"

        def fake_run(cmd, **kwargs):
            raise OSError("calibre-debug not executable")

        monkeypatch.setattr(subprocess, "run", fake_run)

        compiler = KfxCompiler(calibre_debug)
        with pytest.raises(KfxCompilationError, match="not executable"):
            compiler.compile(kpf_path, output_kfx)

    def test_verify_pdoc_true_only_for_pdoc_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calibre_debug = tmp_path / "calibre-debug"
        kfx_path = tmp_path / "book.kfx"
        kfx_path.write_bytes(b"binary-kfx-without-raw-pdoc-literal")

        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            payload = json.dumps({"cde_content_type": "PDOC"})
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"calibre warning\nM2K_META:{payload}\n",
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        compiler = KfxCompiler(calibre_debug)
        assert compiler.verify_pdoc(kfx_path) is True
        cmd = captured["cmd"]
        assert isinstance(cmd, list)
        assert cmd[0] == str(calibre_debug)
        assert cmd[1] == "-c"
        assert "YJ_Book" in cmd[2]
        assert "calibre.customize.ui" in cmd[2]
        assert cmd[3] == str(kfx_path)

    @pytest.mark.parametrize(
        ("returncode", "stdout"),
        [
            (0, 'M2K_META:{"cde_content_type": "EBOK"}\n'),
            (0, 'M2K_META:{"cde_content_type": null}\n'),
            (0, "M2K_META:{}\n"),
            (0, "unexpected output without metadata marker\n"),
            (0, "M2K_META:{invalid-json\n"),
            (1, "Traceback: corrupt KFX container"),
        ],
    )
    def test_verify_pdoc_returns_false_for_non_pdoc_or_corrupt_kfx(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        returncode: int,
        stdout: str,
    ) -> None:
        calibre_debug = tmp_path / "calibre-debug"
        kfx_path = tmp_path / "book.kfx"
        # Intentionally include b"PDOC" in raw bytes to ensure no naive substring grep is used.
        kfx_path.write_bytes(b"fake-kfx-header-PDOC-trap")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        compiler = KfxCompiler(calibre_debug)
        assert compiler.verify_pdoc(kfx_path) is False

    def test_verify_pdoc_returns_false_when_file_missing_or_empty(
        self,
        tmp_path: Path,
    ) -> None:
        compiler = KfxCompiler(tmp_path / "calibre-debug")
        assert compiler.verify_pdoc(tmp_path / "does_not_exist.kfx") is False

        empty_kfx = tmp_path / "empty.kfx"
        empty_kfx.write_bytes(b"")
        assert compiler.verify_pdoc(empty_kfx) is False


class TestKfxCompilerIntegration:
    """Live Calibre KFX Output integration tests."""

    def test_compile_two_page_kpf_and_verify_pdoc(self, tmp_path: Path) -> None:
        calibre_debug = _has_calibre_and_kfx_output()
        if calibre_debug is None:
            pytest.skip("calibre-debug or KFX Output plugin not installed")

        img1 = tmp_path / "page_001.png"
        img2 = tmp_path / "page_002.png"
        Image.new("RGB", (1264, 1680), (255, 255, 255)).save(img1, "PNG")
        Image.new("RGB", (1264, 1680), (240, 240, 240)).save(img2, "PNG")

        pages = [
            PageAsset(
                index=0,
                source_path=img1,
                processed_path=img1,
                width=1264,
                height=1680,
                is_cover=True,
                panels=[
                    PanelBox(x=632, y=50, w=580, h=750, ordinal=1),
                    PanelBox(x=50, y=50, w=550, h=750, ordinal=2),
                ],
            ),
            PageAsset(
                index=1,
                source_path=img2,
                processed_path=img2,
                width=1264,
                height=1680,
                is_cover=False,
                panels=[
                    PanelBox(x=632, y=100, w=580, h=700, ordinal=1),
                    PanelBox(x=50, y=850, w=550, h=700, ordinal=2),
                ],
            ),
        ]

        kpf_path = tmp_path / "sample.kpf"
        kfx_path = tmp_path / "sample.kfx"
        config = ConversionConfig(
            input_path=tmp_path / "input.cbz",
            output_path=kfx_path,
            profile=DeviceProfile.from_name("KPW5"),
            direction=ReadingDirection.RTL,
        )

        KpfBuilder().build(
            pages=pages,
            title="Integration Test Manga",
            config=config,
            output_kpf=kpf_path,
            author="Test Author",
            language="en",
        )

        compiler = KfxCompiler(calibre_debug)
        compiled = compiler.compile(kpf_path, kfx_path)

        assert compiled == kfx_path
        assert kfx_path.is_file()
        assert kfx_path.stat().st_size > 0
        assert compiler.verify_pdoc(kfx_path) is True

        # Verify KFX Output preserves Kindle Create Guided View hierarchy ($439 targets, $157 styles)
        raw_kfx = kfx_path.read_bytes()
        assert b"yj_publisher_panels" in raw_kfx
        inspect_res = subprocess.run(
            [
                str(calibre_debug),
                "-c",
                (
                    "import calibre.customize.ui; "
                    "from calibre_plugins.kfx_output.kfxlib.yj_book import YJ_Book; "
                    f"b = YJ_Book({str(kfx_path)!r}); b.decode_book(); b.check_consistency(); "
                    "st = [f for f in b.fragments if f.ftype == '$157']; "
                    "sl = [f for f in b.fragments if f.ftype == '$259']; "
                    "tgts = [c for f in sl for top in f.value['$146'] for c in top['$146'] if c.get('$159') == '$439']; "
                    "assert len(st) == 2, len(st); "
                    "assert len(tgts) == 4, len(tgts); "
                    "print('GUIDED_VIEW_OK', len(st), len(tgts))"
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "GUIDED_VIEW_OK 2 4" in inspect_res.stdout

        corrupt_kfx = tmp_path / "corrupt.kfx"
        corrupt_kfx.write_bytes(b"not-a-real-kfx-PDOC")
        assert compiler.verify_pdoc(corrupt_kfx) is False
