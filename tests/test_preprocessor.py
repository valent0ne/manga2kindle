"""Tests for ImagePreprocessor, KccBackend, and PillowFallbackBackend."""

from __future__ import annotations

import io
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

from manga2kindle.models import (
    ConversionConfig,
    DeviceProfile,
    PageAsset,
    ReadingDirection,
    SpreadMode,
)
from manga2kindle.platform import PlatformDiscovery
from manga2kindle.preprocessor import (
    ImagePreprocessingError,
    ImagePreprocessor,
    KccBackend,
    PillowFallbackBackend,
)


def _make_config(
    tmp_path: Path,
    *,
    profile: DeviceProfile | None = None,
    direction: ReadingDirection = ReadingDirection.RTL,
    spread_mode: SpreadMode = SpreadMode.SPLIT,
    no_kcc: bool = False,
    kcc_path: Path | None = None,
) -> ConversionConfig:
    return ConversionConfig(
        input_path=tmp_path / "input.cbz",
        output_path=tmp_path / "output.kfx",
        profile=profile or DeviceProfile.from_name("KPW5"),
        direction=direction,
        spread_mode=spread_mode,
        no_kcc=no_kcc,
        kcc_path=kcc_path,
    )


def _save_image(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _make_jpg_bytes(width: int, height: int, color: int = 200) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def test_pillow_fallback_autocrop_gamma_quantize_and_canvas(tmp_path: Path) -> None:
    # 200x300 white image with a 100x200 dark box at (50, 50, 150, 250) and a mid-gray patch.
    # Auto-crop with 5px safety margin -> 110x210 crop.
    # Scaling 110x210 to fit 1264x1680 gives height=1680, width=round(110 * 1680 / 210) = 880.
    img1 = Image.new("L", (200, 300), color=255)
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([50, 50, 149, 249], fill=20)
    draw1.rectangle([75, 100, 124, 199], fill=128)
    src1 = _save_image(img1, tmp_path / "extracted" / "0000_p1.png")

    # Second page: RGBA with transparent border that should stay white (255), not black.
    img2 = Image.new("RGBA", (300, 400), color=(255, 255, 255, 0))
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([40, 40, 259, 359], fill=(100, 100, 100, 255))
    src2 = _save_image(img2, tmp_path / "extracted" / "0001_p2.png")

    pages = [
        PageAsset(index=0, source_path=src1, width=200, height=300, is_cover=True),
        PageAsset(index=1, source_path=src2, width=300, height=400, is_cover=False),
    ]
    config = _make_config(tmp_path, no_kcc=True)
    work_dir = tmp_path / "work"

    result = PillowFallbackBackend().process(pages, work_dir, config)

    assert len(result) == 2
    assert [p.index for p in result] == [0, 1]
    assert [p.is_cover for p in result] == [True, False]

    p0 = result[0]
    assert p0.processed_path is not None and p0.processed_path.is_file()
    assert (p0.width, p0.height) == (1264, 1680)

    with Image.open(p0.processed_path) as out0:
        assert out0.mode == "L"
        assert out0.size == (1264, 1680)
        # White padding on left & right margins outside centered 880px width: (1264-880)/2 = 192px
        assert out0.getpixel((10, 840)) == 255
        assert out0.getpixel((1250, 840)) == 255
        # Verify 5px white safety margin inside the 880px scaled content:
        # 5px at 8x scale = 40px inside 192 -> x=200 is white (255), x=250 is dark!
        assert out0.getpixel((205, 840)) == 255
        assert out0.getpixel((260, 840)) < 50
        # Center mid-gray (128) darkened by gamma 1.6 -> ~84, quantized to 16-level E-Ink (85)
        center_val = out0.getpixel((632, 840))
        assert 68 <= center_val <= 102
        # All pixel values belong to the 16-level E-Ink grayscale palette {0, 17, ..., 255}
        unique_values = set(out0.get_flattened_data())
        assert len(unique_values) <= 16
        assert unique_values.issubset({i * 17 for i in range(16)})
        # Saving as JPEG must succeed cleanly because mode is "L" (not "P")
        out0.save(tmp_path / "verify_jpeg.jpg", format="JPEG")

    p1 = result[1]
    assert p1.processed_path is not None and p1.processed_path.is_file()
    assert (p1.width, p1.height) == (1264, 1680)
    with Image.open(p1.processed_path) as out1:
        assert out1.mode == "L"
        assert out1.getpixel((5, 5)) == 255


def test_pillow_fallback_spread_split_rtl_and_ltr(tmp_path: Path) -> None:
    # 2000x1000 spread: left half dark (34), right half medium-light (187 -> ~153 after gamma)
    spread = Image.new("L", (2000, 1000), color=34)
    draw = ImageDraw.Draw(spread)
    draw.rectangle([1000, 0, 1999, 999], fill=200)
    src = _save_image(spread, tmp_path / "extracted" / "0000_spread.png")

    pages = [
        PageAsset(
            index=0,
            source_path=src,
            width=2000,
            height=1000,
            is_cover=True,
            is_spread=True,
        )
    ]

    # RTL: right half first, left half second
    rtl_cfg = _make_config(
        tmp_path,
        direction=ReadingDirection.RTL,
        spread_mode=SpreadMode.SPLIT,
        no_kcc=True,
    )
    rtl_pages = PillowFallbackBackend().process(pages, tmp_path / "work_rtl", rtl_cfg)
    assert len(rtl_pages) == 2
    assert [p.index for p in rtl_pages] == [0, 1]
    assert [p.is_cover for p in rtl_pages] == [True, False]
    assert [p.is_spread for p in rtl_pages] == [False, False]
    assert [(p.width, p.height) for p in rtl_pages] == [(1264, 1680), (1264, 1680)]

    with Image.open(rtl_pages[0].processed_path) as first, Image.open(
        rtl_pages[1].processed_path
    ) as second:
        # Center of first (right half, original 200) is much lighter than center of second (left half, 34)
        assert first.getpixel((632, 840)) > 120
        assert second.getpixel((632, 840)) < 40

    # LTR: left half first, right half second
    ltr_cfg = _make_config(
        tmp_path,
        direction=ReadingDirection.LTR,
        spread_mode=SpreadMode.SPLIT,
        no_kcc=True,
    )
    ltr_pages = PillowFallbackBackend().process(pages, tmp_path / "work_ltr", ltr_cfg)
    assert len(ltr_pages) == 2
    with Image.open(ltr_pages[0].processed_path) as first, Image.open(
        ltr_pages[1].processed_path
    ) as second:
        assert first.getpixel((632, 840)) < 40
        assert second.getpixel((632, 840)) > 120


def test_pillow_fallback_spread_rotate_and_keep(tmp_path: Path) -> None:
    # 2000x1000 spread with gray background (170) and a black marker in the top-right quadrant.
    # Rotating 90 deg CCW moves the top-right quadrant to the top-left quadrant.
    spread = Image.new("L", (2000, 1000), color=170)
    draw = ImageDraw.Draw(spread)
    draw.rectangle([1500, 0, 1999, 499], fill=0)
    src = _save_image(spread, tmp_path / "extracted" / "0000_spread.png")

    pages = [
        PageAsset(
            index=0,
            source_path=src,
            width=2000,
            height=1000,
            is_cover=True,
            is_spread=True,
        )
    ]

    # ROTATE: 2000x1000 -> 1000x2000 -> scaled to 840x1680 centered horizontally on 1264x1680
    rotate_cfg = _make_config(tmp_path, spread_mode=SpreadMode.ROTATE, no_kcc=True)
    rotated = PillowFallbackBackend().process(pages, tmp_path / "work_rot", rotate_cfg)
    assert len(rotated) == 1
    assert (rotated[0].width, rotated[0].height) == (1264, 1680)
    with Image.open(rotated[0].processed_path) as rot_img:
        assert rot_img.size == (1264, 1680)
        # Left/right white padding outside 840px width ((1264-840)/2 = 212)
        assert rot_img.getpixel((50, 840)) == 255
        assert rot_img.getpixel((1210, 840)) == 255
        # Top-left of content (x ~ 320, y ~ 200) holds the black marker (rotated 90 CCW)
        assert rot_img.getpixel((320, 200)) == 0
        # Bottom-left of content holds gray (170 -> ~136 after gamma)
        assert rot_img.getpixel((320, 1400)) > 100

    # KEEP: 2000x1000 -> fitted cleanly within landscape (2*1264, 1680) -> (2528, 1264)
    keep_cfg = _make_config(tmp_path, spread_mode=SpreadMode.KEEP, no_kcc=True)
    kept = PillowFallbackBackend().process(pages, tmp_path / "work_keep", keep_cfg)
    assert len(kept) == 1
    assert kept[0].is_spread is True
    assert kept[0].width > kept[0].height
    assert (kept[0].width, kept[0].height) == (2528, 1264)
    with Image.open(kept[0].processed_path) as keep_img:
        assert keep_img.size == (2528, 1264)
        # Left side holds darkened gray (~136), top-right quadrant holds black marker (0)
        assert keep_img.getpixel((200, 600)) > 100
        assert keep_img.getpixel((2200, 200)) == 0


def test_pillow_fallback_portrait_wide_banner_not_erroneously_split(tmp_path: Path) -> None:
    """Portrait pages with wide artwork surrounded by white space must not split in SPLIT mode."""
    portrait = Image.new("L", (1000, 1600), color=255)
    draw = ImageDraw.Draw(portrait)
    draw.rectangle([50, 600, 950, 1000], fill=40)
    src = _save_image(portrait, tmp_path / "extracted" / "0000_portrait.png")

    pages = [
        PageAsset(
            index=0,
            source_path=src,
            width=1000,
            height=1600,
            is_cover=True,
            is_spread=False,
        )
    ]
    cfg = _make_config(tmp_path, spread_mode=SpreadMode.SPLIT, no_kcc=True)
    result = PillowFallbackBackend().process(pages, tmp_path / "work_banner", cfg)
    assert len(result) == 1
    assert (result[0].width, result[0].height) == (1264, 1680)
    assert result[0].is_spread is False


def test_kcc_backend_stages_pages_builds_command_and_unpacks_cbz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "src_mixed"
    src_dir.mkdir()
    # Put junk alongside source images to prove KccBackend stages ONLY verified pages into kcc_in
    (src_dir / "corrupt.jpg").write_bytes(b"not an image")
    (src_dir / "notes.txt").write_text("junk")

    p1 = _save_image(Image.new("L", (800, 1200), 100), src_dir / "01.png")
    p2 = _save_image(Image.new("L", (800, 1200), 150), src_dir / "02.png")
    pages = [
        PageAsset(index=0, source_path=p1, width=800, height=1200, is_cover=True),
        PageAsset(index=1, source_path=p2, width=800, height=1200, is_cover=False),
    ]

    fake_kcc = tmp_path / "bin" / "kcc-c2e"
    fake_kcc.parent.mkdir()
    fake_kcc.write_text("#!/bin/sh\n")

    discovery = MagicMock(spec=PlatformDiscovery)
    discovery.find_kcc.return_value = fake_kcc

    captured_cmds: list[list[str]] = []
    captured_envs: list[dict[str, str]] = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        captured_envs.append(kwargs.get("env", {}))
        out_dir = Path(cmd[cmd.index("-o") + 1])
        staging_dir = Path(cmd[-1])
        # Verify staging_dir only contains the two staged images (no junk!)
        staged_files = sorted(f.name for f in staging_dir.iterdir())
        assert len(staged_files) == 2
        assert all(name.endswith(".png") for name in staged_files)

        cbz_out = out_dir / "kcc_in.cbz"
        with zipfile.ZipFile(cbz_out, "w") as zf:
            zf.writestr("__MACOSX/._page_1.jpg", b"junk")
            zf.writestr("page_10.jpg", _make_jpg_bytes(1264, 1680))
            zf.writestr("page_2.jpg", _make_jpg_bytes(1000, 1680))
            zf.writestr("page_1.jpg", _make_jpg_bytes(1264, 1500))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("manga2kindle.preprocessor.subprocess.run", fake_run)

    config = _make_config(
        tmp_path,
        direction=ReadingDirection.RTL,
        spread_mode=SpreadMode.SPLIT,
        no_kcc=False,
        kcc_path=fake_kcc,
    )
    work_dir = tmp_path / "work"
    result = ImagePreprocessor(discovery=discovery).process(pages, work_dir, config)

    discovery.find_kcc.assert_called_once_with(override=fake_kcc)
    assert len(captured_cmds) == 1
    expected_tmp = str(work_dir / "kcc_tmp")
    assert captured_envs[0]["TMPDIR"] == expected_tmp
    assert captured_envs[0]["TEMP"] == expected_tmp
    assert captured_envs[0]["TMP"] == expected_tmp
    assert captured_envs[0]["PYTHONIOENCODING"] == "utf-8"
    assert captured_envs[0]["PYTHONUTF8"] == "1"
    for thread_var in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert captured_envs[0][thread_var] == "1"
    cmd = captured_cmds[0]
    assert cmd == [
        str(fake_kcc),
        "-p",
        "KPW5",
        "-m",
        "-q",
        "-u",
        "-r",
        "0",
        "-f",
        "CBZ",
        "--mozjpeg",
        "-o",
        str(work_dir / "kcc_out"),
        str(work_dir / "kcc_in"),
    ]

    assert len(result) == 3
    assert [p.index for p in result] == [0, 1, 2]
    assert [p.is_cover for p in result] == [True, False, False]
    # Natural order: cover page_1 normalized to 1600x2560, interior page_2 (1000x1680), page_10 (1264x1680)
    assert [(p.width, p.height) for p in result] == [
        (1600, 2560),
        (1000, 1680),
        (1264, 1680),
    ]
    assert all(
        p.processed_path is not None
        and p.processed_path.is_file()
        and p.processed_path.parent == work_dir / "processed"
        for p in result
    )


def test_kcc_backend_ltr_spread_modes_and_custom_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p1 = _save_image(Image.new("L", (800, 1200), 100), tmp_path / "01.png")
    pages = [PageAsset(index=0, source_path=p1, width=800, height=1200, is_cover=True)]

    fake_kcc = tmp_path / "kcc-c2e"
    fake_kcc.write_text("")
    discovery = MagicMock(spec=PlatformDiscovery)
    discovery.find_kcc.return_value = fake_kcc

    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        out_dir = Path(cmd[cmd.index("-o") + 1])
        with zipfile.ZipFile(out_dir / "out.cbz", "w") as zf:
            zf.writestr("0001.jpg", _make_jpg_bytes(1400, 1800))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("manga2kindle.preprocessor.subprocess.run", fake_run)

    custom_profile = DeviceProfile.from_name("CUSTOM", custom_size=(1400, 1800))
    for spread_mode, expected_r, expect_norotate in [
        (SpreadMode.ROTATE, "1", False),
        (SpreadMode.KEEP, "1", True),
    ]:
        captured_cmds.clear()
        cfg = _make_config(
            tmp_path,
            profile=custom_profile,
            direction=ReadingDirection.LTR,
            spread_mode=spread_mode,
            no_kcc=False,
        )
        ImagePreprocessor(discovery=discovery).process(
            pages, tmp_path / f"work_{spread_mode.value}", cfg
        )
        cmd = captured_cmds[0]
        assert "-m" not in cmd
        assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "OTHER"
        assert cmd[cmd.index("-r") + 1] == expected_r
        assert ("--norotate" in cmd) is expect_norotate
        assert cmd[cmd.index("--customwidth") + 1] == "1400"
        assert cmd[cmd.index("--customheight") + 1] == "1800"


def test_kcc_backend_raises_without_silent_fallback_when_missing_or_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p1 = _save_image(Image.new("L", (800, 1200), 100), tmp_path / "01.png")
    pages = [PageAsset(index=0, source_path=p1, width=800, height=1200, is_cover=True)]
    config = _make_config(tmp_path, no_kcc=False)

    # Case 1: kcc-c2e missing
    missing_discovery = MagicMock(spec=PlatformDiscovery)
    missing_discovery.find_kcc.return_value = None
    with pytest.raises(ImagePreprocessingError, match="kcc-c2e"):
        ImagePreprocessor(discovery=missing_discovery).process(
            pages, tmp_path / "w1", config
        )
    assert not (tmp_path / "w1" / "processed").exists() or not list(
        (tmp_path / "w1" / "processed").iterdir()
    )

    # Case 2: kcc-c2e exits non-zero
    fake_kcc = tmp_path / "kcc-c2e"
    fake_kcc.write_text("")
    ok_discovery = MagicMock(spec=PlatformDiscovery)
    ok_discovery.find_kcc.return_value = fake_kcc

    monkeypatch.setattr(
        "manga2kindle.preprocessor.subprocess.run",
        lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(ImagePreprocessingError, match="boom"):
        ImagePreprocessor(discovery=ok_discovery).process(pages, tmp_path / "w2", config)

    # Case 3: kcc-c2e exits 0 but emits no CBZ
    monkeypatch.setattr(
        "manga2kindle.preprocessor.subprocess.run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    with pytest.raises(ImagePreprocessingError, match="no CBZ"):
        ImagePreprocessor(discovery=ok_discovery).process(pages, tmp_path / "w3", config)


def test_image_preprocessor_stretches_cover_to_1600x2560(tmp_path: Path) -> None:
    """Cover page is stretched edge-to-edge to config.cover_size (1600x2560) with no borders."""
    cover_img = Image.new("RGB", (1131, 1696), color=(20, 90, 180))
    cover_src = _save_image(cover_img, tmp_path / "extracted" / "0000_cover.png")
    page_img = Image.new("L", (800, 1200), color=180)
    page_src = _save_image(page_img, tmp_path / "extracted" / "0001_page.png")

    pages = [
        PageAsset(index=0, source_path=cover_src, width=1131, height=1696, is_cover=True),
        PageAsset(index=1, source_path=page_src, width=800, height=1200, is_cover=False),
    ]
    config = _make_config(tmp_path, no_kcc=True)
    result = ImagePreprocessor().process(pages, tmp_path / "work_cover", config)

    assert len(result) == 2
    assert (result[0].width, result[0].height) == (1600, 2560)
    assert result[0].is_cover is True
    assert (result[1].width, result[1].height) == (1264, 1680)
    assert result[1].is_cover is False

    # Stretched edge-to-edge: all edges/corners match cover color with zero white borders.
    with Image.open(result[0].processed_path) as out_cover:
        assert out_cover.size == (1600, 2560)
        assert out_cover.mode == "RGB"
        assert out_cover.getpixel((10, 10)) == (20, 90, 180)
        assert out_cover.getpixel((1590, 2550)) == (20, 90, 180)
        assert out_cover.getpixel((800, 1280)) == (20, 90, 180)


def test_kcc_backend_resolve_max_concurrent_and_semaphore_limits_parallel_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert KccBackend.resolve_max_concurrent(available_mem_gib=16.0, cpu_count=16) == 2
    assert KccBackend.resolve_max_concurrent(available_mem_gib=6.0, cpu_count=16) == 1
    assert KccBackend.resolve_max_concurrent(available_mem_gib=16.0, cpu_count=2) == 1
    assert KccBackend.resolve_max_concurrent(override=3, available_mem_gib=2.0, cpu_count=2) == 3

    p1 = _save_image(Image.new("L", (800, 1200), 100), tmp_path / "src" / "01.png")
    pages = [PageAsset(index=0, source_path=p1, width=800, height=1200, is_cover=True)]

    fake_kcc = tmp_path / "kcc-c2e"
    fake_kcc.write_text("#!/bin/sh\n")
    discovery = MagicMock(spec=PlatformDiscovery)
    discovery.find_kcc.return_value = fake_kcc

    lock = threading.Lock()
    active = 0
    peak_active = 0

    def fake_run(cmd, **kwargs):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            time.sleep(0.05)
            out_dir = Path(cmd[cmd.index("-o") + 1])
            with zipfile.ZipFile(out_dir / "out.cbz", "w") as zf:
                zf.writestr("0001.jpg", _make_jpg_bytes(1264, 1680))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("manga2kindle.preprocessor.subprocess.run", fake_run)

    backend = KccBackend(discovery=discovery, max_concurrent=2)
    config = _make_config(tmp_path, no_kcc=False, kcc_path=fake_kcc)

    def _run_one(idx: int) -> list[PageAsset]:
        return backend.process(pages, tmp_path / f"work_{idx}", config)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_run_one, range(6)))

    assert len(results) == 6
    assert all(len(res) == 1 and res[0].processed_path is not None for res in results)
    assert peak_active <= 2
    assert peak_active >= 1
