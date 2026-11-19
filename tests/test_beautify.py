"""Tests for filename beautification and source renaming."""

from __future__ import annotations

from pathlib import Path

import pytest

from manga2kindle.beautify import beautify_name, beautify_source_file
from manga2kindle.compiler.kfx import KfxCompiler
from manga2kindle.models import ConversionConfig, DetectorConfig, DetectorKind, DeviceProfile
from manga2kindle.packaging.kpf import KpfBuilder
from manga2kindle.pipeline import ConversionPipeline
from manga2kindle.platform import PlatformDiscovery


@pytest.mark.parametrize(
    ("raw_stem", "expected"),
    [
        (
            "Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f)",
            "Frieren - Beyond Journey's End v01 (2021)",
        ),
        (
            "Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f2)",
            "Frieren - Beyond Journey's End v01 (2021)",
        ),
        (
            "Fullmetal Alchemist v01 (Grunbeld)",
            "Fullmetal Alchemist v01",
        ),
        (
            "Witch Hat Atelier v12 (2024) (Digital) (LuCaZ) (F)",
            "Witch Hat Atelier v12 (2024)",
        ),
        (
            "Kagurabachi v01 (2024)",
            "Kagurabachi v01 (2024)",
        ),
        (
            "Chainsaw Man - v01 (2019) [Digital]",
            "Chainsaw Man - v01 (2019)",
        ),
        (
            "One Piece - c1045 (2022) (Digital) [Group]",
            "One Piece - c1045 (2022)",
        ),
        (
            "Berserk v01 (1999) (Digital)",
            "Berserk v01 (1999)",
        ),
        (
            "Series Name - (Digital)",
            "Series Name",
        ),
        (
            "(Digital)",
            "(Digital)",
        ),
        (
            "   ",
            "   ",
        ),
    ],
)
def test_beautify_name(raw_stem: str, expected: str) -> None:
    assert beautify_name(raw_stem) == expected


def test_beautify_source_file_renames_in_place(tmp_path: Path) -> None:
    src = tmp_path / "Witch Hat Atelier v12 (2024) (Digital) (LuCaZ) (F).cbz"
    src.write_bytes(b"cbz-data")

    renamed = beautify_source_file(src)

    expected = tmp_path / "Witch Hat Atelier v12 (2024).cbz"
    assert renamed == expected
    assert expected.is_file()
    assert expected.read_bytes() == b"cbz-data"
    assert not src.exists()


def test_beautify_source_file_noop_when_already_clean(tmp_path: Path) -> None:
    src = tmp_path / "Kagurabachi v01 (2024).cbz"
    src.write_bytes(b"cbz-data")

    result = beautify_source_file(src)

    assert result == src
    assert src.is_file()


def test_beautify_source_file_raises_on_collision(tmp_path: Path) -> None:
    src = tmp_path / "Fullmetal Alchemist v01 (Grunbeld).cbz"
    existing = tmp_path / "Fullmetal Alchemist v01.cbz"
    src.write_bytes(b"new")
    existing.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="already exists"):
        beautify_source_file(src)

    assert src.is_file()
    assert existing.read_bytes() == b"old"


def test_beautify_source_file_ignores_directory(tmp_path: Path) -> None:
    folder = tmp_path / "Manga (Digital)"
    folder.mkdir()

    result = beautify_source_file(folder)

    assert result == folder
    assert folder.is_dir()


def _make_dummy_cbz(cbz_path: Path) -> Path:
    import zipfile
    from PIL import Image

    cbz_path.parent.mkdir(parents=True, exist_ok=True)
    img_path = cbz_path.parent / "_tmp_page.png"
    Image.new("RGB", (400, 600), (255, 255, 255)).save(img_path, format="PNG")
    with zipfile.ZipFile(cbz_path, "w") as zf:
        zf.write(img_path, arcname="001.png")
    img_path.unlink()
    return cbz_path


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, captured_titles: list[str]) -> ConversionPipeline:
    monkeypatch.setattr(
        PlatformDiscovery, "find_calibre", lambda self, override=None: Path("/usr/bin/calibre-debug")
    )
    orig_build = KpfBuilder.build

    def fake_build(self, *, pages, title, config, output_kpf, author=None):
        captured_titles.append(title)
        return orig_build(self, pages=pages, title=title, config=config, output_kpf=output_kpf, author=author)

    monkeypatch.setattr(KpfBuilder, "build", fake_build)
    monkeypatch.setattr(
        KfxCompiler,
        "compile",
        lambda self, kpf_path, output_kfx: output_kfx.write_bytes(b"KFX") and False or output_kfx,
    )
    monkeypatch.setattr(KfxCompiler, "verify_pdoc", lambda self, kfx_path: True)
    return ConversionPipeline()


def test_conversion_config_effective_beautify() -> None:
    profile = DeviceProfile.from_name("KPW5")
    base = ConversionConfig(input_path=Path("a.cbz"), output_path=Path("a.kfx"), profile=profile)
    assert base.beautify is False
    assert base.beautify_source is False
    assert base.effective_beautify is False

    only_out = base.model_copy(update={"beautify": True})
    assert only_out.effective_beautify is True

    only_src = base.model_copy(update={"beautify_source": True})
    assert only_src.effective_beautify is True


def test_pipeline_run_single_beautify_cleans_output_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cbz = _make_dummy_cbz(
        tmp_path / "Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f2).cbz"
    )
    out_dir = tmp_path / "out"
    captured_titles: list[str] = []
    pipeline = _stub_pipeline(monkeypatch, captured_titles)

    cfg = ConversionConfig(
        input_path=cbz,
        output_path=out_dir,
        profile=DeviceProfile.from_name("KPW5"),
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        beautify=True,
    )
    kfx_path = pipeline.run_single(cfg)

    assert kfx_path == out_dir / "Frieren - Beyond Journey's End v01 (2021).kfx"
    assert kfx_path.is_file()
    assert captured_titles == ["Frieren - Beyond Journey's End v01 (2021)"]
    assert cbz.is_file()


def test_pipeline_run_single_beautify_respects_explicit_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cbz = _make_dummy_cbz(tmp_path / "Fullmetal Alchemist v01 (Grunbeld).cbz")
    out_dir = tmp_path / "out"
    captured_titles: list[str] = []
    pipeline = _stub_pipeline(monkeypatch, captured_titles)

    cfg = ConversionConfig(
        input_path=cbz,
        output_path=out_dir,
        profile=DeviceProfile.from_name("KPW5"),
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        beautify=True,
    )
    kfx_path = pipeline.run_single(cfg, title="Custom Title")

    assert kfx_path == out_dir / "Fullmetal Alchemist v01.kfx"
    assert captured_titles == ["Custom Title"]


def test_pipeline_run_single_beautify_source_renames_input_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cbz = _make_dummy_cbz(tmp_path / "Witch Hat Atelier v12 (2024) (Digital) (LuCaZ) (F).cbz")
    out_dir = tmp_path / "out"
    captured_titles: list[str] = []
    pipeline = _stub_pipeline(monkeypatch, captured_titles)

    cfg = ConversionConfig(
        input_path=cbz,
        output_path=out_dir,
        profile=DeviceProfile.from_name("KPW5"),
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        beautify_source=True,
    )
    kfx_path = pipeline.run_single(cfg)

    expected_cbz = tmp_path / "Witch Hat Atelier v12 (2024).cbz"
    assert expected_cbz.is_file()
    assert not cbz.exists()
    assert kfx_path == out_dir / "Witch Hat Atelier v12 (2024).kfx"
    assert captured_titles == ["Witch Hat Atelier v12 (2024)"]


def test_pipeline_run_batch_beautify_detects_collision_before_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_dir = tmp_path / "batch"
    _make_dummy_cbz(in_dir / "Frieren v01 (2021) (Digital) (f).cbz")
    _make_dummy_cbz(in_dir / "Frieren v01 (2021) (Digital) (f2).cbz")
    captured_titles: list[str] = []
    pipeline = _stub_pipeline(monkeypatch, captured_titles)

    cfg = ConversionConfig(
        input_path=in_dir,
        output_path=in_dir,
        profile=DeviceProfile.from_name("KPW5"),
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        beautify_source=True,
    )
    with pytest.raises(ValueError, match="collision"):
        pipeline.run_batch(cfg)

    # Neither source file should have been renamed
    assert (in_dir / "Frieren v01 (2021) (Digital) (f).cbz").is_file()
    assert (in_dir / "Frieren v01 (2021) (Digital) (f2).cbz").is_file()


def test_pipeline_run_batch_beautify_and_beautify_source_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_dir = tmp_path / "batch"
    out_dir = tmp_path / "out"
    _make_dummy_cbz(in_dir / "Frieren - Beyond Journey's End v01 (2021) (Digital) (f).cbz")
    _make_dummy_cbz(in_dir / "Fullmetal Alchemist v01 (Grunbeld).cbz")
    captured_titles: list[str] = []
    pipeline = _stub_pipeline(monkeypatch, captured_titles)

    cfg = ConversionConfig(
        input_path=in_dir,
        output_path=out_dir,
        profile=DeviceProfile.from_name("KPW5"),
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        beautify=True,
        beautify_source=True,
    )
    outputs = pipeline.run_batch(cfg)

    assert sorted(p.name for p in outputs) == [
        "Frieren - Beyond Journey's End v01 (2021).kfx",
        "Fullmetal Alchemist v01.kfx",
    ]
    assert sorted(captured_titles) == [
        "Frieren - Beyond Journey's End v01 (2021)",
        "Fullmetal Alchemist v01",
    ]
    assert (in_dir / "Frieren - Beyond Journey's End v01 (2021).cbz").is_file()
    assert (in_dir / "Fullmetal Alchemist v01.cbz").is_file()


def test_pipeline_run_batch_beautify_output_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_dir = tmp_path / "batch"
    _make_dummy_cbz(in_dir / "Frieren v01 (2021) (Digital) (f).cbz")
    _make_dummy_cbz(in_dir / "Frieren v01 (2021) (Digital) (f2).cbz")
    captured_titles: list[str] = []
    pipeline = _stub_pipeline(monkeypatch, captured_titles)

    cfg = ConversionConfig(
        input_path=in_dir,
        output_path=in_dir,
        profile=DeviceProfile.from_name("KPW5"),
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        beautify=True,
    )
    with pytest.raises(ValueError, match="collision"):
        pipeline.run_batch(cfg)


def test_pipeline_preview_batch_beautify_and_collision(tmp_path: Path) -> None:
    in_dir = tmp_path / "batch_preview"
    out_dir = tmp_path / "preview_out"
    _make_dummy_cbz(in_dir / "Witch Hat Atelier v12 (2024) (Digital) (LuCaZ) (F).cbz")
    pipeline = ConversionPipeline()

    cfg = ConversionConfig(
        input_path=in_dir,
        output_path=out_dir,
        profile=DeviceProfile.from_name("KPW5"),
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        beautify=True,
    )
    preview_dirs = pipeline.preview_batch(cfg)
    assert preview_dirs == [out_dir / "Witch Hat Atelier v12 (2024)_preview"]
    assert preview_dirs[0].is_dir()

    _make_dummy_cbz(in_dir / "Witch Hat Atelier v12 (2024) (Digital) (f2).cbz")
    with pytest.raises(ValueError, match="collision"):
        pipeline.preview_batch(cfg)


def test_beautify_source_does_not_rename_when_preconditions_fail_or_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = DeviceProfile.from_name("KPW5")

    # 1. Missing calibre-debug leaves source file unrenamed in single mode
    cbz = _make_dummy_cbz(tmp_path / "Dandadan v01 (2022) (Digital) (f).cbz")
    monkeypatch.setattr(PlatformDiscovery, "find_calibre", lambda self, override=None: None)
    pipeline = ConversionPipeline()
    cfg_missing_calibre = ConversionConfig(
        input_path=cbz,
        output_path=tmp_path / "out",
        profile=profile,
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        beautify_source=True,
    )
    with pytest.raises(Exception, match="calibre-debug"):
        pipeline.run_single(cfg_missing_calibre)
    assert cbz.is_file()

    # 2. Failed detector resolution leaves source file unrenamed in single mode
    monkeypatch.setattr(
        PlatformDiscovery, "find_calibre", lambda self, override=None: Path("/usr/bin/calibre-debug")
    )
    cfg_bad_detector = ConversionConfig(
        input_path=cbz,
        output_path=tmp_path / "out",
        profile=profile,
        detector=DetectorConfig(kind=DetectorKind.MAGI, allow_remote_code=False),
        no_kcc=True,
        beautify_source=True,
    )
    with pytest.raises(RuntimeError, match="allow_remote_code"):
        pipeline.run_single(cfg_bad_detector)
    assert cbz.is_file()

    # 3. Unsupported extension leaves file unrenamed
    bad_ext = tmp_path / "Notes (Digital).txt"
    bad_ext.write_text("not an archive")
    cfg_bad_ext = ConversionConfig(
        input_path=bad_ext,
        output_path=tmp_path / "out" / "Notes.kfx",
        profile=profile,
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        beautify_source=True,
    )
    with pytest.raises(ValueError, match="Unsupported input format"):
        pipeline.run_single(cfg_bad_ext)
    assert bad_ext.is_file()

    # 4. skip_existing leaves source file unrenamed in single mode
    out_dir = tmp_path / "skip_out"
    out_dir.mkdir()
    existing_kfx = out_dir / "Dandadan v01 (2022).kfx"
    existing_kfx.write_bytes(b"EXISTING")
    cfg_skip_single = ConversionConfig(
        input_path=cbz,
        output_path=out_dir,
        profile=profile,
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        skip_existing=True,
        beautify_source=True,
    )
    assert pipeline.run_single(cfg_skip_single) == existing_kfx
    assert cbz.is_file()

    # 5. Batch mode with missing calibre or all skipped leaves source files unrenamed
    batch_dir = tmp_path / "batch_pre"
    b1 = _make_dummy_cbz(batch_dir / "Dandadan v01 (2022) (Digital) (f).cbz")
    b2 = _make_dummy_cbz(batch_dir / "Dandadan v02 (2022) (Digital) (f).cbz")

    monkeypatch.setattr(PlatformDiscovery, "find_calibre", lambda self, override=None: None)
    batch_cfg = ConversionConfig(
        input_path=batch_dir,
        output_path=tmp_path / "batch_out",
        profile=profile,
        detector=DetectorConfig(kind=DetectorKind.CONTOUR),
        no_kcc=True,
        beautify_source=True,
    )
    with pytest.raises(Exception, match="calibre-debug"):
        pipeline.run_batch(batch_cfg)
    assert b1.is_file()
    assert b2.is_file()

    batch_out = tmp_path / "batch_skip_out"
    batch_out.mkdir()
    (batch_out / "Dandadan v01 (2022).kfx").write_bytes(b"EXISTING_1")
    (batch_out / "Dandadan v02 (2022).kfx").write_bytes(b"EXISTING_2")
    monkeypatch.setattr(
        PlatformDiscovery, "find_calibre", lambda self, override=None: Path("/usr/bin/calibre-debug")
    )
    assert pipeline.run_batch(batch_cfg.model_copy(update={"output_path": batch_out, "skip_existing": True})) == []
    assert b1.is_file()
    assert b2.is_file()
