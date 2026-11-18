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
