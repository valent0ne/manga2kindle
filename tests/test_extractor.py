"""Tests for ArchiveExtractor and natural_sort_key."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import rarfile
from PIL import Image

from manga2kindle.extractor import ArchiveExtractor, natural_sort_key
from manga2kindle.models import PageAsset
from manga2kindle.platform import PlatformDiscovery


def _make_png_bytes(width: int = 100, height: int = 150, color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _write_png(path: Path, width: int = 100, height: int = 150) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_make_png_bytes(width, height))
    return path


def test_natural_sort_key_orders_pages_and_nested_chapters():
    names = ["page_10.png", "page_2.png", "page_1.png"]
    assert sorted(names, key=natural_sort_key) == [
        "page_1.png",
        "page_2.png",
        "page_10.png",
    ]

    nested = [
        "ch_10/page_1.png",
        "ch_2/page_10.png",
        "ch_2\\page_2.png",
        "ch_1/page_1.png",
    ]
    assert sorted(nested, key=natural_sort_key) == [
        "ch_1/page_1.png",
        "ch_2\\page_2.png",
        "ch_2/page_10.png",
        "ch_10/page_1.png",
    ]

    mixed = ["cover.png", "02.png", "01.png", "page_10.png", "page_2.png"]
    assert sorted(mixed, key=natural_sort_key) == [
        "01.png",
        "02.png",
        "cover.png",
        "page_2.png",
        "page_10.png",
    ]


def test_extract_cbz_sorts_naturally_and_skips_junk_entries(tmp_path: Path):
    cbz_path = tmp_path / "manga.cbz"
    work_dir = tmp_path / "work"

    with zipfile.ZipFile(cbz_path, "w") as zf:
        zf.writestr("__MACOSX/._page_1.png", b"appledouble")
        zf.writestr("__MACOSX/page_1.png", _make_png_bytes(100, 150))
        zf.writestr(".DS_Store", b"dsstore")
        zf.writestr("Thumbs.db", b"thumbs")
        zf.writestr("ComicInfo.xml", b"<ComicInfo/>")
        zf.writestr(".hidden.png", _make_png_bytes(100, 150))
        zf.writestr(".hidden_dir/page_0.png", _make_png_bytes(100, 150))
        zf.writestr("readme.txt", b"hello")
        zf.writestr("page_10.png", _make_png_bytes(120, 180))
        zf.writestr("page_2.png", _make_png_bytes(300, 150))  # spread
        zf.writestr("page_1.png", _make_png_bytes(100, 150))  # cover

    pages = ArchiveExtractor().extract(cbz_path, work_dir)

    assert len(pages) == 3
    assert all(isinstance(p, PageAsset) for p in pages)
    assert [p.index for p in pages] == [0, 1, 2]
    assert [p.is_cover for p in pages] == [True, False, False]
    assert [p.is_spread for p in pages] == [False, True, False]
    assert [(p.width, p.height) for p in pages] == [(100, 150), (300, 150), (120, 180)]

    extracted_dir = work_dir / "extracted"
    assert all(p.source_path.parent == extracted_dir for p in pages)
    assert all(p.source_path.is_file() for p in pages)
    staged_names = [p.source_path.name for p in pages]
    assert staged_names == sorted(staged_names)
    assert len(set(staged_names)) == 3
    # Verify only clean image files exist in extracted_dir
    assert sorted(f.name for f in extracted_dir.iterdir()) == staged_names


def test_extract_cbz_unique_filenames_for_duplicate_nested_leaves(tmp_path: Path):
    cbz_path = tmp_path / "multi_ch.cbz"
    work_dir = tmp_path / "work"

    with zipfile.ZipFile(cbz_path, "w") as zf:
        zf.writestr("ch_10/01.png", _make_png_bytes(100, 160))
        zf.writestr("ch_2/01.png", _make_png_bytes(100, 150))
        zf.writestr("ch_1/01.png", _make_png_bytes(100, 140))

    pages = ArchiveExtractor().extract(cbz_path, work_dir)

    assert len(pages) == 3
    assert [(p.width, p.height) for p in pages] == [(100, 140), (100, 150), (100, 160)]
    assert len({p.source_path for p in pages}) == 3


@pytest.mark.parametrize(
    "malicious_name",
    [
        "../evil.png",
        "nested/../../evil.png",
        "/abs/evil.png",
        "..\\evil.png",
        "C:\\evil.png",
    ],
)
def test_extract_cbz_rejects_zip_slip_traversal(tmp_path: Path, malicious_name: str):
    cbz_path = tmp_path / "evil.cbz"
    work_dir = tmp_path / "work"

    with zipfile.ZipFile(cbz_path, "w") as zf:
        zf.writestr("page_1.png", _make_png_bytes())
        zf.writestr(malicious_name, _make_png_bytes())

    with pytest.raises(ValueError, match="traversal|unsafe|path"):
        ArchiveExtractor().extract(cbz_path, work_dir)

    assert not (tmp_path / "evil.png").exists()


def test_extract_folder_of_images_stages_clean_ordered_directory(tmp_path: Path):
    src_dir = tmp_path / "raw_manga"
    _write_png(src_dir / "ch_2" / "page_10.png", 120, 180)
    _write_png(src_dir / "ch_1" / "page_1.png", 100, 150)
    _write_png(src_dir / "ch_1" / "page_2.png", 320, 160)
    (src_dir / ".DS_Store").write_bytes(b"junk")
    (src_dir / "ComicInfo.xml").write_text("<ComicInfo/>")
    (src_dir / "Thumbs.db").write_bytes(b"junk")
    _write_png(src_dir / "__MACOSX" / "ignored.png", 10, 10)

    work_dir = tmp_path / "work"
    pages = ArchiveExtractor().extract(src_dir, work_dir)

    assert len(pages) == 3
    assert [(p.width, p.height) for p in pages] == [(100, 150), (320, 160), (120, 180)]
    assert pages[0].is_cover is True
    assert pages[1].is_cover is False
    assert pages[1].is_spread is True
    assert all(p.source_path.parent == work_dir / "extracted" for p in pages)
    assert (src_dir / "ch_1" / "page_1.png").exists()  # source dir untouched


def test_extract_raises_value_error_on_corrupted_or_zero_byte_image(tmp_path: Path):
    cbz_zero = tmp_path / "zero.cbz"
    with zipfile.ZipFile(cbz_zero, "w") as zf:
        zf.writestr("page_01.png", b"")

    with pytest.raises(ValueError, match="page_01\\.png"):
        ArchiveExtractor().extract(cbz_zero, tmp_path / "work_zero")

    corrupt_dir = tmp_path / "corrupt_dir"
    corrupt_dir.mkdir()
    (corrupt_dir / "broken_page.jpg").write_bytes(b"not-a-real-jpeg-at-all")

    with pytest.raises(ValueError, match="broken_page\\.jpg"):
        ArchiveExtractor().extract(corrupt_dir, tmp_path / "work_corrupt")

    truncated_dir = tmp_path / "truncated_dir"
    truncated_dir.mkdir()
    valid_png = _make_png_bytes(100, 150)
    (truncated_dir / "truncated_page.png").write_bytes(valid_png[: len(valid_png) // 2])

    with pytest.raises(ValueError, match="truncated_page\\.png"):
        ArchiveExtractor().extract(truncated_dir, tmp_path / "work_truncated")


@pytest.mark.parametrize(
    ("tool_name", "attr_name"),
    [
        ("unrar", "UNRAR_TOOL"),
        ("unar", "UNAR_TOOL"),
        ("7z", "SEVENZIP_TOOL"),
    ],
)
def test_extract_cbr_configures_rarfile_and_extracts_pages(
    tmp_path: Path, monkeypatch, tool_name: str, attr_name: str
):
    cbr_path = tmp_path / "manga.cbr"
    cbr_path.write_bytes(b"Rar!stub")
    work_dir = tmp_path / "work_cbr"

    fake_tool = tmp_path / "bin" / tool_name
    fake_tool.parent.mkdir(parents=True)
    fake_tool.write_text("stub")

    discovery = MagicMock(spec=PlatformDiscovery)
    discovery.find_unrar.return_value = fake_tool

    entries = {
        "page_10.png": _make_png_bytes(110, 170),
        "page_2.png": _make_png_bytes(400, 200),
        "page_1.png": _make_png_bytes(100, 160),
        "ComicInfo.xml": b"<ComicInfo/>",
        "__MACOSX/._page_1.png": b"junk",
    }

    fake_infos = []
    for name in entries:
        info = MagicMock()
        info.filename = name
        info.is_dir.return_value = False
        fake_infos.append(info)

    fake_rf = MagicMock()
    fake_rf.__enter__.return_value = fake_rf
    fake_rf.__exit__.return_value = False
    fake_rf.infolist.return_value = fake_infos
    fake_rf.open.side_effect = lambda info: io.BytesIO(entries[info.filename])

    rarfile_cls = MagicMock(return_value=fake_rf)
    monkeypatch.setattr(rarfile, "RarFile", rarfile_cls)
    monkeypatch.setattr(rarfile, attr_name, tool_name)
    monkeypatch.setattr(rarfile, "CURRENT_SETUP", None)

    pages = ArchiveExtractor(discovery=discovery).extract(cbr_path, work_dir)

    discovery.find_unrar.assert_called_once_with()
    assert getattr(rarfile, attr_name) == str(fake_tool)
    rarfile_cls.assert_called_once_with(cbr_path)
    assert len(pages) == 3
    assert [p.is_cover for p in pages] == [True, False, False]
    assert [p.is_spread for p in pages] == [False, True, False]
    assert [(p.width, p.height) for p in pages] == [(100, 160), (400, 200), (110, 170)]


def test_extract_cbr_rejects_zip_slip_traversal(tmp_path: Path, monkeypatch):
    cbr_path = tmp_path / "evil.cbr"
    cbr_path.write_bytes(b"Rar!stub")

    discovery = MagicMock(spec=PlatformDiscovery)
    discovery.find_unrar.return_value = None

    info = MagicMock()
    info.filename = "..\\evil.png"
    info.is_dir.return_value = False

    fake_rf = MagicMock()
    fake_rf.__enter__.return_value = fake_rf
    fake_rf.__exit__.return_value = False
    fake_rf.infolist.return_value = [info]

    monkeypatch.setattr(rarfile, "RarFile", MagicMock(return_value=fake_rf))

    with pytest.raises(ValueError, match="traversal|unsafe|path"):
        ArchiveExtractor(discovery=discovery).extract(cbr_path, tmp_path / "work")


def test_extract_error_cases(tmp_path: Path):
    extractor = ArchiveExtractor()

    with pytest.raises(FileNotFoundError):
        extractor.extract(tmp_path / "missing.cbz", tmp_path / "work")

    bad_file = tmp_path / "book.pdf"
    bad_file.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="Unsupported"):
        extractor.extract(bad_file, tmp_path / "work")

    empty_cbz = tmp_path / "empty.cbz"
    with zipfile.ZipFile(empty_cbz, "w") as zf:
        zf.writestr("ComicInfo.xml", b"<ComicInfo/>")
    with pytest.raises(ValueError, match="No valid image"):
        extractor.extract(empty_cbz, tmp_path / "work_empty")


def test_extract_cbr_skips_rarinfo_symlinks_and_bounds_staged_stem(
    tmp_path: Path, monkeypatch
) -> None:
    cbr_path = tmp_path / "symlinks.cbr"
    cbr_path.write_bytes(b"Rar!stub")
    discovery = MagicMock(spec=PlatformDiscovery)
    discovery.find_unrar.return_value = None

    long_stem = "very_long_page_name_" * 8
    real_info = MagicMock()
    real_info.filename = f"{long_stem}.png"
    real_info.is_dir.return_value = False
    real_info.is_symlink.return_value = False
    real_info.external_attr = 0

    sym_info = MagicMock()
    sym_info.filename = "evil_link.png"
    sym_info.is_dir.return_value = False
    sym_info.is_symlink.return_value = True
    sym_info.external_attr = 0

    fake_rf = MagicMock()
    fake_rf.__enter__.return_value = fake_rf
    fake_rf.__exit__.return_value = False
    fake_rf.infolist.return_value = [sym_info, real_info]
    fake_rf.open.side_effect = lambda info: io.BytesIO(_make_png_bytes(100, 150))

    monkeypatch.setattr(rarfile, "RarFile", MagicMock(return_value=fake_rf))
    pages = ArchiveExtractor(discovery=discovery).extract(cbr_path, tmp_path / "work_sym")
    assert len(pages) == 1
    assert len(pages[0].source_path.stem) <= 75


def test_extract_folder_skips_symlinks(tmp_path: Path) -> None:
    src_dir = tmp_path / "dir_with_symlink"
    src_dir.mkdir()
    outside = tmp_path / "outside.png"
    _write_png(outside, 200, 300)
    _write_png(src_dir / "01.png", 100, 150)
    (src_dir / "02_link.png").symlink_to(outside)

    pages = ArchiveExtractor().extract(src_dir, tmp_path / "work_dir_sym")
    assert [(p.width, p.height) for p in pages] == [(100, 150)]


def test_extract_parallel_workers_preserves_order_and_staging_semantics(tmp_path: Path) -> None:
    # 1. CBZ with 10 naturally sorted pages + one spread
    cbz_path = tmp_path / "parallel.cbz"
    expected_dims = [
        (100, 150),
        (300, 150),  # spread at page 2
        (102, 150),
        (104, 150),
        (106, 150),
        (108, 150),
        (110, 150),
        (112, 150),
        (114, 150),
        (116, 150),
    ]
    with zipfile.ZipFile(cbz_path, "w") as zf:
        for idx in reversed(range(10)):
            w, h = expected_dims[idx]
            zf.writestr(f"page_{idx + 1}.png", _make_png_bytes(w, h))

    work_cbz = tmp_path / "work_cbz"
    cbz_pages = ArchiveExtractor().extract(cbz_path, work_cbz, workers=4)
    assert [p.index for p in cbz_pages] == list(range(10))
    assert [p.is_cover for p in cbz_pages] == [i == 0 for i in range(10)]
    assert [p.is_spread for p in cbz_pages] == [i == 1 for i in range(10)]
    assert [(p.width, p.height) for p in cbz_pages] == expected_dims
    assert not any(p.name.startswith("raw_") for p in work_cbz.iterdir())

    # 2. Image directory preserves source files while staging cleanly with workers=4
    src_dir = tmp_path / "folder_manga"
    src_files = []
    for idx in range(6):
        w, h = expected_dims[idx]
        src_files.append(_write_png(src_dir / f"p_{idx + 1}.png", w, h))

    dir_pages = ArchiveExtractor().extract(src_dir, tmp_path / "work_dir", workers=4)
    assert [p.index for p in dir_pages] == list(range(6))
    assert [(p.width, p.height) for p in dir_pages] == expected_dims[:6]
    assert all(f.is_file() for f in src_files)

    # 3. Corrupted image surfaces ValueError even when parallel workers are used
    bad_cbz = tmp_path / "bad_parallel.cbz"
    with zipfile.ZipFile(bad_cbz, "w") as zf:
        zf.writestr("page_1.png", _make_png_bytes(100, 150))
        zf.writestr("page_2.png", b"not-an-image")
        zf.writestr("page_3.png", _make_png_bytes(100, 150))

    with pytest.raises(ValueError, match="page_2\\.png"):
        ArchiveExtractor().extract(bad_cbz, tmp_path / "work_bad", workers=4)
