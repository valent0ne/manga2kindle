"""Tests for filename beautification and source renaming."""

from __future__ import annotations

from pathlib import Path

import pytest

from manga2kindle.beautify import beautify_name, beautify_source_file


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
