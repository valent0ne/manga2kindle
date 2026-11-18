"""Filename and title beautification utilities for manga archives."""

from __future__ import annotations

import re
from pathlib import Path

_BRACKET_TAG_RE = re.compile(r"\[[^\]]*\]")
_NON_YEAR_PAREN_TAG_RE = re.compile(r"\((?!(?:19|20)\d{2}\))[^)]*\)")
_MULTI_SPACE_RE = re.compile(r"\s+")
_TRAILING_SEPARATOR_RE = re.compile(r"\s+[-_]+$")


def beautify_name(stem: str) -> str:
    """Strip release tags from a filename stem while preserving title, volume/chapter, and year.

    Removes bracketed ``[...]`` tags and parenthesized ``(...)`` tags except
    4-digit release years (``19xx`` / ``20xx``), normalizes whitespace, and trims
    dangling trailing separators. Falls back to ``stem`` if cleaning yields an
    empty string.
    """
    cleaned = _BRACKET_TAG_RE.sub(" ", stem)
    cleaned = _NON_YEAR_PAREN_TAG_RE.sub(" ", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    cleaned = _TRAILING_SEPARATOR_RE.sub("", cleaned).strip()
    return cleaned or stem


def beautify_source_file(path: Path) -> Path:
    """Rename a source archive file in place using :func:`beautify_name`.

    Directories and already-clean filenames are returned unchanged. Raises
    ``FileExistsError`` if a distinct file already exists at the target path.
    """
    if not path.is_file():
        return path

    new_stem = beautify_name(path.stem)
    if new_stem == path.stem:
        return path

    target = path.with_name(f"{new_stem}{path.suffix}")
    if target.exists() and target.resolve() != path.resolve():
        raise FileExistsError(
            f"Cannot beautify '{path.name}': target '{target.name}' already exists."
        )

    path.rename(target)
    return target
