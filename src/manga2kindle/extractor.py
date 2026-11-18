"""Archive extraction, path sanitization, natural sorting, and image verification."""

from __future__ import annotations

import re
import shutil
import stat
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Protocol

import rarfile
from PIL import Image

from manga2kindle.models import PageAsset
from manga2kindle.platform import PlatformDiscovery

IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
)
_IMAGE_EXTENSIONS = IMAGE_EXTENSIONS
_IGNORED_FILENAMES = frozenset({"thumbs.db", "comicinfo.xml"})
_IGNORED_DIRS = frozenset({"__macosx"})
_NATURAL_SPLIT_RE = re.compile(r"(\d+)")
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_STAGED_STEM_LEN = 64
_RAR_CONFIG_LOCK = threading.Lock()


class _ArchiveMember(Protocol):
    filename: str

    def is_dir(self) -> bool: ...


class _ArchiveReader(Protocol):
    def infolist(self) -> list[_ArchiveMember]: ...

    def open(self, member: _ArchiveMember): ...


def natural_sort_key(path: str | Path) -> list[int | str]:
    """Return an alphanumeric sort key treating digit runs as integers."""
    normalized = str(path).replace("\\", "/").lower()
    return [
        int(part) if part.isdigit() else part
        for part in _NATURAL_SPLIT_RE.split(normalized)
    ]


def sanitize_stem(raw_stem: str, fallback: str = "page") -> str:
    """Sanitize a filename stem and bound its length for cross-platform path safety."""
    cleaned = _SAFE_STEM_RE.sub("_", raw_stem).strip("_")[:_MAX_STAGED_STEM_LEN].strip("_")
    return cleaned or fallback


class ArchiveExtractor:
    """Extracts CBZ, CBR, or image directories into verified, naturally ordered pages."""

    def __init__(self, discovery: PlatformDiscovery | None = None) -> None:
        self.discovery = discovery or PlatformDiscovery()

    def extract(self, input_path: Path, work_dir: Path) -> list[PageAsset]:
        """Extract/stage pages into ``work_dir / 'extracted'`` and verify each image."""
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        work_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir = work_dir / "extracted"
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
        extracted_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_dir():
            candidates = self._collect_from_directory(input_path)
            return self._verify_and_stage(candidates, extracted_dir, input_path)

        suffix = input_path.suffix.lower()
        if suffix in {".cbz", ".zip"}:
            with tempfile.TemporaryDirectory(dir=work_dir, prefix="raw_") as tmp_dir:
                with zipfile.ZipFile(input_path, "r") as zf:
                    candidates = self._extract_archive(zf, Path(tmp_dir))
                return self._verify_and_stage(candidates, extracted_dir, input_path)

        if suffix in {".cbr", ".rar"}:
            self._configure_rar_tool()
            with tempfile.TemporaryDirectory(dir=work_dir, prefix="raw_") as tmp_dir:
                with rarfile.RarFile(input_path) as rf:
                    candidates = self._extract_archive(rf, Path(tmp_dir))
                return self._verify_and_stage(candidates, extracted_dir, input_path)

        raise ValueError(f"Unsupported input format '{input_path.suffix}' for {input_path}")

    def _configure_rar_tool(self) -> None:
        with _RAR_CONFIG_LOCK:
            unrar_path = self.discovery.find_unrar()
            if unrar_path is not None:
                stem = unrar_path.stem.lower()
                if stem == "unar":
                    attr = "UNAR_TOOL"
                elif stem in {"7z", "7za", "7zz"}:
                    attr = "SEVENZIP_TOOL"
                else:
                    attr = "UNRAR_TOOL"
                target = str(unrar_path)
                if getattr(rarfile, attr, None) != target:
                    setattr(rarfile, attr, target)
                    rarfile.CURRENT_SETUP = None
            if rarfile.CURRENT_SETUP is None:
                try:
                    rarfile.tool_setup()
                except (rarfile.RarCannotExec, OSError):
                    pass

    @staticmethod
    def validate_member_parts(raw_name: str) -> list[str]:
        """Validate an archive member path against traversal and return clean parts."""
        if "\x00" in raw_name:
            raise ValueError(f"Unsafe archive member path (null byte): {raw_name}")
        normalized = raw_name.replace("\\", "/")
        if normalized.startswith("/") or _DRIVE_PREFIX_RE.match(normalized):
            raise ValueError(f"Unsafe archive member path (absolute path): {raw_name}")

        segments = normalized.split("/")
        if ".." in segments:
            raise ValueError(f"Unsafe archive member path (path traversal): {raw_name}")

        return [part for part in segments if part not in ("", ".")]

    @staticmethod
    def is_ignored_parts(parts: list[str]) -> bool:
        """Return True for metadata, dotfiles, OS junk, or non-image paths."""
        if not parts:
            return True
        if any(part.lower() in _IGNORED_DIRS or part.startswith(".") for part in parts):
            return True
        filename = parts[-1]
        if filename.lower() in _IGNORED_FILENAMES:
            return True
        return Path(filename).suffix.lower() not in IMAGE_EXTENSIONS

    def _collect_from_directory(self, root_dir: Path) -> list[tuple[str, Path]]:
        root_resolved = root_dir.resolve()
        candidates: list[tuple[str, Path]] = []
        for path in root_dir.rglob("*"):
            if path.is_symlink():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
                continue
            rel = path.relative_to(root_dir)
            parts = list(rel.parts)
            if self.is_ignored_parts(parts):
                continue
            candidates.append((rel.as_posix(), path))
        return candidates

    def _extract_archive(
        self, archive: _ArchiveReader, raw_dir: Path
    ) -> list[tuple[str, Path]]:
        raw_root = raw_dir.resolve()
        candidates: list[tuple[str, Path]] = []

        for info in archive.infolist():
            parts = self.validate_member_parts(info.filename)
            if info.is_dir():
                continue
            is_symlink_fn = getattr(info, "is_symlink", None)
            if callable(is_symlink_fn) and is_symlink_fn() is True:
                continue
            ext_attr = getattr(info, "external_attr", 0)
            if isinstance(ext_attr, int) and stat.S_ISLNK(ext_attr >> 16):
                continue
            if self.is_ignored_parts(parts):
                continue

            dest = (raw_root / Path(*parts)).resolve()
            if not dest.is_relative_to(raw_root):
                raise ValueError(
                    f"Unsafe archive member path (escapes target dir): {info.filename}"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            candidates.append(("/".join(parts), dest))

        return candidates

    @staticmethod
    def _verify_and_measure(image_path: Path, display_name: str) -> tuple[int, int]:
        """Verify image integrity via Pillow and return (width, height)."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as exc:
            raise ValueError(
                f"Corrupted or unreadable image '{display_name}': {exc}"
            ) from exc

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid image dimensions ({width}x{height}) for '{display_name}'"
            )
        return width, height

    def _verify_and_stage(
        self,
        candidates: list[tuple[str, Path]],
        extracted_dir: Path,
        input_path: Path,
    ) -> list[PageAsset]:
        if not candidates:
            raise ValueError(f"No valid image files found in {input_path}")

        ordered = sorted(candidates, key=lambda item: natural_sort_key(item[0]))
        pad = max(4, len(str(len(ordered))))
        pages: list[PageAsset] = []

        for index, (rel_name, src_path) in enumerate(ordered):
            width, height = self._verify_and_measure(src_path, rel_name)
            safe_stem = sanitize_stem(src_path.stem, fallback="page")
            staged_name = f"{index:0{pad}d}_{safe_stem}{src_path.suffix.lower()}"
            staged_path = extracted_dir / staged_name
            shutil.copy2(src_path, staged_path)
            pages.append(
                PageAsset(
                    index=index,
                    source_path=staged_path,
                    width=width,
                    height=height,
                    is_cover=(index == 0),
                    is_spread=(width > height),
                )
            )

        return pages
