"""E-Ink image preprocessing via Kindle Comic Converter (KCC) or Pillow fallback."""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from manga2kindle.extractor import natural_sort_key
from manga2kindle.models import (
    ConversionConfig,
    PageAsset,
    ReadingDirection,
    SpreadMode,
)
from manga2kindle.platform import PlatformDiscovery

_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
)
_IGNORED_FILENAMES = frozenset({"thumbs.db", "comicinfo.xml"})
_IGNORED_DIRS = frozenset({"__macosx"})
_SPREAD_FLAGS: dict[SpreadMode, str] = {
    SpreadMode.SPLIT: "0",
    SpreadMode.ROTATE: "1",
    SpreadMode.KEEP: "2",
}


class ImagePreprocessingError(RuntimeError):
    """Raised when image preprocessing fails or a required external tool is unavailable."""


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _measure_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception as exc:
        raise ImagePreprocessingError(f"Unreadable processed image '{path}': {exc}") from exc
    if width <= 0 or height <= 0:
        raise ImagePreprocessingError(
            f"Invalid processed image dimensions ({width}x{height}) for '{path}'"
        )
    return width, height


def _stage_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise ImagePreprocessingError(f"Source page does not exist: {src}")
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _is_valid_archive_image(raw_name: str) -> bool:
    if "\x00" in raw_name:
        return False
    normalized = raw_name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        return False
    if any(part.lower() in _IGNORED_DIRS or part.startswith(".") for part in parts):
        return False
    filename = parts[-1]
    if filename.lower() in _IGNORED_FILENAMES:
        return False
    return Path(filename).suffix.lower() in _IMAGE_EXTENSIONS


class KccBackend:
    """Preprocesses verified page images by staging them cleanly for ``kcc-c2e``."""

    def __init__(self, discovery: PlatformDiscovery | None = None) -> None:
        self.discovery = discovery or PlatformDiscovery()

    def process(
        self,
        pages: list[PageAsset],
        work_dir: Path,
        config: ConversionConfig,
    ) -> list[PageAsset]:
        """Run ``kcc-c2e`` on a clean staging directory and unpack the resulting CBZ."""
        kcc_bin = self.discovery.find_kcc(override=config.kcc_path)
        if kcc_bin is None:
            raise ImagePreprocessingError(
                "kcc-c2e executable not found. Install KindleComicConverter or pass --no-kcc."
            )

        staging_dir = _reset_dir(work_dir / "kcc_in")
        out_dir = _reset_dir(work_dir / "kcc_out")
        processed_dir = _reset_dir(work_dir / "processed")

        self._stage_pages(pages, staging_dir)
        cmd = self._build_command(kcc_bin, staging_dir, out_dir, config)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ImagePreprocessingError(f"Failed to execute kcc-c2e: {exc}") from exc

        if proc.returncode != 0:
            details = (proc.stderr or proc.stdout or "").strip()
            raise ImagePreprocessingError(
                f"kcc-c2e exited with status {proc.returncode}: {details}"
            )

        extracted_paths = self._unpack_cbz_outputs(out_dir, processed_dir)
        preserve_sources = len(extracted_paths) == len(pages)

        result: list[PageAsset] = []
        for idx, out_path in enumerate(extracted_paths):
            width, height = _measure_image(out_path)
            result.append(
                PageAsset(
                    index=idx,
                    source_path=pages[idx].source_path if preserve_sources else out_path,
                    processed_path=out_path,
                    width=width,
                    height=height,
                    is_cover=(idx == 0),
                    is_spread=(width > height),
                )
            )
        return result

    @staticmethod
    def _stage_pages(pages: list[PageAsset], staging_dir: Path) -> None:
        pad = max(4, len(str(len(pages))))
        for idx, page in enumerate(pages):
            dst = staging_dir / f"{idx:0{pad}d}_{page.source_path.name}"
            _stage_file(page.source_path, dst)

    @staticmethod
    def _build_command(
        kcc_bin: Path,
        staging_dir: Path,
        out_dir: Path,
        config: ConversionConfig,
    ) -> list[str]:
        cmd = [str(kcc_bin), "-p", config.profile.kcc_code]
        if config.direction == ReadingDirection.RTL:
            cmd.append("-m")
        cmd.extend(
            [
                "-q",
                "-u",
                "-r",
                _SPREAD_FLAGS[config.spread_mode],
                "-f",
                "CBZ",
                "--mozjpeg",
            ]
        )
        if config.profile.kcc_code == "OTHER":
            cmd.extend(
                [
                    "--customwidth",
                    str(config.profile.width),
                    "--customheight",
                    str(config.profile.height),
                ]
            )
        cmd.extend(["-o", str(out_dir), str(staging_dir)])
        return cmd

    @staticmethod
    def _unpack_cbz_outputs(out_dir: Path, processed_dir: Path) -> list[Path]:
        cbz_files = sorted(out_dir.rglob("*.cbz"), key=natural_sort_key)
        if not cbz_files:
            raise ImagePreprocessingError(f"kcc-c2e produced no CBZ output in {out_dir}")

        cbz_members: list[tuple[Path, list[zipfile.ZipInfo]]] = []
        total_entries = 0
        for cbz_path in cbz_files:
            with zipfile.ZipFile(cbz_path, "r") as zf:
                members = [
                    info
                    for info in zf.infolist()
                    if not info.is_dir() and _is_valid_archive_image(info.filename)
                ]
            members.sort(key=lambda info: natural_sort_key(info.filename))
            if members:
                cbz_members.append((cbz_path, members))
                total_entries += len(members)

        if total_entries == 0:
            raise ImagePreprocessingError("kcc-c2e CBZ output contained no image files")

        pad = max(4, len(str(total_entries)))
        extracted: list[Path] = []
        idx = 0
        for cbz_path, members in cbz_members:
            with zipfile.ZipFile(cbz_path, "r") as zf:
                for info in members:
                    leaf_name = Path(info.filename.replace("\\", "/")).name
                    out_path = processed_dir / f"{idx:0{pad}d}_{leaf_name}"
                    with zf.open(info) as src, out_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(out_path)
                    idx += 1

        return extracted


class PillowFallbackBackend:
    """Pure-Pillow E-Ink preprocessing fallback when KCC is disabled via ``--no-kcc``."""

    _AUTOCROP_MARGIN = 5
    _WHITE_THRESHOLD = 250
    _GAMMA = 1.6

    def __init__(self) -> None:
        self._gamma_lut = [
            round(255 * ((value / 255.0) ** self._GAMMA)) for value in range(256)
        ]
        palette_img = Image.new("P", (1, 1))
        palette_img.putpalette(
            [channel for level in range(16) for channel in (level * 17,) * 3]
        )
        self._eink_palette = palette_img

    def process(
        self,
        pages: list[PageAsset],
        work_dir: Path,
        config: ConversionConfig,
    ) -> list[PageAsset]:
        """Auto-crop, handle spreads, apply gamma 1.6, quantize to 16-level grayscale, and pad."""
        processed_dir = _reset_dir(work_dir / "processed")
        target_size = (config.profile.width, config.profile.height)

        rendered: list[tuple[PageAsset, Image.Image, bool]] = []
        for page in pages:
            if not page.source_path.is_file():
                raise ImagePreprocessingError(
                    f"Source page does not exist: {page.source_path}"
                )
            with Image.open(page.source_path) as raw:
                gray = self._to_grayscale(raw)
                cropped = self._autocrop_white(gray)
                variants, keep_spread = self._handle_spread(cropped, config)
                for variant in variants:
                    final_img = self._render_eink_page(variant, target_size)
                    rendered.append((page, final_img, keep_spread))

        pad = max(4, len(str(len(rendered))))
        output_pages: list[PageAsset] = []
        for idx, (src_page, final_img, is_spread) in enumerate(rendered):
            out_path = processed_dir / f"{idx:0{pad}d}.png"
            final_img.save(out_path, format="PNG", optimize=True)
            width, height = _measure_image(out_path)
            output_pages.append(
                PageAsset(
                    index=idx,
                    source_path=src_page.source_path,
                    processed_path=out_path,
                    width=width,
                    height=height,
                    is_cover=(idx == 0),
                    is_spread=is_spread,
                )
            )
        return output_pages

    @staticmethod
    def _to_grayscale(img: Image.Image) -> Image.Image:
        transposed = ImageOps.exif_transpose(img)
        if transposed.mode in ("RGBA", "LA") or (
            transposed.mode == "P" and "transparency" in transposed.info
        ):
            rgba = transposed.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white_bg, rgba).convert("L")
        if transposed.mode != "L":
            return transposed.convert("L")
        return transposed.copy()

    def _autocrop_white(self, img: Image.Image) -> Image.Image:
        mask = img.point(lambda p: 255 if p < self._WHITE_THRESHOLD else 0, mode="1")
        bbox = mask.getbbox()
        if bbox is None:
            return img
        left, top, right, bottom = bbox
        margin = self._AUTOCROP_MARGIN
        return img.crop(
            (
                max(0, left - margin),
                max(0, top - margin),
                min(img.width, right + margin),
                min(img.height, bottom + margin),
            )
        )

    @staticmethod
    def _handle_spread(
        img: Image.Image, config: ConversionConfig
    ) -> tuple[list[Image.Image], bool]:
        if img.width <= img.height:
            return [img], False

        if config.spread_mode == SpreadMode.SPLIT:
            mid = img.width // 2
            left_half = img.crop((0, 0, mid, img.height))
            right_half = img.crop((mid, 0, img.width, img.height))
            if config.direction == ReadingDirection.RTL:
                return [right_half, left_half], False
            return [left_half, right_half], False

        if config.spread_mode == SpreadMode.ROTATE:
            return [img.transpose(Image.Transpose.ROTATE_90)], False

        return [img], True

    def _render_eink_page(
        self, img: Image.Image, target_size: tuple[int, int]
    ) -> Image.Image:
        target_w, target_h = target_size
        scale = min(target_w / img.width, target_h / img.height)
        new_w = min(target_w, max(1, round(img.width * scale)))
        new_h = min(target_h, max(1, round(img.height * scale)))

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (target_w, target_h), 255)
        canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))

        gamma_adjusted = canvas.point(self._gamma_lut)
        quantized = gamma_adjusted.convert("RGB").quantize(
            palette=self._eink_palette,
            dither=Image.Dither.NONE,
        )
        # Convert back from palette ("P") to grayscale ("L") so PNG/JPEG saving never fails.
        return quantized.convert("L")


class ImagePreprocessor:
    """Facade selecting KCC or Pillow preprocessing according to ConversionConfig."""

    def __init__(self, discovery: PlatformDiscovery | None = None) -> None:
        self.discovery = discovery or PlatformDiscovery()
        self._kcc_backend = KccBackend(discovery=self.discovery)
        self._pillow_backend = PillowFallbackBackend()

    def process(
        self,
        pages: list[PageAsset],
        work_dir: Path,
        config: ConversionConfig,
    ) -> list[PageAsset]:
        """Preprocess extracted pages for target Kindle E-Ink dimensions."""
        if not pages:
            return []
        if config.no_kcc:
            return self._pillow_backend.process(pages, work_dir, config)
        return self._kcc_backend.process(pages, work_dir, config)
