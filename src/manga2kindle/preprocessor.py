"""E-Ink image preprocessing via Kindle Comic Converter (KCC) or Pillow fallback."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from manga2kindle.extractor import ArchiveExtractor, natural_sort_key, sanitize_stem
from manga2kindle.models import (
    ConversionConfig,
    PageAsset,
    ReadingDirection,
    SpreadMode,
)
from manga2kindle.platform import (
    PlatformDiscovery,
    available_memory_gib,
    run_managed_subprocess,
)

_DEFAULT_SUBPROCESS_RUN = subprocess.run

_SPREAD_CLI_ARGS: dict[SpreadMode, list[str]] = {
    SpreadMode.SPLIT: ["-r", "0"],
    SpreadMode.ROTATE: ["-r", "1"],
    # KCC -r 2 means "Both (Split + Rotate)"; -r 1 --norotate keeps spreads unsplit and unrotated.
    SpreadMode.KEEP: ["-r", "1", "--norotate"],
}
_KCC_SINGLE_THREAD_ENV: dict[str, str] = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_KCC_MAX_CONCURRENT_CAP: int = 2
_KCC_CONCURRENT_MEMORY_THRESHOLD_GIB: float = 8.0
_KCC_CONCURRENT_MIN_CPUS: int = 4


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


def _flatten_image(img: Image.Image, mode: str) -> Image.Image:
    """Apply EXIF orientation and composite any alpha/palette transparency onto white."""
    transposed = ImageOps.exif_transpose(img)
    if transposed.mode in ("RGBA", "LA") or (
        transposed.mode == "P" and "transparency" in transposed.info
    ):
        rgba = transposed.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white_bg, rgba).convert(mode)
    if transposed.mode != mode:
        return transposed.convert(mode)
    return transposed.copy()


def _is_valid_archive_image(raw_name: str) -> bool:
    try:
        parts = ArchiveExtractor.validate_member_parts(raw_name)
    except ValueError:
        return False
    return not ArchiveExtractor.is_ignored_parts(parts)


class KccBackend:
    """Preprocesses verified page images by staging them cleanly for ``kcc-c2e``."""

    def __init__(
        self,
        discovery: PlatformDiscovery | None = None,
        *,
        max_concurrent: int | None = None,
    ) -> None:
        self.discovery = discovery or PlatformDiscovery()
        self._max_concurrent = self.resolve_max_concurrent(max_concurrent)
        self._semaphore = threading.BoundedSemaphore(self._max_concurrent)

    @staticmethod
    def resolve_max_concurrent(
        override: int | None = None,
        *,
        available_mem_gib: float | None = None,
        cpu_count: int | None = None,
    ) -> int:
        """Resolve safe simultaneous ``kcc-c2e`` subprocess count for this host."""
        if override is not None:
            return max(1, override)
        cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
        avail = available_mem_gib if available_mem_gib is not None else available_memory_gib()
        if cpus >= _KCC_CONCURRENT_MIN_CPUS and (
            avail is None or avail >= _KCC_CONCURRENT_MEMORY_THRESHOLD_GIB
        ):
            return _KCC_MAX_CONCURRENT_CAP
        return 1

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
        kcc_tmp_dir = _reset_dir(work_dir / "kcc_tmp")
        processed_dir = _reset_dir(work_dir / "processed")

        self._stage_pages(pages, staging_dir)
        cmd = self._build_command(kcc_bin, staging_dir, out_dir, config)
        # Isolate KCC's global checkPre('KCC-') cleanup and cap BLAS threads per child process.
        kcc_tmp_str = str(kcc_tmp_dir)
        env = {
            **os.environ,
            **_KCC_SINGLE_THREAD_ENV,
            "TMPDIR": kcc_tmp_str,
            "TEMP": kcc_tmp_str,
            "TMP": kcc_tmp_str,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }

        try:
            with self._semaphore:
                if subprocess.run is not _DEFAULT_SUBPROCESS_RUN:
                    proc = subprocess.run(
                        cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )
                else:
                    proc = run_managed_subprocess(cmd, env=env)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ImagePreprocessingError(f"Failed to execute kcc-c2e: {exc}") from exc

        if proc.returncode != 0:
            details = (proc.stderr or proc.stdout or "").strip()
            raise ImagePreprocessingError(
                f"kcc-c2e exited with status {proc.returncode}: {details}"
            )

        extracted_paths = self._unpack_cbz_outputs(out_dir, processed_dir)
        preserve_sources = len(extracted_paths) == len(pages)

        if config.workers > 1 and len(extracted_paths) > 1:
            with ThreadPoolExecutor(
                max_workers=min(config.workers, len(extracted_paths))
            ) as pool:
                dimensions = list(pool.map(_measure_image, extracted_paths))
        else:
            dimensions = [_measure_image(out_path) for out_path in extracted_paths]

        result: list[PageAsset] = []
        for idx, (out_path, (width, height)) in enumerate(zip(extracted_paths, dimensions)):
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
                *_SPREAD_CLI_ARGS[config.spread_mode],
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
                    if not info.is_dir()
                    and not (
                        isinstance(getattr(info, "external_attr", 0), int)
                        and stat.S_ISLNK(info.external_attr >> 16)
                    )
                    and _is_valid_archive_image(info.filename)
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
                    raw_leaf = Path(info.filename.replace("\\", "/")).name
                    leaf_path = Path(raw_leaf)
                    safe_leaf_stem = sanitize_stem(leaf_path.stem, fallback="page")
                    out_path = processed_dir / f"{idx:0{pad}d}_{safe_leaf_stem}{leaf_path.suffix.lower()}"
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
        self._thread_local = threading.local()

    @staticmethod
    def _make_eink_palette() -> Image.Image:
        palette_img = Image.new("P", (1, 1))
        palette_img.putpalette(
            [channel for level in range(16) for channel in (level * 17,) * 3]
        )
        return palette_img

    def _get_eink_palette(self) -> Image.Image:
        palette = getattr(self._thread_local, "eink_palette", None)
        if palette is None:
            palette = self._make_eink_palette()
            self._thread_local.eink_palette = palette
        return palette

    def _process_single_page(
        self,
        page_idx: int,
        page: PageAsset,
        processed_dir: Path,
        target_size: tuple[int, int],
        config: ConversionConfig,
    ) -> list[tuple[Path, int, int, bool, Path]]:
        if not page.source_path.is_file():
            raise ImagePreprocessingError(
                f"Source page does not exist: {page.source_path}"
            )
        rendered: list[tuple[Path, int, int, bool, Path]] = []
        with Image.open(page.source_path) as raw:
            gray = self._to_grayscale(raw)
            pre_crop_spread = page.is_spread or (gray.width > gray.height)
            cropped = self._autocrop_white(gray)
            variants, keep_spread = self._handle_spread(
                cropped, pre_crop_spread, config
            )
            for var_idx, variant in enumerate(variants):
                final_img = self._render_eink_page(
                    variant, target_size, keep_spread=keep_spread
                )
                tmp_path = processed_dir / f"__tmp_{page_idx:06d}_{var_idx}.png"
                final_img.save(tmp_path, format="PNG", optimize=True)
                width, height = final_img.size
                final_img.close()
                rendered.append((tmp_path, width, height, keep_spread, page.source_path))
        return rendered

    def process(
        self,
        pages: list[PageAsset],
        work_dir: Path,
        config: ConversionConfig,
    ) -> list[PageAsset]:
        """Auto-crop, handle spreads, apply gamma 1.6, quantize to 16-level grayscale, and pad."""
        processed_dir = _reset_dir(work_dir / "processed")
        if not pages:
            return []
        target_size = (config.profile.width, config.profile.height)
        pad = max(4, len(str(len(pages) * 2)))

        if config.workers > 1 and len(pages) > 1:
            with ThreadPoolExecutor(max_workers=min(config.workers, len(pages))) as pool:
                per_page_results = list(
                    pool.map(
                        lambda item: self._process_single_page(
                            item[0], item[1], processed_dir, target_size, config
                        ),
                        enumerate(pages),
                    )
                )
        else:
            per_page_results = [
                self._process_single_page(
                    page_idx, page, processed_dir, target_size, config
                )
                for page_idx, page in enumerate(pages)
            ]

        output_pages: list[PageAsset] = []
        out_idx = 0
        for page_variants in per_page_results:
            for tmp_path, width, height, keep_spread, source_path in page_variants:
                out_path = processed_dir / f"{out_idx:0{pad}d}.png"
                os.replace(tmp_path, out_path)
                output_pages.append(
                    PageAsset(
                        index=out_idx,
                        source_path=source_path,
                        processed_path=out_path,
                        width=width,
                        height=height,
                        is_cover=(out_idx == 0),
                        is_spread=keep_spread,
                    )
                )
                out_idx += 1
        return output_pages

    @staticmethod
    def _to_grayscale(img: Image.Image) -> Image.Image:
        return _flatten_image(img, "L")

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
        img: Image.Image, is_spread: bool, config: ConversionConfig
    ) -> tuple[list[Image.Image], bool]:
        if not is_spread:
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
        self,
        img: Image.Image,
        target_size: tuple[int, int],
        *,
        keep_spread: bool = False,
    ) -> Image.Image:
        target_w, target_h = target_size
        box_w = target_w * 2 if keep_spread else target_w
        box_h = target_h
        scale = min(box_w / img.width, box_h / img.height)
        new_w = min(box_w, max(1, round(img.width * scale)))
        new_h = min(box_h, max(1, round(img.height * scale)))

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if keep_spread:
            canvas = resized
        else:
            canvas = Image.new("L", (target_w, target_h), 255)
            canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))

        gamma_adjusted = canvas.point(self._gamma_lut)
        quantized = gamma_adjusted.convert("RGB").quantize(
            palette=self._get_eink_palette(),
            dither=Image.Dither.NONE,
        )
        # Convert back from palette ("P") to grayscale ("L") so PNG/JPEG saving never fails.
        return quantized.convert("L")


class ImagePreprocessor:
    """Facade selecting KCC or Pillow preprocessing according to ConversionConfig."""

    def __init__(
        self,
        discovery: PlatformDiscovery | None = None,
        *,
        kcc_max_concurrent: int | None = None,
    ) -> None:
        self.discovery = discovery or PlatformDiscovery()
        self._kcc_backend = KccBackend(
            discovery=self.discovery,
            max_concurrent=kcc_max_concurrent,
        )
        self._pillow_backend = PillowFallbackBackend()

    def process(
        self,
        pages: list[PageAsset],
        work_dir: Path,
        config: ConversionConfig,
    ) -> list[PageAsset]:
        """Preprocess extracted pages and normalize cover to config.cover_size."""
        if not pages:
            return []
        processed = (
            self._pillow_backend.process(pages, work_dir, config)
            if config.no_kcc
            else self._kcc_backend.process(pages, work_dir, config)
        )
        if processed and processed[0].is_cover:
            cover_source = (
                pages[0].source_path
                if pages[0].source_path.is_file()
                else processed[0].processed_path
            )
            if cover_source is not None:
                self._normalize_cover(
                    processed[0],
                    source_path=cover_source,
                    cover_size=config.cover_size,
                )
        return processed

    @staticmethod
    def _normalize_cover(
        cover_page: PageAsset,
        *,
        source_path: Path,
        cover_size: tuple[int, int],
    ) -> None:
        """Stretch the cover image to exact cover_size edge-to-edge without borders."""
        if cover_page.processed_path is None:
            return
        target_w, target_h = cover_size
        with Image.open(source_path) as raw:
            rgb = _flatten_image(raw, "RGB")

        resized = rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)

        out_path = cover_page.processed_path
        if out_path.suffix.lower() in {".jpg", ".jpeg"}:
            resized.save(out_path, format="JPEG", quality=95)
        else:
            resized.save(out_path, format="PNG", optimize=True)
        resized.close()

        cover_page.width, cover_page.height = _measure_image(out_path)
        cover_page.is_spread = False
