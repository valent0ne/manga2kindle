"""End-to-end conversion and panel preview orchestration for manga2kindle."""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from manga2kindle.compiler.kfx import KfxCompilationError, KfxCompiler
from manga2kindle.cv.detector import PanelDetector
from manga2kindle.cv.preview import PanelPreviewRenderer
from manga2kindle.cv.sorter import PanelSorter
from manga2kindle.extractor import ArchiveExtractor, natural_sort_key
from manga2kindle.models import (
    ConversionConfig,
    DeviceProfile,
    PageAsset,
    ReadingDirection,
    SpreadMode,
)
from manga2kindle.packaging.kpf import KpfBuilder
from manga2kindle.platform import PlatformDiscovery
from manga2kindle.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

_BATCH_ARCHIVE_EXTENSIONS = frozenset({".cbz", ".cbr"})


class ConversionPipeline:
    """Orchestrates extraction, preprocessing, CV panel detection, KPF packaging, and KFX compilation."""

    def __init__(
        self,
        discovery: PlatformDiscovery | None = None,
        extractor: ArchiveExtractor | None = None,
        preprocessor: ImagePreprocessor | None = None,
        builder: KpfBuilder | None = None,
    ) -> None:
        self.discovery = discovery or PlatformDiscovery()
        self.extractor = extractor or ArchiveExtractor(discovery=self.discovery)
        self.preprocessor = preprocessor or ImagePreprocessor(discovery=self.discovery)
        self.builder = builder or KpfBuilder()
        self.last_batch_errors: list[tuple[Path, Exception]] = []

    @staticmethod
    def _find_archives(directory: Path) -> list[Path]:
        return sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in _BATCH_ARCHIVE_EXTENSIONS
            ),
            key=natural_sort_key,
        )

    @staticmethod
    def _contains_images(directory: Path) -> bool:
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            parts = list(path.relative_to(directory).parts)
            if not ArchiveExtractor._is_ignored_parts(parts):
                return True
        return False

    def is_batch_input(self, input_path: Path) -> bool:
        """Return True if input_path is a directory of CBZ/CBR archives, False for single input."""
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        if not input_path.is_dir():
            return False
        if self._find_archives(input_path):
            return True
        if self._contains_images(input_path):
            return False
        raise ValueError(
            f"Directory contains no .cbz/.cbr archives or image files: {input_path}"
        )

    @staticmethod
    def resolve_single_output(input_path: Path, output_path: Path | None) -> Path:
        """Resolve exact output .kfx path for a single archive or image folder."""
        if output_path is None:
            return input_path.parent / f"{input_path.stem}.kfx"
        if output_path.suffix.lower() == ".kfx":
            return output_path
        return output_path / f"{input_path.stem}.kfx"

    @staticmethod
    def resolve_batch_output_dir(input_path: Path, output_path: Path | None) -> Path:
        """Resolve output directory for batch archive conversion."""
        if output_path is None:
            return input_path
        if output_path.suffix.lower() == ".kfx":
            raise ValueError(
                f"Batch output path must be a directory, got file path: {output_path}"
            )
        return output_path

    @staticmethod
    def _detect_and_sort_pages(
        pages: list[PageAsset],
        *,
        min_panel_area: float,
        padding: float,
        direction: ReadingDirection,
        workers: int,
    ) -> list[PageAsset]:
        detector = PanelDetector(min_panel_area=min_panel_area, padding=padding)
        sorter = PanelSorter(direction=direction)

        def _process_page(page: PageAsset) -> PageAsset:
            img_path = page.processed_path or page.source_path
            panels = sorter.sort(detector.detect(img_path))
            return page.model_copy(update={"panels": panels})

        if workers > 1 and len(pages) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(pages))) as executor:
                return list(executor.map(_process_page, pages))
        return [_process_page(page) for page in pages]

    def run_single(
        self,
        config: ConversionConfig,
        *,
        title: str | None = None,
        author: str | None = None,
    ) -> Path:
        """Convert a single CBZ/CBR archive or image folder into a verified PDOC KFX."""
        calibre_path = self.discovery.find_calibre(override=config.calibre_path)
        if calibre_path is None:
            raise KfxCompilationError(
                "calibre-debug not found. Install Calibre with the KFX Output plugin "
                "or pass --calibre-path."
            )

        output_kfx = self.resolve_single_output(config.input_path, config.output_path)
        book_title = (title or "").strip() or config.input_path.stem
        work_dir = Path(tempfile.mkdtemp(prefix="manga2kindle_"))

        try:
            pages = self.extractor.extract(config.input_path, work_dir)
            processed = self.preprocessor.process(pages, work_dir, config)
            annotated = self._detect_and_sort_pages(
                processed,
                min_panel_area=config.min_panel_area,
                padding=config.padding,
                direction=config.direction,
                workers=config.workers,
            )

            kpf_path = work_dir / f"{config.input_path.stem}.kpf"
            self.builder.build(
                pages=annotated,
                title=book_title,
                config=config,
                output_kpf=kpf_path,
                author=author,
            )

            compiler = KfxCompiler(calibre_path)
            compiled = compiler.compile(kpf_path, output_kfx)
            if not compiler.verify_pdoc(compiled):
                raise KfxCompilationError(
                    f"Compiled KFX failed PDOC verification: {compiled}"
                )
            return compiled
        finally:
            if not config.keep_temp:
                shutil.rmtree(work_dir, ignore_errors=True)

    def run_batch(
        self,
        config: ConversionConfig,
        *,
        author: str | None = None,
        on_success: Callable[[Path, Path], None] | None = None,
        on_error: Callable[[Path, Exception], None] | None = None,
    ) -> list[Path]:
        """Convert all CBZ/CBR archives in config.input_path concurrently."""
        archives = self._find_archives(config.input_path)
        if not archives:
            raise ValueError(f"No .cbz or .cbr archives found in {config.input_path}")

        out_dir = self.resolve_batch_output_dir(config.input_path, config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.last_batch_errors = []

        batch_workers = min(config.workers, len(archives))
        per_book_workers = max(1, config.workers // batch_workers)

        def _convert_one(archive_path: Path) -> tuple[Path, Path | None, Exception | None]:
            book_config = config.model_copy(
                update={
                    "input_path": archive_path,
                    "output_path": out_dir / f"{archive_path.stem}.kfx",
                    "workers": per_book_workers,
                }
            )
            try:
                out_kfx = self.run_single(
                    book_config,
                    title=archive_path.stem,
                    author=author,
                )
                return archive_path, out_kfx, None
            except Exception as exc:  # noqa: BLE001 - isolate per-book failures in batch mode
                logger.error("Failed to convert %s: %s", archive_path, exc)
                return archive_path, None, exc

        if batch_workers > 1:
            with ThreadPoolExecutor(max_workers=batch_workers) as executor:
                outcomes = list(executor.map(_convert_one, archives))
        else:
            outcomes = [_convert_one(archive) for archive in archives]

        succeeded: list[Path] = []
        errors: list[tuple[Path, Exception]] = []
        for archive_path, out_kfx, exc in outcomes:
            if exc is not None or out_kfx is None:
                err = exc if exc is not None else RuntimeError("Unknown conversion failure")
                errors.append((archive_path, err))
                if on_error is not None:
                    on_error(archive_path, err)
            else:
                succeeded.append(out_kfx)
                if on_success is not None:
                    on_success(archive_path, out_kfx)

        self.last_batch_errors = errors
        return succeeded

    def run(
        self,
        config: ConversionConfig,
        *,
        title: str | None = None,
        author: str | None = None,
        on_success: Callable[[Path, Path], None] | None = None,
        on_error: Callable[[Path, Exception], None] | None = None,
    ) -> list[Path]:
        """Auto-disambiguate single vs batch input and execute conversion."""
        if self.is_batch_input(config.input_path):
            return self.run_batch(
                config,
                author=author,
                on_success=on_success,
                on_error=on_error,
            )
        return [self.run_single(config, title=title, author=author)]

    def preview_panels(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        profile: DeviceProfile | None = None,
        direction: ReadingDirection = ReadingDirection.RTL,
        spread_mode: SpreadMode = SpreadMode.SPLIT,
        padding: float = 0.02,
        min_panel_area: float = 0.03,
        no_kcc: bool = False,
        keep_temp: bool = False,
        workers: int = 4,
        kcc_path: Path | None = None,
    ) -> list[Path]:
        """Extract, preprocess, detect/sort panels, and render numbered PNG debug overlays."""
        resolved_profile = profile or DeviceProfile.from_name("KPW5")
        output_dir.mkdir(parents=True, exist_ok=True)

        config = ConversionConfig(
            input_path=input_path,
            output_path=output_dir,
            profile=resolved_profile,
            direction=direction,
            spread_mode=spread_mode,
            padding=padding,
            min_panel_area=min_panel_area,
            no_kcc=no_kcc,
            keep_temp=keep_temp,
            workers=workers,
            kcc_path=kcc_path,
        )

        work_dir = Path(tempfile.mkdtemp(prefix="manga2kindle_preview_"))
        try:
            pages = self.extractor.extract(input_path, work_dir)
            processed = self.preprocessor.process(pages, work_dir, config)
            annotated = self._detect_and_sort_pages(
                processed,
                min_panel_area=min_panel_area,
                padding=padding,
                direction=direction,
                workers=workers,
            )

            renderer = PanelPreviewRenderer()
            pad = max(4, len(str(len(annotated))))

            def _render_page(page: PageAsset) -> Path:
                img_path = page.processed_path or page.source_path
                out_file = output_dir / f"page_{page.index + 1:0{pad}d}.png"
                return renderer.render(img_path, page.panels, out_file)

            if workers > 1 and len(annotated) > 1:
                with ThreadPoolExecutor(max_workers=min(workers, len(annotated))) as executor:
                    return list(executor.map(_render_page, annotated))
            return [_render_page(page) for page in annotated]
        finally:
            if not keep_temp:
                shutil.rmtree(work_dir, ignore_errors=True)
