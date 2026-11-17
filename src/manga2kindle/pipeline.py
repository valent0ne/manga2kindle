"""End-to-end conversion and panel preview orchestration for manga2kindle."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from manga2kindle.compiler.kfx import KfxCompilationError, KfxCompiler
from manga2kindle.cv.backends import BackendRegistry
from manga2kindle.cv.preview import PanelPreviewRenderer
from manga2kindle.cv.sorter import PanelSorter
from manga2kindle.extractor import ArchiveExtractor, natural_sort_key
from manga2kindle.models import (
    ConversionConfig,
    DetectorConfig,
    DetectorKind,
    DeviceProfile,
    PageAsset,
    PanelBox,
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
        backend_registry: BackendRegistry | None = None,
    ) -> None:
        self.discovery = discovery or PlatformDiscovery()
        self.extractor = extractor or ArchiveExtractor(discovery=self.discovery)
        self.preprocessor = preprocessor or ImagePreprocessor(discovery=self.discovery)
        self.builder = builder or KpfBuilder()
        self._backend_registry = backend_registry
        self.last_batch_errors: list[tuple[Path, Exception]] = []

    def _resolve_registry(
        self,
        detector_config: DetectorConfig,
        registry: BackendRegistry | None = None,
    ) -> BackendRegistry:
        if registry is not None:
            registry.resolve()
            return registry
        if self._backend_registry is not None:
            self._backend_registry.resolve()
            return self._backend_registry
        created = BackendRegistry(detector_config)
        created.resolve()
        return created

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
            if path.is_symlink() or not path.is_file():
                continue
            parts = list(path.relative_to(directory).parts)
            if not ArchiveExtractor.is_ignored_parts(parts):
                return True
        return False

    @staticmethod
    def resolve_input_name(input_path: Path) -> str:
        """Return a clean human-readable base name for an archive file or directory."""
        resolved = input_path.resolve()
        if input_path.is_dir() or resolved.is_dir():
            return resolved.name or "manga"
        return resolved.stem or resolved.name or "manga"

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

    @classmethod
    def resolve_single_output(cls, input_path: Path, output_path: Path | None) -> Path:
        """Resolve exact output .kfx path for a single archive or image folder."""
        base_name = cls.resolve_input_name(input_path)
        if output_path is None:
            return input_path.resolve().parent / f"{base_name}.kfx"
        if output_path.suffix.lower() == ".kfx":
            if output_path.exists() and output_path.is_dir():
                raise ValueError(
                    f"Output .kfx path points to an existing directory: {output_path}"
                )
            return output_path
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(
                f"Output path exists and is not a directory: {output_path}"
            )
        return output_path / f"{base_name}.kfx"

    @staticmethod
    def resolve_batch_output_dir(input_path: Path, output_path: Path | None) -> Path:
        """Resolve output directory for batch archive conversion."""
        if output_path is None:
            return input_path
        if output_path.suffix.lower() == ".kfx":
            raise ValueError(
                f"Batch output path must be a directory, got file path: {output_path}"
            )
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(
                f"Batch output path exists and is not a directory: {output_path}"
            )
        return output_path

    @staticmethod
    def _detect_and_sort_pages(
        pages: list[PageAsset],
        *,
        registry: BackendRegistry,
        direction: ReadingDirection,
        workers: int,
    ) -> list[PageAsset]:
        sorter = PanelSorter(direction=direction)

        def _process_page(page: PageAsset) -> PageAsset:
            img_path = page.processed_path or page.source_path
            panels = registry.detect_and_sort(
                img_path,
                sorter,
                page_width=page.width,
                is_spread=page.is_spread,
            )
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
        registry: BackendRegistry | None = None,
    ) -> Path:
        """Convert a single CBZ/CBR archive or image folder into a verified PDOC KFX."""
        calibre_path = self.discovery.find_calibre(override=config.calibre_path)
        if calibre_path is None:
            raise KfxCompilationError(
                "calibre-debug not found. Install Calibre with the KFX Output plugin "
                "or pass --calibre-path."
            )

        active_registry = self._resolve_registry(config.detector, registry)
        input_name = self.resolve_input_name(config.input_path)
        output_kfx = self.resolve_single_output(config.input_path, config.output_path)
        book_title = (title or "").strip() or input_name
        work_dir = Path(tempfile.mkdtemp(prefix="manga2kindle_"))

        try:
            pages = self.extractor.extract(config.input_path, work_dir)
            processed = self.preprocessor.process(pages, work_dir, config)
            annotated = self._detect_and_sort_pages(
                processed,
                registry=active_registry,
                direction=config.direction,
                workers=config.workers,
            )

            kpf_path = work_dir / f"{input_name}.kpf"
            self.builder.build(
                pages=annotated,
                title=book_title,
                config=config,
                output_kpf=kpf_path,
                author=author,
            )

            compiler = KfxCompiler(calibre_path)
            staged_kfx = work_dir / f"{input_name}.kfx"
            compiled = compiler.compile(kpf_path, staged_kfx)
            if not compiler.verify_pdoc(compiled):
                raise KfxCompilationError(
                    f"Compiled KFX failed PDOC verification: {compiled}"
                )

            output_kfx.parent.mkdir(parents=True, exist_ok=True)
            tmp_dest = output_kfx.with_name(f".{output_kfx.stem}.tmp_{os.getpid()}.kfx")
            try:
                shutil.copy2(compiled, tmp_dest)
                os.replace(tmp_dest, output_kfx)
            finally:
                tmp_dest.unlink(missing_ok=True)
            return output_kfx
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
        seen_names: dict[str, Path] = {}
        targets: list[tuple[Path, Path]] = []
        for archive_path in archives:
            out_kfx = out_dir / f"{archive_path.stem}.kfx"
            key = out_kfx.name.lower()
            if key in seen_names:
                raise ValueError(
                    f"Batch output filename collision for '{out_kfx.name}': "
                    f"'{seen_names[key].name}' and '{archive_path.name}'"
                )
            seen_names[key] = archive_path
            targets.append((archive_path, out_kfx))

        calibre_path = self.discovery.find_calibre(override=config.calibre_path)
        if calibre_path is None:
            raise KfxCompilationError(
                "calibre-debug not found. Install Calibre with the KFX Output plugin "
                "or pass --calibre-path."
            )

        batch_registry = self._resolve_registry(config.detector)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.last_batch_errors = []

        batch_workers = min(config.workers, len(targets))
        per_book_workers = max(1, config.workers // batch_workers)

        def _convert_one(
            archive_path: Path, target_kfx: Path
        ) -> tuple[Path, Path | None, Exception | None]:
            book_config = config.model_copy(
                update={
                    "input_path": archive_path,
                    "output_path": target_kfx,
                    "calibre_path": calibre_path,
                    "workers": per_book_workers,
                }
            )
            try:
                out_kfx = self.run_single(
                    book_config,
                    title=archive_path.stem,
                    author=author,
                    registry=batch_registry,
                )
                return archive_path, out_kfx, None
            except Exception as exc:  # noqa: BLE001 - isolate per-book failures in batch mode
                logger.error("Failed to convert %s: %s", archive_path, exc)
                return archive_path, None, exc

        if batch_workers > 1:
            with ThreadPoolExecutor(max_workers=batch_workers) as executor:
                outcomes = list(
                    executor.map(lambda item: _convert_one(item[0], item[1]), targets)
                )
        else:
            outcomes = [_convert_one(arch, tgt) for arch, tgt in targets]

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

    @staticmethod
    def validate_compare_backends(
        compare: list[DetectorKind],
        detector_config: DetectorConfig,
    ) -> list[DetectorKind]:
        """Validate and deduplicate --compare backends, failing fast on invalid options."""
        unique_kinds = list(dict.fromkeys(compare))
        if not unique_kinds:
            raise ValueError("At least one backend must be specified for --compare.")
        if DetectorKind.AUTO in unique_kinds:
            raise ValueError(
                "--compare does not support 'auto'; specify concrete backends (contour, onnx, magi)."
            )
        if DetectorKind.MAGI in unique_kinds and not detector_config.allow_remote_code:
            raise ValueError(
                "Using 'magi' in --compare requires --allow-remote-code."
            )
        return unique_kinds

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
        detector: DetectorConfig | None = None,
        detector_config: DetectorConfig | None = None,
        compare: list[DetectorKind] | None = None,
        no_kcc: bool = False,
        keep_temp: bool = False,
        workers: int = 4,
        kcc_path: Path | None = None,
        registry: BackendRegistry | None = None,
    ) -> list[Path]:
        """Extract, preprocess, detect/sort panels, and render numbered PNG debug overlays."""
        resolved_profile = profile or DeviceProfile.from_name("KPW5")
        detector_cfg = (
            detector_config
            or detector
            or DetectorConfig(padding=padding, min_panel_area=min_panel_area)
        )

        compare_kinds: list[DetectorKind] | None = None
        if compare is not None:
            compare_kinds = self.validate_compare_backends(compare, detector_cfg)

        output_dir.mkdir(parents=True, exist_ok=True)
        config = ConversionConfig(
            input_path=input_path,
            output_path=output_dir,
            profile=resolved_profile,
            direction=direction,
            spread_mode=spread_mode,
            padding=padding,
            min_panel_area=min_panel_area,
            detector=detector_cfg,
            no_kcc=no_kcc,
            keep_temp=keep_temp,
            workers=workers,
            kcc_path=kcc_path,
        )

        compare_registries: dict[DetectorKind, BackendRegistry] | None = None
        active_registry: BackendRegistry | None = None
        if compare_kinds is not None:
            compare_registries = {}
            for kind in compare_kinds:
                reg = BackendRegistry(detector_cfg.model_copy(update={"kind": kind}))
                reg.resolve()
                compare_registries[kind] = reg
        else:
            active_registry = self._resolve_registry(config.detector, registry)

        work_dir = Path(tempfile.mkdtemp(prefix="manga2kindle_preview_"))
        try:
            pages = self.extractor.extract(input_path, work_dir)
            processed = self.preprocessor.process(pages, work_dir, config)
            renderer = PanelPreviewRenderer()
            pad = max(4, len(str(len(processed))))

            if compare_registries is not None and compare_kinds is not None:
                annotated_by_kind: dict[DetectorKind, list[PageAsset]] = {
                    kind: self._detect_and_sort_pages(
                        processed,
                        registry=reg,
                        direction=direction,
                        workers=workers,
                    )
                    for kind, reg in compare_registries.items()
                }

                def _render_compare_page(page_idx: int) -> Path:
                    base_page = processed[page_idx]
                    img_path = base_page.processed_path or base_page.source_path
                    out_file = output_dir / f"page_{base_page.index + 1:0{pad}d}.png"
                    results: dict[DetectorKind, list[PanelBox]] = {
                        kind: annotated_by_kind[kind][page_idx].panels
                        for kind in compare_kinds
                    }
                    return renderer.render_comparison(img_path, results, out_file)

                indices = list(range(len(processed)))
                if workers > 1 and len(indices) > 1:
                    with ThreadPoolExecutor(max_workers=min(workers, len(indices))) as executor:
                        return list(executor.map(_render_compare_page, indices))
                return [_render_compare_page(idx) for idx in indices]

            assert active_registry is not None
            annotated = self._detect_and_sort_pages(
                processed,
                registry=active_registry,
                direction=direction,
                workers=workers,
            )

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

    def preview_batch(
        self,
        config: ConversionConfig,
        *,
        on_success: Callable[[Path, Path], None] | None = None,
        on_error: Callable[[Path, Exception], None] | None = None,
    ) -> list[Path]:
        """Render panel preview overlays for all CBZ/CBR archives in config.input_path."""
        archives = self._find_archives(config.input_path)
        if not archives:
            raise ValueError(f"No .cbz or .cbr archives found in {config.input_path}")

        out_dir = self.resolve_batch_output_dir(config.input_path, config.output_path)
        batch_registry = self._resolve_registry(config.detector)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.last_batch_errors = []

        batch_workers = min(config.workers, len(archives))
        per_book_workers = max(1, config.workers // batch_workers)

        def _preview_one(archive_path: Path) -> tuple[Path, Path | None, Exception | None]:
            preview_dir = out_dir / f"{archive_path.stem}_preview"
            try:
                self.preview_panels(
                    input_path=archive_path,
                    output_dir=preview_dir,
                    profile=config.profile,
                    direction=config.direction,
                    spread_mode=config.spread_mode,
                    padding=config.padding,
                    min_panel_area=config.min_panel_area,
                    detector_config=config.detector,
                    no_kcc=config.no_kcc,
                    keep_temp=config.keep_temp,
                    workers=per_book_workers,
                    kcc_path=config.kcc_path,
                    registry=batch_registry,
                )
                return archive_path, preview_dir, None
            except Exception as exc:  # noqa: BLE001 - isolate per-book failures in batch mode
                logger.error("Failed to preview %s: %s", archive_path, exc)
                return archive_path, None, exc

        if batch_workers > 1:
            with ThreadPoolExecutor(max_workers=batch_workers) as executor:
                outcomes = list(executor.map(_preview_one, archives))
        else:
            outcomes = [_preview_one(arch) for arch in archives]

        succeeded_dirs: list[Path] = []
        errors: list[tuple[Path, Exception]] = []
        for archive_path, preview_dir, exc in outcomes:
            if exc is not None or preview_dir is None:
                err = exc if exc is not None else RuntimeError("Unknown preview failure")
                errors.append((archive_path, err))
                if on_error is not None:
                    on_error(archive_path, err)
            else:
                succeeded_dirs.append(preview_dir)
                if on_success is not None:
                    on_success(archive_path, preview_dir)

        self.last_batch_errors = errors
        return succeeded_dirs
