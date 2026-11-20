"""End-to-end conversion and panel preview orchestration for manga2kindle."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from manga2kindle.beautify import beautify_name, beautify_source_file
from manga2kindle.compiler.kfx import KfxCompilationError, KfxCompiler
from manga2kindle.cv.backends import BackendRegistry, infer_config_model_target
from manga2kindle.cv.boxes import DEFAULT_PADDING
from manga2kindle.cv.merger import PanelMerger
from manga2kindle.cv.preview import PanelPreviewRenderer
from manga2kindle.cv.sorter import PanelSorter
from manga2kindle.extractor import (
    SUPPORTED_ARCHIVE_EXTENSIONS,
    ArchiveExtractor,
    natural_sort_key,
)
from manga2kindle.models import (
    DEFAULT_COVER_SIZE,
    DEFAULT_PROFILE_NAME,
    ConversionConfig,
    DetectorConfig,
    DetectorKind,
    DeviceProfile,
    PageAsset,
    PanelBox,
    PanelMergeConfig,
    ReadingDirection,
    SpreadMode,
)
from manga2kindle.packaging.kpf import KpfBuilder
from manga2kindle.platform import PlatformDiscovery
from manga2kindle.preprocessor import ImagePreprocessor
from manga2kindle.progress import (
    ConversionStage,
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
)

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
        on_detector_resolved: Callable[[DetectorKind, list[DetectorKind]], None] | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.discovery = discovery or PlatformDiscovery()
        self.extractor = extractor or ArchiveExtractor(discovery=self.discovery)
        self.preprocessor = preprocessor or ImagePreprocessor(discovery=self.discovery)
        self.builder = builder or KpfBuilder()
        self._backend_registry = backend_registry
        self._on_detector_resolved = on_detector_resolved
        self._progress: ProgressReporter = (
            progress if progress is not None else NullProgressReporter()
        )
        self.last_batch_errors: list[tuple[Path, Exception]] = []
        self.last_batch_skipped: list[Path] = []
        self.last_single_skipped: bool = False

    def _resolve_registry(
        self,
        detector_config: DetectorConfig,
        registry: BackendRegistry | None = None,
    ) -> BackendRegistry:
        if registry is not None:
            registry.resolve()
            return registry
        resolved = self._backend_registry or BackendRegistry(detector_config)
        backends = resolved.resolve()
        if self._on_detector_resolved is not None:
            active = resolved.active_backend
            if active is not None:
                self._on_detector_resolved(active.kind, [backend.kind for backend in backends])
        return resolved

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
    def resolve_input_name(input_path: Path, *, beautify: bool = False) -> str:
        """Return a clean human-readable base name for an archive file or directory."""
        resolved = input_path.resolve()
        if input_path.is_dir() or resolved.is_dir():
            raw_name = resolved.name or "manga"
        else:
            raw_name = resolved.stem or resolved.name or "manga"
        return beautify_name(raw_name) if beautify else raw_name

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
    def _available_ram_gib() -> float | None:
        try:
            import psutil
        except ImportError:
            return None
        return psutil.virtual_memory().available / (1024.0**3)

    @classmethod
    def _resolve_batch_workers(
        cls,
        workers: int,
        item_count: int,
        *,
        no_kcc: bool,
        detector_kind: DetectorKind,
        batch_workers_override: int | None = None,
        available_ram_gib: float | None = None,
    ) -> tuple[int, int]:
        """Resolve safe outer book concurrency (tunable or auto-detected from RAM/KCC/GPU)."""
        if item_count <= 1:
            return 1, max(1, workers)
        if batch_workers_override is not None:
            bw = min(max(1, batch_workers_override), item_count)
            return bw, max(1, workers // bw)
        if not no_kcc:
            avail = available_ram_gib if available_ram_gib is not None else cls._available_ram_gib()
            if workers >= 2 and (avail is None or avail >= 8.0):
                bw = min(2, item_count)
                return bw, max(1, workers // bw)
            return 1, max(1, workers)
        batch_workers = min(max(1, workers), item_count)
        per_book_workers = max(1, workers // batch_workers)
        return batch_workers, per_book_workers

    @classmethod
    def resolve_single_output(
        cls,
        input_path: Path,
        output_path: Path | None,
        *,
        beautify: bool = False,
    ) -> Path:
        """Resolve exact output .kfx path for a single archive or image folder."""
        base_name = cls.resolve_input_name(input_path, beautify=beautify)
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
        panel_merge: PanelMergeConfig | None = None,
        padding: float = DEFAULT_PADDING,
        on_page_progress: Callable[[int, int], None] | None = None,
    ) -> list[PageAsset]:
        sorter = PanelSorter(direction=direction)
        total_pages = len(pages)
        if on_page_progress is not None and total_pages > 0:
            on_page_progress(0, total_pages)

        active = registry.active_backend
        if active is not None and active.kind == DetectorKind.MAGI and total_pages > 1:
            paths = [page.processed_path or page.source_path for page in pages]
            widths = [page.width for page in pages]
            spreads = [page.is_spread for page in pages]
            panel_lists = registry.detect_batch_and_sort(
                paths,
                sorter,
                page_widths=widths,
                is_spreads=spreads,
                on_progress=on_page_progress,
            )
            annotated = [
                page.model_copy(update={"panels": panels})
                for page, panels in zip(pages, panel_lists)
            ]
        else:
            def _process_page(page: PageAsset) -> PageAsset:
                img_path = page.processed_path or page.source_path
                panels = registry.detect_and_sort(
                    img_path,
                    sorter,
                    page_width=page.width,
                    is_spread=page.is_spread,
                )
                return page.model_copy(update={"panels": panels})

            if workers > 1 and total_pages > 1:
                annotated_slots: list[PageAsset | None] = [None] * total_pages
                completed = 0
                with ThreadPoolExecutor(max_workers=min(workers, total_pages)) as executor:
                    future_to_idx = {
                        executor.submit(_process_page, page): idx
                        for idx, page in enumerate(pages)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        annotated_slots[idx] = future.result()
                        completed += 1
                        if on_page_progress is not None:
                            on_page_progress(completed, total_pages)
                annotated = [slot for slot in annotated_slots if slot is not None]
            else:
                annotated = []
                for idx, page in enumerate(pages, start=1):
                    annotated.append(_process_page(page))
                    if on_page_progress is not None:
                        on_page_progress(idx, total_pages)

        if panel_merge is not None and panel_merge.enabled:
            merger = PanelMerger(panel_merge)
            return [
                page.model_copy(
                    update={
                        "panels": merger.merge(
                            page.panels,
                            page_width=page.width,
                            page_height=page.height,
                            is_spread=page.is_spread,
                            padding=padding,
                        )
                    }
                )
                for page in annotated
            ]
        return annotated

    def _convert_single(
        self,
        config: ConversionConfig,
        *,
        input_name: str,
        output_kfx: Path,
        calibre_path: Path,
        registry: BackendRegistry,
        title: str | None = None,
        author: str | None = None,
        progress: ProgressReporter | None = None,
        book_id: str | None = None,
    ) -> Path:
        """Execute extraction, preprocessing, detection, KPF packaging, and KFX compilation."""
        book_title = (title or "").strip() or input_name
        reporter = progress or self._progress
        item_id = book_id or str(config.input_path)
        work_dir = Path(tempfile.mkdtemp(prefix="manga2kindle_"))

        try:
            reporter.on_book_stage(
                ProgressEvent(item_id, input_name, ConversionStage.EXTRACTING)
            )
            pages = self.extractor.extract(config.input_path, work_dir)
            reporter.on_book_stage(
                ProgressEvent(
                    item_id,
                    input_name,
                    ConversionStage.PREPROCESSING,
                    total=len(pages),
                )
            )
            processed = self.preprocessor.process(pages, work_dir, config)
            annotated = self._detect_and_sort_pages(
                processed,
                registry=registry,
                direction=config.direction,
                workers=config.workers,
                panel_merge=config.panel_merge,
                padding=config.detector.padding,
                on_page_progress=lambda done, total: reporter.on_book_stage(
                    ProgressEvent(
                        item_id,
                        input_name,
                        ConversionStage.DETECTING,
                        completed=done,
                        total=total,
                    )
                ),
            )

            reporter.on_book_stage(
                ProgressEvent(
                    item_id,
                    input_name,
                    ConversionStage.PACKAGING,
                    total=len(annotated),
                )
            )
            kpf_path = work_dir / f"{input_name}.kpf"
            self.builder.build(
                pages=annotated,
                title=book_title,
                config=config,
                output_kpf=kpf_path,
                author=author,
            )

            reporter.on_book_stage(
                ProgressEvent(item_id, input_name, ConversionStage.COMPILING)
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

    def run_single(
        self,
        config: ConversionConfig,
        *,
        title: str | None = None,
        author: str | None = None,
        registry: BackendRegistry | None = None,
        progress: ProgressReporter | None = None,
    ) -> Path:
        """Convert a single CBZ/CBR archive or image folder into a verified PDOC KFX."""
        self.last_single_skipped = False
        if not config.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {config.input_path}")
        if (
            config.input_path.is_file()
            and config.input_path.suffix.lower() not in SUPPORTED_ARCHIVE_EXTENSIONS
        ):
            raise ValueError(
                f"Unsupported input format '{config.input_path.suffix}' for {config.input_path}"
            )

        reporter = progress or self._progress
        book_id = str(config.input_path)
        effective_beautify = config.effective_beautify
        input_name = self.resolve_input_name(config.input_path, beautify=effective_beautify)
        output_kfx = self.resolve_single_output(
            config.input_path,
            config.output_path,
            beautify=effective_beautify,
        )
        if config.skip_existing and output_kfx.is_file():
            logger.info("Skipping existing output: %s", output_kfx)
            self.last_single_skipped = True
            reporter.on_book_skip(book_id, input_name, output_kfx)
            return output_kfx

        calibre_path = self.discovery.find_calibre(override=config.calibre_path)
        if calibre_path is None:
            raise KfxCompilationError(
                "calibre-debug not found. Install Calibre with the KFX Output plugin "
                "or pass --calibre-path."
            )

        active_registry = self._resolve_registry(config.detector, registry)
        if config.beautify_source and config.input_path.is_file():
            renamed_input = beautify_source_file(config.input_path)
            config = config.model_copy(update={"input_path": renamed_input})

        reporter.on_book_start(book_id, input_name)
        try:
            out_path = self._convert_single(
                config,
                input_name=input_name,
                output_kfx=output_kfx,
                calibre_path=calibre_path,
                registry=active_registry,
                title=title,
                author=author,
                progress=reporter,
                book_id=book_id,
            )
        except Exception as exc:
            reporter.on_book_error(book_id, config.input_path.name, exc)
            raise
        reporter.on_book_complete(book_id, input_name, out_path)
        return out_path

    def run_batch(
        self,
        config: ConversionConfig,
        *,
        author: str | None = None,
        on_success: Callable[[Path, Path], None] | None = None,
        on_error: Callable[[Path, Exception], None] | None = None,
        on_skip: Callable[[Path, Path], None] | None = None,
        progress: ProgressReporter | None = None,
    ) -> list[Path]:
        """Convert all CBZ/CBR archives in config.input_path concurrently."""
        archives = self._find_archives(config.input_path)
        if not archives:
            raise ValueError(f"No .cbz or .cbr archives found in {config.input_path}")

        out_dir = self.resolve_batch_output_dir(config.input_path, config.output_path)
        effective_beautify = config.effective_beautify
        seen_names: dict[str, Path] = {}
        seen_source_names: dict[str, Path] = {}
        planned: list[tuple[Path, str, Path]] = []
        for archive_path in archives:
            target_stem = beautify_name(archive_path.stem) if effective_beautify else archive_path.stem
            out_kfx = out_dir / f"{target_stem}.kfx"
            key = out_kfx.name.lower()
            if key in seen_names:
                raise ValueError(
                    f"Batch output filename collision for '{out_kfx.name}': "
                    f"'{seen_names[key].name}' and '{archive_path.name}'"
                )
            seen_names[key] = archive_path
            if config.beautify_source:
                target_src_name = f"{target_stem}{archive_path.suffix}"
                src_key = target_src_name.lower()
                if src_key in seen_source_names:
                    raise ValueError(
                        f"Batch source filename collision for '{target_src_name}': "
                        f"'{seen_source_names[src_key].name}' and '{archive_path.name}'"
                    )
                seen_source_names[src_key] = archive_path
            planned.append((archive_path, target_stem, out_kfx))

        reporter = progress or self._progress
        self.last_batch_errors = []
        self.last_batch_skipped = []
        reporter.on_batch_start(len(planned), label="Converting")
        try:
            pending_planned: list[tuple[Path, str, Path]] = []
            for archive_path, target_stem, out_kfx in planned:
                if config.skip_existing and out_kfx.is_file():
                    logger.info("Skipping existing output: %s", out_kfx)
                    self.last_batch_skipped.append(out_kfx)
                    reporter.on_book_skip(str(archive_path), target_stem, out_kfx)
                    if on_skip is not None:
                        on_skip(archive_path, out_kfx)
                else:
                    pending_planned.append((archive_path, target_stem, out_kfx))

            if not pending_planned:
                return []

            calibre_path = self.discovery.find_calibre(override=config.calibre_path)
            if calibre_path is None:
                raise KfxCompilationError(
                    "calibre-debug not found. Install Calibre with the KFX Output plugin "
                    "or pass --calibre-path."
                )

            batch_registry = self._resolve_registry(config.detector)

            if config.beautify_source:
                for archive_path, target_stem, _ in pending_planned:
                    if target_stem != archive_path.stem:
                        target_src = archive_path.with_name(f"{target_stem}{archive_path.suffix}")
                        if target_src.exists() and target_src.resolve() != archive_path.resolve():
                            raise FileExistsError(
                                f"Cannot beautify '{archive_path.name}': target '{target_src.name}' already exists."
                            )
                pending_targets = [
                    (beautify_source_file(archive_path), target_stem, out_kfx)
                    for archive_path, target_stem, out_kfx in pending_planned
                ]
            else:
                pending_targets = pending_planned

            out_dir.mkdir(parents=True, exist_ok=True)

            batch_workers, per_book_workers = self._resolve_batch_workers(
                config.workers,
                len(pending_targets),
                no_kcc=config.no_kcc,
                detector_kind=config.detector.kind,
                batch_workers_override=config.batch_workers,
            )

            def _convert_one(
                idx: int, archive_path: Path, target_stem: str, target_kfx: Path
            ) -> tuple[int, Path, str, Path | None, Exception | None]:
                book_id = str(archive_path)
                reporter.on_book_start(book_id, target_stem)
                book_config = config.model_copy(
                    update={
                        "input_path": archive_path,
                        "output_path": target_kfx,
                        "calibre_path": calibre_path,
                        "workers": per_book_workers,
                    }
                )
                try:
                    out_kfx = self._convert_single(
                        book_config,
                        input_name=target_stem,
                        output_kfx=target_kfx,
                        calibre_path=calibre_path,
                        registry=batch_registry,
                        title=target_stem,
                        author=author,
                        progress=reporter,
                        book_id=book_id,
                    )
                    return idx, archive_path, target_stem, out_kfx, None
                except Exception as exc:  # noqa: BLE001 - isolate per-book failures in batch mode
                    logger.error("Failed to convert %s: %s", archive_path, exc)
                    return idx, archive_path, target_stem, None, exc

            indexed_succeeded: list[tuple[int, Path]] = []
            indexed_errors: list[tuple[int, Path, Exception]] = []

            def _handle_outcome(
                outcome: tuple[int, Path, str, Path | None, Exception | None],
            ) -> None:
                idx, archive_path, target_stem, out_kfx, exc = outcome
                book_id = str(archive_path)
                if exc is not None or out_kfx is None:
                    err = exc if exc is not None else RuntimeError("Unknown conversion failure")
                    indexed_errors.append((idx, archive_path, err))
                    reporter.on_book_error(book_id, archive_path.name, err)
                    if on_error is not None:
                        on_error(archive_path, err)
                else:
                    indexed_succeeded.append((idx, out_kfx))
                    reporter.on_book_complete(book_id, target_stem, out_kfx)
                    if on_success is not None:
                        on_success(archive_path, out_kfx)

            if batch_workers > 1:
                with ThreadPoolExecutor(max_workers=batch_workers) as executor:
                    futures = [
                        executor.submit(_convert_one, idx, arch, stem, tgt)
                        for idx, (arch, stem, tgt) in enumerate(pending_targets)
                    ]
                    for future in as_completed(futures):
                        _handle_outcome(future.result())
            else:
                for idx, (arch, stem, tgt) in enumerate(pending_targets):
                    _handle_outcome(_convert_one(idx, arch, stem, tgt))

            indexed_succeeded.sort(key=lambda item: item[0])
            indexed_errors.sort(key=lambda item: item[0])
            self.last_batch_errors = [(arch, err) for _, arch, err in indexed_errors]
            return [out_kfx for _, out_kfx in indexed_succeeded]
        finally:
            reporter.on_batch_finish()

    def run(
        self,
        config: ConversionConfig,
        *,
        title: str | None = None,
        author: str | None = None,
        on_success: Callable[[Path, Path], None] | None = None,
        on_error: Callable[[Path, Exception], None] | None = None,
        on_skip: Callable[[Path, Path], None] | None = None,
        progress: ProgressReporter | None = None,
    ) -> list[Path]:
        """Auto-disambiguate single vs batch input and execute conversion."""
        if self.is_batch_input(config.input_path):
            return self.run_batch(
                config,
                author=author,
                on_success=on_success,
                on_error=on_error,
                on_skip=on_skip,
                progress=progress,
            )
        out_kfx = self.run_single(config, title=title, author=author, progress=progress)
        if self.last_single_skipped:
            if on_skip is not None:
                on_skip(config.input_path, out_kfx)
            return []
        return [out_kfx]

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
        model_target = infer_config_model_target(detector_config)
        if model_target is not None and model_target not in unique_kinds:
            raise ValueError(
                f"Configured detector model targets backend '{model_target.value}', "
                f"which is not in --compare ({', '.join(k.value for k in unique_kinds)})."
            )
        return unique_kinds

    @staticmethod
    def _render_preview_pages(
        total_pages: int,
        render_fn: Callable[[int], Path],
        *,
        workers: int,
        reporter: ProgressReporter,
        book_id: str,
        book_name: str,
    ) -> list[Path]:
        if total_pages > 0:
            reporter.on_book_stage(
                ProgressEvent(
                    book_id,
                    book_name,
                    ConversionStage.RENDERING,
                    completed=0,
                    total=total_pages,
                )
            )
        if workers > 1 and total_pages > 1:
            slots: list[Path | None] = [None] * total_pages
            completed = 0
            with ThreadPoolExecutor(max_workers=min(workers, total_pages)) as executor:
                future_to_idx = {
                    executor.submit(render_fn, idx): idx
                    for idx in range(total_pages)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    slots[idx] = future.result()
                    completed += 1
                    reporter.on_book_stage(
                        ProgressEvent(
                            book_id,
                            book_name,
                            ConversionStage.RENDERING,
                            completed=completed,
                            total=total_pages,
                        )
                    )
            return [path for path in slots if path is not None]

        rendered: list[Path] = []
        for idx in range(total_pages):
            rendered.append(render_fn(idx))
            reporter.on_book_stage(
                ProgressEvent(
                    book_id,
                    book_name,
                    ConversionStage.RENDERING,
                    completed=idx + 1,
                    total=total_pages,
                )
            )
        return rendered

    def preview_panels(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        profile: DeviceProfile | None = None,
        direction: ReadingDirection = ReadingDirection.RTL,
        spread_mode: SpreadMode = SpreadMode.SPLIT,
        padding: float = 0.04,
        min_panel_area: float = 0.03,
        detector: DetectorConfig | None = None,
        detector_config: DetectorConfig | None = None,
        panel_merge: PanelMergeConfig | None = None,
        compare: list[DetectorKind] | None = None,
        no_kcc: bool = False,
        keep_temp: bool = False,
        workers: int = 4,
        kcc_path: Path | None = None,
        cover_size: tuple[int, int] = DEFAULT_COVER_SIZE,
        registry: BackendRegistry | None = None,
        progress: ProgressReporter | None = None,
        book_id: str | None = None,
        book_name: str | None = None,
        _managed_lifecycle: bool = False,
    ) -> list[Path]:
        """Extract, preprocess, detect/sort panels, and render numbered PNG debug overlays."""
        resolved_profile = profile or DeviceProfile.from_name(DEFAULT_PROFILE_NAME)
        detector_cfg = (
            detector_config
            or detector
            or DetectorConfig(padding=padding, min_panel_area=min_panel_area)
        )
        resolved_merge = (
            panel_merge
            if panel_merge is not None
            else PanelMergeConfig.for_profile(
                resolved_profile,
                enabled=detector_cfg.merge_bridged_panels,
            )
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
            panel_merge=resolved_merge,
            no_kcc=no_kcc,
            keep_temp=keep_temp,
            workers=workers,
            kcc_path=kcc_path,
            cover_size=cover_size,
        )

        compare_registries: dict[DetectorKind, BackendRegistry] | None = None
        active_registry: BackendRegistry | None = None
        if compare_kinds is not None:
            model_target = infer_config_model_target(config.detector)
            compare_registries = {}
            for kind in compare_kinds:
                if model_target is not None and kind != model_target:
                    kind_cfg = config.detector.model_copy(
                        update={"kind": kind, "model_path": None, "model_id": None}
                    )
                else:
                    kind_cfg = config.detector.model_copy(update={"kind": kind})
                reg = BackendRegistry(kind_cfg)
                reg.resolve()
                compare_registries[kind] = reg
        else:
            active_registry = self._resolve_registry(config.detector, registry)

        reporter = progress or self._progress
        item_id = book_id or str(input_path)
        display_name = book_name or self.resolve_input_name(input_path)
        if not _managed_lifecycle:
            reporter.on_book_start(item_id, display_name)

        work_dir = Path(tempfile.mkdtemp(prefix="manga2kindle_preview_"))
        try:
            reporter.on_book_stage(
                ProgressEvent(item_id, display_name, ConversionStage.EXTRACTING)
            )
            pages = self.extractor.extract(input_path, work_dir)
            reporter.on_book_stage(
                ProgressEvent(
                    item_id,
                    display_name,
                    ConversionStage.PREPROCESSING,
                    total=len(pages),
                )
            )
            processed = self.preprocessor.process(pages, work_dir, config)
            renderer = PanelPreviewRenderer()
            pad = max(4, len(str(len(processed))))

            if compare_registries is not None and compare_kinds is not None:
                total_compare_pages = len(compare_kinds) * len(processed)
                annotated_by_kind: dict[DetectorKind, list[PageAsset]] = {}
                for kind_idx, (kind, reg) in enumerate(compare_registries.items()):
                    offset = kind_idx * len(processed)
                    annotated_by_kind[kind] = self._detect_and_sort_pages(
                        processed,
                        registry=reg,
                        direction=direction,
                        workers=workers,
                        panel_merge=config.panel_merge,
                        padding=config.detector.padding,
                        on_page_progress=lambda done, _t, base=offset: reporter.on_book_stage(
                            ProgressEvent(
                                item_id,
                                display_name,
                                ConversionStage.DETECTING,
                                completed=base + done,
                                total=total_compare_pages,
                            )
                        ),
                    )

                def _render_compare_page(page_idx: int) -> Path:
                    base_page = processed[page_idx]
                    img_path = base_page.processed_path or base_page.source_path
                    out_file = output_dir / f"page_{base_page.index + 1:0{pad}d}.png"
                    results: dict[DetectorKind, list[PanelBox]] = {
                        kind: annotated_by_kind[kind][page_idx].panels
                        for kind in compare_kinds
                    }
                    return renderer.render_comparison(img_path, results, out_file)

                rendered = self._render_preview_pages(
                    len(processed),
                    _render_compare_page,
                    workers=workers,
                    reporter=reporter,
                    book_id=item_id,
                    book_name=display_name,
                )
            else:
                assert active_registry is not None
                annotated = self._detect_and_sort_pages(
                    processed,
                    registry=active_registry,
                    direction=direction,
                    workers=workers,
                    panel_merge=config.panel_merge,
                    padding=config.detector.padding,
                    on_page_progress=lambda done, total: reporter.on_book_stage(
                        ProgressEvent(
                            item_id,
                            display_name,
                            ConversionStage.DETECTING,
                            completed=done,
                            total=total,
                        )
                    ),
                )

                def _render_page(page_idx: int) -> Path:
                    page = annotated[page_idx]
                    img_path = page.processed_path or page.source_path
                    out_file = output_dir / f"page_{page.index + 1:0{pad}d}.png"
                    return renderer.render(img_path, page.panels, out_file)

                rendered = self._render_preview_pages(
                    len(annotated),
                    _render_page,
                    workers=workers,
                    reporter=reporter,
                    book_id=item_id,
                    book_name=display_name,
                )
        except Exception as exc:
            if not _managed_lifecycle:
                reporter.on_book_error(item_id, input_path.name, exc)
            raise
        finally:
            if not keep_temp:
                shutil.rmtree(work_dir, ignore_errors=True)

        if not _managed_lifecycle:
            reporter.on_book_complete(item_id, display_name, output_dir)
        return rendered

    def preview_batch(
        self,
        config: ConversionConfig,
        *,
        on_success: Callable[[Path, Path], None] | None = None,
        on_error: Callable[[Path, Exception], None] | None = None,
        progress: ProgressReporter | None = None,
    ) -> list[Path]:
        """Render panel preview overlays for all CBZ/CBR archives in config.input_path."""
        archives = self._find_archives(config.input_path)
        if not archives:
            raise ValueError(f"No .cbz or .cbr archives found in {config.input_path}")

        out_dir = self.resolve_batch_output_dir(config.input_path, config.output_path)
        seen_preview_dirs: dict[str, Path] = {}
        preview_targets: list[tuple[Path, str, Path]] = []
        for archive_path in archives:
            stem = beautify_name(archive_path.stem) if config.effective_beautify else archive_path.stem
            preview_dir = out_dir / f"{stem}_preview"
            key = preview_dir.name.lower()
            if key in seen_preview_dirs:
                raise ValueError(
                    f"Batch preview directory collision for '{preview_dir.name}': "
                    f"'{seen_preview_dirs[key].name}' and '{archive_path.name}'"
                )
            seen_preview_dirs[key] = archive_path
            preview_targets.append((archive_path, stem, preview_dir))

        reporter = progress or self._progress
        self.last_batch_errors = []
        reporter.on_batch_start(len(preview_targets), label="Rendering previews")
        try:
            batch_registry = self._resolve_registry(config.detector)
            out_dir.mkdir(parents=True, exist_ok=True)

            batch_workers, per_book_workers = self._resolve_batch_workers(
                config.workers,
                len(preview_targets),
                no_kcc=config.no_kcc,
                detector_kind=config.detector.kind,
                batch_workers_override=config.batch_workers,
            )

            def _preview_one(
                idx: int, archive_path: Path, stem: str, preview_dir: Path
            ) -> tuple[int, Path, str, Path | None, Exception | None]:
                book_id = str(archive_path)
                reporter.on_book_start(book_id, stem)
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
                        panel_merge=config.panel_merge,
                        no_kcc=config.no_kcc,
                        keep_temp=config.keep_temp,
                        workers=per_book_workers,
                        kcc_path=config.kcc_path,
                        cover_size=config.cover_size,
                        registry=batch_registry,
                        progress=reporter,
                        book_id=book_id,
                        book_name=stem,
                        _managed_lifecycle=True,
                    )
                    return idx, archive_path, stem, preview_dir, None
                except Exception as exc:  # noqa: BLE001 - isolate per-book failures in batch mode
                    logger.error("Failed to preview %s: %s", archive_path, exc)
                    return idx, archive_path, stem, None, exc

            indexed_succeeded: list[tuple[int, Path]] = []
            indexed_errors: list[tuple[int, Path, Exception]] = []

            def _handle_outcome(
                outcome: tuple[int, Path, str, Path | None, Exception | None],
            ) -> None:
                idx, archive_path, stem, preview_dir, exc = outcome
                book_id = str(archive_path)
                if exc is not None or preview_dir is None:
                    err = exc if exc is not None else RuntimeError("Unknown preview failure")
                    indexed_errors.append((idx, archive_path, err))
                    reporter.on_book_error(book_id, archive_path.name, err)
                    if on_error is not None:
                        on_error(archive_path, err)
                else:
                    indexed_succeeded.append((idx, preview_dir))
                    reporter.on_book_complete(book_id, stem, preview_dir)
                    if on_success is not None:
                        on_success(archive_path, preview_dir)

            if batch_workers > 1:
                with ThreadPoolExecutor(max_workers=batch_workers) as executor:
                    futures = [
                        executor.submit(_preview_one, idx, arch, stem, pdir)
                        for idx, (arch, stem, pdir) in enumerate(preview_targets)
                    ]
                    for future in as_completed(futures):
                        _handle_outcome(future.result())
            else:
                for idx, (arch, stem, pdir) in enumerate(preview_targets):
                    _handle_outcome(_preview_one(idx, arch, stem, pdir))

            indexed_succeeded.sort(key=lambda item: item[0])
            indexed_errors.sort(key=lambda item: item[0])
            self.last_batch_errors = [(arch, err) for _, arch, err in indexed_errors]
            return [preview_dir for _, preview_dir in indexed_succeeded]
        finally:
            reporter.on_batch_finish()
