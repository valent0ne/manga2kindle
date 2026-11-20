"""Typer CLI entry point for manga2kindle."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Annotated

import cv2
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.table import Table

from manga2kindle import __version__
from manga2kindle.cv import benchmark as cv_benchmark
from manga2kindle.cv import models_hub
from manga2kindle.cv.backends import resolve_auto_chain
from manga2kindle.cv.backends.base import PanelDetectorBackend
from manga2kindle.cv.benchmark import BenchmarkDatasetKind, BenchmarkMetrics
from manga2kindle.models import (
    DEFAULT_COVER_SIZE,
    DEFAULT_PROFILE_NAME,
    ConversionConfig,
    DetectorConfig,
    DetectorKind,
    DevicePreference,
    DeviceProfile,
    PanelMergeConfig,
    ReadingDirection,
    SpreadMode,
)
from manga2kindle.pipeline import ConversionPipeline
from manga2kindle.platform import PlatformDiscovery
from manga2kindle.progress import RichProgressReporter

app = typer.Typer(
    name="manga2kindle",
    help="Convert CBZ/CBR manga into PDOC-tagged KFX files with Guided View panels.",
    no_args_is_help=True,
)
models_app = typer.Typer(
    name="models",
    help="List and download panel detection model weights.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")
console = Console()

_REQUIRED_TOOLS = frozenset({"kcc-c2e", "calibre-debug", "KFX Output"})
_TOOL_REQUIREMENTS: dict[str, str] = {
    "kcc-c2e": "Required*",
    "calibre-debug": "Required",
    "KFX Output": "Required",
    "unrar/unar/7z": "Opt (CBR)",
    "onnxruntime": "Optional",
    "torch": "Optional",
    "Cached Models": "Optional",
    "AUTO Detector": "Optional",
}


def _parse_custom_dimensions(raw: str) -> tuple[int, int]:
    parts = raw.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Expected WIDTHxHEIGHT (e.g. 1264x1680), got '{raw}'")
    return int(parts[0].strip()), int(parts[1].strip())


def _resolve_cover_size(raw: str) -> tuple[int, int]:
    try:
        width, height = _parse_custom_dimensions(raw)
        if width <= 0 or height <= 0:
            raise ValueError("Cover dimensions must be positive integers.")
        return width, height
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _resolve_profile(profile: str, custom_size: str | None) -> DeviceProfile:
    normalized = profile.strip()
    if normalized.upper() == "CUSTOM":
        if not custom_size:
            raise typer.BadParameter(
                "--custom-size WIDTHxHEIGHT is required when --profile CUSTOM is used."
            )
        try:
            dims = _parse_custom_dimensions(custom_size)
            return DeviceProfile.from_name("CUSTOM", custom_size=dims)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if custom_size is not None:
        raise typer.BadParameter(
            "--custom-size can only be used together with --profile CUSTOM."
        )

    if "x" in normalized.lower() and normalized.lower().split("x", 1)[0].isdigit():
        try:
            dims = _parse_custom_dimensions(normalized)
            return DeviceProfile.from_name("CUSTOM", custom_size=dims)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    try:
        return DeviceProfile.from_name(normalized)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _announce_detector(active: DetectorKind, chain: list[DetectorKind]) -> None:
    chain_str = " -> ".join(kind.value for kind in chain)
    if len(chain) > 1:
        console.print(
            f"[dim]Panel detector: {rich_escape(active.value)} "
            f"(fallback: {rich_escape(chain_str)})[/dim]"
        )
    else:
        console.print(f"[dim]Panel detector: {rich_escape(active.value)}[/dim]")


def _build_detector_config(
    *,
    detector: DetectorKind,
    detector_model: str | None,
    device: DevicePreference,
    score_threshold: float,
    allow_remote_code: bool,
    padding: float,
    min_panel_area: float,
    batch_size: int | None = None,
    merge_bridged_panels: bool = True,
) -> DetectorConfig:
    """Build a DetectorConfig from shared CLI detector options."""
    model_path: Path | None = None
    model_id: str | None = None
    if detector_model is not None:
        if detector == DetectorKind.CONTOUR:
            raise ValueError("--detector-model cannot be used with --detector contour.")
        target = models_hub.resolve_detector_model_target(detector_model)
        if detector != DetectorKind.AUTO and target.backend != detector:
            raise ValueError(
                f"--detector-model '{detector_model}' targets backend "
                f"'{target.backend.value}', not '{detector.value}'."
            )
        model_path = target.model_path
        model_id = target.model_id

    return DetectorConfig(
        kind=detector,
        model_path=model_path,
        model_id=model_id,
        device=device,
        score_threshold=score_threshold,
        allow_remote_code=allow_remote_code,
        padding=padding,
        min_panel_area=min_panel_area,
        batch_size=batch_size,
        merge_bridged_panels=merge_bridged_panels,
    )


def _parse_backend_list(
    raw: str | None,
    *,
    option_name: str = "--compare",
) -> list[DetectorKind] | None:
    """Parse a comma-separated backend list into deduplicated DetectorKind enums."""
    if raw is None:
        return None
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not parts:
        raise typer.BadParameter(f"{option_name} requires at least one backend (e.g. contour,onnx).")
    kinds: list[DetectorKind] = []
    for part in parts:
        try:
            kinds.append(DetectorKind(part))
        except ValueError as exc:
            valid = ", ".join(k.value for k in DetectorKind if k != DetectorKind.AUTO)
            raise typer.BadParameter(
                f"Unknown detector backend '{part}' in {option_name}. Choose from: {valid}"
            ) from exc
    return list(dict.fromkeys(kinds))


def _resolve_preview_only_dir(
    input_path: Path,
    output: Path | None,
    *,
    beautify: bool = False,
) -> Path:
    """Resolve the preview output directory for `convert --preview-only` on a single input."""
    default_name = ConversionPipeline.resolve_input_name(input_path, beautify=beautify)
    if output is None:
        return input_path.resolve().parent / f"{default_name}_preview"
    if output.suffix.lower() == ".kfx":
        return output.parent / f"{output.stem}_preview"
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {output}")
    return output


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"manga2kindle {rich_escape(__version__)}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """manga2kindle CLI."""


@app.command()
def convert(
    input_path: Annotated[
        Path,
        typer.Option("-i", "--input", exists=True, help="Input CBZ/CBR file, image folder, or batch directory."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output .kfx file or directory."),
    ] = None,
    profile: Annotated[
        str,
        typer.Option("-p", "--profile", help="Kindle profile: KPW6, KPW5, KS, OASIS, or CUSTOM."),
    ] = DEFAULT_PROFILE_NAME,
    custom_size: Annotated[
        str | None,
        typer.Option("--custom-size", help="Custom WIDTHxHEIGHT when --profile CUSTOM is set."),
    ] = None,
    direction: Annotated[
        ReadingDirection,
        typer.Option("-d", "--direction", case_sensitive=False, help="Reading direction: rtl or ltr."),
    ] = ReadingDirection.RTL,
    spread_mode: Annotated[
        SpreadMode,
        typer.Option("-s", "--spread-mode", case_sensitive=False, help="Spread handling: split, rotate, or keep."),
    ] = SpreadMode.SPLIT,
    padding: Annotated[
        float,
        typer.Option("--padding", min=0.0, max=0.25, help="Panel bounding-box padding ratio."),
    ] = 0.04,
    min_panel_area: Annotated[
        float,
        typer.Option("--min-panel-area", min=0.01, max=0.50, help="Minimum panel area ratio."),
    ] = 0.03,
    detector: Annotated[
        DetectorKind,
        typer.Option(
            "--detector",
            case_sensitive=False,
            help="Panel detector backend: auto, magi, onnx, or contour.",
        ),
    ] = DetectorKind.AUTO,
    detector_model: Annotated[
        str | None,
        typer.Option(
            "--detector-model",
            help="Detector model key, Hugging Face repo ID, or local weights path.",
        ),
    ] = None,
    device: Annotated[
        DevicePreference,
        typer.Option(
            "--device",
            case_sensitive=False,
            help="Inference device for neural detectors: auto, cpu, cuda, or mps.",
        ),
    ] = DevicePreference.AUTO,
    score_threshold: Annotated[
        float,
        typer.Option(
            "--score-threshold",
            min=0.0,
            max=1.0,
            help="Confidence threshold for neural panel detectors.",
        ),
    ] = 0.25,
    allow_remote_code: Annotated[
        bool,
        typer.Option(
            "--allow-remote-code",
            help="Allow executing remote model code (required to enable Magi).",
        ),
    ] = False,
    kcc_path: Annotated[
        Path | None,
        typer.Option("--kcc-path", help="Explicit path to kcc-c2e executable."),
    ] = None,
    no_kcc: Annotated[
        bool,
        typer.Option("--no-kcc", help="Use pure-Pillow preprocessing instead of kcc-c2e."),
    ] = False,
    keep_temp: Annotated[
        bool,
        typer.Option("--keep-temp", help="Retain temporary working directories after conversion."),
    ] = False,
    workers: Annotated[
        int,
        typer.Option("-w", "--workers", min=1, help="Parallel worker thread count."),
    ] = os.cpu_count() or 4,
    batch_workers: Annotated[
        int | None,
        typer.Option(
            "--batch-workers",
            min=1,
            help="Concurrent archives in batch mode (default: auto-detected from RAM/KCC).",
        ),
    ] = None,
    gpu_batch_size: Annotated[
        int | None,
        typer.Option(
            "--gpu-batch-size",
            min=1,
            max=64,
            help="Pages per GPU batch for neural detectors (default: auto-detected from VRAM).",
        ),
    ] = None,
    calibre_path: Annotated[
        Path | None,
        typer.Option("--calibre-path", help="Explicit path to calibre-debug executable."),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option("--title", help="Book title metadata override (defaults to archive/folder name)."),
    ] = None,
    author: Annotated[
        str | None,
        typer.Option("--author", help="Book author metadata."),
    ] = None,
    cover_size: Annotated[
        str,
        typer.Option(
            "--cover-size",
            help="Target cover WIDTHxHEIGHT (stretched to fill without borders).",
        ),
    ] = f"{DEFAULT_COVER_SIZE[0]}x{DEFAULT_COVER_SIZE[1]}",
    preview_only: Annotated[
        bool,
        typer.Option(
            "--preview-only",
            help="Render panel preview PNGs and exit without compiling KPF/KFX.",
        ),
    ] = False,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help="Skip conversion if the output .kfx file already exists.",
        ),
    ] = False,
    beautify: Annotated[
        bool,
        typer.Option(
            "--beautify",
            help="Strip release tags from the output .kfx filename and KFX title metadata.",
        ),
    ] = False,
    beautify_source: Annotated[
        bool,
        typer.Option(
            "--beautify-source",
            help="Rename source .cbz/.cbr file(s) in place and beautify output .kfx and metadata.",
        ),
    ] = False,
    merge_panels: Annotated[
        bool,
        typer.Option(
            "--merge-panels/--no-merge-panels",
            help=(
                "Merge panels bridged by dialogue bubbles or overlap and apply "
                "profile-calibrated mini-panel row/sub-column merging."
            ),
        ),
    ] = True,
) -> None:
    """Convert CBZ, CBR, image folder, or batch directory into PDOC-tagged KFX."""
    device_profile = _resolve_profile(profile, custom_size)
    resolved_cover_size = _resolve_cover_size(cover_size)
    effective_beautify = beautify or beautify_source
    probe_pipeline = ConversionPipeline()

    try:
        if preview_only and beautify_source:
            raise ValueError(
                "--beautify-source cannot be used with --preview-only; use --beautify instead."
            )
        is_batch = probe_pipeline.is_batch_input(input_path)
        if is_batch and title is not None:
            raise ValueError("--title cannot be used with a batch archive directory.")
        if output is None:
            resolved_output = (
                input_path
                if is_batch
                else (
                    _resolve_preview_only_dir(input_path, None, beautify=effective_beautify)
                    if preview_only
                    else ConversionPipeline.resolve_single_output(
                        input_path, None, beautify=effective_beautify
                    )
                )
            )
        elif is_batch:
            resolved_output = ConversionPipeline.resolve_batch_output_dir(input_path, output)
        elif preview_only:
            resolved_output = _resolve_preview_only_dir(
                input_path, output, beautify=effective_beautify
            )
        else:
            resolved_output = ConversionPipeline.resolve_single_output(
                input_path, output, beautify=effective_beautify
            )

        detector_cfg = _build_detector_config(
            detector=detector,
            detector_model=detector_model,
            device=device,
            score_threshold=score_threshold,
            allow_remote_code=allow_remote_code,
            padding=padding,
            min_panel_area=min_panel_area,
            batch_size=gpu_batch_size,
            merge_bridged_panels=merge_panels,
        )
        config = ConversionConfig(
            input_path=input_path,
            output_path=resolved_output,
            profile=device_profile,
            direction=direction,
            spread_mode=spread_mode,
            padding=padding,
            min_panel_area=min_panel_area,
            detector=detector_cfg,
            panel_merge=PanelMergeConfig.for_profile(device_profile, enabled=merge_panels),
            no_kcc=no_kcc,
            keep_temp=keep_temp,
            skip_existing=skip_existing,
            beautify=beautify,
            beautify_source=beautify_source,
            workers=workers,
            batch_workers=batch_workers,
            kcc_path=kcc_path,
            calibre_path=calibre_path,
            cover_size=resolved_cover_size,
        )
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        console.print(f"[red]Error:[/red] {rich_escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    if preview_only:
        if is_batch:
            try:
                with RichProgressReporter(
                    console, is_batch=True, complete_prefix="Rendered preview(s)"
                ) as reporter:
                    pipeline = ConversionPipeline(
                        on_detector_resolved=_announce_detector,
                        progress=reporter,
                    )
                    preview_dirs = pipeline.preview_batch(config)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Batch preview failed:[/red] {rich_escape(str(exc))}")
                raise typer.Exit(code=1) from exc

            total = len(preview_dirs) + len(pipeline.last_batch_errors)
            console.print(
                f"[bold]Batch preview summary:[/bold] {len(preview_dirs)}/{total} succeeded, "
                f"{len(pipeline.last_batch_errors)} failed."
            )
            if pipeline.last_batch_errors:
                raise typer.Exit(code=1)
            return

        try:
            with RichProgressReporter(console, is_batch=False) as reporter:
                pipeline = ConversionPipeline(
                    on_detector_resolved=_announce_detector,
                    progress=reporter,
                )
                rendered = pipeline.preview_panels(
                    input_path=input_path,
                    output_dir=resolved_output,
                    profile=device_profile,
                    direction=direction,
                    spread_mode=spread_mode,
                    padding=padding,
                    min_panel_area=min_panel_area,
                    detector_config=config.detector,
                    panel_merge=config.panel_merge,
                    no_kcc=no_kcc,
                    keep_temp=keep_temp,
                    workers=workers,
                    kcc_path=kcc_path,
                    cover_size=resolved_cover_size,
                )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Preview failed:[/red] {rich_escape(str(exc))}")
            raise typer.Exit(code=1) from exc

        console.print(
            f"[green]Rendered {len(rendered)} panel preview(s) to:[/green] "
            f"{rich_escape(str(resolved_output))}"
        )
        return

    if is_batch:
        try:
            with RichProgressReporter(
                console, is_batch=True, complete_prefix="Converted"
            ) as reporter:
                pipeline = ConversionPipeline(
                    on_detector_resolved=_announce_detector,
                    progress=reporter,
                )
                outputs = pipeline.run_batch(config, author=author)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Batch conversion failed:[/red] {rich_escape(str(exc))}")
            raise typer.Exit(code=1) from exc

        total = len(outputs) + len(pipeline.last_batch_skipped) + len(pipeline.last_batch_errors)
        if pipeline.last_batch_skipped:
            console.print(
                f"[bold]Batch summary:[/bold] {len(outputs)}/{total} succeeded, "
                f"{len(pipeline.last_batch_skipped)} skipped, "
                f"{len(pipeline.last_batch_errors)} failed."
            )
        else:
            console.print(
                f"[bold]Batch summary:[/bold] {len(outputs)}/{total} succeeded, "
                f"{len(pipeline.last_batch_errors)} failed."
            )
        if pipeline.last_batch_errors:
            raise typer.Exit(code=1)
        return

    try:
        with RichProgressReporter(console, is_batch=False) as reporter:
            pipeline = ConversionPipeline(
                on_detector_resolved=_announce_detector,
                progress=reporter,
            )
            out_kfx = pipeline.run_single(config, title=title, author=author)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Conversion failed:[/red] {rich_escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    if pipeline.last_single_skipped:
        console.print(f"[yellow]Skipped (already exists):[/yellow] {rich_escape(str(out_kfx))}")
    else:
        console.print(f"[green]Converted (PDOC verified):[/green] {rich_escape(str(out_kfx))}")


@app.command("preview-panels")
def preview_panels(
    input_path: Annotated[
        Path,
        typer.Option("-i", "--input", exists=True, help="Input CBZ, CBR, or image folder."),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("-o", "--output-dir", help="Directory for numbered PNG panel overlays."),
    ] = None,
    profile: Annotated[
        str,
        typer.Option("-p", "--profile", help="Kindle profile: KPW6, KPW5, KS, OASIS, or CUSTOM."),
    ] = DEFAULT_PROFILE_NAME,
    custom_size: Annotated[
        str | None,
        typer.Option("--custom-size", help="Custom WIDTHxHEIGHT when --profile CUSTOM is set."),
    ] = None,
    direction: Annotated[
        ReadingDirection,
        typer.Option("-d", "--direction", case_sensitive=False, help="Reading direction: rtl or ltr."),
    ] = ReadingDirection.RTL,
    spread_mode: Annotated[
        SpreadMode,
        typer.Option("-s", "--spread-mode", case_sensitive=False, help="Spread handling: split, rotate, or keep."),
    ] = SpreadMode.SPLIT,
    padding: Annotated[
        float,
        typer.Option("--padding", min=0.0, max=0.25, help="Panel bounding-box padding ratio."),
    ] = 0.04,
    min_panel_area: Annotated[
        float,
        typer.Option("--min-panel-area", min=0.01, max=0.50, help="Minimum panel area ratio."),
    ] = 0.03,
    detector: Annotated[
        DetectorKind,
        typer.Option(
            "--detector",
            case_sensitive=False,
            help="Panel detector backend: auto, magi, onnx, or contour.",
        ),
    ] = DetectorKind.AUTO,
    detector_model: Annotated[
        str | None,
        typer.Option(
            "--detector-model",
            help="Detector model key, Hugging Face repo ID, or local weights path.",
        ),
    ] = None,
    device: Annotated[
        DevicePreference,
        typer.Option(
            "--device",
            case_sensitive=False,
            help="Inference device for neural detectors: auto, cpu, cuda, or mps.",
        ),
    ] = DevicePreference.AUTO,
    score_threshold: Annotated[
        float,
        typer.Option(
            "--score-threshold",
            min=0.0,
            max=1.0,
            help="Confidence threshold for neural panel detectors.",
        ),
    ] = 0.25,
    allow_remote_code: Annotated[
        bool,
        typer.Option(
            "--allow-remote-code",
            help="Allow executing remote model code (required to enable Magi).",
        ),
    ] = False,
    kcc_path: Annotated[
        Path | None,
        typer.Option("--kcc-path", help="Explicit path to kcc-c2e executable."),
    ] = None,
    no_kcc: Annotated[
        bool,
        typer.Option("--no-kcc", help="Use pure-Pillow preprocessing instead of kcc-c2e."),
    ] = False,
    keep_temp: Annotated[
        bool,
        typer.Option("--keep-temp", help="Retain temporary staging directories."),
    ] = False,
    workers: Annotated[
        int,
        typer.Option("-w", "--workers", min=1, help="Parallel worker thread count."),
    ] = os.cpu_count() or 4,
    gpu_batch_size: Annotated[
        int | None,
        typer.Option(
            "--gpu-batch-size",
            min=1,
            max=64,
            help="Pages per GPU batch for neural detectors (default: auto-detected from VRAM).",
        ),
    ] = None,
    compare: Annotated[
        str | None,
        typer.Option(
            "--compare",
            help="Comma-separated backends to compare side-by-side (e.g. contour,onnx,magi).",
        ),
    ] = None,
    merge_panels: Annotated[
        bool,
        typer.Option(
            "--merge-panels/--no-merge-panels",
            help=(
                "Merge panels bridged by dialogue bubbles or overlap and apply "
                "profile-calibrated mini-panel row/sub-column merging."
            ),
        ),
    ] = True,
) -> None:
    """Render debug overlays showing detected panels and reading-order arrows."""
    device_profile = _resolve_profile(profile, custom_size)
    compare_kinds = _parse_backend_list(compare, option_name="--compare")
    default_name = ConversionPipeline.resolve_input_name(input_path)
    target_dir = output_dir or (input_path.resolve().parent / f"{default_name}_preview")
    if target_dir.exists() and not target_dir.is_dir():
        console.print(
            f"[red]Error:[/red] Output path exists and is not a directory: "
            f"{rich_escape(str(target_dir))}"
        )
        raise typer.Exit(code=1)
    try:
        effective_detector = detector
        if compare_kinds is not None and detector_model is not None:
            target = models_hub.resolve_detector_model_target(detector_model)
            if target.backend not in compare_kinds:
                raise ValueError(
                    f"--detector-model '{detector_model}' targets backend '{target.backend.value}', "
                    f"which is not in --compare ({', '.join(k.value for k in compare_kinds)})."
                )
            effective_detector = target.backend
        detector_cfg = _build_detector_config(
            detector=effective_detector,
            detector_model=detector_model,
            device=device,
            score_threshold=score_threshold,
            allow_remote_code=allow_remote_code,
            padding=padding,
            min_panel_area=min_panel_area,
            batch_size=gpu_batch_size,
            merge_bridged_panels=merge_panels,
        )
        with RichProgressReporter(console, is_batch=False) as reporter:
            pipeline = ConversionPipeline(
                on_detector_resolved=None if compare_kinds else _announce_detector,
                progress=reporter,
            )
            rendered = pipeline.preview_panels(
                input_path=input_path,
                output_dir=target_dir,
                profile=device_profile,
                direction=direction,
                spread_mode=spread_mode,
                padding=padding,
                min_panel_area=min_panel_area,
                detector_config=detector_cfg,
                panel_merge=PanelMergeConfig.for_profile(device_profile, enabled=merge_panels),
                compare=compare_kinds,
                no_kcc=no_kcc,
                keep_temp=keep_temp,
                workers=workers,
                kcc_path=kcc_path,
            )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Preview failed:[/red] {rich_escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Rendered {len(rendered)} panel preview(s) to:[/green] "
        f"{rich_escape(str(target_dir))}"
    )


@app.command()
def doctor(
    kcc_path: Annotated[
        Path | None,
        typer.Option("--kcc-path", help="Explicit path to kcc-c2e executable."),
    ] = None,
    calibre_path: Annotated[
        Path | None,
        typer.Option("--calibre-path", help="Explicit path to calibre-debug executable."),
    ] = None,
) -> None:
    """Check Python libraries and external CLI dependencies."""
    discovery = PlatformDiscovery()
    statuses = discovery.run_doctor(
        kcc_override=kcc_path,
        calibre_override=calibre_path,
    )

    table = Table(title="manga2kindle Dependency Check", expand=True)
    table.add_column("Dependency", style="bold", no_wrap=True)
    table.add_column("Need", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Version", overflow="fold")
    table.add_column("Location / Details", overflow="fold")

    py_version = sys.version.split()[0]
    table.add_row(
        "Python",
        "Required",
        "[green]Found[/green]",
        rich_escape(py_version),
        rich_escape(sys.executable),
    )
    table.add_row(
        "OpenCV",
        "Required",
        "[green]Found[/green]",
        rich_escape(cv2.__version__),
        rich_escape(cv2.__file__ or "-"),
    )

    missing_required = False
    for status in statuses:
        requirement = _TOOL_REQUIREMENTS.get(status.name, "Optional")
        is_required = status.name in _REQUIRED_TOOLS
        if status.found:
            status_cell = "[green]Found[/green]"
        elif is_required:
            status_cell = "[red]Missing[/red]"
            missing_required = True
        else:
            status_cell = "[yellow]Missing[/yellow]"

        location = str(status.path) if status.path is not None else (status.details or "Not found")
        table.add_row(
            rich_escape(status.name),
            rich_escape(requirement),
            status_cell,
            rich_escape(status.version or "-"),
            rich_escape(location),
        )

    console.print(table)
    console.print("[dim]* kcc-c2e is optional when --no-kcc is passed.[/dim]")
    if missing_required:
        console.print(
            "[red]Missing required dependencies.[/red] Install missing tools or pass explicit "
            "--calibre-path / --kcc-path overrides."
        )
        raise typer.Exit(code=1)


def _safe_is_cached(checker, key: str) -> str:
    try:
        return "[green]Yes[/green]" if checker(key) else "[dim]No[/dim]"
    except RuntimeError:
        return "[yellow]Unknown[/yellow]"


@models_app.command("list")
def models_list() -> None:
    """List registered panel detection models, cache status, and license notices."""
    table = Table(title="manga2kindle Registered Models", expand=True)
    table.add_column("Key", style="bold", no_wrap=True)
    table.add_column("Backend", no_wrap=True)
    table.add_column("Cached", no_wrap=True)
    table.add_column("Repository", overflow="fold")
    table.add_column("License Notice", overflow="fold")

    for spec in models_hub.list_registered_models():
        table.add_row(
            rich_escape(spec.key),
            rich_escape(DetectorKind.ONNX_YOLO.value),
            _safe_is_cached(models_hub.is_model_cached, spec.key),
            rich_escape(f"{spec.repo_id}@{spec.revision[:8]}"),
            rich_escape(spec.license_notice),
        )

    for magi_spec in models_hub.list_registered_magi_models():
        table.add_row(
            rich_escape(magi_spec.key),
            rich_escape(DetectorKind.MAGI.value),
            _safe_is_cached(models_hub.is_magi_cached, magi_spec.key),
            rich_escape(f"{magi_spec.repo_id}@{magi_spec.revision[:8]}"),
            rich_escape(magi_spec.license_notice),
        )

    console.print(table)


@models_app.command("fetch")
def models_fetch(
    name: Annotated[
        str,
        typer.Argument(help="Model key to download (e.g. manga109-yolo-s, manga109-yolo-n, magiv3)."),
    ] = models_hub.DEFAULT_ONNX_MODEL_KEY,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download even if weights are already cached."),
    ] = False,
) -> None:
    """Download and verify a registered panel detection model into the local cache."""
    onnx_spec: models_hub.ModelSpec | None = None
    magi_spec: models_hub.MagiModelSpec | None = None
    try:
        onnx_spec = models_hub.get_model_spec(name)
    except ValueError:
        try:
            magi_spec = models_hub.get_magi_model_spec(name)
        except ValueError as exc:
            supported = ", ".join(models_hub.list_all_model_keys())
            msg = f"Unknown model '{name}'. Registered models: {supported}"
            console.print(f"[red]Error:[/red] {rich_escape(msg)}")
            raise typer.Exit(code=1) from exc

    try:
        if onnx_spec is not None:
            dest = models_hub.fetch_model(onnx_spec.key, force=force)
            notice = onnx_spec.license_notice
        else:
            assert magi_spec is not None
            dest = models_hub.fetch_magi_model(magi_spec.key, force=force)
            notice = magi_spec.license_notice
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {rich_escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Cached model:[/green] {rich_escape(str(dest))}")
    console.print(f"[dim]License notice: {rich_escape(notice)}[/dim]")


def _print_benchmark_notices(
    dataset: BenchmarkDatasetKind,
    backend_kinds: list[DetectorKind],
    model_target: models_hub.ResolvedModelTarget | None,
) -> None:
    """Display dataset and model license/attribution notices before loading."""
    dataset_notice = cv_benchmark.get_dataset_license_notice(dataset)
    if dataset_notice:
        console.print(f"[dim]Dataset license notice: {rich_escape(dataset_notice)}[/dim]")

    for kind in backend_kinds:
        if model_target is not None and model_target.backend == kind and model_target.model_path is not None:
            continue
        scoped_model = (
            model_target.model_id
            if model_target is not None and model_target.backend == kind
            else None
        )
        if kind == DetectorKind.ONNX_YOLO:
            try:
                spec = models_hub.get_model_spec(
                    scoped_model or models_hub.DEFAULT_ONNX_MODEL_KEY
                )
                console.print(
                    f"[dim]Model license ({rich_escape(spec.key)}): "
                    f"{rich_escape(spec.license_notice)}[/dim]"
                )
            except ValueError:
                pass
        elif kind == DetectorKind.MAGI:
            try:
                spec = models_hub.get_magi_model_spec(
                    scoped_model or models_hub.DEFAULT_MAGI_MODEL_KEY
                )
                console.print(
                    f"[dim]Model license ({rich_escape(spec.key)}): "
                    f"{rich_escape(spec.license_notice)}[/dim]"
                )
            except ValueError:
                pass


def _render_benchmark_table(results: list[BenchmarkMetrics], title: str) -> Table:
    table = Table(title=rich_escape(title), expand=True)
    table.add_column("Backend", style="bold", no_wrap=True)
    table.add_column("Pages", justify="right")
    table.add_column("P@50", justify="right")
    table.add_column("R@50", justify="right")
    table.add_column("P@75", justify="right")
    table.add_column("R@75", justify="right")
    table.add_column("mIoU", justify="right")
    table.add_column("Dup%", justify="right")
    table.add_column("Count%", justify="right")
    table.add_column("PairAcc", justify="right")
    table.add_column("Tau", justify="right")
    table.add_column("ms/pg", justify="right")
    table.add_column("RAM MB", justify="right")
    table.add_column("VRAM MB", justify="right")

    for row in results:
        table.add_row(
            rich_escape(row.backend.value),
            str(row.pages_evaluated),
            f"{row.precision_50:.3f}",
            f"{row.recall_50:.3f}",
            f"{row.precision_75:.3f}",
            f"{row.recall_75:.3f}",
            f"{row.mean_iou:.3f}",
            f"{row.nested_duplicate_rate * 100:.1f}%",
            f"{row.exact_count_match * 100:.1f}%",
            f"{row.pairwise_order_acc:.3f}",
            f"{row.kendall_tau:.3f}",
            f"{row.ms_per_page:.1f}",
            f"{row.peak_ram_mb:.1f}",
            f"{row.peak_vram_mb:.1f}",
        )
    return table


@app.command()
def benchmark(
    dataset: Annotated[
        BenchmarkDatasetKind,
        typer.Option(
            "--dataset",
            case_sensitive=False,
            help="Benchmark dataset: popmanga, comix, or local.",
        ),
    ] = BenchmarkDatasetKind.POPMANGA,
    split: Annotated[
        str,
        typer.Option("--split", help="Dataset split for popmanga or comix."),
    ] = "test",
    pages_dir: Annotated[
        Path | None,
        typer.Option("--pages-dir", help="Page images directory (required when --dataset local)."),
    ] = None,
    truth: Annotated[
        Path | None,
        typer.Option("--truth", help="Ground-truth JSON/COCO file (required when --dataset local)."),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Optional local cache directory for HF datasets."),
    ] = None,
    backends: Annotated[
        str,
        typer.Option(
            "--backends",
            help="Comma-separated backends to evaluate (e.g. contour,onnx,magi).",
        ),
    ] = "contour",
    direction: Annotated[
        ReadingDirection | None,
        typer.Option(
            "-d",
            "--direction",
            case_sensitive=False,
            help="Reading direction override (defaults to rtl for popmanga/local, ltr for comix).",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Maximum number of pages to evaluate."),
    ] = None,
    warmup: Annotated[
        int,
        typer.Option("--warmup", min=0, help="Warmup pages excluded from timing."),
    ] = 1,
    device: Annotated[
        DevicePreference,
        typer.Option(
            "--device",
            case_sensitive=False,
            help="Inference device for neural detectors: auto, cpu, cuda, or mps.",
        ),
    ] = DevicePreference.AUTO,
    score_threshold: Annotated[
        float,
        typer.Option(
            "--score-threshold",
            min=0.0,
            max=1.0,
            help="Confidence threshold for neural panel detectors.",
        ),
    ] = 0.25,
    allow_remote_code: Annotated[
        bool,
        typer.Option(
            "--allow-remote-code",
            help="Allow executing remote model code (required to enable Magi).",
        ),
    ] = False,
    detector_model: Annotated[
        str | None,
        typer.Option(
            "--detector-model",
            help="Detector model key, Hugging Face repo ID, or local weights path.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional path to write JSON benchmark report."),
    ] = None,
) -> None:
    """Evaluate panel detection backends against PopManga, CoMix, or a local dataset."""
    backend_kinds = _parse_backend_list(backends, option_name="--backends") or [DetectorKind.CONTOUR]
    if any(k == DetectorKind.AUTO for k in backend_kinds):
        console.print("[red]Error:[/red] Specify concrete backends (contour, onnx, magi) instead of auto.")
        raise typer.Exit(code=1)

    model_target: models_hub.ResolvedModelTarget | None = None
    try:
        if out is not None and out.exists() and out.is_dir():
            raise ValueError(f"--out path exists and is a directory: {out}")

        if dataset == BenchmarkDatasetKind.LOCAL:
            if pages_dir is None or truth is None:
                raise ValueError("--pages-dir and --truth are required when --dataset local is used.")
            if not pages_dir.is_dir():
                raise FileNotFoundError(f"Benchmark pages directory does not exist: {pages_dir}")
            if not truth.is_file():
                raise FileNotFoundError(f"Benchmark truth file does not exist: {truth}")

        if detector_model is not None:
            model_target = models_hub.resolve_detector_model_target(detector_model)
            if model_target.backend not in backend_kinds:
                raise ValueError(
                    f"--detector-model '{detector_model}' targets backend "
                    f"'{model_target.backend.value}', which is not in --backends "
                    f"({', '.join(k.value for k in backend_kinds)})."
                )

        if DetectorKind.MAGI in backend_kinds and not allow_remote_code:
            raise RuntimeError(
                "MagiBackend executes custom model code from ragavsachdeva/magiv3 "
                "(trust_remote_code=True). Pass --allow-remote-code to enable Magi."
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {rich_escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    effective_direction = direction or cv_benchmark.default_direction_for_dataset(dataset)
    _print_benchmark_notices(dataset, backend_kinds, model_target)

    resolved_backends: list[tuple[DetectorKind, PanelDetectorBackend]] = []
    for kind in backend_kinds:
        try:
            scoped_model = (
                detector_model
                if model_target is not None and model_target.backend == kind
                else None
            )
            cfg = _build_detector_config(
                detector=kind,
                detector_model=scoped_model,
                device=device,
                score_threshold=score_threshold,
                allow_remote_code=allow_remote_code,
                padding=0.02,
                min_panel_area=0.03,
                merge_bridged_panels=False,
            )
            backend = resolve_auto_chain(cfg)[0]
            resolved_backends.append((kind, backend))
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[red]Benchmark failed for '{rich_escape(kind.value)}':[/red] "
                f"{rich_escape(str(exc))}"
            )
            raise typer.Exit(code=1) from exc

    try:
        if dataset == BenchmarkDatasetKind.LOCAL:
            assert pages_dir is not None and truth is not None
            samples = cv_benchmark.load_local_dataset(pages_dir, truth, limit=limit)
        elif dataset == BenchmarkDatasetKind.POPMANGA:
            samples = cv_benchmark.load_popmanga_split(
                split,
                cache_dir=cache_dir,
                limit=limit,
                trust_remote_code=allow_remote_code,
            )
        else:
            samples = cv_benchmark.load_comix_split(
                split,
                cache_dir=cache_dir,
                limit=limit,
                trust_remote_code=allow_remote_code,
            )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {rich_escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    results: list[BenchmarkMetrics] = []
    for kind, backend in resolved_backends:
        try:
            metrics = cv_benchmark.run_benchmark(
                samples,
                backend,
                direction=effective_direction,
                warmup=warmup,
            )
            results.append(metrics)
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[red]Benchmark failed for '{rich_escape(kind.value)}':[/red] "
                f"{rich_escape(str(exc))}"
            )
            raise typer.Exit(code=1) from exc

    table = _render_benchmark_table(
        results,
        title=f"Benchmark ({dataset.value}, dir={effective_direction.value}, pages={len(samples)})",
    )
    console.print(table)

    if out is not None:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dataset": dataset.value,
                "direction": effective_direction.value,
                "pages_evaluated": len(samples),
                "results": [
                    {**dataclasses.asdict(item), "backend": item.backend.value}
                    for item in results
                ],
            }
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            console.print(
                f"[red]Failed to write benchmark report:[/red] {rich_escape(str(exc))}"
            )
            raise typer.Exit(code=1) from exc
        console.print(f"[green]Wrote benchmark report:[/green] {rich_escape(str(out))}")
