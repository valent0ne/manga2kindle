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
from rich.table import Table

from manga2kindle import __version__
from manga2kindle.cv import benchmark as cv_benchmark
from manga2kindle.cv import models_hub
from manga2kindle.cv.backends import resolve_auto_chain
from manga2kindle.cv.benchmark import BenchmarkDatasetKind, BenchmarkMetrics
from manga2kindle.models import (
    DEFAULT_COVER_SIZE,
    DEFAULT_PROFILE_NAME,
    ConversionConfig,
    DetectorConfig,
    DetectorKind,
    DevicePreference,
    DeviceProfile,
    ReadingDirection,
    SpreadMode,
)
from manga2kindle.pipeline import ConversionPipeline
from manga2kindle.platform import PlatformDiscovery

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
        if detector == DetectorKind.MAGI:
            model_path, model_id = models_hub.resolve_magi_model_spec(detector_model)
        else:
            model_path, model_id = models_hub.resolve_model_spec(detector_model)

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


def _parse_compare_backends(raw: str | None) -> list[DetectorKind] | None:
    """Parse a comma-separated --compare backend list into DetectorKind enums."""
    if raw is None:
        return None
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not parts:
        raise typer.BadParameter("--compare requires at least one backend (e.g. contour,onnx).")
    kinds: list[DetectorKind] = []
    for part in parts:
        try:
            kinds.append(DetectorKind(part))
        except ValueError as exc:
            valid = ", ".join(k.value for k in DetectorKind if k != DetectorKind.AUTO)
            raise typer.BadParameter(
                f"Unknown detector backend '{part}' in --compare. Choose from: {valid}"
            ) from exc
    return kinds


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
        console.print(f"manga2kindle {__version__}")
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
    zoom_factor: Annotated[
        float,
        typer.Option("-z", "--zoom-factor", min=1.0, help="Guided View magnification zoom factor."),
    ] = 1.5,
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
            help="Merge adjacent panels bridged by shared dialogue bubbles or substantial overlap.",
        ),
    ] = True,
) -> None:
    """Convert CBZ, CBR, image folder, or batch directory into PDOC-tagged KFX."""
    device_profile = _resolve_profile(profile, custom_size)
    resolved_cover_size = _resolve_cover_size(cover_size)
    effective_beautify = beautify or beautify_source
    pipeline = ConversionPipeline()

    try:
        is_batch = pipeline.is_batch_input(input_path)
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
            zoom_factor=zoom_factor,
            padding=padding,
            min_panel_area=min_panel_area,
            detector=detector_cfg,
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
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if preview_only:
        if is_batch:
            try:
                preview_dirs = pipeline.preview_batch(config)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Batch preview failed:[/red] {exc}")
                raise typer.Exit(code=1) from exc

            for preview_dir in preview_dirs:
                console.print(f"[green]Rendered preview(s):[/green] {preview_dir}")
            for failed_archive, err in pipeline.last_batch_errors:
                console.print(f"[red]Failed:[/red] {failed_archive.name} - {err}")

            total = len(preview_dirs) + len(pipeline.last_batch_errors)
            console.print(
                f"[bold]Batch preview summary:[/bold] {len(preview_dirs)}/{total} succeeded, "
                f"{len(pipeline.last_batch_errors)} failed."
            )
            if pipeline.last_batch_errors:
                raise typer.Exit(code=1)
            return

        try:
            rendered = pipeline.preview_panels(
                input_path=input_path,
                output_dir=resolved_output,
                profile=device_profile,
                direction=direction,
                spread_mode=spread_mode,
                padding=padding,
                min_panel_area=min_panel_area,
                detector_config=detector_cfg,
                no_kcc=no_kcc,
                keep_temp=keep_temp,
                workers=workers,
                kcc_path=kcc_path,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Preview failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        console.print(f"[green]Rendered {len(rendered)} panel preview(s) to:[/green] {resolved_output}")
        return

    if is_batch:
        try:
            outputs = pipeline.run_batch(config, author=author)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Batch conversion failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        for out_kfx in outputs:
            console.print(f"[green]Converted:[/green] {out_kfx}")
        for skipped_kfx in pipeline.last_batch_skipped:
            console.print(f"[yellow]Skipped (already exists):[/yellow] {skipped_kfx}")
        for failed_archive, err in pipeline.last_batch_errors:
            console.print(f"[red]Failed:[/red] {failed_archive.name} - {err}")

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
        out_kfx = pipeline.run_single(config, title=title, author=author)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Conversion failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if pipeline.last_single_skipped:
        console.print(f"[yellow]Skipped (already exists):[/yellow] {out_kfx}")
    else:
        console.print(f"[green]Converted (PDOC verified):[/green] {out_kfx}")


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
            help="Merge adjacent panels bridged by shared dialogue bubbles or substantial overlap.",
        ),
    ] = True,
) -> None:
    """Render debug overlays showing detected panels and reading-order arrows."""
    device_profile = _resolve_profile(profile, custom_size)
    compare_kinds = _parse_compare_backends(compare)
    default_name = ConversionPipeline.resolve_input_name(input_path)
    target_dir = output_dir or (input_path.resolve().parent / f"{default_name}_preview")
    if target_dir.exists() and not target_dir.is_dir():
        console.print(f"[red]Error:[/red] Output path exists and is not a directory: {target_dir}")
        raise typer.Exit(code=1)
    pipeline = ConversionPipeline()

    try:
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
        rendered = pipeline.preview_panels(
            input_path=input_path,
            output_dir=target_dir,
            profile=device_profile,
            direction=direction,
            spread_mode=spread_mode,
            padding=padding,
            min_panel_area=min_panel_area,
            detector_config=detector_cfg,
            compare=compare_kinds,
            no_kcc=no_kcc,
            keep_temp=keep_temp,
            workers=workers,
            kcc_path=kcc_path,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Preview failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Rendered {len(rendered)} panel preview(s) to:[/green] {target_dir}")


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
    table.add_row("Python", "Required", "[green]Found[/green]", py_version, sys.executable)
    table.add_row("OpenCV", "Required", "[green]Found[/green]", cv2.__version__, cv2.__file__ or "-")

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
            status.name,
            requirement,
            status_cell,
            status.version or "-",
            location,
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
            spec.key,
            DetectorKind.ONNX_YOLO.value,
            _safe_is_cached(models_hub.is_model_cached, spec.key),
            f"{spec.repo_id}@{spec.revision[:8]}",
            spec.license_notice,
        )

    for magi_spec in models_hub.list_registered_magi_models():
        table.add_row(
            magi_spec.key,
            DetectorKind.MAGI.value,
            _safe_is_cached(models_hub.is_magi_cached, magi_spec.key),
            f"{magi_spec.repo_id}@{magi_spec.revision[:8]}",
            magi_spec.license_notice,
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
    try:
        try:
            spec = models_hub.get_model_spec(name)
            dest = models_hub.fetch_model(spec.key, force=force)
            notice = spec.license_notice
        except ValueError:
            magi_spec = models_hub.get_magi_model_spec(name)
            dest = models_hub.fetch_magi_model(magi_spec.key, force=force)
            notice = magi_spec.license_notice
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Cached model:[/green] {dest}")
    console.print(f"[dim]License notice: {notice}[/dim]")


def _print_benchmark_notices(
    dataset: BenchmarkDatasetKind,
    backend_kinds: list[DetectorKind],
    detector_model: str | None,
) -> None:
    """Display dataset and model license/attribution notices before loading."""
    dataset_notice = cv_benchmark.get_dataset_license_notice(dataset)
    if dataset_notice:
        console.print(f"[dim]Dataset license notice: {dataset_notice}[/dim]")

    for kind in backend_kinds:
        if kind == DetectorKind.ONNX_YOLO:
            try:
                _, model_key = models_hub.resolve_model_spec(
                    detector_model or models_hub.DEFAULT_ONNX_MODEL_KEY
                )
                if model_key is not None:
                    spec = models_hub.get_model_spec(model_key)
                    console.print(f"[dim]Model license ({spec.key}): {spec.license_notice}[/dim]")
            except ValueError:
                pass
        elif kind == DetectorKind.MAGI:
            try:
                _, model_key = models_hub.resolve_magi_model_spec(
                    detector_model or models_hub.DEFAULT_MAGI_MODEL_KEY
                )
                if model_key is not None:
                    spec = models_hub.get_magi_model_spec(model_key)
                    console.print(f"[dim]Model license ({spec.key}): {spec.license_notice}[/dim]")
            except ValueError:
                pass


def _render_benchmark_table(results: list[BenchmarkMetrics], title: str) -> Table:
    table = Table(title=title, expand=True)
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
            row.backend.value,
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
    backend_kinds = _parse_compare_backends(backends) or [DetectorKind.CONTOUR]
    if any(k == DetectorKind.AUTO for k in backend_kinds):
        console.print("[red]Error:[/red] Specify concrete backends (contour, onnx, magi) instead of auto.")
        raise typer.Exit(code=1)

    effective_direction = direction or cv_benchmark.default_direction_for_dataset(dataset)
    _print_benchmark_notices(dataset, backend_kinds, detector_model)

    try:
        if dataset == BenchmarkDatasetKind.LOCAL:
            if pages_dir is None or truth is None:
                raise ValueError("--pages-dir and --truth are required when --dataset local is used.")
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
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    results: list[BenchmarkMetrics] = []
    for kind in backend_kinds:
        try:
            cfg = _build_detector_config(
                detector=kind,
                detector_model=detector_model,
                device=device,
                score_threshold=score_threshold,
                allow_remote_code=allow_remote_code,
                padding=0.02,
                min_panel_area=0.03,
                merge_bridged_panels=False,
            )
            backend = resolve_auto_chain(cfg)[0]
            metrics = cv_benchmark.run_benchmark(
                samples,
                backend,
                direction=effective_direction,
                warmup=warmup,
            )
            results.append(metrics)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Benchmark failed for '{kind.value}':[/red] {exc}")
            raise typer.Exit(code=1) from exc

    table = _render_benchmark_table(
        results,
        title=f"Benchmark ({dataset.value}, dir={effective_direction.value}, pages={len(samples)})",
    )
    console.print(table)

    if out is not None:
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
        console.print(f"[green]Wrote benchmark report:[/green] {out}")
