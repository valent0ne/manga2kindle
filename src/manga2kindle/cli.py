"""Typer CLI entry point for manga2kindle."""

from __future__ import annotations

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
from manga2kindle.models import (
    ConversionConfig,
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
console = Console()

_REQUIRED_TOOLS = frozenset({"kcc-c2e", "calibre-debug", "KFX Output"})
_TOOL_REQUIREMENTS: dict[str, str] = {
    "kcc-c2e": "Required*",
    "calibre-debug": "Required",
    "KFX Output": "Required",
    "Kindle Previewer 4": "Optional",
    "unrar/unar/7z": "Opt (CBR)",
}


def _parse_custom_dimensions(raw: str) -> tuple[int, int]:
    parts = raw.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Expected WIDTHxHEIGHT (e.g. 1264x1680), got '{raw}'")
    return int(parts[0].strip()), int(parts[1].strip())


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
        typer.Option("-p", "--profile", help="Kindle profile: KPW5, KS, OASIS, or CUSTOM."),
    ] = "KPW5",
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
    ] = 0.02,
    min_panel_area: Annotated[
        float,
        typer.Option("--min-panel-area", min=0.01, max=0.50, help="Minimum panel area ratio."),
    ] = 0.03,
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
) -> None:
    """Convert CBZ, CBR, image folder, or batch directory into PDOC-tagged KFX."""
    device_profile = _resolve_profile(profile, custom_size)
    pipeline = ConversionPipeline()

    try:
        is_batch = pipeline.is_batch_input(input_path)
        if is_batch and title is not None:
            raise ValueError("--title cannot be used with a batch archive directory.")
        if output is None:
            resolved_output = (
                input_path
                if is_batch
                else ConversionPipeline.resolve_single_output(input_path, None)
            )
        elif is_batch:
            resolved_output = ConversionPipeline.resolve_batch_output_dir(input_path, output)
        else:
            resolved_output = ConversionPipeline.resolve_single_output(input_path, output)

        config = ConversionConfig(
            input_path=input_path,
            output_path=resolved_output,
            profile=device_profile,
            direction=direction,
            spread_mode=spread_mode,
            zoom_factor=zoom_factor,
            padding=padding,
            min_panel_area=min_panel_area,
            no_kcc=no_kcc,
            keep_temp=keep_temp,
            workers=workers,
            kcc_path=kcc_path,
            calibre_path=calibre_path,
        )
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if is_batch:
        try:
            outputs = pipeline.run_batch(config, author=author)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Batch conversion failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        for out_kfx in outputs:
            console.print(f"[green]Converted:[/green] {out_kfx}")
        for failed_archive, err in pipeline.last_batch_errors:
            console.print(f"[red]Failed:[/red] {failed_archive.name} - {err}")

        total = len(outputs) + len(pipeline.last_batch_errors)
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
        typer.Option("-p", "--profile", help="Kindle profile: KPW5, KS, OASIS, or CUSTOM."),
    ] = "KPW5",
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
    ] = 0.02,
    min_panel_area: Annotated[
        float,
        typer.Option("--min-panel-area", min=0.01, max=0.50, help="Minimum panel area ratio."),
    ] = 0.03,
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
) -> None:
    """Render debug overlays showing detected panels and reading-order arrows."""
    device_profile = _resolve_profile(profile, custom_size)
    default_name = ConversionPipeline.resolve_input_name(input_path)
    target_dir = output_dir or (input_path.resolve().parent / f"{default_name}_preview")
    if target_dir.exists() and not target_dir.is_dir():
        console.print(f"[red]Error:[/red] Output path exists and is not a directory: {target_dir}")
        raise typer.Exit(code=1)
    pipeline = ConversionPipeline()

    try:
        rendered = pipeline.preview_panels(
            input_path=input_path,
            output_dir=target_dir,
            profile=device_profile,
            direction=direction,
            spread_mode=spread_mode,
            padding=padding,
            min_panel_area=min_panel_area,
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
    previewer_path: Annotated[
        Path | None,
        typer.Option("--previewer-path", help="Explicit path to KindlePreviewer4CLI."),
    ] = None,
) -> None:
    """Check Python libraries and external CLI dependencies."""
    discovery = PlatformDiscovery()
    statuses = discovery.run_doctor(
        kcc_override=kcc_path,
        calibre_override=calibre_path,
        previewer_override=previewer_path,
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
