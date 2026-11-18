# Filename Beautification Design

**Date:** 2026-11-18
**Status:** Approved

## Overview

Add optional filename and metadata beautification to `manga2kindle convert` so release tags (e.g., `(Digital)`, `(f)`, `(f2)`, `(F)`, `[Group]`, or other parenthesized/bracketed tags) are stripped from output `.kfx` filenames and embedded KFX `title` metadata while preserving the series title, volume/chapter numbers, and release year `(YYYY)`. A separate flag allows renaming the source `.cbz`/`.cbr` archive files in place.

## Goals

- Provide `--beautify` on `convert` to clean the output `.kfx` filename (single and batch modes), `--preview-only` directory name, and embedded KPF/KFX `title` metadata.
- Provide `--beautify-source` on `convert` to rename source `.cbz`/`.cbr` archive files in place before conversion (and apply `--beautify` to the resulting `.kfx` output and metadata).
- Preserve root title, volume/chapter markers (`v01`, `Vol. 01`, `c001`, etc.), and 4-digit release years (`(19xx)` / `(20xx)`).
- Prevent silent overwrites by detecting filename collisions before renaming source files or starting batch conversion.

## Non-Goals

- Standalone subcommand outside `convert` (YAGNI).
- Parsing or modifying `ComicInfo.xml` inside source archives.
- Modifying explicit `--title` overrides provided by the user.

## Architecture & Component Design

### 1. Beautifier Utility (`src/manga2kindle/beautify.py`)

- `beautify_name(stem: str) -> str`:
  - Removes bracketed tags `[...]` anywhere in the stem.
  - Removes parenthesized tags `(...)` **except** 4-digit release years matching `^\((?:19|20)\d{2}\)$`.
  - Collapses repeated whitespace into single spaces and trims leading/trailing whitespace.
  - Strips dangling trailing separators (`-` or `_`) left behind after tag removal (e.g., `Series Name - ` -> `Series Name`).
  - Falls back to the original `stem` if stripping would yield an empty string.
- `beautify_source_file(path: Path) -> Path`:
  - Returns `path` unchanged for directories or when `beautify_name(path.stem) == path.stem`.
  - Computes `target = path.with_name(f"{new_stem}{path.suffix}")`.
  - Raises `FileExistsError` if a distinct file already exists at `target` (handling case-only renames on case-insensitive filesystems safely).
  - Renames `path` to `target` using `Path.rename` and returns `target`.

### 2. Configuration (`src/manga2kindle/models.py`)

Add two boolean fields to `ConversionConfig`:
- `beautify: bool = False`
- `beautify_source: bool = False`

Add a helper property `effective_beautify` (returning `self.beautify or self.beautify_source`) so `--beautify-source` consistently beautifies output filenames and KFX metadata as well.

### 3. Pipeline Integration (`src/manga2kindle/pipeline.py`)

- `resolve_input_name(input_path: Path, *, beautify: bool = False) -> str` and `resolve_single_output(input_path: Path, output_path: Path | None, *, beautify: bool = False) -> Path` accept `beautify`.
- In `run_single`:
  - If `config.beautify_source` is enabled and `config.input_path.is_file()`, rename the source file via `beautify_source_file` before extraction/conversion.
  - Resolve `input_name` and `output_kfx` using `config.effective_beautify`.
  - `_convert_single` sets `book_title = (title or "").strip() or input_name`, so KPF/KFX `title` metadata is automatically cleaned unless `--title` was explicitly provided.
- In `run_batch`:
  - Compute beautified stems when `config.effective_beautify` is enabled.
  - Validate output `.kfx` filename collisions up front.
  - If `config.beautify_source` is enabled, validate source filename collisions across all archives in the batch *before* renaming any file, then rename each source archive in place before dispatching conversion workers.

### 4. CLI Integration (`src/manga2kindle/cli.py`)

- Add `--beautify` and `--beautify-source` options to `convert`.
- Pass `effective_beautify = beautify or beautify_source` to `resolve_single_output`, `_resolve_preview_only_dir`, and `ConversionConfig`.

## Testing Strategy

- Unit tests in `tests/test_beautify.py` covering `beautify_name` (all tag formats, year preservation, unchanged names, empty fallback, dangling separators) and `beautify_source_file` (in-place rename, no-op, collision error, directory no-op).
- Integration/CLI tests in `tests/test_cli_e2e.py` covering `--beautify` (output `.kfx` name + KPF `title` metadata, explicit `--title` precedence, batch collision detection) and `--beautify-source` (source file renamed in place + output `.kfx` + metadata).
