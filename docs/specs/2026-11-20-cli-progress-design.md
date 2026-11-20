# Built-In CLI Progress Monitoring Design

**Date:** 2026-11-20
**Status:** Approved

## Overview

Add built-in, cross-platform live CLI progress monitoring to `manga2kindle` using the already-bundled `rich` dependency so single-book conversions, batch directory conversions, and `preview-panels` / `--preview-only` runs display real-time completion progress, active worker stages, per-page detection progress, and immediate per-archive streaming results. In addition, document cross-platform (macOS/Linux `sh` and Windows PowerShell) batch high-quality `magiv3` template commands in `README.md`.

## Goals

- Stream batch book completions, skips, and failures in real time via `concurrent.futures.as_completed(...)` instead of buffering all output until the entire batch finishes.
- Preserve deterministic result ordering in `ConversionPipeline.run_batch` and `ConversionPipeline.preview_batch` return lists and `last_batch_errors`.
- Display a thread-safe Rich `Live` progress view (overall bar + ETA in batch mode, active worker spinners with stage and page counts) when running in an interactive terminal (`console.is_terminal`), with zero ANSI escape clutter when output is non-TTY/piped.
- Report granular progress across all pipeline stages (`extracting`, `preprocessing`, `detecting`, `packaging`, `compiling`, `rendering`) including chunk-by-chunk page progress during `MagiBackend.detect_batch`.
- Keep zero new external dependencies and escape all dynamic strings (`rich.markup.escape`) so bracketed manga filenames (`[Group]`) never break Rich rendering.
- Document cross-platform macOS/Linux `sh` and Windows PowerShell high-quality Magi batch template commands in `README.md`.

## Non-Goals

- Adding extra runtime dependencies (e.g., `tqdm`) when `rich` is already a core dependency.
- Modifying KCC (`kcc-c2e`) or Calibre (`calibre-debug`) external subprocess internals to parse child stdout.
- Changing existing CLI summary output strings (`Batch summary: ...`, `Batch preview summary: ...`) relied upon by callers and tests.

## Architecture & Component Design

### 1. Progress Reporting Module (`src/manga2kindle/progress.py`)

- `ConversionStage(str, Enum)`: `EXTRACTING`, `PREPROCESSING`, `DETECTING`, `PACKAGING`, `COMPILING`, `RENDERING`.
- `STAGE_LABELS: dict[ConversionStage, str]`: maps each enum value to its display label.
- `ProgressEvent` (`@dataclass(frozen=True)`):
  - Fields: `book_id: str`, `book_name: str`, `stage: ConversionStage`, `completed: int | None = None`, `total: int | None = None`.
  - `format_detail() -> str`: formats `label`, `label (total pages)`, or `label (completed/total pages)`.
- `ProgressReporter(Protocol)`:
  - `on_batch_start(total, *, label="Converting")`
  - `on_book_start(book_id, book_name)`
  - `on_book_stage(event)`
  - `on_book_complete(book_id, book_name, output_path)`
  - `on_book_skip(book_id, book_name, output_path)`
  - `on_book_error(book_id, source_name, error)`
  - `on_batch_finish()`
- `NullProgressReporter`: no-op default implementation for library callers that do not pass a reporter.
- `RichProgressReporter`:
  - Thread-safe via `threading.Lock()`.
  - Uses `rich.live.Live(Group(overall_progress, worker_progress))` when `show_live` is enabled (`console.is_terminal` by default), and disables live ANSI redraws when non-TTY while still streaming per-archive outcomes in batch mode.
  - Escapes all dynamic text (book names, paths, errors, labels) with `rich.markup.escape`.
  - Forces `transient=True` on exit if an unhandled exception occurred or no overall batch task exists.

### 2. Backend & Registry Page Progress (`src/manga2kindle/cv/backends/magi.py`, `__init__.py`)

- `MagiBackend.detect_batch(image_paths, *, on_progress=None)` invokes `on_progress(len(results), len(image_paths))` after each GPU batch chunk.
- `BackendRegistry.detect_batch` and `detect_batch_and_sort` accept optional `on_progress: Callable[[int, int], None] | None = None`.
- `_supports_on_progress(backend)` inspects `backend.detect_batch` signature so custom/legacy test backends without an `on_progress` parameter remain compatible.

### 3. Pipeline Streaming & Stage Callbacks (`src/manga2kindle/pipeline.py`)

- `ConversionPipeline.__init__` accepts optional `progress: ProgressReporter | None = None` (defaulting to `NullProgressReporter`).
- `run_batch` and `preview_batch` replace `executor.map(...)` with `executor.submit(...)` + `concurrent.futures.as_completed(...)`, invoking `reporter.on_book_complete` / `on_book_error` (and user `on_success` / `on_error` hooks) immediately as each book finishes, then sort collected results by original input index before returning.
- `run_single`, `_convert_single`, and `preview_panels` emit stage transitions and page-level detection/rendering progress.
- Batch filename collision validation and `--skip-existing` early return (when zero books remain) continue to execute before initializing Calibre or neural detectors.

### 4. CLI & Documentation (`src/manga2kindle/cli.py`, `README.md`)

- `convert` (single, batch, `--preview-only` single/batch) and `preview-panels` wrap execution in `with RichProgressReporter(...) as reporter:` and pass `progress=reporter` to `ConversionPipeline`.
- Post-batch `Converted:` / `Skipped:` / `Failed:` print loops are removed from `cli.py` since `RichProgressReporter` prints each line immediately on completion; summary lines remain unchanged.
- `README.md` documents the macOS/Linux (`sh`/`bash`/`zsh`) and Windows (PowerShell) batch `magiv3` commands with `--beautify-source`, max parallelism, `--skip-existing`, and live progress monitoring notes.

## Testing Strategy

- Unit and integration tests in `tests/test_progress.py` covering `ProgressEvent.format_detail`, `RichProgressReporter` non-TTY streaming and TTY live rendering, Rich markup escaping, multi-threaded lock safety, out-of-order completion streaming vs deterministic batch return ordering, stage emissions across `run_single`/`preview_panels`/`preview_batch`, and `MagiBackend`/`BackendRegistry` chunk callbacks.
- CLI end-to-end tests in `tests/test_cli_e2e.py` verifying batch streaming prints each outcome once and preserves bracketed filenames without markup errors.
