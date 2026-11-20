# Built-In CLI Progress Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add thread-safe, cross-platform live CLI progress monitoring with real-time batch streaming to `manga2kindle`, plus cross-platform `README.md` Magi batch templates.

**Architecture:** Introduce a typed `progress.py` module backed by `rich.live.Live` and `rich.progress.Progress` (with a clean non-TTY fallback), add optional chunk progress hooks to `MagiBackend` / `BackendRegistry`, switch `ConversionPipeline` batch executors to `as_completed` streaming with stage/page callbacks, and wire `RichProgressReporter` into `cli.py`.

**Tech Stack:** Python 3.11+, `rich`, `concurrent.futures`, `Enum`/`dataclass`/`Protocol`, `Typer`, `pytest`.

**Spec:** `docs/specs/2026-11-20-cli-progress-design.md`

## Global Constraints

- Python 3.11+ and full compatibility on macOS and Windows using `pathlib.Path`.
- Zero new runtime dependencies; use existing `rich`.
- Typed objects (`Enum`, `dataclass`, `Protocol`), no magic strings, single-responsibility helpers.
- Escape all dynamic values in Rich calls with `rich.markup.escape`.
- Keep `README.md` updated with CLI changes and batch template commands.

## Review Focus

- Manga filenames containing Rich markup tokens (e.g., `[scans] Manga [b] [/tag] [/].cbz`) must never crash Rich or lose bracketed text.
- Non-TTY / piped / `CliRunner` execution must emit clean streaming lines without ANSI progress redraws or duplicate `Converted:`/`Skipped:`/`Failed:` lines.
- Batch runs where `--skip-existing` skips all books must return before initializing Calibre or neural detectors.
- Out-of-order parallel batch completions must stream immediately to the reporter while returning deterministically ordered `list[Path]` and `last_batch_errors`.
- Legacy detector stubs lacking an `on_progress` parameter on `detect_batch` must continue to work seamlessly through `BackendRegistry`.

---

### Task 1: Progress Reporting Module (`src/manga2kindle/progress.py`)

**Files:**
- Create: `src/manga2kindle/progress.py`
- Test: `tests/test_progress.py`

**Interfaces:**
- Produces:
  - `ConversionStage` (`Enum`)
  - `STAGE_LABELS: dict[ConversionStage, str]`
  - `ProgressEvent` (`@dataclass(frozen=True)`)
  - `ProgressReporter` (`Protocol`)
  - `NullProgressReporter`
  - `RichProgressReporter`

- [ ] **Step 1: Write unit tests for `ProgressEvent` and `RichProgressReporter`**
  - Test `ProgressEvent.format_detail()` with `(completed, total)`, `(None, total)`, and `(None, None)`.
  - Test `RichProgressReporter` in non-TTY (`show_live=False`) and TTY (`show_live=True`) modes, Rich markup escaping, exception teardown (`transient=True`), and multi-threaded concurrent updates.
- [ ] **Step 2: Run tests to verify they fail**
  - Run: `uv run --offline --extra dev pytest tests/test_progress.py -v`
- [ ] **Step 3: Implement `src/manga2kindle/progress.py`**
- [ ] **Step 4: Run tests to verify they pass**
  - Run: `uv run --offline --extra dev pytest tests/test_progress.py -v`

---

### Task 2: Backend & Registry Page Progress (`magi.py`, `backends/__init__.py`)

**Files:**
- Modify: `src/manga2kindle/cv/backends/magi.py`
- Modify: `src/manga2kindle/cv/backends/__init__.py`
- Test: `tests/test_progress.py`

**Interfaces:**
- Consumes: `Callable[[int, int], None]`
- Produces:
  - `MagiBackend.detect_batch(image_paths, *, on_progress=None)`
  - `BackendRegistry.detect_batch(image_paths, *, on_progress=None)`
  - `BackendRegistry.detect_batch_and_sort(image_paths, sorter, *, page_widths=None, is_spreads=None, on_progress=None)`

- [ ] **Step 1: Add tests for chunk progress in `MagiBackend.detect_batch` and signature-aware `BackendRegistry.detect_batch`**
- [ ] **Step 2: Run tests to verify they fail**
- [ ] **Step 3: Update `src/manga2kindle/cv/backends/magi.py` and `src/manga2kindle/cv/backends/__init__.py`**
- [ ] **Step 4: Run tests to verify they pass**
  - Run: `uv run --offline --extra dev pytest tests/test_progress.py tests/test_magi_backend.py tests/test_backend_registry.py -v`

---

### Task 3: Pipeline Streaming & Stage Callbacks (`src/manga2kindle/pipeline.py`)

**Files:**
- Modify: `src/manga2kindle/pipeline.py`
- Test: `tests/test_progress.py`

**Interfaces:**
- Consumes: `ConversionStage`, `NullProgressReporter`, `ProgressEvent`, `ProgressReporter` from `manga2kindle.progress`
- Produces:
  - `ConversionPipeline(..., progress: ProgressReporter | None = None)`
  - `as_completed` real-time batch streaming with deterministic sorted return lists in `run_batch` and `preview_batch`
  - Stage and page progress events in `_convert_single`, `run_single`, `run_batch`, `preview_panels`, `preview_batch`

- [ ] **Step 1: Add pipeline streaming and stage-emission tests in `tests/test_progress.py`**
- [ ] **Step 2: Run tests to verify they fail**
- [ ] **Step 3: Update `src/manga2kindle/pipeline.py`**
- [ ] **Step 4: Run tests to verify they pass**
  - Run: `uv run --offline --extra dev pytest tests/test_progress.py tests/test_beautify.py tests/test_backend_registry.py -v`

---

### Task 4: CLI Integration & README Templates (`cli.py`, `README.md`, `tests/test_cli_e2e.py`)

**Files:**
- Modify: `src/manga2kindle/cli.py`
- Modify: `README.md`
- Modify: `tests/test_cli_e2e.py`

**Interfaces:**
- Consumes: `RichProgressReporter` from `manga2kindle.progress`

- [ ] **Step 1: Add CLI streaming uniqueness and bracket-escaping batch tests in `tests/test_cli_e2e.py`**
- [ ] **Step 2: Wire `RichProgressReporter` into `convert` and `preview-panels` in `src/manga2kindle/cli.py` and update `README.md`**
- [ ] **Step 3: Run full test suite and CLI sanity checks**
  - Run: `uv run --offline --extra dev pytest -q`
  - Run: `uv run --offline manga2kindle convert --help`
