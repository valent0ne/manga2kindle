# Filename Beautification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--beautify` and `--beautify-source` CLI options to `manga2kindle convert` to strip release tags from output `.kfx` filenames, embedded KFX `title` metadata, and optionally source `.cbz`/`.cbr` files in place.

**Architecture:** Introduce a focused `beautify.py` module for deterministic tag stripping and safe in-place source renaming, wire `beautify` / `beautify_source` flags through `ConversionConfig` and `ConversionPipeline`, and expose them on the Typer `convert` command.

**Tech Stack:** Python 3.11+, `pathlib.Path`, `re`, `Pydantic`, `Typer`, `pytest`.

**Spec:** `docs/specs/2026-11-18-beautify-filename-design.md`

## Global Constraints

- Python 3.11+ and cross-platform (`macOS` and `Windows`) using `pathlib.Path` exclusively.
- Typed Pydantic models/dataclasses; no magic strings; single-responsibility helpers.
- Conventional Commits without `Co-Authored-By:` trailers; work on `feat/beautify-filename` branched off `develop`.
- Keep `README.md` updated with new CLI flags.

## Review Focus

- Titles with legitimate hyphens (e.g., `Frieren - Beyond Journey's End v01 (2021) (Digital)`) must keep the internal hyphen while stripping trailing release tags.
- 4-digit release years `(19xx)` and `(20xx)` must be preserved while non-year parenthesized/bracketed tags are removed.
- Input stems consisting solely of tags (e.g., `(Digital)`) must fall back to the original stem instead of producing an empty filename.
- Batch `--beautify` and `--beautify-source` must detect target filename collisions *before* renaming any files or starting conversion.
- Explicit `--title` in single-file mode must take precedence over the beautified filename for KFX metadata.

---

### Task 1: Beautifier Module (`src/manga2kindle/beautify.py`)

**Files:**
- Create: `src/manga2kindle/beautify.py`
- Test: `tests/test_beautify.py`

**Interfaces:**
- Produces:
  - `beautify_name(stem: str) -> str`
  - `beautify_source_file(path: Path) -> Path`

- [ ] **Step 1: Write the failing tests in `tests/test_beautify.py`**

Test `beautify_name` on:
- `Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f)` -> `Frieren - Beyond Journey's End v01 (2021)`
- `Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f2)` -> `Frieren - Beyond Journey's End v01 (2021)`
- `Fullmetal Alchemist v01 (Grunbeld)` -> `Fullmetal Alchemist v01`
- `Witch Hat Atelier v12 (2024) (Digital) (LuCaZ) (F)` -> `Witch Hat Atelier v12 (2024)`
- `Kagurabachi v01 (2024)` -> `Kagurabachi v01 (2024)`
- `Chainsaw Man - v01 (2019) [Digital]` -> `Chainsaw Man - v01 (2019)`
- `One Piece - c1045 (2022) (Digital) [Group]` -> `One Piece - c1045 (2022)`
- `Series Name - (Digital)` -> `Series Name`
- `(Digital)` -> `(Digital)` (empty fallback)

Test `beautify_source_file` on:
- Renaming a `.cbz` file in place and returning the new `Path`
- No-op when filename is already clean
- Raising `FileExistsError` when a distinct target file already exists
- Returning directory paths unchanged

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_beautify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manga2kindle.beautify'`

- [ ] **Step 3: Implement `src/manga2kindle/beautify.py`**

Implement `beautify_name` and `beautify_source_file` with clean docstrings and compiled regex constants.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_beautify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

`git add src/manga2kindle/beautify.py tests/test_beautify.py`
`git commit -m "feat: add filename beautifier utility"`

---

### Task 2: Config & Pipeline Integration (`models.py`, `pipeline.py`)

**Files:**
- Modify: `src/manga2kindle/models.py`
- Modify: `src/manga2kindle/pipeline.py`
- Test: `tests/test_cli_e2e.py` (or `tests/test_beautify.py` pipeline unit tests)

**Interfaces:**
- Consumes: `beautify_name`, `beautify_source_file` from `manga2kindle.beautify`
- Produces:
  - `ConversionConfig.beautify: bool = False`
  - `ConversionConfig.beautify_source: bool = False`
  - `ConversionConfig.effective_beautify` property
  - `ConversionPipeline.resolve_input_name(..., beautify: bool = False)`
  - `ConversionPipeline.resolve_single_output(..., beautify: bool = False)`

- [ ] **Step 1: Write failing tests for config & pipeline beautification**

Test:
- `ConversionConfig.effective_beautify` is `True` when either `beautify` or `beautify_source` is `True`.
- `ConversionPipeline.resolve_single_output(..., beautify=True)` strips release tags from output `.kfx` name.
- `ConversionPipeline.run_single` with `beautify=True` writes beautified `.kfx` and passes beautified `title` to `KpfBuilder.build` (and respects explicit `title` override).
- `ConversionPipeline.run_single` with `beautify_source=True` renames source `.cbz` in place, writes beautified `.kfx`, and passes beautified `title` to `KpfBuilder.build`.
- `ConversionPipeline.run_batch` with `beautify=True` detects output `.kfx` collisions after beautifying, and with `beautify_source=True` detects source collisions before renaming any file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_beautify.py -v`
Expected: FAIL

- [ ] **Step 3: Update `src/manga2kindle/models.py` and `src/manga2kindle/pipeline.py`**

Wire `beautify`, `beautify_source`, `effective_beautify`, `resolve_input_name`, `resolve_single_output`, `run_single`, `run_batch`, and `preview_batch`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_beautify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

`git add src/manga2kindle/models.py src/manga2kindle/pipeline.py tests/test_beautify.py`
`git commit -m "feat(pipeline): support beautify and beautify_source options"`

---

### Task 3: CLI Flags & Documentation (`cli.py`, `README.md`)

**Files:**
- Modify: `src/manga2kindle/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli_e2e.py`

- [ ] **Step 1: Write failing CLI tests in `tests/test_cli_e2e.py`**

Test invoking `manga2kindle convert` with `--beautify` and `--beautify-source` via `CliRunner`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli_e2e.py -k beautify -v`
Expected: FAIL with unrecognized option `--beautify`

- [ ] **Step 3: Add `--beautify` and `--beautify-source` to `convert` in `src/manga2kindle/cli.py` and document them in `README.md`**

- [ ] **Step 4: Run full test suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

`git add src/manga2kindle/cli.py README.md tests/test_cli_e2e.py`
`git commit -m "feat(cli): add --beautify and --beautify-source flags"`
