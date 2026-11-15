# `manga2kindle` Design Document

**Date:** 2026-11-15
**Status:** Draft (Pending Approval)

## 1. Intent & Why

Sideloading `.cbz`/`.cbr` manga onto modern Kindle E-Ink hardware (specifically **Kindle Paperwhite 12th Gen**, 1264x1680, 300 PPI) via USB currently forces a compromise:
- Tools like **Kindle Comic Converter (KCC)** produce great E-Ink-tuned images, but their output formats lack native KFX Guided View (double-tap panel-by-panel zoom) or lose cover thumbnails when sideloaded as `EBOK`.
- **Calibre `KFX Output`** produces native `.kfx` with `PDOC` tagging (local cover rendering), but cannot convert fixed-layout comic EPUBs via Kindle Previewer.

`manga2kindle` is a headless, cross-platform (Windows & macOS) Python CLI that unifies E-Ink image preparation, OpenCV panel detection + reading-order sorting, Guided View packaging, and native `PDOC`-tagged `.kfx` compilation into a single automated command.

## 2. Goals, Non-Goals & Anti-Goals

### Goals
- Convert `.cbz`, `.cbr`, and image directories into native `.kfx` files tagged `PDOC` for USB sideloading.
- Detect manga panels automatically with OpenCV, prune duplicates via NMS, and sort in RTL (manga) or LTR (Western comic) reading order.
- Package Guided View (Region Magnification) panel targets so double-tap panel navigation works natively on Kindle.
- Use **KCC (`KindleComicConverter`)** as a required Python dependency for top-quality E-Ink preprocessing, with a `--no-kcc` native Pillow fallback.
- Run headlessly on both **macOS** and **Windows**, with a `doctor` command for dependency diagnostics and `preview-panels` for visual QA.

### Non-Goals
- No GUI or interactive panel editor.
- No DRM removal, cloud/Send-to-Kindle upload, or KDP publishing workflow.
- No reflowable text/OCR extraction from speech bubbles.

### Anti-Goals
- Failing silently into `.mobi` or `EBOK` output without telling the user.
- Leaving temporary extracted archives or intermediate `.kpf` files behind unless `--keep-temp` is passed.
- Hardcoding OS-specific paths without discovery/override support.

## 3. Constraints & Spike Findings

A pre-design spike against the target environment (`macOS`, `Calibre 9.11`, `KFX Output 2.19.0`, `Kindle Previewer 4.0.4`) established:
1. **Kindle Previewer 4 (`KindlePreviewer4CLI`) rejects Fixed-Layout EPUB 3 for KPF:** Any `pre-paginated` / `comic` EPUB passed to KP4 outputs a legacy `.mobi` (`Enhanced Typesetting not supported for this book`). Calibre's `KFX Output` plugin also explicitly does not convert fixed-layout comic EPUBs via Kindle Previewer.
2. **Direct `.kpf` → `.kfx` works cleanly:** Passing a `.kpf` (Kindle Package Format) file to `calibre-debug -r "KFX Output" -- -d <in>.kpf <out>.kfx` compiles a native `.kfx` and tags `cde_content_type=PDOC` automatically (no `--clipping-limit 0` or `--cde-type` flags exist in `KFX Output`; `-d` is the `PDOC` flag).
3. **KCC as a Python package:** `KindleComicConverter` (`ciromattia/kcc`) can be declared as a required dependency from GitHub (`KindleComicConverter @ git+https://github.com/ciromattia/kcc.git`), exposing `kcc-c2e` cross-platform on macOS and Windows.
4. **Cross-platform tooling:** Python 3.11+, `pathlib.Path` everywhere, Windows + macOS executable auto-discovery for `calibre-debug`, `Kindle Previewer 4` (for `doctor` diagnostics), `kcc-c2e`, and `unrar`/`7z`.

## 4. Approaches Considered

### Approach A (Selected): Direct `.kpf` Generation → `calibre-debug -r "KFX Output" -- -d` → `.kfx`
- **How it works:** Stage 4 writes a valid Fixed-Layout Comic `.kpf` archive (`book.kcb` JSON + `book.kdf` SQLite Ion container with page images and Guided View panel regions) directly in Python. Stage 5 compiles `.kpf` → `.kfx` (`PDOC`) via `calibre-debug -r "KFX Output" -- -d`.
- **Trade-offs:** Requires implementing a focused `.kpf` (`KDF`/Ion) writer in Python instead of Jinja2 EPUB XHTML templates, but **actually produces native `.kfx` with Guided View and `PDOC`**, works identically on macOS and Windows, and decouples `.kfx` compilation from Kindle Previewer 3 vs 4 version mismatches in `KFX Output`.

### Approach B (Rejected): Fixed-Layout EPUB 3 → `KindlePreviewer4CLI` → `.mobi`
- **How it works:** Generate Fixed-Layout EPUB 3 with `app-amzn-magnify` XHTML and compile via `KindlePreviewer4CLI`.
- **Why rejected:** Confirmed in spike that KP4 refuses to emit `.kpf` for fixed-layout EPUBs and falls back to `.mobi`, failing the primary `.kfx` + Guided View requirement.

## 5. System Architecture (OOP Design)

Package name: `manga2kindle` (`src/manga2kindle/`), CLI entry point: `manga2kindle`.

### Domain Models (`models.py`)
- `DeviceProfile`: Target display specs (`KPW5`: `1264x1680`, `KS`, `OASIS`, `CUSTOM`).
- `PanelBox`: `(x, y, w, h)` + `ordinal` + normalized/target zoom coordinates.
- `PageAsset`: Source path, processed image path, dimensions, spread flag, detected `list[PanelBox]`.
- `ConversionConfig`: Input/output paths, profile, direction (`rtl`/`ltr`), spread mode (`split`/`rotate`/`keep`), zoom factor, padding, min panel area, `--no-kcc`, `--keep-temp`, workers, binary overrides.

### Pipeline Stages
1. **`ArchiveExtractor` (`extractor.py`):**
   - Extracts `.cbz` (`zipfile`), `.cbr` (`rarfile` + `unrar`), or reads an image directory into a managed temp workspace.
   - Sanitizes filenames, applies natural alphanumeric sorting, verifies images via `Pillow`, and flags cover (`0000`) and spreads (`width > height`).
2. **`ImagePreprocessor` (`preprocessor.py`):**
   - **Primary (Default):** Invokes `kcc-c2e` (from required `KindleComicConverter` package or `--kcc-path`) with `-p KPW5 -m -q -u -r 0 -f CBZ --mozjpeg` (respecting `--spread-mode` and `--profile`), then extracts the optimized images.
   - **Fallback (`--no-kcc`):** Native Pillow pipeline (margin auto-crop with 5px safety buffer, gamma ~1.6, 16-tone grayscale quantization, Lanczos resize + white padding, spread split/rotate/keep).
3. **`PanelDetector` & `PanelSorter` (`cv/detector.py`, `cv/sorter.py`):**
   - OpenCV pipeline: 8-bit grayscale → bilateral filter (`d=9, sigmaColor=75, sigmaSpace=75`) → Otsu inverted binarization → `5x5` morphological closing → `RETR_EXTERNAL` contours → area (`min_panel_area`..98%), aspect ratio (`<= 15`), and ink-density filtering → IoU NMS (`> 0.4` prunes inner/smaller box).
   - Fallback: `[PanelBox(0, 0, page_w, page_h)]` when 0 panels are detected (splash/credit pages).
   - `PanelSorter`: Row-clustering (`row_tolerance_px=50`) with RTL (`x` descending) or LTR (`x` ascending) ordering per row.
4. **`KpfBuilder` (`packaging/kpf.py`):**
   - Packages processed page images, `book.kcb` metadata, and `book.kdf` (SQLite + Amazon Ion binary fragments for `yj_fixed_layout`, storyline/sections, and Guided View panel tap/zoom regions) into `<title>.kpf`.
   - `PanelPreviewRenderer` (`cv/preview.py`): Renders green bounding boxes, ordinal numbers, and red directional flow arrows for `preview-panels`.
5. **`KfxCompiler` & `PlatformDiscovery` (`compiler/kfx.py`, `platform.py`):**
   - Cross-platform (`win32` / `darwin`) discovery for `calibre-debug`, `Kindle Previewer 4`, `kcc-c2e`, and `unrar` with CLI override flags (`--calibre-path`, `--kcc-path`, `--previewer-path`).
   - Invokes `calibre-debug -r "KFX Output" -- -d <book.kpf> <output.kfx>` and verifies the output `.kfx` exists and is tagged `PDOC`.

### CLI (`cli.py`, Typer + Rich)
- `manga2kindle convert`: End-to-end conversion (single archive, folder of images, or batch directory of archives with `--workers`).
- `manga2kindle preview-panels`: Generates visual debug overlays for panel detection & reading-order QA.
- `manga2kindle doctor`: Checks Python, OpenCV, KCC (`kcc-c2e`), Calibre (`calibre-debug` + `KFX Output` plugin), Kindle Previewer 4, and `unrar` on macOS/Windows.

## 6. Left to the Builder

- Verify `KindleComicConverter` GitHub package installation under Python 3.11+/3.14 with `uv` on macOS (and document Windows install notes).
- Verify exact Ion/KDF fragment fields required by `KFX Output`'s `kpf_book.py` for fixed-layout comic pages and Guided View regions so `calibre-debug -r "KFX Output"` emits a valid `.kfx` with working panel zoom.
- Tune default OpenCV thresholds if needed against the sample manga `.cbz` chapters.
