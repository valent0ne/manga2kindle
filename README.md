# manga2kindle

Cross-platform headless CLI for converting `.cbz`, `.cbr`, and image folders into native **`PDOC`-tagged `.kfx`** files with **Guided View** (double-tap panel-by-panel zoom) for Kindle E-Ink devices.

## Features

- **Native `.kfx` with `PDOC` tagging:** Builds fixed-layout `.kpf` packages directly in Python (`KpfBuilder`—SQLite KDF + Amazon Ion 1.0 fragments) and compiles them to `PDOC`-tagged `.kfx` via Calibre's `KFX Output` plugin (`-d`) so USB-sideloaded manga displays cover thumbnails properly on modern Kindles (including Kindle Paperwhite 12th Gen).
- **Guided View Panel Detection (3-Layer Architecture):** Embeds native Kindle magnification targets using pluggable detection backends with automatic fallback:
  - **Layer 1 (`contour`):** Built-in OpenCV morphological contour extraction, containment pruning, IoU NMS, and recursive XY-cut spatial reading-order sorting (RTL manga, LTR comics, and 4-koma vertical strips).
  - **Layer 2 (`onnx`, optional `[onnx]` extra):** ONNX Runtime YOLO panel detector (`manga109-yolo-s` / `manga109-yolo-n`) with SHA-256-verified weight caching.
  - **Layer 3 (`magi`, optional `[magi]` extra):** GPU-accelerated Magi (`magiv3` / `magiv2`) for panel detection and native reading-order prediction (opt-in via `--allow-remote-code`).
  - **`auto` (default):** Resolves `magi` (when GPU, cached weights, and `--allow-remote-code` are present) -> `onnx` (when `onnxruntime` and cached weights are present) -> `contour` fallback.
- **E-Ink Image Preprocessing:** Uses **Kindle Comic Converter (`kcc-c2e`)** by default for E-Ink-tuned dithering/contrast and spread handling, with a pure-Pillow fallback via `--no-kcc`.
- **Visual QA, Benchmarking & Diagnostics:** Includes `preview-panels` (with `--compare` side-by-side backend grids), `convert --preview-only`, `models list`/`fetch`, `benchmark` (PopManga, CoMix, and local JSON/COCO), and `doctor` to verify dependencies.
- **Single, Folder, and Batch Modes:** Automatically handles a single `.cbz`/`.cbr`, a directory of images, or a batch directory of archives using parallel worker threads.
- **Cross-Platform:** Built for macOS and Windows using `pathlib.Path` and automatic executable discovery with explicit `--*-path` overrides.

## Prerequisites

1. **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)**
2. **Calibre** + the **KFX Output** plugin *(required)*
3. **UnRAR / `unar` / 7-Zip** *(optional, required only for `.cbr` archives)*

### Installing System Dependencies

#### macOS (Homebrew)

```bash
brew install --cask calibre
brew install unar                     # optional, for .cbr support
```

#### Windows (`winget` / `choco`)

```powershell
# Using winget
winget install calibre.calibre
winget install 7zip.7zip                 # optional, for .cbr support

# Or using Chocolatey
choco install calibre -y
choco install 7zip -y                    # optional, for .cbr support
```

#### Installing the Calibre `KFX Output` Plugin

1. Open **Calibre** → **Preferences** → **Plugins** → **Get new plugins**.
2. Search for **KFX Output** (by jhowell), install it, and restart Calibre.
3. Verify headless plugin access:

```bash
uv run manga2kindle doctor
```

## Setup

```bash
git clone https://github.com/valent0ne/manga2kindle.git
cd manga2kindle
uv sync
```

Optional extras for neural panel detection and benchmarking:

```bash
uv sync --extra onnx        # Layer 2: ONNX Runtime YOLO detector
uv sync --extra magi        # Layer 3: PyTorch/Transformers Magi detector (GPU)
uv sync --extra benchmark   # PopManga / CoMix dataset evaluation (datasets + psutil)
```

Verify all dependencies:

```bash
uv run manga2kindle doctor
```

## Usage

### 1. Check Dependencies (`doctor`)

```bash
uv run manga2kindle doctor
```

Optional binary overrides:

```bash
uv run manga2kindle doctor \
  --calibre-path "/Applications/calibre.app/Contents/MacOS/calibre-debug" \
  --kcc-path "/path/to/kcc-c2e"
```

### 2. Preview Detected Panels (`preview-panels`)

Generate numbered PNG overlays with green panel boxes, `[1]`/`[2]` ordinals, and red reading-order arrows:

```bash
uv run manga2kindle preview-panels \
  -i "Chapter 01.cbz" \
  --output-dir ./preview \
  -p KPW6 \
  -d rtl \
  -s split
```

(`preview-panels` defaults `-o, --output-dir` to `<input.parent>/<name>_preview`.)

Compare multiple panel detection backends side-by-side:

```bash
uv run manga2kindle preview-panels \
  -i "Chapter 01.cbz" \
  --compare contour,onnx,magi \
  --allow-remote-code
```

### 3. Manage Neural Panel Models (`models`)

List registered models, local cache status, and license notices:

```bash
uv run manga2kindle models list
```

Download and SHA-256-verify weights into your OS-native user cache:

```bash
uv run manga2kindle models fetch manga109-yolo-s
uv run manga2kindle models fetch magiv3
```

### 4. Convert to PDOC KFX (`convert`)

**Single `.cbz` / `.cbr` archive:**

```bash
uv run manga2kindle convert \
  -i "Manga Vol. 01.cbz" \
  -o "Manga Vol. 01.kfx" \
  --title "Manga Vol. 01" \
  --author "Author Name"
```

**Directory of page images (single book):**

```bash
uv run manga2kindle convert -i ./Volume_01_Pages -o ./Volume_01.kfx
```

**Batch directory of `.cbz`/`.cbr` archives:**

```bash
uv run manga2kindle convert -i ./Chapters -o ./KFX_Output -w 4
```

**Using the native Pillow fallback (without `kcc-c2e`):**

```bash
uv run manga2kindle convert -i "Chapter 01.cbz" -o "Chapter 01.kfx" --no-kcc
```

**Rendering panel previews only (without compiling `.kpf`/`.kfx`):**

```bash
uv run manga2kindle convert -i "Chapter 01.cbz" --preview-only
```

**Beautifying output filenames and KFX title metadata (and optionally source files):**

```bash
uv run manga2kindle convert -i ./Chapters -o ./KFX_Output --beautify
uv run manga2kindle convert -i ./Chapters -o ./KFX_Output --beautify-source
```

### 5. Benchmark Panel Detectors (`benchmark`)

Evaluate detection accuracy (`P@50`, `R@50`, `P@75`, `R@75`, `mIoU`, nested duplicate rate, exact count match), reading-order metrics (`PairAcc`, `Kendall's tau`), latency (`ms/pg`), and memory (`RAM`/`VRAM`) against **PopManga** (defaults to `rtl`), **CoMix** (defaults to `ltr`), or a local JSON/COCO dataset:

```bash
# One-command mechanical runner (syncs extras, fetches models, runs benchmark)
uv run python utils/run_benchmark.py --out benchmark.json

# Or invoke the CLI directly:
# PopManga (gated HF dataset: accept terms on Hugging Face & set HF_TOKEN; uses --split unseen)
uv run manga2kindle benchmark --dataset popmanga --split unseen --limit 50 --backends contour,onnx,magi --allow-remote-code --out report.json

# CoMix (defaults --direction to ltr; gated HF dataset)
uv run manga2kindle benchmark --dataset comix --limit 50 --backends contour,onnx

# Local custom JSON or COCO JSON dataset
uv run manga2kindle benchmark --dataset local --pages-dir ./pages --truth ./truth.json --backends contour
```

### Key Flags (`convert`)

| Flag | Default | Description |
| --- | --- | --- |
| `-i, --input` | *(required)* | Input `.cbz`/`.cbr`, image directory, or batch archive directory |
| `-o, --output` | `<input.parent>/<name>.kfx` (single) / `<input>` (batch) | Output `.kfx` file or output directory |
| `-p, --profile` | `KPW6` | Device preset: `KPW6` (1264x1680), `KPW5` (1264x1680), `KS` (1860x2480), `OASIS`, or `CUSTOM` |
| `--custom-size` | `None` | Custom `WIDTHxHEIGHT` (requires `-p CUSTOM`) |
| `--cover-size` | `1600x2560` | Cover `WIDTHxHEIGHT` (stretched to fill without borders) |
| `-d, --direction` | `rtl` | Reading direction (`rtl` or `ltr`) |
| `-s, --spread-mode` | `split` | Double-page spread handling (`split`, `rotate`, or `keep`) |
| `-z, --zoom-factor` | `1.5` | Guided View magnification factor (`>= 1.0`) |
| `--padding` | `0.02` | Panel bounding-box padding ratio (`0.0`–`0.25`) |
| `--min-panel-area` | `0.03` | Minimum panel area ratio (`0.01`–`0.50`) |
| `--detector` | `auto` | Panel detector backend (`auto`, `magi`, `onnx`, or `contour`) |
| `--detector-model` | `None` | Detector model key, Hugging Face repo ID, or local weights path |
| `--device` | `auto` | Inference device for neural detectors (`auto`, `cpu`, `cuda`, or `mps`) |
| `--score-threshold` | `0.25` | Confidence threshold for neural panel detectors (`0.0`–`1.0`) |
| `--merge-panels / --no-merge-panels` | `true` | Merge adjacent panels bridged by shared dialogue bubbles or substantial overlap (also on `preview-panels`) |
| `--allow-remote-code` | `false` | Allow executing remote model code (required to enable `magi`) |
| `--preview-only` | `false` | Render panel preview PNGs and exit without compiling `.kpf`/`.kfx` |
| `--skip-existing` | `false` | Skip conversion if the output `.kfx` file already exists |
| `--beautify` | `false` | Strip release tags from the output `.kfx` filename and KFX title metadata |
| `--beautify-source` | `false` | Rename source `.cbz`/`.cbr` file(s) in place and beautify output `.kfx` and metadata |
| `--no-kcc` | `false` | Use pure-Pillow E-Ink preprocessing instead of `kcc-c2e` |
| `--keep-temp` | `false` | Keep temporary staging and intermediate `.kpf` directories |
| `-w, --workers` | CPU count | Parallel worker threads |
| `--batch-workers` | `auto` | Concurrent archives in batch mode (`auto` from RAM/KCC) |
| `--gpu-batch-size` | `auto` | Pages per GPU batch for neural detectors (`auto` from VRAM) |
| `--title` / `--author` | archive/folder name / `None` | Override KFX book title (single book) and author metadata |
| `--kcc-path` / `--calibre-path` | auto-discovered | Explicit executable path overrides |

## Kindle USB Sideloading (`PDOC`)

1. Connect your Kindle via USB.
2. Copy the generated `.kfx` files into the Kindle's `documents/` folder (for example `/Volumes/Kindle/documents/` on macOS or `E:\documents\` on Windows).
3. Safely eject the Kindle. Because `manga2kindle` tags and verifies every output file as `PDOC`, covers and Guided View panel zoom work natively via USB sideloading without Send-to-Kindle.

## License

This project is licensed under the [GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)](LICENSE).

### Model & Dataset Licenses

Optional neural weights and benchmark datasets are not bundled with `manga2kindle` and remain subject to their respective upstream terms:

- **`manga109-yolo-s` / `manga109-yolo-n` (`deepghs/manga109_yolo`):** Trained on the **Manga109-s** dataset (`hal-utokyo/Manga109-s`); subject to Manga109-s academic/non-commercial dataset terms and the upstream model repository license.
- **`magiv3` / `magiv2` (`ragavsachdeva/magiv3`, `ragavsachdeva/magiv2`):** Subject to non-commercial / personal / research / not-for-profit use only per the upstream model card; commercial use requires a separate license from the authors.
- **PopManga & CoMix Benchmark Datasets:** Subject to their upstream academic/non-commercial research terms and attribution requirements.

## Legal Notice

`manga2kindle` is an independent, open-source conversion utility. It does not distribute, host, or provide access to copyrighted manga or other copyrighted content.

Users are responsible for ensuring they have the necessary rights and permissions to process the files they provide.

The project does not implement DRM removal, decryption, or circumvention of technological protection measures.

`manga2kindle` is not affiliated with, endorsed by, or sponsored by Amazon.

