# manga2kindle

Cross-platform headless CLI for converting `.cbz`, `.cbr`, and image folders into native **`PDOC`-tagged `.kfx`** files with **Guided View** (double-tap panel-by-panel zoom) for Kindle E-Ink devices.

## Features

- **Native `.kfx` with `PDOC` tagging:** Compiles fixed-layout `.kpf` packages directly via Calibre's `KFX Output` plugin (`-d` / `PDOC`) so USB-sideloaded manga displays cover thumbnails properly on modern Kindles (including Kindle Paperwhite 12th Gen).
- **Guided View Panel Detection:** Uses OpenCV morphological contour detection, NMS pruning, and spatial reading-order sorting (RTL manga, LTR comics, and 4-koma vertical strips) to embed native Kindle magnification targets.
- **E-Ink Image Preprocessing:** Uses **Kindle Comic Converter (`kcc-c2e`)** by default for E-Ink-tuned dithering/contrast and spread handling, with a pure-Pillow fallback via `--no-kcc`.
- **Visual QA & Diagnostics:** Includes `preview-panels` to inspect detected bounding boxes and reading-order arrows before converting, and `doctor` to verify external tools.
- **Single, Folder, and Batch Modes:** Automatically handles a single `.cbz`/`.cbr`, a directory of images, or a batch directory of archives using parallel worker threads.
- **Cross-Platform:** Built for macOS and Windows using `pathlib.Path` and automatic executable discovery with explicit `--*-path` overrides.

## Prerequisites

1. **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)**
2. **Calibre** + the **KFX Output** plugin *(required)*
3. **Kindle Previewer 4** *(optional, checked by `doctor`)*
4. **UnRAR / `unar` / 7-Zip** *(optional, required only for `.cbr` archives)*

### Installing System Dependencies

#### macOS (Homebrew)

```bash
brew install --cask calibre
brew install --cask kindle-previewer   # optional
brew install unar                     # optional, for .cbr support
```

#### Windows (`winget` / `choco`)

```powershell
# Using winget
winget install calibre.calibre
winget install Amazon.KindlePreviewer4   # optional
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
  --kcc-path "/path/to/kcc-c2e" \
  --previewer-path "/Applications/Kindle Previewer 4.app/Contents/MacOS/KindlePreviewer4CLI"
```

### 2. Preview Detected Panels (`preview-panels`)

Generate numbered PNG overlays with green panel boxes, `[1]`/`[2]` ordinals, and red reading-order arrows:

```bash
uv run manga2kindle preview-panels \
  -i "Chapter 01.cbz" \
  --output-dir ./preview \
  -p KPW5 \
  -d rtl \
  -s split
```

(`preview-panels` defaults `-o, --output-dir` to `<input.parent>/<name>_preview`.)

### 3. Convert to PDOC KFX (`convert`)

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

### Key Flags (`convert`)

| Flag | Default | Description |
| --- | --- | --- |
| `-i, --input` | *(required)* | Input `.cbz`/`.cbr`, image directory, or batch archive directory |
| `-o, --output` | `<input.parent>/<name>.kfx` (single) / `<input>` (batch) | Output `.kfx` file or output directory |
| `-p, --profile` | `KPW5` | Device preset: `KPW5` (1264x1680), `KS` (1860x2480), `OASIS`, or `CUSTOM` |
| `--custom-size` | `None` | Custom `WIDTHxHEIGHT` (requires `-p CUSTOM`) |
| `-d, --direction` | `rtl` | Reading direction (`rtl` or `ltr`) |
| `-s, --spread-mode` | `split` | Double-page spread handling (`split`, `rotate`, or `keep`) |
| `-z, --zoom-factor` | `1.5` | Guided View magnification factor (`>= 1.0`) |
| `--padding` | `0.02` | Panel bounding-box padding ratio (`0.0`–`0.25`) |
| `--min-panel-area` | `0.03` | Minimum panel area ratio (`0.01`–`0.50`) |
| `--no-kcc` | `false` | Use pure-Pillow E-Ink preprocessing instead of `kcc-c2e` |
| `--keep-temp` | `false` | Keep temporary staging and intermediate `.kpf` directories |
| `-w, --workers` | CPU count | Parallel worker threads |
| `--title` / `--author` | archive/folder name / `None` | Override KFX book title (single book) and author metadata |
| `--kcc-path` / `--calibre-path` | auto-discovered | Explicit executable path overrides (`doctor` also accepts `--previewer-path`) |

## Kindle USB Sideloading (`PDOC`)

1. Connect your Kindle via USB.
2. Copy the generated `.kfx` files into the Kindle's `documents/` folder (for example `/Volumes/Kindle/documents/` on macOS or `E:\documents\` on Windows).
3. Safely eject the Kindle. Because `manga2kindle` tags and verifies every output file as `PDOC`, covers and Guided View panel zoom work natively via USB sideloading without Send-to-Kindle.

## License

This project is licensed under the [MIT License](LICENSE).
