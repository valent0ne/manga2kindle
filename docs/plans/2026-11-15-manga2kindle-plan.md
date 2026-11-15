# `manga2kindle` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `manga2kindle` cross-platform (macOS & Windows) headless Python CLI that converts `.cbz`/`.cbr`/image folders into `PDOC`-tagged `.kfx` manga files with OpenCV-detected Guided View panels.

**Architecture:** A 5-stage OOP pipeline (`ArchiveExtractor` → `ImagePreprocessor` via `KindleComicConverter` or `--no-kcc` Pillow fallback → `PanelDetector`/`PanelSorter` → `KpfBuilder` writing `book.kcb` + `book.kdf` SQLite Ion fragments with explicit `yj_publisher_panels` region magnification → `KfxCompiler` invoking `calibre-debug -r "KFX Output" -- -d`). Supports `convert`, `preview-panels`, and `doctor` subcommands via Typer + Rich.

**Tech Stack:** Python 3.11+, `uv`, `KindleComicConverter`, `opencv-python-headless`, `pillow`, `typer[all]`, `rich`, `pydantic`, `rarfile`, `pytest`, Calibre (`calibre-debug` + `KFX Output`), Kindle Previewer 4 (for `doctor`).

**Spec:** `docs/specs/2026-11-15-manga2kindle-design.md`

## Global Constraints

- Package and CLI name: `manga2kindle` (`src/manga2kindle/`, entry point `manga2kindle = "manga2kindle.cli:app"`).
- Python 3.11+ required; all filesystem paths handled via `pathlib.Path` exclusively (cross-platform macOS & Windows).
- Default device profile: `KPW5` (`1264x1680`), plus `KS` (`1860x2480`), `OASIS` (`1264x1680`), `CUSTOM`.
- Reading directions: `rtl` (default) and `ltr`; spread modes: `split` (default), `rotate`, `keep`.
- Default OpenCV parameters: `zoom_factor=1.5`, `padding=0.02`, `min_panel_area=0.03` (0.01–0.50), `nms_iou_threshold=0.4`, `row_tolerance_px=50`.
- `KindleComicConverter` (`git+https://github.com/ciromattia/kcc.git`) is a required dependency; `--no-kcc` invokes the native Pillow fallback.
- KFX compilation uses `calibre-debug -r "KFX Output" -- -d <book.kpf> <output.kfx>` to enforce `PDOC` tagging; Kindle Previewer 4 is checked by `doctor` (never KP3).
- Follow OOP, KISS, DRY, YAGNI; keep `.gitignore` and `README.md` updated; no leftover temp artifacts unless `--keep-temp` is passed.

## Review Focus

- **Zip-slip / path-traversal & non-image junk in archives:** `.cbz`/`.cbr` files often contain `__MACOSX/`, `.DS_Store`, `ComicInfo.xml`, or `../` entries; `ArchiveExtractor` must reject traversal and skip non-image/hidden entries cleanly (tested in Task 2).
- **Corrupted or zero-byte images inside an archive:** A single bad page must raise a clear contextual error rather than crashing OpenCV/Pillow with an opaque traceback (tested in Task 2).
- **Zero-panel splash/blank pages & edge-bleed boxes:** Pages with no detectable borders must fall back to a full-page box, and bleed-to-edge panels must clamp within `(0, 0, W, H)` with 2px margin (tested in Task 4).
- **Cross-platform paths with spaces/Unicode & Windows executable names:** Paths like `Manga Vol. 01 (Digital).cbz` and Windows `.exe` discovery (`calibre-debug.exe`, `kcc-c2e.exe`, `KindlePreviewer4CLI.exe`, `UnRAR.exe`) must resolve without shell quoting bugs (tested in Task 1 & Task 7).
- **Batch conversion partial failure isolation & temp cleanup:** Batch runs over a folder must clean temp directories on both success and failure unless `--keep-temp` is set, and report per-file failures clearly (tested in Task 8).

---

### Task 1: Project Scaffolding, Domain Models & Platform Discovery

**Files:**
- Create: `pyproject.toml`
- Create: `src/manga2kindle/__init__.py`
- Create: `src/manga2kindle/models.py`
- Create: `src/manga2kindle/platform.py`
- Modify: `.gitignore`
- Test: `tests/test_models_platform.py`

**Interfaces:**
- Produces in `src/manga2kindle/models.py`:
  - `ReadingDirection(str, Enum)`: `RTL = "rtl"`, `LTR = "ltr"`
  - `SpreadMode(str, Enum)`: `SPLIT = "split"`, `ROTATE = "rotate"`, `KEEP = "keep"`
  - `DeviceProfile(BaseModel)`: `name: str`, `width: int`, `height: int`, `kcc_code: str`; classmethod `from_name(name: str, custom_size: tuple[int, int] | None = None) -> DeviceProfile`
  - `PanelBox(BaseModel)`: `x: int`, `y: int`, `w: int`, `h: int`, `ordinal: int = 0` *(already padded by `PanelDetector` and clamped within page bounds)*
  - `ZoomRect(BaseModel)`: `x: int`, `y: int`, `w: int`, `h: int` *(computed from `PanelBox`, `zoom_factor`, and page `(width, height)` for KPF magnification target)*
  - `PageAsset(BaseModel)`: `index: int`, `source_path: Path`, `processed_path: Path | None = None`, `width: int = 0`, `height: int = 0`, `is_cover: bool = False`, `is_spread: bool = False`, `panels: list[PanelBox] = []`
  - `ConversionConfig(BaseModel)`: `input_path: Path`, `output_path: Path`, `profile: DeviceProfile`, `direction: ReadingDirection = ReadingDirection.RTL`, `spread_mode: SpreadMode = SpreadMode.SPLIT`, `zoom_factor: float = 1.5`, `padding: float = 0.02`, `min_panel_area: float = 0.03`, `no_kcc: bool = False`, `keep_temp: bool = False`, `workers: int = Field(default_factory=lambda: os.cpu_count() or 4)`, `kcc_path: Path | None = None`, `calibre_path: Path | None = None`, `previewer_path: Path | None = None`
- Produces in `src/manga2kindle/platform.py`:
  - `ToolStatus(BaseModel)`: `name: str`, `found: bool`, `path: Path | None`, `version: str | None = None`, `details: str = ""`
  - `PlatformDiscovery`: `__init__(self, system: str | None = None)` (defaults to `sys.platform`); `find_kcc(override: Path | None = None) -> Path | None`; `find_calibre(override: Path | None = None) -> Path | None`; `find_previewer4(override: Path | None = None) -> Path | None`; `find_unrar() -> Path | None`; `run_doctor(...) -> list[ToolStatus]`

- [ ] **Step 1: Initialize `pyproject.toml` and `.gitignore`**

Create `pyproject.toml` with `requires-python = ">=3.11"`, dependencies (`opencv-python-headless>=4.8.0`, `pillow>=10.0.0`, `typer[all]>=0.9.0`, `rich>=13.0.0`, `pydantic>=2.0.0`, `rarfile>=4.0`, `KindleComicConverter @ git+https://github.com/ciromattia/kcc.git`), dev extra (`pytest>=8.0.0`), `[project.scripts] manga2kindle = "manga2kindle.cli:app"`.
Append `*.kpf`, `*.kfx`, `*.cbz`, `*.cbr`, `preview/`, `debug_inspect/`, `.superpowers/`, `.worktrees/` to `.gitignore`.

- [ ] **Step 2: Run `uv sync --extra dev` to verify dependency resolution**

Run: `uv sync --extra dev`
Expected: Virtual environment created and all dependencies (including `KindleComicConverter`) resolve cleanly. *(If KCC's PySide6 pin rejects Python 3.14, pin `.python-version` to `3.12` via `uv python pin 3.12` and re-run `uv sync --extra dev`.)*

- [ ] **Step 3: Write failing tests in `tests/test_models_platform.py`**

```python
from pathlib import Path
import pytest
from manga2kindle.models import ConversionConfig, DeviceProfile, PanelBox, ReadingDirection, SpreadMode, ZoomRect
from manga2kindle.platform import PlatformDiscovery

def test_device_profile_presets():
    kpw5 = DeviceProfile.from_name("KPW5")
    assert (kpw5.width, kpw5.height) == (1264, 1680)
    ks = DeviceProfile.from_name("KS")
    assert (ks.width, ks.height) == (1860, 2480)
    custom = DeviceProfile.from_name("CUSTOM", custom_size=(1400, 1900))
    assert (custom.width, custom.height, custom.kcc_code) == (1400, 1900, "OTHER")
    with pytest.raises(ValueError):
        DeviceProfile.from_name("UNKNOWN")

def test_platform_discovery_windows_candidates(tmp_path, monkeypatch):
    fake_calibre = tmp_path / "Calibre2" / "calibre-debug.exe"
    fake_calibre.parent.mkdir(parents=True)
    fake_calibre.write_text("stub")
    disc = PlatformDiscovery(system="win32")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert disc.find_calibre() == fake_calibre

def test_platform_discovery_override_takes_precedence(tmp_path):
    custom = tmp_path / "custom-kcc"
    custom.write_text("stub")
    disc = PlatformDiscovery(system="darwin")
    assert disc.find_kcc(override=custom) == custom
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_models_platform.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'manga2kindle'`).

- [ ] **Step 5: Implement `src/manga2kindle/__init__.py`, `models.py`, and `platform.py`**

Implement the Pydantic models (with `ConversionConfig` validators for bounds: `zoom_factor >= 1.0`, `0.0 <= padding <= 0.25`, `0.01 <= min_panel_area <= 0.50`, `workers >= 1`) and `PlatformDiscovery` covering both `darwin` and `win32` search matrices:
- `find_calibre`: `Path(sys.executable)` siblings, `shutil.which("calibre-debug")` / `calibre-debug.exe`, macOS `/Applications/calibre.app/Contents/MacOS/calibre-debug`, Windows `%ProgramFiles%\Calibre2\calibre-debug.exe`, `%ProgramFiles(x86)%\Calibre2\calibre-debug.exe`, `%LOCALAPPDATA%\Calibre2\calibre-debug.exe`.
- `find_kcc`: override, `Path(sys.executable).parent / ("kcc-c2e.exe" if win32 else "kcc-c2e")`, `shutil.which`.
- `find_previewer4`: macOS `/Applications/Kindle Previewer 4.app/Contents/MacOS/KindlePreviewer4CLI`, Windows `%LOCALAPPDATA%\Amazon\Kindle Previewer 4\KindlePreviewer4CLI.exe` and `...\lib\fc\bin\KindlePreviewer4CLI.exe` and `%ProgramFiles%` variants.
- `find_unrar`: `shutil.which("unrar" | "unar" | "7z")` + Windows `C:\Program Files\WinRAR\UnRAR.exe` and `7-Zip\7z.exe`.
- `run_doctor()` checks Calibre `KFX Output` plugin presence via `calibre-debug -r "KFX Output" -- --help`.

- [ ] **Step 6: Run tests to verify they pass and commit**

Run: `uv run pytest tests/test_models_platform.py -v`
Expected: PASS
```bash
git add pyproject.toml .gitignore src/manga2kindle/__init__.py src/manga2kindle/models.py src/manga2kindle/platform.py tests/test_models_platform.py
git commit -m "feat: scaffold manga2kindle with domain models and cross-platform discovery"
```

---

### Task 2: Archive Extraction & Natural Page Sorting (`ArchiveExtractor`)

**Files:**
- Create: `src/manga2kindle/extractor.py`
- Test: `tests/test_extractor.py`

**Interfaces:**
- Consumes: `PageAsset` from `manga2kindle.models`, `PlatformDiscovery` from `manga2kindle.platform`
- Produces in `src/manga2kindle/extractor.py`:
  - `natural_sort_key(path: str | Path) -> list[int | str]`
  - `ArchiveExtractor`: `__init__(self, discovery: PlatformDiscovery | None = None)`; `extract(self, input_path: Path, work_dir: Path) -> list[PageAsset]` *(resolves unrar/unar/7z via `PlatformDiscovery.find_unrar()` when `.cbr` is extracted; stages verified pages into `work_dir / "extracted"` cleanly)*

- [ ] **Step 1: Write failing tests in `tests/test_extractor.py`**

Include tests for:
1. Natural sorting (`page_1.png`, `page_2.png`, `page_10.png`).
2. Ignoring `__MACOSX/`, `.DS_Store`, `ComicInfo.xml`, and rejecting `../evil.png` zip-slip traversal.
3. Cover flagging (`index == 0`, `is_cover=True`), spread detection (`width > height` → `is_spread=True`), and folder-of-images input.
4. Raising `ValueError` on corrupted/0-byte images via `Image.verify()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_extractor.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `natural_sort_key` and `ArchiveExtractor` in `src/manga2kindle/extractor.py`**

Use `zipfile.ZipFile` for `.cbz`, `rarfile.RarFile` (configuring `rarfile.UNRAR_TOOL` if `unrar_path` is provided) for `.cbr`, and direct copy/scan for directories. Sanitize relative paths, verify each extracted image with `PIL.Image.open(...).verify()` followed by opening dimensions, sort naturally, and return `list[PageAsset]`.

- [ ] **Step 4: Run tests to verify they pass and commit**

Run: `uv run pytest tests/test_extractor.py -v`
Expected: PASS
```bash
git add src/manga2kindle/extractor.py tests/test_extractor.py
git commit -m "feat: add ArchiveExtractor with natural sort, zip-slip guard, and image verification"
```

---

### Task 3: E-Ink Image Preprocessing (`ImagePreprocessor`)

**Files:**
- Create: `src/manga2kindle/preprocessor.py`
- Test: `tests/test_preprocessor.py`

**Interfaces:**
- Consumes: `ConversionConfig`, `PageAsset`, `ReadingDirection`, `SpreadMode` from `manga2kindle.models`; `PlatformDiscovery` from `manga2kindle.platform`; `natural_sort_key` from `manga2kindle.extractor`
- Produces in `src/manga2kindle/preprocessor.py`:
  - `ImagePreprocessingError(RuntimeError)`
  - `ImagePreprocessor`: `__init__(self, discovery: PlatformDiscovery | None = None)`; `process(self, pages: list[PageAsset], work_dir: Path, config: ConversionConfig) -> list[PageAsset]`
  - `KccBackend` and `PillowFallbackBackend` encapsulated behind `ImagePreprocessor`

- [ ] **Step 1: Write failing tests in `tests/test_preprocessor.py`**

Include tests for:
1. Pillow fallback (`--no-kcc`): white-margin auto-crop with 5px safety buffer, gamma adjustment (`1.6`), 16-level grayscale quantized then converted back to `"L"` (saved as optimized PNG/JPEG so Pillow never raises `KeyError`/mode errors), exact `(1264, 1680)` dimensions padded with white, and `PageAsset.width`/`height` updated from actual saved dimensions.
2. Spread handling in Pillow fallback: `split` divides `2000x1000` into two `(1264, 1680)` pages ordered RTL (right half first) or LTR (left half first); `rotate` rotates 90° CCW; `keep` fits width with white padding.
3. KCC backend subprocess command construction: operates on the clean directory of extracted/verified images (never re-reading corrupt archives/non-image entries directly!), building `kcc-c2e -p <kcc_code> [-m if RTL] -q -u -r <0|1|2> -f CBZ --mozjpeg -o <out_dir> <staging_dir>` (with `--customwidth W --customheight H` when `profile.kcc_code == "OTHER"`), unpacking the produced `.cbz`, sorting naturally, updating `width`/`height` and `is_cover`, and raising `ImagePreprocessingError` if KCC fails (`no_kcc=False`) instead of silently falling back.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preprocessor.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `ImagePreprocessor`, `KccBackend`, and `PillowFallbackBackend` in `src/manga2kindle/preprocessor.py`**

Reuse `PlatformDiscovery.find_kcc(override=config.kcc_path)` rather than duplicating `shutil.which` logic. Always stage `pages` from `ArchiveExtractor` (via hardlink/copy into `work_dir / "kcc_in"`) and invoke `kcc-c2e` on that clean staging directory with spread mapping `SPLIT -> 0`, `ROTATE -> 1`, `KEEP -> 2`, and `-m` emitted only for `ReadingDirection.RTL`. Read final `width`/`height` of every processed output image into the returned `PageAsset` objects.

- [ ] **Step 4: Run tests to verify they pass and commit**

Run: `uv run pytest tests/test_preprocessor.py -v`
Expected: PASS
```bash
git add src/manga2kindle/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: add ImagePreprocessor with KCC backend and Pillow fallback"
```

---

### Task 4: OpenCV Panel Detection & Reading-Order Sorter (`PanelDetector`, `PanelSorter`)

**Files:**
- Create: `src/manga2kindle/cv/__init__.py`
- Create: `src/manga2kindle/cv/io.py`
- Create: `src/manga2kindle/cv/detector.py`
- Create: `src/manga2kindle/cv/sorter.py`
- Test: `tests/test_detector_sorter.py`

**Interfaces:**
- Consumes: `PanelBox`, `ReadingDirection` from `manga2kindle.models`
- Produces in `src/manga2kindle/cv/io.py`:
  - `read_image_bgr(path: Path) -> np.ndarray` *(uses `np.fromfile` + `cv2.imdecode` so Windows/Unicode paths never silently return `None`)*
  - `write_image(path: Path, image: np.ndarray) -> Path` *(uses `cv2.imencode` + `Path.write_bytes` for Windows/Unicode safety)*
- Produces in `src/manga2kindle/cv/detector.py`:
  - `PanelDetector`: `__init__(self, min_panel_area: float = 0.03, padding: float = 0.02, iou_threshold: float = 0.4, min_ink_ratio: float = 0.02, max_ink_ratio: float = 0.98)`; `detect(self, image_path: Path) -> list[PanelBox]` *(applies bleed clamping, `padding` expansion once, and bounds clamping before returning `PanelBox`es; returns full-page `PanelBox(0, 0, W, H)` when no candidate survives)*
- Produces in `src/manga2kindle/cv/sorter.py`:
  - `PanelSorter`: `__init__(self, direction: ReadingDirection = ReadingDirection.RTL, row_tolerance_px: int = 50)`; `sort(self, boxes: list[PanelBox]) -> list[PanelBox]` *(assigns 1-based `ordinal` on returned `PanelBox`es; detects 4-koma vertical stack when panels share narrow X span / strict vertical progression)*

- [ ] **Step 1: Write failing tests in `tests/test_detector_sorter.py`**

Include tests for:
1. Synthetic 3-panel manga page image drawn with OpenCV rectangles: verifies `PanelDetector.detect()` finds 3 boxes, filters tiny noise/bubbles, and removes duplicate nested boxes via IoU NMS (`> 0.4`).
2. Blank/splash page with 0 contours above `min_panel_area`: returns single full-page `PanelBox(0, 0, W, H)`.
3. Bleed-to-edge panel touching image borders: bounding box clamped inside `(0, 0, W, H)` with 2px inner safety margin.
4. `PanelSorter.sort()`: RTL (`x` descending within row, `y` ascending across rows), LTR (`x` ascending within row), and 4-koma single-column vertical sequence (`Y0 -> Y3`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_detector_sorter.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `PanelDetector` and `PanelSorter`**

Implement the exact CV pipeline from the spec (`cvtColor` → `bilateralFilter(9, 75, 75)` → `THRESH_BINARY_INV + THRESH_OTSU` → `5x5` `MORPH_CLOSE` → `RETR_EXTERNAL` → area/aspect/ink-density filters → IoU NMS → bleed clamping → padding → full-page fallback) and row-clustering spatial sorter with 4-koma detection.

- [ ] **Step 4: Run tests to verify they pass and commit**

Run: `uv run pytest tests/test_detector_sorter.py -v`
Expected: PASS
```bash
git add src/manga2kindle/cv/__init__.py src/manga2kindle/cv/io.py src/manga2kindle/cv/detector.py src/manga2kindle/cv/sorter.py tests/test_detector_sorter.py
git commit -m "feat: add OpenCV PanelDetector with NMS and RTL/LTR PanelSorter"
```

---

### Task 5: Visual Debug Panel Overlay (`PanelPreviewRenderer`)

**Files:**
- Create: `src/manga2kindle/cv/preview.py`
- Test: `tests/test_preview.py`

**Interfaces:**
- **Consumes:** `PanelBox` from `manga2kindle.models`, `read_image_bgr` / `write_image` from `manga2kindle.cv.io`
- Produces in `src/manga2kindle/cv/preview.py`:
  - `PanelPreviewRenderer`: `render(self, image_path: Path, panels: list[PanelBox], output_path: Path) -> Path`

- [ ] **Step 1: Write failing test in `tests/test_preview.py`**

Render debug overlay on a blank white `600x800` PNG with 2 `PanelBox`es; assert `output_path.exists()`, output dimensions match `(600, 800)`, and green/red overlay pixels were drawn.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preview.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `PanelPreviewRenderer` in `src/manga2kindle/cv/preview.py`**

Draw green (`(0, 200, 0)`) rectangles for each `PanelBox`, ordinal badge `[1]`, `[2]`, ..., and red (`(0, 0, 255)`) `cv2.arrowedLine` from center of panel `i` to panel `i+1`.

- [ ] **Step 4: Run tests to verify they pass and commit**

Run: `uv run pytest tests/test_preview.py -v`
Expected: PASS
```bash
git add src/manga2kindle/cv/preview.py tests/test_preview.py
git commit -m "feat: add PanelPreviewRenderer for preview-panels debug overlays"
```

---

### Task 6: Amazon Ion Writer & `.kpf` Guided View Packager (`IonBinaryWriter`, `KpfBuilder`)

**Files:**
- Create: `src/manga2kindle/packaging/__init__.py`
- Create: `src/manga2kindle/packaging/ion.py`
- Create: `src/manga2kindle/packaging/kpf.py`
- Test: `tests/test_kpf.py`

**Interfaces:**
- Consumes: `ConversionConfig`, `PageAsset`, `PanelBox`, `ReadingDirection`, `ZoomRect` from `manga2kindle.models`
- Produces in `src/manga2kindle/packaging/ion.py`:
  - `YJ_SYMBOLS_MAX_ID = 851` (`YJ_symbols` v10, `max_id=842`, symbols `$10..$851`), `ION_BVM = b"\xe0\x01\x00\xea"`, `IonSymbol`, `IonAnnotation`, `IonStruct`, `IonList`, `IonSExp`, `IonFloat64`, `IonEidRef`, `IonBinaryWriter.encode(value, include_bvm: bool = True) -> bytes`
- Produces in `src/manga2kindle/packaging/kpf.py`:
  - `KpfBuilder`: `compute_zoom_rect(panel: PanelBox, page_w: int, page_h: int, zoom_factor: float) -> ZoomRect`; `build(self, pages: list[PageAsset], title: str, config: ConversionConfig, output_kpf: Path) -> Path`

*(Verified KDF/KPF contract checked against `KFX Output 2.19.0` & `kfxlib`:)*
1. **SQLite schema (`book.kdf`):** Exact table DDL strings required by `KFX Output`'s `KpfContainer`:
   - `CREATE TABLE capabilities(key char(20), version smallint, primary key (key, version)) without rowid` + `INSERT INTO capabilities VALUES ('db.schema', 1)`
   - `CREATE TABLE fragments(id char(40), payload_type char(10), payload_value blob, primary key (id))` *(note: `"path"` rows MUST store UTF-8 `bytes`, e.g. `b"res/rsrc0"`, never `str`)*
   - `CREATE TABLE fragment_properties(id char(40), key char(40), value char(40), primary key (id, key, value)) without rowid`
   - `CREATE TABLE gc_fragment_properties(id varchar(40), key varchar(40), value varchar(40), primary key (id, key, value)) without rowid`
   - `CREATE TABLE gc_reachable(id varchar(40), primary key (id)) without rowid`
   - Explicit `conn.close()` before reading/wrapping `book.kdf` (required on Windows file locking) and `FA 50 0A 5F` 1024-byte fingerprint insertion at offset 1024 and every 1 MiB.
2. **Global Ion fragments:** `$ion_symbol_table` (imports `YJ_symbols` v10 `max_id=842`), `max_id`, `book_navigation`, `book_metadata` (`$490` with `kindle_capability_metadata`: `yj_fixed_layout=1`, `yj_publisher_panels=1`; `kindle_title_metadata`: `book_id`, `title`, `author` if provided, `language`, `cover_image`; `kindle_ebook_metadata`: `selection="enabled"`; `kindle_audit_metadata`: `file_creator="KC"`, `creator_version`), `content_features` (`$585`: `yj_non_pdf_fixed_layout` v2.0 + `yj_publisher_panels` v2.0), `document_data` (`$538`: `16.0`, RTL/LTR layout mode `$375`/`$376`, `max_id`, resource list aux ref, `reading_orders`), `metadata` (`$258`), resource list `$597`, `eidbucket_*` (`$610`), and `yj.section_pid_count_map` (`$611`).
3. **Per-page structure & explicit Guided View panels:** Each page emits `$260` section (`$156: $326` scale-fit), `$259` storyline, image container + leaf (`$270`/`$271` with `$164`/`res/rsrcN`/`$597`), `$609` `<sid>-spm`, and for each sorted `PanelBox` in `page.panels` (`1..N`):
   - Trigger `$608` container (`source_eid`, `$159: $270`, `$156: $324`, `$183: $324`, `%` coords `$59`/`$58`/`$56`/`$57` via `$307`/`$306: $314`, `$684: $441`, `$427: ordinal`, `$426: [{$428: $468, $474: source_eid, $163: target_eid}]`)
   - Target `$608` container (`target_eid`, `$159: $270`, `$156: $324`, `$183: $324`, `%` coords from `ZoomRect`)
   - All `source_eid` and `target_eid` registered in `$259` children (`frag_props`/`gc_frag_props`), `$609` `<sid>-spm`, `$610` `eidbucket_*`, and `yj.section_pid_count_map` count (`3 + 2 * len(panels)`).
4. **KPF ZIP contents:** `book.kdf`, `book.kcb` (clean metadata JSON without unknown keys), and `res/rsrc*` raw image streams (`jpg`/`png`).

- [ ] **Step 1: Write failing tests in `tests/test_kpf.py`**

1. Unit test `IonBinaryWriter` encoding (BVM `0xe0 0x01 0x00 0xea`, `$ion_symbol_table` importing `YJ_symbols` v10, annotated structs, lists, ints, float64s, strings, EID refs).
2. Integration test `KpfBuilder.build()` on 2 synthetic PNG `PageAsset`s with 2 `PanelBox`es each: assert output `.kpf` is a valid ZIP containing `book.kcb`, `book.kdf` (strip `FA 50 0A 5F` fingerprints -> valid SQLite with `capabilities`, `fragments`, `fragment_properties`, `gc_reachable`), and `res/rsrc*` entries; assert panel source/target `$608` fragments with `$426` and `$427` ordinals `1, 2` are present.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kpf.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `IonBinaryWriter` and `KpfBuilder`**

Implement clean Ion 1.0 binary serialization and `.kpf` SQLite/ZIP assembly with explicit OpenCV-derived `PanelBox` source/target Guided View overlays (`ZoomRect` derived cleanly from `zoom_factor` and page dimensions, clamped to `(0, 0, W, H)`). Ensure all SQLite connections are closed in a `finally:` block before reading/wrapping `book.kdf`.

- [ ] **Step 4: Run tests to verify they pass and commit**

Run: `uv run pytest tests/test_kpf.py -v`
Expected: PASS
```bash
git add src/manga2kindle/packaging/__init__.py src/manga2kindle/packaging/ion.py src/manga2kindle/packaging/kpf.py tests/test_kpf.py
git commit -m "feat: add IonBinaryWriter and KpfBuilder for fixed-layout Guided View KPF"
```

---

### Task 7: Headless KFX Compiler (`KfxCompiler`)

**Files:**
- Create: `src/manga2kindle/compiler/__init__.py`
- Create: `src/manga2kindle/compiler/kfx.py`
- Test: `tests/test_kfx_compiler.py`

**Interfaces:**
- Consumes: `PlatformDiscovery` from `manga2kindle.platform`
- Produces in `src/manga2kindle/compiler/kfx.py`:
  - `KfxCompilationError(RuntimeError)`
  - `KfxCompiler`: `__init__(self, calibre_debug_path: Path)`; `compile(self, kpf_path: Path, output_kfx: Path) -> Path`; `verify_pdoc(self, kfx_path: Path) -> bool`

- [ ] **Step 1: Write failing tests in `tests/test_kfx_compiler.py`**

1. Unit test command invocation `[str(calibre_debug_path), "-r", "KFX Output", "--", "-d", str(kpf_path), str(output_kfx)]` and `KfxCompilationError` on non-zero exit or missing output file, plus `verify_pdoc()` returning `True` only when KFX metadata parses `cde_content_type == "PDOC"` and `False` for `EBOK` or missing metadata.
2. Integration test (skipped automatically if `calibre-debug` or `KFX Output` not installed locally): compile a 2-page `.kpf` with panels produced by `KpfBuilder` to `.kfx` via `KfxCompiler.compile()`, and verify `KfxCompiler.verify_pdoc()` returns `True` cleanly.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_kfx_compiler.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `KfxCompiler` in `src/manga2kindle/compiler/kfx.py`**

Invoke `calibre-debug -r "KFX Output" -- -d <kpf_path> <output_kfx>` via `subprocess.run(..., capture_output=True, text=True, check=False)`, verify `.kfx` exists and is non-empty, and implement `verify_pdoc()` structurally (no naive substring grep!): invoke a tiny `calibre-debug -c ...` script importing `calibre_plugins.kfx_output.kfxlib.yj_book.YJ_Book` to inspect `cde_content_type == "PDOC"` (and also validate `KFX` Ion `$490`/`$258` metadata structure directly when mocked in unit tests).

- [ ] **Step 4: Run tests to verify they pass and commit**

Run: `uv run pytest tests/test_kfx_compiler.py -v`
Expected: PASS (including live Calibre `KFX Output` compilation of `KpfBuilder` output!).
```bash
git add src/manga2kindle/compiler/__init__.py src/manga2kindle/compiler/kfx.py tests/test_kfx_compiler.py
git commit -m "feat: add KfxCompiler wrapping calibre-debug KFX Output with PDOC verification"
```

---

### Task 8: Pipeline Orchestrator, Typer CLI, and Documentation

**Files:**
- Create: `src/manga2kindle/pipeline.py`
- Create: `src/manga2kindle/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- Consumes: `models`, `platform`, `extractor`, `preprocessor`, `cv.detector`, `cv.sorter`, `cv.preview`, `packaging.kpf`, `compiler.kfx`
- Produces:
  - `ConversionPipeline` in `src/manga2kindle/pipeline.py`: `run_single(config: ConversionConfig) -> Path`; `run_batch(config: ConversionConfig) -> list[Path]`; `preview_panels(input_path: Path, output_dir: Path, ...) -> list[Path]`
  - `app` (`typer.Typer`) in `src/manga2kindle/cli.py` with `convert`, `preview-panels`, `doctor`

- [ ] **Step 1: Write failing tests in `tests/test_cli_e2e.py`**

Using `typer.testing.CliRunner`:
1. `manga2kindle doctor` renders dependency table cleanly.
2. `manga2kindle preview-panels` on a synthetic `.cbz` creates numbered PNG overlays in `--output-dir`.
3. `manga2kindle convert` on a synthetic `.cbz` with `--no-kcc` (and mocked or real `calibre-debug` when available) produces `.kfx` output and removes temp directory unless `--keep-temp` is passed.
4. Batch `convert` on a directory containing 1 valid `.cbz` and 1 corrupt `.cbz` converts the valid one, logs failure for the corrupt one, and cleans up temp directories.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_e2e.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `ConversionPipeline`, `cli.py`, and update `README.md`**

- `ConversionPipeline`:
  - Uses `ThreadPoolExecutor(max_workers=config.workers)` for parallel page CV/preview/preprocessing and batch archive conversion, and manages temp dirs with `try/finally` cleanup (retaining only when `config.keep_temp=True`).
  - Input disambiguation: when `input_path.is_dir()`, if it contains `.cbz`/`.cbr` files → `run_batch()`; else if it contains images → `run_single()`; else raise clear error.
  - Output resolution: if `-o` ends with `.kfx` (single input), write exact file; if `-o` is a directory or omitted, write `<output_dir or input_path.parent>/<stem>.kfx`.
  - Enforces `KfxCompiler.verify_pdoc(output_kfx)` in `run_single()` (raising `KfxCompilationError` if `False`).
  - `preview_panels()` runs `ArchiveExtractor` + `ImagePreprocessor` (honoring `profile`, `direction`, `spread_mode`, `no_kcc`, `kcc_path`) before `PanelDetector`/`PanelSorter`/`PanelPreviewRenderer` and cleans up its temp work dir via `try/finally`.
- `cli.py` exposes `convert`, `preview-panels`, `doctor` matching all flags from the RFP (`-i/--input`, `-o/--output`, `-p/--profile`, `-d/--direction`, `-s/--spread-mode`, `-z/--zoom-factor`, `--padding`, `--min-panel-area`, `--kcc-path`, `--no-kcc`, `--keep-temp`, `-w/--workers`, `--calibre-path`, `--previewer-path`). Note `--title` and `--author` optional overrides default cleanly from archive/folder stem.
- Update `README.md` with project overview, prerequisites (Calibre + KFX Output plugin, Kindle Previewer 4, UnRAR), macOS (`brew`) & Windows (`winget`/`choco`) install commands, `uv` setup, and CLI usage examples.

- [ ] **Step 4: Run full test suite & live smoke test on sample manga `.cbz`**

Run: `uv run pytest -v`
Expected: All unit and integration tests pass.
Run live smoke test on one sample manga chapter (e.g., `Manga Vol. 01 (Digital).cbz`) outputting to a temporary gitignored folder (do NOT commit output):
- `uv run manga2kindle doctor`
- `uv run manga2kindle preview-panels -i "/path/to/Manga Vol. 01.cbz" -o ./preview`
- `uv run manga2kindle convert -i "/path/to/Manga Vol. 01.cbz" -o ./debug_inspect`

- [ ] **Step 5: Commit**

```bash
git add src/manga2kindle/pipeline.py src/manga2kindle/cli.py README.md tests/test_cli_e2e.py
git commit -m "feat: add ConversionPipeline, Typer CLI, e2e tests, and README documentation"
```
