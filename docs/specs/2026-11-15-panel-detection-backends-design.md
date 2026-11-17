# Pluggable Panel Detection Backends: Design Spec

**Date:** 2026-11-17 (revised from 2026-11-15)
**Status:** Approved, ready for implementation planning
**Related:** [2026-11-15-manga2kindle-design.md](./2026-11-15-manga2kindle-design.md) (§5, Stage 3)

## 1. Intent & Why

Stage 3 today consists of `PanelDetector` (`src/manga2kindle/cv/detector.py`) and `PanelSorter` (`src/manga2kindle/cv/sorter.py`).

Observed failure modes in real manga chapters:
1. **Nested/contained panels:** `PanelDetector.non_max_suppression` only prunes boxes with `IoU > 0.4`. When a small panel sits inside a larger panel, `IoU = area_small / area_large` is well below `0.4`, so both boxes survive and render as separate Guided View targets.
2. **Stacked panels misordered:** `PanelSorter._sort_by_rows` clusters any panel that vertically overlaps a tall neighbor (`overlap_h >= 0.5 * min_h`) into the same row, then sorts that row by `x` before `y`. Two panels stacked one above the other next to a tall panel get scrambled.
3. **Classical contour limits:** Borderless panels, bleeds, diagonal/overlapping layouts, art crossing gutters, and heavy screentones cause missed or merged panels.

This spec introduces a **3-layer pluggable backend architecture** (`CONTOUR` → `ONNX_YOLO` → `MAGI`), fixes both shared geometry/sorting bugs across all applicable layers, adds pre-conversion visual previewing (`preview-panels --compare` and `convert --preview-only`), and adds a reproducible benchmark (`manga2kindle benchmark`) backed by **PopManga** (and optional **CoMix**).

## 2. Goals, Non-Goals & Anti-Goals

### Goals
- Support a **3-layer detection stack** behind a common `PanelDetectorBackend` interface:
  - **Layer 1 (`CONTOUR`):** Zero-GPU, zero-extra-dependency OpenCV baseline (crazy fast).
  - **Layer 2 (`ONNX_YOLO`):** CPU-capable learned model via `onnxruntime` (no `torch`), running cleanly on low-power CPUs (e.g., Intel i3 NAS), Apple Silicon (M-series), and Windows.
  - **Layer 3 (`MAGI`):** SOTA GPU model (`ragavsachdeva/magiv3`, CUDA) with native reading-order prediction.
- Fix **nested-panel suppression** in shared post-processing (`boxes.py`) so contained boxes are pruned for all backends.
- Fix **stacked-panel reading order** in `PanelSorter` (`sorter.py`) for Layer 1 & Layer 2.
- Provide **pre-conversion panel previewing** (`preview-panels` with `--detector` / `--compare`, and `convert --preview-only`) so users can inspect bounding boxes and reading order before mass converting.
- Provide a **reproducible benchmark** (`manga2kindle benchmark`) reporting accuracy, reading-order, and speed/memory metrics across all three layers using **PopManga** (`ragavsachdeva/popmanga_test`).

### Non-Goals
- No model training or fine-tuning pipeline in v1. Use existing pretrained SOTA/community weights.
- No OCR, speaker diarisation, or text extraction in the Kindle output (even when Magi computes them internally).
- No GUI or interactive panel editor.
- No dataset or model weight redistribution in this public repository.

### Anti-Goals
- Silently downloading large weights without explicit user action (`models fetch`).
- Silently falling back in `AUTO` mode without logging/printing which backend ran.
- Making `torch` or `ultralytics` a core dependency.
- Committing dataset files, model weights, or machine-specific local paths to the repository.

## 3. 3-Layer Backend Selection & Scouted Models

| Layer | Backend ID | Model / Implementation | Runtime / Deps | Target Hardware | Reading Order | License Summary (verify against model card) |
|---|---|---|---|---|---|---|
| **1** | `contour` | Classical OpenCV (`ContourBackend`) | `opencv-python-headless` (core) | Any CPU (crazy fast, zero-dep) | `PanelSorter` (fixed) | Project license (`AGPL-3.0-or-later`) |
| **2** | `onnx` | `deepghs/manga109_yolo` pre-exported `model.onnx` (e.g., YOLOv11n/s) | `onnxruntime` + `platformdirs` (`[onnx]` extra; no `torch` / no `ultralytics`) | CPU-capable (NAS, M-series Mac, Windows; optional CoreML/DirectML/CUDA EPs) | `PanelSorter` (fixed) | Trained on Manga109-s (requires Manga109-s attribution; Ultralytics YOLO base weights are `AGPL-3.0`, compatible with this project; do not redistribute raw Manga109-s data) |
| **3** | `magi` | `ragavsachdeva/magiv3` (with `ragavsachdeva/magiv2` as optional fallback variant) | `torch` + `transformers` + `einops` + `timm` (`[magi]` extra, `trust_remote_code=True`) | Discrete GPU / CUDA recommended (e.g., RTX-class) | Native model reading order (`ordered=True`) | Non-commercial / personal / research use only per model card; requires explicit `--allow-remote-code` |

### `AUTO` Policy
Order: **`MAGI` → `ONNX_YOLO` → `CONTOUR`**.
- `MAGI` is selected only if the `[magi]` extra is installed, weights are cached locally, a GPU (`cuda` or `mps`) is available, and `--allow-remote-code` is enabled. If `[magi]`, GPU, and cached weights are all present without `--allow-remote-code`, `AUTO` logs a warning and skips `MAGI`. (Explicit `--detector magi` may specify `--device cpu`.)
- `ONNX_YOLO` is selected if `[onnx]` is installed and weights are cached locally (or `--detector-model` points to a local `.onnx` file).
- `CONTOUR` is always available as the instant fallback.
- `AUTO` logs which backend is active on startup. A load/initialization failure disables that backend for the remainder of the run with a single warning and falls through to the next layer. A zero-panel page result (`panels=[]`) in `AUTO` logs a warning/note and falls through to remaining layers for that page before applying the full-page fallback `[PanelBox(0, 0, page_w, page_h)]`.
- Explicit `--detector <name>` fails fast with an actionable error if dependencies, weights, requested execution provider, or `--allow-remote-code` are missing, and only falls back to full-page `[PanelBox(0, 0, page_w, page_h)]` when detection legitimately returns zero panels on a page.

### Device & Execution Provider Mapping
- `DevicePreference(str, Enum)`: `AUTO = "auto"`, `CPU = "cpu"`, `CUDA = "cuda"`, `MPS = "mps"`.
- **`MagiBackend`:** Maps directly to `torch` devices (`cuda`, `mps`, `cpu`).
- **`OnnxYoloBackend`:** `auto` selects the highest-priority provider available in `onnxruntime.get_available_providers()` (`CUDAExecutionProvider` → `CoreMLExecutionProvider` → `DmlExecutionProvider` → `CPUExecutionProvider`). Explicit `mps` maps to `CoreMLExecutionProvider` on Apple Silicon; explicit `cuda` maps to `CUDAExecutionProvider` (or `DmlExecutionProvider` only when `auto`). Unavailable explicit providers raise an actionable `RuntimeError`.

## 4. Architecture

New or updated modules under `src/manga2kindle/cv/`:

```text
cv/
  backends/
    __init__.py        # BackendRegistry, resolve_auto_chain, detect_with_fallback
    base.py            # PanelDetectorBackend (Protocol), DetectionResult
    contour.py         # ContourBackend (extracted from detector.py)
    onnx_yolo.py       # OnnxYoloBackend (Layer 2: deepghs/manga109_yolo ONNX)
    magi.py            # MagiBackend (Layer 3: ragavsachdeva/magiv3)
  detector.py          # Compatibility wrapper preserving PanelDetector -> list[PanelBox] API
  sorter.py            # Fixed two-tier XY-cut / gutter-aware reading-order sorter
  boxes.py             # Shared geometry: compute_iou, compute_containment, nms_and_prune[_indices], pad_and_clamp
  models_hub.py        # OS cache resolution (platformdirs), pinned downloads, SHA-256 verification, license notices
  benchmark.py         # Evaluation harness + PopManga/CoMix/local loaders
  preview.py           # Extended with side-by-side multi-backend comparison overlays
```

### Shared Geometry & Bugfixes (`boxes.py` & `sorter.py`)
1. **Containment Suppression (`boxes.py`):**
   - Add `compute_containment(box_a, box_b) = inter_area / min(area_a, area_b)` (returning `0.0` if either box has zero area).
   - Provide `nms_and_prune_indices(boxes, iou_threshold=0.4, containment_threshold=0.85, scores=None, preserve_order=False) -> list[int]` and convenience wrapper `nms_and_prune(...) -> list[Rect]`, executed in two stages:
     1. **Containment pass** (sorted by `area` descending, ties by `score` descending): suppress any smaller box strictly contained inside a larger surviving box (`compute_containment(cand, kept) > containment_threshold` and `area(cand) < area(kept)`), ensuring a higher-confidence inset box never suppresses its enclosing outer panel.
     2. **IoU NMS pass** (sorted by `(score, area)` descending when `scores` are provided, else `area` descending): suppress any candidate with `compute_iou(cand, kept) > iou_threshold`.
   - When `preserve_order=True` (used by `MagiBackend`), return surviving indices/boxes in their original input order so native reading order is never scrambled; returning indices also keeps `DetectionResult.scores` 1:1 aligned with `DetectionResult.panels`.
2. **Stacked-Panel Reading Order (`sorter.py`):**
   - Replace flat `_sort_by_rows` in `_sort_group` with **two-tier recursive XY-cut gutter partitioning** (after existing `_sort_spread_halves` when `is_spread=True` and `page_width` is known):
     1. Try **strict** horizontal gutter split (non-overlapping `y` intervals sorted top-to-bottom); if `>= 2` bands, recurse on each band.
     2. Otherwise try **strict** vertical gutter split (non-overlapping `x` intervals ordered RTL `x` descending or LTR `x` ascending); if `>= 2` columns, recurse on each column.
     3. If neither strict cut splits the group (e.g., `padding=0.02` causes 2–15px overlap across tight gutters), retry horizontal then vertical cuts using **inset** intervals (inset each box edge by up to `min(24, int(0.05 * dim))` so minor border/padding overlap never bridges a real gutter).
     4. If still unsplittable (genuinely overlapping/diagonal layout), fall back to `_sort_by_rows` with `row_tolerance_px` and `(y, -x/x)` ordering.
   - Two panels stacked vertically next to a tall neighbor are cleanly isolated into a column and ordered top-to-bottom even after `pad_and_clamp` expansion.

### Domain Types (`models.py`)
- `DetectorKind(str, Enum)`: `AUTO = "auto"`, `CONTOUR = "contour"`, `ONNX_YOLO = "onnx"`, `MAGI = "magi"`.
- `DevicePreference(str, Enum)`: `AUTO = "auto"`, `CPU = "cpu"`, `CUDA = "cuda"`, `MPS = "mps"`.
- `DetectionResult` (dataclass): `panels: list[PanelBox]`, `ordered: bool`, `backend: DetectorKind`, `page_width: int = 0`, `page_height: int = 0`, `scores: list[float] | None = None`.
- `DetectorConfig` (Pydantic): `kind: DetectorKind = DetectorKind.AUTO`, `model_path: Path | None = None`, `model_id: str | None = None`, `device: DevicePreference = DevicePreference.AUTO`, `score_threshold: float = 0.35`, `iou_threshold: float = 0.4`, `containment_threshold: float = 0.85`, `padding: float = 0.02`, `min_panel_area: float = 0.03`, `allow_remote_code: bool = False`.
- `ConversionConfig` (Pydantic): Add `detector: DetectorConfig = Field(default_factory=DetectorConfig)`. Top-level `padding` and `min_panel_area` stay backward-compatible and synchronize with `detector` via a `model_validator` when `detector` uses default field values.

### Interface & Pipeline Integration
```python
class PanelDetectorBackend(Protocol):
    kind: DetectorKind
    def detect(self, image_path: Path) -> DetectionResult: ...
    def detect_batch(self, image_paths: Sequence[Path]) -> list[DetectionResult]: ...
```
- All backends return `PanelBox` in source-page pixel coordinates, clamped via `pad_and_clamp`. When zero panels are detected, `backend.detect()` returns `panels=[]` (no synthetic full-page box) so `AUTO` fallback and `benchmark` see true empty detections; `detect_with_fallback()` (pipeline) and the `PanelDetector` wrapper (`detector.py`) apply `[PanelBox(0, 0, page_w, page_h, ordinal=1)]` when no backend finds panels.
- `min_panel_area` filters contour noise in `ContourBackend`; learned backends (`OnnxYoloBackend`, `MagiBackend`) filter by `score_threshold` (if scored) and degenerate zero/sub-pixel bounds so valid small panels (e.g., 4-koma strips) are not inadvertently dropped.
- Reading order: if `DetectionResult.ordered` is `True` (`MAGI`), preserve it and ensure 1-based `ordinal` (`1..N`); otherwise run `PanelSorter`.
- **Concurrency & Model Reuse (`pipeline.py`):** `ConversionPipeline` resolves and caches backend instances once per run (and across batch workers), rather than reloading weights per page or per book. `OnnxYoloBackend` and `MagiBackend` guard inference with a lock and run `detect_batch()` under `torch.inference_mode()` so multi-worker `run_batch` never triggers GPU OOM or ORT thread oversubscription; `ContourBackend.detect_batch()` may parallelize CPU work across `workers`.

### Models & Cache Hygiene (`models_hub.py`)
- Cached weights live in `platformdirs.user_cache_dir("manga2kindle")` (`pathlib.Path`), never inside the repository.
- `[onnx]` fetches single pinned `.onnx` files over HTTPS via stdlib `urllib.request` (no `huggingface_hub` dependency required) and verifies SHA-256; `[magi]` uses `huggingface_hub.snapshot_download` (pulled by `transformers`) with a pinned commit hash.
- `manga2kindle models fetch <name>` (and `models list`) prints model source, size, and license notice before downloading; runtime inference loads strictly from local cache (`local_files_only=True`) and never downloads implicitly.

## 5. CLI & Pre-Conversion Preview

- **`convert`:**
  - `--detector [auto|contour|onnx|magi]` (default `auto`)
  - `--detector-model PATH_OR_ID` (resolved as `model_path` when it exists on disk, ends with `.onnx`, or has path separators/drive/dot prefix; otherwise resolved as a registered hub key or HF `owner/repo` `model_id`; missing explicit paths fail fast)
  - `--device [auto|cpu|cuda|mps]` (default `auto`)
  - `--score-threshold FLOAT` (default `0.35`, `[0.0, 1.0]`)
  - `--allow-remote-code` (required for `magi`)
  - `--preview-only` (supports both single and batch inputs: runs extraction, preprocessing, and panel detection/sorting, writes preview overlays to `<input.parent>/<name>_preview` / `<out_dir>/<stem>_preview` or explicit dir, and skips `calibre-debug` lookup + KPF/KFX compilation)
- **`preview-panels`:**
  - Accepts all `--detector*`, `--device`, `--score-threshold`, `--allow-remote-code` flags.
  - `--compare contour,onnx,magi`: accepts a comma-separated list of explicit backends (rejects `auto`; fails fast if any requested backend is unavailable), and renders **side-by-side tiled comparison PNGs** with per-backend color-coded boxes, `[ordinal]` badges, reading-order arrows, and a `BACKEND (N panels)` header banner so the user can visually compare layers before batch conversion.
- **`models` subcommands:**
  - `manga2kindle models list`: show available registered models, cache paths, cache status, and licenses.
  - `manga2kindle models fetch <name>`: download pinned model artifacts with SHA-256 verification and license notice.
- **`doctor`:**
  - Reports `onnxruntime` (and active providers), `torch` + CUDA/MPS availability, cached models, and resolved `AUTO` layer as optional diagnostics (never causing `doctor` to exit nonzero when absent).
- **Optional extras in `pyproject.toml`:**
  - `onnx = ["onnxruntime>=1.17", "platformdirs>=4.0"]`
  - `magi = ["torch>=2.2", "transformers>=4.40", "einops>=0.7", "timm>=0.9", "platformdirs>=4.0"]` (plus `scipy` / `torchvision` if required by the pinned `magiv3` commit)
  - `benchmark = ["datasets>=2.18", "psutil>=5.9"]` (for loading Hugging Face benchmark splits and cross-platform Windows/macOS RAM tracking)

## 6. Licensing & Public Repo Compliance

- Project license is **AGPL-3.0-or-later**.
- **Never** commit datasets, model weights, sample manga pages, or machine-specific local paths to git. Ensure `.gitignore` covers `benchmarks/`, `data/`, `*_preview/`, `*.onnx`, `*.safetensors`, `*.pt`, `*.bin`, `*.pth`, `*.parquet`, `*.arrow`, and local cache folders.
- **Model & Dataset Terms:**
  - `ragavsachdeva/magiv3` & `ragavsachdeva/popmanga_test`: **non-commercial / personal / research use only** (PopManga also requires Hugging Face login + terms acceptance). Displayed at `models fetch` / `benchmark` time and documented in `README.md`.
  - `deepghs/manga109_yolo`: trained on `Manga109-s`; document attribution in `README.md` and `docs/`.
  - `emanuelevivoli/comix-v0_1-pages` (CoMix): `CC BY-SA 4.0` / `CC0` (per-component metadata).

## 7. Benchmark & Evaluation (`manga2kindle benchmark`)

### Dataset Choice & Protocol
- **Primary Benchmark:** **`ragavsachdeva/popmanga_test` (PopManga)** (~2,000 modern manga pages with COCO-style panel bounding boxes **and** reading-order ground truth).
  - Chosen because it is the official benchmark for Magi, requires no manual access request (only HF login/terms), and evaluates both **panel detection** and **reading order** out of the box.
- **Optional Cross-Domain Evaluation:** **`emanuelevivoli/comix-v0_1-pages` (CoMix)** for testing generalization on Western/diverse comic layouts, plus any user-supplied local folder (`--pages DIR --truth FILE`, supporting COCO JSON or a simple JSON list of `{image, panels: [[x, y, w, h], ...]}` in reading order).
- *Why not Manga109-s / MangaSeg as primary:* Manga109-s requires a manual access approval gate and lacks reading-order annotations; MangaSeg adds pixel masks on top of Manga109 whereas Kindle Guided View targets rectangular regions.
- **Evaluation Protocol:**
  - Evaluates raw dataset page images directly (`backend.detect()` without full-page fallback or KCC/Pillow resizing) so coordinates match dataset ground truth.
  - `--limit N` selects the same deterministic slice across all compared detectors (`--direction` defaults to `rtl` for `popmanga` and `ltr` for `comix`, overridable via CLI).
  - Reading-order metrics (`pairwise_order_acc`, `kendall_tau`) perform **1-to-1 greedy IoU matching (`IoU >= 0.50`)** between predicted and ground-truth panels per page, then score the induced order over matched pairs (`n >= 2`).
  - Latency excludes a warmup pass and calls `torch.cuda.synchronize()` before stopping timers on CUDA; `peak_ram_mb` uses `psutil` RSS tracking and `peak_vram_mb` uses `torch.cuda.max_memory_allocated()`.

### Metrics Reported
`manga2kindle benchmark --dataset popmanga --detectors contour,onnx,magi --limit 200 --out report.json`
Also supports `--split`, `--direction`, `--device`, `--score-threshold`, `--allow-remote-code`, and `--pages DIR --truth FILE`.
Prints a Rich comparison table and writes a JSON summary with:
- **Panel Recall & Precision** at `IoU >= 0.50` and `IoU >= 0.75`
- **Mean Best-Match IoU**
- **Nested-Duplicate Rate** (count of surviving boxes with `containment > 0.85`, verifying the nested-panel fix)
- **Exact Panel Count Match Rate**
- **Reading-Order Accuracy** (pairwise order accuracy & Kendall's tau against PopManga ground truth)
- **Latency (`ms/page`)** on CPU and GPU
- **Peak Memory (`RAM MB` / `VRAM MB`)**

## 8. Testing Plan

- **Unit tests (no weights or network):**
  - `boxes.py`: `compute_iou`, `compute_containment`, `nms_and_prune` (explicitly testing nested small-inside-large boxes), `pad_and_clamp`.
  - `sorter.py`: regression tests for stacked panels next to tall panels (both RTL and LTR), 4-koma stacks, and spreads.
  - `backends/`: `BackendRegistry`, `AUTO` fallback chain (`MAGI → ONNX_YOLO → CONTOUR`) using stub backends.
  - `onnx_yolo.py`: post-processing & de-letterboxing on synthetic raw YOLO output tensors.
  - `benchmark.py`: metric calculations (precision/recall, Kendall tau, pairwise accuracy, containment rate) on synthetic boxes.
- **Integration tests:**
  - `preview-panels --compare` and `convert --preview-only` with stub backends.
- **Model/Benchmark smoke tests (`@pytest.mark.model`, skipped by default in CI):**
  - End-to-end inference and benchmark run on local GPU/CPU when weights and local PopManga cache are present.

## 9. Phasing

1. **P1 (Geometry & Sorter Fixes + Interface):** Extract `boxes.py` with containment suppression, fix `sorter.py` XY-cut stacked-panel ordering, introduce `PanelDetectorBackend` + `ContourBackend`, keep `PanelDetector` shim.
2. **P2 (Layer 2 — `OnnxYoloBackend`):** Implement `deepghs/manga109_yolo` ONNX backend, `models fetch`, and `AUTO` fallback.
3. **P3 (Layer 3 — `MagiBackend`):** Implement `ragavsachdeva/magiv3` GPU backend with native reading order and `--allow-remote-code`.
4. **P4 (Preview & Benchmark):** Add `preview-panels --compare`, `convert --preview-only`, and `manga2kindle benchmark` (PopManga + CoMix), then record benchmark results (metrics only, no raw data) in `docs/`.

## 10. Left to the Builder

- Verify exact tensor input/output shapes and class index mapping of the pinned `deepghs/manga109_yolo` ONNX file before implementing `OnnxYoloBackend`.
- Verify `ragavsachdeva/magiv3` `predict_detections_and_associations` return schema and reading-order output format against the pinned HF commit hash.
- Verify PopManga (`ragavsachdeva/popmanga_test`) dataset schema (panel box coordinates + reading-order field names) when writing the benchmark loader.
