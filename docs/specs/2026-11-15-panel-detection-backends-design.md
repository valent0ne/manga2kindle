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
- `MAGI` is selected only if the `[magi]` extra is installed, weights are cached locally, a GPU (`cuda` or `mps`) is available, and `--allow-remote-code` is enabled (or configured).
- `ONNX_YOLO` is selected if `[onnx]` is installed and weights are cached locally.
- `CONTOUR` is always available as the instant fallback.
- Any load failure or zero-panel result in `AUTO` logs a warning and falls through to the next layer before the full-page fallback.
- Explicit `--detector <name>` fails fast with an actionable message if dependencies/weights are missing.

## 4. Architecture

New or updated modules under `src/manga2kindle/cv/`:

```text
cv/
  backends/
    __init__.py        # BackendRegistry, create_backend(kind, config)
    base.py            # PanelDetectorBackend (Protocol), DetectionResult
    contour.py         # ContourBackend (moved from detector.py)
    onnx_yolo.py       # OnnxYoloBackend (Layer 2: deepghs/manga109_yolo ONNX)
    magi.py            # MagiBackend (Layer 3: ragavsachdeva/magiv3)
  detector.py          # Thin compatibility shim re-exporting ContourBackend as PanelDetector
  sorter.py            # Fixed XY-cut / gutter-aware reading-order sorter
  boxes.py             # Shared geometry: compute_iou, compute_containment, nms_and_prune, pad_and_clamp
  benchmark.py         # Evaluation harness + PopManga/CoMix loaders
  preview.py           # Extended for multi-backend color-coded / side-by-side overlays
```

### Shared Geometry & Bugfixes (`boxes.py` & `sorter.py`)
1. **Containment Suppression (`boxes.py`):**
   - Add `compute_containment(inner, outer) = inter_area / min(area_a, area_b)`.
   - In `nms_and_prune(boxes, iou_threshold=0.4, containment_threshold=0.85)`: sort boxes by area descending (or confidence descending when scores are provided, keeping the larger box when one strictly contains the other), and suppress any smaller box whose `IoU > iou_threshold` **or** `containment > containment_threshold`.
   - Reused by all three backends so nested/contained panels never render separately.
2. **Stacked-Panel Reading Order (`sorter.py`):**
   - Replace flat `_sort_by_rows` with **recursive XY-cut / gutter-aware partitioning**:
     - First try splitting the box set along horizontal gutters (vertical gaps where no box spans across, or minimal overlap) into top-to-bottom bands.
     - Within each band, split along vertical gutters (horizontal gaps where no box spans across) into columns ordered RTL (`x` descending) or LTR (`x` ascending).
     - Recurse on each sub-group until 1 box remains (or fall back to top-to-bottom, then RTL/LTR when boxes overlap without a clean cut).
   - Two panels stacked vertically next to a tall neighbor are now cleanly isolated into a sub-column and ordered top-to-bottom.

### Domain Types (`models.py`)
- `DetectorKind(str, Enum)`: `AUTO`, `CONTOUR`, `ONNX_YOLO`, `MAGI`.
- `DetectionResult` (dataclass): `panels: list[PanelBox]`, `ordered: bool`, `backend: DetectorKind`, `scores: list[float] | None`.
- `DetectorConfig` (Pydantic): `kind`, `model_path: Path | None`, `model_id: str | None`, `device` (`auto|cpu|cuda|mps`), `score_threshold`, `iou_threshold`, `containment_threshold`, `padding`, `min_panel_area`, `allow_remote_code: bool`.

### Interface
```python
class PanelDetectorBackend(Protocol):
    kind: DetectorKind
    def detect(self, image_path: Path) -> DetectionResult: ...
    def detect_batch(self, image_paths: Sequence[Path]) -> list[DetectionResult]: ...
```
- All backends return `PanelBox` in source-page pixel coordinates, clamped to bleed-safe bounds.
- Reading order: if `DetectionResult.ordered` is `True` (`MAGI`), preserve it; otherwise run `PanelSorter`.

### Models & Cache Hygiene
- Cached weights live in the OS user cache directory via `platformdirs` (`pathlib.Path`), never inside the repository.
- `manga2kindle models fetch <name>` downloads pinned revisions, verifies SHA-256, and prints model source, size, and license terms before/after download.

## 5. CLI & Pre-Conversion Preview

- **`convert`:**
  - `--detector [auto|contour|onnx|magi]` (default `auto`)
  - `--detector-model PATH_OR_ID`
  - `--device [auto|cpu|cuda|mps]`
  - `--score-threshold FLOAT`
  - `--allow-remote-code` (required for `magi`)
  - `--preview-only` (runs extraction, preprocessing, and panel detection/sorting, writes preview overlays, and exits without KPF/KFX compilation—ideal for checking a chapter before mass converting)
- **`preview-panels`:**
  - Accepts all `--detector*`, `--device`, `--score-threshold`, `--allow-remote-code` flags.
  - `--compare contour,onnx,magi`: renders side-by-side or color-coded comparison PNGs with backend name, panel count, and reading-order arrows so the user can visually compare layers before batch conversion.
- **`doctor`:**
  - Reports `onnxruntime` (and active providers), `torch` + CUDA/MPS availability, cached models, and active default `AUTO` layer.
- **Optional extras in `pyproject.toml`:**
  - `onnx = ["onnxruntime>=1.17", "platformdirs>=4.0"]`
  - `magi = ["torch>=2.2", "transformers>=4.40", "einops>=0.7", "timm>=0.9", "platformdirs>=4.0"]`
  - `benchmark = ["datasets>=2.18"]` (for streaming/loading Hugging Face benchmark splits locally)

## 6. Licensing & Public Repo Compliance

- Project license is **AGPL-3.0-or-later**.
- **Never** commit datasets, model weights, sample manga pages, or machine-specific local paths to git. Ensure `.gitignore` covers `benchmarks/`, `data/`, `*.onnx`, `*.safetensors`, `*.pt`, and local cache folders.
- **Model & Dataset Terms:**
  - `ragavsachdeva/magiv3` & `ragavsachdeva/popmanga_test`: **non-commercial / personal / research use only** (PopManga also requires Hugging Face login + terms acceptance). Displayed at `models fetch` / `benchmark` time and documented in `README.md`.
  - `deepghs/manga109_yolo`: trained on `Manga109-s`; document attribution in `README.md` and `docs/`.
  - `emanuelevivoli/comix-v0_1-pages` (CoMix): `CC BY-SA 4.0` / `CC0` (per-component metadata).

## 7. Benchmark & Evaluation (`manga2kindle benchmark`)

### Dataset Choice
- **Primary Benchmark:** **`ragavsachdeva/popmanga_test` (PopManga)** (~2,000 modern manga pages with COCO-style panel bounding boxes **and** reading-order ground truth).
  - Chosen because it is the official benchmark for Magi, requires no manual access request (only HF login/terms), and evaluates both **panel detection** and **reading order** out of the box.
- **Optional Cross-Domain Evaluation:** **`emanuelevivoli/comix-v0_1-pages` (CoMix)** for testing generalization on Western/diverse comic layouts, plus any user-supplied local folder (`--pages DIR --truth FILE`).
- *Why not Manga109-s / MangaSeg as primary:* Manga109-s requires a manual access approval gate and lacks reading-order annotations; MangaSeg adds pixel masks on top of Manga109 whereas Kindle Guided View targets rectangular regions.

### Metrics Reported
`manga2kindle benchmark --dataset popmanga --detectors contour,onnx,magi --limit 200 --out report.json`
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
