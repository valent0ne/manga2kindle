# Pluggable 3-Layer Panel Detection Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a 3-layer pluggable panel detection architecture (`CONTOUR` → `ONNX_YOLO` → `MAGI`), fix nested-panel suppression and stacked-panel reading order, add pre-conversion comparison previews, and build a PopManga/CoMix benchmark harness.

**Architecture:** Extract shared bounding-box geometry (`boxes.py`) with containment pruning, upgrade `PanelSorter` (`sorter.py`) to recursive XY-cut gutter partitioning, introduce a `PanelDetectorBackend` protocol with `ContourBackend`, `OnnxYoloBackend`, and `MagiBackend` behind an `AUTO` fallback chain, and expose `--compare`, `--preview-only`, `models fetch`, and `benchmark` in the Typer CLI.

**Tech Stack:** Python 3.11+, OpenCV (`opencv-python-headless`), Pydantic v2, Typer + Rich, `onnxruntime` + `platformdirs` (`[onnx]` extra), `torch` + `transformers` + `einops` + `timm` (`[magi]` extra), `datasets` (`[benchmark]` extra), `pytest`.

**Spec:** [2026-11-15-panel-detection-backends-design.md](../specs/2026-11-15-panel-detection-backends-design.md)

## Global Constraints

- Target Python 3.11+ on both macOS and Windows using `pathlib.Path` exclusively for filesystem paths.
- Keep `torch` and `ultralytics` out of core dependencies; learned backends live in optional extras (`[onnx]`, `[magi]`, `[benchmark]`).
- Use structured `Pydantic` models or `dataclass`es and `Enum`s / named constants—no untyped dicts for internal state and no magic strings.
- Never commit dataset files, model weights, sample manga chapters, or machine-specific local paths to the repository.
- Honor all model and dataset licenses (display non-commercial/attribution notices at `models fetch` and `benchmark` time and in `README.md`).
- Follow Conventional Commits without `Co-Authored-By` trailers.

## Review Focus

1. **Nested small-inside-large panel:** A small panel completely inside a larger panel (`IoU < 0.4` but `containment > 0.85`) must be pruned across all backends (`Task 1`).
2. **Stacked panels beside a tall panel:** Two panels stacked vertically next to a tall panel that overlaps both vertically must sort top-to-bottom in reading order (`Task 2`).
3. **Letterbox coordinate round-trip on non-square pages:** `OnnxYoloBackend` detections on tall/wide manga pages must map back to exact source-page pixel bounds without clipping or offset (`Task 4`).
4. **Missing `--allow-remote-code` for `MAGI`:** Explicit `--detector magi` without `--allow-remote-code` must fail with an actionable error; `AUTO` must skip `MAGI` with a warning (`Task 5`, `Task 6`).
5. **Zero-panel splash page in `AUTO`:** When a learned backend returns 0 panels on a splash page, `AUTO` walks remaining backends before falling back to full-page `[PanelBox(0, 0, page_w, page_h)]` (`Task 6`).

---

### Task 1: Shared Geometry & Containment Suppression (`cv/boxes.py`)

**Files:**
- Create: `src/manga2kindle/cv/boxes.py`
- Modify: `src/manga2kindle/cv/detector.py`
- Test: `tests/test_boxes.py`

**Interfaces:**
- Produces:
  - `Rect = tuple[int, int, int, int]`
  - `compute_iou(box_a: Rect, box_b: Rect) -> float`
  - `compute_containment(box_a: Rect, box_b: Rect) -> float`
  - `nms_and_prune(boxes: list[Rect], iou_threshold: float = 0.4, containment_threshold: float = 0.85, scores: list[float] | None = None) -> list[Rect]`
  - `pad_and_clamp(rect: Rect, page_w: int, page_h: int, padding: float = 0.02, bleed_margin: int = 2) -> PanelBox`

- [ ] **Step 1: Write failing tests for containment suppression and geometry helpers**

```python
from manga2kindle.cv.boxes import compute_containment, compute_iou, nms_and_prune, pad_and_clamp


def test_compute_containment_nested_box_is_one():
    outer = (0, 0, 400, 600)
    inner = (50, 50, 100, 100)
    assert compute_iou(outer, inner) < 0.1
    assert compute_containment(outer, inner) == 1.0


def test_nms_and_prune_removes_nested_small_panel():
    outer = (10, 10, 500, 700)
    inner = (60, 80, 120, 140)
    kept = nms_and_prune([inner, outer], iou_threshold=0.4, containment_threshold=0.85)
    assert kept == [outer]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_boxes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manga2kindle.cv.boxes'`

- [ ] **Step 3: Implement `src/manga2kindle/cv/boxes.py` and update `detector.py` to import from it**

Extract `compute_iou` and `pad_and_clamp` from `detector.py`. Implement `compute_containment(box_a, box_b) -> float` as `inter_area / min(area_a, area_b)`. In `nms_and_prune`, sort by area descending (when `scores is None`) or `(score, area)` descending, and suppress any candidate where `compute_iou(cand, kept_box) > iou_threshold` or `compute_containment(cand, kept_box) > containment_threshold`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_boxes.py tests/test_detector_sorter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/manga2kindle/cv/boxes.py src/manga2kindle/cv/detector.py tests/test_boxes.py
git commit -m "fix(cv): extract shared boxes module with containment suppression"
```

---

### Task 2: XY-Cut Stacked-Panel Reading-Order Fix (`cv/sorter.py`)

**Files:**
- Modify: `src/manga2kindle/cv/sorter.py`
- Test: `tests/test_detector_sorter.py`

**Interfaces:**
- Consumes: `PanelBox`, `ReadingDirection`
- Produces: `PanelSorter.sort(boxes: list[PanelBox], *, page_width: int | None = None, is_spread: bool = False) -> list[PanelBox]`

- [ ] **Step 1: Write failing tests for stacked panels beside a tall panel (RTL and LTR)**

```python
def test_stacked_panels_next_to_tall_panel_rtl():
    # Right column: tall panel; Left column: two stacked panels overlapping tall panel's y-range
    tall_right = PanelBox(x=520, y=20, w=460, h=700)
    top_left = PanelBox(x=20, y=30, w=460, h=320)
    bottom_left = PanelBox(x=40, y=380, w=440, h=330)

    ordered = PanelSorter(direction=ReadingDirection.RTL).sort([bottom_left, top_left, tall_right])
    assert [(b.x, b.y, b.ordinal) for b in ordered] == [
        (520, 20, 1),
        (20, 30, 2),
        (40, 380, 3),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector_sorter.py -k stacked -v`
Expected: FAIL (currently `bottom_left` with `x=40` sorts before `top_left` with `x=20` in RTL row clustering)

- [ ] **Step 3: Implement recursive XY-cut gutter partitioning in `PanelSorter._sort_group`**

Implement `_xy_cut_sort(boxes: list[PanelBox]) -> list[PanelBox]`:
1. Attempt horizontal gutter split (sweep `y` intervals to find non-overlapping horizontal bands sorted top-to-bottom); if `>= 2` bands, recurse on each band.
2. Otherwise attempt vertical gutter split (sweep `x` intervals to find non-overlapping vertical columns ordered RTL `x` descending or LTR `x` ascending); if `>= 2` columns, recurse on each column.
3. If no clean gutter exists (overlapping panels), fall back to row-clustering with `row_tolerance_px` and `(y, -x/x)` tie-breaking.

- [ ] **Step 4: Run tests to verify all sorter tests pass**

Run: `pytest tests/test_detector_sorter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/manga2kindle/cv/sorter.py tests/test_detector_sorter.py
git commit -m "fix(cv): use recursive XY-cut sorting for stacked manga panels"
```

---

### Task 3: Backend Protocol, Domain Types & `ContourBackend`

**Files:**
- Modify: `src/manga2kindle/models.py`
- Create: `src/manga2kindle/cv/backends/base.py`
- Create: `src/manga2kindle/cv/backends/contour.py`
- Modify: `src/manga2kindle/cv/detector.py`
- Test: `tests/test_backends_contour.py`

**Interfaces:**
- Produces:
  - `DetectorKind(str, Enum)`: `AUTO = "auto"`, `CONTOUR = "contour"`, `ONNX_YOLO = "onnx"`, `MAGI = "magi"`
  - `DevicePreference(str, Enum)`: `AUTO = "auto"`, `CPU = "cpu"`, `CUDA = "cuda"`, `MPS = "mps"`
  - `DetectionResult(dataclass)`: `panels: list[PanelBox]`, `ordered: bool`, `backend: DetectorKind`, `scores: list[float] | None = None`
  - `DetectorConfig(BaseModel)`: `kind`, `model_path`, `model_id`, `device`, `score_threshold`, `iou_threshold`, `containment_threshold`, `padding`, `min_panel_area`, `allow_remote_code`
  - `PanelDetectorBackend(Protocol)`: `kind: DetectorKind`, `detect(image_path: Path) -> DetectionResult`, `detect_batch(image_paths: Sequence[Path]) -> list[DetectionResult]`
  - `ContourBackend` in `cv/backends/contour.py`

- [ ] **Step 1: Write failing tests for `ContourBackend` and `PanelDetector` shim**

Verify `ContourBackend().detect(page_path)` returns `DetectionResult(ordered=False, backend=DetectorKind.CONTOUR, ...)` and `PanelDetector` remains backward-compatible.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends_contour.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add domain types to `models.py`, create `base.py` & `contour.py`, and turn `detector.py` into a shim**

- [ ] **Step 4: Run all existing + new CV tests**

Run: `pytest tests/test_backends_contour.py tests/test_detector_sorter.py tests/test_models_platform.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/manga2kindle/models.py src/manga2kindle/cv/backends/base.py src/manga2kindle/cv/backends/contour.py src/manga2kindle/cv/detector.py tests/test_backends_contour.py
git commit -m "refactor(cv): add PanelDetectorBackend protocol and ContourBackend"
```

---

### Task 4: Layer 2 — `OnnxYoloBackend` & Model Hub (`models fetch`)

**Files:**
- Modify: `pyproject.toml` (add `[onnx]` optional extra: `onnxruntime>=1.17`, `platformdirs>=4.0`)
- Create: `src/manga2kindle/cv/models_hub.py` (cache dir resolution via `platformdirs`, model registry with pinned HF revisions, SHA-256 verification, license/attribution display, `fetch_model(name: str) -> Path`)
- Create: `src/manga2kindle/cv/backends/onnx_yolo.py`
- Test: `tests/test_onnx_yolo.py`

**Interfaces:**
- Consumes: `boxes.nms_and_prune`, `boxes.pad_and_clamp`, `DetectionResult`, `DetectorConfig`, `DetectorKind`
- Produces:
  - `OnnxYoloBackend(config: DetectorConfig)` implementing `PanelDetectorBackend`
  - `letterbox_image(image_bgr: np.ndarray, target_size: int) -> tuple[np.ndarray, float, tuple[int, int]]`
  - `decode_yolo_output(raw_output: np.ndarray, ...) -> tuple[list[Rect], list[float]]`
  - `fetch_model(model_key: str) -> Path`

- [ ] **Step 1: Write failing unit tests for letterbox round-trip and canned YOLO tensor decoding**

Test `letterbox_image` + `decode_yolo_output` on synthetic non-square page (`1264x1680`) and canned `(1, 8, N)` / `(1, N, 6)` raw tensor outputs with nested boxes verifying containment pruning and source-coordinate clamping (no model file required).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_onnx_yolo.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `models_hub.py` and `onnx_yolo.py`, add `[onnx]` extra in `pyproject.toml`**

Verify the exact ONNX I/O layout of `deepghs/manga109_yolo` (`model.onnx`) during implementation, pin its HF revision and SHA-256 in `models_hub.py`, filter to the `frame` (panel) class constant, and apply `nms_and_prune` + `pad_and_clamp`.

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/test_onnx_yolo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/manga2kindle/cv/models_hub.py src/manga2kindle/cv/backends/onnx_yolo.py tests/test_onnx_yolo.py
git commit -m "feat(cv): add Layer 2 OnnxYoloBackend and model cache hub"
```

---

### Task 5: Layer 3 — `MagiBackend` (`ragavsachdeva/magiv3`)

**Files:**
- Modify: `pyproject.toml` (add `[magi]` optional extra: `torch>=2.2`, `transformers>=4.40`, `einops>=0.7`, `timm>=0.9`, `platformdirs>=4.0`)
- Create: `src/manga2kindle/cv/backends/magi.py`
- Test: `tests/test_magi_backend.py`

**Interfaces:**
- Consumes: `DetectionResult`, `DetectorConfig`, `DetectorKind`, `boxes.nms_and_prune`, `boxes.pad_and_clamp`
- Produces: `MagiBackend(config: DetectorConfig)` returning `DetectionResult(ordered=True, backend=DetectorKind.MAGI, ...)`

- [ ] **Step 1: Write failing unit tests for `--allow-remote-code` gate and mocked Magi output parsing**

Test that `MagiBackend` raises a clear `RuntimeError` when `allow_remote_code=False`, and test parsing mocked `predict_detections_and_associations` outputs into ordered `PanelBox`es with containment pruning.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_magi_backend.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/manga2kindle/cv/backends/magi.py` and add `[magi]` extra to `pyproject.toml`**

Pin `ragavsachdeva/magiv3` revision in `models_hub.py`, gate `trust_remote_code=True` behind `config.allow_remote_code`, select `cuda` / `mps` / `cpu` per `config.device`, and preserve Magi's reading order (`ordered=True`).

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/test_magi_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/manga2kindle/cv/backends/magi.py src/manga2kindle/cv/models_hub.py tests/test_magi_backend.py
git commit -m "feat(cv): add Layer 3 MagiBackend for GPU SOTA detection"
```

---

### Task 6: Backend Registry, `AUTO` Chain, Pipeline & `doctor`

**Files:**
- Create: `src/manga2kindle/cv/backends/__init__.py`
- Modify: `src/manga2kindle/pipeline.py`
- Modify: `src/manga2kindle/cli.py`
- Test: `tests/test_backend_registry.py`

**Interfaces:**
- Consumes: `ContourBackend`, `OnnxYoloBackend`, `MagiBackend`, `PanelSorter`
- Produces: `BackendRegistry`, `resolve_auto_chain(config: DetectorConfig) -> list[PanelDetectorBackend]`, `detect_with_fallback(image_path: Path, ...) -> DetectionResult`

- [ ] **Step 1: Write failing tests for `AUTO` fallback chain (`MAGI → ONNX_YOLO → CONTOUR`) and zero-panel fallback**

Use stub backends to verify:
1. `AUTO` tries `MAGI` → `ONNX_YOLO` → `CONTOUR` and logs/records warnings on load failure or 0-panel return.
2. When `DetectionResult.ordered` is `True`, `PanelSorter` is skipped; when `False`, `PanelSorter` runs.
3. Explicit `--detector onnx` raises on missing weights instead of falling back.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `cv/backends/__init__.py`, wire into `pipeline.py`, and update `doctor` + `models fetch` in `cli.py`**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_backend_registry.py tests/test_cli_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/manga2kindle/cv/backends/__init__.py src/manga2kindle/pipeline.py src/manga2kindle/cli.py tests/test_backend_registry.py
git commit -m "feat(pipeline): wire 3-layer backend registry, AUTO fallback, and models fetch"
```

---

### Task 7: Pre-Conversion Preview (`preview-panels --compare` & `convert --preview-only`)

**Files:**
- Modify: `src/manga2kindle/cv/preview.py`
- Modify: `src/manga2kindle/pipeline.py`
- Modify: `src/manga2kindle/cli.py`
- Test: `tests/test_preview.py`

**Interfaces:**
- Consumes: `BackendRegistry`, `PanelPreviewRenderer`
- Produces:
  - `PanelPreviewRenderer.render_comparison(image_path: Path, results: dict[DetectorKind, list[PanelBox]], output_path: Path) -> Path`
  - CLI flags: `preview-panels --detector ... --compare contour,onnx,magi` and `convert --preview-only`

- [ ] **Step 1: Write failing tests for comparison overlay rendering and `convert --preview-only`**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preview.py -v`
Expected: FAIL

- [ ] **Step 3: Implement side-by-side / color-coded comparison rendering in `preview.py` and wire `--compare` + `--preview-only` in `pipeline.py` and `cli.py`**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preview.py tests/test_cli_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/manga2kindle/cv/preview.py src/manga2kindle/pipeline.py src/manga2kindle/cli.py tests/test_preview.py
git commit -m "feat(cli): add preview-panels --compare and convert --preview-only"
```

---

### Task 8: PopManga / CoMix Benchmark Harness (`cv/benchmark.py`)

**Files:**
- Modify: `pyproject.toml` (add `[benchmark]` optional extra: `datasets>=2.18`)
- Modify: `.gitignore` (ensure `benchmarks/`, `data/`, `*.onnx`, `*.safetensors`, `*.pt` are ignored)
- Create: `src/manga2kindle/cv/benchmark.py`
- Modify: `src/manga2kindle/cli.py` (add `manga2kindle benchmark` command)
- Modify: `README.md` (document 3-layer backends, `models fetch`, `preview-panels --compare`, `benchmark`, and model/dataset licenses)
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces:
  - `BenchmarkMetrics(dataclass)`: `backend`, `precision_50`, `recall_50`, `precision_75`, `recall_75`, `mean_iou`, `nested_duplicate_rate`, `exact_count_match`, `pairwise_order_acc`, `kendall_tau`, `ms_per_page`, `peak_ram_mb`, `peak_vram_mb`
  - `evaluate_predictions(...) -> BenchmarkMetrics`
  - `load_popmanga_split(...)` (reads from local HF cache, never commits data)

- [ ] **Step 1: Write failing unit tests for all benchmark metric functions on synthetic boxes**

Test `evaluate_predictions` with known predicted vs ground-truth boxes and reading orders, verifying `precision_50`, `recall_50`, `nested_duplicate_rate`, `pairwise_order_acc`, and `kendall_tau`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `cv/benchmark.py`, `manga2kindle benchmark` CLI command, `.gitignore` updates, and `README.md` docs**

- [ ] **Step 4: Run full test suite**

Run: `pytest -v`
Expected: PASS (all unit & integration tests green; `@pytest.mark.model` tests skipped unless weights/cache present)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore README.md src/manga2kindle/cv/benchmark.py src/manga2kindle/cli.py tests/test_benchmark.py
git commit -m "feat(benchmark): add PopManga/CoMix evaluation harness and CLI command"
```
