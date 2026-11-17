# Pluggable 3-Layer Panel Detection Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a 3-layer pluggable panel detection architecture (`CONTOUR` → `ONNX_YOLO` → `MAGI`), fix nested-panel suppression and stacked-panel reading order, add pre-conversion comparison previews, and build a PopManga/CoMix benchmark harness.

**Architecture:** Extract shared bounding-box geometry (`boxes.py`) with containment pruning, upgrade `PanelSorter` (`sorter.py`) to recursive XY-cut gutter partitioning, introduce a `PanelDetectorBackend` protocol with `ContourBackend`, `OnnxYoloBackend`, and `MagiBackend` behind an `AUTO` fallback chain, and expose `--compare`, `--preview-only`, `models fetch`, and `benchmark` in the Typer CLI.

**Tech Stack:** Python 3.11+, OpenCV (`opencv-python-headless`), Pydantic v2, Typer + Rich, `onnxruntime` + `platformdirs` (`[onnx]` extra), `torch` + `transformers` + `einops` + `timm` (`[magi]` extra), `datasets` + `psutil` (`[benchmark]` extra), `pytest`.

**Spec:** [2026-11-15-panel-detection-backends-design.md](../specs/2026-11-15-panel-detection-backends-design.md)

## Global Constraints

- Target Python 3.11+ on both macOS and Windows using `pathlib.Path` exclusively for filesystem paths.
- Keep `torch` and `ultralytics` out of core dependencies; learned backends live in optional extras (`[onnx]`, `[magi]`, `[benchmark]`).
- Use structured `Pydantic` models or `dataclass`es and `Enum`s / named constants—no untyped dicts for internal state and no magic strings.
- Never commit dataset files, model weights, sample manga chapters, or machine-specific local paths to the repository.
- Honor all model and dataset licenses (display non-commercial/attribution notices at `models fetch` and `benchmark` time and in `README.md`).
- Preserve all existing public/test contracts (`PanelDetector`, `compute_iou`, `PanelPreviewRenderer.render`, `PlatformDiscovery.run_doctor`, `ConversionConfig` defaults/validators).
- Follow Conventional Commits without `Co-Authored-By` trailers.

## Review Focus

1. **Nested small-inside-large panel:** A small panel completely inside a larger panel (`IoU < 0.4` but `containment > 0.85`) must be pruned across all backends—even when the inner box has higher detection confidence than the outer panel (`Task 1`).
2. **Stacked panels beside a tall panel:** Two panels stacked vertically next to a tall panel that overlaps both vertically—including when `padding=0.02` causes minor gutter overlap—must sort top-to-bottom in reading order (`Task 2`).
3. **Letterbox coordinate round-trip on non-square pages:** `OnnxYoloBackend` detections on tall/wide manga pages must map back to exact source-page pixel bounds without clipping or offset (`Task 4`).
4. **Magi reading-order preservation:** Containment pruning in `MagiBackend` (`preserve_order=True`) must never reorder Magi's native reading sequence (`Task 5`).
5. **Missing `--allow-remote-code` for `MAGI`:** Explicit `--detector magi` without `--allow-remote-code` must fail with an actionable error; `AUTO` must skip `MAGI` with a warning only when `[magi]` + GPU + cached weights are present (`Task 5`, `Task 6`).
6. **Zero-panel splash page in `AUTO`:** When a learned backend returns 0 panels on a splash page, `AUTO` walks remaining backends before falling back to full-page `[PanelBox(0, 0, page_w, page_h, ordinal=1)]`, while raw `backend.detect()` returns `panels=[]` for `benchmark` accuracy (`Task 6`, `Task 8`).
7. **`convert --preview-only` without Calibre:** `--preview-only` must run cleanly without `calibre-debug` installed on both single and batch inputs (`Task 7`).

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
  - `nms_and_prune_indices(boxes: Sequence[Rect], iou_threshold: float = 0.4, containment_threshold: float = 0.85, scores: Sequence[float] | None = None, preserve_order: bool = False) -> list[int]`
  - `nms_and_prune(boxes: Sequence[Rect], iou_threshold: float = 0.4, containment_threshold: float = 0.85, scores: Sequence[float] | None = None, preserve_order: bool = False) -> list[Rect]`
  - `pad_and_clamp(rect: Rect, page_w: int, page_h: int, padding: float = 0.02, bleed_margin: int = 2) -> PanelBox`

- [ ] **Step 1: Write failing tests for containment suppression, score alignment, order preservation, and geometry helpers**

```python
from manga2kindle.cv.boxes import (
    compute_containment,
    compute_iou,
    nms_and_prune,
    nms_and_prune_indices,
    pad_and_clamp,
)


def test_compute_containment_nested_box_is_one():
    outer = (0, 0, 400, 600)
    inner = (50, 50, 100, 100)
    assert compute_iou(outer, inner) < 0.1
    assert compute_containment(outer, inner) == 1.0
    assert compute_containment(outer, (0, 0, 0, 10)) == 0.0


def test_nms_and_prune_removes_nested_small_panel_even_with_higher_score():
    outer = (10, 10, 500, 700)
    inner = (60, 80, 120, 140)
    kept = nms_and_prune(
        [inner, outer],
        scores=[0.95, 0.78],
        iou_threshold=0.4,
        containment_threshold=0.85,
    )
    assert kept == [outer]


def test_nms_and_prune_indices_preserve_order():
    small_first = (20, 20, 100, 100)
    outer_second = (150, 20, 400, 500)
    inner_third = (200, 80, 80, 80)
    indices = nms_and_prune_indices(
        [small_first, outer_second, inner_third],
        preserve_order=True,
    )
    assert indices == [0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_boxes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manga2kindle.cv.boxes'`

- [ ] **Step 3: Implement `src/manga2kindle/cv/boxes.py` and update `detector.py` to delegate to it**

Extract `compute_iou` and `pad_and_clamp` (guarding `page_w <= 0 or page_h <= 0`) from `detector.py`. Implement `compute_containment(box_a, box_b) -> float` as `inter_area / min(area_a, area_b)` (returning `0.0` when `min_area <= 0`). Implement two-stage `nms_and_prune_indices`:
1. Containment pass sorted by `area` descending (ties by `score` descending): drop any smaller candidate with `compute_containment(cand, kept) > containment_threshold` and `area(cand) < area(kept)`.
2. IoU NMS pass sorted by `(score, area)` descending when `scores` are given else `area` descending: drop any candidate with `compute_iou(cand, kept) > iou_threshold`.
3. If `preserve_order=True`, sort surviving indices ascending to match input order.
Keep `PanelDetector.non_max_suppression` and `PanelDetector._pad_and_clamp` in `detector.py` delegating to `boxes.py` so existing `tests/test_detector_sorter.py` assertions continue passing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_boxes.py tests/test_detector_sorter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/manga2kindle/cv/boxes.py src/manga2kindle/cv/detector.py tests/test_boxes.py
git commit -m "fix(cv): extract shared boxes module with containment suppression"
```

---

### Task 2: Two-Tier XY-Cut Stacked-Panel Reading-Order Fix (`cv/sorter.py`)

**Files:**
- Modify: `src/manga2kindle/cv/sorter.py`
- Test: `tests/test_detector_sorter.py`

**Interfaces:**
- Consumes: `PanelBox`, `ReadingDirection`
- Produces: `PanelSorter.sort(boxes: list[PanelBox], *, page_width: int | None = None, is_spread: bool = False) -> list[PanelBox]`

- [ ] **Step 1: Write failing tests for stacked panels beside a tall panel (RTL, LTR, and padding-overlap cases)**

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


def test_stacked_panels_next_to_tall_panel_with_padding_overlap_ltr():
    # Simulates 2% pad_and_clamp expansion causing ~10px gutter overlap between columns/rows
    tall_left = PanelBox(x=20, y=20, w=490, h=700)
    top_right = PanelBox(x=500, y=25, w=460, h=340)
    bottom_right = PanelBox(x=502, y=358, w=458, h=360)

    ordered = PanelSorter(direction=ReadingDirection.LTR).sort([bottom_right, top_right, tall_left])
    assert [(b.x, b.y, b.ordinal) for b in ordered] == [
        (20, 20, 1),
        (500, 25, 2),
        (502, 358, 3),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector_sorter.py -k stacked -v`
Expected: FAIL (currently `bottom_left` with `x=40` sorts before `top_left` with `x=20` in RTL row clustering)

- [ ] **Step 3: Implement two-tier recursive XY-cut gutter partitioning in `PanelSorter._sort_group`**

Implement `_xy_cut_sort(boxes: list[PanelBox]) -> list[PanelBox]` (keeping `_sort_spread_halves` at the top level):
1. Attempt **strict** horizontal gutter split (sweep `y` intervals to find non-overlapping horizontal bands sorted top-to-bottom); if `>= 2` bands, recurse on each band.
2. Otherwise attempt **strict** vertical gutter split (sweep `x` intervals to find non-overlapping vertical columns ordered RTL `x` descending or LTR `x` ascending); if `>= 2` columns, recurse on each column.
3. If neither strict cut splits `boxes`, retry horizontal then vertical split using **inset** intervals (inset each box edge by `min(24, int(0.05 * dim))`, guarded so each interval retains `>= 1px`), resolving minor `pad_and_clamp` gutter overlap without cutting across real panel cores.
4. If still unsplittable (genuinely overlapping/diagonal panels), fall back to `_sort_by_rows` using `row_tolerance_px`.

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
- Modify: `src/manga2kindle/cv/__init__.py`
- Test: `tests/test_backends_contour.py`

**Interfaces:**
- Produces:
  - `DetectorKind(str, Enum)`: `AUTO = "auto"`, `CONTOUR = "contour"`, `ONNX_YOLO = "onnx"`, `MAGI = "magi"`
  - `DevicePreference(str, Enum)`: `AUTO = "auto"`, `CPU = "cpu"`, `CUDA = "cuda"`, `MPS = "mps"`
  - `DetectionResult(dataclass)`: `panels: list[PanelBox]`, `ordered: bool`, `backend: DetectorKind`, `page_width: int = 0`, `page_height: int = 0`, `scores: list[float] | None = None`
  - `DetectorConfig(BaseModel)`: `kind`, `model_path`, `model_id`, `device`, `score_threshold`, `iou_threshold`, `containment_threshold`, `padding`, `min_panel_area`, `allow_remote_code`
  - `ConversionConfig.detector: DetectorConfig` (synchronized with top-level `padding` / `min_panel_area` via `model_validator` so legacy callers passing `ConversionConfig(padding=...)` propagate cleanly)
  - `PanelDetectorBackend(Protocol)`: `kind: DetectorKind`, `detect(image_path: Path) -> DetectionResult`, `detect_batch(image_paths: Sequence[Path]) -> list[DetectionResult]`
  - `ContourBackend` in `cv/backends/contour.py` (returns `panels=[]` when zero contours survive; `detect_batch` supports `workers`)
  - `PanelDetector` wrapper in `cv/detector.py` delegating to `ContourBackend` and returning `list[PanelBox]` with full-page fallback

- [ ] **Step 1: Write failing tests for `ContourBackend`, `ConversionConfig.detector` sync, and `PanelDetector` wrapper**

Verify:
- `ContourBackend().detect(page_path)` returns `DetectionResult(ordered=False, backend=DetectorKind.CONTOUR, page_width=..., page_height=..., ...)` and returns `panels=[]` on a blank page.
- `PanelDetector` wrapper preserves `list[PanelBox]` return type and full-page fallback on blank pages.
- `ConversionConfig(padding=0.05, min_panel_area=0.04, ...)` propagates into `config.detector` when `detector` uses defaults.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends_contour.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add domain types to `models.py`, create `base.py` & `contour.py`, update `detector.py` wrapper & `cv/__init__.py`**

- [ ] **Step 4: Run all existing + new CV tests**

Run: `pytest tests/test_backends_contour.py tests/test_detector_sorter.py tests/test_models_platform.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/manga2kindle/models.py src/manga2kindle/cv/backends/base.py src/manga2kindle/cv/backends/contour.py src/manga2kindle/cv/detector.py src/manga2kindle/cv/__init__.py tests/test_backends_contour.py
git commit -m "refactor(cv): add PanelDetectorBackend protocol and ContourBackend"
```

---

### Task 4: Layer 2 — `OnnxYoloBackend` & Model Hub (`models fetch`)

**Files:**
- Modify: `pyproject.toml` (add `[onnx]` optional extra: `onnxruntime>=1.17`, `platformdirs>=4.0`)
- Create: `src/manga2kindle/cv/models_hub.py` (OS cache dir resolution via `platformdirs`, model registry with pinned HF revisions, SHA-256 verification, license/attribution display, `urllib.request` download for `.onnx`, `fetch_model(name: str) -> Path`, `resolve_model_spec(raw: str) -> tuple[Path | None, str | None]`)
- Create: `src/manga2kindle/cv/backends/onnx_yolo.py`
- Test: `tests/test_onnx_yolo.py`

**Interfaces:**
- Consumes: `boxes.nms_and_prune_indices`, `boxes.pad_and_clamp`, `DetectionResult`, `DetectorConfig`, `DetectorKind`, `DevicePreference`
- Produces:
  - `OnnxYoloBackend(config: DetectorConfig)` implementing `PanelDetectorBackend`
  - `letterbox_image(image_bgr: np.ndarray, target_size: int) -> tuple[np.ndarray, float, tuple[int, int]]`
  - `decode_yolo_output(raw_output: np.ndarray, ...) -> tuple[list[Rect], list[float]]`
  - `select_ort_providers(device: DevicePreference, available_providers: Sequence[str]) -> list[str]`
  - `fetch_model(model_key: str) -> Path`, `is_model_cached(model_key: str) -> bool`, `list_registered_models() -> list[ModelSpec]`

- [ ] **Step 1: Write failing unit tests for letterbox round-trip, ORT provider selection, canned YOLO tensor decoding, and `models_hub` SHA-256 verification**

Test:
- `letterbox_image` + `decode_yolo_output` on synthetic non-square page (`1264x1680`) and canned `(1, 8, N)` / `(1, N, 6)` raw tensor outputs with nested boxes verifying containment pruning, aligned `DetectionResult.scores`, non-clipping of small valid panels (`< 3%` area), and source-coordinate clamping (no model file required).
- `select_ort_providers` mapping `AUTO` (`CUDAExecutionProvider` -> `CoreMLExecutionProvider` -> `DmlExecutionProvider` -> `CPUExecutionProvider`), `MPS` -> `CoreMLExecutionProvider`, and raising `RuntimeError` when an explicit provider is unavailable.
- `resolve_model_spec` distinguishing filesystem `.onnx`/path strings vs HF repo IDs/aliases, and `models_hub` SHA-256 mismatch error handling.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_onnx_yolo.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `models_hub.py` and `onnx_yolo.py`, add `[onnx]` extra in `pyproject.toml`**

Verify the exact ONNX I/O layout of `deepghs/manga109_yolo` (`model.onnx`) during implementation, pin its HF revision and SHA-256 in `models_hub.py`, download via stdlib `urllib.request` (no `huggingface_hub` dependency in `[onnx]`), filter to the `frame` (panel) class constant and `score_threshold`, apply `nms_and_prune_indices` + `pad_and_clamp`, guard `session.run()` with a lock, and load strictly from local cache during inference.

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
- Modify: `pyproject.toml` (add `[magi]` optional extra: `torch>=2.2`, `transformers>=4.40`, `einops>=0.7`, `timm>=0.9`, `platformdirs>=4.0`, plus `scipy`/`torchvision` if imported by pinned `magiv3` commit)
- Modify: `src/manga2kindle/cv/models_hub.py` (add `ragavsachdeva/magiv3` + optional `ragavsachdeva/magiv2` pinned revisions and `snapshot_download` support)
- Create: `src/manga2kindle/cv/backends/magi.py`
- Test: `tests/test_magi_backend.py`

**Interfaces:**
- Consumes: `DetectionResult`, `DetectorConfig`, `DetectorKind`, `DevicePreference`, `boxes.nms_and_prune_indices`, `boxes.pad_and_clamp`
- Produces: `MagiBackend(config: DetectorConfig)` returning `DetectionResult(ordered=True, backend=DetectorKind.MAGI, ...)`

- [ ] **Step 1: Write failing unit tests for `--allow-remote-code` gate and mocked Magi output parsing**

Test that:
- `MagiBackend` raises an actionable `RuntimeError` when `allow_remote_code=False`.
- Mocked `predict_detections_and_associations` outputs (with BGR/grayscale input converted to RGB) parse into `PanelBox`es with `containment` pruning (`preserve_order=True`) preserving Magi's native reading order and 1-based `ordinal`s.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_magi_backend.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/manga2kindle/cv/backends/magi.py`, update `models_hub.py`, and add `[magi]` extra to `pyproject.toml`**

Verify `ragavsachdeva/magiv3` exact return schema and remote-code imports against its pinned HF commit hash. Gate `trust_remote_code=True` behind `config.allow_remote_code`, enforce `local_files_only=True` at inference time, select `cuda` / `mps` / `cpu` per `config.device`, run under `torch.inference_mode()` protected by a thread lock, and preserve Magi's reading order (`ordered=True`, `preserve_order=True`, 1-based `ordinal`).

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/test_magi_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/manga2kindle/cv/backends/magi.py src/manga2kindle/cv/models_hub.py tests/test_magi_backend.py
git commit -m "feat(cv): add Layer 3 MagiBackend for GPU SOTA detection"
```

---

### Task 6: Backend Registry, `AUTO` Chain, Pipeline, `models` & `doctor`

**Files:**
- Create: `src/manga2kindle/cv/backends/__init__.py`
- Modify: `src/manga2kindle/pipeline.py`
- Modify: `src/manga2kindle/platform.py`
- Modify: `src/manga2kindle/cli.py`
- Test: `tests/test_backend_registry.py`

**Interfaces:**
- Consumes: `ContourBackend`, `OnnxYoloBackend`, `MagiBackend`, `PanelSorter`, `models_hub`
- Produces:
  - `BackendRegistry`
  - `resolve_auto_chain(config: DetectorConfig) -> list[PanelDetectorBackend]`
  - `detect_with_fallback(image_path: Path, ...) -> DetectionResult` (returns full-page `[PanelBox(0, 0, page_w, page_h, ordinal=1)]` when all candidate backends return `panels=[]`)
  - CLI `models` subcommand group (`models list`, `models fetch <name>`) and `--detector`, `--detector-model`, `--device`, `--score-threshold`, `--allow-remote-code` flags on `convert` and `preview-panels`

- [ ] **Step 1: Write failing tests for `AUTO` fallback chain (`MAGI → ONNX_YOLO → CONTOUR`), model reuse, and zero-panel fallback**

Use stub backends to verify:
1. `AUTO` tries `MAGI` → `ONNX_YOLO` → `CONTOUR`, logs active backend, disables a load-failed backend once per session, and falls through per page on 0-panel return before applying full-page `[PanelBox(0, 0, page_w, page_h, ordinal=1)]`.
2. `AUTO` skips `MAGI` with a single warning when `[magi]` + GPU + cached weights exist but `allow_remote_code=False`.
3. When `DetectionResult.ordered` is `True`, `PanelSorter` is skipped and 1-based ordinals are preserved; when `False`, `PanelSorter` runs.
4. Explicit `--detector onnx` raises on missing weights instead of falling back.
5. `ConversionPipeline` reuses a single resolved backend/session across pages and across batch workers (never instantiating per-page).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `cv/backends/__init__.py`, wire into `pipeline.py`, update `platform.py` + `doctor` + `models` + `convert` flags in `cli.py`**

Ensure `doctor` reports `onnxruntime` (and active providers), `torch` + CUDA/MPS, cached models, and resolved `AUTO` layer as optional diagnostics so existing `PlatformDiscovery.run_doctor` monkeypatches and `test_doctor_exits_zero_when_only_optional_tools_missing` continue to pass.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_backend_registry.py tests/test_cli_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/manga2kindle/cv/backends/__init__.py src/manga2kindle/pipeline.py src/manga2kindle/platform.py src/manga2kindle/cli.py tests/test_backend_registry.py
git commit -m "feat(pipeline): wire 3-layer backend registry, AUTO fallback, and models fetch"
```

---

### Task 7: Pre-Conversion Preview (`preview-panels --compare` & `convert --preview-only`)

**Files:**
- Modify: `src/manga2kindle/cv/preview.py`
- Modify: `src/manga2kindle/pipeline.py`
- Modify: `src/manga2kindle/cli.py`
- Test: `tests/test_preview.py`
- Modify: `tests/test_cli_e2e.py` (add CLI coverage for `--compare` and `--preview-only`)

**Interfaces:**
- Consumes: `BackendRegistry`, `PanelPreviewRenderer`
- Produces:
  - `PanelPreviewRenderer.render_comparison(image_path: Path, results: dict[DetectorKind, list[PanelBox]], output_path: Path) -> Path` (renders side-by-side tiled comparison PNGs with per-backend distinct box colors, `BACKEND (N panels)` header banner, `[ordinal]` badges, and reading-order arrows, sharing the core overlay helper with `render()`)
  - `ConversionPipeline.preview_panels(..., detector_config: DetectorConfig | None = None, compare: list[DetectorKind] | None = None)`
  - CLI flags: `preview-panels --detector ... --compare contour,onnx,magi` and `convert --preview-only`

- [ ] **Step 1: Write failing tests for side-by-side comparison rendering, `--compare` validation, and `convert --preview-only`**

Test:
- `render_comparison` writes a horizontally tiled image (`width = sum(tile_widths)`), draws header banners with backend names & panel counts, and preserves `render()` unchanged.
- `preview-panels --compare contour,onnx` with stub backends succeeds; `--compare auto` fails fast; `--compare magi` without `--allow-remote-code` fails fast.
- `convert --preview-only` on single and batch inputs succeeds even when `calibre-debug` is absent (`find_calibre` returns `None`), emits PNG overlays to the preview directory, and writes no `.kfx` file.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preview.py tests/test_cli_e2e.py -k "comparison or preview_only" -v`
Expected: FAIL

- [ ] **Step 3: Implement side-by-side comparison rendering in `preview.py` and wire `--compare` + `--preview-only` in `pipeline.py` and `cli.py`**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_preview.py tests/test_cli_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/manga2kindle/cv/preview.py src/manga2kindle/pipeline.py src/manga2kindle/cli.py tests/test_preview.py tests/test_cli_e2e.py
git commit -m "feat(cli): add preview-panels --compare and convert --preview-only"
```

---

### Task 8: PopManga / CoMix Benchmark Harness (`cv/benchmark.py`)

**Files:**
- Modify: `pyproject.toml` (add `[benchmark]` optional extra: `datasets>=2.18`, `psutil>=5.9`)
- Modify: `.gitignore` (ensure `benchmarks/`, `data/`, `*_preview/`, `*.onnx`, `*.safetensors`, `*.pt`, `*.bin`, `*.pth`, `*.parquet`, `*.arrow` are ignored)
- Create: `src/manga2kindle/cv/benchmark.py`
- Modify: `src/manga2kindle/cli.py` (add `manga2kindle benchmark` command)
- Modify: `README.md` (document 3-layer backends, `models fetch` / `models list`, `preview-panels --compare`, `convert --preview-only`, `benchmark`, and model/dataset licenses)
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces:
  - `BenchmarkMetrics(dataclass)`: `backend`, `precision_50`, `recall_50`, `precision_75`, `recall_75`, `mean_iou`, `nested_duplicate_rate`, `exact_count_match`, `pairwise_order_acc`, `kendall_tau`, `ms_per_page`, `peak_ram_mb`, `peak_vram_mb`
  - `evaluate_predictions(...) -> BenchmarkMetrics` (uses 1-to-1 greedy IoU matching at `IoU >= 0.50` before scoring `pairwise_order_acc` and `kendall_tau` on matched pairs with `n >= 2`)
  - `load_popmanga_split(...)`, `load_comix_split(...)`, `load_local_dataset(pages_dir: Path, truth_file: Path)` (read from local HF cache or user dir, never commit data)

- [ ] **Step 1: Write failing unit tests for all benchmark metric functions and local JSON/COCO loaders**

Test `evaluate_predictions` with known predicted vs ground-truth boxes and reading orders (including mismatched panel counts and nested duplicates), verifying `precision_50`, `recall_50`, `precision_75`, `recall_75`, `mean_iou`, `nested_duplicate_rate`, `exact_count_match`, `pairwise_order_acc`, and `kendall_tau`, plus `load_local_dataset`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `cv/benchmark.py`, `manga2kindle benchmark` CLI command, `.gitignore` updates, and `README.md` docs**

Ensure `benchmark`:
- Prints model/dataset license & attribution notices before loading.
- Evaluates raw page images via `backend.detect()` (no full-page fallback or KCC/Pillow resizing), running `PanelSorter(direction=...)` only when `not result.ordered`.
- Defaults `--direction` to `rtl` for `popmanga` and `ltr` for `comix`.
- Uses deterministic `--limit` slicing across detectors, excludes warmup from `ms_per_page`, calls `torch.cuda.synchronize()` for CUDA timing, and tracks `psutil` RAM + `torch.cuda.max_memory_allocated()` VRAM cleanly on both Windows and macOS.

- [ ] **Step 4: Run full test suite**

Run: `pytest -v`
Expected: PASS (all unit & integration tests green; `@pytest.mark.model` tests skipped unless weights/cache present)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore README.md src/manga2kindle/cv/benchmark.py src/manga2kindle/cli.py tests/test_benchmark.py
git commit -m "feat(benchmark): add PopManga/CoMix evaluation harness and CLI command"
```
