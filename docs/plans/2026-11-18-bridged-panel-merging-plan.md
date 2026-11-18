# Bridged Panel Merging Post-Processor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a geometry-based panel-merging post-processor that merges adjacent panels bridged by a shared dialogue bubble or substantial overlap, while enforcing 7" Kindle readability guardrails and preserving reading order and detection scores.

**Architecture:** Define merge default constants in `src/manga2kindle/models.py` (avoiding circular imports with `cv/`), implement a shared post-processing pipeline (`nms_and_prune_indices` -> optional `merge_bridged_panel_clusters` -> `pad_and_clamp`) in `src/manga2kindle/cv/boxes.py`, expose zero-cost text boxes from `MagiBackend` (`"texts"` + `"is_essential_text"`) and `OnnxYoloBackend` (`MANGA109_TEXT_CLASS_ID = 3`), wire `ContourBackend` and `DetectorConfig`, and expose `--merge-panels/--no-merge-panels` on `convert` and `preview-panels` while keeping `benchmark` unmerged.

**Tech Stack:** Python 3.11+, Pydantic v2, NumPy, OpenCV, Typer, Pytest.

**Spec:** [docs/specs/2026-11-18-bridged-panel-merging-design.md](../specs/2026-11-18-bridged-panel-merging-design.md)

## Global Constraints

- Target Python 3.11+ on Windows and macOS; use `pathlib.Path` exclusively for filesystem paths.
- Use `Enum`s / named module constants (no magic strings or unshared magic numbers).
- Use typed objects (`@dataclass(frozen=True)`, Pydantic `BaseModel`) over untyped dicts/tuples for structured data.
- Preserve backward compatibility for `ContourBackend(...)` kwargs/positional callers, `PanelDetector`, `ModelSpec`, and `decode_yolo_output(...) -> tuple[list[Rect], list[float]]`.
- Keep `DetectionResult` (`src/manga2kindle/cv/backends/base.py`) and `PanelSorter` (`src/manga2kindle/cv/sorter.py`) unchanged.
- `manga2kindle benchmark` must force `merge_bridged_panels=False` so PopManga/CoMix metrics are unaffected.
- Work on a feature branch branched off `develop` (`feat/bridged-panel-merging`), using Conventional Commits without `Co-Authored-By:` trailers.

## Review Focus

- **Tiny text-box / stray punctuation bridge:** A tiny text detection (`area / page_area < 0.0005`) straddling a 2px panel seam must NOT bridge two large panels.
- **L-shaped / diagonal union dead-space & engulfment:** Merging two diagonal panels whose axis-aligned union is mostly empty (`sum_member_areas / union_area < 0.55`) or swallows an unrelated 3rd panel above `containment_threshold` must be rejected.
- **Transitive full-page collapse:** Chain-merging 3+ panels across a page must stop before `union_area / page_area > 0.65` (`max_merged_area_ratio`), prioritizing strongest bridges first deterministically.
- **Magi `"texts"` / `"is_essential_text"` anomalies:** `texts=None`, NumPy array values, mismatched `is_essential_text` lengths, or `is_essential_text=False` (decorative SFX spanning panels) must not crash or trigger unwanted merges.
- **Custom 1-class / 3-class ONNX model:** Passing a custom `.onnx` via `--detector-model` (`text_class_id=None`) or a tensor with `< 4` classes (`4 + text_class_id >= arr.shape[0]`, e.g. `(1, 7, N)`) must skip text extraction without raising.

---

### Task 1: Core Merge Geometry, Guardrails, Shared Finalizer & Config (`models.py`, `boxes.py`)

**Files:**
- Modify: `src/manga2kindle/models.py:122-136`
- Modify: `src/manga2kindle/cv/boxes.py:1-173`
- Test: `tests/test_boxes.py`

- [ ] **Step 1: Write failing unit tests in `tests/test_boxes.py`**
  Cover:
  - `merge_bridged_panel_clusters` merges two side-by-side panels bridged by a text box (`>= 0.15` overlap in each) into one `MergedCluster` with sorted `member_indices`.
  - Text box wholly inside one panel, below `text_bridge_overlap`, or below `min_text_area_ratio` (tiny noise) does NOT merge.
  - Degenerate panel `boxes` and `text_boxes` (`w <= 0` or `h <= 0`) are ignored safely (degenerate panel boxes are not emitted as `1x1` panels).
  - Text box bridging 3+ panels links all pairs and merges strongest-first, respecting `max_merged_area_ratio`.
  - Direct panel overlap (`compute_containment >= panel_overlap_threshold`) merges; below threshold does not.
  - `min_fill_ratio` rejects L-shaped/diagonal unions with excessive dead space; engulfment guardrail rejects merges whose union contains an unrelated third panel (`>= containment_threshold`).
  - Ordering preserves `min(member_indices)` and raises `ValueError` on non-positive `page_width`/`page_height` (even when `boxes` is empty).
  - `merge_bridged_panels` wrapper returns `list[Rect]`.
  - `finalize_panel_boxes` runs NMS -> optional merge -> `pad_and_clamp`, assigns `ordinal = 1..N` **only** when `preserve_order=True` (leaving `ordinal=0` when `preserve_order=False`), slices `kept_scores` post-NMS before aggregating `max(kept_scores[m] for m in c.member_indices)`, and returns `([], [])` when `boxes=[]` with `scores=[]` vs `([], None)` when `scores=None`.
  - `DetectorConfig` validates `merge_bridged_panels`, `text_bridge_overlap`, `panel_overlap_threshold`, `max_merged_area_ratio`, `min_fill_ratio`.

- [ ] **Step 2: Run pytest on `tests/test_boxes.py` to confirm failure**
  Run: `pytest tests/test_boxes.py -v`

- [ ] **Step 3: Define merge constants and `DetectorConfig` fields in `src/manga2kindle/models.py`**
  - Define module constants in `models.py` (which has no internal `cv` imports, preventing circular imports):
    - `DEFAULT_TEXT_BRIDGE_OVERLAP = 0.15`
    - `DEFAULT_PANEL_OVERLAP_THRESHOLD = 0.25`
    - `DEFAULT_MAX_MERGED_AREA_RATIO = 0.65`
    - `DEFAULT_MIN_FILL_RATIO = 0.55`
    - `DEFAULT_MIN_TEXT_AREA_RATIO = 0.0005`
  - Add to `DetectorConfig`:
    - `merge_bridged_panels: bool = True`
    - `text_bridge_overlap: float = Field(default=DEFAULT_TEXT_BRIDGE_OVERLAP, gt=0.0, le=1.0)`
    - `panel_overlap_threshold: float = Field(default=DEFAULT_PANEL_OVERLAP_THRESHOLD, gt=0.0, le=1.0)`
    - `max_merged_area_ratio: float = Field(default=DEFAULT_MAX_MERGED_AREA_RATIO, gt=0.0, le=1.0)`
    - `min_fill_ratio: float = Field(default=DEFAULT_MIN_FILL_RATIO, gt=0.0, le=1.0)`

- [ ] **Step 4: Implement `MergedCluster`, `merge_bridged_panel_clusters`, `merge_bridged_panels`, and `finalize_panel_boxes` in `src/manga2kindle/cv/boxes.py`**
  - Import and re-export `DEFAULT_TEXT_BRIDGE_OVERLAP`, `DEFAULT_PANEL_OVERLAP_THRESHOLD`, `DEFAULT_MAX_MERGED_AREA_RATIO`, `DEFAULT_MIN_FILL_RATIO`, `DEFAULT_MIN_TEXT_AREA_RATIO` from `manga2kindle.models`.
  - Add `@dataclass(frozen=True) class MergedCluster` with `rect: Rect` and `member_indices: tuple[int, ...]`.
  - Implement `merge_bridged_panel_clusters(...)`:
    - Validate `page_width > 0 and page_height > 0` first; filter `valid_indices = [i for i, b in enumerate(boxes) if b[2] > 0 and b[3] > 0]`; return `[]` if `not valid_indices`.
    - Ignore `text_boxes` with `w <= 0 or h <= 0` or `area(t) / page_area < min_text_area_ratio`.
    - Text-bridge all-pairs edges over `valid_indices`, weighted by `min(frac_a, frac_b)`.
    - Panel-overlap edges over `valid_indices` using `compute_containment(boxes[i], boxes[j]) >= panel_overlap_threshold`.
    - Deduplicate edges keeping `max(weight)` per `(i, j)` (`i < j`), sorted by `(-weight, i, j)`.
    - Guarded Union-Find tracking `sum_member_areas` (sum of individual `boxes[m]` areas, not union area) and checking `max_merged_area_ratio`, `min_fill_ratio`, and third-panel engulfment (`compute_containment(union_rect, boxes[k]) >= containment_threshold` for any `k` in `valid_indices` outside both clusters).
    - Emit `MergedCluster` sorted by `member_indices[0]`.
  - Implement `merge_bridged_panels(...) -> list[Rect]` wrapper.
  - Implement `finalize_panel_boxes(boxes, page_w, page_h, *, text_boxes=(), scores=None, iou_threshold=DEFAULT_IOU_THRESHOLD, containment_threshold=DEFAULT_CONTAINMENT_THRESHOLD, padding=DEFAULT_PADDING, bleed_margin=DEFAULT_BLEED_MARGIN, preserve_order=False, merge_bridged_panels=True, text_bridge_overlap=DEFAULT_TEXT_BRIDGE_OVERLAP, panel_overlap_threshold=DEFAULT_PANEL_OVERLAP_THRESHOLD, max_merged_area_ratio=DEFAULT_MAX_MERGED_AREA_RATIO, min_fill_ratio=DEFAULT_MIN_FILL_RATIO) -> tuple[list[PanelBox], list[float] | None]`:
    - Validate `page_w > 0 and page_h > 0`.
    - Run `nms_and_prune_indices(..., preserve_order=preserve_order)`.
    - Slice `kept_rects = [boxes[i] for i in surviving]` and `kept_scores = [float(scores[i]) for i in surviving] if scores is not None else None`.
    - If `merge_bridged_panels` and `kept_rects`: run `merge_bridged_panel_clusters(kept_rects, page_w, page_h, ...)` and aggregate `out_scores = [max(kept_scores[m] for m in c.member_indices) for c in clusters]`.
    - Call `pad_and_clamp(...)` for each rect; if `preserve_order=True`, set `ordinal=idx` (`1..N`) via `.model_copy(update={"ordinal": idx})`, else leave `ordinal=0`.

- [ ] **Step 5: Run pytest on `tests/test_boxes.py` and commit**
  Run: `pytest tests/test_boxes.py -v`
  Commit: `feat(cv): add bridged panel merging geometry and config`

---

### Task 2: `MagiBackend` Text-Bridge Extraction & Shared Finalizer (`magi.py`)

**Files:**
- Modify: `src/manga2kindle/cv/backends/magi.py:15-35,223-268`
- Test: `tests/test_magi_backend.py`

- [ ] **Step 1: Write failing tests in `tests/test_magi_backend.py`**
  Cover:
  - Mocked `page_result` with `"panels"` and bridging `"texts"` merges bridged panels while preserving Magi reading order (`ordinal=1, 2, ...`).
  - `"is_essential_text"` with `False` on a bridging SFX box suppresses merging for that box, while `True` merges.
  - `texts=None`, missing `"texts"`, mismatched `is_essential_text` length, or NumPy array `"texts"` / `"is_essential_text"` does not crash.
  - `detect_batch` with bridging `"texts"` merges per page cleanly.
  - `merge_bridged_panels=False` keeps panels unmerged.

- [ ] **Step 2: Run pytest on `tests/test_magi_backend.py` to confirm failure**
  Run: `pytest tests/test_magi_backend.py -v`

- [ ] **Step 3: Update `src/manga2kindle/cv/backends/magi.py`**
  - Define module constants `_MAGI_PANELS_KEY = "panels"`, `_MAGI_TEXTS_KEY = "texts"`, `_MAGI_ESSENTIAL_TEXT_KEY = "is_essential_text"`.
  - Add helper `_extract_essential_text_rects(page_result, orig_w, orig_h) -> list[Rect]` using explicit `is not None` checks (safe for `None` and `np.ndarray`) and filtering by `is_essential_text` when present with matching length. Also guard `raw_panels` with `is not None`.
  - Replace the `nms_and_prune_indices` + `pad_and_clamp` block in `_parse_page_result` with `finalize_panel_boxes(..., text_boxes=text_rects, preserve_order=True, ...)` using `self._config` settings.

- [ ] **Step 4: Run pytest on `tests/test_magi_backend.py` and commit**
  Run: `pytest tests/test_magi_backend.py -v`
  Commit: `feat(magi): merge panels bridged by essential text detections`

---

### Task 3: `OnnxYoloBackend` Multi-Class Text Decoding & Score-Preserving Merge (`models_hub.py`, `onnx_yolo.py`)

**Files:**
- Modify: `src/manga2kindle/cv/models_hub.py:18,40-89`
- Modify: `src/manga2kindle/cv/backends/onnx_yolo.py:14-35,176-354`
- Test: `tests/test_onnx_yolo.py`

- [ ] **Step 1: Write failing tests in `tests/test_onnx_yolo.py`**
  Cover:
  - `ModelSpec` for `manga109-yolo-s` and `manga109-yolo-n` sets `text_class_id == 3`.
  - `decode_yolo_detections` extracts both class `2` (`frame`) and class `3` (`text`) from raw `(1, 8, N)` and NMS-exported `(1, N, 6)` outputs.
  - Exact-boundary 3-class `(1, 7, N)` tensor with `text_class_id=3` skips text extraction safely; negative `text_class_id` raises `ValueError`.
  - Duplicate raw text-box anchors are pruned via NMS before bridging.
  - `OnnxYoloBackend` loaded via registry `model_id` merges panels bridged by class-3 text boxes, aggregates score via `max(...)`, and keeps `len(panels) == len(scores)`.
  - Custom `.onnx` model via `model_path` (`text_class_id=None`) skips text extraction safely.
  - Existing `decode_yolo_output` signature and return type remain unchanged.
  - `merge_bridged_panels=False` keeps panels unmerged.

- [ ] **Step 2: Run pytest on `tests/test_onnx_yolo.py` to confirm failure**
  Run: `pytest tests/test_onnx_yolo.py -v`

- [ ] **Step 3: Update `src/manga2kindle/cv/models_hub.py` and `src/manga2kindle/cv/backends/onnx_yolo.py`**
  - In `models_hub.py`: add `MANGA109_TEXT_CLASS_ID = 3` and append `text_class_id: int | None = None` at the end of `ModelSpec` (set to `MANGA109_TEXT_CLASS_ID` for `manga109-yolo-s` and `manga109-yolo-n`).
  - In `onnx_yolo.py`:
    - Add `@dataclass(frozen=True) class YoloDetections(panel_rects, panel_scores, text_rects, text_scores)` and `@dataclass(frozen=True) class _ResolvedOnnxModel(model_path, panel_class_id, text_class_id, input_size)`.
    - Implement `decode_yolo_detections(..., text_class_id: int | None = None) -> YoloDetections` (validate `text_class_id >= 0` when not `None`; skip text extraction when `None` or `4 + text_class_id >= arr.shape[0]`).
    - Keep `decode_yolo_output(...)` delegating to `decode_yolo_detections(..., text_class_id=None)` and returning `(det.panel_rects, det.panel_scores)`.
    - In `OnnxYoloBackend.detect`: when `self._config.merge_bridged_panels` and `det.text_rects`, run `nms_and_prune(det.text_rects, iou_threshold=self._config.iou_threshold, containment_threshold=self._config.containment_threshold, scores=det.text_scores)`, then call `finalize_panel_boxes(..., text_boxes=pruned_text_rects, scores=det.panel_scores, preserve_order=False, ...)`.

- [ ] **Step 4: Run pytest on `tests/test_onnx_yolo.py` and commit**
  Run: `pytest tests/test_onnx_yolo.py -v`
  Commit: `feat(onnx): decode text class and merge bridged panels`

---

### Task 4: `ContourBackend` & `BackendRegistry` Wiring (`contour.py`, `backends/__init__.py`)

**Files:**
- Modify: `src/manga2kindle/cv/backends/contour.py:13-88`
- Modify: `src/manga2kindle/cv/backends/__init__.py:42-48`
- Test: `tests/test_backends_contour.py`, `tests/test_backend_registry.py`

- [ ] **Step 1: Write failing tests in `tests/test_backends_contour.py` and `tests/test_backend_registry.py`**
  Cover:
  - `ContourBackend` accepts optional `merge_bridged_panels`, `panel_overlap_threshold`, `max_merged_area_ratio`, `min_fill_ratio` kwargs appended after `max_ink_ratio`, preserving existing positional/keyword callers and `PanelDetector()`.
  - Overlapping contours below `iou_threshold` merge when `merge_bridged_panels=True` and stay separate when `False`.
  - `_build_contour_backend(config)` and `BackendRegistry(config).resolve()` forward the new `DetectorConfig` fields.

- [ ] **Step 2: Run pytest on contour and registry tests to confirm failure**
  Run: `pytest tests/test_backends_contour.py tests/test_backend_registry.py -v`

- [ ] **Step 3: Update `src/manga2kindle/cv/backends/contour.py` and `src/manga2kindle/cv/backends/__init__.py`**
  - Append the new kwargs with defaults from `boxes.py` after `max_ink_ratio` in `ContourBackend.__init__`.
  - Replace `nms_and_prune` + `pad_and_clamp` in `ContourBackend.detect` with `finalize_panel_boxes(..., text_boxes=(), scores=None, preserve_order=False, ...)`.
  - Forward `merge_bridged_panels`, `panel_overlap_threshold`, `max_merged_area_ratio`, `min_fill_ratio` in `_build_contour_backend(config)`.

- [ ] **Step 4: Run pytest on contour and registry tests and commit**
  Run: `pytest tests/test_backends_contour.py tests/test_backend_registry.py -v`
  Commit: `feat(contour): wire bridged panel merging into ContourBackend and registry`

---

### Task 5: CLI `--merge-panels/--no-merge-panels`, `benchmark` Isolation & `README.md` (`cli.py`, `README.md`)

**Files:**
- Modify: `src/manga2kindle/cli.py:111-141,340-405,605-645,977-995`
- Modify: `README.md`
- Test: `tests/test_cli_e2e.py`, `tests/test_benchmark.py`

- [ ] **Step 1: Write failing CLI and benchmark tests in `tests/test_cli_e2e.py` and `tests/test_benchmark.py`**
  Cover:
  - `convert` (including `--preview-only`) and `preview-panels` (including `--compare`) default `merge_bridged_panels=True` and accept `--no-merge-panels` to set `merge_bridged_panels=False`.
  - `benchmark` forces `merge_bridged_panels=False` on its `DetectorConfig`.

- [ ] **Step 2: Run pytest on CLI and benchmark tests to confirm failure**
  Run: `pytest tests/test_cli_e2e.py tests/test_benchmark.py -v`

- [ ] **Step 3: Update `src/manga2kindle/cli.py` and `README.md`**
  - Add `merge_bridged_panels: bool = True` parameter to `_build_detector_config`.
  - Add `--merge-panels/--no-merge-panels` Typer option (default `True`, no spaces around `/`) to `convert` and `preview_panels` and pass to `_build_detector_config`.
  - Pass `merge_bridged_panels=False` in `benchmark`.
  - Update `README.md` CLI options documentation for `--merge-panels / --no-merge-panels`.

- [ ] **Step 4: Run full test suite and commit**
  Run: `pytest -q`
  Commit: `feat(cli): add --merge-panels flag and document in README`
