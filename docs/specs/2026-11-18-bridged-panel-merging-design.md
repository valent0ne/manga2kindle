# Bridged Panel Merging Post-Processor: Design Spec

**Date:** 2026-11-18
**Status:** Reviewed & Ready for Planning
**Related:** [2026-11-15-panel-detection-backends-design.md](./2026-11-15-panel-detection-backends-design.md) (§3, §4)

## 1. Intent & Why

On a **7" Kindle**, Guided View relies on splitting manga pages into readable panel crops. However, **overly granular splits** damage the reading experience:

- Learned detectors (`magi`, `onnx`) are trained on semantic panel annotations (e.g., Manga109), so they routinely split two adjacent panels even when a **dialogue bubble** crosses the gutter between them.
- In Guided View, splitting across a bridging bubble slices the speech bubble in half across two consecutive zoom steps.
- Classical `contour` detection naturally keeps ink-connected panels together, but struggles on borderless, dark-background, or full-bleed pages.

This spec adds a **lightweight, geometry-based panel-merging post-processor** that runs after containment/IoU NMS and before padding/clamping, combining ML borderless accuracy with intact speech bubbles—while guarding against over-merging so crops remain comfortable to read on a 7" screen.

## 2. Goals, Non-Goals & Anti-Goals

### Goals
- Merge adjacent panel boxes when a detected **text/dialogue box bridges both panels**, preserving the full bubble in a single Guided View frame.
- Merge adjacent panel boxes when they **overlap substantially** below the NMS IoU cutoff (e.g., overlapping inset/character panels).
- Enforce **readability guardrails** (`max_merged_area_ratio` and `min_fill_ratio`) so transitive or L-shaped merges cannot collapse a page into an unreadable full-page view or engulf unrelated panels on a 7" Kindle.
- Preserve **native reading order** for ordered backends (`magi`) by assigning each merged cluster its earliest constituent's ordinal, and preserve **score alignment** for scored backends (`onnx`).
- Expose a clean CLI/config toggle (`--merge-panels` / `--no-merge-panels`, enabled by default for conversion and previews, disabled by default for `benchmark`).

### Non-Goals
- No new ML model, OCR engine, or separate bubble-segmentation network.
- No text extraction or speaker diarisation in Kindle output.
- No changes to `DetectionResult` (`src/manga2kindle/cv/backends/base.py`) or `PanelSorter` (`src/manga2kindle/cv/sorter.py`).

### Anti-Goals
- Running the merge pass *after* `pad_and_clamp` (artificial 2% padding would create false overlaps/bridges).
- Unbounded transitive chaining or L-shaped dead-space unions that engulf unrelated panels.
- Breaking existing `ContourBackend(...)` kwargs callers or `decode_yolo_output(...)` return signatures.
- Altering `manga2kindle benchmark` PopManga/CoMix ground-truth metrics by merging semantic panels during evaluation.

## 3. Architecture & Data Flow

### Shared Post-Processing Tail (`boxes.py`)

In each backend (`MagiBackend`, `OnnxYoloBackend`, `ContourBackend`), single-page post-processing currently runs `nms_and_prune_indices` followed by `pad_and_clamp`. We extract a shared helper `finalize_panel_boxes(...)` in `src/manga2kindle/cv/boxes.py`:

```text
raw panel rects (+ optional raw text rects, optional scores)
  → nms_and_prune_indices(...)
  → merge_bridged_panel_clusters(...)   # ← NEW: only when merge_bridged_panels=True
  → pad_and_clamp(...) + score aggregation
```

Running `merge_bridged_panel_clusters` **after** containment/IoU pruning and **before** `pad_and_clamp` ensures:
1. Duplicate/nested detections are already removed.
2. Padding never manufactures false overlaps or false text bridges.
3. Each final merged box receives a single clean `pad_and_clamp` pass.
4. For scored backends (`onnx`), each merged cluster aggregates its score as `max(scores[i] for i in cluster.member_indices)`.

### Text-Box Extraction (Zero Extra Inference Cost)

Both ML backends already detect text regions in the same forward pass:
- **`MagiBackend` (`src/manga2kindle/cv/backends/magi.py`):**
  - `predict_detections_and_associations` returns `"texts"` (in `[x1, y1, x2, y2]` source-pixel coordinates) and optionally `"is_essential_text"` (boolean mask distinguishing dialogue from decorative/SFX text).
  - Read `raw_texts = page_result.get(_MAGI_TEXTS_KEY) if isinstance(page_result, dict) else None` and guard with `if raw_texts is not None:` (avoiding `None` or NumPy array truthiness errors).
  - If `"is_essential_text"` is present with matching length, filter `"texts"` to essential text boxes so decorative SFX spanning multiple panels does not trigger unintended merges.
  - Parse via `_xyxy_to_rect(box, orig_w, orig_h)`.
- **`OnnxYoloBackend` (`src/manga2kindle/cv/backends/onnx_yolo.py` & `models_hub.py`):**
  - Define `MANGA109_TEXT_CLASS_ID = 3` and `text_class_id: int | None = None` on `ModelSpec` (set to `MANGA109_TEXT_CLASS_ID` for `manga109-yolo-s` and `manga109-yolo-n`, and `None` for arbitrary custom `--detector-model` files).
  - Keep `decode_yolo_output(...) -> tuple[list[Rect], list[float]]` unchanged for backward compatibility; add a typed `@dataclass(frozen=True) class YoloDetections` and `decode_yolo_detections(...) -> YoloDetections` returning both panel and text boxes/scores.
  - When `text_class_id is not None` and the model output has enough classes (`4 + text_class_id < arr.shape[0]` in raw format, or matching class ID in NMS-exported format), decode `"text"` detections filtered by `config.score_threshold` and run `nms_and_prune` on them so raw YOLO anchor duplicates do not unpredictably trigger bridges.
- **`ContourBackend` (`src/manga2kindle/cv/backends/contour.py`):**
  - Preserve existing kwargs-based `__init__` signature (`min_panel_area`, `padding`, `iou_threshold`, `containment_threshold`, `min_ink_ratio`, `max_ink_ratio`), adding optional `merge_bridged_panels`, `panel_overlap_threshold`, `max_merged_area_ratio`, and `min_fill_ratio` kwargs with defaults from `boxes.py`.
  - Forward these fields in `_build_contour_backend(config)` (`src/manga2kindle/cv/backends/__init__.py`).
  - Supplies `text_boxes=()` (only overlap-based merging applies when enabled).

## 4. Merge Algorithm (`merge_bridged_panel_clusters` & `merge_bridged_panels`)

Implemented as pure, typed helpers in `src/manga2kindle/cv/boxes.py` using `@dataclass(frozen=True) class MergedCluster`:
- `rect: Rect`
- `member_indices: tuple[int, ...]` (sorted ascending)

Constants in `src/manga2kindle/cv/boxes.py`:
- `DEFAULT_TEXT_BRIDGE_OVERLAP = 0.15`
- `DEFAULT_PANEL_OVERLAP_THRESHOLD = 0.25`
- `DEFAULT_MAX_MERGED_AREA_RATIO = 0.65`
- `DEFAULT_MIN_FILL_RATIO = 0.55`
- `DEFAULT_MIN_TEXT_AREA_RATIO = 0.0005`

### Inputs & Validation
- `boxes: Sequence[Rect]` — surviving unpadded `(x, y, w, h)` panel boxes in backend order.
- `page_width: int`, `page_height: int` — must be positive (`> 0`); returns `[]` immediately when `boxes` is empty. Degenerate boxes (`w <= 0 or h <= 0`) are ignored.
- `text_boxes: Sequence[Rect] = ()` — optional detected text/dialogue boxes.

### Step-by-Step Logic
1. **Candidate Edges:** Build weighted merge candidates `(i, j)` with `i < j`:
   - **Text-bridge rule:** Ignore any `t` in `text_boxes` with `area(t) <= 0` or `area(t) / page_area < min_text_area_ratio` (prevents tiny punctuation/noise on a seam from merging large panels). For each surviving `t`, compute `frac_k = intersection_area(t, boxes[k]) / area(t)` for every panel `k`, keeping panels where `frac_k >= text_bridge_overlap`. If 2 or more panels match, emit candidate `(a, b)` for **all pairs** `a < b` in the matched set with weight `min(frac_a, frac_b)`.
   - **Panel-overlap rule:** For each pair `(i, j)` with `i < j`, compute `overlap = compute_containment(boxes[i], boxes[j])` (`intersection / min(area_i, area_j)`). If `overlap >= panel_overlap_threshold`, emit candidate `(i, j)` with weight `overlap`.
   - **Deduplication & Deterministic Order:** Deduplicate by `(i, j)` keeping `max(weight)`, and sort edges by `(-weight, i, j)` so the strongest bridges/overlaps always merge first regardless of input order.
2. **Guarded Union-Find:** Process sorted edges `(i, j)`:
   - Skip if `find(i) == find(j)`.
   - Compute `union_rect` of clusters `root(i)` and `root(j)` and `union_area = union_rect.w * union_rect.h`.
   - **Area-cap guardrail:** Reject if `union_area / page_area > max_merged_area_ratio`.
   - **Dead-space / L-shape guardrail:** Let `sum_member_areas` be the sum of individual `boxes[m]` areas in the combined cluster. Reject if `sum_member_areas / union_area < min_fill_ratio` (prevents diagonal/L-shaped panels from creating a huge empty bounding box).
   - **Engulfment guardrail:** Reject if `union_rect` would contain any unmerged panel `k` outside the two clusters with `compute_containment(union_rect, boxes[k]) >= containment_threshold`.
3. **Order-Preserving Output:**
   - Each surviving cluster emits `MergedCluster(rect=union_rect, member_indices=tuple(sorted(members)))`.
   - Clusters are sorted by `member_indices[0]` so `MagiBackend` (`ordered=True`) preserves its native reading order seamlessly.
   - `merge_bridged_panels(...) -> list[Rect]` wraps `merge_bridged_panel_clusters(...)` for callers that only need rectangles.

## 5. Configuration, CLI & Documentation

Add validated fields to `DetectorConfig` (`src/manga2kindle/models.py`):
- `merge_bridged_panels: bool = True`
- `text_bridge_overlap: float = Field(default=DEFAULT_TEXT_BRIDGE_OVERLAP, gt=0.0, le=1.0)`
- `panel_overlap_threshold: float = Field(default=DEFAULT_PANEL_OVERLAP_THRESHOLD, gt=0.0, le=1.0)`
- `max_merged_area_ratio: float = Field(default=DEFAULT_MAX_MERGED_AREA_RATIO, gt=0.0, le=1.0)`
- `min_fill_ratio: float = Field(default=DEFAULT_MIN_FILL_RATIO, gt=0.0, le=1.0)`

CLI (`src/manga2kindle/cli.py`):
- Add `--merge-panels / --no-merge-panels` (default: `True`) to `convert` and `preview-panels`.
- In `benchmark`, force `merge_bridged_panels=False` (or default `False`) so PopManga/CoMix semantic-panel evaluation metrics remain unaffected.

Documentation (`README.md`):
- Document `--merge-panels / --no-merge-panels` under the `convert` / `preview-panels` CLI options.

## 6. Testing & Verification Plan

1. **Unit tests (`tests/test_boxes.py`):**
   - Text box bridging two side-by-side panels merges them into one `MergedCluster` with sorted `member_indices`.
   - Text box wholly inside one panel, below `text_bridge_overlap`, or below `min_text_area_ratio` does not merge.
   - Text box bridging 3+ panels links all pairs and merges strongest-first, respecting `max_merged_area_ratio`.
   - Direct panel overlap above `panel_overlap_threshold` merges; below threshold does not.
   - `min_fill_ratio` rejects L-shaped/diagonal merges with excessive dead space; engulfment guardrail rejects merges that swallow an unrelated third panel.
   - Cluster ordering preserves `min(member_indices)` and validates positive page dimensions.
2. **Backend & registry integration tests (`tests/test_magi_backend.py`, `tests/test_onnx_yolo.py`, `tests/test_backends_contour.py`, `tests/test_backend_registry.py`):**
   - `MagiBackend`: `"texts"` bridging merges panels and preserves reading order; `is_essential_text=False` suppresses decorative/SFX bridges; `texts=None` or `np.ndarray` does not crash; `merge_bridged_panels=False` disables merging.
   - `OnnxYoloBackend`: `decode_yolo_detections` extracts class `3` text boxes with NMS; bridging text merges panels and aggregates score via `max(...)`; custom 1-class model (`text_class_id=None` or `< 4` classes) skips text extraction safely; `decode_yolo_output` remains backward compatible.
   - `ContourBackend` & `BackendRegistry`: forwards new config fields without breaking existing kwargs callers.
3. **CLI & Benchmark tests (`tests/test_cli.py`):**
   - `--merge-panels / --no-merge-panels` wires through `convert` and `preview-panels`; `benchmark` keeps `merge_bridged_panels=False`.
4. **Real-page visual verification:**
   - Run `preview-panels --compare magi,onnx,contour` with `--merge-panels` vs `--no-merge-panels` on real manga pages to verify split dialogue bubbles are merged cleanly without over-merging full pages.
