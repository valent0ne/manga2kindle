# Pluggable Panel Detection Backends: Design Spec

**Date:** 2026-11-15
**Status:** Draft, open questions resolved (§10), ready for planning by a follow-up agent
**Related:** [2026-11-15-manga2kindle-design.md](./2026-11-15-manga2kindle-design.md) (§5, Stage 3)

## 1. Intent & Why

Stage 3 today is a single class, `PanelDetector` in `src/manga2kindle/cv/detector.py`. It uses:
- bilateral filter
- Otsu inverted threshold
- 5x5 morphological close
- `RETR_EXTERNAL` contours
- area, aspect and ink-ratio filters
- IoU NMS
- padding

It is a classical baseline. Its known failure modes are:
- borderless panels and bleeds
- diagonal or overlapping panels
- art or speech bubbles that cross gutters
- screentone and dense backgrounds that merge neighbouring panels
- per-title threshold tuning
- no notion of reading order beyond the row-clustering `PanelSorter`

Panel quality decides whether Guided View on the Kindle is usable. This spec adds learned detection backends behind a common interface. The contour detector stays as the zero-dependency default and as a fallback.

## 2. Goals, Non-Goals

### Goals
- Introduce a `PanelDetectorBackend` interface with several selectable implementations.
- Add at least one learned backend that handles borderless and non-rectilinear layouts.
- Keep the base install light. Learned backends are optional extras.
- Add a reproducible benchmark so the backends are compared with data, not opinion.
- Optionally use a model's reading order, with `PanelSorter` as the fallback.
- Work on macOS and Windows with Python 3.11+. GPU is optional.

### Non-Goals
- No training pipeline in v1. Use existing pretrained weights. Fine-tuning is a later stage (§9).
- No OCR, character detection, captioning or speaker diarisation. Magi can do these but they are out of scope (see the main design's Non-Goals).
- No GUI or interactive panel editor.
- No network calls at conversion time other than a one-time, explicit model download.

### Anti-Goals
- Silently downloading large weights without telling the user.
- Silently falling back to a different backend. Log it and show it in the CLI output.
- Making `torch` a core dependency.
- Shipping or bundling weights whose license conflicts with this project (§6).

## 3. Scouted Options

Findings come from web research on 2026-11-15. Search summaries were not primary sources, so every item marked **verify** must be checked against the model card or repo before implementation.

| # | Option | Type | Outputs | Runtime / deps | Strength | Weakness | License (verify) |
|---|--------|------|---------|----------------|----------|----------|------------------|
| A | **Magiv3** (`ragavsachdeva/magiv3`, Oxford, Sachdeva & Zisserman) | Vision-language model (Florence-2-style, next-token detection) | Panels, text, characters, tail boxes; plus OCR, captions and reading order | `transformers` + `torch`, `trust_remote_code=True`, CUDA recommended, large | Likely best accuracy. Handles borderless, overlapping and irregular layouts natively. No confidence thresholds or post-process heuristics. | Heavy, slow on CPU, remote code execution risk, needs a GPU to be practical | Non-commercial or research use, per the model card. **Verify**: compatibility with this repo's AGPL-3.0 |
| B | **Magiv2** (`ragavsachdeva/magiv2`) | DETR-style detector plus assignment heads | Panels, text, characters; reading order via a DAG; speaker association | `transformers` + `torch` | Proven, well documented, lighter than Magiv3 | Panel detector is fixed-threshold. Same license limits as Magi. | Same family. **Verify** |
| C | **YOLO fine-tuned on Manga109**, e.g. `deepghs/manga109_yolo` (classes: frame/panel, text, body, face; nano to XL) | Single-stage detector | Panel boxes (also text, body, face) | Ultralytics `torch`, or exported **ONNX** via `onnxruntime` | Fast, runs on CPU, ONNX keeps install light, trivial NMS | Boxes only. No reading order. Trained on Manga109, so a dataset-license question. Weak on Western comics and webtoons. | Weights may be Apache/MIT. The Ultralytics library is AGPL-3.0. Manga109-s is non-commercial. **Verify** |
| D | **Community panel-only YOLO ONNX** (e.g. `mednasserallah/manga-panel-detector-yolo26n-onnx`; reported panel mAP50 above 0.98) | Single-stage detector, panel class only | Panel boxes | `onnxruntime` only (no torch) | Smallest dependency footprint. Good candidate for the default learned backend. | Claimed metrics are unverified and probably in-distribution. Provenance and training data unclear. | **Verify** |
| E | **SAM / SAM2 plus panel prompts or fine-tune** | Promptable segmentation | Pixel masks. Gives polygons and real shapes. | `torch`, large weights | Best fit for slanted or polygonal panels. Could drive non-rectangular Guided View regions later. | Slow, needs prompts or auto-mask post-processing. Zero-shot SAM over-segments (characters and bubbles are returned as masks too). Kindle Guided View still wants rectangles. | Apache-2.0 (SAM). **Verify** |
| F | **Kumiko** (`njean42/kumiko`) | Classical (OpenCV: Canny, Hough, contours, plus reading-order logic) | Panels and reading order | OpenCV only | Different heuristics from ours, such as gutter-based recursive splitting. Useful as a second classical reference. | Same family of failure modes as the current detector. | GPL-3.0. **Verify**. Prefer to re-implement the idea, not vendor the code. |
| G | **Datasets / benchmarks**: Manga109 and the segmentation updates, eBDtheque, Comics100 / Comics Datasets Framework, CoMix | Evaluation data | Panel boxes and masks | n/a | Ground truth for the benchmark (§7) | Manga109 and CoMix have licensing and redistribution limits. Never commit them. | **Verify** each |

### Recommendation
1. **Interface first.** Add a `PanelDetectorBackend` protocol and make the current detector its first implementation (`ContourBackend`).
2. **Portable learned backend: C or D**, via ONNX Runtime. It is the cheapest to ship, works on CPU, and is the middle step of `AUTO`. Choose between them after the benchmark in §7.
3. **Quality backend: Magiv3 (A)** behind an extra, and the first choice in `AUTO` when installed and a GPU is usable. Include its reading order.
4. **Out of scope for v1:** SAM (E). Re-evaluate it if non-rectangular Guided View regions become a goal.
5. **Auto mode (quality-first, per §10)**: Magi → ONNX YOLO → contour. Each step falls back to the next on load failure or zero panels. The order is provisional until the benchmark confirms it.

## 4. Architecture

New or changed modules under `src/manga2kindle/cv/`:

```
cv/
  backends/
    __init__.py        # BackendRegistry, create_backend(kind, config)
    base.py            # PanelDetectorBackend (Protocol or ABC), DetectionResult
    contour.py         # ContourBackend (current PanelDetector logic, moved)
    onnx_yolo.py       # OnnxYoloBackend  (options C/D)
    magi.py            # MagiBackend      (option A/B, optional extra)
  detector.py          # thin compatibility shim, re-exporting ContourBackend as PanelDetector
  sorter.py            # unchanged, used when result.ordered is False
  boxes.py             # shared helpers: compute_iou, nms, pad_and_clamp (DRY, extracted)
  benchmark.py         # evaluation harness (§7)
```

### Domain types (`models.py`, typed, no dicts)
- `DetectorKind(str, Enum)`: `AUTO`, `CONTOUR`, `ONNX_YOLO`, `MAGI`.
- `DetectionResult` (dataclass): `panels: list[PanelBox]`, `ordered: bool`, `backend: DetectorKind`, `scores: list[float] | None`.
- `DetectorConfig` (Pydantic): `kind`, `model_path: Path | None`, `model_id: str | None`, `device` (`auto|cpu|cuda|mps`), `score_threshold`, `iou_threshold`, `padding`, `min_panel_area`.

### Interface
```python
class PanelDetectorBackend(Protocol):
    kind: DetectorKind
    def detect(self, image_path: Path) -> DetectionResult: ...
    def detect_batch(self, image_paths: Sequence[Path]) -> list[DetectionResult]: ...  # default: loop
```

### Behaviour rules
- Every backend returns `PanelBox` in **source-page pixel coordinates**, clamped to the page. Letterbox and resize are undone inside the backend.
- Shared post-processing (NMS, min area, padding, bleed clamp) lives in `boxes.py` and is reused by all backends. Do not copy it.
- If a backend returns zero panels, the pipeline uses the existing full-page fallback. In `AUTO` mode it first walks the rest of the chain (§10) before the full-page fallback.
- Reading order: if `DetectionResult.ordered` is true (Magi), keep it. Otherwise call `PanelSorter` with the configured direction.
- Model load failure in `AUTO` mode logs a warning and moves to the next backend in the chain (Magi → ONNX YOLO → contour). In an explicit mode (`--detector onnx`) it is a hard error with an actionable message.
- No magic strings. Class names from models (`"frame"`, `"panel"`) go in named constants or an enum mapping.

### Models and weights
- Weights live in the platform cache dir (`platformdirs`; macOS `~/Library/Caches`, Windows `%LOCALAPPDATA%`), via `pathlib.Path`.
- Add `manga2kindle models fetch <name>`: explicit download, prints size, source and license, and verifies a pinned SHA-256.
- Pin model revision or commit hash. For Magi, pin the HF revision too, because `trust_remote_code=True` executes repo code. Document this risk and require an explicit opt-in flag (`--allow-remote-code`).

## 5. CLI & Config Changes

- `convert` and `preview-panels` get:
  - `--detector [auto|contour|onnx|magi]` (default `auto`)
  - `--detector-model PATH_OR_ID`
  - `--device [auto|cpu|cuda|mps]`
  - `--score-threshold FLOAT`
  - `--allow-remote-code` (Magi only)
- `doctor`: report `onnxruntime`, `torch` (if present), cached models and active device.
- `preview-panels`: print the backend used per page and draw a different colour per backend if run with several.
- New `manga2kindle benchmark` command (§7).
- Optional extras in `pyproject.toml`:
  - `onnx = ["onnxruntime>=1.17", "platformdirs"]`
  - `magi = ["torch", "transformers", "einops", "timm"]` (finalise versions against the model card)
  - Keep the base dependencies unchanged.
- Update `README.md` (flags, install extras, model download, license notes) as required by `AGENTS.md`.

## 6. Licensing & Compliance (blocking, resolve before shipping any backend)

- The project is **AGPL-3.0-or-later**. Check each model and dataset license for compatibility, and whether the CLI may *download and run* weights as opposed to *redistribute* them. Never bundle weights in the wheel or repo.
- The Magi models are described as non-commercial or research use only. Make that visible at `models fetch` time and in the README.
- Manga109-trained weights inherit the dataset's non-commercial terms. State this.
- The Ultralytics runtime is AGPL-3.0. Prefer ONNX Runtime so the base and `onnx` extra do not need it.
- Record the license and source URL of every backend in a `docs/` table. Implementers must open the real model cards. This spec's summaries are not authoritative.

## 7. Evaluation (build this before choosing the default)

Goal: pick the default learned backend and the `AUTO` policy from measured results.

- **Dataset:** a local, git-ignored folder `benchmarks/pages/` of about 40 to 100 pages with hand-corrected panel boxes (JSON or COCO). Mix:
  - clean grids
  - borderless
  - diagonal or overlapping
  - splash
  - screentone-heavy
  - Western comic
  - webtoon-style strip
  - Use the user's own sample chapters. Optionally add a Manga109 subset where the license allows local use.
- **Metrics per backend:**
  - panel recall and precision at IoU 0.5 and 0.75
  - mean best-match IoU
  - exact page count match
  - reading-order accuracy (Kendall tau or pairwise order accuracy)
  - ms/page on CPU, and on GPU if available
  - peak RAM
- **Harness:** `manga2kindle benchmark --pages DIR --truth FILE --detectors contour,onnx,magi` prints a Rich table and writes JSON. Metric code is pure and unit-tested with synthetic boxes.
- **Annotation helper (optional):** `preview-panels` can export current detections as editable ground-truth JSON.
- Add `benchmarks/`, `*.onnx`, `*.safetensors`, `*.pt` and the model cache to `.gitignore`.

## 8. Testing Plan

- **Unit (no models):**
  - `boxes.py` helpers, IoU, NMS and clamping, with the existing tests moved
  - `ContourBackend` regression, so existing `tests/test_detector_sorter.py` still passes
  - registry and `AUTO` fallback with fake backends
  - metric functions
  - letterbox and de-letterbox coordinate round-trip
- **Backend tests:** `OnnxYoloBackend` post-processing is tested on canned raw output tensors, with no model file needed. A real-weights smoke test is marked `@pytest.mark.model` and skipped by default.
- **Integration:** the CLI e2e test runs with `--detector contour`. Add one with a stub backend that returns fixed panels.
- **Verification before completion:** `pytest` green, and a benchmark run recorded in `docs/` showing contour against the learned backend on the sample set.

## 9. Phasing

1. **P1:** Extract `boxes.py`, add the interface and `ContourBackend`, and keep `PanelDetector` as a shim. No behaviour change.
2. **P2:** `benchmark` command and metrics, plus a small annotated set.
3. **P3:** `OnnxYoloBackend` (evaluate C and D), `models fetch`, `AUTO` mode.
4. **P4:** `MagiBackend` as an optional extra, including reading order.
5. **P5 (later):** fine-tune a YOLO or RT-DETR on the user's corrected pages if a gap remains. Evaluate SAM for polygon panels.

## 10. Decisions (answered by the user, 2026-11-15)

1. **Personal use only.** The non-commercial terms of Magi and Manga109 are acceptable. Still show them at `models fetch` time and in the README. Do not redistribute weights.
2. **GPU: available if needed.** Magiv3 is viable, but CPU-only must keep working through the contour and ONNX backends.
3. **No golden set yet.** The contour method looked promising in manual tests. Building the golden set is part of this work (P2) and is a prerequisite for choosing defaults. Western comics and webtoons stay out of scope unless the user's own pages include them.
4. **`AUTO` prefers quality.** Order: `MAGI` (if its extra is installed, its weights are cached and a usable device exists) → `ONNX_YOLO` (if available) → `CONTOUR`. Any failure or zero-panel result moves to the next backend, with a warning shown in the CLI. Because there is no golden set yet, the order is provisional. P2 may change it if the benchmark shows the contour backend is as good on the user's pages.

### Consequences for the plan
- Because the contour detector already looks promising, the benchmark (§7) is the gate. Any learned backend must beat contour on the golden set (recall at IoU 0.5 and reading-order accuracy) before it is made part of `AUTO`.
- Golden set creation: run `preview-panels` on 40 to 60 of the user's own pages and export the detections as editable ground-truth JSON. The user then corrects them by hand. Prefer a tiny helper (export and import JSON) over a GUI editor, which stays a non-goal.
- Reading order is a quality factor. When Magi is used, its order wins over `PanelSorter`.

## 11. Left to the Builder

- Confirm every license and metric above from primary sources. The scouting results are secondary.
- Check the exact ONNX I/O format (input size, output layout, class ids) of the chosen model.
- Verify the Magiv3 API (`predict_detections_and_associations` and its output schema) against the pinned revision.
- Find the box-merge and padding defaults that suit Guided View on the Paperwhite 12th gen (1264x1680).
