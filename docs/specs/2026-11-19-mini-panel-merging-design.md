# Profile-Calibrated Mini-Panel Merging Post-Processor: Design Spec

**Date:** 2026-11-19
**Status:** Implemented
**Related:** [2026-11-18-bridged-panel-merging-design.md](./2026-11-18-bridged-panel-merging-design.md)

## 1. Intent & Why

On a **7" Kindle** (`KPW6`, `1264×1680`), even after dialogue-bridged panel merging (`v0.6.0`), high-density action/reaction layouts often contain runs of 3+ tiny horizontal panels or narrow vertically stacked mini sub-columns beside a dominant hero panel:

- Stepping through 3–5 adjacent mini panels in one compact horizontal band forces tedious rapid page taps while showing heavily zoomed low-detail crops.
- Vertically stacked mini panels sharing one narrow column beside a large hero panel behave visually as one beats column next to the hero panel.
- Conversely, taller multi-column dialogue bands (such as `3` tall columns spanning ~45% of page height) remain comfortable individual crops on 7" devices while only becoming overly fragmented on large displays (`KS`, 10.2").

This design adds **Mode B** — a profile-calibrated, post-sort `PanelMerger` layered directly after `v0.6.1`'s bridged-panel & bubble-contour pipeline (`merge_bridged_panel_clusters` + `PanelSorter`).

## 2. Architecture & Pipeline Placement

```text
Backend detection (magi / onnx / contour)
  → merge_bridged_panel_clusters + expand_text_boxes_to_bubbles + pad_and_clamp (0.04)
  → PanelSorter (XY-cut gutter sorting + ReadingDirection ordinals)
  → PanelMerger (padding-aware Mode B mini-panel merge + ordinal renumbering)   # ← NEW
```

Why post-sort rather than pre-padding inside `boxes.py`:
1. `PanelSorter` establishes exact RTL/LTR reading order across XY gutters and spreads; `PanelMerger` only merges **contiguous** reading-order slices, preventing out-of-order jumps.
2. Bridged dialogue merges (`merge_bridged_panel_clusters`) and bubble/margin expansions have already resolved shared text boxes and contour recovery beforehand.
3. Because `v0.6.1` emits panels padded with `DEFAULT_PADDING = 0.04`, `PanelMerger` unpads panel rectangles via `unpad_rect` before checking height/width/area ratios, uncapped inset gutter splits (`inset_max_px=None`), and external overlap containment, while emitting the exact union of the already-padded member boxes (preserving bubble clearance without double-padding).

## 3. Merge Rules

1. **Rule 1 — Compact Horizontal Mini-Panel Tiers (`>= min_tier_panels`, default `3`):**
   - Split the page (or spread half) into horizontal bands by inset X/Y gutters.
   - If the whole band is contiguous in reading order, has at least `min_tier_panels`, total unpadded height ratio `<= max_tier_height_ratio`, every unpadded member has width ratio `<= max_tier_panel_width_ratio` and area ratio `<= max_tier_panel_area_ratio`, and its unpadded union does not overlap any external panel beyond `max_external_overlap_ratio`, merge the band into one frame (supports nested stacked sub-panels inside compact bands).
2. **Rule 2 — Narrow Stacked Mini Sub-Columns (`>= min_subcol_panels`, default `2`):**
   - When Rule 1 skips a mixed/taller band, split it into vertical columns (`len(cols) >= 2`).
   - For any column containing `>= 2` vertically stacked panels (`len(split_by_gutter(col, HORIZONTAL)) >= 2`), merge the column if it is reading-order contiguous, within `max_subcol_width_ratio`, `max_subcol_height_ratio`, `max_subcol_panel_area_ratio`, and `max_external_overlap_ratio`.
   - Dominant hero panels stay standalone, and 4-koma single-column stacks (`len(cols) == 1`) are never merged.

## 4. Profile Size-Tier Calibration (`DeviceSizeTier`)

Display presets and `--custom-size` profiles are classified by short edge (`min(width, height)`):

| Tier (`DeviceSizeTier`) | Short Edge | Profiles | Bridged Area Cap (`max_merged_area_ratio`) | Tier Height Cap | Max Panel W / Area | Sub-Col W / H / Panel Area |
| --- | --- | --- | --- | --- | --- | --- |
| `COMPACT` | `< 1150px` | 6" Basic (`CUSTOM` e.g. `1072×1448`) | `0.45` | `0.32` | `0.45` / `0.15` | `0.40` / `0.48` / `0.12` |
| `STANDARD` *(default)* | `1150–1599px` | `KPW6`, `KPW5`, `OASIS` | `0.55` | `0.37` | `0.55` / `0.20` | `0.45` / `0.55` / `0.15` |
| `LARGE` | `>= 1600px` | `KS` (Kindle Scribe) | `0.70` | `0.46` | `0.60` / `0.25` | `0.50` / `0.65` / `0.20` |

Additional bridged/gutter safety guards in `boxes.py` / `magi.py`:
- **Partial-slice bridge guard (`_MAX_EXTERNAL_SLICE_OVERLAP = 0.25`):** rejects L-shaped bridge unions that slice `>= 25%` across unrelated panels instead of engulfing them.
- **Nearest zero-conflict gutter text expansion & floating non-essential narration enclosure:** when unmerged gutter text sits near `>= 2` panels (`prox_matches`) — including floating `is_essential_text=False` narration captions with `< text_bridge_overlap` panel overlap (`margin_text_boxes`) — expands the closest zero-conflict panel up to `_MAX_TEXT_EXPANSION_AREA_RATIO = 0.85` without overlapping sibling panels or triggering panel merges.
- **Essential-only alcove fallback (`_ESSENTIAL_ALCOVE_MAX_GAP_RATIO = 0.12`, `_MAX_ALCOVE_ADDED_AREA_RATIO = 0.08`):** when an essential dialogue box (`text_boxes` only, never `margin_text_boxes`) sits in open white space beyond the 4% halo, expands the nearest zero-conflict cluster up to 12% gap provided added area stays `<= 8%` of the page and adds zero overlap with sibling panels.

`ConversionConfig` auto-calibrates `panel_merge: PanelMergeConfig` and unset `detector.max_merged_area_ratio` from `profile` while preserving any explicitly overridden fields and respecting `detector.merge_bridged_panels`. The existing `--merge-panels / --no-merge-panels` CLI flag controls both bridged/bubble merging and Mode B mini-panel merging together (`benchmark` continues to keep `merge_bridged_panels=False` and skips `PanelMerger`).
