# Profile-Calibrated Mini-Panel Merging: Implementation Plan

**Date:** 2026-11-19
**Spec:** [2026-11-19-mini-panel-merging-design.md](../specs/2026-11-19-mini-panel-merging-design.md)
**Branch Base:** `develop` (`v0.6.1` + `pytorch-metric-learning<3.0` pin)

## Tasks

- [x] **1. Sync `develop` (`v0.6.1`) & pin `pytorch-metric-learning<3.0`:** Pin `>=2.0,<3.0` in `pyproject.toml`, update `uv.lock`, merge `fix/pin-pytorch-metric-learning` into `develop`, verify `magiv3` compatibility and 245 baseline tests.
- [x] **2. Extract shared gutter-splitting & unpadding helpers (`cv/boxes.py`, `cv/sorter.py`):** Add `CutAxis`, `GUTTER_INSET_MAX_PX`, `GUTTER_INSET_RATIO`, `panel_to_rect`, `union_rect`, `unpad_rect`, `axis_interval`, and `split_by_gutter` (supporting configurable `inset_ratio` and uncapped `inset_max_px=None`), delegate `PanelSorter` to `boxes.py`, and add unit tests in `tests/test_boxes.py`.
- [x] **3. Add `DeviceSizeTier` & `PanelMergeConfig` (`models.py`):** Add `DeviceSizeTier`, tier presets, `DeviceProfile.size_tier`, `PanelMergeConfig.for_tier`/`for_profile`, `ConversionConfig.panel_merge` auto-calibration validator, and unit tests in `tests/test_models_platform.py`.
- [x] **4. Implement padding-aware `PanelMerger` (`cv/merger.py`, `cv/__init__.py`):** Implement Rule 1 (compact horizontal tiers), Rule 2 (stacked mini sub-columns), unpadded threshold/overlap checks, uncapped high-res padding inset splitting, spread-half guards, duplicate-instance safety, and `tests/test_merger.py`.
- [x] **5. Wire into `pipeline.py`, `cli.py`, `README.md`, and `tests/test_cli_e2e.py`:** Invoke `PanelMerger` post-sort in `ConversionPipeline._detect_and_sort_pages` (`_convert_single`, `preview_panels`, `preview_batch`), pass profile-calibrated `PanelMergeConfig` from CLI under `--merge-panels/--no-merge-panels`, update `README.md`, add CLI/E2E tests, and verify full `pytest` suite.
- [x] **6. Real-manga verification:** Run `magiv3` `preview-panels` and coverage diagnostics on `Kagurabachi 069 (2025).cbz`, `Kagurabachi 070 (2025).cbz`, and unseen `Kagurabachi 071 (2025).cbz`, verify per-page merge counts (unmerged vs bridged-only vs bridged + Mode B) and text coverage (`57/57` pages `>= 90%` text coverage, `0/57` `> 0.55` bridged collapses), and visually inspect pages across all three chapters.
