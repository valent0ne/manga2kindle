# SOTA End-to-End Throughput Implementation Plan

**Goal:** Maximize end-to-end throughput of `manga2kindle` across GPU (`magiv3`) and multi-core CPU/IO stages on Windows and macOS without degrading panel accuracy or causing memory/process leaks.

**Spec:** [`2026-11-20-sota-throughput-design.md`](../specs/2026-11-20-sota-throughput-design.md)

## Tasks

- [x] **Step 0 — Branching:** Create feature branch `feat/sota-throughput` off `develop`.
- [x] **Step 1 — Florence-2 / Magi v3 Fast Primitives (`src/manga2kindle/cv/backends/_florence2_fast.py`):** Implement `FastNoRepeatNGramLogitsProcessor`, `supports_fast_florence2`, `compact_beam_search` (single batched sync per step + dynamic finished-page compaction + non-4-tuple KV fallback), and `Florence2FastRunner` (`prepare_cpu`, `slice_prepared`, `infer_gpu`, `decode_cpu`).
- [x] **Step 2 — 3-Stage Pipelined `MagiBackend` (`src/manga2kindle/cv/backends/magi.py`):** Add free-VRAM-aware `resolve_magi_batch_size` (up to 32), CUDA TF32 + `sdpa`/`eager` fallback, separate `_processor_lock`, overlapped `_detect_batch_fast` and `_detect_batch_legacy`, and `workers` parameter.
- [x] **Step 3 — Registry Kwarg Forwarding (`src/manga2kindle/cv/backends/__init__.py`):** Generalize `_supports_on_progress` to `_supports_kwarg` and forward `workers` into `detect_batch` / `detect_batch_and_sort`.
- [x] **Step 4 — Pre-Load RAM Sampling & Ultra Concurrency (`src/manga2kindle/pipeline.py`):** Sample `_available_ram_gib()` before `_resolve_registry()`, raise ultra auto `batch_workers` cap to `6` (`>= 12 GiB`, `workers >= 8`), and forward `workers` to extraction and batch detection.
- [x] **Step 5 — Process Tree Cleanup & Parallel Preprocessing (`src/manga2kindle/platform.py`, `preprocessor.py`, `compiler/kfx.py`):** Implement `run_managed_subprocess` (Win32 Job Object / POSIX `killpg`), wire into KCC and Calibre runners cleanly, parallelize KCC output measurement and `PillowFallbackBackend` rendering.
- [x] **Step 6 — Fast Extraction & KPF Packaging (`src/manga2kindle/extractor.py`, `packaging/kpf.py`):** Parallelize image verification + zero-copy `os.replace` staging in `ArchiveExtractor`; switch KPF `res/*` entries to `ZIP_STORED`, bulk SQLite `executemany` with safe cursor closure.
- [x] **Step 7 — Unit & Integration Tests + Docs:** Add tests across `test_magi_backend.py`, `test_backend_registry.py`, `test_preprocessor.py`, `test_extractor.py`, and `test_kpf.py`; add spec/plan docs and update `README.md`.
- [x] **Step 8 — Full Verification & Merge to `develop`:** Run full `uv run pytest -q`, live 64-page GPU `MagiBackend` throughput + `mIoU` equivalence verification, isolated 3-volume end-to-end batch smoke test, commit cleanly on `feat/sota-throughput`, and merge into `develop`.
