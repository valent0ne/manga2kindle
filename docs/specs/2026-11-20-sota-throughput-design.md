# SOTA End-to-End Throughput Design

**Date:** 2026-11-20
**Status:** Approved

## Overview

Eliminate GPU, CPU, I/O, and Windows/POSIX process-lifecycle bottlenecks across `manga2kindle` so the standard single-command batch CLI (`manga2kindle convert ... --detector magi --detector-model magiv3 -w <cores>`) saturates modern multi-core CPUs and CUDA/MPS GPUs simultaneously without changing panel detection quality (`float16`, `num_beams=3`, exact n-gram constraint, identical panel merging/sorting semantics) or increasing peak VRAM/commit instability.

## Root Causes Diagnosed

Profiling an 8-core/16-thread host (`31.16 GiB` RAM, RTX 5070 Ti `16 GiB` VRAM) converting `.cbz` volumes revealed five stacked bottlenecks:

1. **Windows commit-charge self-throttling in `ConversionPipeline`:** Both `run_batch()` and `preview_batch()` queried `available_memory_gib()` *after* `_resolve_registry(config.detector)` loaded `magiv3` + PyTorch CUDA contexts (consuming ~6–7 GiB of Windows virtual commit charge), causing `_resolve_batch_workers` to underestimate available RAM and throttle concurrent books on high-core systems.
2. **Orphaned KCC `multiprocessing.spawn` worker trees on Windows:** Running external subprocesses (`kcc-c2e`, `calibre-debug`) via raw `subprocess.run` left spawned worker descendants alive if the CLI was cancelled or timed out, silently leaking RAM and Windows commit charge.
3. **Unoptimized HF Florence-2 beam decode in `MagiBackend` (`magiv3`):**
   - HF `NoRepeatNGramLogitsProcessor` executed 27,648+ `.tolist()` GPU-to-CPU syncs (~47.2% of decode time).
   - `BeamSearchScorer.process` performed per-beam `.item()` syncs (~27.1% of remaining decode time), and finished pages kept running through decoder steps until the slowest straggler finished.
   - Redundant encoder/image passes (`lm.generate` re-ran vision encoding or affinity heads unused for panel extraction), eager attention instead of PyTorch SDPA, and no overlapped CPU image decoding / tokenization / post-processing around GPU inference.
4. **Sequential CPU/IO pipeline stages:** Archive extraction verified images sequentially and copied ephemeral extracted files instead of renaming them; `KccBackend` measured processed image dimensions sequentially; `PillowFallbackBackend` processed pages serially; `KpfBuilder` deflated already-compressed JPEG/PNG images (`ZIP_DEFLATED`) and executed per-row SQLite inserts under default `DELETE` journaling.
5. **Conservative VRAM and batch concurrency ceilings:** Auto `batch_workers` capped at `4` even when `available >= 12 GiB` and `workers >= 8`, and auto `--gpu-batch-size` only inspected total VRAM up to `16` without accounting for 24 GB+ GPUs (`32`) or currently free VRAM via `torch.cuda.mem_get_info`.

## Architecture & Changes

### 1. Accelerated Florence-2 / Magi v3 Engine (`src/manga2kindle/cv/backends/_florence2_fast.py`)

Activated only when `supports_fast_florence2(model, processor)` confirms concrete Florence-2 / Magi v3 contracts (so `magiv2`, non-Florence models, and unit test mocks stay on `predict_detections_and_associations`):

- **`FastNoRepeatNGramLogitsProcessor`:** Vectorized `unfold` + `scatter_reduce_(..., reduce="amin", include_self=True)` n-gram banning on-device with cached `(-inf, +inf)` scalars, plus an exact `nonzero(as_tuple=True)` fallback for Apple Silicon MPS or backends lacking `scatter_reduce_`.
- **`compact_beam_search`:** Runs Florence-2's `language_model` with a single batched D2H/H2D transfer per decode step and dynamically compacts finished pages out of `input_ids`, `enc_hidden`, `enc_mask`, and Florence-2 4-tensor `past_key_values` (gracefully falling back to `lm.generate` if the model uses a non-4-tuple KV structure).
- **`Florence2FastRunner`:** Splits detection into three explicit stages (`prepare_cpu`, `infer_gpu`, `decode_cpu`):
  - Reuses image features + `encoder_outputs` for `get_essential_text_logits` only when `merge_bridged_panels=True`.
  - Runs `processor.postprocess_output` inside `torch.inference_mode()` because Magi v3's processor performs in-place `masked_fill_` on CPU `generated_ids`.
  - Supports zero-copy batch slicing (`slice_prepared`) for CUDA OOM recursive halving without re-tokenizing.

### 2. Three-Stage Pipelined `MagiBackend` (`src/manga2kindle/cv/backends/magi.py`, `__init__.py`)

- **Free- + total-VRAM batch sizing (`resolve_magi_batch_size`):** Scales up to `32` (`>= 20 GiB` total/free), `16` (`>= 15 GiB` total / `>= 13.5 GiB` free), `8` (`>= 12 GiB` / `>= 10 GiB`), `4` (`>= 8 GiB` / `>= 6.5 GiB`), or `2`.
- **Backend acceleration:** Enables CUDA TF32 (`matmul.allow_tf32`, `cudnn.allow_tf32`), loads with `attn_implementation="sdpa"` and automatically falls back to `"eager"` on unsupported custom remote code, keeps `float16` on CUDA/MPS, and separates `_processor_lock` from GPU `_lock`.
- **Overlapped CPU/GPU execution:** `_detect_batch_fast` (and overlapped `_detect_batch_legacy`) runs parallel OpenCV image decoding + tokenization for chunk `N+1` while chunk `N` executes on GPU and chunk `N-1` decodes/merges panels on CPU threads. Forwards optional `workers` via `BackendRegistry._supports_kwarg`.

### 3. Pre-Load RAM Sampling & Batch Scaling (`src/manga2kindle/pipeline.py`)

- `run_batch()` and `preview_batch()` sample `_available_ram_gib()` *before* `_resolve_registry(config.detector)` so model commit charge does not suppress `batch_workers`.
- Adds an ultra concurrency tier (`available_ram_gib >= 12.0` and `workers >= 8`) scaling auto `batch_workers` up to `6` concurrent books (while `KccBackend._semaphore` continues capping simultaneous `kcc-c2e` subprocesses cleanly).
- Forwards `workers` to `ArchiveExtractor.extract(...)` and `BackendRegistry.detect_batch_and_sort(...)`.

### 4. Managed Process Trees & Parallel CPU/IO Stages

- **Managed subprocess trees (`src/manga2kindle/platform.py`, `preprocessor.py`, `compiler/kfx.py`):** `run_managed_subprocess` binds external child trees to a Win32 Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) on Windows and an isolated POSIX process group (`start_new_session=True` + `SIGTERM`/`SIGKILL` via `os.killpg`) on macOS/Linux. Both `KccBackend` and `KfxCompiler` retain the `_DEFAULT_SUBPROCESS_RUN` guard so monkeypatched tests remain compatible.
- **Parallel preprocessing (`src/manga2kindle/preprocessor.py`):** `KccBackend` measures unpacked CBZ images in parallel across `config.workers`; `PillowFallbackBackend` processes pages across `ThreadPoolExecutor(config.workers)` using thread-local E-Ink palettes and deterministic temp-then-rename indexing.
- **Parallel extraction & zero-copy staging (`src/manga2kindle/extractor.py`):** Verifies/measures candidate images across `ThreadPoolExecutor(workers)` and uses `os.replace` (`move_sources=True`) when staging ephemeral `raw_*` archive extracts (preserving `shutil.copy2` for user directories).
- **Fast KPF packaging (`src/manga2kindle/packaging/kpf.py`):** Writes `res/*` media entries with `zipfile.ZIP_STORED` (preserving `ZIP_DEFLATED` for `book.kdf` and `book.kcb`), bulk-inserts SQLite fragments via `executemany` with `PRAGMA synchronous = OFF` and `PRAGMA journal_mode = MEMORY`, and explicitly closes SQLite cursors before closing connections (preventing Windows file-handle locks).

## Trade-Offs Rejected

- **Multi-replica GPU workers (threads with CUDA streams or separate subprocesses):** Benchmarked empirically on RTX 5070 Ti (`magiv3`, 32 real pages): 2 threads + CUDA streams yielded `3.65 pg/s` vs `3.60 pg/s` (+1.4%, blocked by GIL in `transformers.generation`); 2 subprocesses regressed to `3.41 pg/s` due to WDDM CUDA context switching and doubled VRAM/commit footprint. Rejected in favor of single-process Florence-2 fast decoding.
- **`bfloat16` precision for `magiv3`:** Benchmarked slower (`3.34 pg/s` vs `3.62 pg/s` in `float16`) and altered detections (`22/32` exact vs baseline vs `32/32` for `float16` + SDPA). Kept `float16`.

## Measured Results

Verified on the Windows 8C/16T + RTX 5070 Ti (16 GiB) host:

| Benchmark / Verification Check | Baseline | After Overhaul | Notes |
| --- | --- | --- | --- |
| Live 64-page `MagiBackend` (`magiv3`, CUDA, `bs=16`, `workers=8`) | `2.58 pg/s` | **`8.03 pg/s`** (`7.974s`) | `3.11x` speedup; peak `6.40 GiB` alloc / `7.71 GiB` reserved |
| 16-page panel detection & reading-order equivalence | — | **`16/16` exact count, `1.000000` mIoU** | 47/47 panel pairs bit/pixel-identical (`min_iou=1.000000`) |
| 3-volume `Ichi the Witch` CLI batch smoke test (`-w 16`, `magiv3`, KCC + Calibre KFX) | — | **`106.82s` total (`3/3` succeeded)** | `batch_workers=(3, 5)`, `0` new orphaned Python processes (`106 -> 106`) |
| Full `pytest` suite (`uv run pytest -q`) | `302 passed` | **`311 passed` (`30.79s`)** | 0 failures / errors |
