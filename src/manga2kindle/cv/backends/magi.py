"""Layer 3 GPU SOTA panel detection backend powered by ragavsachdeva/magiv3."""

from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from manga2kindle.cv.backends._florence2_fast import (
    Florence2FastRunner,
    PreparedFlorenceBatch,
    supports_fast_florence2,
)
from manga2kindle.cv.backends.base import DetectionResult
from manga2kindle.cv.backends.contour import ContourBackend
from manga2kindle.cv.boxes import (
    DEFAULT_BLEED_MARGIN,
    Rect,
    compute_containment,
    finalize_panel_boxes,
    intersection_area,
)
from manga2kindle.cv.bubbles import expand_text_boxes_to_bubbles
from manga2kindle.cv.io import read_image_bgr
from manga2kindle.cv.models_hub import (
    DEFAULT_MAGI_MODEL_KEY,
    MagiModelSpec,
    get_cached_magi_dir,
    get_magi_model_spec,
    is_magi_cached,
    resolve_magi_model_spec,
)
from manga2kindle.models import DetectorConfig, DetectorKind, DevicePreference


_BYTES_PER_GIB = 1024**3
_VRAM_TIER_XLARGE_GIB = 20.0
_VRAM_TIER_LARGE_GIB = 15.0
_VRAM_TIER_MEDIUM_GIB = 12.0
_VRAM_TIER_SMALL_GIB = 8.0
_FREE_VRAM_TIER_XLARGE_GIB = 20.0
_FREE_VRAM_TIER_LARGE_GIB = 13.5
_FREE_VRAM_TIER_MEDIUM_GIB = 10.0
_FREE_VRAM_TIER_SMALL_GIB = 6.5
_BATCH_SIZE_XLARGE = 32
_BATCH_SIZE_LARGE = 16
_BATCH_SIZE_MEDIUM = 8
_BATCH_SIZE_SMALL = 4
_BATCH_SIZE_FALLBACK = 2
_MAGI_CPU_WORKERS_CAP = 8
_MAGI_PANELS_KEY = "panels"
_MAGI_TEXTS_KEY = "texts"
_MAGI_ESSENTIAL_TEXT_KEY = "is_essential_text"
_CONTOUR_RECOVERY_OVERLAP_THRESHOLD = 0.15
_ROW_ALIGNMENT_Y_RATIO = 0.5


def _batch_size_for_total_vram(total_gib: float) -> int:
    if total_gib >= _VRAM_TIER_XLARGE_GIB:
        return _BATCH_SIZE_XLARGE
    if total_gib >= _VRAM_TIER_LARGE_GIB:
        return _BATCH_SIZE_LARGE
    if total_gib >= _VRAM_TIER_MEDIUM_GIB:
        return _BATCH_SIZE_MEDIUM
    if total_gib >= _VRAM_TIER_SMALL_GIB:
        return _BATCH_SIZE_SMALL
    return _BATCH_SIZE_FALLBACK


def _batch_size_for_free_vram(free_gib: float) -> int:
    if free_gib >= _FREE_VRAM_TIER_XLARGE_GIB:
        return _BATCH_SIZE_XLARGE
    if free_gib >= _FREE_VRAM_TIER_LARGE_GIB:
        return _BATCH_SIZE_LARGE
    if free_gib >= _FREE_VRAM_TIER_MEDIUM_GIB:
        return _BATCH_SIZE_MEDIUM
    if free_gib >= _FREE_VRAM_TIER_SMALL_GIB:
        return _BATCH_SIZE_SMALL
    return _BATCH_SIZE_FALLBACK


def resolve_magi_batch_size(device: str, torch_mod: Any, configured: int | None) -> int:
    """Resolve Magi inference batch size from explicit config or total/free GPU VRAM."""
    if configured is not None:
        return max(1, configured)
    if device == "cuda":
        try:
            cuda_mod = getattr(torch_mod, "cuda", None)
            if cuda_mod is not None:
                props = cuda_mod.get_device_properties(0)
                total_gib = getattr(props, "total_memory", 0) / _BYTES_PER_GIB
                tier = _batch_size_for_total_vram(total_gib)
                mem_get_info = getattr(cuda_mod, "mem_get_info", None)
                if callable(mem_get_info):
                    try:
                        free_bytes, _ = mem_get_info(0)
                        if isinstance(free_bytes, (int, float)) and free_bytes > 0:
                            tier = min(tier, _batch_size_for_free_vram(free_bytes / _BYTES_PER_GIB))
                    except Exception:  # noqa: BLE001
                        pass
                return tier
        except Exception:  # noqa: BLE001
            return _BATCH_SIZE_FALLBACK
        return _BATCH_SIZE_FALLBACK
    if device == "mps":
        return _BATCH_SIZE_FALLBACK
    return 1


def _import_magi_deps() -> tuple[Any, Any]:
    """Import torch and transformers lazily so core installs do not require them."""
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "torch and transformers are required for MagiBackend. "
            "Install the Magi extra with: pip install manga2kindle[magi]"
        ) from exc
    return torch, transformers


def select_torch_device(device: DevicePreference, torch_module: Any) -> str:
    """Resolve DevicePreference to a concrete PyTorch device string ('cuda', 'mps', or 'cpu')."""
    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(torch_module.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())

    if device == DevicePreference.AUTO:
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    if device == DevicePreference.CUDA:
        if not cuda_available:
            raise RuntimeError("Requested device 'cuda' is unavailable in PyTorch.")
        return "cuda"

    if device == DevicePreference.MPS:
        if not mps_available:
            raise RuntimeError("Requested device 'mps' is unavailable in PyTorch.")
        return "mps"

    if device == DevicePreference.CPU:
        return "cpu"

    raise ValueError(f"Unsupported device preference: {device}")


def to_rgb_array(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or grayscale ndarray to an RGB uint8 ndarray."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Expected 2D grayscale or 3/4-channel BGR(A) image, got shape {image.shape}")


def _xyxy_to_rect(box: Sequence[float], page_w: int, page_h: int) -> Rect | None:
    """Convert [x1, y1, x2, y2] coordinates to a clamped (x, y, w, h) Rect, or None if degenerate."""
    if len(box) < 4:
        return None
    x1 = int(round(max(0.0, min(float(page_w), float(box[0])))))
    y1 = int(round(max(0.0, min(float(page_h), float(box[1])))))
    x2 = int(round(max(0.0, min(float(page_w), float(box[2])))))
    y2 = int(round(max(0.0, min(float(page_h), float(box[3])))))
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    return (x1, y1, w, h)


class MagiBackend:
    """GPU SOTA Magi panel detector with native reading-order prediction."""

    kind: DetectorKind = DetectorKind.MAGI

    def __init__(self, config: DetectorConfig) -> None:
        if not config.allow_remote_code:
            raise RuntimeError(
                "MagiBackend executes custom model code from ragavsachdeva/magiv3 "
                "(trust_remote_code=True). Pass --allow-remote-code (or set "
                "allow_remote_code=True) to opt in."
            )

        self._config = config
        self._lock = threading.Lock()
        self._processor_lock = threading.Lock()
        self._contour_fallback = ContourBackend(min_panel_area=config.min_panel_area)
        self._torch, transformers = _import_magi_deps()
        self._device = select_torch_device(config.device, self._torch)
        self._configure_torch_backends()
        self._batch_size = resolve_magi_batch_size(self._device, self._torch, config.batch_size)

        model_dir, spec = self._resolve_local_model_dir(config)
        auto_class_name = spec.auto_model_class if spec is not None else "AutoModelForCausalLM"
        uses_processor = spec.uses_processor if spec is not None else True

        torch_dtype = self._torch.float16 if self._device in ("cuda", "mps") else self._torch.float32
        model_cls = getattr(transformers, auto_class_name)
        hf_logging = getattr(getattr(transformers, "utils", None), "logging", None)
        prev_verbosity = hf_logging.get_verbosity() if hf_logging is not None else None
        if hf_logging is not None:
            hf_logging.set_verbosity_error()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
                warnings.filterwarnings("ignore", message=".*does not directly inherit from.*GenerationMixin.*")
                warnings.filterwarnings("ignore", message=".*use_reentrant=False.*")
                model = self._load_pretrained_model(
                    model_cls,
                    model_dir,
                    torch_dtype=torch_dtype,
                )
                self._model = model.to(self._device).eval()

                if uses_processor:
                    self._processor: Any = transformers.AutoProcessor.from_pretrained(
                        str(model_dir),
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                else:
                    self._processor = None
        finally:
            if hf_logging is not None and prev_verbosity is not None:
                hf_logging.set_verbosity(prev_verbosity)

        self._fast_runner: Florence2FastRunner | None = (
            Florence2FastRunner(self._model, self._processor, self._torch)
            if supports_fast_florence2(self._model, self._processor)
            else None
        )

    def _configure_torch_backends(self) -> None:
        if self._device != "cuda":
            return
        backends = getattr(self._torch, "backends", None)
        cuda_backend = getattr(backends, "cuda", None)
        matmul_backend = getattr(cuda_backend, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = True
        cudnn_backend = getattr(backends, "cudnn", None)
        if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = True

    @staticmethod
    def _load_pretrained_model(
        model_cls: Any,
        model_dir: Path,
        *,
        torch_dtype: Any,
    ) -> Any:
        for attn_impl in ("sdpa", "eager"):
            try:
                return model_cls.from_pretrained(
                    str(model_dir),
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl,
                )
            except (ValueError, NotImplementedError, AttributeError, RuntimeError):
                if attn_impl == "eager":
                    raise
        raise RuntimeError(f"Failed to load Magi model from {model_dir}")

    @staticmethod
    def _resolve_local_model_dir(config: DetectorConfig) -> tuple[Path, MagiModelSpec | None]:
        if config.model_path is not None:
            if not config.model_path.is_dir():
                raise FileNotFoundError(f"Magi model directory not found: {config.model_path}")
            return config.model_path, None

        raw_spec = config.model_id or DEFAULT_MAGI_MODEL_KEY
        local_path, model_key = resolve_magi_model_spec(raw_spec)
        if local_path is not None:
            if not local_path.is_dir():
                raise FileNotFoundError(f"Magi model directory not found: {local_path}")
            return local_path, None

        assert model_key is not None
        spec = get_magi_model_spec(model_key)
        cached_dir = get_cached_magi_dir(spec.key)
        if not is_magi_cached(spec.key):
            raise FileNotFoundError(
                f"Magi model '{spec.key}' is not cached at {cached_dir}. "
                f"Fetch it first with: manga2kindle models fetch {spec.key}"
            )
        return cached_dir, spec

    def detect(self, image_path: Path) -> DetectionResult:
        """Detect panels on a single page image while preserving Magi's native reading order."""
        return self.detect_batch([image_path])[0]

    def _infer_chunk(self, rgb_images: list[np.ndarray]) -> list[Any]:
        with self._lock, self._torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
            warnings.filterwarnings("ignore", message=".*use_reentrant=False.*")
            if self._processor is not None:
                return list(
                    self._model.predict_detections_and_associations(
                        rgb_images,
                        self._processor,
                    )
                )
            return list(self._model.predict_detections_and_associations(rgb_images))

    def _extract_essential_text_rects(
        self,
        page_result: Any,
        orig_w: int,
        orig_h: int,
    ) -> list[Rect]:
        if not isinstance(page_result, dict):
            return []
        raw_texts = page_result.get(_MAGI_TEXTS_KEY)
        if raw_texts is None:
            return []
        text_list = list(raw_texts)
        if not text_list:
            return []

        essential_mask = page_result.get(_MAGI_ESSENTIAL_TEXT_KEY)
        mask_list: list[Any] | None = None
        if essential_mask is not None:
            candidate_mask = list(essential_mask)
            if len(candidate_mask) == len(text_list):
                mask_list = candidate_mask

        rects: list[Rect] = []
        for idx, box in enumerate(text_list):
            if mask_list is not None and not bool(mask_list[idx]):
                continue
            rect = _xyxy_to_rect(box, orig_w, orig_h)
            if rect is not None:
                rects.append(rect)
        return rects

    def _extract_floating_non_essential_text_rects(
        self,
        page_result: Any,
        panel_rects: Sequence[Rect],
        orig_w: int,
        orig_h: int,
    ) -> list[Rect]:
        """Extract non-essential texts floating outside all panels (e.g. gutter narration)."""
        if not isinstance(page_result, dict) or not panel_rects:
            return []
        raw_texts = page_result.get(_MAGI_TEXTS_KEY)
        essential_mask = page_result.get(_MAGI_ESSENTIAL_TEXT_KEY)
        if raw_texts is None or essential_mask is None:
            return []
        text_list = list(raw_texts)
        mask_list = list(essential_mask)
        if not text_list or len(mask_list) != len(text_list):
            return []

        floating: list[Rect] = []
        overlap_limit = self._config.text_bridge_overlap
        for idx, box in enumerate(text_list):
            if bool(mask_list[idx]):
                continue
            rect = _xyxy_to_rect(box, orig_w, orig_h)
            if rect is None:
                continue
            area = rect[2] * rect[3]
            if area <= 0:
                continue
            if all((intersection_area(rect, p_rect) / area) < overlap_limit for p_rect in panel_rects):
                floating.append(rect)
        return floating

    def _recover_uncovered_contour_panels(
        self,
        magi_rects: list[Rect],
        rgb_image: np.ndarray,
    ) -> list[Rect]:
        """Recover clear contour-detected panels that Magi missed completely."""
        contour_rects = self._contour_fallback.detect_raw_rects(rgb_image, is_rgb=True)
        if not contour_rects:
            return magi_rects

        merged = list(magi_rects)
        for c_rect in contour_rects:
            c_area = c_rect[2] * c_rect[3]
            if c_area <= 0:
                continue
            overlaps_existing = any(
                compute_containment(c_rect, m_rect) >= _CONTOUR_RECOVERY_OVERLAP_THRESHOLD
                or (intersection_area(c_rect, m_rect) / c_area) >= _CONTOUR_RECOVERY_OVERLAP_THRESHOLD
                for m_rect in merged
            )
            if overlaps_existing:
                continue

            # Insert at a natural RTL reading-order position by vertical center, then right-to-left.
            c_cy = c_rect[1] + c_rect[3] / 2.0
            c_cx = c_rect[0] + c_rect[2] / 2.0
            insert_idx = len(merged)
            for idx, m_rect in enumerate(merged):
                m_cy = m_rect[1] + m_rect[3] / 2.0
                m_cx = m_rect[0] + m_rect[2] / 2.0
                row_tol = _ROW_ALIGNMENT_Y_RATIO * min(c_rect[3], m_rect[3])
                if c_cy < m_cy - row_tol or (
                    abs(c_cy - m_cy) <= row_tol and c_cx > m_cx
                ):
                    insert_idx = idx
                    break
            merged.insert(insert_idx, c_rect)
        return merged

    def _parse_page_result(
        self,
        page_result: Any,
        orig_w: int,
        orig_h: int,
        image: np.ndarray | None = None,
    ) -> DetectionResult:
        raw_panels = page_result.get(_MAGI_PANELS_KEY) if isinstance(page_result, dict) else None
        if raw_panels is None:
            raw_panels = []

        rects: list[Rect] = []
        for box in raw_panels:
            rect = _xyxy_to_rect(box, orig_w, orig_h)
            if rect is not None:
                rects.append(rect)

        if image is not None and rects and self._config.merge_bridged_panels:
            rects = self._recover_uncovered_contour_panels(rects, image)

        if not rects:
            return DetectionResult(
                panels=[],
                ordered=True,
                backend=self.kind,
                page_width=orig_w,
                page_height=orig_h,
                scores=None,
            )

        text_rects = (
            self._extract_essential_text_rects(page_result, orig_w, orig_h)
            if self._config.merge_bridged_panels
            else []
        )
        floating_margin_texts = (
            self._extract_floating_non_essential_text_rects(page_result, rects, orig_w, orig_h)
            if self._config.merge_bridged_panels
            else []
        )
        if image is not None and text_rects:
            text_rects = expand_text_boxes_to_bubbles(image, text_rects, is_rgb=True)

        panels, _ = finalize_panel_boxes(
            rects,
            orig_w,
            orig_h,
            text_boxes=text_rects,
            margin_text_boxes=floating_margin_texts,
            iou_threshold=self._config.iou_threshold,
            containment_threshold=self._config.containment_threshold,
            padding=self._config.padding,
            bleed_margin=DEFAULT_BLEED_MARGIN,
            preserve_order=True,
            merge_bridged_panels=self._config.merge_bridged_panels,
            text_bridge_overlap=self._config.text_bridge_overlap,
            panel_overlap_threshold=self._config.panel_overlap_threshold,
            max_merged_area_ratio=self._config.max_merged_area_ratio,
            min_fill_ratio=self._config.min_fill_ratio,
        )

        return DetectionResult(
            panels=panels,
            ordered=True,
            backend=self.kind,
            page_width=orig_w,
            page_height=orig_h,
            scores=None,
        )

    @staticmethod
    def _load_rgb_page(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        """Decode an image file from disk and convert it to RGB with (width, height)."""
        bgr_image = read_image_bgr(path)
        orig_h, orig_w = bgr_image.shape[:2]
        return to_rgb_array(bgr_image), (orig_w, orig_h)

    def _current_batch_size(self) -> int:
        with self._lock:
            return max(1, self._batch_size)

    def _pad_and_concat_ids(self, left_ids: Any, right_ids: Any) -> Any:
        max_len = max(int(left_ids.shape[1]), int(right_ids.shape[1]))
        tokenizer = getattr(self._processor, "tokenizer", None)
        pad_id = int(getattr(tokenizer, "pad_token_id", 1))
        if int(left_ids.shape[1]) < max_len:
            pad = self._torch.full(
                (int(left_ids.shape[0]), max_len - int(left_ids.shape[1])),
                pad_id,
                dtype=left_ids.dtype,
                device=left_ids.device,
            )
            left_ids = self._torch.cat([left_ids, pad], dim=1)
        if int(right_ids.shape[1]) < max_len:
            pad = self._torch.full(
                (int(right_ids.shape[0]), max_len - int(right_ids.shape[1])),
                pad_id,
                dtype=right_ids.dtype,
                device=right_ids.device,
            )
            right_ids = self._torch.cat([right_ids, pad], dim=1)
        return self._torch.cat([left_ids, right_ids], dim=0)

    def _infer_prepared_single_oom(
        self,
        prepared: PreparedFlorenceBatch,
        *,
        need_ess: bool,
    ) -> tuple[Any, list[list[bool]] | None]:
        assert self._fast_runner is not None
        try:
            with self._lock, warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
                warnings.filterwarnings("ignore", message=".*use_reentrant=False.*")
                return self._fast_runner.infer_gpu(prepared, need_essential_text=need_ess)
        except RuntimeError as exc:
            count = len(prepared.rgb_images)
            if "out of memory" not in str(exc).lower() or count <= 1:
                raise
            mid = count // 2
            with self._lock:
                cuda_mod = getattr(self._torch, "cuda", None)
                if cuda_mod is not None and hasattr(cuda_mod, "empty_cache"):
                    cuda_mod.empty_cache()
                self._batch_size = max(1, min(self._batch_size, mid))
            left_ids, left_ess = self._infer_prepared_single_oom(
                self._fast_runner.slice_prepared(prepared, 0, mid),
                need_ess=need_ess,
            )
            right_ids, right_ess = self._infer_prepared_single_oom(
                self._fast_runner.slice_prepared(prepared, mid, count),
                need_ess=need_ess,
            )
            combined_ess = (
                (left_ess + right_ess)
                if left_ess is not None and right_ess is not None
                else None
            )
            return self._pad_and_concat_ids(left_ids, right_ids), combined_ess

    def _infer_prepared_with_oom_fallback(
        self, prepared: PreparedFlorenceBatch
    ) -> tuple[Any, list[list[bool]] | None]:
        """Run fast Florence-2 GPU inference, respecting shrunken batch size and OOM halving."""
        assert self._fast_runner is not None
        need_ess = bool(self._config.merge_bridged_panels)
        total = len(prepared.rgb_images)
        step = self._current_batch_size()
        if total <= step:
            return self._infer_prepared_single_oom(prepared, need_ess=need_ess)

        all_ids: Any = None
        all_ess: list[list[bool]] | None = [] if need_ess else None
        for sub_start in range(0, total, step):
            sub_batch = self._fast_runner.slice_prepared(
                prepared, sub_start, min(total, sub_start + step)
            )
            sub_ids, sub_ess = self._infer_prepared_single_oom(sub_batch, need_ess=need_ess)
            all_ids = sub_ids if all_ids is None else self._pad_and_concat_ids(all_ids, sub_ids)
            if all_ess is not None and sub_ess is not None:
                all_ess.extend(sub_ess)
        return all_ids, all_ess

    def _infer_chunk_with_oom_fallback(self, rgb_images: list[np.ndarray]) -> list[Any]:
        """Run GPU inference, recursively halving batch size on CUDA OOM."""
        try:
            return self._infer_chunk(rgb_images)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(rgb_images) == 1:
                raise
            mid = len(rgb_images) // 2
            with self._lock:
                cuda_mod = getattr(self._torch, "cuda", None)
                if cuda_mod is not None and hasattr(cuda_mod, "empty_cache"):
                    cuda_mod.empty_cache()
                self._batch_size = max(1, min(self._batch_size, mid))
            return (
                self._infer_chunk_with_oom_fallback(rgb_images[:mid])
                + self._infer_chunk_with_oom_fallback(rgb_images[mid:])
            )

    def _detect_batch_fast(
        self,
        image_paths: Sequence[Path],
        *,
        cpu_workers: int,
        on_progress: Callable[[int, int], None] | None,
    ) -> list[DetectionResult]:
        assert self._fast_runner is not None
        runner = self._fast_runner
        total_pages = len(image_paths)
        results: list[DetectionResult] = []

        with (
            ThreadPoolExecutor(max_workers=cpu_workers) as pool,
            ThreadPoolExecutor(max_workers=1) as prep_pool,
            ThreadPoolExecutor(max_workers=1) as post_pool,
        ):
            def _prep_chunk(
                paths: Sequence[Path],
            ) -> tuple[PreparedFlorenceBatch, list[tuple[int, int]]]:
                load_futs = [pool.submit(self._load_rgb_page, p) for p in paths]
                loaded = [f.result() for f in load_futs]
                rgb_images = [item[0] for item in loaded]
                dims = [item[1] for item in loaded]
                with self._processor_lock:
                    prepared = runner.prepare_cpu(rgb_images)
                return prepared, dims

            def _post_chunk(
                gen_cpu: Any,
                rgb_images: list[np.ndarray],
                ess_masks: list[list[bool]] | None,
                dims: list[tuple[int, int]],
            ) -> list[DetectionResult]:
                with self._processor_lock:
                    raw_chunk = runner.decode_cpu(gen_cpu, rgb_images, ess_masks)
                parse_futs = [
                    pool.submit(
                        self._parse_page_result,
                        raw_chunk[idx] if idx < len(raw_chunk) else {},
                        orig_w,
                        orig_h,
                        rgb_images[idx],
                    )
                    for idx, (orig_w, orig_h) in enumerate(dims)
                ]
                parsed = [f.result() for f in parse_futs]
                rgb_images.clear()
                return parsed

            def _drain_post(pending: Future[list[DetectionResult]] | None) -> None:
                if pending is None:
                    return
                results.extend(pending.result())
                if on_progress is not None:
                    on_progress(len(results), total_pages)

            start = 0
            first_end = min(total_pages, self._current_batch_size())
            prep_future = prep_pool.submit(_prep_chunk, image_paths[start:first_end])
            pending_post: Future[list[DetectionResult]] | None = None

            while start < total_pages:
                prepared, dims = prep_future.result()
                next_start = start + len(dims)
                if next_start < total_pages:
                    next_end = min(total_pages, next_start + self._current_batch_size())
                    prep_future = prep_pool.submit(_prep_chunk, image_paths[next_start:next_end])

                gen_cpu, ess_masks = self._infer_prepared_with_oom_fallback(prepared)
                _drain_post(pending_post)
                pending_post = post_pool.submit(
                    _post_chunk,
                    gen_cpu,
                    prepared.rgb_images,
                    ess_masks,
                    dims,
                )
                start = next_start

            _drain_post(pending_post)

        return results

    def _detect_batch_legacy(
        self,
        image_paths: Sequence[Path],
        *,
        cpu_workers: int,
        on_progress: Callable[[int, int], None] | None,
    ) -> list[DetectionResult]:
        total_pages = len(image_paths)
        results: list[DetectionResult] = []

        with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
            def _submit_load(
                paths: Sequence[Path],
            ) -> list[Future[tuple[np.ndarray, tuple[int, int]]]]:
                return [pool.submit(self._load_rgb_page, path) for path in paths]

            def _drain_parse(
                pending: tuple[list[Future[DetectionResult]], list[np.ndarray], list[Any]] | None,
            ) -> None:
                if pending is None:
                    return
                parse_futs, old_rgbs, old_loaded = pending
                results.extend(fut.result() for fut in parse_futs)
                old_rgbs.clear()
                old_loaded.clear()
                if on_progress is not None:
                    on_progress(len(results), total_pages)

            start = 0
            first_end = min(total_pages, self._current_batch_size())
            current_futures = _submit_load(image_paths[start:first_end])
            pending_parse: (
                tuple[list[Future[DetectionResult]], list[np.ndarray], list[Any]] | None
            ) = None

            while start < total_pages:
                loaded = [fut.result() for fut in current_futures]
                rgb_images = [item[0] for item in loaded]
                dims = [item[1] for item in loaded]

                next_start = start + len(loaded)
                if next_start < total_pages:
                    next_end = min(total_pages, next_start + self._current_batch_size())
                    current_futures = _submit_load(image_paths[next_start:next_end])
                else:
                    current_futures = []

                step = self._current_batch_size()
                if len(rgb_images) <= step:
                    raw_chunk = self._infer_chunk_with_oom_fallback(rgb_images)
                else:
                    raw_chunk = []
                    for sub_start in range(0, len(rgb_images), step):
                        raw_chunk.extend(
                            self._infer_chunk_with_oom_fallback(
                                rgb_images[sub_start : sub_start + step]
                            )
                        )

                _drain_parse(pending_parse)
                parse_futures = [
                    pool.submit(
                        self._parse_page_result,
                        raw_chunk[idx] if idx < len(raw_chunk) else {},
                        orig_w,
                        orig_h,
                        rgb_images[idx],
                    )
                    for idx, (orig_w, orig_h) in enumerate(dims)
                ]
                pending_parse = (parse_futures, rgb_images, loaded)
                start = next_start

            _drain_parse(pending_parse)

        return results

    def detect_batch(
        self,
        image_paths: Sequence[Path],
        *,
        on_progress: Callable[[int, int], None] | None = None,
        workers: int | None = None,
    ) -> list[DetectionResult]:
        """Detect panels across a sequence of page images in VRAM-tuned GPU batches."""
        if not image_paths:
            return []

        total_pages = len(image_paths)
        if total_pages == 1:
            rgb, (orig_w, orig_h) = self._load_rgb_page(image_paths[0])
            if self._fast_runner is not None:
                with self._processor_lock:
                    prepared = self._fast_runner.prepare_cpu([rgb])
                gen_cpu, ess_masks = self._infer_prepared_with_oom_fallback(prepared)
                with self._processor_lock:
                    raw_chunk = self._fast_runner.decode_cpu(gen_cpu, [rgb], ess_masks)
            else:
                raw_chunk = self._infer_chunk_with_oom_fallback([rgb])
            page_result = raw_chunk[0] if raw_chunk else {}
            parsed = self._parse_page_result(page_result, orig_w, orig_h, image=rgb)
            if on_progress is not None:
                on_progress(1, 1)
            return [parsed]

        default_workers = min(os.cpu_count() or 4, _MAGI_CPU_WORKERS_CAP)
        cpu_workers = min(total_pages, max(1, workers) if workers is not None else default_workers)

        if self._fast_runner is not None:
            return self._detect_batch_fast(
                image_paths,
                cpu_workers=cpu_workers,
                on_progress=on_progress,
            )
        return self._detect_batch_legacy(
            image_paths,
            cpu_workers=cpu_workers,
            on_progress=on_progress,
        )
