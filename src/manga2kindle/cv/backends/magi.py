"""Layer 3 GPU SOTA panel detection backend powered by ragavsachdeva/magiv3."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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
_MAGI_PANELS_KEY = "panels"
_MAGI_TEXTS_KEY = "texts"
_MAGI_ESSENTIAL_TEXT_KEY = "is_essential_text"
_CONTOUR_RECOVERY_OVERLAP_THRESHOLD = 0.15
_ROW_ALIGNMENT_Y_RATIO = 0.5


def resolve_magi_batch_size(device: str, torch_mod: Any, configured: int | None) -> int:
    """Resolve Magi inference batch size from explicit config or available GPU VRAM."""
    if configured is not None:
        return max(1, configured)
    if device == "cuda":
        try:
            cuda_mod = getattr(torch_mod, "cuda", None)
            if cuda_mod is not None:
                props = cuda_mod.get_device_properties(0)
                total_gib = getattr(props, "total_memory", 0) / _BYTES_PER_GIB
                if total_gib >= 12.0:
                    return 8
                if total_gib >= 8.0:
                    return 4
                return 2
        except Exception:  # noqa: BLE001
            return 2
        return 2
    if device == "mps":
        return 2
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
        self._contour_fallback = ContourBackend(min_panel_area=config.min_panel_area)
        self._torch, transformers = _import_magi_deps()
        self._device = select_torch_device(config.device, self._torch)
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
                model = model_cls.from_pretrained(
                    str(model_dir),
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
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
        if image is not None and text_rects:
            text_rects = expand_text_boxes_to_bubbles(image, text_rects, is_rgb=True)

        panels, _ = finalize_panel_boxes(
            rects,
            orig_w,
            orig_h,
            text_boxes=text_rects,
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

    def detect_batch(self, image_paths: Sequence[Path]) -> list[DetectionResult]:
        """Detect panels across a sequence of page images in VRAM-tuned GPU batches."""
        if not image_paths:
            return []

        results: list[DetectionResult] = []
        step = max(1, self._batch_size)
        for start in range(0, len(image_paths), step):
            chunk_paths = image_paths[start : start + step]
            rgb_images: list[np.ndarray] = []
            dims: list[tuple[int, int]] = []
            for path in chunk_paths:
                bgr_image = read_image_bgr(path)
                orig_h, orig_w = bgr_image.shape[:2]
                dims.append((orig_w, orig_h))
                rgb_images.append(to_rgb_array(bgr_image))

            try:
                raw_chunk = self._infer_chunk(rgb_images)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or len(rgb_images) == 1:
                    raise
                cuda_mod = getattr(self._torch, "cuda", None)
                if cuda_mod is not None and hasattr(cuda_mod, "empty_cache"):
                    cuda_mod.empty_cache()
                raw_chunk = [self._infer_chunk([img])[0] for img in rgb_images]

            for idx, (orig_w, orig_h) in enumerate(dims):
                page_result = raw_chunk[idx] if idx < len(raw_chunk) else {}
                results.append(
                    self._parse_page_result(
                        page_result,
                        orig_w,
                        orig_h,
                        image=rgb_images[idx],
                    )
                )

        return results
