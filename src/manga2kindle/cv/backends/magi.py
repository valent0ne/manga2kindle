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
from manga2kindle.cv.boxes import (
    DEFAULT_BLEED_MARGIN,
    Rect,
    nms_and_prune_indices,
    pad_and_clamp,
)
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
        self._torch, transformers = _import_magi_deps()
        self._device = select_torch_device(config.device, self._torch)

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
        bgr_image = read_image_bgr(image_path)
        orig_h, orig_w = bgr_image.shape[:2]
        rgb_image = to_rgb_array(bgr_image)

        with self._lock, self._torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
            warnings.filterwarnings("ignore", message=".*use_reentrant=False.*")
            if self._processor is not None:
                raw_results = self._model.predict_detections_and_associations(
                    [rgb_image],
                    self._processor,
                )
            else:
                raw_results = self._model.predict_detections_and_associations([rgb_image])

        page_result = raw_results[0] if raw_results else {}
        raw_panels = page_result.get("panels", []) if isinstance(page_result, dict) else []

        rects: list[Rect] = []
        for box in raw_panels:
            rect = _xyxy_to_rect(box, orig_w, orig_h)
            if rect is not None:
                rects.append(rect)

        if not rects:
            return DetectionResult(
                panels=[],
                ordered=True,
                backend=self.kind,
                page_width=orig_w,
                page_height=orig_h,
                scores=None,
            )

        surviving = nms_and_prune_indices(
            rects,
            iou_threshold=self._config.iou_threshold,
            containment_threshold=self._config.containment_threshold,
            preserve_order=True,
        )

        panels = [
            pad_and_clamp(
                rects[idx],
                orig_w,
                orig_h,
                padding=self._config.padding,
                bleed_margin=DEFAULT_BLEED_MARGIN,
            ).model_copy(update={"ordinal": ordinal})
            for ordinal, idx in enumerate(surviving, start=1)
        ]

        return DetectionResult(
            panels=panels,
            ordered=True,
            backend=self.kind,
            page_width=orig_w,
            page_height=orig_h,
            scores=None,
        )

    def detect_batch(self, image_paths: Sequence[Path]) -> list[DetectionResult]:
        """Detect panels across a sequence of page images, preserving page order."""
        return [self.detect(path) for path in image_paths]
