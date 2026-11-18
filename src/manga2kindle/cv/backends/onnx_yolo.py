"""ONNX Runtime YOLO backend (Layer 2) for manga panel detection."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from manga2kindle.cv.backends.base import DetectionResult
from manga2kindle.cv.boxes import (
    DEFAULT_BLEED_MARGIN,
    Rect,
    finalize_panel_boxes,
    nms_and_prune,
)
from manga2kindle.cv.bubbles import expand_text_boxes_to_bubbles
from manga2kindle.cv.io import read_image_bgr
from manga2kindle.cv.models_hub import (
    DEFAULT_ONNX_MODEL_KEY,
    MANGA109_FRAME_CLASS_ID,
    get_cached_model_path,
    get_model_spec,
    resolve_model_spec,
)
from manga2kindle.models import DetectorConfig, DetectorKind, DevicePreference

PANEL_CLASS_ID: int = MANGA109_FRAME_CLASS_ID
DEFAULT_INPUT_SIZE: int = 640
_LETTERBOX_FILL: tuple[int, int, int] = (114, 114, 114)

_AUTO_PROVIDER_ORDER: tuple[str, ...] = (
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)

_EXPLICIT_PROVIDER_MAP: dict[DevicePreference, str] = {
    DevicePreference.CPU: "CPUExecutionProvider",
    DevicePreference.CUDA: "CUDAExecutionProvider",
    DevicePreference.MPS: "CoreMLExecutionProvider",
}


def _import_onnxruntime() -> Any:
    """Import onnxruntime lazily so core installs do not require it."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for OnnxYoloBackend. "
            "Install the ONNX extra with: pip install manga2kindle[onnx]"
        ) from exc
    return ort


def select_ort_providers(
    device: DevicePreference,
    available_providers: Sequence[str],
) -> list[str]:
    """Select ONNX Runtime execution providers matching the requested DevicePreference."""
    available = set(available_providers)

    if device == DevicePreference.AUTO:
        selected = [p for p in _AUTO_PROVIDER_ORDER if p in available]
        if not selected:
            raise RuntimeError(
                "No supported ONNX Runtime execution provider is available "
                f"(saw: {list(available_providers)})"
            )
        return selected

    required = _EXPLICIT_PROVIDER_MAP.get(device)
    if required is None:
        raise ValueError(f"Unsupported device preference: {device}")
    if required not in available:
        raise RuntimeError(
            f"Requested execution provider '{required}' for device '{device.value}' "
            f"is unavailable (available: {list(available_providers)})"
        )

    if required != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        return [required, "CPUExecutionProvider"]
    return [required]


def letterbox_image(
    image_bgr: np.ndarray,
    target_size: int = DEFAULT_INPUT_SIZE,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and pad a BGR image to a square NCHW float32 RGB tensor in [0, 1].

    Returns ``(tensor, scale, (pad_left, pad_top))``.
    """
    if image_bgr.ndim < 2 or image_bgr.shape[0] <= 0 or image_bgr.shape[1] <= 0:
        raise ValueError(f"Expected non-empty 2D/3D image, got shape {image_bgr.shape}")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(target_size / orig_h, target_size / orig_w)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)

    dw = target_size - new_w
    dh = target_size - new_h
    pad_left = dw // 2
    pad_top = dh // 2
    pad_right = dw - pad_left
    pad_bottom = dh - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=_LETTERBOX_FILL,
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    nchw = np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)
    return nchw, scale, (pad_left, pad_top)


def _is_nms_exported_format(arr_2d: np.ndarray) -> bool:
    """Distinguish (N, 6) NMS-exported outputs from (4 + C, N) raw YOLO outputs."""
    if arr_2d.shape[1] != 6:
        return False
    if arr_2d.shape[0] != 8:
        return True
    # Ambiguous (8, 6) shape: check whether column 5 holds integer class IDs and column 4 holds scores.
    scores_valid = bool(np.all((arr_2d[:, 4] >= 0.0) & (arr_2d[:, 4] <= 1.0)))
    cls_int = bool(np.all((arr_2d[:, 5] >= 0.0) & (arr_2d[:, 5] == np.floor(arr_2d[:, 5]))))
    boxes_ordered = bool(np.all((arr_2d[:, 2] >= arr_2d[:, 0]) & (arr_2d[:, 3] >= arr_2d[:, 1])))
    return scores_valid and cls_int and boxes_ordered


def _unletterbox_and_clamp(
    x1_lb: float,
    y1_lb: float,
    x2_lb: float,
    y2_lb: float,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> Rect | None:
    """Map letterboxed pixel coordinates back to clamped (x, y, w, h) source Rect."""
    x1 = (x1_lb - pad_x) / scale
    y1 = (y1_lb - pad_y) / scale
    x2 = (x2_lb - pad_x) / scale
    y2 = (y2_lb - pad_y) / scale

    ix1 = int(round(max(0.0, min(float(orig_w), x1))))
    iy1 = int(round(max(0.0, min(float(orig_h), y1))))
    ix2 = int(round(max(0.0, min(float(orig_w), x2))))
    iy2 = int(round(max(0.0, min(float(orig_h), y2))))

    w = ix2 - ix1
    h = iy2 - iy1
    if w <= 0 or h <= 0:
        return None
    return (ix1, iy1, w, h)


@dataclass(frozen=True)
class YoloDetections:
    """Decoded panel and optional text detections from a YOLO ONNX output tensor."""

    panel_rects: list[Rect]
    panel_scores: list[float]
    text_rects: list[Rect]
    text_scores: list[float]


@dataclass(frozen=True)
class _ResolvedOnnxModel:
    """Resolved local ONNX model path and class/input metadata."""

    model_path: Path
    panel_class_id: int
    text_class_id: int | None
    input_size: int


def decode_yolo_detections(
    raw_output: np.ndarray,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad: tuple[int, int],
    score_threshold: float,
    panel_class_id: int = PANEL_CLASS_ID,
    text_class_id: int | None = None,
) -> YoloDetections:
    """Decode raw YOLOv8/v11 (1, 4+C, N) or NMS-exported (1, N, 6) ONNX output."""
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"orig_w and orig_h must be positive, got ({orig_w}, {orig_h})")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if panel_class_id < 0:
        raise ValueError(f"panel_class_id must be non-negative, got {panel_class_id}")
    if text_class_id is not None and text_class_id < 0:
        raise ValueError(f"text_class_id must be non-negative, got {text_class_id}")

    arr = np.asarray(raw_output, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {arr.shape}")
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D YOLO output tensor, got shape {arr.shape}")

    pad_x, pad_y = pad
    panel_rects: list[Rect] = []
    panel_scores: list[float] = []
    text_rects: list[Rect] = []
    text_scores: list[float] = []

    if _is_nms_exported_format(arr):
        for row in arr:
            x1_lb, y1_lb, x2_lb, y2_lb, score, cls_val = row
            cls_id = int(round(float(cls_val)))
            conf = float(score)
            if conf < score_threshold:
                continue
            if cls_id == panel_class_id:
                rect = _unletterbox_and_clamp(
                    float(x1_lb), float(y1_lb), float(x2_lb), float(y2_lb),
                    orig_w, orig_h, scale, pad_x, pad_y,
                )
                if rect is not None:
                    panel_rects.append(rect)
                    panel_scores.append(conf)
            elif text_class_id is not None and cls_id == text_class_id:
                rect = _unletterbox_and_clamp(
                    float(x1_lb), float(y1_lb), float(x2_lb), float(y2_lb),
                    orig_w, orig_h, scale, pad_x, pad_y,
                )
                if rect is not None:
                    text_rects.append(rect)
                    text_scores.append(conf)
        return YoloDetections(panel_rects, panel_scores, text_rects, text_scores)

    if arr.shape[0] < 5:
        raise ValueError(f"Expected at least 5 channels (4 box + >=1 class), got shape {arr.shape}")
    if 4 + panel_class_id >= arr.shape[0]:
        raise ValueError(
            f"panel_class_id={panel_class_id} out of range for tensor shape {arr.shape}"
        )

    extract_text = text_class_id is not None and (4 + text_class_id) < arr.shape[0]
    predictions = arr.T  # (N, 4 + num_classes)
    for pred in predictions:
        class_scores = pred[4:]
        best_cls = int(np.argmax(class_scores))
        is_panel = best_cls == panel_class_id
        is_text = extract_text and best_cls == text_class_id
        if not (is_panel or is_text):
            continue
        conf = float(class_scores[best_cls])
        if conf < score_threshold:
            continue

        cx, cy, bw, bh = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
        if bw <= 0.0 or bh <= 0.0:
            continue

        half_w = bw / 2.0
        half_h = bh / 2.0
        rect = _unletterbox_and_clamp(
            cx - half_w, cy - half_h, cx + half_w, cy + half_h,
            orig_w, orig_h, scale, pad_x, pad_y,
        )
        if rect is None:
            continue
        if is_panel:
            panel_rects.append(rect)
            panel_scores.append(conf)
        else:
            text_rects.append(rect)
            text_scores.append(conf)

    return YoloDetections(panel_rects, panel_scores, text_rects, text_scores)


def decode_yolo_output(
    raw_output: np.ndarray,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad: tuple[int, int],
    score_threshold: float,
    panel_class_id: int = PANEL_CLASS_ID,
) -> tuple[list[Rect], list[float]]:
    """Decode raw YOLOv8/v11 (1, 4+C, N) or NMS-exported (1, N, 6) ONNX output."""
    det = decode_yolo_detections(
        raw_output,
        orig_w=orig_w,
        orig_h=orig_h,
        scale=scale,
        pad=pad,
        score_threshold=score_threshold,
        panel_class_id=panel_class_id,
        text_class_id=None,
    )
    return det.panel_rects, det.panel_scores


class OnnxYoloBackend:
    """CPU/EP-accelerated ONNX YOLO panel detector using local cached weights only."""

    kind: DetectorKind = DetectorKind.ONNX_YOLO

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        resolved = self._resolve_local_model(config)
        self._model_path = resolved.model_path
        self._panel_class_id = resolved.panel_class_id
        self._text_class_id = resolved.text_class_id

        ort = _import_onnxruntime()
        providers = select_ort_providers(config.device, ort.get_available_providers())
        self._session = ort.InferenceSession(str(self._model_path), providers=providers)

        input_meta = self._session.get_inputs()[0]
        self._input_name: str = input_meta.name
        self._input_size: int = self._resolve_input_size(input_meta.shape, resolved.input_size)

    @staticmethod
    def _resolve_local_model(config: DetectorConfig) -> _ResolvedOnnxModel:
        if config.model_path is not None:
            if not config.model_path.is_file():
                raise FileNotFoundError(f"ONNX model file not found: {config.model_path}")
            return _ResolvedOnnxModel(
                model_path=config.model_path,
                panel_class_id=PANEL_CLASS_ID,
                text_class_id=None,
                input_size=DEFAULT_INPUT_SIZE,
            )

        raw_spec = config.model_id or DEFAULT_ONNX_MODEL_KEY
        local_path, model_key = resolve_model_spec(raw_spec)
        if local_path is not None:
            if not local_path.is_file():
                raise FileNotFoundError(f"ONNX model file not found: {local_path}")
            return _ResolvedOnnxModel(
                model_path=local_path,
                panel_class_id=PANEL_CLASS_ID,
                text_class_id=None,
                input_size=DEFAULT_INPUT_SIZE,
            )

        assert model_key is not None
        spec = get_model_spec(model_key)
        cached_path = get_cached_model_path(spec.key)
        if not cached_path.is_file():
            raise FileNotFoundError(
                f"ONNX model '{spec.key}' is not cached at {cached_path}. "
                f"Fetch it first with: manga2kindle models fetch {spec.key}"
            )
        return _ResolvedOnnxModel(
            model_path=cached_path,
            panel_class_id=spec.panel_class_id,
            text_class_id=spec.text_class_id,
            input_size=spec.input_size,
        )

    @staticmethod
    def _resolve_input_size(shape: Sequence[Any], fallback: int) -> int:
        if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0:
            return int(shape[2])
        return fallback

    def detect(self, image_path: Path) -> DetectionResult:
        """Detect manga panels on a single page image using ONNX Runtime."""
        image = read_image_bgr(image_path)
        orig_h, orig_w = image.shape[:2]
        tensor, scale, pad = letterbox_image(image, target_size=self._input_size)

        with self._lock:
            outputs = self._session.run(None, {self._input_name: tensor})

        raw_output = np.asarray(outputs[0])
        active_text_class_id = (
            self._text_class_id if self._config.merge_bridged_panels else None
        )
        det = decode_yolo_detections(
            raw_output,
            orig_w=orig_w,
            orig_h=orig_h,
            scale=scale,
            pad=pad,
            score_threshold=self._config.score_threshold,
            panel_class_id=self._panel_class_id,
            text_class_id=active_text_class_id,
        )

        pruned_text_rects = (
            nms_and_prune(
                det.text_rects,
                iou_threshold=self._config.iou_threshold,
                containment_threshold=self._config.containment_threshold,
                scores=det.text_scores,
            )
            if self._config.merge_bridged_panels and det.text_rects
            else []
        )
        if pruned_text_rects:
            pruned_text_rects = expand_text_boxes_to_bubbles(image, pruned_text_rects)

        panels, kept_scores = finalize_panel_boxes(
            det.panel_rects,
            orig_w,
            orig_h,
            text_boxes=pruned_text_rects,
            scores=det.panel_scores,
            iou_threshold=self._config.iou_threshold,
            containment_threshold=self._config.containment_threshold,
            padding=self._config.padding,
            bleed_margin=DEFAULT_BLEED_MARGIN,
            preserve_order=False,
            merge_bridged_panels=self._config.merge_bridged_panels,
            text_bridge_overlap=self._config.text_bridge_overlap,
            panel_overlap_threshold=self._config.panel_overlap_threshold,
            max_merged_area_ratio=self._config.max_merged_area_ratio,
            min_fill_ratio=self._config.min_fill_ratio,
        )

        return DetectionResult(
            panels=panels,
            ordered=False,
            backend=self.kind,
            page_width=orig_w,
            page_height=orig_h,
            scores=kept_scores,
        )

    def detect_batch(self, image_paths: Sequence[Path]) -> list[DetectionResult]:
        """Detect panels across a sequence of page images, preserving order."""
        return [self.detect(path) for path in image_paths]
