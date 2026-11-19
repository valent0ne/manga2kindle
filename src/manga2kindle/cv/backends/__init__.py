"""Pluggable panel detection backends, AUTO fallback chain, and session registry."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path

from manga2kindle.cv.backends.base import DetectionResult, PanelDetectorBackend
from manga2kindle.cv.backends.contour import ContourBackend
from manga2kindle.cv.backends.magi import MagiBackend
from manga2kindle.cv.backends.onnx_yolo import OnnxYoloBackend, select_ort_providers
from manga2kindle.cv.io import read_image_bgr
from manga2kindle.cv.models_hub import (
    DEFAULT_MAGI_MODEL_KEY,
    DEFAULT_ONNX_MODEL_KEY,
    is_magi_cached,
    is_model_cached,
    resolve_detector_model_target,
    resolve_magi_model_spec,
)
from manga2kindle.cv.sorter import PanelSorter
from manga2kindle.models import (
    DetectorConfig,
    DetectorKind,
    DevicePreference,
    PanelBox,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BackendRegistry",
    "ContourBackend",
    "DetectionResult",
    "MagiBackend",
    "OnnxYoloBackend",
    "PanelDetectorBackend",
    "describe_auto_chain",
    "detect_with_fallback",
    "infer_config_model_target",
    "resolve_auto_chain",
]


def _build_contour_backend(config: DetectorConfig) -> ContourBackend:
    return ContourBackend(
        min_panel_area=config.min_panel_area,
        padding=config.padding,
        iou_threshold=config.iou_threshold,
        containment_threshold=config.containment_threshold,
        merge_bridged_panels=config.merge_bridged_panels,
        panel_overlap_threshold=config.panel_overlap_threshold,
        max_merged_area_ratio=config.max_merged_area_ratio,
        min_fill_ratio=config.min_fill_ratio,
    )


def _probe_magi_readiness(config: DetectorConfig) -> tuple[bool, str]:
    """Return (True, 'ready') if Magi deps, GPU, and local weights are all available."""
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError:
        return False, "magi dependencies not installed"

    cuda_ok = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps_backend is not None and mps_backend.is_available())

    if config.device == DevicePreference.CPU:
        return False, "device=cpu disables GPU Magi auto-selection"
    if config.device == DevicePreference.CUDA and not cuda_ok:
        return False, "CUDA unavailable"
    if config.device == DevicePreference.MPS and not mps_ok:
        return False, "MPS unavailable"
    if config.device == DevicePreference.AUTO and not (cuda_ok or mps_ok):
        return False, "no GPU available"

    if config.model_path is not None:
        has_weights = config.model_path.is_dir()
    else:
        raw_spec = config.model_id or DEFAULT_MAGI_MODEL_KEY
        try:
            local_path, model_key = resolve_magi_model_spec(raw_spec)
        except ValueError:
            return False, "invalid magi model spec"
        if local_path is not None:
            has_weights = local_path.is_dir()
        elif model_key is not None:
            try:
                has_weights = is_magi_cached(model_key)
            except (ValueError, RuntimeError):
                has_weights = False
        else:
            has_weights = False

    if not has_weights:
        return False, "magi weights not cached"

    return True, "ready"


def infer_config_model_target(config: DetectorConfig) -> DetectorKind | None:
    """Infer the explicit backend targeted by config.model_path or config.model_id, if any."""
    if config.model_path is not None:
        if not config.model_path.exists():
            raise FileNotFoundError(f"Detector model path not found: {config.model_path}")
        return DetectorKind.MAGI if config.model_path.is_dir() else DetectorKind.ONNX_YOLO
    if config.model_id is not None:
        return resolve_detector_model_target(config.model_id).backend
    return None


def describe_auto_chain(config: DetectorConfig | None = None) -> tuple[list[DetectorKind], str]:
    """Describe the AUTO fallback chain and any readiness note without side-effect warnings or model loads."""
    cfg = config or DetectorConfig()
    chain: list[DetectorKind] = []
    note = ""

    magi_ready, _reason = _probe_magi_readiness(cfg)
    if magi_ready:
        if cfg.allow_remote_code:
            chain.append(DetectorKind.MAGI)
        else:
            note = "magi ready with --allow-remote-code"

    try:
        import onnxruntime as ort

        select_ort_providers(cfg.device, ort.get_available_providers())
        if is_model_cached(cfg.model_id or DEFAULT_ONNX_MODEL_KEY):
            chain.append(DetectorKind.ONNX_YOLO)
    except (ImportError, RuntimeError, ValueError):
        pass

    chain.append(DetectorKind.CONTOUR)
    return chain, note


def resolve_auto_chain(config: DetectorConfig) -> list[PanelDetectorBackend]:
    """Resolve an ordered list of instantiated panel detector backends for config.kind."""
    if config.kind == DetectorKind.CONTOUR:
        return [_build_contour_backend(config)]
    if config.kind == DetectorKind.ONNX_YOLO:
        return [OnnxYoloBackend(config)]
    if config.kind == DetectorKind.MAGI:
        return [MagiBackend(config)]

    backends: list[PanelDetectorBackend] = []
    explicit_target = infer_config_model_target(config)
    if explicit_target == DetectorKind.MAGI:
        backends = [MagiBackend(config), _build_contour_backend(config)]
        logger.info(
            "Resolved AUTO panel detector chain (active=%s): %s",
            backends[0].kind.value,
            [backend.kind.value for backend in backends],
        )
        return backends
    if explicit_target == DetectorKind.ONNX_YOLO:
        backends = [OnnxYoloBackend(config), _build_contour_backend(config)]
        logger.info(
            "Resolved AUTO panel detector chain (active=%s): %s",
            backends[0].kind.value,
            [backend.kind.value for backend in backends],
        )
        return backends

    magi_ready, _reason = _probe_magi_readiness(config)
    if magi_ready:
        if not config.allow_remote_code:
            logger.warning(
                "Skipping Magi backend in AUTO mode: GPU and cached weights are available, "
                "but allow_remote_code=False. Pass --allow-remote-code to enable Magi."
            )
        else:
            try:
                backends.append(MagiBackend(config))
            except (RuntimeError, FileNotFoundError, OSError, ImportError, ValueError) as exc:
                logger.warning("Magi backend failed to initialize in AUTO mode; falling back: %s", exc)

    try:
        backends.append(OnnxYoloBackend(config))
    except (RuntimeError, FileNotFoundError, OSError, ImportError, ValueError) as exc:
        logger.debug("ONNX YOLO backend unavailable in AUTO mode; falling back: %s", exc)

    backends.append(_build_contour_backend(config))
    logger.info(
        "Resolved AUTO panel detector chain (active=%s): %s",
        backends[0].kind.value,
        [backend.kind.value for backend in backends],
    )
    return backends


def detect_with_fallback(
    image_path: Path,
    backends: Sequence[PanelDetectorBackend],
    *,
    disabled_backends: set[DetectorKind] | None = None,
    lock: threading.Lock | None = None,
) -> DetectionResult:
    """Run candidate backends in order, disabling load-failed backends once and falling back on 0 panels."""
    last_w = 0
    last_h = 0

    for backend in backends:
        if disabled_backends is not None:
            if lock is not None:
                with lock:
                    is_disabled = backend.kind in disabled_backends
            else:
                is_disabled = backend.kind in disabled_backends
            if is_disabled:
                continue

        try:
            result = backend.detect(image_path)
        except (RuntimeError, FileNotFoundError, OSError, ImportError, ValueError) as exc:
            if disabled_backends is not None:
                if lock is not None:
                    with lock:
                        should_log = backend.kind not in disabled_backends
                        disabled_backends.add(backend.kind)
                else:
                    should_log = backend.kind not in disabled_backends
                    disabled_backends.add(backend.kind)
                if should_log:
                    logger.warning(
                        "Backend '%s' failed during detection and is disabled for this session: %s",
                        backend.kind.value,
                        exc,
                    )
            continue

        if result.page_width > 0 and result.page_height > 0:
            last_w = result.page_width
            last_h = result.page_height

        if result.panels:
            return result

    if last_w <= 0 or last_h <= 0:
        img = read_image_bgr(image_path)
        last_h, last_w = img.shape[:2]

    return DetectionResult(
        panels=[PanelBox(x=0, y=0, w=last_w, h=last_h, ordinal=1)],
        ordered=True,
        backend=DetectorKind.CONTOUR,
        page_width=last_w,
        page_height=last_h,
    )


class BackendRegistry:
    """Thread-safe session cache for resolved panel detection backends."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        *,
        backends: Sequence[PanelDetectorBackend] | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self._lock = threading.Lock()
        self._backends: list[PanelDetectorBackend] | None = (
            list(backends) if backends is not None else None
        )
        self._disabled: set[DetectorKind] = set()

    def resolve(self) -> list[PanelDetectorBackend]:
        """Resolve and cache the backend chain once per registry instance."""
        if self._backends is None:
            with self._lock:
                if self._backends is None:
                    self._backends = resolve_auto_chain(self.config)
        return self._backends

    @property
    def active_backend(self) -> PanelDetectorBackend | None:
        """Return the highest-priority non-disabled backend, if any."""
        backends = self.resolve()
        with self._lock:
            for backend in backends:
                if backend.kind not in self._disabled:
                    return backend
        return None

    def detect(self, image_path: Path) -> DetectionResult:
        """Detect panels on a single page using the cached backend chain and fallback."""
        return detect_with_fallback(
            image_path,
            self.resolve(),
            disabled_backends=self._disabled,
            lock=self._lock,
        )

    @staticmethod
    def _sort_result(
        result: DetectionResult,
        sorter: PanelSorter,
        *,
        page_width: int = 0,
        is_spread: bool = False,
    ) -> list[PanelBox]:
        if result.ordered:
            return [
                panel.model_copy(update={"ordinal": idx}) if panel.ordinal <= 0 else panel
                for idx, panel in enumerate(result.panels, start=1)
            ]
        effective_width = page_width if page_width > 0 else result.page_width
        return sorter.sort(
            result.panels,
            page_width=effective_width,
            is_spread=is_spread,
        )

    def detect_and_sort(
        self,
        image_path: Path,
        sorter: PanelSorter,
        *,
        page_width: int = 0,
        is_spread: bool = False,
    ) -> list[PanelBox]:
        """Detect panels and apply PanelSorter only when the backend did not already order them."""
        result = self.detect(image_path)
        return self._sort_result(
            result,
            sorter,
            page_width=page_width,
            is_spread=is_spread,
        )

    def detect_batch(self, image_paths: Sequence[Path]) -> list[DetectionResult]:
        """Detect panels across a batch of pages, falling back per-session on error or per-page on empty."""
        if not image_paths:
            return []

        backends = self.resolve()
        active_backends: list[PanelDetectorBackend]
        with self._lock:
            active_backends = [b for b in backends if b.kind not in self._disabled]

        for idx, backend in enumerate(active_backends):
            try:
                batch_results = list(backend.detect_batch(image_paths))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Backend %s failed during batch detection (%s); disabling for this session.",
                    backend.kind.value,
                    exc,
                )
                with self._lock:
                    self._disabled.add(backend.kind)
                continue

            if len(batch_results) != len(image_paths):
                logger.warning(
                    "Backend %s returned %d results for %d pages; falling back per-page.",
                    backend.kind.value,
                    len(batch_results),
                    len(image_paths),
                )
                return [self.detect(path) for path in image_paths]

            remaining = active_backends[idx + 1 :]
            resolved: list[DetectionResult] = []
            for path, res in zip(image_paths, batch_results):
                if res.panels:
                    resolved.append(res)
                else:
                    resolved.append(
                        detect_with_fallback(
                            path,
                            remaining,
                            disabled_backends=self._disabled,
                            lock=self._lock,
                        )
                    )
            return resolved

        return [self.detect(path) for path in image_paths]

    def detect_batch_and_sort(
        self,
        image_paths: Sequence[Path],
        sorter: PanelSorter,
        *,
        page_widths: Sequence[int] | None = None,
        is_spreads: Sequence[bool] | None = None,
    ) -> list[list[PanelBox]]:
        """Detect and sort panels for a sequence of pages using batched backend inference."""
        results = self.detect_batch(image_paths)
        widths = page_widths if page_widths is not None else [0] * len(results)
        spreads = is_spreads if is_spreads is not None else [False] * len(results)
        return [
            self._sort_result(
                res,
                sorter,
                page_width=widths[idx] if idx < len(widths) else 0,
                is_spread=spreads[idx] if idx < len(spreads) else False,
            )
            for idx, res in enumerate(results)
        ]
