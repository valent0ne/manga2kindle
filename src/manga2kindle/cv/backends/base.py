"""Protocol and result types for pluggable manga panel detection backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from manga2kindle.models import DetectorKind, PanelBox


@dataclass
class DetectionResult:
    """Output returned by a panel detector backend for a single page image."""

    panels: list[PanelBox]
    ordered: bool
    backend: DetectorKind
    page_width: int = 0
    page_height: int = 0
    scores: list[float] | None = None


@runtime_checkable
class PanelDetectorBackend(Protocol):
    """Interface implemented by all panel detection backends."""

    kind: DetectorKind

    def detect(self, image_path: Path) -> DetectionResult:
        """Detect panels on a single page image."""
        ...

    def detect_batch(self, image_paths: Sequence[Path]) -> list[DetectionResult]:
        """Detect panels across a sequence of page images, preserving order."""
        ...
