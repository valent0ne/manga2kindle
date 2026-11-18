"""Domain models and configuration schemas for the manga2kindle pipeline."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class ReadingDirection(str, Enum):
    """Supported panel and page reading directions."""

    RTL = "rtl"
    LTR = "ltr"


class SpreadMode(str, Enum):
    """How double-page landscape spreads are handled during preprocessing."""

    SPLIT = "split"
    ROTATE = "rotate"
    KEEP = "keep"


class DetectorKind(str, Enum):
    """Supported panel detection backend identifiers."""

    AUTO = "auto"
    CONTOUR = "contour"
    ONNX_YOLO = "onnx"
    MAGI = "magi"


class DevicePreference(str, Enum):
    """Preferred inference compute device for neural panel detectors."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


DEFAULT_COVER_SIZE: tuple[int, int] = (1600, 2560)


_DEVICE_PRESETS: dict[str, tuple[int, int, str]] = {
    "KPW5": (1264, 1680, "KPW5"),
    "KS": (1860, 2480, "KS"),
    "OASIS": (1264, 1680, "KO"),
}


class DeviceProfile(BaseModel):
    """Target Kindle display dimensions and KCC profile identifier."""

    name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    kcc_code: str

    @classmethod
    def from_name(
        cls,
        name: str,
        custom_size: tuple[int, int] | None = None,
    ) -> DeviceProfile:
        """Build a DeviceProfile from a preset name or CUSTOM dimensions."""
        normalized = name.strip().upper()
        if normalized == "CUSTOM":
            if custom_size is None:
                raise ValueError("custom_size=(width, height) is required for CUSTOM profile")
            width, height = custom_size
            if width <= 0 or height <= 0:
                raise ValueError("custom_size dimensions must be positive integers")
            return cls(name="CUSTOM", width=width, height=height, kcc_code="OTHER")

        preset = _DEVICE_PRESETS.get(normalized)
        if preset is None:
            supported = ", ".join([*_DEVICE_PRESETS.keys(), "CUSTOM"])
            raise ValueError(f"Unknown device profile '{name}'. Supported: {supported}")

        width, height, kcc_code = preset
        return cls(name=normalized, width=width, height=height, kcc_code=kcc_code)


class PanelBox(BaseModel):
    """Detected manga panel bounding box clamped within page bounds."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    ordinal: int = Field(default=0, ge=0)


class ZoomRect(BaseModel):
    """KPF magnification target rectangle computed from a PanelBox and zoom factor."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class PageAsset(BaseModel):
    """Represents a single manga page across extraction, preprocessing, and detection."""

    index: int = Field(ge=0)
    source_path: Path
    processed_path: Path | None = None
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    is_cover: bool = False
    is_spread: bool = False
    panels: list[PanelBox] = Field(default_factory=list)


class DetectorConfig(BaseModel):
    """Configuration for panel detection backends and post-processing."""

    kind: DetectorKind = DetectorKind.AUTO
    model_path: Path | None = None
    model_id: str | None = None
    device: DevicePreference = DevicePreference.AUTO
    score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    containment_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    padding: float = Field(default=0.02, ge=0.0, le=0.25)
    min_panel_area: float = Field(default=0.03, ge=0.01, le=0.50)
    batch_size: int | None = Field(default=None, ge=1, le=64)
    allow_remote_code: bool = False


class ConversionConfig(BaseModel):
    """Runtime configuration for converting a CBZ/CBR/image folder into KFX."""

    input_path: Path
    output_path: Path
    profile: DeviceProfile
    direction: ReadingDirection = ReadingDirection.RTL
    spread_mode: SpreadMode = SpreadMode.SPLIT
    zoom_factor: float = Field(default=1.5, ge=1.0)
    padding: float = Field(default=0.02, ge=0.0, le=0.25)
    min_panel_area: float = Field(default=0.03, ge=0.01, le=0.50)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    no_kcc: bool = False
    keep_temp: bool = False
    skip_existing: bool = False
    beautify: bool = False
    beautify_source: bool = False
    workers: int = Field(default_factory=lambda: os.cpu_count() or 4, ge=1)
    batch_workers: int | None = Field(default=None, ge=1)
    kcc_path: Path | None = None
    calibre_path: Path | None = None
    cover_size: tuple[int, int] = Field(default=DEFAULT_COVER_SIZE)

    @property
    def effective_beautify(self) -> bool:
        """Return True when output filenames and metadata should be beautified."""
        return self.beautify or self.beautify_source

    @model_validator(mode="after")
    def _sync_detector_fields(self) -> ConversionConfig:
        """Keep top-level padding/min_panel_area synchronized with detector config."""
        for field_name in ("padding", "min_panel_area"):
            top_explicit = field_name in self.model_fields_set
            det_explicit = field_name in self.detector.model_fields_set
            if top_explicit and not det_explicit:
                setattr(self.detector, field_name, getattr(self, field_name))
            elif det_explicit and not top_explicit:
                setattr(self, field_name, getattr(self.detector, field_name))
        return self

    @field_validator("cover_size")
    @classmethod
    def _validate_cover_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        width, height = value
        if width <= 0 or height <= 0:
            raise ValueError("cover_size dimensions must be positive integers")
        return value
