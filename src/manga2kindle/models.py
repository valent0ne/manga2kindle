"""Domain models and configuration schemas for the manga2kindle pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
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


class DeviceSizeTier(str, Enum):
    """Screen-size tier used to calibrate Guided View panel merging thresholds."""

    COMPACT = "compact"
    STANDARD = "standard"
    LARGE = "large"


DEFAULT_COVER_SIZE: tuple[int, int] = (1600, 2560)
DEFAULT_PROFILE_NAME: str = "KPW6"
COMPACT_MAX_SHORT_EDGE: int = 1150
LARGE_MIN_SHORT_EDGE: int = 1600


@dataclass(frozen=True)
class _TierMergePreset:
    min_tier_panels: int
    max_tier_height_ratio: float
    max_tier_panel_width_ratio: float
    max_tier_panel_area_ratio: float
    min_subcol_panels: int
    max_subcol_width_ratio: float
    max_subcol_height_ratio: float
    max_subcol_panel_area_ratio: float
    max_external_overlap_ratio: float
    max_bridged_area_ratio: float


_PANEL_MERGE_TIER_PRESETS: dict[DeviceSizeTier, _TierMergePreset] = {
    DeviceSizeTier.COMPACT: _TierMergePreset(
        min_tier_panels=3,
        max_tier_height_ratio=0.32,
        max_tier_panel_width_ratio=0.45,
        max_tier_panel_area_ratio=0.15,
        min_subcol_panels=2,
        max_subcol_width_ratio=0.40,
        max_subcol_height_ratio=0.48,
        max_subcol_panel_area_ratio=0.12,
        max_external_overlap_ratio=0.15,
        max_bridged_area_ratio=0.45,
    ),
    DeviceSizeTier.STANDARD: _TierMergePreset(
        min_tier_panels=3,
        max_tier_height_ratio=0.37,
        max_tier_panel_width_ratio=0.55,
        max_tier_panel_area_ratio=0.20,
        min_subcol_panels=2,
        max_subcol_width_ratio=0.45,
        max_subcol_height_ratio=0.55,
        max_subcol_panel_area_ratio=0.15,
        max_external_overlap_ratio=0.15,
        max_bridged_area_ratio=0.55,
    ),
    DeviceSizeTier.LARGE: _TierMergePreset(
        min_tier_panels=3,
        max_tier_height_ratio=0.46,
        max_tier_panel_width_ratio=0.60,
        max_tier_panel_area_ratio=0.25,
        min_subcol_panels=2,
        max_subcol_width_ratio=0.50,
        max_subcol_height_ratio=0.65,
        max_subcol_panel_area_ratio=0.20,
        max_external_overlap_ratio=0.15,
        max_bridged_area_ratio=0.70,
    ),
}


_DEVICE_PRESETS: dict[str, tuple[int, int, str]] = {
    "KPW6": (1264, 1680, "KPW6"),
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

    @property
    def size_tier(self) -> DeviceSizeTier:
        """Classify display size tier from the shorter display edge."""
        short_edge = min(self.width, self.height)
        if short_edge < COMPACT_MAX_SHORT_EDGE:
            return DeviceSizeTier.COMPACT
        if short_edge >= LARGE_MIN_SHORT_EDGE:
            return DeviceSizeTier.LARGE
        return DeviceSizeTier.STANDARD

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


DEFAULT_TEXT_BRIDGE_OVERLAP = 0.05
DEFAULT_PANEL_OVERLAP_THRESHOLD = 0.25
DEFAULT_MAX_MERGED_AREA_RATIO = 0.55
DEFAULT_MIN_FILL_RATIO = 0.55
DEFAULT_MIN_TEXT_AREA_RATIO = 0.0005


class DetectorConfig(BaseModel):
    """Configuration for panel detection backends and post-processing."""

    kind: DetectorKind = DetectorKind.AUTO
    model_path: Path | None = None
    model_id: str | None = None
    device: DevicePreference = DevicePreference.AUTO
    score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    containment_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    padding: float = Field(default=0.04, ge=0.0, le=0.25)
    min_panel_area: float = Field(default=0.03, ge=0.01, le=0.50)
    batch_size: int | None = Field(default=None, ge=1, le=64)
    allow_remote_code: bool = False
    merge_bridged_panels: bool = True
    text_bridge_overlap: float = Field(default=DEFAULT_TEXT_BRIDGE_OVERLAP, gt=0.0, le=1.0)
    panel_overlap_threshold: float = Field(default=DEFAULT_PANEL_OVERLAP_THRESHOLD, gt=0.0, le=1.0)
    max_merged_area_ratio: float = Field(default=DEFAULT_MAX_MERGED_AREA_RATIO, gt=0.0, le=1.0)
    min_fill_ratio: float = Field(default=DEFAULT_MIN_FILL_RATIO, gt=0.0, le=1.0)


class PanelMergeConfig(BaseModel):
    """Profile-calibrated rules for merging compact mini-panel rows and stacked sub-columns."""

    enabled: bool = True
    min_tier_panels: int = Field(default=3, ge=2, le=12)
    max_tier_height_ratio: float = Field(default=0.37, gt=0.0, le=1.0)
    max_tier_panel_width_ratio: float = Field(default=0.55, gt=0.0, le=1.0)
    max_tier_panel_area_ratio: float = Field(default=0.20, gt=0.0, le=1.0)
    min_subcol_panels: int = Field(default=2, ge=2, le=12)
    max_subcol_width_ratio: float = Field(default=0.45, gt=0.0, le=1.0)
    max_subcol_height_ratio: float = Field(default=0.55, gt=0.0, le=1.0)
    max_subcol_panel_area_ratio: float = Field(default=0.15, gt=0.0, le=1.0)
    max_external_overlap_ratio: float = Field(default=0.15, ge=0.0, le=1.0)

    @classmethod
    def for_tier(cls, tier: DeviceSizeTier, *, enabled: bool = True) -> PanelMergeConfig:
        """Return calibrated merge thresholds for a DeviceSizeTier."""
        preset = _PANEL_MERGE_TIER_PRESETS[tier]
        preset_values = {
            f.name: getattr(preset, f.name)
            for f in fields(preset)
            if f.name in cls.model_fields
        }
        return cls(enabled=enabled, **preset_values)

    @classmethod
    def for_profile(cls, profile: DeviceProfile, *, enabled: bool = True) -> PanelMergeConfig:
        """Return calibrated merge thresholds for a target DeviceProfile."""
        return cls.for_tier(profile.size_tier, enabled=enabled)


class ConversionConfig(BaseModel):
    """Runtime configuration for converting a CBZ/CBR/image folder into KFX."""

    input_path: Path
    output_path: Path
    profile: DeviceProfile
    direction: ReadingDirection = ReadingDirection.RTL
    spread_mode: SpreadMode = SpreadMode.SPLIT
    padding: float = Field(default=0.04, ge=0.0, le=0.25)
    min_panel_area: float = Field(default=0.03, ge=0.01, le=0.50)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    panel_merge: PanelMergeConfig = Field(default_factory=PanelMergeConfig)
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

    @model_validator(mode="after")
    def _sync_panel_merge_for_profile(self) -> ConversionConfig:
        """Calibrate unset panel_merge and detector bridge thresholds from the active DeviceProfile."""
        explicit = self.panel_merge.model_fields_set
        effective_enabled = (
            self.detector.merge_bridged_panels
            if "enabled" not in explicit and "merge_bridged_panels" in self.detector.model_fields_set
            else self.panel_merge.enabled
        )
        calibrated = PanelMergeConfig.for_profile(
            self.profile,
            enabled=effective_enabled,
        )
        updates: dict[str, object] = {
            field_info.name: getattr(calibrated, field_info.name)
            for field_info in fields(_TierMergePreset)
            if field_info.name in PanelMergeConfig.model_fields and field_info.name not in explicit
        }
        if "enabled" not in explicit:
            updates["enabled"] = calibrated.enabled
        if updates:
            self.panel_merge = self.panel_merge.model_copy(update=updates)

        if "max_merged_area_ratio" not in self.detector.model_fields_set:
            preset = _PANEL_MERGE_TIER_PRESETS[self.profile.size_tier]
            self.detector = self.detector.model_copy(
                update={"max_merged_area_ratio": preset.max_bridged_area_ratio}
            )
        return self

    @field_validator("cover_size")
    @classmethod
    def _validate_cover_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        width, height = value
        if width <= 0 or height <= 0:
            raise ValueError("cover_size dimensions must be positive integers")
        return value
