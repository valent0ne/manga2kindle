from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from manga2kindle.models import (
    DEFAULT_PROFILE_NAME,
    ConversionConfig,
    DetectorConfig,
    DeviceProfile,
    DeviceSizeTier,
    PageAsset,
    PanelBox,
    PanelMergeConfig,
    ReadingDirection,
    SpreadMode,
)
from manga2kindle.platform import PlatformDiscovery, ToolStatus


def test_device_profile_presets():
    assert DEFAULT_PROFILE_NAME == "KPW6"
    kpw6 = DeviceProfile.from_name("KPW6")
    assert (kpw6.width, kpw6.height, kpw6.kcc_code) == (1264, 1680, "KPW6")
    kpw5 = DeviceProfile.from_name("KPW5")
    assert (kpw5.width, kpw5.height, kpw5.kcc_code) == (1264, 1680, "KPW5")
    ks = DeviceProfile.from_name("KS")
    assert (ks.width, ks.height, ks.kcc_code) == (1860, 2480, "KS")
    oasis = DeviceProfile.from_name("OASIS")
    assert (oasis.width, oasis.height, oasis.kcc_code) == (1264, 1680, "KO")
    custom = DeviceProfile.from_name("CUSTOM", custom_size=(1400, 1900))
    assert (custom.width, custom.height, custom.kcc_code) == (1400, 1900, "OTHER")
    with pytest.raises(ValueError):
        DeviceProfile.from_name("CUSTOM")
    with pytest.raises(ValueError):
        DeviceProfile.from_name("UNKNOWN")


def test_conversion_config_defaults_and_validators(tmp_path):
    profile = DeviceProfile.from_name("KPW5")
    cfg = ConversionConfig(
        input_path=tmp_path / "in.cbz",
        output_path=tmp_path / "out.kfx",
        profile=profile,
    )
    assert cfg.direction == ReadingDirection.RTL
    assert cfg.spread_mode == SpreadMode.SPLIT
    assert cfg.padding == 0.04
    assert cfg.min_panel_area == 0.03
    assert cfg.no_kcc is False
    assert cfg.keep_temp is False
    assert cfg.workers >= 1
    assert cfg.cover_size == (1600, 2560)

    with pytest.raises(ValidationError):
        ConversionConfig(
            input_path=tmp_path / "in.cbz",
            output_path=tmp_path / "out.kfx",
            profile=profile,
            cover_size=(0, 2560),
        )

    with pytest.raises(ValidationError):
        ConversionConfig(
            input_path=tmp_path / "in.cbz",
            output_path=tmp_path / "out.kfx",
            profile=profile,
            padding=-0.01,
        )
    with pytest.raises(ValidationError):
        ConversionConfig(
            input_path=tmp_path / "in.cbz",
            output_path=tmp_path / "out.kfx",
            profile=profile,
            padding=0.30,
        )
    with pytest.raises(ValidationError):
        ConversionConfig(
            input_path=tmp_path / "in.cbz",
            output_path=tmp_path / "out.kfx",
            profile=profile,
            min_panel_area=0.005,
        )
    with pytest.raises(ValidationError):
        ConversionConfig(
            input_path=tmp_path / "in.cbz",
            output_path=tmp_path / "out.kfx",
            profile=profile,
            min_panel_area=0.60,
        )
    with pytest.raises(ValidationError):
        ConversionConfig(
            input_path=tmp_path / "in.cbz",
            output_path=tmp_path / "out.kfx",
            profile=profile,
            workers=0,
        )


def test_panel_and_page_models(tmp_path):
    panel = PanelBox(x=10, y=20, w=300, h=400, ordinal=1)
    page = PageAsset(
        index=0,
        source_path=tmp_path / "0000.png",
        width=1264,
        height=1680,
        is_cover=True,
        panels=[panel],
    )
    assert panel.ordinal == 1
    assert page.is_cover is True
    assert len(page.panels) == 1


def test_platform_discovery_windows_candidates(tmp_path, monkeypatch):
    fake_calibre = tmp_path / "Calibre2" / "calibre-debug.exe"
    fake_calibre.parent.mkdir(parents=True)
    fake_calibre.write_text("stub")
    disc = PlatformDiscovery(system="win32")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "x86"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert disc.find_calibre() == fake_calibre


def test_platform_discovery_override_takes_precedence(tmp_path):
    custom = tmp_path / "custom-kcc"
    custom.write_text("stub")
    disc = PlatformDiscovery(system="darwin")
    assert disc.find_kcc(override=custom) == custom


def test_platform_discovery_kcc_windows(tmp_path, monkeypatch):
    bin_dir = tmp_path / "venv"
    bin_dir.mkdir(parents=True)
    fake_kcc = bin_dir / "kcc-c2e.exe"
    fake_kcc.write_text("stub")
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python.exe"))
    monkeypatch.setattr("shutil.which", lambda _: None)

    disc = PlatformDiscovery(system="win32")
    assert disc.find_kcc() == fake_kcc


def test_platform_discovery_unrar_which(tmp_path, monkeypatch):
    fake_unar = tmp_path / "unar"
    fake_unar.write_text("stub")
    monkeypatch.setattr(
        "shutil.which",
        lambda name: str(fake_unar) if name == "unar" else None,
    )
    disc = PlatformDiscovery(system="darwin")
    assert disc.find_unrar() == fake_unar


def test_platform_discovery_run_doctor(tmp_path, monkeypatch):
    kcc = tmp_path / "kcc-c2e"
    calibre = tmp_path / "calibre-debug"
    unrar = tmp_path / "unrar"
    for p in (kcc, calibre, unrar):
        p.write_text("stub")

    disc = PlatformDiscovery(system="darwin")
    monkeypatch.setattr(disc, "find_kcc", lambda override=None: kcc)
    monkeypatch.setattr(disc, "find_calibre", lambda override=None: calibre)
    monkeypatch.setattr(disc, "find_unrar", lambda: unrar)

    recorded_cmds: list[list[str]] = []
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if isinstance(cmd, str):
            return real_run(cmd, **kwargs)
        recorded_cmds.append(cmd)
        assert kwargs.get("encoding") == "utf-8"
        assert kwargs.get("errors") == "replace"
        if "KFX Output" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="KFX Output 2.19.0", stderr="")
        if "--help" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="comic2ebook v13.0.1 - ...", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="1.0.0", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    statuses = disc.run_doctor()
    by_name = {s.name: s for s in statuses}
    assert all(isinstance(s, ToolStatus) for s in statuses)
    assert by_name["kcc-c2e"].found is True
    assert by_name["kcc-c2e"].version == "comic2ebook v13.0.1 - ..."
    assert any(cmd[0] == str(kcc) and cmd[1] == "--help" for cmd in recorded_cmds)
    assert by_name["calibre-debug"].found is True
    assert by_name["KFX Output"].found is True
    assert by_name["unrar/unar/7z"].found is True


def test_platform_discovery_find_unrar_includes_7zz(tmp_path, monkeypatch):
    fake_7zz = tmp_path / "7zz"
    fake_7zz.write_text("stub")
    monkeypatch.setattr(
        "shutil.which",
        lambda name: str(fake_7zz) if name == "7zz" else None,
    )
    disc = PlatformDiscovery(system="darwin")
    assert disc.find_unrar() == fake_7zz


def test_platform_discovery_run_doctor_missing_tools(monkeypatch):
    disc = PlatformDiscovery(system="darwin")
    monkeypatch.setattr(disc, "find_kcc", lambda override=None: None)
    monkeypatch.setattr(disc, "find_calibre", lambda override=None: None)
    monkeypatch.setattr(disc, "find_unrar", lambda: None)

    statuses = disc.run_doctor()
    by_name = {s.name: s for s in statuses}
    assert by_name["kcc-c2e"].found is False
    assert by_name["calibre-debug"].found is False
    assert by_name["KFX Output"].found is False
    assert by_name["KFX Output"].details == "calibre-debug not found"
    assert by_name["unrar/unar/7z"].found is False


def test_device_profile_size_tier_and_panel_merge_calibration(tmp_path: Path) -> None:
    assert DeviceProfile.from_name("KPW6").size_tier == DeviceSizeTier.STANDARD
    assert DeviceProfile.from_name("KPW5").size_tier == DeviceSizeTier.STANDARD
    assert DeviceProfile.from_name("OASIS").size_tier == DeviceSizeTier.STANDARD
    assert DeviceProfile.from_name("KS").size_tier == DeviceSizeTier.LARGE
    assert DeviceProfile.from_name("CUSTOM", custom_size=(1072, 1448)).size_tier == DeviceSizeTier.COMPACT
    # Landscape custom screen still classifies by short edge
    assert DeviceProfile.from_name("CUSTOM", custom_size=(1680, 1264)).size_tier == DeviceSizeTier.STANDARD
    assert DeviceProfile.from_name("CUSTOM", custom_size=(2400, 1700)).size_tier == DeviceSizeTier.LARGE

    compact_cfg = PanelMergeConfig.for_tier(DeviceSizeTier.COMPACT)
    standard_cfg = PanelMergeConfig.for_profile(DeviceProfile.from_name("KPW6"))
    large_cfg = PanelMergeConfig.for_profile(DeviceProfile.from_name("KS"))
    assert compact_cfg.max_tier_height_ratio < standard_cfg.max_tier_height_ratio < large_cfg.max_tier_height_ratio
    assert PanelMergeConfig.for_tier(DeviceSizeTier.LARGE, enabled=False).enabled is False

    # ConversionConfig auto-calibrates panel_merge from profile while respecting explicit overrides
    cfg_ks = ConversionConfig(
        input_path=tmp_path / "in.cbz",
        output_path=tmp_path / "out.kfx",
        profile=DeviceProfile.from_name("KS"),
    )
    assert cfg_ks.panel_merge == large_cfg

    cfg_override = ConversionConfig(
        input_path=tmp_path / "in.cbz",
        output_path=tmp_path / "out.kfx",
        profile=DeviceProfile.from_name("KS"),
        panel_merge=PanelMergeConfig(max_tier_height_ratio=0.41),
    )
    assert cfg_override.panel_merge.max_tier_height_ratio == 0.41
    assert cfg_override.panel_merge.max_subcol_height_ratio == large_cfg.max_subcol_height_ratio

    # detector.merge_bridged_panels=False disables panel_merge.enabled unless explicitly overridden
    cfg_det_disabled = ConversionConfig(
        input_path=tmp_path / "in.cbz",
        output_path=tmp_path / "out.kfx",
        profile=DeviceProfile.from_name("KPW6"),
        detector=DetectorConfig(merge_bridged_panels=False),
    )
    assert cfg_det_disabled.panel_merge.enabled is False

    cfg_merge_explicit = ConversionConfig(
        input_path=tmp_path / "in.cbz",
        output_path=tmp_path / "out.kfx",
        profile=DeviceProfile.from_name("KPW6"),
        detector=DetectorConfig(merge_bridged_panels=False),
        panel_merge=PanelMergeConfig(enabled=True),
    )
    assert cfg_merge_explicit.panel_merge.enabled is True

