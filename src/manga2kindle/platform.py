"""Cross-platform discovery and diagnostics for external CLI dependencies."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel


class ToolStatus(BaseModel):
    """Diagnostic status for an external tool or Calibre plugin."""

    name: str
    found: bool
    path: Path | None = None
    version: str | None = None
    details: str = ""


class PlatformDiscovery:
    """Finds external executables across macOS and Windows environments."""

    def __init__(self, system: str | None = None) -> None:
        self.system = system or sys.platform
        self._is_windows = self.system.startswith("win")

    @staticmethod
    def _first_existing(candidates: list[Path]) -> Path | None:
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _which(names: list[str]) -> Path | None:
        for name in names:
            resolved = shutil.which(name)
            if resolved:
                return Path(resolved)
        return None

    def _env_dirs(self, *var_names: str) -> list[Path]:
        dirs: list[Path] = []
        for var in var_names:
            value = os.environ.get(var)
            if value:
                dirs.append(Path(value))
        return dirs

    @staticmethod
    def _probe_command(cmd: list[str], timeout: int = 10) -> tuple[int | None, str]:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, str(exc)
        output = (proc.stdout or proc.stderr or "").strip()
        return proc.returncode, output

    def find_kcc(self, override: Path | None = None) -> Path | None:
        """Locate the kcc-c2e CLI executable."""
        if override is not None:
            return override if override.is_file() else None

        exe_name = "kcc-c2e.exe" if self._is_windows else "kcc-c2e"
        py_bin = Path(sys.executable).parent
        candidates = [py_bin / exe_name]
        if self._is_windows:
            candidates.append(py_bin / "Scripts" / exe_name)

        found = self._first_existing(candidates)
        if found is not None:
            return found

        return self._which([exe_name, "kcc-c2e"])

    def find_calibre(self, override: Path | None = None) -> Path | None:
        """Locate Calibre's calibre-debug executable."""
        if override is not None:
            return override if override.is_file() else None

        exe_name = "calibre-debug.exe" if self._is_windows else "calibre-debug"
        sibling = Path(sys.executable).parent / exe_name
        if sibling.is_file():
            return sibling

        which_hit = self._which([exe_name, "calibre-debug"])
        if which_hit is not None:
            return which_hit

        if self._is_windows:
            roots = self._env_dirs("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
            candidates = [root / "Calibre2" / "calibre-debug.exe" for root in roots]
        else:
            candidates = [
                Path("/Applications/calibre.app/Contents/MacOS/calibre-debug"),
                Path.home() / "Applications/calibre.app/Contents/MacOS/calibre-debug",
            ]
        return self._first_existing(candidates)

    def find_previewer4(self, override: Path | None = None) -> Path | None:
        """Locate Kindle Previewer 4 CLI executable."""
        if override is not None:
            return override if override.is_file() else None

        exe_name = "KindlePreviewer4CLI.exe" if self._is_windows else "KindlePreviewer4CLI"
        which_hit = self._which([exe_name, "KindlePreviewer4CLI"])
        if which_hit is not None:
            return which_hit

        if self._is_windows:
            roots = self._env_dirs("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)")
            candidates: list[Path] = []
            for root in roots:
                kp4_dir = root / "Amazon" / "Kindle Previewer 4"
                candidates.append(kp4_dir / "KindlePreviewer4CLI.exe")
                candidates.append(kp4_dir / "lib" / "fc" / "bin" / "KindlePreviewer4CLI.exe")
        else:
            candidates = [
                Path("/Applications/Kindle Previewer 4.app/Contents/MacOS/KindlePreviewer4CLI"),
                Path.home()
                / "Applications/Kindle Previewer 4.app/Contents/MacOS/KindlePreviewer4CLI",
            ]
        return self._first_existing(candidates)

    def find_unrar(self) -> Path | None:
        """Locate an archive extractor capable of handling CBR (RAR) archives."""
        names = (
            ["unrar.exe", "unrar", "unar", "7z.exe", "7z", "7zz.exe", "7zz", "7za.exe", "7za"]
            if self._is_windows
            else ["unrar", "unar", "7zz", "7z", "7za"]
        )
        which_hit = self._which(names)
        if which_hit is not None:
            return which_hit

        if self._is_windows:
            roots = self._env_dirs("ProgramFiles", "ProgramFiles(x86)")
            candidates: list[Path] = []
            for root in roots:
                candidates.append(root / "WinRAR" / "UnRAR.exe")
                candidates.append(root / "7-Zip" / "7z.exe")
                candidates.append(root / "7-Zip" / "7zz.exe")
                candidates.append(root / "7-Zip" / "7za.exe")
            candidates.extend(
                [
                    Path(r"C:\Program Files\WinRAR\UnRAR.exe"),
                    Path(r"C:\Program Files\7-Zip\7z.exe"),
                    Path(r"C:\Program Files\7-Zip\7zz.exe"),
                    Path(r"C:\Program Files\7-Zip\7za.exe"),
                ]
            )
            return self._first_existing(candidates)
        return None

    def _check_tool(
        self,
        name: str,
        path: Path | None,
        version_args: list[str] | None = None,
    ) -> ToolStatus:
        if path is None:
            return ToolStatus(name=name, found=False, path=None, details="Not found")
        version: str | None = None
        if version_args is not None:
            code, output = self._probe_command([str(path), *version_args])
            if code == 0 and output:
                version = output.splitlines()[0]
        return ToolStatus(name=name, found=True, path=path, version=version)

    def run_doctor(
        self,
        kcc_override: Path | None = None,
        calibre_override: Path | None = None,
        previewer_override: Path | None = None,
    ) -> list[ToolStatus]:
        """Run diagnostic checks for all external dependencies required by manga2kindle."""
        kcc_path = self.find_kcc(override=kcc_override)
        calibre_path = self.find_calibre(override=calibre_override)
        kp4_path = self.find_previewer4(override=previewer_override)
        unrar_path = self.find_unrar()

        statuses = [
            self._check_tool("kcc-c2e", kcc_path, ["--help"]),
            self._check_tool("calibre-debug", calibre_path, ["--version"]),
        ]

        if calibre_path is None:
            statuses.append(
                ToolStatus(
                    name="KFX Output",
                    found=False,
                    path=None,
                    details="calibre-debug not found",
                )
            )
        else:
            code, output = self._probe_command(
                [str(calibre_path), "-r", "KFX Output", "--", "--help"]
            )
            statuses.append(
                ToolStatus(
                    name="KFX Output",
                    found=code == 0,
                    path=calibre_path if code == 0 else None,
                    version=output.splitlines()[0] if code == 0 and output else None,
                    details="" if code == 0 else (output or "KFX Output plugin not installed"),
                )
            )

        statuses.append(self._check_tool("Kindle Previewer 4", kp4_path))
        statuses.append(self._check_tool("unrar/unar/7z", unrar_path))
        return statuses
