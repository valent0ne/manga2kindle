"""Cross-platform discovery and diagnostics for external CLI dependencies."""

from __future__ import annotations

import ctypes
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_BYTES_PER_GIB: float = 1024.0**3
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: int = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: int = 9
_POSIX_TERM_GRACE_SECONDS: float = 2.0


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _windows_avail_commit_gib() -> float | None:
    """Return the minimum of available Windows physical RAM and commit charge in GiB, or None."""
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return min(float(status.ullAvailPhys), float(status.ullAvailPageFile)) / _BYTES_PER_GIB
    except (AttributeError, OSError):
        return None


def available_memory_gib() -> float | None:
    """Return effective available memory (min of physical RAM and Windows commit limit) in GiB."""
    avail_phys: float | None = None
    try:
        import psutil

        avail_phys = float(psutil.virtual_memory().available) / _BYTES_PER_GIB
    except ImportError:
        avail_phys = None

    if sys.platform.startswith("win"):
        avail_commit = _windows_avail_commit_gib()
        if avail_phys is not None and avail_commit is not None:
            return min(avail_phys, avail_commit)
        return avail_commit if avail_commit is not None else avail_phys

    return avail_phys


def _win32_create_kill_job() -> tuple[int, Any] | None:
    """Create a Win32 Job Object configured to kill all assigned descendants on handle close."""
    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class _JobBasicLimitInfo(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _JobExtendedLimitInfo(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _JobBasicLimitInfo),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _JobExtendedLimitInfo()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(job)
            return None
        return int(job), kernel32
    except (AttributeError, OSError, ImportError):
        return None


def _close_win32_job(job_pair: tuple[int, Any] | None) -> None:
    if job_pair is None:
        return
    job_handle, kernel32 = job_pair
    try:
        kernel32.CloseHandle(job_handle)
    except (AttributeError, OSError):
        pass


def _terminate_posix_group(pid: int) -> None:
    killpg = getattr(os, "killpg", None)
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    if killpg is None:
        return
    try:
        killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    try:
        import time

        deadline = time.monotonic() + _POSIX_TERM_GRACE_SECONDS
        while time.monotonic() < deadline:
            try:
                killpg(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        killpg(pid, sigkill)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_managed_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an external CLI and guarantee full process-tree cleanup on exit or interrupt."""
    if sys.platform.startswith("win"):
        job_pair = _win32_create_kill_job()
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if job_pair is not None:
            job_handle, kernel32 = job_pair
            proc_handle = getattr(proc, "_handle", None)
            if proc_handle is None or not kernel32.AssignProcessToJobObject(
                job_handle, int(proc_handle)
            ):
                _close_win32_job(job_pair)
                job_pair = None
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout, stderr=stderr)
        except BaseException:
            if job_pair is not None:
                _close_win32_job(job_pair)
                job_pair = None
            else:
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                proc.wait(timeout=_POSIX_TERM_GRACE_SECONDS)
            except (subprocess.TimeoutExpired, OSError):
                pass
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
            raise
        finally:
            _close_win32_job(job_pair)

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout, stderr=stderr)
    except BaseException:
        _terminate_posix_group(proc.pid)
        try:
            proc.wait(timeout=_POSIX_TERM_GRACE_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass
        raise


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
    ) -> list[ToolStatus]:
        """Run diagnostic checks for all external dependencies required by manga2kindle."""
        kcc_path = self.find_kcc(override=kcc_override)
        calibre_path = self.find_calibre(override=calibre_override)
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

        statuses.append(self._check_tool("unrar/unar/7z", unrar_path))
        statuses.extend(self._check_detector_diagnostics())
        return statuses

    @staticmethod
    def _check_detector_diagnostics() -> list[ToolStatus]:
        """Collect optional ML runtime, cached model, and AUTO detector chain diagnostics."""
        results: list[ToolStatus] = []

        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            results.append(
                ToolStatus(
                    name="onnxruntime",
                    found=True,
                    version=getattr(ort, "__version__", None),
                    details=f"providers: {', '.join(providers)}" if providers else "no providers",
                )
            )
        except ImportError:
            results.append(
                ToolStatus(
                    name="onnxruntime",
                    found=False,
                    details="Optional (pip install manga2kindle[onnx])",
                )
            )

        try:
            import torch

            cuda_mod = getattr(torch, "cuda", None)
            cuda_ok = bool(cuda_mod.is_available()) if cuda_mod is not None else False
            backends_mod = getattr(torch, "backends", None)
            mps_backend = getattr(backends_mod, "mps", None) if backends_mod is not None else None
            mps_ok = bool(mps_backend is not None and mps_backend.is_available())
            results.append(
                ToolStatus(
                    name="torch",
                    found=True,
                    version=getattr(torch, "__version__", None),
                    details=f"CUDA={cuda_ok}, MPS={mps_ok}",
                )
            )
        except ImportError:
            results.append(
                ToolStatus(
                    name="torch",
                    found=False,
                    details="Optional (pip install manga2kindle[magi])",
                )
            )

        from manga2kindle.cv.models_hub import (
            is_magi_cached,
            is_model_cached,
            list_registered_magi_models,
            list_registered_models,
        )

        cached_keys: list[str] = []
        try:
            for spec in list_registered_models():
                if is_model_cached(spec.key):
                    cached_keys.append(spec.key)
            for magi_spec in list_registered_magi_models():
                if is_magi_cached(magi_spec.key):
                    cached_keys.append(magi_spec.key)
        except RuntimeError:
            cached_keys = []

        results.append(
            ToolStatus(
                name="Cached Models",
                found=bool(cached_keys),
                details=", ".join(cached_keys)
                if cached_keys
                else "None cached (run: manga2kindle models fetch)",
            )
        )

        from manga2kindle.cv.backends import describe_auto_chain
        from manga2kindle.models import DetectorConfig

        chain, note = describe_auto_chain(DetectorConfig())
        chain_names = [kind.value for kind in chain]
        details = f"chain: {' -> '.join(chain_names)}"
        if note:
            details = f"{details} ({note})"
        results.append(
            ToolStatus(
                name="AUTO Detector",
                found=True,
                version=chain_names[0] if chain_names else None,
                details=details,
            )
        )
        return results
