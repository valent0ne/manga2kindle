"""Headless Calibre KFX Output compiler wrapper and structural PDOC verifier."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
import json
import os
import subprocess
from pathlib import Path

from manga2kindle.compiler._calibre_worker import (
    CDE_CONTENT_TYPE_KEY,
    CalibreWorkerAction,
    KFX_OUTPUT_PLUGIN_NAME,
    META_PREFIX,
    PDOC_CLI_FLAG,
    PDOC_CONTENT_TYPE,
)
from manga2kindle.platform import run_managed_subprocess

_WORKER_SCRIPT_NAME = "_calibre_worker.py"
_DEFAULT_SUBPROCESS_RUN = subprocess.run


class KfxCompilationError(RuntimeError):
    """Raised when calibre-debug KFX Output fails or produces invalid output."""


class KfxCompiler:
    """Compiles KPF packages to PDOC-tagged KFX via Calibre's KFX Output plugin."""

    def __init__(self, calibre_debug_path: Path) -> None:
        self.calibre_debug_path = Path(calibre_debug_path)

    def _run_calibre(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        cmd = [str(self.calibre_debug_path), *args]
        if subprocess.run is not _DEFAULT_SUBPROCESS_RUN:
            return subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        return run_managed_subprocess(cmd, env=env)

    @staticmethod
    def _validate_compiled_output(output_kfx: Path) -> None:
        if not output_kfx.is_file():
            raise KfxCompilationError(
                f"KFX Output exited cleanly but output file is missing: {output_kfx}"
            )
        if output_kfx.stat().st_size == 0:
            raise KfxCompilationError(
                f"KFX Output produced an empty file (0 bytes): {output_kfx}"
            )

    @staticmethod
    @contextmanager
    def _worker_script_path() -> Iterator[Path]:
        worker_resource = resources.files(__package__).joinpath(_WORKER_SCRIPT_NAME)
        with resources.as_file(worker_resource) as worker_path:
            yield worker_path.resolve()

    @staticmethod
    def _has_pdoc_marker(stdout: str) -> bool:
        for line in reversed(stdout.splitlines()):
            if line.startswith(META_PREFIX):
                raw_json = line[len(META_PREFIX) :]
                try:
                    data = json.loads(raw_json)
                except (json.JSONDecodeError, TypeError, ValueError):
                    return False
                return isinstance(data, dict) and data.get(CDE_CONTENT_TYPE_KEY) == PDOC_CONTENT_TYPE
        return False

    def _run_compile_command(
        self, args: list[str], output_kfx: Path
    ) -> subprocess.CompletedProcess[str]:
        output_kfx.parent.mkdir(parents=True, exist_ok=True)
        # Prevent stale pre-existing outputs from masking silent KFX Output exit-0 failures.
        output_kfx.unlink(missing_ok=True)
        try:
            proc = self._run_calibre(args)
        except (OSError, subprocess.SubprocessError) as exc:
            raise KfxCompilationError(
                f"Failed to invoke calibre-debug at {self.calibre_debug_path}: {exc}"
            ) from exc

        if proc.returncode != 0:
            details = (proc.stderr or proc.stdout or "").strip()
            raise KfxCompilationError(
                f"KFX Output failed with exit code {proc.returncode}"
                + (f": {details}" if details else "")
            )

        self._validate_compiled_output(output_kfx)
        return proc

    def compile(self, kpf_path: Path, output_kfx: Path) -> Path:
        """Compile a .kpf package into a PDOC-tagged .kfx file using calibre-debug."""
        resolved_kpf = kpf_path.resolve()
        resolved_kfx = output_kfx.resolve()
        self._run_compile_command(
            [
                "-r",
                KFX_OUTPUT_PLUGIN_NAME,
                "--",
                PDOC_CLI_FLAG,
                str(resolved_kpf),
                str(resolved_kfx),
            ],
            output_kfx,
        )
        return output_kfx

    def verify_pdoc(self, kfx_path: Path) -> bool:
        """Verify structurally via YJ_Book that kfx_path has cde_content_type == 'PDOC'."""
        if not kfx_path.is_file() or kfx_path.stat().st_size == 0:
            return False

        try:
            with self._worker_script_path() as worker_path:
                proc = self._run_calibre(
                    [
                        "-e",
                        str(worker_path),
                        "--",
                        CalibreWorkerAction.VERIFY.value,
                        str(kfx_path.resolve()),
                    ]
                )
        except (OSError, subprocess.SubprocessError):
            return False

        if proc.returncode != 0:
            return False

        return self._has_pdoc_marker(proc.stdout or "")

    def compile_and_verify(self, kpf_path: Path, output_kfx: Path) -> Path:
        """Compile a .kpf to .kfx and verify PDOC tagging in one calibre-debug invocation."""
        if (
            type(self).compile is not _DEFAULT_COMPILE
            or type(self).verify_pdoc is not _DEFAULT_VERIFY_PDOC
            or "compile" in self.__dict__
            or "verify_pdoc" in self.__dict__
        ):
            compiled = self.compile(kpf_path, output_kfx)
            if not self.verify_pdoc(compiled):
                raise KfxCompilationError(
                    f"Compiled KFX failed PDOC verification: {compiled}"
                )
            return compiled

        with self._worker_script_path() as worker_path:
            proc = self._run_compile_command(
                [
                    "-e",
                    str(worker_path),
                    "--",
                    CalibreWorkerAction.COMPILE_AND_VERIFY.value,
                    str(kpf_path.resolve()),
                    str(output_kfx.resolve()),
                ],
                output_kfx,
            )
        if not self._has_pdoc_marker(proc.stdout or ""):
            raise KfxCompilationError(
                f"Compiled KFX failed PDOC verification: {output_kfx}"
            )
        return output_kfx


_DEFAULT_COMPILE = KfxCompiler.compile
_DEFAULT_VERIFY_PDOC = KfxCompiler.verify_pdoc
