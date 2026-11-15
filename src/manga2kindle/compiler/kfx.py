"""Headless Calibre KFX Output compiler wrapper and structural PDOC verifier."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_META_PREFIX = "M2K_META:"
_VERIFY_PDOC_SCRIPT = (
    "import json, sys; "
    "import calibre.customize.ui; "
    "from calibre_plugins.kfx_output.kfxlib.yj_book import YJ_Book; "
    "meta = YJ_Book(sys.argv[1]).get_metadata(); "
    f"print({_META_PREFIX!r} + json.dumps({{'cde_content_type': getattr(meta, 'cde_content_type', None)}}))"
)


class KfxCompilationError(RuntimeError):
    """Raised when calibre-debug KFX Output fails or produces invalid output."""


class KfxCompiler:
    """Compiles KPF packages to PDOC-tagged KFX via Calibre's KFX Output plugin."""

    def __init__(self, calibre_debug_path: Path) -> None:
        self.calibre_debug_path = Path(calibre_debug_path)

    def _run_calibre(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.calibre_debug_path), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def compile(self, kpf_path: Path, output_kfx: Path) -> Path:
        """Compile a .kpf package into a PDOC-tagged .kfx file using calibre-debug."""
        output_kfx.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = self._run_calibre(
                ["-r", "KFX Output", "--", "-d", str(kpf_path), str(output_kfx)]
            )
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

        if not output_kfx.is_file():
            raise KfxCompilationError(
                f"KFX Output exited cleanly but output file is missing: {output_kfx}"
            )

        if output_kfx.stat().st_size == 0:
            raise KfxCompilationError(
                f"KFX Output produced an empty file (0 bytes): {output_kfx}"
            )

        return output_kfx

    def verify_pdoc(self, kfx_path: Path) -> bool:
        """Verify structurally via YJ_Book that kfx_path has cde_content_type == 'PDOC'."""
        if not kfx_path.is_file() or kfx_path.stat().st_size == 0:
            return False

        try:
            proc = self._run_calibre(["-c", _VERIFY_PDOC_SCRIPT, str(kfx_path)])
        except (OSError, subprocess.SubprocessError):
            return False

        if proc.returncode != 0:
            return False

        for line in reversed((proc.stdout or "").splitlines()):
            if line.startswith(_META_PREFIX):
                raw_json = line[len(_META_PREFIX) :]
                try:
                    data = json.loads(raw_json)
                except (json.JSONDecodeError, TypeError, ValueError):
                    return False
                return isinstance(data, dict) and data.get("cde_content_type") == "PDOC"

        return False
