"""Self-contained worker script executed inside Calibre's Python via calibre-debug -e."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
import json
from pathlib import Path
import sys

META_PREFIX = "M2K_META:"
KFX_OUTPUT_PLUGIN_NAME = "KFX Output"
PDOC_CONTENT_TYPE = "PDOC"
CDE_CONTENT_TYPE_KEY = "cde_content_type"
PDOC_CLI_FLAG = "-d"


class CalibreWorkerAction(str, Enum):
    """Supported actions for the Calibre worker script."""

    VERIFY = "verify"
    COMPILE_AND_VERIFY = "compile-and-verify"


def emit_pdoc_metadata(kfx_path: Path) -> None:
    """Inspect a compiled KFX file via YJ_Book and print its metadata marker."""
    import calibre.customize.ui  # noqa: F401 - initializes Calibre plugin loader
    from calibre_plugins.kfx_output.kfxlib.yj_book import YJ_Book

    meta = YJ_Book(str(kfx_path)).get_metadata()
    payload = {CDE_CONTENT_TYPE_KEY: getattr(meta, CDE_CONTENT_TYPE_KEY, None)}
    print(f"{META_PREFIX}{json.dumps(payload)}")


def compile_and_verify(kpf_path: Path, output_kfx: Path) -> None:
    """Compile a KPF package to KFX via KFX Output and print PDOC metadata."""
    import calibre.customize.ui as ui

    plugin = ui.find_plugin(KFX_OUTPUT_PLUGIN_NAME)
    if plugin is None:
        raise RuntimeError(f"{KFX_OUTPUT_PLUGIN_NAME} plugin not installed")

    exit_code = plugin.cli_main(
        [KFX_OUTPUT_PLUGIN_NAME, PDOC_CLI_FLAG, str(kpf_path), str(output_kfx)]
    )
    if isinstance(exit_code, int) and exit_code != 0:
        raise SystemExit(exit_code)

    if output_kfx.is_file() and output_kfx.stat().st_size > 0:
        emit_pdoc_metadata(output_kfx)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch CLI arguments passed to calibre-debug -e."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise ValueError("Missing worker action")

    action = CalibreWorkerAction(args[0])
    if action is CalibreWorkerAction.VERIFY:
        if len(args) != 2:
            raise ValueError(f"Expected 2 arguments for {action.value}, got {len(args)}")
        emit_pdoc_metadata(Path(args[1]))
    elif action is CalibreWorkerAction.COMPILE_AND_VERIFY:
        if len(args) != 3:
            raise ValueError(f"Expected 3 arguments for {action.value}, got {len(args)}")
        compile_and_verify(Path(args[1]), Path(args[2]))
    return 0


if __name__ == "__main__":
    main()
