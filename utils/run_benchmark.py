#!/usr/bin/env python3
"""Cross-platform helper to sync extras, fetch models, and run the 3-layer benchmark."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_uv() -> str:
    uv_exe = shutil.which("uv")
    if uv_exe is not None:
        return uv_exe
    raise SystemExit(
        "Error: 'uv' executable not found on PATH. "
        "Install uv (https://docs.astral.sh/uv/getting-started/installation/) and retry."
    )


def _run_step(cmd: list[str], *, dry_run: bool = False) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mechanically sync dependencies, fetch models, and benchmark manga2kindle backends.",
    )
    parser.add_argument(
        "--dataset",
        default="popmanga",
        choices=("popmanga", "comix", "local"),
        help="Benchmark dataset (default: popmanga).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (default: unseen for popmanga, test for comix).",
    )
    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=None,
        help="Page images directory when --dataset local is used.",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="Ground-truth JSON/COCO file when --dataset local is used.",
    )
    parser.add_argument(
        "--backends",
        default="contour,onnx,magi",
        help="Comma-separated backends to benchmark (default: contour,onnx,magi).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of pages to evaluate (default: 50).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup pages excluded from timing (default: 2).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write JSON benchmark report.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip 'uv sync --all-extras'.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip 'manga2kindle models fetch' for selected backends.",
    )
    parser.add_argument(
        "--no-allow-remote-code",
        action="store_true",
        help="Do not pass --allow-remote-code (disables Magi and HF script datasets).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    uv_exe = _find_uv()
    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    allow_remote_code = not args.no_allow_remote_code

    if args.dataset in ("popmanga", "comix") and not os.environ.get("HF_TOKEN"):
        print(
            "Note: HF_TOKEN is not set. Gated Hugging Face datasets require logging in "
            "(hf auth login or HF_TOKEN=...) and accepting dataset terms on the Hub.",
            flush=True,
        )

    if not args.skip_sync:
        _run_step([uv_exe, "sync", "--all-extras"], dry_run=args.dry_run)

    if not args.skip_fetch:
        if "onnx" in backends:
            _run_step(
                [uv_exe, "run", "--no-sync", "manga2kindle", "models", "fetch", "manga109-yolo-s"],
                dry_run=args.dry_run,
            )
        if "magi" in backends:
            _run_step(
                [uv_exe, "run", "--no-sync", "manga2kindle", "models", "fetch", "magiv3"],
                dry_run=args.dry_run,
            )

    split = args.split or ("unseen" if args.dataset == "popmanga" else "test")
    bench_cmd = [
        uv_exe,
        "run",
        "--no-sync",
        "manga2kindle",
        "benchmark",
        "--dataset",
        args.dataset,
        "--split",
        split,
        "--backends",
        ",".join(backends),
        "--limit",
        str(args.limit),
        "--warmup",
        str(args.warmup),
        "--device",
        args.device,
    ]
    if allow_remote_code:
        bench_cmd.append("--allow-remote-code")
    if args.pages_dir is not None:
        bench_cmd.extend(["--pages-dir", str(args.pages_dir)])
    if args.truth is not None:
        bench_cmd.extend(["--truth", str(args.truth)])
    if args.out is not None:
        bench_cmd.extend(["--out", str(args.out)])

    _run_step(bench_cmd, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
