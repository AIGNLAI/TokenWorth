#!/usr/bin/env python3
"""Run the published local hash ALFWorld suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "local.yaml"
METHODS = (
    "tokenworth", "rag", "memrl", "reflexion", "memp",
    "evosc", "evoskill", "mem0", "memskill", "no_memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        choices=METHODS,
        help="Run only this method; repeat to select multiple methods.",
    )
    parser.add_argument("--api-base", help="Override OPENAI_BASE_URL.")
    parser.add_argument("--model", help="Override the configured model name.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs",
        help="Root directory for generated outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run at most this many new episodes per selected method.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    selected = args.method or list(METHODS)
    command = [
        sys.executable,
        str(ROOT / "run_continual_eval.py"),
        "--config",
        str(CONFIG),
        "--methods",
        ",".join(selected),
        "--output-root",
        str(args.output_root),
    ]
    if args.api_base:
        command.extend(["--api-base", args.api_base])
    if args.model:
        command.extend(["--model", args.model])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
