#!/usr/bin/env python3
"""Run one published BigCodeBench suite, one method at a time."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suite",
        choices=sorted(path.name for path in CONFIG_ROOT.iterdir() if path.is_dir()),
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Run only this config stem; repeat to select multiple methods.",
    )
    parser.add_argument("--limit", type=int, help="Process only N pending tasks.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs",
        help="Root directory for generated outputs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = CONFIG_ROOT / args.suite
    configs = sorted(config_dir.glob("*.json"))
    if args.method:
        wanted = set(args.method)
        configs = [path for path in configs if path.stem in wanted]
        missing = wanted - {path.stem for path in configs}
        if missing:
            raise SystemExit(f"Unknown method config(s): {', '.join(sorted(missing))}")
    for config_path in configs:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        module = (
            "experiments.method_runner"
            if "method" in config
            else "experiments.memory_runner"
        )
        output_dir = args.output_root / args.suite / config_path.stem
        command = [
            sys.executable,
            "-m",
            module,
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
