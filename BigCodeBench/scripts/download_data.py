#!/usr/bin/env python3
"""Download and deterministically materialize BigCodeBench v0.1.4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    revision: str
    source_sha256: str
    output_name: str
    output_sha256: str
    expected_rows: int


SOURCE_PATH = "data/v0.1.4-00000-of-00001.parquet"
SPECS = (
    DatasetSpec(
        repo_id="bigcode/bigcodebench",
        revision="b74c0d0bf70d2c0bc459be537895cca163007f1a",
        source_sha256=(
            "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267"
        ),
        output_name="BigCodeBench-v0.1.4.jsonl",
        output_sha256=(
            "0877e9216fed1730a97bada5f7c7e9b0024a06532c13ee1158e8569a66fe7dc9"
        ),
        expected_rows=1140,
    ),
    DatasetSpec(
        repo_id="bigcode/bigcodebench-hard",
        revision="298d2cc7b96612e15e47313c3603ee124cee0c1f",
        source_sha256=(
            "73a4270b43feb81abefad7bb1b592937c768ea45c3263bedc1cabd30a9a6ce79"
        ),
        output_name="BigCodeBench-Hard-v0.1.4.jsonl",
        output_sha256=(
            "495bd97ec58d63b48869e5b9f0395bfdb1e521d964ca921c706f507168d5b10b"
        ),
        expected_rows=148,
    ),
)

REQUIRED_FIELDS = {
    "task_id",
    "complete_prompt",
    "instruct_prompt",
    "canonical_solution",
    "code_prompt",
    "test",
    "entry_point",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(rows: Iterable[dict[str, Any]], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(spec: DatasetSpec, output_dir: Path, cache_dir: Path) -> set[str]:
    source = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            filename=SOURCE_PATH,
            revision=spec.revision,
            cache_dir=cache_dir,
        )
    )
    actual_source_hash = file_sha256(source)
    if actual_source_hash != spec.source_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {spec.repo_id}: expected "
            f"{spec.source_sha256}, got {actual_source_hash}"
        )

    dataset = load_dataset("parquet", data_files=str(source), split="train")
    if len(dataset) != spec.expected_rows:
        raise RuntimeError(
            f"Unexpected row count for {spec.repo_id}: expected "
            f"{spec.expected_rows}, got {len(dataset)}"
        )
    missing = REQUIRED_FIELDS.difference(dataset.column_names)
    if missing:
        raise RuntimeError(f"Missing fields in {spec.repo_id}: {sorted(missing)}")

    destination = output_dir / spec.output_name
    write_jsonl(dataset, destination)
    output_hash = file_sha256(destination)
    if output_hash != spec.output_sha256:
        raise RuntimeError(
            f"Non-reproducible JSONL for {spec.repo_id}: expected "
            f"{spec.output_sha256}, got {output_hash}"
        )
    print(f"{destination}: {len(dataset)} rows, sha256={output_hash}")
    return set(dataset["task_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bigcodebench-v0.1.4"),
        help="directory for the two JSONL files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face cache (default: OUTPUT_DIR/.cache)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = (args.cache_dir or output_dir / ".cache").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    full_ids = materialize(SPECS[0], output_dir, cache_dir)
    hard_ids = materialize(SPECS[1], output_dir, cache_dir)
    if not hard_ids.issubset(full_ids):
        raise RuntimeError("BigCodeBench-Hard contains task IDs absent from Full")

    print("\nSet these environment variables:")
    print(f"BIGCODEBENCH_OVERRIDE_PATH={output_dir / SPECS[0].output_name}")
    print(f"BIGCODEBENCH_HARD_OVERRIDE_PATH={output_dir / SPECS[1].output_name}")


if __name__ == "__main__":
    main()
