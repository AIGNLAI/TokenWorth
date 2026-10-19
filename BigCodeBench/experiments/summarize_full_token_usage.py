"""Sum solver and memory-building API usage for a local experiment round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "no_memory", "rag", "tokenworth", "memrl", "memp",
    "mem0", "evoskill", "evosc", "reflexion", "memskill",
)
FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sum_metadata(records: list[dict], key: str) -> tuple[dict[str, int], int]:
    sums = {field: 0 for field in FIELDS}
    missing = 0
    for record in records:
        metadata = record.get(key) or {}
        if any(metadata.get(field) is None for field in FIELDS[:3]):
            missing += 1
            continue
        for field in FIELDS:
            sums[field] += int(metadata.get(field) or 0)
    return sums, missing


def summarize(round_dir: Path) -> dict[str, dict]:
    result = {}
    for method in METHODS:
        method_dir = round_dir / method
        rows = read_jsonl(method_dir / "results.jsonl")
        calls = read_jsonl(method_dir / "auxiliary_usage.jsonl")
        main, main_missing = sum_metadata(rows, "generation_metadata")
        auxiliary, auxiliary_missing = sum_metadata(calls, "generation_metadata")
        result[method] = {
            "completed_tasks": len(rows),
            "auxiliary_calls": len(calls),
            "main": main,
            "auxiliary": auxiliary,
            "all": {field: main[field] + auxiliary[field] for field in FIELDS},
            "missing_main_usage": main_missing,
            "missing_auxiliary_usage": auxiliary_missing,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_dir", type=Path)
    parser.add_argument("--write", action="store_true", help="Write token_usage_summary.json")
    args = parser.parse_args()
    result = summarize(args.round_dir)
    if args.write:
        path = args.round_dir / "token_usage_summary.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("method\ttasks\taux_calls\tinput\toutput\ttotal\tmissing_usage")
    for method, item in result.items():
        all_usage = item["all"]
        print(f"{method}\t{item['completed_tasks']}\t{item['auxiliary_calls']}\t"
              f"{all_usage['prompt_tokens']}\t{all_usage['completion_tokens']}\t"
              f"{all_usage['total_tokens']}\t"
              f"{item['missing_main_usage'] + item['missing_auxiliary_usage']}")


if __name__ == "__main__":
    main()
