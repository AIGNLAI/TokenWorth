#!/usr/bin/env python3
"""Report every run, separating comparable complete runs from diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORED_CALLBACK_SET = {"saving", "abnormal", "memory_monitor"}
VALID_STATUSES = {
    "completed",
    "task_limit_reached",
    "agent_validation_failed",
    "agent_context_limit",
}


@dataclass
class RunSummary:
    group: str
    output_dir: str
    task: str
    method: str
    method_parameters: str
    model_name: str
    base_url: str
    protocol_hash: str
    method_config_hash: str
    implementation_hash: str
    planned_repeats: int
    planned_group_id: str
    run_id: str
    primary_metric: str
    primary_score: float | None
    f1_score: float | None
    accuracy: float
    sample_count: int
    expected_sample_count: int
    correct_count: int
    incorrect_count: int
    completed_count: int
    task_limit_reached_count: int
    agent_validation_failed_count: int
    agent_unknown_error_count: int
    task_environment_error_count: int
    agent_context_limit_count: int
    other_status_count: int
    eligible: bool
    invalid_reasons: str
    agent_prompt_tokens: int
    agent_completion_tokens: int
    auxiliary_prompt_tokens: int
    auxiliary_completion_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    call_count: int
    budget_exhausted_count: int
    prompt_guard_count: int
    latency_seconds: float
    usage_complete: bool
    output_tokens_per_task: float | None
    max_memory_injection_chars: int | None


def _load_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return fallback


def _get_method(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for callback in (
        config.get("assignment_config", {}).get("callback_dict", {}).values()
    ):
        name = callback.get("name", "unknown").removesuffix("_callback")
        if name not in IGNORED_CALLBACK_SET:
            defaults = (
                config.get("callback_dict", {}).get(name, {}).get("parameters", {})
            )
            return name, {**defaults, **callback.get("custom_parameters", {})}
    return "standard", {}


def _get_nested_float(value: dict[str, Any], *keys: str) -> float | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return float(current) if math.isfinite(current) else None
    return None


def _usage_summary(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    # Include every attempt, even an interrupted/uncommitted sample or an API retry.
    counters = dict(
        agent_prompt_tokens=0,
        agent_completion_tokens=0,
        auxiliary_prompt_tokens=0,
        auxiliary_completion_tokens=0,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        total_tokens=0,
        call_count=0,
        budget_exhausted_count=0,
        prompt_guard_count=0,
        latency_seconds=0.0,
        usage_complete=True,
    )
    ledger = directory / "usage.jsonl"
    if not ledger.is_file():
        counters["usage_complete"] = False
        return counters
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError("invalid usage entry")
        except (ValueError, TypeError):
            counters["usage_complete"] = False
            continue
        if entry.get("attempt", 1) != 0:
            counters["call_count"] += 1
        if entry.get("status") in {"budget_exhausted", "prompt_guard"}:
            counters[entry["status"] + "_count"] += 1
        phase = entry.get("phase")
        if phase not in {"agent", "auxiliary"}:
            counters["usage_complete"] = False
        latency = entry.get("latency_seconds", entry.get("latency", 0.0))
        if (
            isinstance(latency, (int, float))
            and math.isfinite(latency)
            and latency >= 0
        ):
            counters["latency_seconds"] += latency
        else:
            counters["usage_complete"] = False
        if entry.get("missing_usage") or entry.get("usage_missing"):
            counters["usage_complete"] = False
        for token_type in ("prompt_tokens", "completion_tokens", "total_tokens"):
            count = entry.get(token_type)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                counters["usage_complete"] = False
                continue
            counters[
                "total_" + token_type if token_type != "total_tokens" else token_type
            ] += count
            if phase in {"agent", "auxiliary"} and token_type != "total_tokens":
                counters[phase + "_" + token_type] += count
    if not ledger.read_text(encoding="utf-8").strip():
        counters["usage_complete"] = False
    return counters


def _audit_request_journal(
    directory: Path, manifest: dict[str, Any]
) -> tuple[list[str], int | None]:
    journal = directory / "request_attempts.jsonl"
    if not journal.is_file():
        # A complete stream can fail the prompt guard before making any network request.
        usage_path = directory / "usage.jsonl"
        if usage_path.exists():
            try:
                entries = [
                    json.loads(line)
                    for line in usage_path.read_text().splitlines()
                    if line.strip()
                ]
                if entries and all(
                    isinstance(entry, dict)
                    and entry.get("attempt") == 0
                    and entry.get("status") in {"prompt_guard", "budget_exhausted"}
                    for entry in entries
                ):
                    return [], 0
            except (ValueError, OSError):
                pass
        if manifest.get("protocol", {}).get("generation_budget") is not None:
            return ["missing durable request journal"], None
        return [], None
    reasons = []
    started, finished = set(), set()
    for path, keys in ((journal, started), (directory / "usage.jsonl", finished)):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                attempt = entry.get("attempt")
                if not isinstance(attempt, int):
                    raise ValueError("missing attempt")
                if attempt == 0:
                    continue
                key = (entry["request_id"], attempt)
                if key in keys:
                    reasons.append("duplicate request attempt in cost journal")
                keys.add(key)
            except (ValueError, TypeError, KeyError, AttributeError):
                reasons.append("malformed request journal or usage record")
    if started != finished:
        reasons.append("request journal and usage differ: unknown request cost")
    return reasons, len(started | finished)


def _audit_generation_budget(directory: Path, manifest: dict[str, Any]) -> list[str]:
    budget = manifest.get("protocol", {}).get("generation_budget")
    if not isinstance(budget, int) or budget <= 0:
        return []
    ledger = directory / "usage.jsonl"
    if not ledger.is_file():
        return ["missing usage ledger for generation budget audit"]
    totals: Counter[str] = Counter()
    reasons = []
    expected = {str(index) for index in manifest.get("sample_order", [])}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            index = str(entry.get("sample_index"))
            count = entry.get("completion_tokens")
            if (
                entry.get("missing_usage")
                or entry.get("usage_missing")
                or not isinstance(count, int)
                or count < 0
            ):
                reasons.append(
                    "generation budget cannot be audited: missing token usage"
                )
                continue
            if index not in expected:
                reasons.append("usage ledger contains sample outside protocol")
            totals[index] += count
        except (ValueError, AttributeError):
            reasons.append("generation budget cannot be audited: malformed ledger")
    if any(count > budget for count in totals.values()):
        reasons.append(
            "per-task total agent and auxiliary output exceeds generation budget"
        )
    return reasons


def _summarize_output(outputs_dir: Path, config_path: Path) -> RunSummary:
    directory = config_path.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    manifest = _load_json(directory / "runtime_manifest.json", {})
    status = _load_json(directory / "run_status.json", {})
    metric = _load_json(directory / "metric.json", {})
    runs = _load_json(directory / "runs.json", None)
    reasons: list[str] = []
    for label, value in (
        ("config", config),
        ("runtime manifest", manifest),
        ("run status", status),
        ("metric", metric),
    ):
        if not isinstance(value, dict) or not value:
            reasons.append(f"missing or invalid {label}")
    config = config if isinstance(config, dict) else {}
    manifest = manifest if isinstance(manifest, dict) else {}
    status = status if isinstance(status, dict) else {}
    metric = metric if isinstance(metric, dict) else {}
    if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
        reasons.append("missing or invalid runs")
        runs = []
    task = str(config.get("assignment_config", {}).get("task", ""))
    if manifest.get("task") != task:
        reasons.append("manifest task differs from config")
    method, parameters = _get_method(config)
    if manifest.get("method") and manifest["method"] != method:
        reasons.append("manifest method differs from config")
    expected = manifest.get("sample_order")
    actual = [run.get("sample_index") for run in runs]
    if not isinstance(expected, list) or not expected:
        reasons.append("missing nonempty expected sample_order")
        expected = []
    else:
        if len({json.dumps(x, sort_keys=True) for x in expected}) != len(expected):
            reasons.append("duplicate expected sample IDs")
        if actual != expected:
            reasons.append("sample IDs or order differ from protocol")
    if len({json.dumps(x, sort_keys=True) for x in actual}) != len(actual):
        reasons.append("duplicate result sample IDs")
    if any(run.get("task_name") not in {None, task} for run in runs):
        reasons.append("result task differs from protocol")
    if status.get("status") != "completed" or status.get("valid") is not True:
        reasons.append("run is not marked completed and valid")
    for key, count in (
        ("expected_sample_count", len(expected)),
        ("completed_sample_count", len(runs)),
    ):
        if status.get(key) != count:
            reasons.append(f"run status {key} mismatch")
    if status.get("invalid_reasons"):
        reasons.append("run status records invalid reasons")
    protocol_hash = str(manifest.get("protocol_hash", ""))
    method_hash = str(manifest.get("method_config_hash", ""))
    if not protocol_hash or not method_hash or not manifest.get("implementation_hash"):
        reasons.append("missing protocol, method config, or implementation hash")
    outcomes: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    f1_values: list[float] = []
    injection_values = []
    injection_budget = (
        manifest.get("protocol", {}).get("resource_budget", {}).get("injection_chars")
    )
    for run in runs:
        injection = run.get("memory_injection_chars")
        if (
            isinstance(injection, int)
            and not isinstance(injection, bool)
            and injection >= 0
        ):
            injection_values.append(injection)
        elif injection_budget is not None:
            reasons.append("missing/invalid memory injection character count")
        if (
            isinstance(injection_budget, int)
            and isinstance(injection, int)
            and injection > injection_budget
        ):
            reasons.append("memory injection exceeds protocol character budget")
        evaluation = run.get("evaluation_record") or {}
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        outcome = evaluation.get("outcome", "unknown")
        sample_status = run.get("sample_status", "unknown")
        outcomes[outcome] += 1
        statuses[sample_status] += 1
        if outcome not in {"correct", "incorrect"}:
            reasons.append("unknown/unset evaluation outcome")
        if sample_status not in VALID_STATUSES:
            reasons.append(f"invalid sample status: {sample_status}")
        if task == "knowledge_graph":
            details = evaluation.get("detail_dict") or {}
            f1 = _get_nested_float(details, "f1_score")
            if f1 is None:
                # Unexecuted legitimate failures have F1=0; completed answers need an evaluator F1.
                if sample_status == "completed":
                    reasons.append("completed KG sample missing F1")
                f1 = 0.0
            if not 0 <= f1 <= 1:
                reasons.append("KG F1 outside [0, 1]")
            f1_values.append(f1)
    accuracy = outcomes["correct"] / len(runs) if runs else 0.0
    saved_success = _get_nested_float(
        metric, "overall", "evaluation_outcome", "correct"
    )
    if saved_success is None or not math.isclose(saved_success, accuracy, abs_tol=1e-9):
        reasons.append("metric success rate differs from sample results")
    f1_score = statistics.fmean(f1_values) if f1_values else None
    if task == "knowledge_graph":
        saved_f1 = _get_nested_float(metric, "overall", "additional", "f1_score")
        if (
            saved_f1 is None
            or f1_score is None
            or not math.isclose(saved_f1, f1_score, abs_tol=1e-9)
        ):
            reasons.append("metric F1 differs from sample results")
    protocol = _load_json(REPO_ROOT / "data/manifest.json", {})
    primary = manifest.get("primary_metric") or protocol.get("benchmarks", {}).get(
        task, {}
    ).get("primary_metric")
    if primary not in {"success_rate", "f1_score"}:
        reasons.append("missing/unsupported primary metric")
    primary_score = f1_score if primary == "f1_score" else accuracy
    identities = manifest.get("model_identity", {})
    identities = identities if isinstance(identities, dict) else {}
    model_names, base_urls = [], []
    for identity in identities.values():
        if isinstance(identity, dict):
            model_names.append(str(identity.get("model_name", "")))
            base_urls.append(str(identity.get("base_url", "")))
    if not identities:
        reasons.append("missing resolved model identity")
    if status.get("checkpoint_generation") is not None:
        try:
            from src.utils.checkpoint import CheckpointManager

            manager = CheckpointManager(str(directory))
            committed = manager.peek_sessions()
            pointer = _load_json(directory / "checkpoints" / "CURRENT", {})
            if (
                committed != runs
                or pointer.get("generation") != status["checkpoint_generation"]
            ):
                reasons.append(
                    "exported sessions/status differ from committed checkpoint"
                )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            reasons.append("committed checkpoint is missing or corrupt")
    usage = _usage_summary(directory, manifest)
    reasons.extend(_audit_generation_budget(directory, manifest))
    journal_reasons, actual_calls = _audit_request_journal(directory, manifest)
    reasons.extend(journal_reasons)
    if journal_reasons:
        usage["usage_complete"] = False
    if actual_calls is not None:
        usage["call_count"] = actual_calls
    return RunSummary(
        group="valid" if not reasons else "invalid",
        output_dir=directory.relative_to(outputs_dir).as_posix(),
        task=task,
        method=method,
        method_parameters=json.dumps(parameters, ensure_ascii=False, sort_keys=True),
        model_name=", ".join(model_names),
        base_url=", ".join(base_urls),
        protocol_hash=protocol_hash,
        method_config_hash=method_hash,
        implementation_hash=str(manifest.get("implementation_hash", "")),
        planned_repeats=0,
        planned_group_id="",
        run_id=str(manifest.get("run_id", directory.name)),
        primary_metric=str(primary or ""),
        primary_score=primary_score,
        f1_score=f1_score,
        accuracy=accuracy,
        sample_count=len(runs),
        expected_sample_count=len(expected),
        correct_count=outcomes["correct"],
        incorrect_count=outcomes["incorrect"],
        completed_count=statuses["completed"],
        task_limit_reached_count=statuses["task_limit_reached"],
        agent_validation_failed_count=statuses["agent_validation_failed"],
        agent_unknown_error_count=statuses["agent_unknown_error"],
        task_environment_error_count=statuses["task_environment_error"],
        agent_context_limit_count=statuses["agent_context_limit"],
        other_status_count=sum(
            count
            for name, count in statuses.items()
            if name not in VALID_STATUSES
            and name not in {"agent_unknown_error", "task_environment_error"}
        ),
        eligible=not reasons,
        invalid_reasons="; ".join(dict.fromkeys(reasons)),
        **usage,
        max_memory_injection_chars=max(injection_values) if injection_values else None,
        output_tokens_per_task=(
            usage["total_completion_tokens"] / len(expected)
            if expected and usage["usage_complete"]
            else None
        ),
    )


def validate_output(directory: Path) -> tuple[bool, str]:
    """The launcher uses the same acceptance rules as the published report."""
    if not (directory / "config.yaml").is_file():
        return False, "missing config.yaml"
    try:
        summary = _summarize_output(directory.parent, directory / "config.yaml")
        return summary.eligible, summary.invalid_reasons
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return False, f"unreadable output: {type(error).__name__}"


def _write_csv(path: Path, rows: Iterable[Any]) -> None:
    rows = [
        asdict(row) if hasattr(row, "__dataclass_fields__") else row for row in rows
    ]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _get_primary_summary_list(summaries: Iterable[RunSummary]) -> list[RunSummary]:
    return [summary for summary in summaries if summary.eligible]


def _aggregate(summaries: Iterable[RunSummary]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[RunSummary]] = defaultdict(list)
    seen: set[str] = set()
    for summary in _get_primary_summary_list(summaries):
        # A copied result cannot be counted as an independent repetition.
        if summary.run_id in seen:
            continue
        seen.add(summary.run_id)
        key = (
            summary.task,
            summary.protocol_hash,
            summary.implementation_hash,
            summary.method,
            summary.method_config_hash,
            summary.primary_metric,
        )
        groups[key].append(summary)
    rows = []
    for (task, protocol, implementation, method, method_hash, primary), runs in sorted(
        groups.items()
    ):
        values = [run.primary_score for run in runs if run.primary_score is not None]
        planned_groups = {
            run.planned_group_id: run.planned_repeats
            for run in runs
            if run.planned_repeats
        }
        planned_count = sum(planned_groups.values()) + sum(
            not run.planned_repeats for run in runs
        )
        known_costs = [
            run.output_tokens_per_task
            for run in runs
            if run.output_tokens_per_task is not None
        ]
        cost_complete = len(known_costs) == len(runs)
        rows.append(
            dict(
                task=task,
                protocol_hash=protocol,
                implementation_hash=implementation,
                method=method,
                method_config_hash=method_hash,
                primary_metric=primary,
                repeats=len(values),
                planned_repeats=planned_count,
                complete_group=len(values) == planned_count,
                mean=statistics.fmean(values),
                std=statistics.stdev(values) if len(values) > 1 else None,
                output_tokens_per_task_mean=(
                    statistics.fmean(known_costs) if cost_complete else None
                ),
                output_tokens_per_task_std=(
                    statistics.stdev(known_costs)
                    if cost_complete and len(known_costs) > 1
                    else None
                ),
                agent_output_tokens_per_task_mean=(
                    statistics.fmean(
                        run.agent_completion_tokens / run.expected_sample_count
                        for run in runs
                    )
                    if cost_complete
                    else None
                ),
                auxiliary_output_tokens_per_task_mean=(
                    statistics.fmean(
                        run.auxiliary_completion_tokens / run.expected_sample_count
                        for run in runs
                    )
                    if cost_complete
                    else None
                ),
                calls_per_task_mean=statistics.fmean(
                    run.call_count / run.expected_sample_count for run in runs
                ),
                usage_complete=cost_complete,
                delta_output_tokens_percent=None,
                outputs=json.dumps([run.output_dir for run in runs]),
            )
        )
    standards = {}
    for row in rows:
        if row["method"] == "standard" and row["complete_group"]:
            key = (row["task"], row["protocol_hash"], row["implementation_hash"])
            # Never choose a favorable Standard configuration to define the denominator.
            standards.setdefault(key, []).append(row["output_tokens_per_task_mean"])
    for row in rows:
        values = standards.get(
            (row["task"], row["protocol_hash"], row["implementation_hash"]), []
        )
        if (
            len(values) == 1
            and values[0] is not None
            and values[0] > 0
            and row["output_tokens_per_task_mean"] is not None
        ):
            row["delta_output_tokens_percent"] = (
                100 * (row["output_tokens_per_task_mean"] - values[0]) / values[0]
            )
    return rows


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _write_markdown(path: Path, summaries: list[RunSummary]) -> None:
    lines = [
        "# Experiment results",
        "",
        "Only complete, valid runs with the same runtime protocol are compared. Every repetition is retained; no best-run selection. Different method configurations are reported separately. KG includes F1 and exact-match success rate.",
        "",
        "## Comparable repetitions",
        "",
        "| Task | Protocol | Implementation | Method | Config | Metric | Repeats | Complete | Mean | Sample SD |",
        "|---|---|---|---|---|---|---:|---|---:|---:|",
    ]
    for row in _aggregate(summaries):
        lines.append(
            "| "
            + " | ".join(
                map(
                    _cell,
                    [
                        row["task"],
                        row["protocol_hash"][:12],
                        row["implementation_hash"][:12],
                        row["method"],
                        row["method_config_hash"][:12],
                        row["primary_metric"],
                        f"{row['repeats']}/{row['planned_repeats']}",
                        row["complete_group"],
                        _rate(row["mean"]),
                        _rate(row["std"]),
                    ],
                )
            )
            + " |"
        )
    lines += [
        "",
        "Incomplete groups are diagnostic only; their surviving-run means must not be used as a final comparison. SD is unavailable for a single repetition. Repetitions share the exact task order fixed by the protocol. Copied run IDs count once.",
        "",
        "## All runs",
        "",
        "| Output | Task | Method | Samples | Eligible | Primary | Success / KG EM | KG F1 | Output tokens/task | Calls | Reasons |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in summaries:
        lines.append(
            "| "
            + " | ".join(
                map(
                    _cell,
                    [
                        run.output_dir,
                        run.task,
                        run.method,
                        f"{run.sample_count}/{run.expected_sample_count}",
                        run.eligible,
                        _rate(run.primary_score),
                        _rate(run.accuracy),
                        _rate(run.f1_score),
                        (
                            f"{run.output_tokens_per_task:.2f}"
                            if run.output_tokens_per_task is not None
                            else "unknown"
                        ),
                        run.call_count,
                        run.invalid_reasons,
                    ],
                )
            )
            + " |"
        )
    lines += [
        "",
        "Token totals in all_outputs.csv include main and auxiliary calls, all retries, and uncommitted samples. Missing usage is marked usage_complete=false; partial known totals are not presented as exact costs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_report(outputs_dir: Path, report_dir: Path) -> list[RunSummary]:
    outputs_dir = outputs_dir.resolve()
    paths = sorted(outputs_dir.rglob("config.yaml"))
    summaries = []
    for path in paths:
        try:
            summaries.append(_summarize_output(outputs_dir, path))
        except (OSError, ValueError, TypeError, AttributeError) as error:
            print(
                f"Cannot summarize {path.relative_to(outputs_dir)}: {type(error).__name__}"
            )
    # The batch preselects repetitions before any outcome is observed.
    plans = {}
    for batch_path in outputs_dir.rglob("batch_manifest.json"):
        batch = _load_json(batch_path, {})
        if not isinstance(batch, dict):
            continue
        records = batch.get("runs", [])
        if not isinstance(records, list):
            continue
        counts = Counter(
            record.get("source_config_path")
            for record in records
            if isinstance(record, dict)
        )
        for record in records:
            if isinstance(record, dict) and record.get("output_dir"):
                plans[str(Path(record["output_dir"]).resolve())] = (
                    counts[record.get("source_config_path")],
                    str(batch_path) + "::" + str(record.get("source_config_path")),
                )
    for summary in summaries:
        summary.planned_repeats, summary.planned_group_id = plans.get(
            str(outputs_dir / summary.output_dir), (0, "")
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "all_outputs.csv", summaries)
    _write_csv(report_dir / "repetitions.csv", _aggregate(summaries))
    _write_csv(
        report_dir / "comparisons.csv",
        [row for row in _aggregate(summaries) if row["complete_group"]],
    )
    _write_markdown(report_dir / "summary.md", summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/statistics"))
    args = parser.parse_args()
    summaries = generate_report(args.outputs_dir, args.report_dir)
    print(f"Wrote {len(summaries)} run summaries to {args.report_dir}.")


if __name__ == "__main__":
    main()
