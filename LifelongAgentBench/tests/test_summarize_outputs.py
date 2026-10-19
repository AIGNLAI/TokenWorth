from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.summarize_outputs import (
    _aggregate,
    _summarize_output,
    generate_report,
    validate_output,
)


def write_run(
    root: Path,
    name: str,
    *,
    task: str = "db_bench",
    correct: int = 1,
    count: int = 2,
    protocol: str = "same-protocol",
    method_hash: str = "same-method",
    f1: float = 0.5,
) -> Path:
    output = root / name
    output.mkdir()
    config = {
        "assignment_config": {
            "task": task,
            "callback_dict": {
                "method": {"name": "TokenWorth", "custom_parameters": {}}
            },
        }
    }
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    manifest = {
        "task": task,
        "method": "TokenWorth",
        "sample_order": [str(i) for i in range(count)],
        "protocol_hash": protocol,
        "method_config_hash": method_hash,
        "implementation_hash": "fixture-source",
        "run_id": name,
        "model_identity": {
            "model": {"model_name": "fixture-model", "base_url": "http://fixture/v1"}
        },
        "primary_metric": "success_rate",
    }
    (output / "runtime_manifest.json").write_text(json.dumps(manifest))
    (output / "run_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "valid": True,
                "expected_sample_count": count,
                "completed_sample_count": count,
                "invalid_reasons": [],
            }
        )
    )
    runs = [
        {
            "task_name": task,
            "sample_index": str(i),
            "sample_status": "completed",
            "evaluation_record": {
                "outcome": "correct" if i < correct else "incorrect",
                "detail_dict": {"f1_score": f1},
            },
        }
        for i in range(count)
    ]
    (output / "runs.json").write_text(json.dumps(runs))
    metric = {
        "overall": {
            "evaluation_outcome": {"correct": correct / count},
            "additional": {"f1_score": f1},
        }
    }
    (output / "metric.json").write_text(json.dumps(metric))
    return output


def summary(root: Path, output: Path):
    return _summarize_output(root, output / "config.yaml")


def rewrite(path: Path, update):
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


def test_kg_f1_is_preserved_with_exact_match_and_paper_primary(tmp_path):
    output = write_run(tmp_path, "kg", task="knowledge_graph", correct=0, f1=0.5)
    run = summary(tmp_path, output)
    assert run.eligible, run.invalid_reasons
    assert run.f1_score == 0.5
    assert run.accuracy == 0
    assert run.primary_score == 0
    assert run.expected_sample_count == 2  # The selected stream, not hard-coded 150.
    generate_report(tmp_path, tmp_path / "statistics")
    assert "50.00%" in (tmp_path / "statistics" / "summary.md").read_text()


@pytest.mark.parametrize("ids", [["1", "0"], ["0", "0"], ["0", "different"]])
def test_equal_counts_do_not_hide_wrong_sample_ids_or_order(tmp_path, ids):
    output = write_run(tmp_path, "bad")
    rewrite(
        output / "runs.json",
        lambda runs: [run.update(sample_index=index) for run, index in zip(runs, ids)],
    )
    assert not validate_output(output)[0]
    assert "sample IDs or order differ" in summary(tmp_path, output).invalid_reasons


@pytest.mark.parametrize(
    "status",
    [
        "agent_unknown_error",
        "task_environment_error",
        "task_unknown_error",
        "agent_out_of_memory",
    ],
)
def test_infrastructure_errors_never_enter_primary_table(tmp_path, status):
    output = write_run(tmp_path, "bad")
    rewrite(output / "runs.json", lambda runs: runs[0].update(sample_status=status))
    run = summary(tmp_path, output)
    assert not run.eligible
    assert not _aggregate([run])


def test_evaluator_error_with_completed_status_is_invalid(tmp_path):
    output = write_run(tmp_path, "unknown")
    rewrite(
        output / "runs.json",
        lambda runs: runs[0]["evaluation_record"].update(outcome="unknown"),
    )
    assert "unknown/unset evaluation" in summary(tmp_path, output).invalid_reasons


def test_normal_validation_and_context_budget_failures_remain_valid(tmp_path):
    output = write_run(tmp_path, "valid-failures", correct=0)
    rewrite(
        output / "runs.json",
        lambda runs: [
            runs[0].update(sample_status="agent_validation_failed"),
            runs[1].update(sample_status="agent_context_limit"),
        ],
    )
    assert summary(tmp_path, output).eligible


def test_saved_metrics_are_checked_against_actual_sessions(tmp_path):
    output = write_run(tmp_path, "wrong-metric", task="knowledge_graph")
    rewrite(
        output / "metric.json",
        lambda metric: metric["overall"]["additional"].update(f1_score=0.9),
    )
    assert "metric F1 differs" in summary(tmp_path, output).invalid_reasons


def test_repeats_are_averaged_without_best_selection_or_protocol_mixing(tmp_path):
    first = summary(tmp_path, write_run(tmp_path, "first", count=10, correct=9))
    second = summary(tmp_path, write_run(tmp_path, "second", count=10, correct=1))
    other_model = summary(
        tmp_path, write_run(tmp_path, "other-model", protocol="other-model-protocol")
    )
    other_config = summary(
        tmp_path,
        write_run(tmp_path, "other-config", method_hash="other-hyperparameters"),
    )
    rows = _aggregate([first, second, other_model, other_config])
    assert len(rows) == 3
    same = next(row for row in rows if row["repeats"] == 2)
    assert same["mean"] == 0.5
    assert same["std"] == pytest.approx(0.565685424949238)
    assert json.loads(same["outputs"]) == ["first", "second"]


def test_copied_result_is_not_an_independent_repetition(tmp_path):
    first = summary(tmp_path, write_run(tmp_path, "first"))
    copied = summary(tmp_path, write_run(tmp_path, "copied"))
    copied.run_id = first.run_id
    assert _aggregate([first, copied])[0]["repeats"] == 1


def test_all_attempts_and_auxiliary_tokens_are_counted_but_missing_usage_is_unknown(
    tmp_path,
):
    output = write_run(tmp_path, "cost")
    entries = [
        {
            "sample_index": "0",
            "phase": "agent",
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
            "latency_seconds": 1,
        },
        {
            "sample_index": "0",
            "phase": "auxiliary",
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9,
            "latency_seconds": 2,
        },
        {
            "sample_index": "not-committed",
            "phase": "agent",
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12,
            "latency_seconds": 3,
        },
    ]
    ledger = output / "usage.jsonl"
    ledger.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    run = summary(tmp_path, output)
    assert run.usage_complete
    assert run.total_completion_tokens == 9
    assert run.agent_completion_tokens == 7
    assert run.auxiliary_completion_tokens == 2
    assert run.output_tokens_per_task == 4.5
    assert run.call_count == 3
    with ledger.open("a") as file:
        file.write(
            json.dumps({"phase": "agent", "missing_usage": True, "status": "error"})
            + "\n"
        )
    run = summary(tmp_path, output)
    assert run.call_count == 4
    assert not run.usage_complete
    assert run.output_tokens_per_task is None
    assert (
        run.total_completion_tokens == 9
    )  # Known subtotal is retained and clearly qualified.


def test_legacy_outputs_without_runtime_identity_are_diagnostics_only(tmp_path):
    output = write_run(tmp_path, "legacy")
    (output / "runtime_manifest.json").unlink()
    assert not summary(tmp_path, output).eligible


def test_budget_checks_main_and_auxiliary_total_for_each_task(tmp_path):
    output = write_run(tmp_path, "over-budget")
    rewrite(
        output / "runtime_manifest.json",
        lambda value: value.update(protocol={"generation_budget": 128}),
    )
    entries = [
        {
            "sample_index": "0",
            "phase": phase,
            "request_id": phase,
            "attempt": 1,
            "prompt_tokens": 1,
            "completion_tokens": tokens,
            "total_tokens": tokens + 1,
            "latency_seconds": 0.1,
        }
        for phase, tokens in [("agent", 100), ("auxiliary", 29)]
    ]
    (output / "usage.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n"
    )
    (output / "request_attempts.jsonl").write_text(
        "\n".join(
            json.dumps({"request_id": entry["request_id"], "attempt": 1})
            for entry in entries
        )
        + "\n"
    )
    run = summary(tmp_path, output)
    assert not run.eligible
    assert "output exceeds generation budget" in run.invalid_reasons
    assert run.total_completion_tokens == 129


def test_request_killed_before_usage_arrives_is_invalid_and_cost_unknown(tmp_path):
    output = write_run(tmp_path, "inflight")
    (output / "request_attempts.jsonl").write_text(
        json.dumps({"request_id": "unfinished", "attempt": 1}) + "\n"
    )
    run = summary(tmp_path, output)
    assert not run.eligible
    assert "unknown request cost" in run.invalid_reasons
    assert run.call_count == 1
    assert not run.usage_complete
    assert run.output_tokens_per_task is None


def test_guard_events_are_not_network_calls_and_journal_does_not_double_count(tmp_path):
    output = write_run(tmp_path, "journal")
    attempt = {
        "sample_index": "0",
        "phase": "agent",
        "request_id": "actual",
        "attempt": 1,
        "status": "completed",
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
        "latency_seconds": 1,
    }
    guard = {
        "sample_index": "0",
        "phase": "auxiliary",
        "request_id": "guard",
        "attempt": 0,
        "status": "budget_exhausted",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0,
    }
    (output / "usage.jsonl").write_text(
        json.dumps(attempt) + "\n" + json.dumps(guard) + "\n"
    )
    (output / "request_attempts.jsonl").write_text(
        json.dumps({"request_id": "actual", "attempt": 1}) + "\n"
    )
    run = summary(tmp_path, output)
    assert run.eligible
    assert run.call_count == 1
    assert run.budget_exhausted_count == 1
    assert run.output_tokens_per_task == 1


def test_incomplete_preselected_repeats_are_not_final_comparisons(tmp_path):
    output = write_run(tmp_path, "only-one")
    absent = tmp_path / "not-finished"
    (tmp_path / "batch_manifest.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"source_config_path": "same-config", "output_dir": str(output)},
                    {"source_config_path": "same-config", "output_dir": str(absent)},
                ]
            }
        )
    )
    generate_report(tmp_path, tmp_path / "statistics")
    assert (tmp_path / "statistics" / "comparisons.csv").read_text() == ""
    repetitions = (tmp_path / "statistics" / "repetitions.csv").read_text()
    assert "False" in repetitions


def test_source_changes_separate_repetition_groups(tmp_path):
    first = summary(tmp_path, write_run(tmp_path, "first"))
    second = summary(tmp_path, write_run(tmp_path, "changed"))
    second.implementation_hash = "new-algorithm-source"
    assert len(_aggregate([first, second])) == 2


@pytest.mark.parametrize("injection", [None, -1, 901, True])
def test_paper_protocol_requires_a_valid_injection_count_within_budget(
    tmp_path, injection
):
    output = write_run(tmp_path, "bad-injection")
    rewrite(
        output / "runtime_manifest.json",
        lambda value: value.update(
            protocol={"resource_budget": {"injection_chars": 900}}
        ),
    )
    rewrite(
        output / "runs.json",
        lambda runs: [run.update(memory_injection_chars=injection) for run in runs],
    )
    assert not summary(tmp_path, output).eligible
    assert "injection" in summary(tmp_path, output).invalid_reasons


def test_injection_at_paper_limit_is_valid(tmp_path):
    output = write_run(tmp_path, "at-limit")
    rewrite(
        output / "runtime_manifest.json",
        lambda value: value.update(
            protocol={"resource_budget": {"injection_chars": 900}}
        ),
    )
    rewrite(
        output / "runs.json",
        lambda runs: [run.update(memory_injection_chars=900) for run in runs],
    )
    assert summary(tmp_path, output).eligible
    assert summary(tmp_path, output).max_memory_injection_chars == 900


def test_prompt_guard_without_network_requests_has_known_zero_cost(tmp_path):
    output = write_run(tmp_path, "only-guards", correct=0)
    entries = [
        {
            "sample_index": str(index),
            "phase": "agent",
            "request_id": f"guard-{index}",
            "attempt": 0,
            "status": "prompt_guard",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0,
        }
        for index in range(2)
    ]
    (output / "usage.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n"
    )
    rewrite(
        output / "runtime_manifest.json",
        lambda value: value.update(protocol={"generation_budget": 128}),
    )
    run = summary(tmp_path, output)
    assert run.eligible, run.invalid_reasons
    assert run.usage_complete
    assert run.call_count == 0
    assert run.output_tokens_per_task == 0


def test_exported_results_must_match_authoritative_checkpoint(tmp_path):
    from src.utils.checkpoint import CheckpointManager

    output = write_run(tmp_path, "checkpoint")
    manager = CheckpointManager(str(output))
    manager.commit(json.loads((output / "runs.json").read_text()))
    rewrite(
        output / "run_status.json",
        lambda status: status.update(checkpoint_generation=manager.generation),
    )
    assert summary(tmp_path, output).eligible
    rewrite(output / "runs.json", lambda runs: runs[0].update(memory_injection_chars=9))
    assert (
        "differ from committed checkpoint" in summary(tmp_path, output).invalid_reasons
    )
