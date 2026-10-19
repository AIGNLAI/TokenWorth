import copy
import json
import math

import pytest

from experiments.methods import TaskOutcome, create_method
from experiments.method_runner import MethodExperimentRunner
from tokenworth_core.runtime import CheckpointStore


def outcome(index=0, status="pass", code="def task_func(): return 1"):
    return TaskOutcome(1, "simple", str(index), "Return one.", code, status, "")


def test_cached_residual_positive_bank_and_no_auxiliary_calls(tmp_path):
    def forbidden(_):
        raise AssertionError("TokenWorth must not call another LLM")

    method = create_method("TokenWorth", tmp_path, {"embedding_dim": 8}, forbidden)
    method.context(method.retrieve(outcome().prompt))
    method.apply(outcome())
    items = method.retrieve(outcome().prompt)
    cached = copy.deepcopy(method.core._selected[0])
    assert len(method.context(items)) <= 4096
    method.apply(outcome(1, "fail"))
    method.save()
    pair = method.core.state["pairs"][0]
    assert pair["label"] == pytest.approx(-1 - math.tanh(cached["base_score"] / 2))
    assert pair["feature"] == cached["feature"]
    assert len(method.core.state["memories"]) == 1
    restored = create_method("TokenWorth", tmp_path, {"embedding_dim": 8}, forbidden)
    assert restored.core.state == method.core.state


def test_legacy_state_and_infrastructure_rewards_are_rejected(tmp_path):
    method = create_method("TokenWorth", tmp_path, {"embedding_dim": 8})
    method.context(method.retrieve("Return one"))
    with pytest.raises(RuntimeError, match="infrastructure"):
        method.apply(outcome(status="eval_error"))
    assert method.core.state["step"] == 0
    (tmp_path / "method_state.json").write_text(
        json.dumps({"method": "TokenWorth", "step": 1, "state": {"memories": []}})
    )
    with pytest.raises(ValueError, match="legacy|Legacy"):
        create_method("TokenWorth", tmp_path, {"embedding_dim": 8})


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from experiments import method_runner as module

    class Generator:
        calls = 0
        fail_at = None

        def __init__(self, config):
            self.last_generation_metadata = {}

        def generate(self, *args):
            type(self).calls += 1
            if self.fail_at == type(self).calls:
                raise RuntimeError("service fault")
            self.last_generation_metadata = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
            return "```python\ndef task_func():\n    return 1\n```"

    class Evaluator:
        status = "pass"

        def __init__(self, config, split):
            self.config = config

        def evaluate(self, *args):
            return self.status, ""

        def close(self):
            pass

    tasks = {
        str(i): {
            "instruct_prompt": "Return one",
            "entry_point": "task_func",
            "test": "",
            "code_prompt": "def task_func():",
        }
        for i in range(2)
    }
    monkeypatch.setattr(module, "OpenAICompatibleGenerator", Generator)
    monkeypatch.setattr(module, "TaskEvaluator", Evaluator)
    monkeypatch.setattr(module, "load_ordering_datasets", lambda: (tasks, {}))
    config = {
        "method": "TokenWorth",
        "model": {"name": "fixed-test-model", "base_url": "http://example.invalid"},
        "evaluation": {"backend": "gradio"},
        "method_parameters": {"embedding_dim": 8},
        "prompt": {"instruction_prefix": "Write Python"},
        "run": {"output_dir": str(tmp_path), "epochs": 1, "resume": True},
    }
    return config, Generator, Evaluator


def test_partial_resume_and_completed_resume_make_no_extra_calls(harness, tmp_path):
    config, generator, _ = harness
    first = MethodExperimentRunner(copy.deepcopy(config))
    first.run(1)
    assert not json.loads((tmp_path / "results_summary.json").read_text())[
        "is_complete"
    ]
    MethodExperimentRunner(copy.deepcopy(config)).run()
    assert generator.calls == 2
    store = CheckpointStore(tmp_path)
    assert store.method["state"]["paper"]["metrics"]["pair_writes"] == 1
    assert json.loads((tmp_path / "results_summary.json").read_text())["is_complete"]
    MethodExperimentRunner(copy.deepcopy(config)).run()
    assert generator.calls == 2


def test_committed_crash_recovers_without_replay(harness, tmp_path, monkeypatch):
    config, generator, _ = harness
    runner = MethodExperimentRunner(copy.deepcopy(config))
    monkeypatch.setattr(
        runner.method,
        "save",
        lambda: (_ for _ in ()).throw(RuntimeError("after commit")),
    )
    with pytest.raises(RuntimeError, match="after commit"):
        runner.run(1)
    assert (tmp_path / "active_task.json").exists()
    MethodExperimentRunner(copy.deepcopy(config)).run()
    assert generator.calls == 2
    assert len(CheckpointStore(tmp_path).records) == 2


def test_uncommitted_request_failure_refuses_replay(harness, tmp_path):
    config, generator, _ = harness
    MethodExperimentRunner(copy.deepcopy(config)).run(1)
    generator.fail_at = 2
    with pytest.raises(RuntimeError, match="service fault"):
        MethodExperimentRunner(copy.deepcopy(config)).run()
    with pytest.raises(RuntimeError, match="replay refused"):
        MethodExperimentRunner(copy.deepcopy(config)).run()
    assert generator.calls == 2
    assert len(CheckpointStore(tmp_path).records) == 1


def test_model_identity_drift_rejected_before_call(harness):
    config, generator, _ = harness
    MethodExperimentRunner(copy.deepcopy(config)).run(1)
    altered = copy.deepcopy(config)
    altered["model"]["base_url"] = "http://different.invalid"
    with pytest.raises(RuntimeError, match="identity changed"):
        MethodExperimentRunner(altered).run()
    assert generator.calls == 1


def test_evaluator_fault_never_updates_or_completes(harness, tmp_path):
    config, generator, evaluator = harness
    evaluator.status = "eval_error"
    runner = MethodExperimentRunner(copy.deepcopy(config))
    with pytest.raises(RuntimeError, match="valid external reward"):
        runner.run()
    assert runner.method.core.state["step"] == 0
    assert not (tmp_path / "checkpoint.json").exists()
    assert not (tmp_path / "results_summary.json").exists()


def test_corrupted_result_is_detected(harness, tmp_path):
    config, _, _ = harness
    MethodExperimentRunner(copy.deepcopy(config)).run(1)
    path = next((tmp_path / "records").iterdir())
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        MethodExperimentRunner(copy.deepcopy(config)).run()


def test_long_solution_is_an_outline_not_a_truncated_verified_program(tmp_path):
    method = create_method(
        "TokenWorth", tmp_path, {"embedding_dim": 8, "max_memory_chars": 256}
    )
    code = "def task_func():\n" + "    x=1\n" * 100 + "    return x\n"
    method.context(method.retrieve("Return one"))
    method.apply(outcome(code=code))
    snippet = method.core.state["memories"][0]["snippet"]
    assert len(snippet) <= 256
    assert "outline (not executable code)" in snippet
    assert "Verified Python solution" not in snippet


def test_truncated_rendered_memory_is_labeled_as_incomplete(tmp_path):
    method = create_method(
        "TokenWorth", tmp_path, {"embedding_dim": 8, "max_context_chars": 256}
    )
    code = "def task_func():\n" + "    x=1\n" * 100 + "    return x\n"
    method.context(method.retrieve("Return one"))
    method.apply(outcome(code=code))
    context = method.context(method.retrieve("Return one"))
    assert len(context) <= 256
    assert "Incomplete excerpt" in context
