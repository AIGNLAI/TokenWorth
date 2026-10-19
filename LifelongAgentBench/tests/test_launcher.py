from __future__ import annotations

import json
import sys
import httpx
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import run_experiments as launcher
from src.utils.checkpoint import CheckpointManager
from src.utils.runtime import (
    implementation_hash,
    resolved_model_identity,
    UncommittedInferenceError,
)


def args_for(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", *map(str, args)])
    return launcher.parse_args()


def config_file(tmp_path: Path, *, task="knowledge_graph") -> Path:
    path = tmp_path / "fixture.yaml"
    config = {
        "assignment_config": {
            "task": task,
            "sample_order": ["0"],
            "language_model_list": [{"name": "model"}],
            "callback_dict": {},
            "agent": {"name": "agent"},
        },
        "language_model_dict": {
            "model": {
                "module": "src.language_models.instance.openai_language_model.OpenaiLanguageModel",
                "parameters": {
                    "model_name": "env:TEST_BATCH_MODEL",
                    "api_key": "EMPTY",
                    "trust_env": False,
                },
            }
        },
        "task_dict": {
            task: {
                "module": "unused.fixture",
                "parameters": {"sparql_url": "http://old/sparql"},
            }
        },
    }
    path.write_text(yaml.safe_dump(config))
    return path


def write_completed(spec: launcher.RunSpec, *, task="knowledge_graph") -> None:
    config = yaml.safe_load(spec.generated_config_path.read_text())
    (spec.output_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (spec.output_dir / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "task": task,
                "method": "standard",
                "sample_order": ["0"],
                "protocol_hash": "protocol",
                "method_config_hash": "config",
                "run_id": spec.name,
                "primary_metric": "success_rate",
                "model_identity": resolved_model_identity(config),
                "implementation_hash": implementation_hash(launcher.REPO_ROOT),
            }
        )
    )
    (spec.output_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "valid": True,
                "expected_sample_count": 1,
                "completed_sample_count": 1,
                "invalid_reasons": [],
            }
        )
    )
    sessions = [
        {
            "task_name": task,
            "sample_index": "0",
            "sample_status": "completed",
            "evaluation_record": {
                "outcome": "correct",
                "detail_dict": {"f1_score": 1.0},
            },
        }
    ]
    (spec.output_dir / "runs.json").write_text(json.dumps(sessions))
    (spec.output_dir / "metric.json").write_text(
        json.dumps(
            {
                "overall": {
                    "evaluation_outcome": {"correct": 1.0},
                    "additional": {"f1_score": 1.0},
                }
            }
        )
    )


def test_uniform_overrides_and_repeats_persist_in_batch_manifest(tmp_path, monkeypatch):
    source = config_file(tmp_path)
    args = args_for(
        monkeypatch,
        source,
        "--output-root",
        tmp_path / "batch",
        "--repeats",
        2,
        "--kg-sparql-url",
        "http://new/sparql",
        "--llm-model-name",
        "fixed-model",
        "--llm-base-url",
        "http://new/v1",
        "--llm-timeout",
        9,
    )
    specs = launcher._make_run_specs(args)
    assert len(specs) == 2
    for spec in specs:
        config = yaml.safe_load(spec.generated_config_path.read_text())
        assert (
            config["task_dict"]["knowledge_graph"]["parameters"]["sparql_url"]
            == "http://new/sparql"
        )
        assert (
            config["language_model_dict"]["model"]["parameters"]["model_name"]
            == "fixed-model"
        )
        assert config["language_model_dict"]["model"]["parameters"]["timeout"] == 9
    assert (
        len(json.loads((args.batch_root / "batch_manifest.json").read_text())["runs"])
        == 2
    )
    with pytest.raises(ValueError, match="use --resume"):
        launcher._make_run_specs(args)


def test_resume_skips_complete_and_reuses_unfinished_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BATCH_MODEL", "original")
    source = config_file(tmp_path)
    args = args_for(
        monkeypatch, source, "--output-root", tmp_path / "batch", "--repeats", 2
    )
    specs = launcher._make_run_specs(args)
    write_completed(specs[0])
    resumed = args_for(monkeypatch, "--resume", args.batch_root)
    pending = launcher._make_run_specs(resumed)
    assert [spec.output_dir for spec in pending] == [specs[1].output_dir]
    monkeypatch.setenv("TEST_BATCH_MODEL", "different")
    with pytest.raises(ValueError, match="identity changed"):
        launcher._make_run_specs(resumed)


def test_resume_refuses_modified_saved_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BATCH_MODEL", "original")
    args = args_for(
        monkeypatch, config_file(tmp_path), "--output-root", tmp_path / "batch"
    )
    spec = launcher._make_run_specs(args)[0]
    with spec.generated_config_path.open("a") as stream:
        stream.write("# changed\n")
    resumed = args_for(monkeypatch, "--resume", args.batch_root)
    with pytest.raises(ValueError, match="config changed"):
        launcher._make_run_specs(resumed)


def test_dirty_inference_cannot_be_replayed_by_batch_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BATCH_MODEL", "original")
    args = args_for(
        monkeypatch, config_file(tmp_path), "--output-root", tmp_path / "batch"
    )
    spec = launcher._make_run_specs(args)[0]
    CheckpointManager(str(spec.output_dir)).commit([])
    (spec.output_dir / "usage.jsonl").write_text(
        json.dumps({"sample_index": "0", "attempt": 1, "completion_tokens": 3}) + "\n"
    )
    resumed = args_for(monkeypatch, "--resume", args.batch_root)
    with pytest.raises(UncommittedInferenceError, match="uncommitted model request"):
        launcher._make_run_specs(resumed)


def test_zero_exit_without_complete_results_is_a_batch_failure(tmp_path, monkeypatch):
    args = args_for(
        monkeypatch,
        config_file(tmp_path),
        "--output-root",
        tmp_path / "batch",
        "--poll-interval",
        ".01",
    )
    spec = launcher._make_run_specs(args)[0]
    fake = launcher.RunSpec(
        spec.name,
        spec.source_config_path,
        spec.generated_config_path,
        spec.output_dir,
        spec.log_path,
        [sys.executable, "-c", "pass"],
        None,
    )
    assert launcher._run_specs([fake], args) == 1
    record = args.batch_manifest["runs"][0]
    assert record["status"] == "failed"
    assert "missing config.yaml" in record["reason"]


def test_valid_zero_exit_is_accepted_and_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BATCH_MODEL", "original")
    args = args_for(
        monkeypatch,
        config_file(tmp_path),
        "--output-root",
        tmp_path / "batch",
        "--poll-interval",
        ".01",
    )
    spec = launcher._make_run_specs(args)[0]
    write_completed(spec)
    fake = launcher.RunSpec(
        spec.name,
        spec.source_config_path,
        spec.generated_config_path,
        spec.output_dir,
        spec.log_path,
        [sys.executable, "-c", "pass"],
        None,
    )
    assert launcher._run_specs([fake], args) == 0
    launcher.generate_report(args.batch_root, args.batch_root / "statistics")
    assert (args.batch_root / "statistics" / "summary.md").is_file()


def test_preflight_sends_a_completion_and_fails_fast_on_503(tmp_path, monkeypatch):
    import openai  # Load SDK classes before replacing the transport client.

    seen, state = [], {"status": 200}

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        if state["status"] == 200:
            response = {
                "id": "probe",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 5,
                },
            }
        else:
            response = {
                "error": {"message": "fixture unavailable", "type": "server_error"}
            }
        return httpx.Response(state["status"], json=response)

    original_client = httpx.Client

    class MockClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "Client", MockClient)
    monkeypatch.setenv("TEST_BATCH_MODEL", "fixture-model")
    config = yaml.safe_load(config_file(tmp_path).read_text())
    config["language_model_dict"]["model"]["parameters"][
        "base_url"
    ] = "http://fixture/v1"
    launcher._model_preflight(config)
    assert len(seen) == 1
    assert seen[0]["model"] == "fixture-model"
    assert seen[0]["max_tokens"] == 1
    assert not list(tmp_path.rglob("usage.jsonl"))
    state["status"] = 503
    with pytest.raises(RuntimeError, match="preflight failed"):
        launcher._model_preflight(config)
    assert len(seen) == 2  # No hidden SDK retries.


def test_preflight_validates_data_and_each_distinct_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BATCH_MODEL", "fixture-model")
    source = config_file(tmp_path)
    args = args_for(
        monkeypatch, source, "--output-root", tmp_path / "batch", "--repeats", 2
    )
    specs = launcher._make_run_specs(args)
    calls = []
    monkeypatch.setattr(
        launcher, "_model_preflight", lambda config: calls.append("model")
    )
    monkeypatch.setattr(launcher, "_kg_preflight", lambda url: calls.append(url))
    launcher._preflight(specs, args)
    assert calls == ["model", "http://old/sparql"]
    assert all(record["model_identity"] for record in args.batch_manifest["runs"])


def test_image_overrides_are_written_to_effective_task_config(tmp_path, monkeypatch):
    source = config_file(tmp_path, task="os_interaction")
    args = args_for(
        monkeypatch,
        source,
        "--output-root",
        tmp_path / "batch",
        "--os-image",
        "review-os:fixed",
    )
    spec = launcher._make_run_specs(args)[0]
    config = yaml.safe_load(spec.generated_config_path.read_text())
    assert (
        config["task_dict"]["os_interaction"]["parameters"]["container_image"]
        == "review-os:fixed"
    )


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0},
        {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 9},
    ],
)
def test_unverifiable_or_over_budget_provider_fails_before_any_child(
    tmp_path, monkeypatch, usage
):
    import openai

    def respond(request):
        response = {
            "id": "probe",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        return httpx.Response(200, json=response)

    original_client = httpx.Client

    class MockClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "Client", MockClient)
    monkeypatch.setenv("TEST_BATCH_MODEL", "fixture")
    source = config_file(tmp_path)
    args_for(
        monkeypatch,
        source,
        "--output-root",
        tmp_path / "batch",
        "--llm-base-url",
        "http://fixture/v1",
    )
    children = []
    monkeypatch.setattr(launcher, "_run_specs", lambda *args: children.append(args))
    assert launcher.main() == 1
    assert not children
    manifest = json.loads((tmp_path / "batch" / "batch_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "preflight" in manifest["failure_reason"]


def test_preflight_can_compute_missing_total_from_verified_prompt_and_completion(
    tmp_path, monkeypatch
):
    import openai

    def respond(request):
        return httpx.Response(
            200,
            json={
                "id": "probe",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    original_client = httpx.Client

    class MockClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "Client", MockClient)
    monkeypatch.setenv("TEST_BATCH_MODEL", "fixture")
    config = yaml.safe_load(config_file(tmp_path).read_text())
    config["language_model_dict"]["model"]["parameters"][
        "base_url"
    ] = "http://fixture/v1"
    launcher._model_preflight(config)
