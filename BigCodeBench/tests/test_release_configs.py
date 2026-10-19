import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
LOCAL_METHODS = {
    "no_memory",
    "rag",
    "mem0",
    "memp",
    "memrl",
    "memskill",
    "reflexion",
    "evoskill",
    "evosc",
    "tokenworth",
}
API_METHODS = set(LOCAL_METHODS)


def load_suite(name: str) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((CONFIG_ROOT / name).glob("*.json"))
    }


@pytest.mark.parametrize(
    ("suite", "expected"),
    [
        ("local", LOCAL_METHODS),
        ("api_no_thinking", API_METHODS),
        ("api_thinking", API_METHODS),
    ],
)
def test_published_inventory(suite, expected):
    assert set(load_suite(suite)) == expected


def test_configs_are_flattened_and_portable():
    for path in CONFIG_ROOT.glob("*/*.json"):
        text = path.read_text(encoding="utf-8")
        config = json.loads(text)
        assert "extends" not in config
        assert not Path(config["evaluation"]["dataset_path"]).is_absolute()
        assert not Path(config["run"]["output_dir"]).is_absolute()
        assert config["model"]["base_url"] == "${OPENAI_BASE_URL}"
        assert config["evaluation"]["dataset_path"] == "${BIGCODEBENCH_OVERRIDE_PATH}"
        assert config["run"]["output_dir"] == f"outputs/{path.parent.name}/{path.stem}"


def test_local_protocol():
    configs = load_suite("local")
    for config in configs.values():
        assert config["model"]["name"] == "qwen2.5-14b-instruct"
        assert config["model"]["max_tokens"] == 1280
        assert "thinking_mode" not in config["model"]
    assert (
        configs["tokenworth"]["method_parameters"]["implementation_variant"] == "paper"
    )
    assert configs["evosc"]["method_parameters"] == {
        "allow_zero_retrieval": True,
        "max_context_chars": 4096,
        "top_k": 1,
        "top_k_success": 1,
        "top_k_failure": 1,
        "min_retrieval_score": 0.1,
        "max_success_count": 20,
        "max_failure_count": 20,
        "max_summary_count": 10,
        "min_experience_chars": 0,
        "max_experience_chars": 4096,
        "min_summary_chars": 0,
        "max_summary_chars": 4096,
        "auxiliary_max_tokens": 1280,
    }


@pytest.mark.parametrize(
    ("suite", "thinking_mode", "max_tokens"),
    [
        ("api_no_thinking", "disabled", 1280),
        ("api_thinking", "enabled", 4096),
    ],
)
def test_api_protocol(suite, thinking_mode, max_tokens):
    configs = load_suite(suite)
    for name, config in configs.items():
        assert config["model"]["name"] == "deepseek/deepseek-v4-flash"
        assert config["model"]["thinking_mode"] == thinking_mode
        assert config["model"]["max_tokens"] == max_tokens
        if name == "tokenworth":
            assert config["method_parameters"]["implementation_variant"] == "paper"
            assert config["method_parameters"]["candidate_count"] == 12
            assert config["method_parameters"]["base_update_interval"] == 50
            assert config["method_parameters"]["base_regularization"] == 0.1
        elif name != "no_memory":
            assert (
                config["method_parameters"]["retrieval_variant"]
                == "tokenworth_hash_cosine"
            )
        assert config["model"]["max_retries"] == 0
        assert config["model"]["transient_max_retries"] >= 0
    if thinking_mode == "enabled":
        for name, config in configs.items():
            if name != "no_memory":
                assert config["auxiliary"]["thinking_mode"] == "disabled"


def test_no_results_are_published():
    forbidden = {"results.jsonl", "results_summary.json", "method_state.json"}
    tracked = subprocess.run(
        ["git", "ls-files", "--full-name", "--", ROOT.name],
        cwd=ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not any(Path(path).name in forbidden for path in tracked)
    assert not any(Path(path).parts[:2] == (ROOT.name, "outputs") for path in tracked)
