from pathlib import Path
import subprocess
import unittest

import yaml

from continual_alfworld.methods import METHODS


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "local.yaml"
EXPECTED_METHODS = {
    "tokenworth",
    "rag",
    "memrl",
    "reflexion",
    "memp",
    "evosc",
    "evoskill",
    "mem0",
    "memskill",
    "no_memory",
}


def load_release_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_only_published_methods_are_registered():
    assert set(METHODS) == EXPECTED_METHODS


def test_release_config_is_flattened_and_portable():
    text = CONFIG_PATH.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    assert "extends" not in config
    for value in (
        config["experiment"]["output_root"],
        config["environment"]["alfworld_config"],
        config["environment"]["data_root"],
        config["model"]["env_file"],
    ):
        assert not Path(value).is_absolute()
    assert config["experiment"]["output_root"] == "outputs"
    assert config["environment"]["alfworld_config"] == "configs/alfworld_base.yaml"
    assert config["environment"]["data_root_env"] == "ALFWORLD_DATA_ROOT"


def test_published_protocol():
    config = load_release_config()
    assert config["experiment"]["seed"] == 42
    assert config["phases"]["train"] == {
        "enabled": True,
        "games": 300,
        "epochs": 1,
        "sampling": "stratified",
    }
    assert config["phases"]["valid_seen"]["games"] == -1
    assert config["phases"]["valid_unseen"]["games"] == -1
    assert config["model"]["name"] == "qwen2.5-14b-instruct"
    assert config["model"]["action_max_tokens"] == 128
    assert config["model"]["auxiliary_max_tokens"] == 512
    assert config["agent"]["max_steps"] == 50
    assert config["agent"]["loop_guard_enabled"] is False
    assert config["agent"]["memory_prompt_format"] == "method_reference"
    assert config["memory"]["update_phases"] == ["train", "valid_seen", "valid_unseen"]


def test_shared_hash_protocol():
    config = load_release_config()
    assert config["retrieval"] == {
        "backend": "tokenworth_feature_hashing_cosine",
        "dimensions": 512,
        "top_k": 1,
        "minimum_score": 0.0,
        "character_ngram_min": 3,
        "character_ngram_max": 5,
    }
    assert config["methods"]["rag"]["retrieval_backend"] == (
        "tokenworth_feature_hashing_cosine"
    )
    assert config["methods"]["tokenworth"]["implementation_variant"] == "paper"
    assert config["methods"]["tokenworth"]["candidate_count"] == 12
    assert config["methods"]["tokenworth"]["base_update_interval"] == 50
    assert config["methods"]["tokenworth"]["positive_ratio_threshold"] == 0.45


def test_no_results_or_model_weights_are_published():
    forbidden_names = {
        "summary.json",
        "trajectories.jsonl",
        "method_state.json",
        "game_manifest.json",
        "resolved_config.yaml",
    }
    # Only published files matter: an installed venv or a user's ignored local
    # run must not be mistaken for a committed release artifact.
    tracked = subprocess.run(
        ["git", "ls-files", "--full-name", "--", ROOT.name],
        cwd=ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    paths = [Path(path) for path in tracked]
    assert not any(path.name in forbidden_names for path in paths)
    assert not any(path.suffix in {".pt", ".bin", ".safetensors"} for path in paths)
    assert not any(path.parts[:2] == (ROOT.name, "outputs") for path in paths)


class ReleaseTests(unittest.TestCase):
    def test_registered_method_inventory(self):
        test_only_published_methods_are_registered()

    def test_flattened_portable_config(self):
        test_release_config_is_flattened_and_portable()

    def test_protocol_values(self):
        test_published_protocol()

    def test_hash_settings(self):
        test_shared_hash_protocol()

    def test_artifact_exclusion(self):
        test_no_results_or_model_weights_are_published()


if __name__ == "__main__":
    unittest.main()
