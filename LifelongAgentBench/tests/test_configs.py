from importlib import import_module
from pathlib import Path
import re
from typing import Any

from run import ConfigUtility, apply_language_model_overrides
from src.typings import AssignmentConfig
from src.utils import ConfigLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = {
    "db": ("db_bench", 3, 128),
    "knowledge_graph": ("knowledge_graph", 15, 128),
    "os_interaction": ("os_interaction", 5, 512),
}
OUTPUT_BENCHMARK_NAMES = {
    "db": "db",
    "knowledge_graph": "knowledge_graph",
    "os_interaction": "os",
}
METHODS = {
    "EvoSC",
    "EvoSkill",
    "Mem0",
    "MemP",
    "MemRL",
    "MemSkill",
    "RAG",
    "Reflexion",
    "TokenWorth",
    "standard",
}


def _module_paths(value: Any):
    if isinstance(value, dict):
        module_path = value.get("module")
        if isinstance(module_path, str):
            yield module_path
        for child in value.values():
            yield from _module_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from _module_paths(child)


def test_config_layout() -> None:
    config_root = PROJECT_ROOT / "configs"
    assert {path.name for path in config_root.iterdir()} == {
        "tasks",
        "methods",
        "examples",
        "experiments",
        "utils",
    }
    assert {path.stem for path in (config_root / "tasks").glob("*.yaml")} == {
        "db",
        "knowledge_graph",
        "os_interaction",
    }
    assert {path.stem for path in (config_root / "methods").glob("*.yaml")} == (
        METHODS - {"standard"}
    ) | {"abnormal", "saving"}
    data_root = PROJECT_ROOT / "data/lifelongagent_bench"
    assert {path.name for path in data_root.iterdir() if path.is_dir()} == {
        "db",
        "kg",
        "os",
    }


def test_experiment_matrix_resolves_to_fixed_protocol() -> None:
    config_count = 0
    for directory, (task_name, max_round, max_tokens) in BENCHMARKS.items():
        paths = sorted(
            (PROJECT_ROOT / "configs/experiments" / directory).glob("*.yaml")
        )
        assert {path.stem for path in paths} == METHODS

        for path in paths:
            raw_config = ConfigLoader().load_from(path)
            assignment, _, _ = ConfigUtility.read_raw_config(raw_config)
            task_parameters = raw_config["task_dict"][task_name]["parameters"]

            assert raw_config["assignment_config"]["task"] == task_name
            expected_output_dir = (
                f"outputs/runs/{path.stem}-{OUTPUT_BENCHMARK_NAMES[directory]}-"
                "{TIMESTAMP_MS}"
            )
            assert raw_config["assignment_config"]["output_dir"] == expected_output_dir
            assert re.fullmatch(
                rf"outputs/runs/{re.escape(path.stem)}-"
                rf"{OUTPUT_BENCHMARK_NAMES[directory]}-"
                r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}",
                assignment.output_dir,
            )
            assert task_parameters["max_round"] == max_round
            assert (
                assignment.agent.parameters["inference_config_dict"]["max_tokens"]
                == max_tokens
            )
            assert (PROJECT_ROOT / task_parameters["data_file_path"]).is_file()
            system_prompt_path = task_parameters["system_prompt_factory"]["parameters"][
                "system_prompt_dict_path"
            ]
            assert (PROJECT_ROOT / system_prompt_path).is_file()

            if task_name == "knowledge_graph":
                assert task_parameters["allow_final_answer_after_round_limit"] is True
                assert (PROJECT_ROOT / task_parameters["ontology_dir_path"]).is_dir()

            for module_path in set(_module_paths(raw_config)):
                module_name, attribute_name = module_path.rsplit(".", 1)
                assert hasattr(import_module(module_name), attribute_name)
            config_count += 1

    assert config_count == 30


def test_smoke_config_and_model_overrides() -> None:
    raw_config = ConfigLoader().load_from(
        PROJECT_ROOT / "configs/examples/TokenWorth_db_smoke.yaml"
    )
    assignment, _, _ = ConfigUtility.read_raw_config(raw_config)
    assert list(assignment.sample_order) == ["0", "1", "2"]
    assert re.fullmatch(
        r"outputs/runs/smoke/TokenWorth-db-"
        r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}",
        assignment.output_dir,
    )

    kg_config = ConfigLoader().load_from(
        PROJECT_ROOT / "configs/examples/TokenWorth_knowledge_graph_smoke.yaml"
    )
    kg_assignment, _, _ = ConfigUtility.read_raw_config(kg_config)
    assert list(kg_assignment.sample_order) == ["0"]
    assert kg_config["assignment_config"]["task"] == "knowledge_graph"
    assert kg_config["task_dict"]["knowledge_graph"]["parameters"]["max_round"] == 15

    overridden = apply_language_model_overrides(
        raw_config,
        language_model_name=None,
        model_name="provider/model",
        base_url="https://provider.example/v1",
        api_key="env:OPENAI_API_KEY",
        maximum_prompt_token_count=8192,
        timeout=60,
        max_retries=2,
        trust_env=False,
    )
    parameters = overridden["language_model_dict"]["experiment_model"]["parameters"]
    assert parameters["model_name"] == "provider/model"
    assert parameters["base_url"] == "https://provider.example/v1"
    assert parameters["api_key"] == "env:OPENAI_API_KEY"
    assert parameters["trust_env"] is False


def test_millisecond_output_directory_comparison() -> None:
    assert AssignmentConfig.is_output_dir_equal(
        "outputs/runs/EvoSC-db-{TIMESTAMP_MS}",
        "outputs/runs/EvoSC-db-2026-08-24-10-15-30-123",
    )
