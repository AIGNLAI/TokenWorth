import json

import pytest

from experiments.methods import METHODS, TaskOutcome, create_method
from experiments.methods.shared_retrieval import HASH_COSINE_VARIANT


EXPECTED_METHODS = {
    "RAG",
    "Reflexion",
    "Mem0",
    "MemP",
    "MemRL",
    "MemSkill",
    "EvoSC",
    "EvoSkill",
    "TokenWorth",
    "NoMemory",
}


def auxiliary(prompt: str) -> str:
    if "Mem0" in prompt:
        return json.dumps(
            {
                "memories": [
                    {
                        "memory": "Validate edge cases.",
                        "category": "procedure",
                        "entities": [],
                    }
                ]
            }
        )
    if "detailed_steps" in prompt:
        return json.dumps(
            {
                "detailed_steps": ["Validate inputs."],
                "abstractions": ["Separate parsing."],
            }
        )
    if "cautions" in prompt:
        return json.dumps({"cautions": ["Check empty inputs."], "deprecate": []})
    if "MemSkill" in prompt:
        return json.dumps(
            {"actions": [{"action": "insert", "memory_item": "Validate inputs."}]}
        )
    if "Return JSON" in prompt and "name" in prompt:
        return json.dumps(
            {
                "name": "validation",
                "description": "Validate boundaries",
                "triggers": ["empty input"],
                "instruction": "Check empty values first.",
                "tags": ["validation"],
                "examples": [],
            }
        )
    return "- Validate boundary conditions.\n- Prefer explicit error handling."


def outcome() -> TaskOutcome:
    return TaskOutcome(
        epoch=1,
        phase="simple",
        task_id="BigCodeBench/0",
        prompt="Write a function that parses JSON and handles empty input.",
        code="def task_func(value): return value",
        status="pass",
        error="",
    )


def test_only_published_methods_are_registered():
    assert set(METHODS) == EXPECTED_METHODS


@pytest.mark.parametrize("method_name", sorted(EXPECTED_METHODS))
def test_methods_round_trip_state(tmp_path, method_name):
    state_dir = tmp_path / method_name
    method = create_method(method_name, state_dir, {}, auxiliary)
    method.context(method.retrieve(outcome().prompt))
    method.apply(outcome())
    method.save()
    restored = create_method(method_name, state_dir, {}, auxiliary)
    assert restored.step == 1
    assert restored.state_path.exists()


@pytest.mark.parametrize("variant", ["simplified", "full"])
def test_legacy_tokenworth_variants_are_rejected(tmp_path, variant):
    with pytest.raises(ValueError, match="Legacy"):
        create_method("TokenWorth", tmp_path, {"implementation_variant": variant})


@pytest.mark.parametrize(
    "method_name", sorted(EXPECTED_METHODS - {"TokenWorth", "NoMemory"})
)
def test_hash_cosine_variant_is_available(tmp_path, method_name):
    method = create_method(
        method_name,
        tmp_path / method_name,
        {"retrieval_variant": HASH_COSINE_VARIANT},
        auxiliary,
    )
    assert method.summary()["retrieval_variant"] == HASH_COSINE_VARIANT
