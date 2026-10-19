from pathlib import Path
from run import ConfigUtility
from src.utils import ConfigLoader
from src.language_models.language_model import LanguageModel


class UncalledModel(LanguageModel):
    def __init__(self):
        super().__init__({"user": "user", "agent": "assistant"})

    def _inference(self, *args):
        raise AssertionError("Constructor validation must not perform inference.")


def test_all_matrix_callbacks_construct_with_shared_budgets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs/experiments").glob("*/*.yaml"))
    assert len(paths) == 30
    for path in paths:
        raw = ConfigLoader().load_from(path)
        assignment, _, _ = ConfigUtility.read_raw_config(raw)
        budget = raw["assignment_config"]["resource_budget"]
        assert assignment.generation_budget == budget["generation_budget"]
        assert assignment.agent.parameters["action_normalization"] == "first_action_v1"
        assert budget["memory_capacity"] == 64
        assert budget["retrieval_top_k"] == 1
        for factory in assignment.callback_dict.values():
            if factory.module.rsplit(".", 1)[-1] in {"CurrentSessionSavingCallback", "ConsecutiveAbnormalAgentInferenceProcessHandlingCallback", "MemoryUsageMonitoringCallback"}:
                continue
            if "language_model" in factory.parameters:
                factory.parameters["language_model"] = UncalledModel()
            callback = factory.create()
            if hasattr(callback, "max_memory_count"):
                assert callback.max_memory_count <= 64
            if hasattr(callback, "use_evaluator_outcome_only"):
                assert callback.use_evaluator_outcome_only is True
            if hasattr(callback, "auxiliary_max_tokens"):
                assert callback.auxiliary_max_tokens == assignment.generation_budget
            if hasattr(callback, "max_skill_count") and hasattr(callback, "max_memory_count"):
                assert callback.max_skill_count + callback.max_memory_count == 64
