import json
from types import SimpleNamespace

import pytest

from src.callbacks.instance.EvoSC import EvoSCCallback
from src.callbacks.instance.EvoSkill import EvoSkillCallback, EvoSkillRecord, EvoSkillStatus, EvoSkillAction, EvoSkillFailureRecord
from src.language_models.language_model import LanguageModel
from src.typings import ChatHistoryItem, Role, Session, TaskName, SessionEvaluationOutcome


class SkillModel(LanguageModel):
    def __init__(self):
        super().__init__({"user": "user", "agent": "assistant"})
        self.prompts = []

    def _inference(self, batch, *args):
        self.prompts.append(batch[0].get_item_deep_copy(-1).content)
        content = ({"action": "create", "proposed_skill": "check system state", "justification": "avoid premature finish"} if len(self.prompts) == 1 else
                   {"name": "check_state", "description": "Verify state", "triggers": ["system task"], "instruction": "Check the final environment state before finishing.", "tags": ["verification"]})
        return [ChatHistoryItem(role=Role.AGENT, content=json.dumps(content))]


@pytest.mark.parametrize("chars", [800, 900, 4096])
def test_evskill_long_failure_evidence_preserves_schema_and_builds_skill(chars):
    model = SkillModel()
    cb = EvoSkillCallback(model, max_trajectory_length=chars, failure_batch_size=8)
    cb.failure_record_list = [EvoSkillFailureRecord(query="q" + str(i), trajectory="x"*2000,
                                retrieved_skill_name_list=[], outcome=SessionEvaluationOutcome.INCORRECT, created_step=i) for i in range(8)]
    session = Session(task_name=TaskName.DB_BENCH, sample_index="0")
    cb._evolve_from_failures(SimpleNamespace(current_session=session))
    assert len(model.prompts) == 2
    assert "Return JSON only" in model.prompts[1] and '"instruction"' in model.prompts[1]
    assert len(cb.skill_list) == 1
    assert cb.skill_list[0].instruction == "Check the final environment state before finishing."


def skill(identifier, status=EvoSkillStatus.ACTIVE, parent=None):
    return EvoSkillRecord(id=identifier, task_name="db_bench", name=identifier, description="skill",
                         triggers=[], instruction="procedure", examples=[], tags=[], status=status,
                         action=EvoSkillAction.CREATE, parent_skill_id=parent, source_failure_queries=[],
                         created_step=0, updated_step=0)


def test_skill_capacity_includes_candidate_and_discards_archived_bodies():
    cb = EvoSkillCallback(SkillModel(), max_skill_count=64)
    cb.skill_list = [skill(str(i)) for i in range(64)] + [skill("old", EvoSkillStatus.ARCHIVED), skill("candidate", EvoSkillStatus.CANDIDATE, "0")]
    cb._prune_skills()
    assert len(cb.skill_list) == 64
    assert {item.id for item in cb.skill_list} >= {"candidate", "0"}
    assert all(item.status != EvoSkillStatus.ARCHIVED for item in cb.skill_list)


def test_summary_budget_counts_individual_persistent_records():
    cb = EvoSCCallback(SkillModel(), max_success_count=32, max_failure_count=31, max_summary_count=1)
    cb.consolidated_summary_dict = {"old": ["old1", "old2"], "new": ["new1", "new2", "new3", "new4", "new5"]}
    cb._prune_summary_items()
    assert cb.consolidated_summary_dict == {"new": ["new5"]}
    assert cb.max_success_count + cb.max_failure_count + sum(map(len, cb.consolidated_summary_dict.values())) == 64


@pytest.mark.parametrize("method", ["RAG", "MemRL"])
@pytest.mark.parametrize("limit", ["task_limit_reached", "agent_context_limit"])
def test_verified_success_remains_positive_feedback_after_budget_limit(method, limit):
    from run import ConfigUtility
    from src.utils import ConfigLoader
    from pathlib import Path
    from src.typings import SampleStatus
    root = Path(__file__).resolve().parents[1]
    raw = ConfigLoader().load_from(root / f"configs/experiments/db/{method}.yaml")
    assignment, _, _ = ConfigUtility.read_raw_config(raw)
    cb = assignment.callback_dict["callback_0"].create()
    session = Session(task_name=TaskName.DB_BENCH, sample_index="0", sample_status=SampleStatus(limit))
    session.evaluation_record.outcome = SessionEvaluationOutcome.CORRECT
    for role, content in [(Role.USER, "schema"), (Role.AGENT, "ready"), (Role.USER, "update state"), (Role.AGENT, "Action: Operation\n```sql\nUPDATE t SET v=1;\n```"), (Role.USER, "OK"), (Role.AGENT, "")]:
        session.chat_history.inject(ChatHistoryItem(role=role, content=content))
    cb.on_task_complete(SimpleNamespace(current_session=session, session_context=SimpleNamespace(agent=SimpleNamespace(get_role_dict=lambda: {Role.USER: "user", Role.AGENT: "assistant"}))))
    records = cb.experience_record_list if method == "RAG" else cb.episode_record_list
    assert len(records) == 1
