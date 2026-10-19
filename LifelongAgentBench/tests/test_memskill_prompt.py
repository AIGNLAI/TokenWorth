from types import SimpleNamespace

import pytest

from src.callbacks.instance.MemSkill import MemSkillCallback
from src.typings import (
    ChatHistoryItem,
    Role,
    SampleStatus,
    Session,
    SessionEvaluationOutcome,
    TaskName,
)


@pytest.mark.parametrize(
    "task_name, budget",
    [
        (TaskName.DB_BENCH, 900),
        (TaskName.OS_INTERACTION, 800),
        (TaskName.KNOWLEDGE_GRAPH, 4096),
    ],
)
def test_long_trajectory_keeps_executor_rules_and_learns(
    task_name, budget, monkeypatch
):
    callback = MemSkillCallback(
        language_model=None, top_k=1, top_k_skills=1, max_trajectory_length=budget
    )
    session = Session(
        task_name=task_name, sample_index="0", sample_status=SampleStatus.COMPLETED
    )
    for role, content in [
        (Role.USER, "Instructions"),
        (Role.AGENT, "Ready"),
        (Role.USER, "Inspect log files"),
        (Role.AGENT, "Action: Operation\n```sql\nSELECT 1;\n```"),
        (Role.USER, "Observation " * 5000),
    ]:
        session.chat_history.inject(ChatHistoryItem(role=role, content=content))
    session.evaluation_record.outcome = SessionEvaluationOutcome.CORRECT
    captured = []

    def executor(model, prompt, **kwargs):
        captured.append(prompt)
        assert "Output format" in prompt
        assert "MEMORY ITEM:" in prompt
        assert "MEMORY INDEX is 0-based" in prompt
        assert "Action type: INSERT only." in prompt
        return (
            "ACTION: INSERT\nMEMORY ITEM: Inspect log file permissions before reading."
        )

    monkeypatch.setattr("src.callbacks.instance.MemSkill.run_single_prompt", executor)
    args = SimpleNamespace(
        current_session=session,
        session_context=SimpleNamespace(
            agent=SimpleNamespace(
                get_role_dict=lambda: {Role.USER: "user", Role.AGENT: "assistant"}
            )
        ),
    )
    callback.on_task_complete(args)
    assert len(captured) == 1
    assert len(callback.memory_record_list) == 1
    assert (
        callback.memory_record_list[0].memory
        == "Inspect log file permissions before reading."
    )
    variable = (
        captured[0]
        .split("Input Text Chunk:\n", 1)[1]
        .split("\n\nRetrieved Memories", 1)[0]
    )
    assert len(variable) <= budget
