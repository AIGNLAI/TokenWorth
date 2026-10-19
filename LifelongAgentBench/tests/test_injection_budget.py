import pytest
from src.utils.injection_budget import enforce_injection_budget
from src.typings import Session, TaskName, ChatHistoryItem, Role


def session_with(content):
    session = Session(task_name=TaskName.DB_BENCH, sample_index="0")
    session.chat_history.inject(ChatHistoryItem(role=Role.USER, content=content))
    return session


def test_budget_includes_title_and_separator_without_cutting_task_schema():
    original = "Execute SQL using the required action schema.\n\nNow solve the question."
    injected = "Retrieved experience:\n" + "x" * 900 + "\n\n"
    session = session_with(original.replace("\n\nNow", "\n\n" + injected + "Now"))
    enforce_injection_budget(session, original, 900)
    text = session.chat_history.get_item_deep_copy(0).content
    assert len(text) - len(original) == session.memory_injection_chars == 900
    assert text.startswith("Execute SQL using the required action schema.")
    assert text.endswith("Now solve the question.")


def test_callback_cannot_replace_or_remove_task_instructions():
    session = session_with("Output JSON.")
    with pytest.raises(ValueError, match="modified task instructions"):
        enforce_injection_budget(session, "Output SQL.", 900)


def test_no_memory_does_not_consume_injection_budget():
    session = session_with("task")
    enforce_injection_budget(session, "task", 900)
    assert session.memory_injection_chars == 0
