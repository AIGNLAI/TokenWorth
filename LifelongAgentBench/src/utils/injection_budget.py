"""Enforce one common character budget on the entire injected memory block."""
from src.typings import ChatHistoryItem, Role, Session


def enforce_injection_budget(session: Session, original: str, chars: int) -> None:
    if chars <= 0:
        raise ValueError("Memory injection budget must be positive.")
    first = session.chat_history.get_item_deep_copy(0)
    if first.role != Role.USER:
        raise ValueError("Memory must be injected into the user instruction slot.")
    updated = first.content
    prefix = 0
    while prefix < min(len(original), len(updated)) and original[prefix] == updated[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(original) - prefix, len(updated) - prefix) and original[-suffix - 1] == updated[-suffix - 1]:
        suffix += 1
    original_end = len(original) - suffix
    updated_end = len(updated) - suffix
    if original_end != prefix:
        raise ValueError("Memory callback modified task instructions rather than inserting memory.")
    injected = updated[prefix:updated_end]
    if len(injected) > chars:
        injected = injected[: max(0, chars - 2)].rstrip() + "\n\n"
        injected = injected[:chars]
        updated = original[:prefix] + injected + original[original_end:]
        session.chat_history.set(0, ChatHistoryItem(role=Role.USER, content=updated))
    session.memory_injection_chars = len(injected)
