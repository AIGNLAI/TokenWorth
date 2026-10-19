"""Deterministic, low-loss serialization for ALFWorld trajectory memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Set, Tuple


WELCOME = "-= Welcome to TextWorld, ALFRED! =-"
STATE_CHANGING_ACTIONS = {
    "open",
    "close",
    "take",
    "move",
    "clean",
    "heat",
    "cool",
    "slice",
    "use",
}


def normalize_text(value: object) -> str:
    """Collapse display whitespace without changing observation words."""

    return " ".join(str(value or "").split())


def _middle_truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 5:
        return text[:limit]
    left = (limit - 5) // 2
    right = limit - 5 - left
    return text[:left] + " ... " + text[-right:]


def _initial_state(observation: object) -> str:
    """Remove UI boilerplate and the separately stored task, not state facts."""

    text = normalize_text(observation).replace(WELCOME, "").strip()
    marker = "Your task is to:"
    if marker in text:
        text = text.split(marker, 1)[0].strip()
    return text or "(initial state unavailable)"


def _result_observation(transcript: Sequence[Dict[str, Any]], index: int) -> str:
    """Read the action result from the normalized in-memory outcome."""

    turn = transcript[index]
    result = turn.get("next_observation")
    if result is not None and normalize_text(result):
        return normalize_text(result)
    if index + 1 < len(transcript):
        return normalize_text(transcript[index + 1].get("observation"))
    return "(terminal observation unavailable)"


def _step_ranges(indices: Sequence[int]) -> str:
    if not indices:
        return "none"
    ranges: List[str] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else "{}-{}".format(start, previous))
        start = previous = value
    ranges.append(str(start) if start == previous else "{}-{}".format(start, previous))
    return ",".join(ranges)


@dataclass(frozen=True)
class StateDeltaTrace:
    content: str
    format: str
    source_chars: int
    unbounded_chars: int
    stored_chars: int
    total_transitions: int
    retained_transitions: int
    repeat_references: int
    budget_reduced: bool

    def metadata(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("content")
        return payload


def _render_trace(
    task: str,
    task_type: str,
    success: bool,
    declared_steps: int,
    initial_state: str,
    transitions: Sequence[Tuple[str, str]],
    selected: Set[int],
) -> Tuple[str, int]:
    lines = [
        "Task: " + task,
        "Task type: " + task_type,
        "Outcome: {}; steps={}".format("SUCCESS" if success else "FAILURE", declared_steps),
        "Past episode initial state (not the current environment): " + initial_state,
        "Transitions (action => resulting observation delta):",
    ]
    definitions: Dict[Tuple[str, str], int] = {}
    omitted: List[int] = []
    references = 0

    def flush_omitted() -> None:
        if omitted:
            lines.append("[omitted transitions for budget: {}]".format(_step_ranges(omitted)))
            omitted[:] = []

    for offset, pair in enumerate(transitions):
        step = offset + 1
        if offset not in selected:
            omitted.append(step)
            continue
        flush_omitted()
        if pair in definitions:
            lines.append("{}. REPEAT step {} (identical action and result)".format(step, definitions[pair]))
            references += 1
        else:
            action, result = pair
            lines.append("{}. {} => {}".format(step, action, result))
            definitions[pair] = step
    flush_omitted()
    return "\n".join(lines), references


def build_state_delta_trace(
    task: object,
    task_type: object,
    success: bool,
    steps: int,
    transcript: Sequence[Dict[str, Any]],
    max_chars: int,
) -> StateDeltaTrace:
    """Serialize S0 followed by action/result deltas, preserving the terminal result.

    Exact repeated transitions are represented by backward references. If an
    exceptional trace exceeds ``max_chars``, whole low-priority transitions are
    omitted and explicitly identified; raw prefix truncation is never used.
    """

    clean_task = normalize_text(task)
    clean_type = normalize_text(task_type)
    initial = _initial_state(transcript[0].get("observation")) if transcript else "(initial state unavailable)"
    transitions = [
        (normalize_text(turn.get("action")), _result_observation(transcript, index))
        for index, turn in enumerate(transcript)
    ]
    source = "\n".join(
        [clean_task, initial]
        + ["{} => {}".format(action, result) for action, result in transitions]
    )
    selected = set(range(len(transitions)))
    full, full_references = _render_trace(
        clean_task, clean_type, success, steps, initial, transitions, selected
    )
    if max_chars <= 0 or len(full) <= max_chars:
        return StateDeltaTrace(
            content=full,
            format="state_delta",
            source_chars=len(source),
            unbounded_chars=len(full),
            stored_chars=len(full),
            total_transitions=len(transitions),
            retained_transitions=len(transitions),
            repeat_references=full_references,
            budget_reduced=False,
        )

    # Preserve the terminal result first, then state-changing operations, the
    # first decision, and remaining observations. Each candidate is accepted
    # only if the complete, line-oriented representation still fits.
    selected = {len(transitions) - 1} if transitions else set()
    priorities: List[int] = []
    priorities.extend(
        index
        for index, (action, _) in enumerate(transitions[:-1])
        if action.split(" ", 1)[0].lower() in STATE_CHANGING_ACTIONS
    )
    if transitions:
        priorities.append(0)
    priorities.extend(range(max(0, len(transitions) - 1)))
    seen: Set[int] = set(selected)
    for index in priorities:
        if index in seen:
            continue
        candidate = set(selected)
        candidate.add(index)
        rendered, _ = _render_trace(
            clean_task, clean_type, success, steps, initial, transitions, candidate
        )
        if len(rendered) <= max_chars:
            selected = candidate
        seen.add(index)

    rendered, references = _render_trace(
        clean_task, clean_type, success, steps, initial, transitions, selected
    )
    if len(rendered) > max_chars:
        # Only relevant for unrealistically tiny budgets: shorten S0 while
        # retaining both its beginning/end and the terminal transition.
        excess = len(rendered) - max_chars
        shorter_initial = _middle_truncate(initial, max(40, len(initial) - excess))
        rendered, references = _render_trace(
            clean_task, clean_type, success, steps, shorter_initial, transitions, selected
        )
    if len(rendered) > max_chars:
        raise ValueError(
            "memory.max_item_chars={} cannot hold trajectory header and terminal transition".format(max_chars)
        )
    return StateDeltaTrace(
        content=rendered,
        format="state_delta",
        source_chars=len(source),
        unbounded_chars=len(full),
        stored_chars=len(rendered),
        total_transitions=len(transitions),
        retained_transitions=len(selected),
        repeat_references=references,
        budget_reduced=True,
    )
