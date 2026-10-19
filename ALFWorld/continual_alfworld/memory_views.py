"""Safe, compact views of stored ALFWorld memories for prompt injection.

The state-delta representation is useful as a loss-light storage format, but
its observations and concrete entity identifiers describe a *past* world.
This module derives a procedure-only view that can be shown to an agent
without exposing that past layout.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence


PROCEDURE_VIEW_FORMAT = "procedure_only"

_STEP_RE = re.compile(r"^(?P<number>\d+)\.\s+(?P<body>.+)$")
_ENTITY_ID_RE = re.compile(r"\b([A-Za-z][A-Za-z_-]*)\s+\d+\b")


def _normalize(value: object) -> str:
    return " ".join(str(value or "").split())


def _redact_entity_ids(text: str) -> str:
    """Remove concrete ALFWorld entity numbers from otherwise safe metadata."""

    return _ENTITY_ID_RE.sub(r"\1 <id>", text)


def generalize_action(action: object) -> str:
    """Convert an ALFWorld action into a layout-independent procedure step.

    Known action grammars retain their operation and semantic argument roles,
    while discarding object/receptacle names and numbers. Unsupported actions
    are not copied into the prompt because their arguments may reveal a past
    scene.
    """

    text = _normalize(action).lower()
    if not text:
        return "<empty action omitted>"

    if text in {"inventory", "look"}:
        return text
    if re.fullmatch(r"go to .+", text):
        return "go to <receptacle>"
    if re.fullmatch(r"open .+", text):
        return "open <receptacle>"
    if re.fullmatch(r"close .+", text):
        return "close <receptacle>"
    if re.fullmatch(r"take .+ from .+", text):
        return "take <target_object> from <source_receptacle>"

    put_match = re.fullmatch(r"put .+?\s+(in|on)\s+.+", text)
    if put_match:
        return "put <target_object> {} <destination_receptacle>".format(
            put_match.group(1)
        )
    if re.fullmatch(r"move .+ to .+", text):
        return "move <target_object> to <destination_receptacle>"
    if re.fullmatch(r"clean .+ with .+", text):
        return "clean <target_object> with <cleaning_appliance>"
    if re.fullmatch(r"heat .+ with .+", text):
        return "heat <target_object> with <heating_appliance>"
    if re.fullmatch(r"cool .+ with .+", text):
        return "cool <target_object> with <cooling_appliance>"
    if re.fullmatch(r"slice .+ with .+", text):
        return "slice <target_object> with <tool>"
    if re.fullmatch(r"(?:use|toggle|turn on|turn off) .+", text):
        verb = text.split(" ", 1)[0]
        if text.startswith("turn on "):
            verb = "turn on"
        elif text.startswith("turn off "):
            verb = "turn off"
        return verb + " <device>"
    if re.fullmatch(r"(?:examine|inspect) .+", text):
        return text.split(" ", 1)[0] + " <target_object>"
    if re.fullmatch(r"look at .+?(?: with| under) .+", text):
        connector = "with" if " with " in text else "under"
        return "look at <target_object> {} <device>".format(connector)

    return "<unrecognized action omitted>"


def _metadata(trace: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in trace.splitlines():
        for key in ("Task", "Task type", "Outcome"):
            prefix = key + ":"
            if line.startswith(prefix):
                values[key] = _redact_entity_ids(_normalize(line[len(prefix) :]))
                break
    return values


def _procedure_lines(trace: str) -> List[str]:
    lines: List[str] = []
    in_transitions = False
    for raw_line in trace.splitlines():
        line = raw_line.strip()
        if line.startswith("Transitions (action =>"):
            in_transitions = True
            continue
        if not in_transitions:
            continue
        if line.startswith("[omitted transitions for budget:"):
            # The range contains step indices, not past entity identifiers.
            lines.append(line)
            continue
        match = _STEP_RE.match(line)
        if not match:
            # Never pass through unknown lines from the observation-bearing
            # section of a stored trace.
            continue
        number = match.group("number")
        body = match.group("body")
        repeat = re.fullmatch(r"REPEAT step (\d+) \(identical action and result\)", body)
        if repeat:
            lines.append("{}. repeat generalized step {}".format(number, repeat.group(1)))
            continue
        action, separator, _observation = body.partition("=>")
        if not separator:
            continue
        lines.append("{}. {}".format(number, generalize_action(action)))
    return lines


def _render(headers: Dict[str, str], steps: Sequence[str]) -> str:
    lines = [
        "Task: " + headers.get("Task", "(unknown)"),
        "Task type: " + headers.get("Task type", "(unknown)"),
        "Outcome: " + headers.get("Outcome", "(unknown)"),
        "Past procedure only (actions generalized; past observations omitted):",
    ]
    lines.extend(steps or ["(no usable procedure steps)"])
    return "\n".join(lines)


def render_procedure_only_view(
    state_delta_trace: object, max_chars: Optional[int] = 1200
) -> str:
    """Render ``state_delta`` text as a safe procedure-only prompt view.

    Args:
        state_delta_trace: Content produced by ``build_state_delta_trace``.
        max_chars: Maximum returned characters. ``None`` or a non-positive
            value disables this view-level budget.

    Returns:
        A view retaining Task, Task type, Outcome, and generalized action
        order. The initial state and every resulting observation are omitted.

    Raises:
        ValueError: If the input is not recognizable as a state-delta trace,
            or if ``max_chars`` is too small even for the safe header.
    """

    trace = str(state_delta_trace or "")
    if "Transitions (action => resulting observation delta):" not in trace:
        raise ValueError("procedure-only view requires a state_delta trace")

    headers = _metadata(trace)
    steps = _procedure_lines(trace)
    rendered = _render(headers, steps)
    if max_chars is None or max_chars <= 0 or len(rendered) <= max_chars:
        return rendered

    # Keep complete generalized steps. Prefer the first decision and terminal
    # step, then add intermediate steps in chronological order while they fit.
    selected: List[str] = []
    if steps:
        selected.append(steps[0])
        if len(steps) > 1:
            selected.append(steps[-1])
    omitted = max(0, len(steps) - len(selected))
    marker = "[{} generalized steps omitted for view budget]".format(omitted)
    compact_steps = selected[:1] + ([marker] if omitted else []) + selected[1:]
    rendered = _render(headers, compact_steps)

    for candidate in steps[1:-1]:
        trial_selected = compact_steps[:-1] if len(steps) > 1 else compact_steps[:]
        marker_index = next(
            (index for index, value in enumerate(trial_selected) if value.startswith("[")),
            len(trial_selected),
        )
        trial_selected.insert(marker_index, candidate)
        remaining = omitted - 1
        trial_selected = [
            "[{} generalized steps omitted for view budget]".format(remaining)
            if value.startswith("[")
            else value
            for value in trial_selected
        ]
        if remaining == 0:
            trial_selected = [value for value in trial_selected if not value.startswith("[")]
        if len(steps) > 1:
            trial_selected.append(steps[-1])
        trial = _render(headers, trial_selected)
        if len(trial) > max_chars:
            break
        compact_steps = trial_selected
        omitted = remaining
        rendered = trial

    if len(rendered) > max_chars:
        raise ValueError(
            "max_chars={} cannot hold procedure view header and boundary steps".format(
                max_chars
            )
        )
    return rendered


__all__ = [
    "PROCEDURE_VIEW_FORMAT",
    "generalize_action",
    "render_procedure_only_view",
]
