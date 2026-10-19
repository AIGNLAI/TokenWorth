"""Conservative loop detection for an in-progress ALFWorld episode.

The detector is deliberately independent of the runner.  It accepts either
the runner's compact history entries (where ``observation`` is the result of
the action) or transcript entries (where ``next_observation`` is the result).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LoopDetection:
    """Result returned by :func:`detect_loop`.

    ``reason`` is one of ``repeated_action``, ``alternating_cycle``, or
    ``repeated_transition``.  The object can be used directly as a boolean.
    """

    detected: bool
    reason: Optional[str] = None
    span: int = 0
    occurrences: int = 0
    pattern: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.detected


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _result_observation(turn: Mapping[str, Any]) -> str:
    """Return the post-action observation from either supported history form."""

    if "next_observation" in turn and turn.get("next_observation") is not None:
        return _normalize(turn.get("next_observation"))
    return _normalize(turn.get("observation"))


def _alternating_suffix(
    actions: Sequence[str], observations: Sequence[str], repeat_limit: int
) -> LoopDetection:
    """Recognize an ABAB suffix without flagging a progressing interaction.

    With observations available, two identical action/result cycles are
    sufficient.  If results are missing, the action cycle must repeat
    ``repeat_limit`` times, because action-only ABAB is weaker evidence.
    """

    if len(actions) < 4:
        return LoopDetection(False)
    first, second = actions[-2], actions[-1]
    if not first or not second or first == second:
        return LoopDetection(False)

    repetitions = 0
    cursor = len(actions) - 2
    while cursor >= 0 and actions[cursor] == first and actions[cursor + 1] == second:
        repetitions += 1
        cursor -= 2
    if repetitions < 2:
        return LoopDetection(False)

    start = len(actions) - repetitions * 2
    cycle_observations = observations[start:]
    have_results = all(cycle_observations)
    stable_results = have_results and all(
        cycle_observations[index] == cycle_observations[index % 2]
        for index in range(2, len(cycle_observations))
    )
    if not stable_results and (have_results or repetitions < repeat_limit):
        return LoopDetection(False)

    return LoopDetection(
        True,
        reason="alternating_cycle",
        span=repetitions * 2,
        occurrences=repetitions,
        pattern=(first, second),
    )


def detect_loop(
    history: Sequence[Mapping[str, Any]], *, window: int = 6, repeat_limit: int = 3
) -> LoopDetection:
    """Detect a likely action loop in the most recent episode history.

    Detection is limited to the last ``window`` turns and covers:

    * the same action issued consecutively ``repeat_limit`` times;
    * a stable two-action cycle (ABAB);
    * the same action/result transition seen ``repeat_limit`` times.

    Empty or malformed actions are retained as boundaries but never treated as
    loop evidence.  The function does not mutate ``history``.
    """

    if window < 1:
        raise ValueError("window must be at least 1")
    if repeat_limit < 2:
        raise ValueError("repeat_limit must be at least 2")

    recent = list(history[-window:])
    if not recent:
        return LoopDetection(False)
    actions = [_normalize(turn.get("action")) for turn in recent]
    observations = [_result_observation(turn) for turn in recent]

    # Exact consecutive action repetition is strong evidence even when an
    # environment happens to word successive failure observations differently.
    final_action = actions[-1]
    if final_action:
        consecutive = 0
        for action in reversed(actions):
            if action != final_action:
                break
            consecutive += 1
        if consecutive >= repeat_limit:
            return LoopDetection(
                True,
                reason="repeated_action",
                span=consecutive,
                occurrences=consecutive,
                pattern=(final_action,),
            )

    alternating = _alternating_suffix(actions, observations, repeat_limit)
    if alternating:
        return alternating

    # Require a non-empty post-action observation.  Repeated actions with
    # absent results are handled only by the stricter checks above.
    transitions = [
        (action, observation)
        for action, observation in zip(actions, observations)
        if action and observation
    ]
    counts = Counter(transitions)
    # Only diagnose a repeated transition when the newest turn itself supplies
    # the repeated evidence.  A malformed trailing entry must not resurrect an
    # older pattern that may already have been left behind.
    if actions[-1] and observations[-1]:
        final_transition = (actions[-1], observations[-1])
        occurrences = counts[final_transition]
        if occurrences >= repeat_limit:
            return LoopDetection(
                True,
                reason="repeated_transition",
                span=len(recent),
                occurrences=occurrences,
                pattern=(final_transition[0], final_transition[1]),
            )

    return LoopDetection(False)


def loop_break_navigation_candidates(
    actions: Sequence[str], visited_counts: Mapping[str, int]
) -> Tuple[str, ...]:
    """Return navigation actions that force progress after a detected loop.

    Prefer destinations not yet visited in the current episode.  If every
    currently admissible destination has been visited, retain only the least
    visited destinations.  Non-navigation actions are intentionally excluded:
    the forced step is a one-step escape from the current search cycle, after
    which the full admissible action space is restored.
    """

    navigation = [action for action in actions if _normalize(action).startswith("go to ")]
    if not navigation:
        return ()
    counts = [int(visited_counts.get(_normalize(action), 0)) for action in navigation]
    minimum = min(counts)
    return tuple(
        action
        for action, count in zip(navigation, counts)
        if count == minimum
    )
