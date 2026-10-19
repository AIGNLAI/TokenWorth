"""Lightweight, deterministic gating helpers for TokenWorth on ALFWorld.

The gate deliberately uses low-dimensional, interpretable signals.  It parses
the task structure, rejects incompatible memories, creates reproducible
no-memory probes, and compares a memory's empirical success estimate with a
per-task-type no-memory baseline.

The module has no dependency on :mod:`methods` so it can be reused without a
circular import.  All persistent state is plain JSON-compatible data.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


_TYPE_TO_OPERATION = {
    "pick_and_place_simple": "place",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "two_object",
    "look_at_obj_in_light": "light",
}

_OBJECT_ALIASES = {
    "coffee machine": "coffeemachine",
    "counter top": "countertop",
    "desk lamp": "desklamp",
    "dining table": "diningtable",
    "floor lamp": "floorlamp",
    "garbage can": "garbagecan",
    "side table": "sidetable",
    "sink basin": "sinkbasin",
}


def _normalize(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    return " ".join(text.split())


def _canonical_task_type(value: object) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())).strip("_")


def _canonical_object(value: object) -> str:
    text = _normalize(value)
    text = re.sub(r"^(?:an|a|the|some)\s+", "", text)
    text = re.sub(r"\s+\d+$", "", text)
    return _OBJECT_ALIASES.get(text, text)


@dataclass(frozen=True)
class TaskSignature:
    """Structured portion of an ALFWorld natural-language task."""

    task_type: str
    operation: str
    target: str
    destination: str


def parse_alfworld_task(task: object, task_type: object = "") -> TaskSignature:
    """Parse operation, target, and destination from an ALFWorld task.

    ``task_type`` is authoritative for the operation when it is one of the six
    standard ALFWorld task types.  The text parser also handles common task
    paraphrases used by model-generated summaries, such as ``put a cool cup in
    microwave`` and ``examine the keychain with the desklamp``.
    """

    text = _normalize(task)
    clean_type = _canonical_task_type(task_type)
    operation = _TYPE_TO_OPERATION.get(clean_type, "")
    target = ""
    destination = ""

    light = re.search(
        r"(?:look at|examine)\s+(?:an|a|the|some)?\s*(.+?)\s+"
        r"(?:under|with|using)\s+(?:an|a|the)?\s*(.+)$",
        text,
    )
    state_then_place = re.search(
        r"(?:clean|heat|cool)\s+(?:an|a|the|some)?\s*(.+?)\s+and\s+"
        r"(?:then\s+)?(?:put|place)\s+(?:it|them)\s+"
        r"(?:in|on|into|onto|in the|on the)\s+(.+)$",
        text,
    )
    state_adjective_place = re.search(
        r"(?:put|place)\s+(?:an|a|the|some)?\s*"
        r"(clean|cleaned|heat|heated|hot|cool|cooled|cold)\s+(.+?)\s+"
        r"(?:in|on|into|onto)\s+(?:an|a|the)?\s*(.+)$",
        text,
    )
    two_place = re.search(
        r"(?:put|place)\s+(?:two|2)\s+(.+?)\s+"
        r"(?:in|on|into|onto)\s+(?:an|a|the)?\s*(.+)$",
        text,
    )
    simple_place = re.search(
        r"(?:put|place|move)\s+(?:an|a|the|some)?\s*(.+?)\s+"
        r"(?:in|on|into|onto|to)\s+(?:an|a|the)?\s*(.+)$",
        text,
    )

    if light:
        operation = operation or "light"
        target, destination = light.groups()
    elif state_then_place:
        operation = operation or text.split(" ", 1)[0]
        target, destination = state_then_place.groups()
    elif state_adjective_place:
        adjective, target, destination = state_adjective_place.groups()
        inferred = {
            "clean": "clean",
            "cleaned": "clean",
            "heat": "heat",
            "heated": "heat",
            "hot": "heat",
            "cool": "cool",
            "cooled": "cool",
            "cold": "cool",
        }[adjective]
        operation = operation or inferred
    elif two_place:
        operation = operation or "two_object"
        target, destination = two_place.groups()
    elif simple_place:
        operation = operation or "place"
        target, destination = simple_place.groups()

    return TaskSignature(
        task_type=clean_type,
        operation=operation,
        target=_canonical_object(target),
        destination=_canonical_object(destination),
    )


def signature_from_candidate(candidate: Mapping[str, Any]) -> TaskSignature:
    """Read a task signature from an existing TokenWorth memory record."""

    task = candidate.get("query") or candidate.get("task") or ""
    if not task:
        content = str(candidate.get("content", ""))
        match = re.search(r"^Task:\s*(.+)$", content, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            task = match.group(1)
    return parse_alfworld_task(task, candidate.get("task_type", ""))


@dataclass(frozen=True)
class CompatibilityConfig:
    """Hard-filter switches; destination matching is optional by design."""

    exact_task_type: bool = True
    exact_operation: bool = True
    exact_target: bool = True
    exact_destination: bool = False
    allow_unknown: bool = False

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "CompatibilityConfig":
        source = value or {}
        return cls(
            exact_task_type=bool(source.get("exact_task_type_gate", source.get("exact_task_type", True))),
            exact_operation=bool(source.get("exact_operation_gate", source.get("exact_operation", True))),
            exact_target=bool(source.get("exact_target_gate", source.get("exact_target", True))),
            exact_destination=bool(source.get("exact_destination_gate", source.get("exact_destination", False))),
            allow_unknown=bool(source.get("gate_allow_unknown", source.get("allow_unknown", False))),
        )


def compatibility_mismatches(
    current: TaskSignature,
    candidate: TaskSignature,
    config: CompatibilityConfig = CompatibilityConfig(),
) -> Tuple[str, ...]:
    """Return required fields that make ``candidate`` incompatible."""

    checks = (
        ("task_type", config.exact_task_type),
        ("operation", config.exact_operation),
        ("target", config.exact_target),
        ("destination", config.exact_destination),
    )
    mismatches: List[str] = []
    for field, required in checks:
        if not required:
            continue
        left = getattr(current, field)
        right = getattr(candidate, field)
        if not left or not right:
            if not config.allow_unknown:
                mismatches.append(field + "_unknown")
        elif left != right:
            mismatches.append(field)
    return tuple(mismatches)


def compatible_candidate(
    current: TaskSignature,
    candidate: TaskSignature,
    config: CompatibilityConfig = CompatibilityConfig(),
) -> bool:
    return not compatibility_mismatches(current, candidate, config)


def filter_compatible_candidates(
    task: object,
    task_type: object,
    candidates: Iterable[Mapping[str, Any]],
    config: CompatibilityConfig = CompatibilityConfig(),
) -> List[Mapping[str, Any]]:
    current = parse_alfworld_task(task, task_type)
    return [
        item
        for item in candidates
        if compatible_candidate(current, signature_from_candidate(item), config)
    ]


def deterministic_probe(seed: int, episode_key: object, task: object, rate: float) -> bool:
    """Return a reproducible Bernoulli decision without mutable RNG state."""

    probability = max(0.0, min(1.0, float(rate)))
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    payload = "{}\x1f{}\x1f{}".format(int(seed), episode_key, _normalize(task))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    unit = int.from_bytes(digest, "big") / float(1 << 64)
    return unit < probability


@dataclass
class BaselineEstimate:
    alpha: float
    beta: float
    ema: float
    count: int

    @property
    def beta_mean(self) -> float:
        return self.alpha / max(self.alpha + self.beta, 1e-12)


class TaskTypeBaseline:
    """Per-task-type no-memory success baseline using Beta and EMA estimates."""

    def __init__(
        self,
        state: Optional[Mapping[str, Any]] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        ema_alpha: float = 0.2,
        beta_weight: float = 0.5,
    ) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 <= beta_weight <= 1.0:
            raise ValueError("beta_weight must be in [0, 1]")
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.ema_alpha = float(ema_alpha)
        self.beta_weight = float(beta_weight)
        self._values: Dict[str, BaselineEstimate] = {}
        for key, raw in dict(state or {}).items():
            if not isinstance(raw, Mapping):
                continue
            self._values[_canonical_task_type(key) or "unknown"] = BaselineEstimate(
                alpha=float(raw.get("alpha", self.prior_alpha)),
                beta=float(raw.get("beta", self.prior_beta)),
                ema=float(raw.get("ema", self.prior_alpha / (self.prior_alpha + self.prior_beta))),
                count=int(raw.get("count", 0)),
            )

    def _new(self) -> BaselineEstimate:
        mean = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        return BaselineEstimate(self.prior_alpha, self.prior_beta, mean, 0)

    def details(self, task_type: object) -> BaselineEstimate:
        key = _canonical_task_type(task_type) or "unknown"
        value = self._values.get(key)
        if value is None:
            value = self._new()
            self._values[key] = value
        return value

    def update(self, task_type: object, success: bool) -> float:
        value = self.details(task_type)
        reward = 1.0 if success else 0.0
        value.alpha += reward
        value.beta += 1.0 - reward
        value.ema = reward if value.count == 0 else (
            self.ema_alpha * reward + (1.0 - self.ema_alpha) * value.ema
        )
        value.count += 1
        return self.estimate(task_type)

    def estimate(self, task_type: object) -> float:
        value = self.details(task_type)
        return self.beta_weight * value.beta_mean + (1.0 - self.beta_weight) * value.ema

    def count(self, task_type: object) -> int:
        return self.details(task_type).count

    def to_state(self) -> Dict[str, Dict[str, Any]]:
        return {key: asdict(value) for key, value in sorted(self._values.items())}


@dataclass(frozen=True)
class GateDecision:
    use_memory: bool
    reason: str
    no_memory_probe: bool
    baseline_success: float
    memory_success: Optional[float]
    predicted_advantage: Optional[float]
    mismatches: Tuple[str, ...] = ()


class LightweightAdvantageGate:
    """Deterministic probe and empirical relative-advantage gate.

    A candidate's ``help_count``/``harm_count`` form a Beta posterior for its
    success probability.  An optional model prediction can be blended into the
    posterior.  Supplying ``predicted_label`` expects the TokenWorth convention
    of roughly ``[-1, 1]`` and maps it to a success probability.
    """

    def __init__(
        self,
        baseline: Optional[TaskTypeBaseline] = None,
        probe_rate: float = 0.2,
        advantage_threshold: float = 0.03,
        min_baseline_feedback: int = 4,
        min_memory_feedback: int = 4,
        memory_prior_alpha: float = 1.0,
        memory_prior_beta: float = 1.0,
        model_weight: float = 0.25,
        eval_use_unvalidated: bool = False,
    ) -> None:
        if memory_prior_alpha <= 0 or memory_prior_beta <= 0:
            raise ValueError("memory Beta prior parameters must be positive")
        if not 0.0 <= probe_rate <= 1.0:
            raise ValueError("probe_rate must be in [0, 1]")
        if not 0.0 <= model_weight <= 1.0:
            raise ValueError("model_weight must be in [0, 1]")
        self.baseline = baseline or TaskTypeBaseline()
        self.probe_rate = float(probe_rate)
        self.advantage_threshold = float(advantage_threshold)
        self.min_baseline_feedback = max(0, int(min_baseline_feedback))
        self.min_memory_feedback = max(0, int(min_memory_feedback))
        self.memory_prior_alpha = float(memory_prior_alpha)
        self.memory_prior_beta = float(memory_prior_beta)
        self.model_weight = float(model_weight)
        self.eval_use_unvalidated = bool(eval_use_unvalidated)

    @classmethod
    def from_mapping(
        cls,
        settings: Optional[Mapping[str, Any]],
        baseline_state: Optional[Mapping[str, Any]] = None,
    ) -> "LightweightAdvantageGate":
        """Build from TokenWorth YAML settings and persisted baseline state."""

        source = settings or {}
        shared_minimum = int(source.get("gate_min_feedback", 4))
        baseline = TaskTypeBaseline(
            state=baseline_state,
            prior_alpha=float(source.get("gate_baseline_prior_alpha", 1.0)),
            prior_beta=float(source.get("gate_baseline_prior_beta", 1.0)),
            ema_alpha=float(source.get("gate_baseline_ema_alpha", 0.2)),
            beta_weight=float(source.get("gate_baseline_beta_weight", 0.5)),
        )
        return cls(
            baseline=baseline,
            probe_rate=float(source.get("gate_probe_rate", 0.2)),
            advantage_threshold=float(source.get("gate_advantage_threshold", 0.03)),
            min_baseline_feedback=int(source.get("gate_min_baseline_feedback", shared_minimum)),
            min_memory_feedback=int(source.get("gate_min_memory_feedback", shared_minimum)),
            memory_prior_alpha=float(source.get("gate_memory_prior_alpha", 1.0)),
            memory_prior_beta=float(source.get("gate_memory_prior_beta", 1.0)),
            model_weight=float(source.get("gate_model_weight", 0.25)),
            eval_use_unvalidated=bool(source.get("gate_eval_use_unvalidated", False)),
        )

    def should_probe(self, seed: int, episode_key: object, task: object) -> bool:
        return deterministic_probe(seed, episode_key, task, self.probe_rate)

    def update_no_memory(self, task_type: object, success: bool) -> float:
        return self.baseline.update(task_type, success)

    def memory_success_probability(
        self,
        candidate: Mapping[str, Any],
        predicted_success: Optional[float] = None,
        predicted_label: Optional[float] = None,
    ) -> float:
        help_count = max(0, int(candidate.get("help_count", 0)))
        harm_count = max(0, int(candidate.get("harm_count", 0)))
        empirical = (self.memory_prior_alpha + help_count) / (
            self.memory_prior_alpha + self.memory_prior_beta + help_count + harm_count
        )
        if predicted_success is None and predicted_label is not None:
            predicted_success = (max(-1.0, min(1.0, float(predicted_label))) + 1.0) / 2.0
        if predicted_success is None:
            return empirical
        model_value = max(0.0, min(1.0, float(predicted_success)))
        return (1.0 - self.model_weight) * empirical + self.model_weight * model_value

    def predicted_advantage(
        self,
        task_type: object,
        candidate: Mapping[str, Any],
        predicted_success: Optional[float] = None,
        predicted_label: Optional[float] = None,
    ) -> float:
        memory_value = self.memory_success_probability(
            candidate, predicted_success=predicted_success, predicted_label=predicted_label
        )
        return memory_value - self.baseline.estimate(task_type)

    def decide(
        self,
        task: object,
        task_type: object,
        candidate: Mapping[str, Any],
        *,
        learning: bool,
        seed: int,
        episode_key: object,
        compatibility: CompatibilityConfig = CompatibilityConfig(),
        predicted_success: Optional[float] = None,
        predicted_label: Optional[float] = None,
    ) -> GateDecision:
        current = parse_alfworld_task(task, task_type)
        candidate_signature = signature_from_candidate(candidate)
        mismatches = compatibility_mismatches(current, candidate_signature, compatibility)
        baseline = self.baseline.estimate(task_type)
        if mismatches:
            return GateDecision(False, "incompatible", False, baseline, None, None, mismatches)
        if learning and self.should_probe(seed, episode_key, task):
            return GateDecision(False, "no_memory_probe", True, baseline, None, None)

        memory_value = self.memory_success_probability(
            candidate, predicted_success=predicted_success, predicted_label=predicted_label
        )
        advantage = memory_value - baseline
        memory_feedback = max(0, int(candidate.get("help_count", 0))) + max(
            0, int(candidate.get("harm_count", 0))
        )
        if self.baseline.count(task_type) < self.min_baseline_feedback:
            use_memory = learning or self.eval_use_unvalidated
            return GateDecision(use_memory, "baseline_warmup", False, baseline, memory_value, advantage)
        if memory_feedback < self.min_memory_feedback:
            use_memory = learning or self.eval_use_unvalidated
            return GateDecision(use_memory, "memory_warmup", False, baseline, memory_value, advantage)
        use_memory = advantage > self.advantage_threshold
        return GateDecision(
            use_memory,
            "positive_advantage" if use_memory else "insufficient_advantage",
            False,
            baseline,
            memory_value,
            advantage,
        )
