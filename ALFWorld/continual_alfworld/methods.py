"""Standardized continual-memory methods for the ALFWorld benchmark.

These are deliberately protocol-compatible adaptations, not claims of exact
reproduction.  They share retrieval, memory/context budgets, task order, and
the same base language model.  Method-specific code only decides what to store,
how to update it, and how retrieved items are ranked.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .client import parse_json_object
from .memory_views import PROCEDURE_VIEW_FORMAT, render_procedure_only_view
from .retrieval import (
    bm25_rank,
    normalize,
    rank,
    tokenworth_hash_rank,
    tokenworth_hashed_vector,
    truncate,
    unique_strings,
)
from .tokenworth_gate import (
    CompatibilityConfig,
    LightweightAdvantageGate,
    filter_compatible_candidates,
)
from .trajectory import StateDeltaTrace, build_state_delta_trace


AuxiliaryCall = Callable[[str, str], str]


REPEAT_REFERENCE_PATTERN = re.compile(
    r"(?m)^(?P<step>\d+)\. REPEAT step (?P<source>\d+) "
    r"\(identical action and result\)$"
)


def render_repeat_reference_prompt(content: str, style: str) -> str:
    """Render stored repeat references for prompt injection only."""

    if style == "repeat_step":
        return content
    if style != "no_progress_duplicate":
        raise ValueError("unknown memory.repeat_reference_prompt_style: " + style)
    return REPEAT_REFERENCE_PATTERN.sub(
        lambda match: (
            "{}. [NO-PROGRESS DUPLICATE OMITTED: same action and result as past "
            "step {}; this is compression metadata, not an action to repeat]"
        ).format(match.group("step"), match.group("source")),
        content,
    )


@dataclass
class EpisodeOutcome:
    phase: str
    epoch: int
    episode_id: str
    task: str
    task_type: str
    gamefile: str
    success: bool
    steps: int
    transcript: List[Dict[str, str]]
    memory_disabled_after_loop: bool = False
    loop_break_trigger_count: int = 0
    loop_break_forced_actions: int = 0

    def compact(self, max_chars: int = 10000) -> str:
        turns = []
        for turn in self.transcript:
            turns.append(
                {
                    "observation": truncate(str(turn.get("observation", "")), 700),
                    "action": str(turn.get("action", "")),
                }
            )
        payload = {
            "task": self.task,
            "task_type": self.task_type,
            "success": self.success,
            "steps": self.steps,
            "turns": turns,
        }
        return truncate(json.dumps(payload, ensure_ascii=False), max_chars)


@dataclass
class RetrievedMemory:
    id: str
    score: float
    content: str
    kind: str = "memory"


class BaseMethod:
    name = "base"
    implementation_mode = "standardized_online_adaptation"

    def __init__(
        self,
        config: Dict[str, Any],
        state_path: Path,
        auxiliary_call: AuxiliaryCall,
    ) -> None:
        self.config = config
        self.state_path = state_path
        self.auxiliary_call = auxiliary_call
        self.step = 0
        self.state: Dict[str, Any] = self.initial_state()
        self.last_retrieved: List[RetrievedMemory] = []
        self._retrieval_observation = ""
        self.learning = False
        self.auxiliary_parse_errors = 0
        self.load()

    def initial_state(self) -> Dict[str, Any]:
        return {}

    @property
    def top_k(self) -> int:
        return int(self.config.get("retrieval", {}).get("top_k", 3))

    @property
    def dimensions(self) -> int:
        return int(self.config.get("retrieval", {}).get("dimensions", 512))

    @property
    def minimum_score(self) -> float:
        return float(self.config.get("retrieval", {}).get("minimum_score", 0.0))

    @property
    def shared_retrieval_backend(self) -> str:
        return str(
            self.config.get("retrieval", {}).get("backend", "stable_feature_hashing")
        )

    @property
    def max_records(self) -> int:
        return int(self.config.get("memory", {}).get("max_records", 128))

    @property
    def max_item_chars(self) -> int:
        return int(self.config.get("memory", {}).get("max_item_chars", 900))

    @property
    def context_chars(self) -> int:
        return int(self.config.get("memory", {}).get("context_chars", 6000))

    def new_id(self, prefix: str) -> str:
        return "{}-{}-{}".format(prefix, self.step, uuid.uuid4().hex[:8])

    def load(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("method") != self.name:
            raise ValueError("state method mismatch at {}".format(self.state_path))
        self.step = int(payload.get("step", 0))
        self.state = dict(payload.get("state") or {})
        self.auxiliary_parse_errors = int(payload.get("auxiliary_parse_errors", 0))

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.name,
            "implementation_mode": self.implementation_mode,
            "step": self.step,
            "auxiliary_parse_errors": self.auxiliary_parse_errors,
            "state": self.state,
            "saved_at": time.time(),
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(str(temporary), str(self.state_path))

    def _call_json(self, prompt: str, purpose: str) -> Dict[str, Any]:
        try:
            return parse_json_object(self.auxiliary_call(prompt, purpose))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.auxiliary_parse_errors += 1
            return {}

    def trajectory(self, outcome: EpisodeOutcome, max_chars: int = 10000) -> str:
        """Render the shared episode representation used by memory managers."""

        trajectory_format = str(
            self.config.get("memory", {}).get("trajectory_format", "state_delta")
        )
        if trajectory_format != "state_delta":
            raise ValueError("unknown memory.trajectory_format: " + trajectory_format)
        return build_state_delta_trace(
            task=outcome.task,
            task_type=outcome.task_type,
            success=outcome.success,
            steps=outcome.steps,
            transcript=outcome.transcript,
            max_chars=max_chars,
        ).content

    def _rank_records(
        self,
        query: str,
        records: Sequence[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if not records:
            return []
        documents = [
            "{} {} {}".format(
                record.get("task_type", ""),
                " ".join(record.get("tags", [])),
                record.get("content", record.get("instruction", "")),
            )
            for record in records
        ]
        scored = self._rank_texts(
            query,
            documents,
            self.top_k if top_k is None else top_k,
            self.minimum_score,
        )
        return [(score, records[index]) for score, index in scored]

    def _rank_texts(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int,
        minimum_score: float = -1.0,
    ) -> List[Tuple[float, int]]:
        backend = self.shared_retrieval_backend
        if backend == "stable_feature_hashing":
            return rank(
                query,
                documents,
                top_k,
                minimum_score,
                self.dimensions,
            )
        if backend == "tokenworth_feature_hashing_cosine":
            retrieval = self.config.get("retrieval", {})
            return tokenworth_hash_rank(
                query,
                documents,
                top_k,
                minimum_score,
                self.dimensions,
                int(retrieval.get("character_ngram_min", 3)),
                int(retrieval.get("character_ngram_max", 5)),
            )
        raise ValueError("unknown shared retrieval.backend: " + backend)

    def _as_retrieved(
        self, scored: Iterable[Tuple[float, Dict[str, Any]]]
    ) -> List[RetrievedMemory]:
        result = []
        for score, record in scored:
            result.append(
                RetrievedMemory(
                    id=str(record.get("id", "")),
                    score=round(float(score), 6),
                    content=str(record.get("content", record.get("instruction", ""))),
                    kind=str(record.get("kind", "memory")),
                )
            )
        return result

    def retrieve(self, task: str, task_type: str) -> List[RetrievedMemory]:
        del task, task_type
        self.last_retrieved = []
        return []

    def set_retrieval_context(self, observation: object) -> None:
        """Provide generic environment context without structured benchmark state."""

        self._retrieval_observation = str(observation or "")

    def render_context(self, memories: Sequence[RetrievedMemory]) -> str:
        if not memories:
            return "(none)"
        chunks: List[str] = []
        used = 0
        repeat_style = str(
            self.config.get("memory", {}).get(
                "repeat_reference_prompt_style", "repeat_step"
            )
        )
        for index, memory in enumerate(memories, 1):
            prompt_content = render_repeat_reference_prompt(
                memory.content, repeat_style
            )
            chunk = "[{} | {} | score={:.3f}]\n{}".format(
                index, memory.kind, memory.score, prompt_content
            )
            if used + len(chunk) > self.context_chars:
                remaining = self.context_chars - used
                if remaining > 60:
                    chunks.append(truncate(chunk, remaining))
                break
            chunks.append(chunk)
            used += len(chunk) + 2
        return "\n\n".join(chunks)

    def observe(self, outcome: EpisodeOutcome) -> None:
        del outcome
        self.step += 1

    def summary(self) -> Dict[str, Any]:
        update_phases = list(
            self.config.get("memory", {}).get("update_phases", ["train"])
        )
        return {
            "method": self.name,
            "implementation_mode": self.implementation_mode,
            "training_episodes": self.step if update_phases == ["train"] else None,
            "memory_update_episodes": self.step,
            "memory_update_phases": update_phases,
            "auxiliary_parse_errors": self.auxiliary_parse_errors,
            "trajectory_format": str(
                self.config.get("memory", {}).get("trajectory_format", "state_delta")
            ),
            "retrieval_top_k": self.top_k,
            "shared_retrieval_backend": self.shared_retrieval_backend,
            "memory_item_chars": self.max_item_chars,
            "memory_context_chars": self.context_chars,
        }

    def _record(
        self,
        content: object,
        task_type: str,
        kind: str,
        tags: Optional[Sequence[object]] = None,
        prefix: str = "mem",
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(content, str) or not normalize(content):
            return None
        return {
            "id": self.new_id(prefix),
            "content": truncate(content.strip(), self.max_item_chars),
            "task_type": task_type,
            "kind": kind,
            "tags": unique_strings(tags or [], 8),
            "created_step": self.step,
            "updated_step": self.step,
        }

    @staticmethod
    def _deduplicate(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        result = []
        for record in reversed(records):
            key = normalize(str(record.get("content", record.get("instruction", ""))))
            if key and key not in seen:
                seen.add(key)
                result.append(record)
        return list(reversed(result))


class NoMemory(BaseMethod):
    name = "no_memory"


class Mem0(BaseMethod):
    name = "mem0"

    def initial_state(self) -> Dict[str, Any]:
        return {"memories": []}

    def retrieve(self, task: str, task_type: str) -> List[RetrievedMemory]:
        scored = self._rank_records(task_type + " " + task, self.state["memories"])
        self.last_retrieved = self._as_retrieved(scored)
        return self.last_retrieved

    def observe(self, outcome: EpisodeOutcome) -> None:
        memories = self.state["memories"]
        retrieved = [asdict(item) for item in self.last_retrieved]
        prompt = """You are the Mem0 memory manager in an embodied household agent.
Given one completed trajectory and the memories retrieved before it, return only JSON:
{{"operations":[{{"op":"add|update|delete|noop","target_id":"existing id or empty","content":"durable reusable rule","tags":["short tags"]}}]}}
Keep at most 3 operations. Store reusable environment/action knowledge, not episode-specific prose.
Outcome: {outcome}
Retrieved: {retrieved}
""".format(
            outcome=self.trajectory(outcome),
            retrieved=json.dumps(retrieved, ensure_ascii=False),
        )
        response = self._call_json(prompt, "memory_mem0")
        allowed = {item.id for item in self.last_retrieved}
        by_id = {str(item.get("id")): item for item in memories}
        for operation in list(response.get("operations") or [])[:3]:
            if not isinstance(operation, dict):
                continue
            op = str(operation.get("op", "noop")).lower()
            target = str(operation.get("target_id", ""))
            if op == "add":
                record = self._record(
                    operation.get("content"),
                    outcome.task_type,
                    "memory",
                    operation.get("tags") or [],
                    "mem0",
                )
                if record:
                    memories.append(record)
            elif op == "update" and target in allowed and target in by_id:
                content = operation.get("content")
                if isinstance(content, str) and normalize(content):
                    by_id[target]["content"] = truncate(
                        content.strip(), self.max_item_chars
                    )
                    by_id[target]["tags"] = unique_strings(
                        operation.get("tags") or [], 8
                    )
                    by_id[target]["updated_step"] = self.step
            elif op == "delete" and target in allowed:
                memories[:] = [
                    item for item in memories if str(item.get("id")) != target
                ]
        self.state["memories"] = self._deduplicate(memories)[-self.max_records :]
        self.step += 1

    def summary(self) -> Dict[str, Any]:
        result = super().summary()
        result["memory_records"] = len(self.state["memories"])
        return result


class MemP(BaseMethod):
    name = "memp"

    def initial_state(self) -> Dict[str, Any]:
        return {"procedures": []}

    def retrieve(self, task: str, task_type: str) -> List[RetrievedMemory]:
        active = [
            item
            for item in self.state["procedures"]
            if item.get("status", "active") == "active"
        ]
        scored = self._rank_records(task_type + " " + task, active)
        self.last_retrieved = self._as_retrieved(scored)
        return self.last_retrieved

    def observe(self, outcome: EpisodeOutcome) -> None:
        procedures = self.state["procedures"]
        by_id = {str(item["id"]): item for item in procedures}
        for memory in self.last_retrieved:
            record = by_id.get(memory.id)
            if record:
                field = "support" if outcome.success else "contradiction"
                record[field] = int(record.get(field, 0)) + 1
                record["uses"] = int(record.get("uses", 0)) + 1

        if outcome.success:
            schema = '{"detailed_steps":["..."] ,"abstraction":"...","tags":["..."]}'
            instruction = "Extract up to two concrete procedures and one abstract transferable procedure."
            purpose = "memory_memp_success"
        else:
            schema = '{"cautions":["..."],"revisions":[{"target_id":"id","content":"..."}],"deprecate":["id"]}'
            instruction = "Extract up to two cautions and propose revisions/deprecations only for retrieved procedure ids."
            purpose = "memory_memp_failure"
        prompt = """You manage procedural memory for an embodied household agent.
{instruction} Return JSON only with schema {schema}
Trajectory: {outcome}
Retrieved procedures: {retrieved}
""".format(
            instruction=instruction,
            schema=schema,
            outcome=self.trajectory(outcome),
            retrieved=json.dumps(
                [asdict(item) for item in self.last_retrieved], ensure_ascii=False
            ),
        )
        response = self._call_json(prompt, purpose)
        allowed = {item.id for item in self.last_retrieved}
        if outcome.success:
            additions = [
                (item, "detailed")
                for item in unique_strings(response.get("detailed_steps") or [], 2)
            ]
            abstraction = response.get("abstraction")
            if isinstance(abstraction, str) and normalize(abstraction):
                additions.append((abstraction, "abstract"))
            for content, kind in additions:
                record = self._record(
                    content, outcome.task_type, kind, response.get("tags") or [], "proc"
                )
                if record:
                    record.update(
                        {
                            "status": "active",
                            "support": 1,
                            "contradiction": 0,
                            "uses": 0,
                        }
                    )
                    procedures.append(record)
        else:
            for content in unique_strings(response.get("cautions") or [], 2):
                record = self._record(content, outcome.task_type, "caution", [], "proc")
                if record:
                    record.update(
                        {
                            "status": "active",
                            "support": 0,
                            "contradiction": 1,
                            "uses": 0,
                        }
                    )
                    procedures.append(record)
            for revision in list(response.get("revisions") or [])[:2]:
                if not isinstance(revision, dict):
                    continue
                target = str(revision.get("target_id", ""))
                content = revision.get("content")
                if (
                    target in allowed
                    and target in by_id
                    and isinstance(content, str)
                    and normalize(content)
                ):
                    by_id[target]["content"] = truncate(
                        content.strip(), self.max_item_chars
                    )
                    by_id[target]["updated_step"] = self.step
            for target in unique_strings(response.get("deprecate") or [], 3):
                if target in allowed and target in by_id:
                    by_id[target]["status"] = "deprecated"
        for record in procedures:
            if int(record.get("uses", 0)) >= 3 and int(
                record.get("contradiction", 0)
            ) > int(record.get("support", 0)):
                record["status"] = "deprecated"
        active = [item for item in procedures if item.get("status") == "active"]
        deprecated = [item for item in procedures if item.get("status") != "active"]
        self.state["procedures"] = (
            deprecated[-max(8, self.max_records // 4) :] + self._deduplicate(active)
        )[-self.max_records :]
        self.step += 1

    def summary(self) -> Dict[str, Any]:
        result = super().summary()
        result["procedure_records"] = len(self.state["procedures"])
        result["active_procedures"] = sum(
            item.get("status") == "active" for item in self.state["procedures"]
        )
        return result


class MemRL(BaseMethod):
    name = "memrl"

    def initial_state(self) -> Dict[str, Any]:
        return {"experiences": []}

    def retrieve(self, task: str, task_type: str) -> List[RetrievedMemory]:
        experiences = self.state["experiences"]
        candidate_k = int(
            self.config.get("methods", {}).get("memrl", {}).get("candidate_k", 8)
        )
        candidates = self._rank_records(
            task_type + " " + task, experiences, candidate_k
        )
        similarity_weight = float(
            self.config.get("methods", {})
            .get("memrl", {})
            .get("similarity_weight", 0.5)
        )
        q_weight = 1.0 - similarity_weight
        rescored = []
        for similarity, record in candidates:
            q_value = max(-1.0, min(1.0, float(record.get("q", 0.0))))
            final = similarity_weight * similarity + q_weight * ((q_value + 1.0) / 2.0)
            rescored.append((final, record))
        rescored.sort(key=lambda item: item[0], reverse=True)
        self.last_retrieved = self._as_retrieved(rescored[: self.top_k])
        return self.last_retrieved

    def observe(self, outcome: EpisodeOutcome) -> None:
        experiences = self.state["experiences"]
        by_id = {str(item["id"]): item for item in experiences}
        reward = 1.0 if outcome.success else -1.0
        alpha = float(self.config.get("methods", {}).get("memrl", {}).get("alpha", 0.3))
        for memory in self.last_retrieved:
            record = by_id.get(memory.id)
            if record:
                record["q"] = (1.0 - alpha) * float(
                    record.get("q", 0.0)
                ) + alpha * reward
                record["uses"] = int(record.get("uses", 0)) + 1
                record["updated_step"] = self.step
        prompt = """Summarize a completed ALFWorld trajectory into one reusable experience.
Return JSON only: {{"content":"a concise successful procedure or corrective lesson","tags":["..."]}}
The content must state what action sequence or check helps future tasks; omit narration.
Trajectory: {outcome}
""".format(
            outcome=self.trajectory(outcome)
        )
        response = self._call_json(prompt, "memory_memrl")
        record = self._record(
            response.get("content"),
            outcome.task_type,
            "experience",
            response.get("tags") or [],
            "qexp",
        )
        if record:
            duplicate = next(
                (
                    item
                    for item in experiences
                    if normalize(str(item.get("content", "")))
                    == normalize(str(record.get("content", "")))
                ),
                None,
            )
            if duplicate is None:
                record.update(
                    {"q": reward, "uses": 0, "source_success": outcome.success}
                )
                experiences.append(record)
            else:
                # Preserve the identity and accumulated value of an experience;
                # a duplicate extraction is another observation, not a new item.
                if str(duplicate.get("id")) not in {
                    item.id for item in self.last_retrieved
                }:
                    duplicate["q"] = (1.0 - alpha) * float(
                        duplicate.get("q", 0.0)
                    ) + alpha * reward
                duplicate["updated_step"] = self.step
                duplicate["tags"] = unique_strings(
                    list(duplicate.get("tags", [])) + list(record.get("tags", [])), 8
                )
        experiences.sort(
            key=lambda item: (
                float(item.get("q", 0.0)),
                int(item.get("updated_step", 0)),
            )
        )
        self.state["experiences"] = experiences[-self.max_records :]
        self.step += 1

    def summary(self) -> Dict[str, Any]:
        result = super().summary()
        records = self.state["experiences"]
        result.update(
            {
                "experience_records": len(records),
                "mean_q": sum(float(item.get("q", 0.0)) for item in records)
                / max(1, len(records)),
            }
        )
        return result


class Reflexion(BaseMethod):
    name = "reflexion"

    def initial_state(self) -> Dict[str, Any]:
        return {"reflections": []}

    def retrieve(self, task: str, task_type: str) -> List[RetrievedMemory]:
        scored = self._rank_records(task_type + " " + task, self.state["reflections"])
        self.last_retrieved = self._as_retrieved(scored)
        return self.last_retrieved

    def observe(self, outcome: EpisodeOutcome) -> None:
        if not outcome.success:
            prompt = """Reflect on this failed ALFWorld trajectory. Return JSON only:
{{"failure_pattern":"...","root_cause":"...","corrective_plan":"...","checks":["..."]}}
The corrective plan must be operational and reusable on a later episode.
Trajectory: {outcome}
""".format(
                outcome=self.trajectory(outcome)
            )
            response = self._call_json(prompt, "memory_reflexion")
            pieces = [
                "Failure pattern: " + str(response.get("failure_pattern", "")),
                "Root cause: " + str(response.get("root_cause", "")),
                "Corrective plan: " + str(response.get("corrective_plan", "")),
            ]
            checks = unique_strings(response.get("checks") or [], 4)
            if checks:
                pieces.append("Checks: " + "; ".join(checks))
            record = self._record(
                "\n".join(pieces), outcome.task_type, "reflection", [], "refl"
            )
            if record and normalize(str(response.get("corrective_plan", ""))):
                self.state["reflections"].append(record)
            per_type = int(
                self.config.get("methods", {})
                .get("reflexion", {})
                .get("max_per_task_type", 3)
            )
            kept: List[Dict[str, Any]] = []
            for task_type in sorted(
                {str(item.get("task_type")) for item in self.state["reflections"]}
            ):
                subset = [
                    item
                    for item in self.state["reflections"]
                    if item.get("task_type") == task_type
                ]
                kept.extend(subset[-per_type:])
            self.state["reflections"] = kept[-self.max_records :]
        self.step += 1

    def summary(self) -> Dict[str, Any]:
        result = super().summary()
        result["reflection_records"] = len(self.state["reflections"])
        return result


class MemSkill(BaseMethod):
    name = "memskill"

    def initial_state(self) -> Dict[str, Any]:
        primitives = [
            (
                "insert",
                "Add a new reusable memory when the trajectory reveals genuinely new knowledge.",
            ),
            (
                "update",
                "Revise a retrieved memory when evidence makes it incomplete or misleading.",
            ),
            (
                "delete",
                "Delete a retrieved memory only when repeated evidence shows it is harmful.",
            ),
            (
                "noop",
                "Make no memory change when the trajectory adds no durable knowledge.",
            ),
        ]
        return {
            "memories": [],
            "operation_skills": [
                {
                    "id": "skill-" + name,
                    "name": name,
                    "instruction": instruction,
                    "triggers": [name],
                    "utility": 0.0,
                    "uses": 0,
                    "created_step": 0,
                }
                for name, instruction in primitives
            ],
            "failures": [],
        }

    def retrieve(self, task: str, task_type: str) -> List[RetrievedMemory]:
        scored = self._rank_records(task_type + " " + task, self.state["memories"])
        self.last_retrieved = self._as_retrieved(scored)
        return self.last_retrieved

    def _select_skills(self, query: str) -> List[Dict[str, Any]]:
        skills = self.state["operation_skills"]
        documents = [
            "{} {}".format(" ".join(item.get("triggers", [])), item["instruction"])
            for item in skills
        ]
        similarity = {
            index: score
            for score, index in self._rank_texts(query, documents, len(skills), -1.0)
        }
        scored = []
        for index, skill in enumerate(skills):
            utility = (float(skill.get("utility", 0.0)) + 1.0) / 2.0
            scored.append((0.6 * similarity.get(index, 0.0) + 0.4 * utility, skill))
        scored.sort(key=lambda item: item[0], reverse=True)
        count = int(
            self.config.get("methods", {}).get("memskill", {}).get("action_top_k", 3)
        )
        return [item for _, item in scored[:count]]

    def observe(self, outcome: EpisodeOutcome) -> None:
        query = outcome.task + " " + self.trajectory(outcome, 3000)
        selected = self._select_skills(query)
        prompt = """Execute memory-operation skills after an embodied-agent episode.
Return only JSON: {{"operations":[{{"skill_id":"id","op":"insert|update|delete|noop","target_id":"retrieved id or empty","content":"reusable rule","tags":["..."]}}]}}
Only update/delete retrieved ids. Use at most 3 operations.
Available operation skills: {skills}
Retrieved memories: {retrieved}
Trajectory: {outcome}
""".format(
            skills=json.dumps(selected, ensure_ascii=False),
            retrieved=json.dumps(
                [asdict(item) for item in self.last_retrieved], ensure_ascii=False
            ),
            outcome=self.trajectory(outcome),
        )
        response = self._call_json(prompt, "memory_memskill_executor")
        reward = 1.0 if outcome.success else -1.0
        beta = float(
            self.config.get("methods", {}).get("memskill", {}).get("utility_alpha", 0.2)
        )
        selected_by_id = {str(item["id"]): item for item in selected}
        memories = self.state["memories"]
        by_id = {str(item["id"]): item for item in memories}
        allowed = {item.id for item in self.last_retrieved}
        for operation in list(response.get("operations") or [])[:3]:
            if not isinstance(operation, dict):
                continue
            skill = selected_by_id.get(str(operation.get("skill_id", "")))
            if skill:
                skill["uses"] = int(skill.get("uses", 0)) + 1
                skill["utility"] = (1.0 - beta) * float(
                    skill.get("utility", 0.0)
                ) + beta * reward
            op = str(operation.get("op", "noop")).lower()
            target = str(operation.get("target_id", ""))
            if op == "insert":
                record = self._record(
                    operation.get("content"),
                    outcome.task_type,
                    "memory",
                    operation.get("tags") or [],
                    "msmem",
                )
                if record:
                    memories.append(record)
            elif op == "update" and target in allowed and target in by_id:
                content = operation.get("content")
                if isinstance(content, str) and normalize(content):
                    by_id[target]["content"] = truncate(
                        content.strip(), self.max_item_chars
                    )
                    by_id[target]["updated_step"] = self.step
            elif op == "delete" and target in allowed:
                memories[:] = [item for item in memories if str(item["id"]) != target]
        self.state["memories"] = self._deduplicate(memories)[-self.max_records :]
        if not outcome.success:
            self.state["failures"].append(
                {
                    "task": outcome.task,
                    "task_type": outcome.task_type,
                    "trace": self.trajectory(outcome, 3500),
                }
            )
            self.state["failures"] = self.state["failures"][-32:]

        interval = int(
            self.config.get("methods", {})
            .get("memskill", {})
            .get("designer_interval", 25)
        )
        max_skills = int(
            self.config.get("methods", {})
            .get("memskill", {})
            .get("max_operation_skills", 16)
        )
        if interval > 0 and (self.step + 1) % interval == 0 and self.state["failures"]:
            design_prompt = """Design one reusable memory-management operation skill from recent failures.
Return JSON only: {{"name":"short name","triggers":["..."],"instruction":"precise memory operation policy"}}
Do not design an environment-navigation skill; design how memory should be maintained.
Failures: {failures}
Current skills: {skills}
""".format(
                failures=json.dumps(self.state["failures"][-8:], ensure_ascii=False),
                skills=json.dumps(self.state["operation_skills"], ensure_ascii=False),
            )
            design = self._call_json(design_prompt, "memory_memskill_designer")
            if isinstance(design.get("instruction"), str) and normalize(
                design["instruction"]
            ):
                skill = {
                    "id": self.new_id("opskill"),
                    "name": truncate(str(design.get("name", "adaptive")), 80),
                    "instruction": truncate(
                        design["instruction"].strip(), self.max_item_chars
                    ),
                    "triggers": unique_strings(design.get("triggers") or [], 8),
                    "utility": 0.0,
                    "uses": 0,
                    "created_step": self.step,
                }
                self.state["operation_skills"].append(skill)
                learned = self.state["operation_skills"][4:]
                learned.sort(
                    key=lambda item: (
                        float(item.get("utility", 0.0)),
                        int(item.get("created_step", 0)),
                    )
                )
                self.state["operation_skills"] = (
                    self.state["operation_skills"][:4]
                    + learned[-max(0, max_skills - 4) :]
                )
                self.state["failures"] = []
        self.step += 1

    def summary(self) -> Dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "memory_records": len(self.state["memories"]),
                "operation_skills": len(self.state["operation_skills"]),
            }
        )
        return result


class EvoSkill(BaseMethod):
    name = "evoskill"

    def initial_state(self) -> Dict[str, Any]:
        return {
            "active_skills": [],
            "candidate": None,
            "failures": [],
            "feedback": [],
            "recent_results": [],
        }

    def retrieve(self, task: str, task_type: str) -> List[RetrievedMemory]:
        records = list(self.state["active_skills"])
        if self.state.get("candidate"):
            records.append(self.state["candidate"])
        scored = self._rank_records(task_type + " " + task, records)
        self.last_retrieved = self._as_retrieved(scored)
        return self.last_retrieved

    def _maybe_finish_validation(self) -> None:
        candidate = self.state.get("candidate")
        if not candidate:
            return
        window = int(
            self.config.get("methods", {})
            .get("evoskill", {})
            .get("validation_window", 20)
        )
        results = list(candidate.get("validation_results") or [])
        if len(results) < window:
            return
        candidate_rate = sum(bool(value) for value in results) / max(1, len(results))
        baseline_rate = float(candidate.get("baseline_rate", 0.0))
        delta = float(
            self.config.get("methods", {}).get("evoskill", {}).get("minimum_delta", 0.0)
        )
        accepted = candidate_rate >= baseline_rate + delta
        if accepted:
            parent_id = str(candidate.get("parent_id", ""))
            if parent_id:
                self.state["active_skills"] = [
                    item
                    for item in self.state["active_skills"]
                    if str(item["id"]) != parent_id
                ]
            candidate.pop("validation_results", None)
            candidate.pop("baseline_rate", None)
            candidate.pop("parent_id", None)
            candidate["kind"] = "skill"
            self.state["active_skills"].append(candidate)
        self.state["feedback"].append(
            {
                "step": self.step,
                "accepted": accepted,
                "candidate_rate": candidate_rate,
                "baseline_rate": baseline_rate,
            }
        )
        self.state["feedback"] = self.state["feedback"][-32:]
        self.state["candidate"] = None

    def _maybe_propose(self) -> None:
        settings = self.config.get("methods", {}).get("evoskill", {})
        interval = int(settings.get("evolution_interval", 25))
        minimum_failures = int(settings.get("minimum_failures", 3))
        if self.state.get("candidate") or not interval or (self.step + 1) % interval:
            return
        if len(self.state["failures"]) < minimum_failures:
            return
        proposer_prompt = """Act as an EvoSkill proposer. Analyze failed embodied-agent trajectories and current skills.
Return JSON only: {{"action":"create|edit","target_id":"existing skill id or empty","rationale":"...","skill_brief":"..."}}
Failures: {failures}
Current skills: {skills}
Prior validation feedback: {feedback}
""".format(
            failures=json.dumps(self.state["failures"][-8:], ensure_ascii=False),
            skills=json.dumps(self.state["active_skills"], ensure_ascii=False),
            feedback=json.dumps(self.state["feedback"][-8:], ensure_ascii=False),
        )
        proposal = self._call_json(proposer_prompt, "memory_evoskill_proposer")
        target_id = str(proposal.get("target_id", ""))
        allowed = {str(item["id"]) for item in self.state["active_skills"]}
        if str(proposal.get("action", "create")) == "edit" and target_id not in allowed:
            target_id = ""
        builder_prompt = """Act as an EvoSkill builder. Turn this proposal into one operational ALFWorld skill.
Return JSON only: {{"name":"...","triggers":["task cues"],"instruction":"ordered, executable guidance"}}
Proposal: {proposal}
""".format(
            proposal=json.dumps(proposal, ensure_ascii=False)
        )
        built = self._call_json(builder_prompt, "memory_evoskill_builder")
        if not isinstance(built.get("instruction"), str) or not normalize(
            built["instruction"]
        ):
            return
        recent = self.state["recent_results"][
            -int(settings.get("baseline_window", 20)) :
        ]
        baseline = sum(bool(value) for value in recent) / max(1, len(recent))
        self.state["candidate"] = {
            "id": self.new_id("candidate"),
            "name": truncate(str(built.get("name", "candidate skill")), 100),
            "content": truncate(built["instruction"].strip(), self.max_item_chars),
            "instruction": truncate(built["instruction"].strip(), self.max_item_chars),
            "triggers": unique_strings(built.get("triggers") or [], 8),
            "tags": unique_strings(built.get("triggers") or [], 8),
            "task_type": "general",
            "kind": "candidate_skill",
            "parent_id": target_id,
            "baseline_rate": baseline,
            "validation_results": [],
            "created_step": self.step,
        }
        self.state["failures"] = []

    def observe(self, outcome: EpisodeOutcome) -> None:
        candidate = self.state.get("candidate")
        if candidate and any(
            item.id == str(candidate["id"]) for item in self.last_retrieved
        ):
            candidate["validation_results"].append(outcome.success)
        self.state["recent_results"].append(outcome.success)
        self.state["recent_results"] = self.state["recent_results"][-100:]
        if not outcome.success:
            self.state["failures"].append(
                {
                    "task": outcome.task,
                    "task_type": outcome.task_type,
                    "trace": self.trajectory(outcome, 3500),
                }
            )
            self.state["failures"] = self.state["failures"][-32:]
        self._maybe_finish_validation()
        self._maybe_propose()
        max_skills = int(
            self.config.get("methods", {}).get("evoskill", {}).get("max_skills", 32)
        )
        self.state["active_skills"] = self.state["active_skills"][-max_skills:]
        self.step += 1

    def summary(self) -> Dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "active_skills": len(self.state["active_skills"]),
                "candidate_under_validation": bool(self.state.get("candidate")),
                "validation_events": len(self.state["feedback"]),
            }
        )
        return result


class EvoSC(BaseMethod):
    name = "evosc"
    implementation_mode = "textual_consolidation_surrogate_api_only"

    def initial_state(self) -> Dict[str, Any]:
        return {"experiences": [], "consolidations": []}

    def retrieve(self, task: str, task_type: str) -> List[RetrievedMemory]:
        records = self.state["consolidations"] + self.state["experiences"]
        scored = self._rank_records(task_type + " " + task, records)
        self.last_retrieved = self._as_retrieved(scored)
        return self.last_retrieved

    def observe(self, outcome: EpisodeOutcome) -> None:
        extract_prompt = """Extract one compact reusable experience from an ALFWorld trajectory.
Return JSON only: {{"content":"actionable rule including important preconditions","tags":["..."]}}
Trajectory: {outcome}
""".format(
            outcome=self.trajectory(outcome)
        )
        response = self._call_json(extract_prompt, "memory_evosc_extract")
        record = self._record(
            response.get("content"),
            outcome.task_type,
            "success_experience" if outcome.success else "failure_experience",
            response.get("tags") or [],
            "scexp",
        )
        if record:
            record["source_success"] = outcome.success
            self.state["experiences"].append(record)

        interval = int(
            self.config.get("methods", {})
            .get("evosc", {})
            .get("consolidation_interval", 16)
        )
        batch_size = int(
            self.config.get("methods", {})
            .get("evosc", {})
            .get("consolidation_batch_size", 24)
        )
        if (
            interval > 0
            and (self.step + 1) % interval == 0
            and self.state["experiences"]
        ):
            source = self.state["experiences"][-batch_size:]
            consolidate_prompt = """Self-consolidate episodic memories into durable ALFWorld knowledge.
Return JSON only: {{"summary":"compact procedures plus failure-avoidance rules","tags":["..."]}}
Resolve contradictions; preserve preconditions and task-type differences.
Experiences: {experiences}
Previous consolidated memories: {previous}
""".format(
                experiences=json.dumps(source, ensure_ascii=False),
                previous=json.dumps(
                    self.state["consolidations"][-3:], ensure_ascii=False
                ),
            )
            merged = self._call_json(consolidate_prompt, "memory_evosc_consolidate")
            summary = self._record(
                merged.get("summary"),
                "general",
                "consolidation",
                merged.get("tags") or [],
                "sc",
            )
            if summary:
                self.state["consolidations"].append(summary)
                source_ids = {str(item["id"]) for item in source}
                self.state["experiences"] = [
                    item
                    for item in self.state["experiences"]
                    if str(item["id"]) not in source_ids
                ]
        total = self.state["consolidations"] + self.state["experiences"]
        if len(total) > self.max_records:
            overflow = len(total) - self.max_records
            self.state["experiences"] = self.state["experiences"][overflow:]
            total = self.state["consolidations"] + self.state["experiences"]
            if len(total) > self.max_records:
                self.state["consolidations"] = self.state["consolidations"][
                    -self.max_records :
                ]
                self.state["experiences"] = []
        self.step += 1

    def summary(self) -> Dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "episodic_records": len(self.state["experiences"]),
                "consolidated_records": len(self.state["consolidations"]),
            }
        )
        return result


def _trajectory_memory(
    outcome: EpisodeOutcome, config: Dict[str, Any], max_chars: int
) -> Tuple[str, Optional[Dict[str, Any]]]:
    trajectory_format = str(
        config.get("memory", {}).get("trajectory_format", "state_delta")
    )
    if trajectory_format != "state_delta":
        raise ValueError("unknown memory.trajectory_format: " + trajectory_format)
    trace: StateDeltaTrace = build_state_delta_trace(
        task=outcome.task,
        task_type=outcome.task_type,
        success=outcome.success,
        steps=outcome.steps,
        transcript=outcome.transcript,
        max_chars=max_chars,
    )
    return trace.content, trace.metadata()


class RAG(BaseMethod):
    """Episodic RAG with configurable retrieval and trajectory write policies."""

    name = "rag"
    implementation_mode = "adapted_from_tokenworth_rag_callback"

    def initial_state(self) -> Dict[str, Any]:
        return {"experiences": []}

    @property
    def writes_only_successes(self) -> bool:
        return bool(self.settings.get("writes_only_successes", True))

    @property
    def settings(self) -> Dict[str, Any]:
        return self.config.get("methods", {}).get("rag", {})

    @property
    def retrieval_backend(self) -> str:
        return str(self.settings.get("retrieval_backend", "bm25"))

    def _hash_embedding(self, text: str) -> List[float]:
        return tokenworth_hashed_vector(
            text,
            dimensions=int(self.settings.get("embedding_dimensions", self.dimensions)),
            character_ngram_min=int(self.settings.get("character_ngram_min", 3)),
            character_ngram_max=int(self.settings.get("character_ngram_max", 5)),
        )

    def retrieve(self, task: str, task_type: str) -> List[RetrievedMemory]:
        del task_type
        records = self.state["experiences"]
        if self.retrieval_backend == "bm25":
            scored = bm25_rank(
                task,
                [str(item.get("search_text", "")) for item in records],
                self.top_k,
            )
        elif self.retrieval_backend == "tokenworth_feature_hashing_cosine":
            query_embedding = self._hash_embedding(task)
            document_field = str(self.settings.get("hash_document_field", "query"))
            if document_field not in {"query", "search_text"}:
                raise ValueError("unknown rag hash_document_field: " + document_field)
            scored = []
            for index, record in enumerate(records):
                embedding = record.get("hash_embedding")
                if not embedding or record.get("hash_document_field") != document_field:
                    embedding = self._hash_embedding(
                        str(record.get(document_field, ""))
                    )
                score = TokenWorth._cosine_dense(query_embedding, embedding)
                scored.append((score, index))
            scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            scored = scored[: self.top_k]
        else:
            raise ValueError("unknown rag retrieval_backend: " + self.retrieval_backend)
        self.last_retrieved = [
            RetrievedMemory(
                id=str(records[index]["id"]),
                score=round(float(score), 6),
                content=str(records[index]["content"]),
                kind=str(records[index].get("kind", "trajectory")),
            )
            for score, index in scored
        ]
        return self.last_retrieved

    def observe(self, outcome: EpisodeOutcome) -> None:
        if outcome.success or not self.writes_only_successes:
            snippet, compression = _trajectory_memory(
                outcome, self.config, self.max_item_chars
            )
            kind = (
                "successful_trajectory"
                if outcome.success
                else "failed_trajectory_counterexample"
            )
            record = self._record(
                snippet,
                outcome.task_type,
                kind,
                [outcome.task_type, "success" if outcome.success else "failure"],
                "rag",
            )
            if record:
                if compression is not None:
                    record["compression"] = compression
                record["source_success"] = outcome.success
                record["query"] = outcome.task
                record["search_text"] = outcome.task + "\n" + snippet
                if self.retrieval_backend == "tokenworth_feature_hashing_cosine":
                    document_field = str(
                        self.settings.get("hash_document_field", "query")
                    )
                    if document_field not in {"query", "search_text"}:
                        raise ValueError(
                            "unknown rag hash_document_field: " + document_field
                        )
                    record["hash_document_field"] = document_field
                    record["hash_embedding"] = self._hash_embedding(
                        str(record[document_field])
                    )
                self.state["experiences"].append(record)
                # The reference callback uses insertion-order FIFO eviction.
                self.state["experiences"] = self.state["experiences"][
                    -self.max_records :
                ]
        self.step += 1

    def summary(self) -> Dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "experience_records": len(self.state["experiences"]),
                "retrieval_backend": self.retrieval_backend,
                "writes_only_successes": self.writes_only_successes,
                "successful_records": sum(
                    bool(
                        item.get(
                            "source_success",
                            item.get("kind") == "successful_trajectory",
                        )
                    )
                    for item in self.state["experiences"]
                ),
                "failed_records": sum(
                    not bool(
                        item.get(
                            "source_success",
                            item.get("kind") == "successful_trajectory",
                        )
                    )
                    for item in self.state["experiences"]
                ),
                "budget_reduced_records": sum(
                    bool(item.get("compression", {}).get("budget_reduced"))
                    for item in self.state["experiences"]
                ),
            }
        )
        return result


class TokenWorth(BaseMethod):
    """ALFWorld trajectory adapter for the shared paper TokenWorth core."""

    name = "tokenworth"
    implementation_mode = "paper-v2"

    def __init__(self, *args, **kwargs):
        from tokenworth_core import TokenWorthCore

        super().__init__(*args, **kwargs)
        settings = dict(self.config.get("methods", {}).get("tokenworth", {}))
        variant = settings.pop("implementation_variant", "paper")
        if variant != "paper":
            raise ValueError(
                "Only paper TokenWorth is supported; start a new output directory"
            )
        parameters = {
            "embedding_dim": self.dimensions,
            "top_k": self.top_k,
            "capacity": self.max_records,
            "character_ngram_min": self.config.get("retrieval", {}).get(
                "character_ngram_min", 3
            ),
            "character_ngram_max": self.config.get("retrieval", {}).get(
                "character_ngram_max", 5
            ),
            **settings,
        }
        existed = self.state_path.exists()
        self.core = TokenWorthCore(
            parameters, self.state.get("paper") if existed else None
        )
        if existed and (
            "paper" not in self.state or self.step != self.core.state["step"]
        ):
            raise ValueError(
                "Legacy/incomplete TokenWorth state; use a new output directory"
            )
        if self.context_chars < 128 or self.max_item_chars < 1:
            raise ValueError(
                "TokenWorth requires positive snippet and usable bounded context limits"
            )
        self.state["paper"] = self.core.state

    def _embedding(self, text):
        return self.core.encode(text)

    @staticmethod
    def _cosine_dense(left, right):
        from tokenworth_core import cosine

        return cosine(left, right)

    def retrieve(self, task, task_type):
        selected = self.core.retrieve(task)
        self.last_retrieved = [
            RetrievedMemory(
                str(record["id"]), score, record["snippet"], "verified_experience"
            )
            for score, record in selected
        ]
        return self.last_retrieved

    def render_context(self, memories):
        chunks, injected, used = [], [], 0
        for memory in memories:
            header = "[Verified past experience {}]\n".format(memory.id)
            remaining = self.context_chars - used - len(header) - (2 if chunks else 0)
            if remaining < 1:
                break
            snippet = truncate(memory.content, remaining)
            if not snippet:
                break
            chunk = header + snippet
            chunks.append(chunk)
            used += len(chunk) + (2 if len(chunks) > 1 else 0)
            injected.append(int(memory.id))
        self.core.confirm_injection(injected)
        self.last_retrieved = [m for m in memories if int(m.id) in injected]
        return "\n\n".join(chunks) if chunks else "(none)"

    def finish_without_update(self):
        self.core.discard_retrieval()

    def observe(self, outcome):
        # Only the externally verified final won flag supplies the reward.
        snippet = (
            self.trajectory(outcome, self.max_item_chars) if outcome.success else ""
        )
        self.core.feedback(
            outcome.success,
            snippet,
            {
                "episode_id": outcome.episode_id,
                "phase": outcome.phase,
                "epoch": outcome.epoch,
            },
        )
        self.step = self.core.state["step"]

    def save(self):
        self.core.validate_state()
        super().save()

    def summary(self):
        return {
            **super().summary(),
            "implementation": self.implementation_mode,
            "parameters": dict(self.core.parameters),
            "memory_count": len(self.core.state["memories"]),
            "metrics": dict(self.core.state["metrics"]),
        }


METHODS = {
    "no_memory": NoMemory,
    "rag": RAG,
    "tokenworth": TokenWorth,
    "mem0": Mem0,
    "memp": MemP,
    "memrl": MemRL,
    "memskill": MemSkill,
    "evoskill": EvoSkill,
    "evosc": EvoSC,
    "reflexion": Reflexion,
}


def create_method(
    name: str,
    config: Dict[str, Any],
    state_path: Path,
    auxiliary_call: AuxiliaryCall,
) -> BaseMethod:
    normalized_name = name.strip().lower().replace("-", "_")
    aliases = {"nomemory": "no_memory", "none": "no_memory"}
    normalized_name = aliases.get(normalized_name, normalized_name)
    if normalized_name not in METHODS:
        raise ValueError(
            "unknown method {!r}; choose from {}".format(name, ", ".join(METHODS))
        )
    return METHODS[normalized_name](config, state_path, auxiliary_call)
