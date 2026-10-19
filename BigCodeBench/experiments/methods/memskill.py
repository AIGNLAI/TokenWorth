"""MemSkill method implementation."""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import Counter
from typing import Any, Sequence

from .core import (
    MemoryMethod,
    RetrievedItem,
    TaskOutcome,
    bm25_scores,
    cosine,
    memory_length_requirement,
    normalize,
    parse_bullets,
    parse_json_object,
    rank_indices,
    string_list,
    tokenize,
    truncate,
)

class MemSkillMethod(MemoryMethod):
    method_name = "MemSkill"
    DEFAULT_SKILLS = [
        {"name": "insert_new_memory", "description": "Insert missing durable coding facts, constraints, procedures, or lessons.", "update_type": "insert"},
        {"name": "update_existing_memory", "description": "Revise a retrieved memory using corrected or more specific information.", "update_type": "update"},
        {"name": "delete_invalid_memory", "description": "Delete a retrieved memory explicitly contradicted by the evaluation.", "update_type": "delete"},
        {"name": "no_operation", "description": "Make no change when no durable information is available.", "update_type": "noop"},
    ]

    def restore(self) -> None:
        super().restore()
        self.state.setdefault("memories", [])
        self.state.setdefault("skills", [{**skill, "use_count": 0, "success_count": 0, "failure_count": 0} for skill in self.DEFAULT_SKILLS])
        self.state.setdefault("hard_cases", [])
        self._enforce_capacity()

    def _capacity_limits(self) -> tuple[int, int, int, int]:
        total = int(self.config.get("max_memory_count", 50))
        memory_limit = int(self.config.get("max_memory_record_count", 38))
        skill_limit = int(self.config.get("max_skill_count", 4))
        hard_case_limit = int(self.config.get("hard_case_buffer_size", 8))
        if min(total, memory_limit, skill_limit, hard_case_limit) < 1:
            raise ValueError("MemSkill memory capacities must be positive")
        if skill_limit < len(self.DEFAULT_SKILLS):
            raise ValueError(
                f"MemSkill requires at least {len(self.DEFAULT_SKILLS)} operation skills"
            )
        if memory_limit + skill_limit + hard_case_limit > total:
            raise ValueError(
                "MemSkill component capacities exceed max_memory_count: "
                f"{memory_limit} + {skill_limit} + {hard_case_limit} > {total}"
            )
        return total, memory_limit, skill_limit, hard_case_limit

    @staticmethod
    def _skill_score(skill: dict[str, Any]) -> tuple[float, int]:
        attempts = skill["success_count"] + skill["failure_count"]
        success_rate = skill["success_count"] / attempts if attempts else 0.0
        return success_rate, skill["use_count"]

    def _enforce_capacity(self) -> None:
        _, memory_limit, skill_limit, hard_case_limit = self._capacity_limits()
        memories = self.state["memories"]
        memories.sort(
            key=lambda record: (
                record["strength"],
                record["access_count"],
                record["last_access_step"],
            ),
            reverse=True,
        )
        del memories[memory_limit:]

        default_names = {skill["name"] for skill in self.DEFAULT_SKILLS}
        defaults = [skill for skill in self.state["skills"] if skill["name"] in default_names]
        learned = [skill for skill in self.state["skills"] if skill["name"] not in default_names]
        learned.sort(key=self._skill_score, reverse=True)
        self.state["skills"] = defaults[:skill_limit] + learned[: max(0, skill_limit - len(defaults))]
        self.state["hard_cases"] = self.state["hard_cases"][-hard_case_limit:]

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state["memories"]
        eligible = [i for i, record in enumerate(records) if not self.config.get("correct_memory_only", True) or record["source_outcome"] == "pass"]
        scores = bm25_scores(prompt, [records[i]["search_text"] for i in eligible])
        ranked = sorted(((scores[n] + 0.5 * records[i]["strength"] + 1.0 / (self.step - records[i]["last_access_step"] + 1), i) for n, i in enumerate(eligible) if scores[n] >= self.retrieval_threshold("min_retrieval_score", 0.1)), reverse=True)
        selected = ranked[: int(self.config.get("top_k", 8))]
        self.last_retrieved = [index for _, index in selected]
        for _, index in selected:
            records[index]["access_count"] += 1
            records[index]["last_access_step"] = self.step
        return [
            RetrievedItem.from_record(
                records[i],
                score,
                truncate(
                    records[i]["memory"],
                    int(self.config.get("max_injected_memory_chars_per_item", 1500)),
                ),
            )
            for score, i in selected
        ]

    def _rank_skills(
        self,
        query: str,
        skills: Sequence[dict[str, Any]],
        top_k: int,
    ) -> list[int]:
        """Rank operation skills; retrieval variants may override this hook."""
        return rank_indices(
            query,
            [skill["description"] for skill in skills],
            top_k,
        )

    def _upsert(self, text: str, outcome: TaskOutcome) -> None:
        if not text.strip() or (not outcome.success and self.config.get("correct_memory_only", True)):
            return
        minimum = int(self.config.get("min_memory_chars", 600))
        maximum = int(self.config.get("max_memory_chars", 1500))
        text = text.strip()
        if len(text) < minimum:
            self.state["rejected_too_short"] = int(self.state.get("rejected_too_short", 0)) + 1
            return
        text = truncate(text, maximum)
        records = self.state["memories"]
        existing = next((record for record in records if normalize(record["memory"]) == normalize(text)), None)
        if existing:
            existing["strength"] = min(3.0, existing["strength"] + (0.4 if outcome.success else 0.2))
            existing["access_count"] += 1
            existing["last_access_step"] = self.step
            return
        memory_id = int(self.state.get("next_id", 1))
        self.state["next_id"] = memory_id + 1
        records.append({"id": memory_id, "memory": text, "source_query": outcome.prompt, "search_text": f"{outcome.prompt}\n{text}", "source_outcome": outcome.status, "source_epoch": outcome.epoch, "source_phase": outcome.phase, "source_task_id": outcome.task_id, "source_prompt": outcome.prompt, "source_code": outcome.code, "source_status": outcome.status, "source_error": outcome.error, "created_step": self.step, "last_access_step": self.step, "access_count": 1, "strength": 1.0 if outcome.success else 0.6})

    def update(self, outcome: TaskOutcome) -> None:
        super().update(outcome)
        records = self.state["memories"]
        retrieved = [records[i]["memory"] for i in self.last_retrieved if i < len(records)]
        skills = self.state["skills"]
        skill_indices = self._rank_skills(
            outcome.trajectory + "\n" + "\n".join(retrieved),
            skills,
            int(self.config.get("top_k_skills", 3)),
        )
        selected_skills = [skills[i] for i in skill_indices]
        total_limit = int(
            self.config.get(
                "max_auxiliary_memory_chars_total",
                self.config.get("max_memory_chars", 1500),
            )
        )
        length_requirement = memory_length_requirement(
            "Inserted or updated memory text",
            int(self.config.get("min_memory_chars", 600)),
            int(self.config.get("max_memory_chars", 1500)),
        )
        response = self.auxiliary(
            "You are the MemSkill memory executor.\n"
            f"Trajectory:\n{outcome.trajectory}\nRetrieved memories (0-based): {retrieved}\n"
            f"Selected operations: {selected_skills}\n"
            "For a verified success, insert or update a reusable memory when the trajectory contains a "
            "specific transferable procedure. Every inserted or updated memory must state applicability, "
            "exact steps, rationale, edge cases, and a compact code idiom; never emit generic advice. "
            f"{length_requirement} The combined memory_item and "
            f"updated_memory text across all actions must not exceed {total_limit} characters.\n"
            'Return JSON only: {"actions":[{"action":"insert|update|delete|noop","memory_item":"...","memory_index":0,"updated_memory":"..."}]}'
        )
        try:
            actions = parse_json_object(response).get("actions", [])
        except Exception:
            actions = []
        deleted: set[int] = set()
        remaining = total_limit
        if outcome.success or not self.config.get("correct_memory_only", True):
            for action in actions if isinstance(actions, list) else []:
                if not isinstance(action, dict):
                    continue
                kind = str(action.get("action", "noop")).lower()
                local_index = action.get("memory_index")
                if kind == "insert":
                    memory_item = truncate(
                        str(action.get("memory_item", "")).strip(),
                        int(self.config.get("max_memory_chars", 1500)),
                    )
                    if len(memory_item) > remaining:
                        self.state["rejected_over_total_budget"] = int(
                            self.state.get("rejected_over_total_budget", 0)
                        ) + 1
                        continue
                    if len(memory_item) >= int(self.config.get("min_memory_chars", 600)):
                        remaining -= len(memory_item)
                    self._upsert(memory_item, outcome)
                elif kind == "update" and isinstance(local_index, int) and 0 <= local_index < len(self.last_retrieved):
                    index = self.last_retrieved[local_index]
                    updated = truncate(
                        str(action.get("updated_memory", "")).strip(),
                        int(self.config.get("max_memory_chars", 1500)),
                    )
                    minimum = int(self.config.get("min_memory_chars", 600))
                    if len(updated) > remaining:
                        self.state["rejected_over_total_budget"] = int(
                            self.state.get("rejected_over_total_budget", 0)
                        ) + 1
                        continue
                    if len(updated) >= minimum:
                        remaining -= len(updated)
                        records[index]["memory"] = updated
                        records[index]["search_text"] = f"{outcome.prompt}\n{records[index]['memory']}"
                        records[index].update(
                            source_epoch=outcome.epoch,
                            source_phase=outcome.phase,
                            source_task_id=outcome.task_id,
                            source_prompt=outcome.prompt,
                            source_code=outcome.code,
                            source_status=outcome.status,
                            source_error=outcome.error,
                        )
                    else:
                        self.state["rejected_too_short"] = int(self.state.get("rejected_too_short", 0)) + 1
                elif kind == "delete" and isinstance(local_index, int) and 0 <= local_index < len(self.last_retrieved):
                    deleted.add(self.last_retrieved[local_index])
        self.state["memories"] = [record for index, record in enumerate(records) if index not in deleted]
        for index in skill_indices:
            skills[index]["use_count"] += 1
            skills[index]["success_count" if outcome.success else "failure_count"] += 1
        if not outcome.success:
            self.state["hard_cases"].append({"query": outcome.prompt, "trajectory": outcome.trajectory, "retrieved": retrieved, "skills": [skill["name"] for skill in selected_skills], "step": self.step})
        self._enforce_capacity()

    def summary(self) -> dict[str, Any]:
        total, memory_limit, skill_limit, hard_case_limit = self._capacity_limits()
        counts = {
            "memories": len(self.state["memories"]),
            "skills": len(self.state["skills"]),
            "hard_cases": len(self.state["hard_cases"]),
        }
        return {
            **super().summary(),
            "memory_count": sum(counts.values()),
            "max_memory_count": total,
            "component_counts": counts,
            "component_capacities": {
                "memories": memory_limit,
                "skills": skill_limit,
                "hard_cases": hard_case_limit,
            },
            "average_memory_chars": (
                sum(len(record["memory"]) for record in self.state["memories"])
                / len(self.state["memories"])
                if self.state["memories"] else 0.0
            ),
            "rejected_too_short": int(self.state.get("rejected_too_short", 0)),
        }
