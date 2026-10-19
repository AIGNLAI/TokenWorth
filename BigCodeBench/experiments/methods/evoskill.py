"""EvoSkill method implementation."""

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

class EvoSkillMethod(MemoryMethod):
    method_name = "EvoSkill"

    def restore(self) -> None:
        super().restore()
        self.state.setdefault("skills", [])
        self.state.setdefault("failures", [])
        self.state.setdefault("feedback", [])
        self.state.setdefault("recent_outcomes", [])
        self.state.setdefault("iteration", 0)
        self._enforce_capacity()

    def _capacity_limits(self) -> tuple[int, int, int, int]:
        total = int(self.config.get("max_memory_count", 50))
        skill_limit = int(self.config.get("max_skill_count", 30))
        failure_limit = int(self.config.get("max_failure_buffer_size", 12))
        feedback_limit = int(self.config.get("feedback_history_size", 8))
        if min(total, skill_limit, failure_limit, feedback_limit) < 1:
            raise ValueError("EvoSkill memory capacities must be positive")
        if skill_limit + failure_limit + feedback_limit > total:
            raise ValueError(
                "EvoSkill component capacities exceed max_memory_count: "
                f"{skill_limit} + {failure_limit} + {feedback_limit} > {total}"
            )
        return total, skill_limit, failure_limit, feedback_limit

    @staticmethod
    def _skill_score(skill: dict[str, Any]) -> float:
        attempts = skill["success_count"] + skill["failure_count"]
        return skill["success_count"] / attempts if attempts else skill["validation_score"]

    def _enforce_capacity(self) -> None:
        _, skill_limit, failure_limit, feedback_limit = self._capacity_limits()
        skills = self.state["skills"]
        candidates = [skill for skill in skills if skill["status"] == "candidate"][-1:]
        active = [skill for skill in skills if skill["status"] == "active"]
        active.sort(
            key=lambda skill: (
                self._skill_score(skill),
                skill["usage_count"],
                skill["created_step"],
            ),
            reverse=True,
        )
        self.state["skills"] = active[: max(0, skill_limit - len(candidates))] + candidates
        self.state["failures"] = self.state["failures"][-failure_limit:]
        self.state["feedback"] = self.state["feedback"][-feedback_limit:]

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        skills = self.state["skills"]
        eligible = [i for i, skill in enumerate(skills) if skill["status"] in {"active", "candidate"}]
        scores = bm25_scores(prompt, [skills[i]["search_text"] for i in eligible])
        ranked = sorted(
            ((score, eligible[local]) for local, score in enumerate(scores) if score >= self.retrieval_threshold("min_retrieval_score", 0.1)),
            reverse=True,
        )
        selected = [index for _, index in ranked[: int(self.config.get("top_k", 4))]]
        self.last_retrieved = selected
        return [
            RetrievedItem.from_record(
                skills[i],
                bm25_scores(prompt, [skills[i]["search_text"]])[0],
                f"Skill: {skills[i]['name']}\nTriggers: {'; '.join(skills[i]['triggers'])}\nProcedure:\n{skills[i]['instruction']}",
            )
            for i in selected
        ]

    def update(self, outcome: TaskOutcome) -> None:
        super().update(outcome)
        skills = self.state["skills"]
        retrieved_skill_names = [
            skills[index]["name"]
            for index in self.last_retrieved
            if index < len(skills)
        ]
        for index in self.last_retrieved:
            if index >= len(skills):
                continue
            skills[index]["usage_count"] += 1
            skills[index]["success_count" if outcome.success else "failure_count"] += 1
        recent = self.state["recent_outcomes"]
        recent.append(1 if outcome.success else 0)
        del recent[: max(0, len(recent) - int(self.config.get("baseline_window", 50)))]
        candidate = next((skill for skill in skills if skill["status"] == "candidate"), None)
        if candidate:
            candidate["validation_total_count"] += 1
            candidate["validation_correct_count"] += int(outcome.success)
            candidate["validation_score"] = candidate["validation_correct_count"] / candidate["validation_total_count"]
            if candidate["validation_total_count"] >= int(self.config.get("validation_window", 25)):
                threshold = max(float(self.config.get("min_acceptance_rate", 0.0)), candidate["baseline_score"] + float(self.config.get("min_score_delta", 0.0)))
                accepted = candidate["validation_score"] > threshold
                candidate["status"] = "active" if accepted else "rejected"
                self.state["feedback"].append(
                    {
                        "iteration": self.state["iteration"],
                        "candidate_skill_name": candidate["name"],
                        "decision": "accepted" if accepted else "rejected",
                        "baseline_score": candidate["baseline_score"],
                        "validation_score": candidate["validation_score"],
                        "created_step": self.step,
                    }
                )
                self._enforce_capacity()
                skills = self.state["skills"]
        if not outcome.success:
            self.state["failures"].append({"query": outcome.prompt, "trajectory": outcome.trajectory, "skills": retrieved_skill_names, "step": self.step, "source_epoch": outcome.epoch, "source_phase": outcome.phase, "source_task_id": outcome.task_id, "source_prompt": outcome.prompt, "source_code": outcome.code, "source_status": outcome.status, "source_error": outcome.error})
            del self.state["failures"][: max(0, len(self.state["failures"]) - int(self.config.get("max_failure_buffer_size", 64)))]
        if any(skill["status"] == "candidate" for skill in skills):
            return
        interval = int(self.config.get("evolution_interval", 25))
        batch_size = int(self.config.get("failure_batch_size", 4))
        if self.step % interval or len(self.state["failures"]) < batch_size:
            return
        failures = self.state["failures"][-batch_size:]
        length_requirement = memory_length_requirement(
            "The instruction",
            int(self.config.get("min_skill_chars", 600)),
            int(self.config.get("max_skill_chars", 1500)),
        )
        response = self.auxiliary(
            "You are EvoSkill. Build one general coding skill from these failures. Avoid task-specific literals "
            "but preserve exact technical triggers, operations, invariants, and checks. "
            f"{length_requirement} It must be a numbered procedure, not a placeholder such as 'checklist'.\n"
            f"Failures: {failures}\nExisting: {[s['name'] for s in skills if s['status']=='active']}\n"
            f"Recent feedback: {self.state['feedback']}\n"
            'Return JSON: {"name":"...","description":"...","triggers":[],"instruction":"numbered procedure","tags":[],"examples":[],"target_skill":null}'
        )
        try:
            payload = parse_json_object(response)
        except Exception:
            return
        if not all(str(payload.get(key, "")).strip() for key in ("name", "description", "instruction")):
            return
        instruction = str(payload["instruction"]).strip()
        minimum = int(self.config.get("min_skill_chars", 600))
        maximum = int(self.config.get("max_skill_chars", 1500))
        if len(instruction) < minimum or normalize(instruction) == "checklist":
            self.state["rejected_too_short"] = int(self.state.get("rejected_too_short", 0)) + 1
            return
        instruction = truncate(instruction, maximum)
        source = failures[-1]
        self.state["iteration"] += 1
        baseline = sum(recent) / len(recent) if recent else 0.0
        text = "\n".join([str(payload.get("name")), str(payload.get("description")), " ".join(string_list(payload.get("triggers"), 8)), instruction, " ".join(string_list(payload.get("tags"), 10))])
        skills.append({"id": str(uuid.uuid4()), "name": str(payload["name"]), "description": str(payload["description"]), "triggers": string_list(payload.get("triggers"), 8), "instruction": instruction, "tags": string_list(payload.get("tags"), 10), "examples": string_list(payload.get("examples"), 3), "search_text": text, "status": "candidate", "parent_skill_id": None, "source_epoch": source.get("source_epoch"), "source_phase": source.get("source_phase", "unknown"), "source_task_id": source.get("source_task_id", "unknown"), "source_prompt": source.get("source_prompt", source.get("query", "")), "source_code": source.get("source_code", ""), "source_status": source.get("source_status", "fail"), "source_error": source.get("source_error", ""), "created_step": self.step, "usage_count": 0, "success_count": 0, "failure_count": 0, "validation_total_count": 0, "validation_correct_count": 0, "baseline_score": baseline, "validation_score": 0.0})
        del self.state["failures"][-batch_size:]
        self._enforce_capacity()

    def summary(self) -> dict[str, Any]:
        total, skill_limit, failure_limit, feedback_limit = self._capacity_limits()
        counts = {
            "skills": len(self.state["skills"]),
            "failures": len(self.state["failures"]),
            "feedback": len(self.state["feedback"]),
        }
        return {
            **super().summary(),
            "memory_count": sum(counts.values()),
            "max_memory_count": total,
            "component_counts": counts,
            "component_capacities": {
                "skills": skill_limit,
                "failures": failure_limit,
                "feedback": feedback_limit,
            },
            "average_skill_chars": (
                sum(len(skill["instruction"]) for skill in self.state["skills"])
                / len(self.state["skills"])
                if self.state["skills"] else 0.0
            ),
            "rejected_too_short": int(self.state.get("rejected_too_short", 0)),
        }
