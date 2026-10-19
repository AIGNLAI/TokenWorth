"""MemRL: procedural memories with relevance- and value-driven retrieval."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from typing import Any

from .core import (
    MemoryMethod,
    RetrievedItem,
    TaskOutcome,
    memory_length_requirement,
    tokenize,
    truncate,
)


class MemRLMethod(MemoryMethod):
    method_name = "MemRL"
    STOPWORDS = {
        "the", "and", "for", "from", "with", "this", "that", "return",
        "name", "into", "are", "has",
    }

    @classmethod
    def _terms(cls, text: str) -> Counter[str]:
        return Counter(
            token for token in tokenize(text)
            if len(token) > 2 and token not in cls.STOPWORDS
        )

    @classmethod
    def _similarity(cls, left: str, right: str) -> float:
        """Cosine similarity over lexical term-frequency vectors."""
        left_terms = cls._terms(left)
        right_terms = cls._terms(right)
        if not left_terms or not right_terms:
            return 0.0
        common = left_terms.keys() & right_terms.keys()
        dot = sum(left_terms[token] * right_terms[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left_terms.values()))
        right_norm = math.sqrt(sum(value * value for value in right_terms.values()))
        return dot / (left_norm * right_norm)

    @staticmethod
    def _q_score(q_value: float) -> float:
        return (max(-1.0, min(1.0, q_value)) + 1.0) / 2.0

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state.setdefault("episodes", [])
        threshold = self.retrieval_threshold("similarity_threshold", 0.20)
        candidates = []
        for index, record in enumerate(records):
            similarity = self._similarity(prompt, record["search_text"])
            if similarity >= threshold:
                candidates.append((similarity, index))

        # Phase 1 is a locality filter. Phase 2 combines task similarity with
        # learned utility; unlike the previous formula, both terms are in [0, 1].
        candidates.sort(reverse=True)
        candidates = candidates[: int(self.config.get("candidate_count", 16))]
        similarity_weight = float(self.config.get("weight_similarity", 0.5))
        q_weight = float(self.config.get("weight_q", 0.5))
        denominator = similarity_weight + q_weight
        if denominator <= 0:
            similarity_weight, q_weight, denominator = 1.0, 0.0, 1.0
        ranked = [
            (
                (
                    similarity_weight * similarity
                    + q_weight * self._q_score(float(records[index]["q_value"]))
                ) / denominator,
                index,
            )
            for similarity, index in candidates
        ]
        ranked.sort(reverse=True)

        top_k = int(self.config.get("top_k", 5))
        epsilon = float(self.config.get("epsilon", 0.0))
        rng = random.Random(int(self.config.get("random_seed", 42)) + self.step)
        if ranked and epsilon > 0 and rng.random() < epsilon:
            selected = rng.sample(ranked, min(top_k, len(ranked)))
            selected.sort(reverse=True)
        else:
            selected = ranked[:top_k]

        self.last_retrieved = [index for _, index in selected]
        for _, index in selected:
            records[index]["retrieve_count"] += 1
            records[index]["last_access_step"] = self.step
        return [
            RetrievedItem.from_record(
                records[index], score, records[index]["snippet"]
            )
            for score, index in selected
        ]

    def _proceduralization_prompt(self, outcome: TaskOutcome) -> str:
        minimum = int(self.config.get("min_memory_chars", 600))
        maximum = int(self.config.get("max_memory_chars", 1500))
        length_requirement = memory_length_requirement(
            "The rendered memory", minimum, maximum
        )
        return f"""You are building reusable procedural memory for future Python coding tasks.
Distill the successful task below into a strategy, not a task transcript and not a copy of the full solution.
The memory must be specific enough to change a future implementation decision.

Return exactly one JSON object with these fields:
- problem_pattern: concise description of when this memory applies
- key_insight: the non-obvious reason the solution works
- procedure: 3 to 6 concrete ordered steps
- pitfalls: 1 to 3 concrete failure modes or edge cases
- code_pattern: a short Python idiom or pseudocode fragment, never the full task solution
- retrieval_terms: up to 8 discriminative technical terms

{length_requirement} Avoid generic advice such as
"validate inputs", "handle edge cases", or "write clean code" unless you name the exact condition and action.

Task:
{outcome.prompt}

Verified code:
{outcome.code}
"""

    def _render_procedural_memory(self, payload: dict[str, Any]) -> str:
        def text(name: str) -> str:
            value = payload.get(name, "")
            return value.strip() if isinstance(value, str) else ""

        def strings(name: str, limit: int) -> list[str]:
            value = payload.get(name, [])
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                return []
            return [
                item.strip() for item in value
                if isinstance(item, str) and item.strip()
            ][:limit]

        pattern = text("problem_pattern")
        insight = text("key_insight")
        procedure = strings("procedure", 6)
        pitfalls = strings("pitfalls", 3)
        code_pattern = text("code_pattern")
        retrieval_terms = strings(
            "retrieval_terms", int(self.config.get("max_keywords", 8))
        )
        if not pattern or not insight or len(procedure) < 3:
            return ""

        lines = [
            f"[Applicable Pattern] {pattern}",
            f"[Key Insight] {insight}",
            "[Procedure]",
        ]
        lines.extend(f"{index}. {step}" for index, step in enumerate(procedure, 1))
        if pitfalls:
            lines.append("[Pitfalls]")
            lines.extend(f"- {item}" for item in pitfalls)
        if code_pattern:
            lines.extend(("[Code Pattern]", code_pattern))
        if retrieval_terms:
            lines.append("[Retrieval Terms] " + ", ".join(retrieval_terms))
        return "\n".join(lines)

    def _build_memory(self, outcome: TaskOutcome) -> tuple[str, str]:
        minimum = int(self.config.get("min_memory_chars", 600))
        maximum = int(self.config.get("max_memory_chars", 1500))
        if minimum < 0 or maximum < max(1, minimum):
            raise ValueError("MemRL requires 0 <= min_memory_chars <= max_memory_chars")
        strategy = str(self.config.get("memory_build_strategy", "proceduralization"))
        rendered = ""
        response = (
            self.auxiliary(self._proceduralization_prompt(outcome))
            if strategy == "proceduralization"
            else ""
        )
        if response:
            try:
                candidate = response
                if "```json" in response:
                    candidate = response.split("```json", 1)[1].split("```", 1)[0]
                elif "```" in response:
                    candidate = response.split("```", 1)[1].split("```", 1)[0]
                rendered = self._render_procedural_memory(json.loads(candidate))
            except (json.JSONDecodeError, TypeError, ValueError):
                rendered = ""

        if rendered and len(rendered) >= minimum:
            return truncate(rendered, maximum), "proceduralized"

        # Never admit a tiny generic memory: retain bounded verified evidence
        # when distillation fails or is underspecified.
        fallback = (
            f"[Applicable Pattern]\n{truncate(outcome.prompt, maximum // 2)}\n"
            f"[Verified Implementation]\n{outcome.code}"
        )
        fallback = truncate(fallback, maximum)
        if len(fallback) < minimum:
            return "", "rejected_too_short"
        return fallback, "verified_fallback"

    def _find_duplicate(self, search_text: str) -> int | None:
        threshold = float(self.config.get("add_similarity_threshold", 0.9))
        best: tuple[float, int] | None = None
        for index, record in enumerate(self.state.setdefault("episodes", [])):
            similarity = self._similarity(search_text, record["search_text"])
            if similarity >= threshold and (best is None or similarity > best[0]):
                best = (similarity, index)
        return None if best is None else best[1]

    def update(self, outcome: TaskOutcome) -> None:
        super().update(outcome)
        records = self.state.setdefault("episodes", [])
        reward = (
            float(self.config.get("success_reward", 1.0))
            if outcome.success
            else float(self.config.get("failure_reward", -1.0))
        )
        learning_rate = float(self.config.get("learning_rate", 0.3))
        q_minimum = float(self.config.get("q_min_threshold", -0.8))
        for index in self.last_retrieved:
            record = records[index]
            record["q_value"] += learning_rate * (reward - record["q_value"])
            record["q_value"] = max(q_minimum, min(1.0, record["q_value"]))
            record["success_count" if outcome.success else "failure_count"] += 1
            record["last_access_step"] = self.step

        if not outcome.success and not self.config.get("store_incorrect_episodes", False):
            return

        snippet, build_status = self._build_memory(outcome)
        if not snippet:
            rejected = self.state.setdefault("rejected_builds", {})
            rejected[build_status] = int(rejected.get(build_status, 0)) + 1
            return
        search_text = f"{outcome.prompt}\n{snippet}"
        duplicate_index = self._find_duplicate(search_text)
        if duplicate_index is not None:
            duplicate = records[duplicate_index]
            if len(snippet) > len(duplicate["snippet"]):
                duplicate["snippet"] = snippet
                duplicate["query"] = outcome.prompt
                duplicate["search_text"] = search_text
                duplicate["build_status"] = build_status
                duplicate.update(
                    source_epoch=outcome.epoch,
                    source_phase=outcome.phase,
                    source_task_id=outcome.task_id,
                    source_prompt=outcome.prompt,
                    source_code=outcome.code,
                    source_status=outcome.status,
                    source_error=outcome.error,
                )
            duplicate["last_access_step"] = self.step
            duplicate["merge_count"] = int(duplicate.get("merge_count", 0)) + 1
            merge_gain = float(self.config.get("reward_merge_gain", 0.1))
            duplicate["q_value"] += merge_gain * (reward - duplicate["q_value"])
            duplicate["q_value"] = max(q_minimum, min(1.0, duplicate["q_value"]))
            return

        memory_id = int(self.state.get("next_id", 1))
        self.state["next_id"] = memory_id + 1
        records.append({
            "id": memory_id,
            "query": outcome.prompt,
            "snippet": snippet,
            "search_text": search_text,
            "outcome": outcome.status,
            "build_status": build_status,
            "source_epoch": outcome.epoch,
            "source_phase": outcome.phase,
            "source_task_id": outcome.task_id,
            "source_prompt": outcome.prompt,
            "source_code": outcome.code,
            "source_status": outcome.status,
            "source_error": outcome.error,
            "q_value": 0.0,
            "retrieve_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "merge_count": 0,
            "created_step": self.step,
            "last_access_step": self.step,
        })
        records.sort(
            key=lambda record: (
                record["q_value"], record["retrieve_count"], record["last_access_step"]
            ),
            reverse=True,
        )
        del records[int(self.config.get("max_memory_count", 50)):]

    def summary(self) -> dict[str, Any]:
        records = self.state.get("episodes", [])
        statuses = Counter(record.get("build_status", "unknown") for record in records)
        lengths = [len(record.get("snippet", "")) for record in records]
        return {
            **super().summary(),
            "memory_count": len(records),
            "average_memory_chars": sum(lengths) / len(lengths) if lengths else 0.0,
            "build_status_counts": dict(statuses),
            "rejected_builds": dict(self.state.get("rejected_builds", {})),
        }
