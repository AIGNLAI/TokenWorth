"""EvoSC method implementation."""

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

class EvoSCMethod(MemoryMethod):
    method_name = "EvoSC"

    def restore(self) -> None:
        super().restore()
        self.state.setdefault("successes", [])
        self.state.setdefault("failures", [])
        self.state.setdefault("summaries", [])

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        candidates: list[tuple[str, dict[str, Any]]] = []
        candidates.extend(("summary", item) for item in self.state["summaries"])
        candidates.extend(("success", item) for item in self.state["successes"])
        candidates.extend(("failure", item) for item in self.state["failures"])
        limits = {"summary": 2, "success": min(2, int(self.config.get("top_k_success", 4))), "failure": min(2, int(self.config.get("top_k_failure", 4)))}
        selected: list[RetrievedItem] = []
        for kind in ("summary", "success", "failure"):
            local = [(offset, item) for offset, (candidate_kind, item) in enumerate(candidates) if candidate_kind == kind]
            if not local:
                continue
            scores = bm25_scores(prompt, [item["search_text"] for _, item in local])
            ranked = sorted(((scores[n], offset, item) for n, (offset, item) in enumerate(local) if scores[n] >= self.retrieval_threshold("min_retrieval_score", 0.1)), reverse=True)[: limits[kind]]
            selected.extend(
                RetrievedItem.from_record(item, score, f"[{kind}] {item['content']}")
                for score, _, item in ranked
            )
        selected.sort(key=lambda item: item.score, reverse=True)
        return selected[: int(self.config.get("top_k", 5))]

    def update(self, outcome: TaskOutcome) -> None:
        super().update(outcome)
        kind = "successes" if outcome.success else "failures"
        instruction = "reusable success strategies" if outcome.success else "failure patterns and corrective insights"
        experience_requirement = memory_length_requirement(
            "The experience",
            int(self.config.get("min_experience_chars", 350)),
            int(self.config.get("max_experience_chars", 1100)),
        )
        response = self.auxiliary(
            f"Extract one high-value {instruction} for a coding agent. Preserve exact applicability, "
            "technical operations, rationale, edge conditions, and a compact code idiom while removing "
            "task-specific literals. Avoid generic advice.\n"
            f"{outcome.trajectory}\n{experience_requirement} "
            'Return JSON only: {"experience":"..."}'
        )
        try:
            experience = str(parse_json_object(response).get("experience", "")).strip()
        except Exception:
            experience = response.strip()
        minimum = int(self.config.get("min_experience_chars", 350))
        if not experience or len(experience) < minimum:
            self.state["rejected_experiences"] = int(self.state.get("rejected_experiences", 0)) + 1
            return
        experience = truncate(experience, int(self.config.get("max_experience_chars", 1100)))
        new_items = [experience]
        records = self.state[kind]
        next_id = int(self.state.get("next_id", 1))
        for text in new_items:
            records.append({"id": next_id, "content": text, "search_text": f"{outcome.prompt}\n{text}", "task_id": outcome.task_id, "source_epoch": outcome.epoch, "source_phase": outcome.phase, "source_task_id": outcome.task_id, "source_prompt": outcome.prompt, "source_code": outcome.code, "source_status": outcome.status, "source_error": outcome.error})
            next_id += 1
        self.state["next_id"] = next_id
        maximum = int(self.config.get("max_success_count" if outcome.success else "max_failure_count", 64))
        del records[: max(0, len(records) - maximum)]
        current = "\n".join(f"- {item['content']}" for item in self.state["summaries"][-5:])
        summary_requirement = memory_length_requirement(
            "The summary",
            int(self.config.get("min_summary_chars", 500)),
            int(self.config.get("max_summary_chars", 1200)),
        )
        consolidated = self.auxiliary(
            "Consolidate the memories below into one high-precision cross-task strategy. Preserve concrete "
            "triggers, algorithms, library/API constraints, failure boundaries, and verification checks. "
            "Do not reduce them to generic coding advice.\n"
            f"Current:\n{current}\nLatest:\n{new_items}\n{summary_requirement} "
            'Return JSON only: {"summary":"..."}'
        )
        try:
            summary_text = str(parse_json_object(consolidated).get("summary", "")).strip()
        except Exception:
            summary_text = consolidated.strip()
        summaries = []
        if summary_text and len(summary_text) >= int(self.config.get("min_summary_chars", 500)):
            summary_text = truncate(summary_text, int(self.config.get("max_summary_chars", 1200)))
            summaries.append({"id": next_id, "content": summary_text, "search_text": summary_text, "source_epoch": outcome.epoch, "source_phase": outcome.phase, "source_task_id": outcome.task_id, "source_prompt": outcome.prompt, "source_code": outcome.code, "source_status": outcome.status, "source_error": outcome.error})
            next_id += 1
        else:
            self.state["rejected_summaries"] = int(self.state.get("rejected_summaries", 0)) + 1
        if summaries:
            previous = self.state["summaries"]
            self.state["summaries"] = (previous + summaries)[-int(self.config.get("max_summary_count", 32)) :]
            self.state["next_id"] = next_id

    def summary(self) -> dict[str, Any]:
        groups = {
            "successes": self.state.get("successes", []),
            "failures": self.state.get("failures", []),
            "summaries": self.state.get("summaries", []),
        }
        lengths = [len(item["content"]) for records in groups.values() for item in records]
        return {
            **super().summary(),
            "memory_count": sum(len(records) for records in groups.values()),
            "component_counts": {name: len(records) for name, records in groups.items()},
            "average_memory_chars": sum(lengths) / len(lengths) if lengths else 0.0,
            "rejected_experiences": int(self.state.get("rejected_experiences", 0)),
            "rejected_summaries": int(self.state.get("rejected_summaries", 0)),
        }
