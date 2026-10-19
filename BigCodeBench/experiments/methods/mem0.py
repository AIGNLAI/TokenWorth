"""Mem0 method implementation."""

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

class Mem0Method(MemoryMethod):
    method_name = "Mem0"

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state.setdefault("memories", [])
        eligible = [i for i, record in enumerate(records) if record["outcome"] == "pass"] or list(range(len(records)))
        documents = [records[i]["search_text"] for i in eligible]
        lexical = bm25_scores(prompt, documents)
        scored = []
        for local, index in enumerate(eligible):
            if lexical[local] < self.retrieval_threshold("min_retrieval_score", 0.1):
                continue
            record = records[index]
            recency = math.exp(-0.015 * max(0, self.step - record["last_access_step"]))
            score = lexical[local] + 1.5 * record["importance"] + 0.15 * record["mentions"] + recency
            scored.append((score, index))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected = scored[: int(self.config.get("top_k", 8))]
        self.last_retrieved = [index for _, index in selected]
        for _, index in selected:
            records[index]["last_access_step"] = self.step
            records[index]["mentions"] += 1
            records[index]["importance"] += 0.05
        return [
            RetrievedItem.from_record(records[i], score, records[i]["memory"])
            for score, i in selected
        ]

    def update(self, outcome: TaskOutcome) -> None:
        super().update(outcome)
        minimum = int(self.config.get("min_memory_chars", 180))
        maximum = int(self.config.get("max_memory_chars", 500))
        total_limit = int(
            self.config.get("max_auxiliary_memory_chars_total", maximum)
        )
        item_limit = int(self.config.get("memories_per_update", 3))
        length_requirement = memory_length_requirement(
            "Each memory", minimum, maximum
        )
        response = self.auxiliary(
            "You are Mem0, a long-term memory extractor for a coding agent.\n"
            f"{outcome.trajectory}\nExtract at most {item_limit} durable, atomic, reusable memories. "
            f"{length_requirement} State the exact applicability, "
            "action, rationale, and a concrete edge condition. Reject generic advice. "
            f"The combined memory text across all returned items must not exceed {total_limit} characters. "
            'Return JSON: {"memories":[{"memory":"...","category":"fact|constraint|procedure","entities":[]}]}'
        )
        try:
            items = parse_json_object(response).get("memories", [])
        except Exception:
            return
        records = self.state.setdefault("memories", [])
        remaining = total_limit
        for item in items[:item_limit] if isinstance(items, list) else []:
            if not isinstance(item, dict) or not str(item.get("memory", "")).strip():
                continue
            text = str(item["memory"]).strip()
            if len(text) < minimum:
                self.state["rejected_too_short"] = int(self.state.get("rejected_too_short", 0)) + 1
                continue
            text = truncate(text, maximum)
            if len(text) > remaining:
                self.state["rejected_over_total_budget"] = int(
                    self.state.get("rejected_over_total_budget", 0)
                ) + 1
                continue
            remaining -= len(text)
            existing = next((record for record in records if normalize(record["memory"]) == normalize(text)), None)
            if existing:
                existing["mentions"] += 1
                existing["importance"] += 0.4 if outcome.success else 0.2
                existing["last_access_step"] = self.step
                if outcome.success:
                    existing["outcome"] = "pass"
                continue
            memory_id = int(self.state.get("next_id", 1))
            self.state["next_id"] = memory_id + 1
            records.append({"id": memory_id, "memory": text, "category": str(item.get("category", "fact")), "entities": string_list(item.get("entities"), 10), "source_query": outcome.prompt, "search_text": f"{outcome.prompt}\n{text}", "outcome": outcome.status, "source_epoch": outcome.epoch, "source_phase": outcome.phase, "source_task_id": outcome.task_id, "source_prompt": outcome.prompt, "source_code": outcome.code, "source_status": outcome.status, "source_error": outcome.error, "created_step": self.step, "last_access_step": self.step, "mentions": 1, "importance": 1.0 if outcome.success else 0.7})
        records.sort(key=lambda record: (record["importance"], record["mentions"], record["last_access_step"]), reverse=True)
        del records[int(self.config.get("max_memory_count", 512)) :]

    def summary(self) -> dict[str, Any]:
        records = self.state.get("memories", [])
        lengths = [len(record["memory"]) for record in records]
        return {
            **super().summary(),
            "memory_count": len(records),
            "average_memory_chars": sum(lengths) / len(lengths) if lengths else 0.0,
            "rejected_too_short": int(self.state.get("rejected_too_short", 0)),
        }
