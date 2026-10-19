"""RAG method implementation."""

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
    normalize,
    parse_bullets,
    parse_json_object,
    rank_indices,
    string_list,
    tokenize,
    truncate,
)

class RAGMethod(MemoryMethod):
    method_name = "RAG"

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state.setdefault("experiences", [])
        indices = rank_indices(prompt, [record["search_text"] for record in records], int(self.config.get("top_k", 4)))
        scores = bm25_scores(prompt, [record["search_text"] for record in records])
        indices = [
            index for index in indices
            if scores[index] >= self.retrieval_threshold("min_retrieval_score", 0.1)
        ]
        self.last_retrieved = indices
        return [
            RetrievedItem.from_record(
                records[i], scores[i], records[i]["snippet"], include_method_memory=False
            )
            for i in indices
        ]

    def update(self, outcome: TaskOutcome) -> None:
        super().update(outcome)
        if not outcome.success:
            return
        records = self.state.setdefault("experiences", [])
        minimum = int(self.config.get("min_memory_chars", 600))
        maximum = int(self.config.get("max_memory_chars", 1500))
        snippet = truncate(outcome.trajectory, maximum)
        if len(snippet) < minimum:
            self.state["rejected_too_short"] = int(
                self.state.get("rejected_too_short", 0)
            ) + 1
            return
        records.append(
            {
                "id": int(self.state.get("next_id", 1)),
                "query": outcome.prompt,
                "snippet": snippet,
                "search_text": f"{outcome.prompt}\n{outcome.code}",
                "task_id": outcome.task_id,
                "source_epoch": outcome.epoch,
                "source_phase": outcome.phase,
                "source_task_id": outcome.task_id,
                "source_prompt": outcome.prompt,
                "source_code": outcome.code,
                "source_status": outcome.status,
                "source_error": outcome.error,
            }
        )
        self.state["next_id"] = records[-1]["id"] + 1
        del records[: max(0, len(records) - int(self.config.get("max_memory_count", 256)))]

    def summary(self) -> dict[str, Any]:
        records = self.state.get("experiences", [])
        lengths = [len(record["snippet"]) for record in records]
        return {
            **super().summary(),
            "memory_count": len(records),
            "average_memory_chars": sum(lengths) / len(lengths) if lengths else 0.0,
            "rejected_too_short": int(self.state.get("rejected_too_short", 0)),
        }
