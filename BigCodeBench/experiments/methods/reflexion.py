"""Reflexion method implementation."""

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

class ReflexionMethod(MemoryMethod):
    method_name = "Reflexion"

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state.setdefault("reflections", [])
        indices = rank_indices(prompt, [record["search_text"] for record in records], int(self.config.get("top_k", 4)))
        scores = bm25_scores(prompt, [record["search_text"] for record in records])
        indices = [
            index for index in indices
            if scores[index] >= self.retrieval_threshold("min_retrieval_score", 0.1)
        ]
        self.last_retrieved = indices
        return [
            RetrievedItem.from_record(records[i], scores[i], records[i]["reflection"])
            for i in indices
        ]

    def update(self, outcome: TaskOutcome) -> None:
        super().update(outcome)
        if outcome.success:
            return
        length_requirement = memory_length_requirement(
            "The complete reflection",
            int(self.config.get("min_memory_chars", 400)),
            int(self.config.get("max_memory_chars", 1500)),
        )
        response = self.auxiliary(
            "You are writing reusable failure memory for a coding agent.\n"
            f"Task:\n{outcome.prompt}\n\nFailed attempt:\n{outcome.trajectory}\n\n"
            "Diagnose the exact failed assumption and produce a correction that can change a future "
            "implementation. Include applicability, root cause, corrected procedure, and verification checks. "
            f"{length_requirement} Avoid generic advice. "
            'Return JSON only: {"failure_pattern":"...","root_cause":"...",'
            '"corrective_procedure":"...","verification_checks":["..."]}'
        )
        records = self.state.setdefault("reflections", [])
        try:
            payload = parse_json_object(response)
            checks = string_list(payload.get("verification_checks"), 4)
            reflection = "\n".join(
                part for part in (
                    f"[Failure Pattern] {str(payload.get('failure_pattern', '')).strip()}",
                    f"[Root Cause] {str(payload.get('root_cause', '')).strip()}",
                    f"[Corrective Procedure] {str(payload.get('corrective_procedure', '')).strip()}",
                    "[Verification Checks]\n" + "\n".join(f"- {item}" for item in checks),
                ) if part.split("]", 1)[-1].strip()
            )
        except Exception:
            reflection = response.strip()
        minimum = int(self.config.get("min_memory_chars", 400))
        if not reflection or len(reflection) < minimum:
            self.state["rejected_too_short"] = int(self.state.get("rejected_too_short", 0)) + 1
            return
        reflection = truncate(reflection, int(self.config.get("max_memory_chars", 1500)))
        if any(normalize(record["reflection"]) == normalize(reflection) for record in records):
            return
        next_id = int(self.state.get("next_id", 1))
        records.append({"id": next_id, "query": outcome.prompt, "reflection": reflection, "search_text": f"{outcome.prompt}\n{reflection}", "source_epoch": outcome.epoch, "source_phase": outcome.phase, "source_task_id": outcome.task_id, "source_prompt": outcome.prompt, "source_code": outcome.code, "source_status": outcome.status, "source_error": outcome.error})
        self.state["next_id"] = next_id + 1
        del records[: max(0, len(records) - int(self.config.get("max_memory_count", 256)))]

    def summary(self) -> dict[str, Any]:
        records = self.state.get("reflections", [])
        lengths = [len(record["reflection"]) for record in records]
        return {
            **super().summary(),
            "memory_count": len(records),
            "average_memory_chars": sum(lengths) / len(lengths) if lengths else 0.0,
            "rejected_too_short": int(self.state.get("rejected_too_short", 0)),
        }
