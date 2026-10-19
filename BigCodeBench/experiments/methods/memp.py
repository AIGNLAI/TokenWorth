"""MemP method implementation."""

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

class MemPMethod(MemoryMethod):
    method_name = "MemP"

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        records = self.state.setdefault("procedures", [])
        eligible = [i for i, record in enumerate(records) if record["status"] == "active"]
        scores = bm25_scores(prompt, [records[i]["search_text"] for i in eligible])
        ranked = sorted(
            ((scores[n] + 2.0 * (records[i]["support_count"] - 0.75 * records[i]["contradiction_count"]), i) for n, i in enumerate(eligible) if scores[n] >= self.retrieval_threshold("min_retrieval_score", 0.1)),
            reverse=True,
        )
        selected: list[tuple[float, int]] = []
        for kind, limit in (("abstract", 1), ("detailed", 2), ("caution", 1)):
            selected.extend([item for item in ranked if records[item[1]]["kind"] == kind][:limit])
        seen = {index for _, index in selected}
        selected.extend(item for item in ranked if item[1] not in seen)
        selected = selected[: min(int(self.config.get("top_k", 8)), 4)]
        self.last_retrieved = [index for _, index in selected]
        return [
            RetrievedItem.from_record(
                records[i], score, f"[{records[i]['kind']}] {records[i]['memory']}"
            )
            for score, i in selected
        ]

    def _upsert(self, text: str, kind: str, outcome: TaskOutcome) -> None:
        minimum = int(self.config.get("min_memory_chars", 180))
        maximum = int(self.config.get("max_memory_chars", 900))
        text = text.strip()
        if not text or len(text) < minimum:
            self.state["rejected_too_short"] = int(self.state.get("rejected_too_short", 0)) + 1
            return
        text = truncate(text, maximum)
        records = self.state.setdefault("procedures", [])
        existing = next((record for record in records if normalize(record["memory"]) == normalize(text)), None)
        if existing:
            existing.update(status="active", support_count=existing["support_count"] + 1, last_access_step=self.step)
            return
        memory_id = int(self.state.get("next_id", 1))
        self.state["next_id"] = memory_id + 1
        records.append({"id": memory_id, "memory": text.strip(), "kind": kind, "status": "active", "source_query": outcome.prompt, "search_text": f"{outcome.prompt}\n{kind}\n{text}", "source_epoch": outcome.epoch, "source_phase": outcome.phase, "source_task_id": outcome.task_id, "source_prompt": outcome.prompt, "source_code": outcome.code, "source_status": outcome.status, "source_error": outcome.error, "created_step": self.step, "last_access_step": self.step, "support_count": 1, "contradiction_count": 0})

    def update(self, outcome: TaskOutcome) -> None:
        super().update(outcome)
        minimum = int(self.config.get("min_memory_chars", 180))
        maximum = int(self.config.get("max_memory_chars", 900))
        total_limit = int(
            self.config.get("max_auxiliary_memory_chars_total", maximum)
        )
        item_requirement = memory_length_requirement(
            "Each item", minimum, maximum
        )
        caution_requirement = memory_length_requirement(
            "Each caution", minimum, maximum
        )
        if outcome.success:
            prompt = (
                "Distill this verified coding trajectory into reusable procedural memory. "
                "Every item must name its applicability, exact operations, rationale, and edge conditions; "
                "do not return generic coding advice.\n"
                f"{outcome.trajectory}\nReturn JSON with detailed_steps (max 2) and abstractions (max 1). "
                f"{item_requirement} The combined text of detailed_steps "
                f"and abstractions must not exceed {total_limit} characters."
            )
        else:
            retrieved = [self.state["procedures"][i]["memory"] for i in self.last_retrieved]
            prompt = (
                "Update procedural memory after failure. Diagnose the concrete failed assumption and give "
                "a transferable correction with an explicit trigger and check.\n"
                f"Retrieved: {retrieved}\n{outcome.trajectory}\nReturn JSON with cautions (max 2) "
                f"and deprecate (exact retrieved text, max 2). {caution_requirement} "
                f"The combined text of cautions must not exceed {total_limit} characters."
            )
        response = self.auxiliary(prompt)
        try:
            payload = parse_json_object(response)
        except Exception:
            return
        remaining = total_limit

        def upsert_with_budget(values: list[str], kind: str) -> None:
            nonlocal remaining
            for value in values:
                text = value.strip()
                if len(text) < minimum:
                    self._upsert(text, kind, outcome)
                    continue
                text = truncate(text, maximum)
                if len(text) > remaining:
                    self.state["rejected_over_total_budget"] = int(
                        self.state.get("rejected_over_total_budget", 0)
                    ) + 1
                    continue
                remaining -= len(text)
                self._upsert(text, kind, outcome)

        if outcome.success:
            upsert_with_budget(string_list(payload.get("detailed_steps"), 2), "detailed")
            upsert_with_budget(string_list(payload.get("abstractions"), 1), "abstract")
        else:
            upsert_with_budget(string_list(payload.get("cautions"), 2), "caution")
            for text in string_list(payload.get("deprecate"), 2):
                for record in self.state.setdefault("procedures", []):
                    if normalize(record["memory"]) == normalize(re.sub(r"^\[(?:detailed|abstract|caution)\]\s*", "", text)):
                        record["contradiction_count"] += 1
                        if record["contradiction_count"] > record["support_count"]:
                            record["status"] = "deprecated"
        records = self.state.setdefault("procedures", [])
        records.sort(key=lambda record: (record["status"] == "active", record["support_count"] - record["contradiction_count"], record["last_access_step"]), reverse=True)
        del records[int(self.config.get("max_memory_count", 256)) :]

    def summary(self) -> dict[str, Any]:
        records = self.state.get("procedures", [])
        lengths = [len(record["memory"]) for record in records]
        return {
            **super().summary(),
            "memory_count": len(records),
            "average_memory_chars": sum(lengths) / len(lengths) if lengths else 0.0,
            "rejected_too_short": int(self.state.get("rejected_too_short", 0)),
        }
