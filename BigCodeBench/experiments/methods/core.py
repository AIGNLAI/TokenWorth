"""Shared contracts and utilities for BigCodeBench memory methods."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
LIST_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$")
AuxiliaryCall = Callable[[str], str]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + ("..." if limit >= 3 else "")


def memory_length_requirement(subject: str, minimum: int, maximum: int) -> str:
    """Render a length instruction, treating zero as no numeric lower bound."""
    if minimum > 0:
        return f"{subject} must be {minimum}-{maximum} characters."
    return f"{subject} must be non-empty and at most {maximum} characters."


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    if not documents:
        return []
    query_tokens = tokenize(query)
    counters = [Counter(tokenize(document)) for document in documents]
    lengths = [sum(counter.values()) for counter in counters]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    frequencies: Counter[str] = Counter()
    for counter in counters:
        frequencies.update(counter.keys())
    scores: list[float] = []
    for counter, length in zip(counters, lengths):
        score = 0.0
        for token in query_tokens:
            frequency = counter.get(token, 0)
            if not frequency:
                continue
            document_frequency = frequencies[token]
            inverse = math.log(
                1.0
                + (len(documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + 1.5 * (
                0.25 + 0.75 * (length / average if average else 0.0)
            )
            score += inverse * frequency * 2.5 / denominator
        scores.append(score)
    return scores


def rank_indices(query: str, documents: Sequence[str], top_k: int) -> list[int]:
    scored = [
        (score, index) for index, score in enumerate(bm25_scores(query, documents))
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [index for _, index in scored[: max(0, top_k)]]


def parse_bullets(text: str, limit: int = 3) -> list[str]:
    values = []
    for line in text.splitlines():
        match = LIST_RE.match(line)
        if match and normalize(match.group(1)) not in {
            normalize(item) for item in values
        }:
            values.append(match.group(1).strip())
            if len(values) >= limit:
                break
    if not values and text.strip():
        values.append(text.strip())
    return values


def parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.I)
    candidate = match.group(1) if match else text[text.find("{") : text.rfind("}") + 1]
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("auxiliary response is not a JSON object")
    return value


def string_list(value: Any, limit: int = 5) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        key = normalize(item)
        if key and key not in seen:
            result.append(item.strip())
            seen.add(key)
        if len(result) >= limit:
            break
    return result


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return (
        dot / (left_norm * right_norm)
        if left_norm > 1e-12 and right_norm > 1e-12
        else 0.0
    )


@dataclass
class TaskOutcome:
    epoch: int
    phase: str
    task_id: str
    prompt: str
    code: str
    status: str
    error: str

    @property
    def success(self) -> bool:
        return self.status == "pass"

    @property
    def trajectory(self) -> str:
        return truncate(
            f"Task:\n{self.prompt}\n\nGenerated code:\n{self.code}\n\n"
            f"Evaluation: {self.status}\n{self.error}",
            12000,
        )


@dataclass
class RetrievedItem:
    memory_id: int | str
    score: float
    content: str
    epoch: int | None = None
    phase: str = "unknown"
    task_id: str = "unknown"
    prompt: str = ""
    code: str = ""
    status: str = "unknown"
    error: str = ""
    include_method_memory: bool = True

    @classmethod
    def from_record(
        cls,
        record: dict[str, Any],
        score: float,
        content: str,
        *,
        include_method_memory: bool = True,
    ) -> "RetrievedItem":
        """Build an injected item from the supported method-state fields."""
        return cls(
            memory_id=record.get("id", record.get("mem_id", "unknown")),
            score=score,
            content=content,
            epoch=record.get("source_epoch", record.get("epoch")),
            phase=str(record.get("source_phase", record.get("phase", "unknown"))),
            task_id=str(
                record.get(
                    "source_task_id",
                    record.get("task_id", "unknown"),
                )
            ),
            prompt=str(
                record.get(
                    "source_prompt",
                    record.get("source_query", record.get("query", "")),
                )
            ),
            code=str(record.get("source_code", record.get("code", ""))),
            status=str(
                record.get(
                    "source_status",
                    record.get("source_outcome", record.get("outcome", "unknown")),
                )
            ),
            error=str(record.get("source_error", record.get("error", ""))),
            include_method_memory=include_method_memory,
        )


class MemoryMethod:
    method_name = "base"

    def __init__(
        self,
        state_dir: Path,
        config: dict[str, Any],
        auxiliary_call: AuxiliaryCall | None = None,
    ) -> None:
        self.state_dir = state_dir
        self.config = config
        self.auxiliary_call = auxiliary_call
        self.step = 0
        self.state: dict[str, Any] = {}
        self.last_retrieved: list[int] = []
        self.restore()

    @property
    def state_path(self) -> Path:
        return self.state_dir / "method_state.json"

    def restore(self) -> None:
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.step = int(payload.get("step", 0))
            self.state = payload.get("state", {})

    def save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"method": self.method_name, "step": self.step, "state": self.state},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        self.last_retrieved = []
        return []

    def retrieval_threshold(self, key: str, default: float) -> float:
        """Return the method gate, or disable rejection when configured.

        ``top_k`` is a maximum when zero retrieval is enabled.  A method may
        therefore return fewer candidates, including none, when its native
        relevance/value score does not clear the configured threshold.
        """
        if not self.config.get("allow_zero_retrieval", True):
            return float("-inf")
        return float(self.config.get(key, default))

    def update(self, outcome: TaskOutcome) -> None:
        self.step += 1

    def apply(self, outcome: TaskOutcome) -> bool:
        """Apply an outcome once, making task updates resume-safe."""
        key = f"{outcome.epoch}:{outcome.task_id}"
        applied = self.state.setdefault("_applied_tasks", [])
        if key in applied:
            return False
        self.update(outcome)
        applied.append(key)
        return True

    def context(self, items: Sequence[RetrievedItem]) -> str:
        if not items:
            return ""
        # This budget covers the complete injected context, including headers.
        # Per-memory limits remain method-specific snippet/trajectory settings.
        configured_limit = int(
            self.config.get(
                "max_context_chars",
                self.config.get("max_injected_chars_total", 8000),
            )
        )
        # A non-positive value explicitly disables the aggregate injection cap.
        limit = configured_limit if configured_limit > 0 else None
        header = "[Retrieved Memory Context]"
        parts = [header]
        remaining = None if limit is None else max(0, limit - len(header) - 1)
        for index, item in enumerate(items, 1):
            epoch = item.epoch if item.epoch is not None else "unknown"
            successful = item.status.strip().lower() in {
                "pass",
                "passed",
                "success",
                "successful",
                "correct",
            }
            code = (
                item.code
                if successful
                else "# Omitted: failed code is not a reusable solution."
            )
            evaluation = json.dumps(
                {"status": item.status, "error": item.error},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            block = (
                f"### Memory {index} (id={item.memory_id}, sim={item.score:.6f})\n"
                f"[BCB] epoch={epoch} phase={item.phase} task_id={item.task_id}\n"
                f"[PROMPT] {item.prompt}\n"
                "[GENERATED CODE] \n"
                "'''python \n"
                f"{code}\n"
                "'''\n"
                f"[EVAL] {evaluation}"
            )
            if item.include_method_memory and item.content.strip():
                block += f"\n[METHOD MEMORY]\n{item.content.strip()}"
            if remaining is not None:
                if remaining <= 0:
                    break
                block = truncate(block, remaining)
            parts.append(block)
            if remaining is not None:
                remaining -= len(block) + 1
        return "\n".join(parts)

    def auxiliary(self, prompt: str) -> str:
        if self.auxiliary_call is None:
            return ""
        try:
            return self.auxiliary_call(prompt)
        except Exception as error:
            errors = self.state.setdefault("auxiliary_errors", [])
            errors.append({"step": self.step, "error": str(error)})
            del errors[:-20]
            raise

    def summary(self) -> dict[str, Any]:
        return {"method": self.method_name, "step": self.step}
