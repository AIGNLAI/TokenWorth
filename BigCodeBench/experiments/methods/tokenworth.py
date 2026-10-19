"""BigCodeBench adapter for the shared manuscript-aligned TokenWorth core."""

from __future__ import annotations

import ast
from copy import deepcopy
from typing import Any, Sequence

from tokenworth_core import VERSION, TokenWorthCore
from .core import MemoryMethod, RetrievedItem, TaskOutcome, truncate


class TokenWorthMethod(MemoryMethod):
    method_name = "TokenWorth"

    def restore(self) -> None:
        existed = self.state_path.exists()
        super().restore()
        config = dict(self.config)
        variant = config.pop("implementation_variant", "paper")
        if variant != "paper":
            raise ValueError(
                "Legacy simplified/full TokenWorth is unsupported; use implementation_variant=paper and a new output directory"
            )
        capacity = config.pop("max_memory_count", 50)
        self.max_memory_chars = int(config.pop("max_memory_chars", 4096))
        self.max_context_chars = int(config.pop("max_context_chars", 4096))
        if self.max_memory_chars < 1 or self.max_context_chars < 128:
            raise ValueError(
                "TokenWorth requires positive snippet and usable bounded context limits"
            )
        self.core = TokenWorthCore(
            {**config, "capacity": capacity},
            self.state.get("paper") if existed else None,
        )
        if existed and (
            "paper" not in self.state or self.step != self.core.state["step"]
        ):
            raise ValueError(
                "Incompatible legacy/incomplete TokenWorth state; use a new output directory"
            )
        self.state["paper"] = self.core.state

    def retrieve(self, prompt: str) -> list[RetrievedItem]:
        selected = self.core.retrieve(prompt)
        self.last_retrieved = [record["id"] for _, record in selected]
        return [
            RetrievedItem.from_record(
                {**record, **record.get("metadata", {})}, score, record["snippet"]
            )
            for score, record in selected
        ]

    def context(self, items: Sequence[RetrievedItem]) -> str:
        if not items:
            self.core.confirm_injection([])
            return ""
        parts = ["[Retrieved Memory Context]"]
        used = len(parts[0])
        injected = []
        for item in items:
            header = "\n### Verified past experience {}\n".format(item.memory_id)
            remaining = self.max_context_chars - used - len(header)
            if remaining < 1:
                break
            content = item.content
            if len(content) > remaining:
                marker = "[Incomplete excerpt of a verified experience]\n"
                if remaining <= len(marker):
                    break
                content = marker + content
            snippet = truncate(content, remaining)
            if not snippet:
                break
            parts.append(header + snippet)
            used += len(header) + len(snippet)
            injected.append(item.memory_id)
        self.core.confirm_injection(injected)
        self.last_retrieved = injected
        return "".join(parts) if injected else ""

    def update(self, outcome: TaskOutcome) -> None:
        if outcome.status not in {"pass", "fail", "timeout"}:
            raise RuntimeError(
                "Evaluator infrastructure error is not TokenWorth feedback: "
                + outcome.status
            )
        # Verified generated code only; no reference implementation, tests or
        # extra LLM calls are supplied to the learner.
        snippet = ""
        if outcome.success:
            prompt = truncate(outcome.prompt, min(768, self.max_memory_chars // 4))
            if not outcome.code.strip():
                raise RuntimeError(
                    "Evaluator marked empty generated code as successful"
                )
            tree = ast.parse(outcome.code)
            full = "Task: {}\nVerified Python solution:\n{}".format(
                prompt, outcome.code
            )
            if len(full) <= self.max_memory_chars:
                snippet = full
            else:
                # Never present a syntactically truncated program as a complete
                # verified solution. A bounded outline is a procedural memory.
                lines = []
                for node in tree.body:
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        lines.append(ast.unparse(node))
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        lines.append(
                            "Function {}({})".format(node.name, ast.unparse(node.args))
                        )
                    elif isinstance(node, ast.ClassDef):
                        lines.append("Class " + node.name)
                calls = list(
                    dict.fromkeys(
                        ast.unparse(node.func)
                        for node in ast.walk(tree)
                        if isinstance(node, ast.Call)
                    )
                )[:64]
                lines.append("APIs used: " + ", ".join(calls))
                snippet = truncate(
                    "Task: {}\nSuccessful solution outline (not executable code):\n{}".format(
                        prompt, "\n".join(lines)
                    ),
                    self.max_memory_chars,
                )
        self.core.feedback(
            outcome.success,
            snippet,
            {
                "source_epoch": outcome.epoch,
                "source_phase": outcome.phase,
                "source_task_id": outcome.task_id,
                "source_status": outcome.status,
            },
        )
        self.step = self.core.state["step"]

    def save(self) -> None:
        self.core.validate_state()
        super().save()

    def summary(self) -> dict[str, Any]:
        return {
            **super().summary(),
            "implementation": VERSION,
            "parameters": deepcopy(self.core.parameters),
            "memory_count": len(self.core.state["memories"]),
            "metrics": dict(self.core.state["metrics"]),
        }
