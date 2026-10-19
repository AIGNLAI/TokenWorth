"""Run memory methods on BigCodeBench in fixed task order."""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any


from tokenworth_core.runtime import (
    CheckpointStore,
    RunGuard,
    atomic_json,
    fingerprint,
    redacted_config,
    source_identity,
)
from experiments.methods import TaskOutcome, create_method
from experiments.memory_runner import (
    DEFAULT_BCB_SYSTEM_PROMPT,
    OpenAICompatibleGenerator,
    TaskEvaluator,
    append_jsonl,
    build_results_summary,
    build_system_prompt,
    build_user_prompt,
    load_json,
    load_ordering_datasets,
    ordered_tasks,
    task_order_manifest,
    sanitize_generated_code,
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_method_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    parent = config.get("extends")
    if not parent:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return deep_merge(load_method_config(parent_path), config)


class MethodExperimentRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.method_name = str(config["method"])
        self.split = config.get("dataset", {}).get("split", "instruct")
        self.output_dir = Path(config["run"]["output_dir"])
        self.results_path = self.output_dir / "results.jsonl"
        self.pending_path = self.output_dir / "pending_task.json"
        self.auxiliary_usage_path = self.output_dir / "auxiliary_usage.jsonl"
        self._active_auxiliary_task: tuple[int, str] | None = None
        self._active_auxiliary_calls: list[dict[str, Any]] | None = None
        backend = config["evaluation"].get("backend")
        if backend not in {"docker", "gradio"}:
            raise ValueError(
                "Memory methods require immediate per-task evaluation; "
                "use evaluation.backend=docker or gradio"
            )
        if not config["model"].get("name") or config["model"]["name"] == "auto":
            raise ValueError(
                "Use an explicit model.name so resume can freeze model identity"
            )
        self.generator = OpenAICompatibleGenerator(config["model"])
        auxiliary_config = deepcopy(config["model"])
        auxiliary_config["temperature"] = 0.0
        auxiliary_config["max_tokens"] = int(
            config.get("method_parameters", {}).get("auxiliary_max_tokens")
            or config.get("auxiliary", {}).get("max_tokens", 512)
        )
        auxiliary_config["thinking_mode"] = config.get("auxiliary", {}).get(
            "thinking_mode", auxiliary_config.get("thinking_mode")
        )
        self.auxiliary_generator = OpenAICompatibleGenerator(auxiliary_config)
        self.evaluator = TaskEvaluator(config["evaluation"], self.split)
        self.method = create_method(
            self.method_name,
            self.output_dir,
            config.get("method_parameters", {}),
            self._generate_auxiliary,
        )

    def _generate_auxiliary(self, prompt: str) -> str:
        """Record every successful auxiliary API response, including discarded memory."""
        if self._active_auxiliary_task is None or self._active_auxiliary_calls is None:
            raise RuntimeError("auxiliary call outside an active task")
        response = self.auxiliary_generator.generate(prompt, "")
        epoch, task_id = self._active_auxiliary_task
        call = {
            "method": self.method_name,
            "epoch": epoch,
            "task_id": task_id,
            "call_index": len(self._active_auxiliary_calls) + 1,
            "generation_metadata": dict(
                self.auxiliary_generator.last_generation_metadata
            ),
        }
        # The append-only ledger also preserves calls made before an interrupted
        # task is retried. Those calls consumed tokens even if no memory survived.
        append_jsonl(self.auxiliary_usage_path, call)
        self._active_auxiliary_calls.append(call)
        return response

    def _apply_with_usage(self, outcome: TaskOutcome, record: dict[str, Any]) -> None:
        self._active_auxiliary_task = (outcome.epoch, outcome.task_id)
        self._active_auxiliary_calls = list(
            record.get("auxiliary_generation_metadata", [])
        )
        try:
            self.method.apply(outcome)
            record["auxiliary_generation_metadata"] = self._active_auxiliary_calls
        finally:
            self._active_auxiliary_task = None
            self._active_auxiliary_calls = None

    def completed(self) -> set[tuple[int, str]]:
        return {(r["epoch"], r["task_id"]) for r in getattr(self, "records", [])}

    def run(self, limit: int | None = None) -> None:
        if limit is not None and limit < 1:
            raise ValueError("--limit must be positive")
        full, hard = load_ordering_datasets()
        tasks = ordered_tasks(full, hard)
        manifest = task_order_manifest(tasks)
        evaluator_identity = dict(self.config["evaluation"])
        if evaluator_identity.get("backend") == "docker":
            image = evaluator_identity.get(
                "image", "bigcodebench-evaluator-local:latest"
            )
            immutable = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if not immutable.startswith("sha256:"):
                raise RuntimeError("Cannot resolve immutable evaluator image")
            evaluator_identity["image_id"] = immutable
            self.evaluator.config = deepcopy(self.evaluator.config)
            self.evaluator.config["image"] = immutable
        identity = {
            "config": redacted_config(self.config),
            "stream": fingerprint(tasks),
            "implementation": source_identity(Path(__file__).resolve().parents[1]),
            "evaluator": evaluator_identity,
            "runtime_model": {
                label: {
                    "base_url": getattr(generator, "resolved_base_url", None),
                    "thinking_mode": getattr(generator, "thinking_mode", None),
                }
                for label, generator in (
                    ("solver", self.generator),
                    ("auxiliary", self.auxiliary_generator),
                )
            },
        }
        self.guard = RunGuard(self.output_dir, identity)
        self.store = CheckpointStore(self.output_dir)
        planned = [
            (epoch, phase, task_id, task)
            for epoch in range(1, self.config["run"].get("epochs", 1) + 1)
            for phase, task_id, task in tasks
        ]
        self.records = []
        if self.store.method is not None:
            self.records = self.store.records
            expected = [(e, t) for e, _, t, _ in planned]
            actual = [(r["epoch"], r["task_id"]) for r in self.records]
            if actual != expected[: len(actual)] or len(actual) > len(expected):
                raise RuntimeError("Checkpoint is not the exact ordered task prefix")
            if any(
                r["status"] not in {"pass", "fail", "timeout"} for r in self.records
            ):
                raise RuntimeError("Checkpoint contains invalid evaluator rewards")
            atomic_json(self.method.state_path, self.store.method)
            self.method = create_method(
                self.method_name,
                self.output_dir,
                self.config.get("method_parameters", {}),
                self._generate_auxiliary,
            )
            if self.method.step != len(self.records):
                raise RuntimeError("Checkpoint state/result step mismatch")
        self.guard.recover(
            ["{}:{}".format(r["epoch"], r["task_id"]) for r in self.records]
        )
        self._rewrite_results()
        if self.records and not self.config["run"].get("resume", True):
            raise RuntimeError(
                "Existing run cannot be restarted; use a new output directory"
            )
        for generator in (self.generator, self.auxiliary_generator):
            generator.journal_path = self.output_dir / "request_events.jsonl"
        atomic_json(
            self.output_dir / "resolved_config.json", redacted_config(self.config)
        )
        atomic_json(self.output_dir / "task_order.json", manifest)
        processed = 0
        try:
            for epoch, phase, task_id, task in planned[len(self.records) :]:
                if limit is not None and processed >= limit:
                    break
                self.run_task(epoch, phase, task_id, task)
                processed += 1
        finally:
            self.evaluator.close()
        summary = build_results_summary(self.records, self.method_name, manifest)
        summary["method_state"] = self.method.summary()
        summary["completed_tasks"] = len(self.records)
        summary["planned_tasks"] = len(planned)
        summary["is_complete"] = len(self.records) == len(planned)
        summary["protocol"] = "appendix-paper-v2"
        atomic_json(self.output_dir / "results_summary.json", summary)

    def _rewrite_results(self):
        self.store.export_jsonl(self.results_path)

    def run_task(
        self, epoch: int, phase: str, task_id: str, task: dict[str, Any]
    ) -> None:
        key = "{}:{}".format(epoch, task_id)
        self.guard.start(key)
        for generator in (self.generator, self.auxiliary_generator):
            generator.active_task_key = key
        task_prompt = task[f"{self.split}_prompt"]
        retrieved = self.method.retrieve(task_prompt)
        memory_context = self.method.context(retrieved)
        system_context = build_system_prompt(
            self.config.get("prompt", {}).get(
                "system_prompt", DEFAULT_BCB_SYSTEM_PROMPT
            ),
            memory_context,
        )
        user_prompt = build_user_prompt(
            task_prompt, self.split, self.config["prompt"]["instruction_prefix"]
        )
        raw_code = self.generator.generate(user_prompt, system_context)
        solution = sanitize_generated_code(raw_code, task["entry_point"])
        status, error = self.evaluator.evaluate(task_id, task, solution)
        if status not in {"pass", "fail", "timeout"}:
            raise RuntimeError(
                "Evaluator did not produce a valid external reward: " + status
            )
        outcome = TaskOutcome(
            epoch, phase, task_id, task_prompt, solution, status, error
        )
        record = {
            "method": self.method_name,
            "epoch": epoch,
            "phase": phase,
            "task_id": task_id,
            "retrieved": [
                {
                    "memory_id": i.memory_id,
                    "score": i.score,
                    "content_chars": len(i.content),
                }
                for i in retrieved
            ],
            "memory_context_chars": len(memory_context),
            "system_context_chars": len(system_context),
            "raw_code": raw_code,
            "generation_metadata": dict(self.generator.last_generation_metadata),
            "auxiliary_generation_metadata": [],
            "solution": solution,
            "status": status,
            "error": error,
        }
        self._apply_with_usage(outcome, record)
        # One atomic authoritative generation contains both the entire learner
        # state and its evaluated result prefix. No auxiliary call is replayed.
        payload = {
            "method": self.method_name,
            "step": self.method.step,
            "state": self.method.state,
        }
        self.store.commit(payload, record)
        self.records = self.store.records
        self.method.save()
        self._rewrite_results()
        self.pending_path.unlink(missing_ok=True)
        self.guard.finish(key)
        print(
            f"method={self.method_name} epoch={epoch} phase={phase} task={task_id} status={status} step={self.method.step}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override solver model.max_tokens/max_completion_tokens",
    )
    parser.add_argument(
        "--auxiliary-thinking-mode",
        choices=("enabled", "disabled"),
        help="Override thinking only for auxiliary memory-building calls",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("enabled", "disabled"),
        help="Explicitly enable or disable model thinking for API calls",
    )
    parser.add_argument(
        "--transient-max-retries",
        type=int,
        help="Bounded retries for explicitly rejected rate-limit requests (0..10)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_method_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    if args.base_url:
        config["model"]["base_url"] = args.base_url
    if args.max_tokens is not None:
        if args.max_tokens < 1:
            raise ValueError("--max-tokens must be positive")
        config["model"]["max_tokens"] = args.max_tokens
    if args.auxiliary_thinking_mode:
        config.setdefault("auxiliary", {})[
            "thinking_mode"
        ] = args.auxiliary_thinking_mode
    if args.thinking_mode:
        config["model"]["thinking_mode"] = args.thinking_mode
    if args.transient_max_retries is not None:
        config["model"]["transient_max_retries"] = args.transient_max_retries
    if args.output_dir:
        config["run"]["output_dir"] = str(args.output_dir)
    MethodExperimentRunner(config).run(args.limit)


if __name__ == "__main__":
    main()
