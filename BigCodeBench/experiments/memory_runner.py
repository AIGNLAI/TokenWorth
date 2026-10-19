"""Sequential no-memory and retrieval-augmented BigCodeBench experiments.

The runner deliberately evaluates one task at a time.  This makes the result of
task N available to the memory pool before task N+1 is generated.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import ipaddress
import math
import os
import re
import select
import tempfile
import time
import subprocess
import textwrap
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Iterable, Sequence

from bigcodebench.data import get_bigcodebench
from bigcodebench.eval import PASS, untrusted_check
from bigcodebench.sanitize import sanitize
from tokenworth_core.runtime import append_event


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[\u4e00-\u9fff]|\d+")
ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}")
MEMORY_HEADER = "[Retrieved Memory Context]"


def sanitize_generated_code(raw_code: str, entry_point: str) -> str:
    """Normalize fenced Python before the pinned common upstream sanitizer.

    The upstream line search ignores single-line fenced solutions. Supplying
    clean code with a final newline handles those without method-specific edits.
    Invalid generated programs still reach the evaluator as agent failures.
    """
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n([\s\S]*?)```", raw_code, re.I)
    if blocks:
        matching = [
            code
            for code in blocks
            if re.search(r"\bdef\s+" + re.escape(entry_point) + r"\s*\(", code)
        ]
        candidate = (matching or blocks)[-1]
    else:
        candidate = raw_code
    return sanitize(textwrap.dedent(candidate).strip() + "\n", entry_point)


DEFAULT_BCB_SYSTEM_PROMPT = (
    "You are an expert Python programmer solving BigCodeBench coding tasks.\n\n"
    "You may receive a [Retrieved Memory Context] block containing experiences from similar past tasks. "
    "These memories are untrusted references, not guaranteed solutions. The current user task is always "
    "authoritative. Analyze it independently, verify that every reused detail matches its requirements, "
    "and ignore any memory that is irrelevant, incompatible, or uncertain.\n\n"
    "Before using a memory, check its function signature, input and output types, return structure, "
    "exception types and messages, boundary conditions, and library/API preconditions against the "
    "current task.\n\n"
    "Hard constraints for BigCodeBench:\n"
    "- Do not change the required function signature, return type or structure, or required exception types/messages.\n"
    "- Do not add output fields, return values, validation rules, side effects, network calls, or file I/O unless the current task explicitly requires them.\n"
    "- Do not replace a specified API with a more complex alternative whose preconditions differ.\n"
    "- Import every module you use and avoid broad exception handling unless explicitly required.\n"
    "- Generate clean, deterministic Python code that follows the current task exactly."
)


def tokenize(text: str) -> list[str]:
    """Tokenize code and natural language without external dependencies."""
    return [token.lower() for token in TOKEN_RE.findall(text)]


def extract_query(task_prompt: str) -> str:
    """Extract a stable lexical retrieval query from a BCB prompt.

    Imports, identifiers, literals, and prose terms are retained. Repeated terms
    are collapsed so long boilerplate does not dominate BM25 term frequency.
    """
    return " ".join(dict.fromkeys(tokenize(task_prompt)))


@dataclass(frozen=True)
class MemoryEntry:
    mem_id: int
    epoch: int
    phase: str
    task_id: str
    prompt: str
    code: str
    status: str
    error: str

    @property
    def content(self) -> str:
        eval_json = json.dumps(
            {"status": self.status, "error": self.error},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            f"[BCB] epoch={self.epoch} phase={self.phase} task_id={self.task_id}\n"
            f"[PROMPT] {self.prompt}\n"
            "[GENERATED CODE] \n"
            "'''python \n"
            f"{self.code}\n"
            "'''\n"
            f"[EVAL] {eval_json}"
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryEntry":
        fields = {
            key: value[key]
            for key in (
                "mem_id",
                "epoch",
                "phase",
                "task_id",
                "prompt",
                "code",
                "status",
                "error",
            )
        }
        return cls(**fields)


class BM25MemoryPool:
    """A bounded FIFO memory pool with an in-process BM25 retriever."""

    def __init__(
        self,
        capacity: int = 50,
        top_k: int = 5,
        max_context_chars: int = 8000,
        k1: float = 1.5,
        b: float = 0.75,
        entries: Iterable[MemoryEntry] = (),
    ) -> None:
        if capacity < 1 or top_k < 1 or max_context_chars < len(MEMORY_HEADER):
            raise ValueError("Invalid memory pool limits")
        self.capacity = capacity
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.k1 = k1
        self.b = b
        self._entries: deque[MemoryEntry] = deque(entries, maxlen=capacity)

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    def append(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def retrieve(self, query: str) -> list[tuple[MemoryEntry, float]]:
        if not self._entries:
            return []
        docs = [tokenize(entry.content) for entry in self._entries]
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        avgdl = sum(map(len, docs)) / len(docs) or 1.0
        doc_freq = Counter(
            term for doc in docs for term in set(doc) if term in query_terms
        )
        scored: list[tuple[MemoryEntry, float]] = []
        n_docs = len(docs)
        for entry, doc in zip(self._entries, docs):
            frequencies = Counter(doc)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                idf = math.log(
                    1.0 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5)
                )
                denom = frequency + self.k1 * (1.0 - self.b + self.b * len(doc) / avgdl)
                score += idf * frequency * (self.k1 + 1.0) / denom
            if score > 0:
                scored.append((entry, score))
        scored.sort(key=lambda item: (-item[1], -item[0].mem_id))
        return scored[: self.top_k]

    def format_context(self, retrieved: Sequence[tuple[MemoryEntry, float]]) -> str:
        if not retrieved:
            return ""
        context = MEMORY_HEADER
        for index, (entry, score) in enumerate(retrieved, start=1):
            block = (
                f"\n### Memory {index} (id={entry.mem_id}, sim={score:.6f})\n"
                f"{entry.content}"
            )
            remaining = self.max_context_chars - len(context)
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining]
            context += block
            if len(block) < remaining:
                continue
            break
        return context

    def save(self, path: Path) -> None:
        payload = {
            "capacity": self.capacity,
            "next_mem_id": self.next_mem_id,
            "entries": [
                {**asdict(entry), "content": entry.content} for entry in self._entries
            ],
        }
        atomic_write_json(path, payload)

    @property
    def next_mem_id(self) -> int:
        return max((entry.mem_id for entry in self._entries), default=0) + 1

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        capacity: int,
        top_k: int,
        max_context_chars: int,
        k1: float,
        b: float,
    ) -> "BM25MemoryPool":
        entries: list[MemoryEntry] = []
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = [MemoryEntry.from_dict(item) for item in payload["entries"]]
        return cls(capacity, top_k, max_context_chars, k1, b, entries)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return expand_environment(value)


def load_task_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Load a BigCodeBench JSONL file while preserving its task order."""
    tasks: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                task = json.loads(line)
                tasks[task["task_id"]] = task
    return tasks


def load_ordering_datasets() -> (
    tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]
):
    """Load Full and Hard without letting the Full override shadow Hard.

    BigCodeBench's ``BIGCODEBENCH_OVERRIDE_PATH`` is global: when it is set,
    ``get_bigcodebench(subset="hard")`` incorrectly returns that same override.
    Prefer the corresponding cached Hard file (or an explicit Hard override) in
    that case.
    """
    full_override = os.getenv("BIGCODEBENCH_OVERRIDE_PATH")
    if not full_override:
        return get_bigcodebench(subset="full"), get_bigcodebench(subset="hard")

    full_path = Path(full_override)
    hard_override = os.getenv("BIGCODEBENCH_HARD_OVERRIDE_PATH")
    if hard_override:
        hard_path = Path(hard_override)
    else:
        hard_path = full_path.with_name(
            full_path.name.replace("BigCodeBench-", "BigCodeBench-Hard-", 1)
        )
    if not hard_path.is_file():
        raise FileNotFoundError(
            "Hard subset not found; set BIGCODEBENCH_HARD_OVERRIDE_PATH to its JSONL path: "
            f"{hard_path}"
        )
    return load_task_jsonl(full_path), load_task_jsonl(hard_path)


def expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, str):
        return ENV_RE.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value


def ordered_tasks(
    full: dict[str, dict[str, Any]], hard: dict[str, dict[str, Any]]
) -> list[tuple[str, str, dict[str, Any]]]:
    """Return Full - Hard first, followed by the Hard subset."""
    hard_ids = set(hard)
    simple = [
        ("simple", task_id, task)
        for task_id, task in full.items()
        if task_id not in hard_ids
    ]
    difficult = [
        ("hard", task_id, full[task_id]) for task_id in hard if task_id in full
    ]
    if len(simple) + len(difficult) != len(full):
        raise ValueError("Hard task IDs are inconsistent with the full dataset")
    return simple + difficult


def task_order_manifest(
    tasks: Sequence[tuple[str, str, dict[str, Any]]]
) -> dict[str, Any]:
    order = [
        {"index": index, "phase": phase, "task_id": task_id}
        for index, (phase, task_id, _) in enumerate(tasks)
    ]
    canonical = json.dumps(order, ensure_ascii=False, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "total": len(order),
        "simple": sum(item["phase"] == "simple" for item in order),
        "hard": sum(item["phase"] == "hard" for item in order),
        "order": order,
    }


def build_results_summary(
    records: Sequence[dict[str, Any]],
    mode: str,
    manifest: dict[str, Any],
    pool: BM25MemoryPool | None = None,
) -> dict[str, Any]:
    """Build a compact, machine-readable Pass@1 summary."""
    status_counts = Counter(record["status"] for record in records)
    phases: dict[str, dict[str, Any]] = {}
    for phase in ("simple", "hard"):
        phase_records = [record for record in records if record.get("phase") == phase]
        phase_counts = Counter(record["status"] for record in phase_records)
        phase_total = len(phase_records)
        phases[phase] = {
            "total": phase_total,
            "status_counts": dict(phase_counts),
            "pass_at_1": (
                phase_counts.get(PASS, 0) / phase_total if phase_total else 0.0
            ),
        }
    total = len(records)
    summary: dict[str, Any] = {
        "mode": mode,
        "complete": total == manifest["total"],
        "total": total,
        "expected_total": manifest["total"],
        "status_counts": dict(status_counts),
        "pass_at_1": status_counts.get(PASS, 0) / total if total else 0.0,
        "phases": phases,
        "task_order_sha256": manifest["sha256"],
    }
    if mode == "rag" and pool is not None:
        summary["memory"] = {
            "capacity": pool.capacity,
            "retained_entries": len(pool.entries),
            "successful_entries_created": pool.next_mem_id - 1,
            "next_mem_id": pool.next_mem_id,
        }
    return summary


def build_user_prompt(task_prompt: str, split: str, instruction_prefix: str) -> str:
    if split == "complete":
        return f"{instruction_prefix}\n```\n{task_prompt.strip()}\n```"
    return f"{instruction_prefix}\n{task_prompt.strip()}"


def build_system_prompt(base_prompt: str, memory_context: str = "") -> str:
    """Compose a stable BCB system prompt with optional retrieved memory.

    Keeping the base prompt present for both memory and no-memory runs prevents
    the mere presence of a system role from becoming an experiment confound.
    Memory is appended after the rules so that it remains subordinate context.
    """
    parts = [part.strip() for part in (base_prompt, memory_context) if part.strip()]
    return "\n\n".join(parts)


class OpenAICompatibleGenerator:
    def __init__(self, config: dict[str, Any]) -> None:
        from openai import OpenAI

        api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"), "none")
        base_url = config.get("base_url") or os.getenv("OPENAI_BASE_URL")
        hostname = urlparse(base_url or "").hostname or ""
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = hostname.lower() == "localhost"
        client_options = {}
        if loopback:
            # A workstation's global proxy must not route its local model
            # endpoint to an external proxy and turn it into a 502 response.
            import httpx

            client_options["http_client"] = httpx.Client(trust_env=False)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=config.get("timeout_seconds", 180),
            max_retries=0,
            **client_options,
        )
        budget = config.get("max_tokens", 1280)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("model.max_tokens must be a positive integer")
        self.config = config
        self.resolved_base_url = str(
            getattr(self.client, "base_url", config.get("base_url"))
        )
        self.last_generation_metadata: dict[str, Any] = {}
        self.journal_path = None
        self.active_task_key = None
        self.transient_max_retries = int(config.get("transient_max_retries", 3))
        if not 0 <= self.transient_max_retries <= 10:
            raise ValueError("model.transient_max_retries must be bounded in [0,10]")
        self.thinking_mode = config.get("thinking_mode") or os.getenv(
            "BCB_LLM_THINKING"
        )
        if self.thinking_mode not in {None, "enabled", "disabled"}:
            raise ValueError(
                "model.thinking_mode/BCB_LLM_THINKING must be 'enabled', "
                f"'disabled', or unset; found {self.thinking_mode!r}"
            )

    def generate(self, user_prompt: str, system_context: str) -> str:
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        messages.append({"role": "user", "content": user_prompt})
        kwargs: dict[str, Any] = {
            "model": self.config["name"],
            "messages": messages,
            "temperature": self.config.get("temperature", 0.0),
            "n": 1,
        }
        max_tokens = self.config.get("max_tokens", 1280)
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        if self.thinking_mode is not None:
            kwargs["extra_body"] = {"thinking": {"type": self.thinking_mode}}
        transient_retries = 0
        while True:
            append_event(
                self.journal_path,
                {
                    "event": "request_start",
                    "task": self.active_task_key,
                    "attempt": transient_retries + 1,
                },
            )
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as error:
                import openai

                append_event(
                    self.journal_path,
                    {
                        "event": "request_error",
                        "task": self.active_task_key,
                        "error_type": type(error).__name__,
                        "status": getattr(error, "status_code", None),
                    },
                )
                # A transport timeout or 5xx may hide an accepted, charged
                # request. Only explicit rejected rate limits may be retried.
                retryable = isinstance(error, openai.RateLimitError)
                if not retryable:
                    raise
                transient_retries += 1
                if transient_retries > self.transient_max_retries:
                    raise
                delay = min(2 ** min(transient_retries - 1, 2), 4)
                limit = str(self.transient_max_retries)
                print(
                    "Transient model API error "
                    f"({type(error).__name__}); retry={transient_retries}/{limit} "
                    f"in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        usage = getattr(response, "usage", None)
        reported = {
            key: getattr(usage, key, None)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        append_event(
            self.journal_path,
            {
                "event": "request_response",
                "task": self.active_task_key,
                "usage": reported,
            },
        )
        for key in ("prompt_tokens", "completion_tokens"):
            value = reported[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError("Missing/invalid provider usage: " + key)
        if (
            reported["completion_tokens"] < 1
            or reported["completion_tokens"] > max_tokens
        ):
            raise RuntimeError(
                "Provider output token count violates the request budget"
            )
        if (
            not response.choices
            or not isinstance(response.choices[0].message.content, str)
            or not response.choices[0].message.content.strip()
        ):
            raise RuntimeError("Model returned an empty or malformed response")
        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        completion_details = (
            getattr(usage, "completion_tokens_details", None) if usage else None
        )
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning", None)
        if reasoning is None:
            model_extra = getattr(message, "model_extra", None) or {}
            reasoning = model_extra.get("reasoning_content") or model_extra.get(
                "reasoning"
            )
        self.last_generation_metadata = {
            "requested_max_completion_tokens": max_tokens,
            "finish_reason": getattr(choice, "finish_reason", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
            "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else None,
            "visible_output_chars": len(message.content or ""),
        }
        return message.content or ""


class TaskEvaluator:
    def __init__(self, config: dict[str, Any], split: str) -> None:
        self.config = config
        self.split = split
        self._gradio_client = None
        self._docker_worker: subprocess.Popen[str] | None = None
        self._docker_worker_tasks = 0

    def evaluate(
        self, task_id: str, task: dict[str, Any], solution: str
    ) -> tuple[str, str]:
        backend = self.config.get("backend", "gradio")
        if backend in {"deferred_gradio", "deferred_docker"}:
            return "pending", ""
        if backend == "docker":
            return self._evaluate_docker(task_id, solution)
        if backend == "local":
            return self._evaluate_local(task, solution)
        if backend == "gradio":
            return self._evaluate_gradio(task_id, solution)
        raise ValueError(f"Unsupported evaluator backend: {backend}")

    def _start_docker_worker(self) -> subprocess.Popen[str]:
        if self._docker_worker is not None:
            return self._docker_worker
        dataset_path = Path(self.config["dataset_path"]).expanduser().resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"BigCodeBench dataset file not found: {dataset_path}"
            )
        container_dataset_path = "/data/BigCodeBench.jsonl"
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--read-only",
            "--pids-limit",
            str(self.config.get("pids_limit", 256)),
            "--memory",
            self.config.get("container_memory", "64g"),
            "--cpus",
            str(self.config.get("container_cpus", 8)),
            "--tmpfs",
            "/tmp:rw,exec,nosuid,size=4g",
            "--tmpfs",
            "/home/evaluator:rw,nosuid,size=256m",
            "-v",
            f"{dataset_path}:{container_dataset_path}:ro",
            "-e",
            f"BIGCODEBENCH_OVERRIDE_PATH={container_dataset_path}",
        ]
        for name, value in self.config.get("container_environment", {}).items():
            command.extend(["-e", f"{name}={value}"])
        command.append(self.config.get("image", "bigcodebench-evaluator-local:latest"))
        self._docker_worker = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        assert self._docker_worker.stdout is not None
        ready = json.loads(self._docker_worker.stdout.readline())
        if not ready.get("ready") or ready.get("tasks") != 1140:
            raise RuntimeError(f"Docker evaluator failed to initialize: {ready}")
        self._docker_worker_tasks = 0
        atexit.register(self.close)
        return self._docker_worker

    def _evaluate_docker(self, task_id: str, solution: str) -> tuple[str, str]:
        max_tasks = int(self.config.get("docker_worker_max_tasks", 50))
        if max_tasks > 0 and self._docker_worker_tasks >= max_tasks:
            self.close()
        worker = self._start_docker_worker()
        assert worker.stdin is not None and worker.stdout is not None
        request = {
            "task_id": task_id,
            "solution": solution,
            "min_time_limit": self.config.get("min_time_limit", 1),
            "gt_time_limit": self.config.get("gt_time_limit", 20),
            "max_as_limit": self.config.get("max_as_limit", 30 * 1024),
            "max_data_limit": self.config.get("max_data_limit", 30 * 1024),
            "max_stack_limit": self.config.get("max_stack_limit", 10),
        }
        worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        worker.stdin.flush()
        response_timeout = float(self.config.get("docker_response_timeout_seconds", 60))
        # A submitted program can leave a child process holding the worker's
        # stdout and emit protocol-shaped noise despite the worker redirect.
        # Ignore such lines until the worker's actual status object arrives.
        deadline = time.monotonic() + max(0.0, response_timeout)
        ignored_lines: list[str] = []
        while True:
            readable, _, _ = select.select(
                [worker.stdout], [], [], max(0.0, deadline - time.monotonic())
            )
            if not readable:
                self.close()
                raise RuntimeError(
                    "Docker evaluator response timeout; no valid task reward"
                )
            response_line = worker.stdout.readline()
            if not response_line:
                self.close()
                raise RuntimeError("Docker evaluator exited without returning a result")
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError:
                ignored_lines.append(response_line.rstrip()[:500])
                continue
            if isinstance(response, dict) and "status" in response:
                break
            ignored_lines.append(response_line.rstrip()[:500])
        self._docker_worker_tasks += 1
        details = response.get("details") or {}
        error = json.dumps(details, ensure_ascii=False) if details else ""
        return response["status"], error

    def close(self) -> None:
        if self._docker_worker is None:
            return
        if self._docker_worker.stdin:
            self._docker_worker.stdin.close()
        try:
            self._docker_worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._docker_worker.terminate()
        self._docker_worker = None
        self._docker_worker_tasks = 0

    def _evaluate_local(self, task: dict[str, Any], solution: str) -> tuple[str, str]:
        calibrated = task["code_prompt"] + "\n    pass\n" + solution
        status, details = untrusted_check(
            calibrated,
            task["test"],
            task["entry_point"],
            self.config.get("max_as_limit", 30 * 1024),
            self.config.get("max_data_limit", 30 * 1024),
            self.config.get("max_stack_limit", 10),
            self.config.get("min_time_limit", 1),
            self.config.get("gt_time_limit", 20),
        )
        error = json.dumps(details, ensure_ascii=False) if details else ""
        return status or "timeout", error

    def _evaluate_gradio(self, task_id: str, solution: str) -> tuple[str, str]:
        from gradio_client import Client, handle_file

        if self._gradio_client is None:
            self._gradio_client = Client(self.config["endpoint"])
        # The hosted evaluator accepts numeric suffixes and adds
        # ``BigCodeBench/`` internally.
        selective_id = task_id.split("/", 1)[-1]
        sample_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", encoding="utf-8", delete=False
            ) as sample:
                sample.write(
                    json.dumps({"task_id": task_id, "solution": solution}) + "\n"
                )
                sample_path = sample.name
            attempts = self.config.get("request_attempts", 3)
            for attempt in range(1, attempts + 1):
                try:
                    results, _ = self._gradio_client.predict(
                        split=self.split,
                        subset="full",
                        samples=handle_file(sample_path),
                        pass_k="1",
                        parallel=1,
                        min_time_limit=self.config.get("min_time_limit", 1),
                        max_as_limit=self.config.get("max_as_limit", 30 * 1024),
                        max_data_limit=self.config.get("max_data_limit", 30 * 1024),
                        max_stack_limit=self.config.get("max_stack_limit", 10),
                        calibrated=True,
                        check_gt_only=False,
                        no_gt=self.config.get("no_gt", False),
                        selective_evaluate=selective_id,
                        api_name="/predict",
                    )
                    item = results["eval"][task_id][0]
                    details = item.get("details") or {}
                    return item["status"], json.dumps(details, ensure_ascii=False)
                except Exception as error:  # network service errors are retryable here
                    if attempt == attempts:
                        return "eval_error", str(error)
                    time.sleep(min(2**attempt, 8))
        finally:
            if sample_path:
                Path(sample_path).unlink(missing_ok=True)
        return "eval_error", "remote evaluator returned no result"


class ExperimentRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.mode = config["mode"]
        if self.mode not in {"no_memory", "rag"}:
            raise ValueError("mode must be no_memory or rag")
        self.split = config.get("dataset", {}).get("split", "instruct")
        self.output_dir = Path(config["run"]["output_dir"])
        self.results_path = self.output_dir / "results.jsonl"
        self.memory_path = self.output_dir / "memory_pool.json"
        rag = config.get("rag", {})
        self.pool = BM25MemoryPool.load(
            self.memory_path,
            capacity=rag.get("pool_size", 50),
            top_k=rag.get("top_k", 5),
            max_context_chars=rag.get("max_context_chars", 8000),
            k1=rag.get("bm25_k1", 1.5),
            b=rag.get("bm25_b", 0.75),
        )
        self.generator = OpenAICompatibleGenerator(config["model"])
        self.evaluator = TaskEvaluator(config["evaluation"], self.split)

    def completed(self) -> set[tuple[int, str]]:
        completed: set[tuple[int, str]] = set()
        if self.config["run"].get("resume", True) and self.results_path.exists():
            for line in self.results_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                completed.add((item["epoch"], item["task_id"]))
        return completed

    def run(self, limit: int | None = None) -> None:
        full, hard = load_ordering_datasets()
        tasks = ordered_tasks(full, hard)
        completed = self.completed()
        processed = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.output_dir / "resolved_config.json", self.config)
        atomic_write_json(
            self.output_dir / "task_order.json", task_order_manifest(tasks)
        )
        limit_reached = False
        for epoch in range(1, self.config["run"].get("epochs", 1) + 1):
            for phase, task_id, task in tasks:
                if (epoch, task_id) in completed:
                    continue
                if limit is not None and processed >= limit:
                    limit_reached = True
                    break
                self.run_task(epoch, phase, task_id, task)
                processed += 1
            if limit_reached:
                break
        backend = self.config["evaluation"].get("backend")
        if backend == "deferred_gradio":
            self.finalize_deferred_gradio(tasks)
        elif backend == "deferred_docker":
            self.finalize_deferred_docker(tasks)
        records = [
            json.loads(line)
            for line in self.results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = task_order_manifest(tasks)
        atomic_write_json(
            self.output_dir / "results_summary.json",
            build_results_summary(records, self.mode, manifest, self.pool),
        )

    def finalize_deferred_docker(
        self, tasks: Sequence[tuple[str, str, dict[str, Any]]]
    ) -> None:
        """Evaluate generated no-memory tasks locally with resumable progress."""
        if self.config["run"].get("epochs", 1) != 1:
            raise ValueError("deferred_docker currently requires epochs=1")
        records = [
            json.loads(line)
            for line in self.results_path.read_text(encoding="utf-8").splitlines()
        ]
        latest = {record["task_id"]: record for record in records}
        expected_ids = [task_id for _, task_id, _ in tasks if task_id in latest]
        if not expected_ids:
            return

        checkpoint_path = self.output_dir / "local_eval_results.jsonl"
        evaluations: dict[str, dict[str, str]] = {}
        if checkpoint_path.exists():
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                evaluations[item["task_id"]] = item

        try:
            for task_id in expected_ids:
                if task_id in evaluations:
                    continue
                status, error = self.evaluator._evaluate_docker(
                    task_id, latest[task_id]["solution"]
                )
                item = {"task_id": task_id, "status": status, "error": error}
                append_jsonl(checkpoint_path, item)
                evaluations[task_id] = item
                print(
                    f"local_eval={len(evaluations)}/{len(expected_ids)} "
                    f"task={task_id} status={status}",
                    flush=True,
                )
        finally:
            self.evaluator.close()

        for task_id in expected_ids:
            record = latest[task_id]
            record["status"] = evaluations[task_id]["status"]
            record["error"] = evaluations[task_id]["error"]
        atomic_write_jsonl(
            self.results_path, (latest[task_id] for task_id in expected_ids)
        )
        atomic_write_json(
            self.output_dir / "local_eval_summary.json",
            {
                "total": len(expected_ids),
                "statuses": dict(
                    Counter(evaluations[task_id]["status"] for task_id in expected_ids)
                ),
            },
        )

    def finalize_deferred_gradio(
        self, tasks: Sequence[tuple[str, str, dict[str, Any]]]
    ) -> None:
        """Evaluate a complete no-memory epoch in one hosted API call."""
        if self.config["run"].get("epochs", 1) != 1:
            raise ValueError("deferred_gradio currently requires epochs=1")
        records = [
            json.loads(line)
            for line in self.results_path.read_text(encoding="utf-8").splitlines()
        ]
        latest = {record["task_id"]: record for record in records}
        expected_ids = [task_id for _, task_id, _ in tasks]
        if any(task_id not in latest for task_id in expected_ids):
            return
        samples_path = self.output_dir / "samples.jsonl"
        atomic_write_jsonl(
            samples_path,
            (
                {"task_id": task_id, "solution": latest[task_id]["solution"]}
                for task_id in expected_ids
            ),
        )
        from gradio_client import Client, handle_file

        evaluation = self.config["evaluation"]
        client = Client(evaluation["endpoint"])
        attempts = evaluation.get("request_attempts", 3)
        for attempt in range(1, attempts + 1):
            try:
                results, pass_at_k = client.predict(
                    split=self.split,
                    subset="full",
                    samples=handle_file(str(samples_path)),
                    pass_k="1",
                    parallel=evaluation.get("parallel", -1),
                    min_time_limit=evaluation.get("min_time_limit", 1),
                    max_as_limit=evaluation.get("max_as_limit", 30 * 1024),
                    max_data_limit=evaluation.get("max_data_limit", 30 * 1024),
                    max_stack_limit=evaluation.get("max_stack_limit", 10),
                    calibrated=True,
                    check_gt_only=False,
                    no_gt=evaluation.get("no_gt", False),
                    selective_evaluate="",
                    api_name="/predict",
                )
                break
            except Exception:
                if attempt == attempts:
                    raise
                time.sleep(min(2**attempt, 8))
        atomic_write_json(self.output_dir / "eval_results.json", results)
        atomic_write_json(self.output_dir / "pass_at_k.json", pass_at_k)
        for task_id in expected_ids:
            item = results["eval"][task_id][0]
            record = latest[task_id]
            record["status"] = item["status"]
            details = item.get("details") or {}
            record["error"] = json.dumps(details, ensure_ascii=False) if details else ""
        atomic_write_jsonl(
            self.results_path, (latest[task_id] for task_id in expected_ids)
        )

    def run_task(
        self, epoch: int, phase: str, task_id: str, task: dict[str, Any]
    ) -> None:
        task_prompt = task[f"{self.split}_prompt"]
        query = extract_query(task_prompt)
        retrieved = self.pool.retrieve(query) if self.mode == "rag" else []
        memory_context = self.pool.format_context(retrieved) if retrieved else ""
        system_context = build_system_prompt(
            self.config.get("prompt", {}).get(
                "system_prompt", DEFAULT_BCB_SYSTEM_PROMPT
            ),
            memory_context,
        )
        user_prompt = build_user_prompt(
            task_prompt,
            self.split,
            self.config["prompt"]["instruction_prefix"],
        )
        raw_code = self.generator.generate(user_prompt, system_context)
        model_code = sanitize_generated_code(raw_code, task["entry_point"])
        status, error = self.evaluator.evaluate(task_id, task, model_code)
        record = {
            "epoch": epoch,
            "phase": phase,
            "task_id": task_id,
            "query": query,
            "retrieved": [
                {"mem_id": entry.mem_id, "similarity": score}
                for entry, score in retrieved
            ],
            "memory_context_chars": len(memory_context),
            "system_context_chars": len(system_context),
            "raw_code": raw_code,
            "generation_metadata": dict(
                getattr(self.generator, "last_generation_metadata", {})
            ),
            "solution": model_code,
            "status": status,
            "error": error,
        }
        append_jsonl(self.results_path, record)
        if self.mode == "rag" and status == PASS:
            self.pool.append(
                MemoryEntry(
                    mem_id=self.pool.next_mem_id,
                    epoch=epoch,
                    phase=phase,
                    task_id=task_id,
                    prompt=task_prompt,
                    code=model_code,
                    status=status,
                    error=error,
                )
            )
            self.pool.save(self.memory_path)
        print(
            f"epoch={epoch} phase={phase} task={task_id} status={status} "
            f"memories={len(self.pool.entries)} retrieved={len(retrieved)}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", help="Override model.name from the config")
    parser.add_argument("--base-url", help="Override model.base_url from the config")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override model.max_tokens/max_completion_tokens",
    )
    parser.add_argument(
        "--evaluation-backend",
        choices=("docker", "local", "gradio", "deferred_docker", "deferred_gradio"),
        help="Override evaluation.backend",
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
    parser.add_argument("--output-dir", type=Path, help="Override run.output_dir")
    parser.add_argument("--limit", type=int, help="Only process N pending tasks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if args.model:
        config["model"]["name"] = args.model
    if args.base_url:
        config["model"]["base_url"] = args.base_url
    if args.max_tokens is not None:
        if args.max_tokens < 1:
            raise ValueError("--max-tokens must be positive")
        config["model"]["max_tokens"] = args.max_tokens
    if args.evaluation_backend:
        config["evaluation"]["backend"] = args.evaluation_backend
    if args.thinking_mode:
        config["model"]["thinking_mode"] = args.thinking_mode
    if args.transient_max_retries is not None:
        config["model"]["transient_max_retries"] = args.transient_max_retries
    if args.output_dir:
        config["run"]["output_dir"] = str(args.output_dir)
    if not config["model"].get("name"):
        raise ValueError("Set BCB_MODEL or pass --model")
    ExperimentRunner(config).run(limit=args.limit)


if __name__ == "__main__":
    main()
