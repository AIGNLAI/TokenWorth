from experiments.memory_runner import (
    BM25MemoryPool,
    DEFAULT_BCB_SYSTEM_PROMPT,
    ExperimentRunner,
    OpenAICompatibleGenerator,
    TaskEvaluator,
    build_results_summary,
    MEMORY_HEADER,
    MemoryEntry,
    build_system_prompt,
    expand_environment,
    extract_query,
    load_ordering_datasets,
    ordered_tasks,
    task_order_manifest,
)


def test_docker_evaluator_mounts_only_the_dataset_file(tmp_path, monkeypatch):
    from io import StringIO
    from types import SimpleNamespace

    dataset_path = tmp_path / "BigCodeBench-v0.1.4.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    commands = []

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(
            stdin=StringIO(),
            stdout=StringIO('{"ready": true, "tasks": 1140}\n'),
            wait=lambda timeout: 0,
            terminate=lambda: None,
        )

    monkeypatch.setattr(
        "experiments.memory_runner.subprocess.Popen",
        fake_popen,
    )
    evaluator = TaskEvaluator(
        {"dataset_path": str(dataset_path), "image": "test-evaluator"},
        "instruct",
    )

    evaluator._start_docker_worker()
    evaluator.close()

    command = commands[0]
    mount = command[command.index("-v") + 1]
    assert mount == f"{dataset_path.resolve()}:/data/BigCodeBench.jsonl:ro"
    assert str(dataset_path.parent) + ":/data:ro" not in command
    environment = command[command.index("-e") + 1]
    assert environment == "BIGCODEBENCH_OVERRIDE_PATH=/data/BigCodeBench.jsonl"


def test_openai_generator_can_explicitly_disable_thinking(monkeypatch):
    from types import SimpleNamespace

    import httpx
    import openai

    requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                raise openai.RateLimitError(
                    "rate limited",
                    response=httpx.Response(
                        429, request=httpx.Request("POST", "https://example.test/v1")
                    ),
                    body={},
                )
            message = SimpleNamespace(content="```python\ndef target(): return 1\n```")
            usage = SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=1200,
                total_tokens=1300,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=900),
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="length")],
                usage=usage,
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: fake_client)
    monkeypatch.setattr("experiments.memory_runner.time.sleep", lambda _delay: None)
    generator = OpenAICompatibleGenerator(
        {
            "name": "test-model",
            "thinking_mode": "disabled",
            "transient_max_retries": 1,
            "max_tokens": 1280,
        }
    )

    generator.generate("Solve the task", "Follow the contract")

    assert len(requests) == 2
    assert requests[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert requests[1]["max_completion_tokens"] == 1280
    assert generator.last_generation_metadata["finish_reason"] == "length"
    assert generator.last_generation_metadata["reasoning_tokens"] == 900


def entry(mem_id: int, prompt: str, status: str = "pass") -> MemoryEntry:
    return MemoryEntry(
        mem_id=mem_id,
        epoch=1,
        phase="simple",
        task_id=f"BigCodeBench/{mem_id}",
        prompt=prompt,
        code=f"def f(): return {mem_id}",
        status=status,
        error="",
    )


def test_memory_template_matches_protocol():
    content = entry(1, "Parse a JSON document").content
    assert content.startswith(
        "[BCB] epoch=1 phase=simple task_id=BigCodeBench/1\n"
        "[PROMPT] Parse a JSON document\n"
        "[GENERATED CODE] \n'''python \n"
    )
    assert content.endswith('[EVAL] {"status":"pass","error":""}')


def test_fifo_capacity_and_bm25_top_k():
    pool = BM25MemoryPool(capacity=3, top_k=2, max_context_chars=8000)
    pool.append(entry(1, "json parsing dictionary"))
    pool.append(entry(2, "image resize pixels"))
    pool.append(entry(3, "json serialization encoder"))
    pool.append(entry(4, "json load nested dictionary"))

    assert [item.mem_id for item in pool.entries] == [2, 3, 4]
    retrieved = pool.retrieve(extract_query("load and parse nested JSON"))
    assert len(retrieved) == 2
    assert {item.mem_id for item, _ in retrieved} == {3, 4}


def test_context_is_system_message_shaped_and_length_limited():
    pool = BM25MemoryPool(capacity=50, top_k=5, max_context_chars=180)
    pool.append(entry(1, "json " * 100))
    context = pool.format_context(pool.retrieve("json"))
    assert context.startswith(MEMORY_HEADER + "\n### Memory 1")
    assert len(context) == 180


def test_system_prompt_is_stable_and_memory_is_appended_after_rules():
    without_memory = build_system_prompt(DEFAULT_BCB_SYSTEM_PROMPT)
    with_memory = build_system_prompt(
        DEFAULT_BCB_SYSTEM_PROMPT,
        "[Retrieved Memory Context]\n### Memory 1",
    )

    assert without_memory == DEFAULT_BCB_SYSTEM_PROMPT.strip()
    assert with_memory.startswith(without_memory)
    assert with_memory.endswith("[Retrieved Memory Context]\n### Memory 1")
    assert "current user task is always authoritative" in with_memory
    assert "function signature" in with_memory
    assert "return type or structure" in with_memory
    assert "exception types/messages" in with_memory


def test_simple_tasks_are_before_hard_tasks():
    full = {f"BigCodeBench/{i}": {"value": i} for i in range(5)}
    hard = {"BigCodeBench/1": {}, "BigCodeBench/4": {}}
    tasks = ordered_tasks(full, hard)
    assert [(phase, task_id) for phase, task_id, _ in tasks] == [
        ("simple", "BigCodeBench/0"),
        ("simple", "BigCodeBench/2"),
        ("simple", "BigCodeBench/3"),
        ("hard", "BigCodeBench/1"),
        ("hard", "BigCodeBench/4"),
    ]
    first = task_order_manifest(tasks)
    second = task_order_manifest(ordered_tasks(full, hard))
    assert first["sha256"] == second["sha256"]
    assert (first["total"], first["simple"], first["hard"]) == (5, 3, 2)


def test_results_summary_includes_phases_and_rag_memory():
    records = [
        {"phase": "simple", "status": "pass"},
        {"phase": "simple", "status": "fail"},
        {"phase": "hard", "status": "timeout"},
    ]
    manifest = {"total": 3, "sha256": "order-hash"}
    pool = BM25MemoryPool(capacity=50, top_k=5, max_context_chars=8000)
    pool.append(entry(1, "json"))

    summary = build_results_summary(records, "rag", manifest, pool)

    assert summary["complete"] is True
    assert summary["status_counts"] == {"pass": 1, "fail": 1, "timeout": 1}
    assert summary["pass_at_1"] == 1 / 3
    assert summary["phases"]["simple"]["pass_at_1"] == 0.5
    assert summary["memory"] == {
        "capacity": 50,
        "retained_entries": 1,
        "successful_entries_created": 1,
        "next_mem_id": 2,
    }


def test_full_override_does_not_shadow_hard_subset(tmp_path, monkeypatch):
    full_path = tmp_path / "BigCodeBench-v0.1.4.jsonl"
    hard_path = tmp_path / "BigCodeBench-Hard-v0.1.4.jsonl"
    full_path.write_text(
        "\n".join('{"task_id":"BigCodeBench/%d"}' % index for index in range(3)) + "\n"
    )
    hard_path.write_text('{"task_id":"BigCodeBench/2"}\n')
    monkeypatch.setenv("BIGCODEBENCH_OVERRIDE_PATH", str(full_path))
    monkeypatch.delenv("BIGCODEBENCH_HARD_OVERRIDE_PATH", raising=False)

    full, hard = load_ordering_datasets()

    assert list(full) == ["BigCodeBench/0", "BigCodeBench/1", "BigCodeBench/2"]
    assert list(hard) == ["BigCodeBench/2"]


def test_unset_environment_variable_becomes_empty(monkeypatch):
    monkeypatch.delenv("MISSING_BCB_VALUE", raising=False)
    assert expand_environment("${MISSING_BCB_VALUE}") == ""


class FakeGenerator:
    def __init__(self):
        self.system_contexts = []

    def generate(self, user_prompt, system_context):
        self.system_contexts.append(system_context)
        return "```python\ndef target():\n    return 1\n```"


class FakeEvaluator:
    def __init__(self, statuses):
        self.statuses = iter(statuses)

    def evaluate(self, task_id, task, solution):
        return next(self.statuses), "test error"


class FakeDeferredEvaluator:
    def __init__(self):
        self.closed = False

    def _evaluate_docker(self, task_id, solution):
        return ("pass" if task_id.endswith("1") else "fail"), solution

    def close(self):
        self.closed = True


def test_only_successful_tasks_update_memory(tmp_path):
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.mode = "rag"
    runner.split = "instruct"
    runner.output_dir = tmp_path
    runner.results_path = tmp_path / "results.jsonl"
    runner.memory_path = tmp_path / "memory_pool.json"
    runner.config = {"prompt": {"instruction_prefix": "Solve:"}}
    runner.pool = BM25MemoryPool(capacity=50, top_k=5, max_context_chars=8000)
    runner.generator = FakeGenerator()
    runner.evaluator = FakeEvaluator(["fail", "pass"])
    task = {
        "instruct_prompt": "Return one from target.",
        "entry_point": "target",
    }

    runner.run_task(1, "simple", "BigCodeBench/1", task)
    assert runner.pool.entries == ()
    runner.run_task(1, "simple", "BigCodeBench/2", task)
    assert len(runner.pool.entries) == 1
    assert runner.pool.entries[0].task_id == "BigCodeBench/2"
    saved = __import__("json").loads(runner.memory_path.read_text())
    assert saved["entries"][0]["content"].startswith("[BCB] epoch=1")
    assert runner.generator.system_contexts[0] == DEFAULT_BCB_SYSTEM_PROMPT.strip()
    assert runner.generator.system_contexts[1].startswith(
        DEFAULT_BCB_SYSTEM_PROMPT.strip()
    )


def test_deferred_docker_evaluation_is_checkpointed_and_merged(tmp_path):
    import json

    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.config = {"run": {"epochs": 1}}
    runner.output_dir = tmp_path
    runner.results_path = tmp_path / "results.jsonl"
    runner.evaluator = FakeDeferredEvaluator()
    records = [
        {"task_id": "BigCodeBench/1", "solution": "one", "status": "pending"},
        {"task_id": "BigCodeBench/2", "solution": "two", "status": "pending"},
    ]
    runner.results_path.write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    tasks = [
        ("simple", "BigCodeBench/1", {}),
        ("hard", "BigCodeBench/2", {}),
    ]

    runner.finalize_deferred_docker(tasks)

    merged = [json.loads(line) for line in runner.results_path.read_text().splitlines()]
    assert [item["status"] for item in merged] == ["pass", "fail"]
    assert (tmp_path / "local_eval_results.jsonl").read_text().count("\n") == 2
    assert json.loads((tmp_path / "local_eval_summary.json").read_text()) == {
        "total": 2,
        "statuses": {"pass": 1, "fail": 1},
    }
    assert runner.evaluator.closed


def test_limit_still_finalizes_generated_deferred_docker_tasks(tmp_path, monkeypatch):
    import json

    tasks = [
        ("simple", "BigCodeBench/1", {}),
        ("simple", "BigCodeBench/2", {}),
    ]
    monkeypatch.setattr(
        "experiments.memory_runner.load_ordering_datasets", lambda: ({}, {})
    )
    monkeypatch.setattr(
        "experiments.memory_runner.ordered_tasks", lambda _full, _hard: tasks
    )

    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.config = {
        "run": {"epochs": 1, "resume": True},
        "evaluation": {"backend": "deferred_docker"},
    }
    runner.mode = "no_memory"
    runner.pool = None
    runner.output_dir = tmp_path
    runner.results_path = tmp_path / "results.jsonl"
    runner.evaluator = FakeDeferredEvaluator()
    runner.completed = lambda: set()

    def fake_run_task(epoch, phase, task_id, _task):
        record = {
            "epoch": epoch,
            "phase": phase,
            "task_id": task_id,
            "solution": "generated",
            "status": "pending",
            "error": "",
        }
        with runner.results_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")

    runner.run_task = fake_run_task
    runner.run(limit=1)

    records = [
        json.loads(line) for line in runner.results_path.read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["task_id"] == "BigCodeBench/1"
    assert records[0]["status"] == "pass"
    summary = json.loads((tmp_path / "results_summary.json").read_text())
    assert summary["total"] == 1
    assert summary["expected_total"] == 2
    assert summary["complete"] is False


def test_one_line_fenced_solution_survives_common_sanitization():
    import ast
    from experiments.memory_runner import sanitize_generated_code

    solution = sanitize_generated_code(
        "```python\ndef task_func(): return 1\n```", "task_func"
    )
    assert ast.parse(solution).body[0].name == "task_func"
    assert "```" not in solution


def test_transport_failure_is_not_retried_and_is_journaled(tmp_path, monkeypatch):
    import httpx
    import json
    import openai
    import pytest
    from types import SimpleNamespace

    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        raise openai.APIConnectionError(
            request=httpx.Request("POST", "https://example.invalid")
        )

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    generator = OpenAICompatibleGenerator(
        {"name": "fixed-model", "max_tokens": 128, "transient_max_retries": 3}
    )
    generator.journal_path = tmp_path / "requests.jsonl"
    generator.active_task_key = "task-1"
    with pytest.raises(openai.APIConnectionError):
        generator.generate("solve", "contract")
    assert len(attempts) == 1
    events = [
        json.loads(line) for line in generator.journal_path.read_text().splitlines()
    ]
    assert [e["event"] for e in events] == ["request_start", "request_error"]
    assert events[0]["task"] == "task-1"


def test_missing_or_over_budget_usage_aborts_generation(monkeypatch):
    import openai
    import pytest
    from types import SimpleNamespace

    for usage in (
        None,
        SimpleNamespace(prompt_tokens=3, completion_tokens=129, total_tokens=132),
    ):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"), finish_reason="stop"
                )
            ],
            usage=usage,
        )
        monkeypatch.setattr(
            openai,
            "OpenAI",
            lambda **kwargs: SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **kw: response)
                )
            ),
        )
        generator = OpenAICompatibleGenerator(
            {"name": "fixed-model", "max_tokens": 128}
        )
        with pytest.raises(RuntimeError):
            generator.generate("solve", "contract")
