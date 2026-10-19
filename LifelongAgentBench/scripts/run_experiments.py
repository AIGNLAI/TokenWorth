#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import datetime as dt
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_outputs import generate_report, validate_output


def _deep_merge(base_item: object, new_item: object) -> object:
    if isinstance(base_item, dict) and isinstance(new_item, dict):
        ret = dict(base_item)
        for key, value in new_item.items():
            if key in ret:
                ret[key] = _deep_merge(ret[key], value)
            else:
                ret[key] = value
        return ret
    if isinstance(base_item, list) and isinstance(new_item, list):
        return [*base_item, *new_item]
    return new_item


class LightweightConfigLoader:
    def __init__(self) -> None:
        self.loading: set[Path] = set()
        self.loaded: dict[Path, object] = {}

    def load_from(self, path: str | Path) -> dict:
        path = Path(path).resolve()
        if path in self.loading:
            raise RuntimeError(f"Circular import detected: {path}")
        if path in self.loaded:
            loaded = self.loaded[path]
            return yaml.safe_load(yaml.safe_dump(loaded))
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as f:
            if path.suffix in {".yaml", ".yml"}:
                config = yaml.safe_load(f)
            else:
                raise RuntimeError(f"Unknown file type: {path}")
        self.loading.add(path)
        try:
            config = self._parse_imports(path.parent, config)
        finally:
            self.loading.remove(path)
        self.loaded[path] = config
        return self._parse_default_and_overwrite(yaml.safe_load(yaml.safe_dump(config)))

    def _parse_imports(self, base_dir: Path, raw_config: object) -> object:
        if isinstance(raw_config, dict):
            ret: object = {}
            raw_config = dict(raw_config)
            if "import" in raw_config:
                import_value = raw_config.pop("import")
                if isinstance(import_value, str):
                    ret = _deep_merge(ret, self.load_from(base_dir / import_value))
                elif isinstance(import_value, list):
                    for item in import_value:
                        if not isinstance(item, str):
                            raise TypeError(
                                f"Import list must contain strings, found {type(item)}"
                            )
                        ret = _deep_merge(ret, self.load_from(base_dir / item))
                else:
                    raise TypeError(f"Unknown import value: {import_value}")
            parsed = {
                key: self._parse_imports(base_dir, value)
                for key, value in raw_config.items()
            }
            return _deep_merge(ret, parsed)
        if isinstance(raw_config, list):
            return [self._parse_imports(base_dir, item) for item in raw_config]
        return raw_config

    def _parse_default_and_overwrite(self, config: object) -> object:
        if isinstance(config, dict):
            if not config:
                return {}
            config = dict(config)
            overwrite = None
            default = None
            if "overwrite" in config:
                overwrite = self._parse_default_and_overwrite(config.pop("overwrite"))
            if "default" in config:
                default = self._parse_default_and_overwrite(config.pop("default"))
            ret = {}
            for key, value in config.items():
                parsed_value = self._parse_default_and_overwrite(value)
                if overwrite is not None:
                    parsed_value = _deep_merge(parsed_value, overwrite)
                if default is not None:
                    parsed_value = _deep_merge(default, parsed_value)
                ret[key] = parsed_value
            return ret
        if isinstance(config, list):
            return [self._parse_default_and_overwrite(item) for item in config]
        return config


@dataclass(frozen=True)
class RunSpec:
    name: str
    source_config_path: Path
    generated_config_path: Path
    output_dir: Path
    log_path: Path
    command: list[str]
    cuda_visible_devices: Optional[str]


@dataclass
class ActiveRun:
    spec: RunSpec
    process: subprocess.Popen
    log_file: object
    started_at: float


@dataclass(frozen=True)
class CompletedRun:
    spec: RunSpec
    return_code: int
    elapsed_seconds: float


def _timestamp() -> str:
    now = dt.datetime.now()
    return f'{now.strftime("%Y-%m-%d-%H-%M-%S")}-{now.microsecond // 1000:03d}'


def _path_for_config_display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _slugify_config_path(config_path: Path) -> str:
    display_path = _path_for_config_display(config_path.with_suffix(""))
    experiment_prefix = "configs/experiments/"
    if display_path.startswith(experiment_prefix):
        relative_parts = Path(display_path[len(experiment_prefix) :]).parts
        if len(relative_parts) == 2:
            benchmark, method = relative_parts
            benchmark = {
                "db": "db",
                "knowledge_graph": "knowledge_graph",
                "os_interaction": "os",
            }.get(benchmark, benchmark)
            display_path = f"{method}-{benchmark}"
    else:
        examples_prefix = "configs/examples/"
        if display_path.startswith(examples_prefix):
            display_path = display_path[len(examples_prefix) :]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", display_path).strip("-")
    return slug[-140:] or "experiment"


def _deduplicate_path(path: Path, used: set[Path]) -> Path:
    candidate = path
    suffix = 2
    while candidate in used or candidate.exists():
        candidate = path.with_name(f"{path.name}-{suffix}")
        suffix += 1
    used.add(candidate)
    return candidate


def _repo_relative_or_absolute(path: Path) -> str:
    if path.is_absolute():
        try:
            return path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _apply_overrides(raw_config: dict, args: argparse.Namespace) -> dict:
    from run import apply_language_model_overrides

    raw_config = apply_language_model_overrides(
        raw_config,
        args.llm_name,
        args.llm_model_name,
        args.llm_base_url,
        args.llm_api_key,
        args.llm_maximum_prompt_token_count,
        args.llm_timeout,
        args.llm_max_retries,
        False if args.llm_ignore_env_proxy else None,
    )
    if (
        args.kg_sparql_url
        and raw_config["assignment_config"]["task"] == "knowledge_graph"
    ):
        raw_config["task_dict"]["knowledge_graph"]["parameters"][
            "sparql_url"
        ] = args.kg_sparql_url
    task = raw_config["assignment_config"]["task"]
    image = (
        args.db_image
        if task == "db_bench"
        else args.os_image if task == "os_interaction" else None
    )
    if image:
        raw_config["task_dict"][task]["parameters"]["container_image"] = image
    return raw_config


def _prepare_generated_config(
    source: Path, output: Path, args: argparse.Namespace
) -> Path:
    config = _apply_overrides(LightweightConfigLoader().load_from(source), args)
    config["assignment_config"]["output_dir"] = str(output)
    # This config persists. Keep all model credentials as environment references.
    for info in config.get("language_model_dict", {}).values():
        key = info.get("parameters", {}).get("api_key")
        if key and not key.startswith("env:") and key != "EMPTY":
            raise ValueError(
                "Model API keys must use an env: reference in saved batch configs"
            )
    output.mkdir(parents=True, exist_ok=False)
    path = output / "launcher_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _build_env(cuda_visible_devices: Optional[str]) -> dict[str, str]:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + previous if previous else "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def _spec_record(spec: RunSpec) -> dict[str, Any]:
    return dict(
        name=spec.name,
        source_config_path=str(spec.source_config_path),
        generated_config_path=str(spec.generated_config_path),
        output_dir=str(spec.output_dir),
        log_path=str(spec.log_path),
        cuda_visible_devices=spec.cuda_visible_devices,
        config_sha256=_file_hash(spec.generated_config_path),
        model_identity=None,
        status="pending",
    )


def _make_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    if args.resume:
        return _resume_specs(args)
    root = _resolve_path(args.output_root or f"outputs/batch-{_timestamp()}")
    if root.exists():
        raise ValueError(f"Batch directory exists: {root}; use --resume to continue it")
    root.mkdir(parents=True)
    args.batch_root = root
    entrypoint = REPO_ROOT / "run.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(entrypoint)
    devices = [
        device.strip()
        for device in (args.cuda_devices or "").split(",")
        if device.strip()
    ]
    if args.cuda_devices and not devices:
        raise ValueError("--cuda-devices was provided but no device ID was parsed")
    specs, used = [], set()
    for repeat in range(args.repeats):
        for source_name in args.config_paths:
            source = _resolve_path(source_name)
            if not source.is_file():
                raise FileNotFoundError(source)
            name = f"{_slugify_config_path(source)}-repeat-{repeat + 1}"
            output = _deduplicate_path(root / name, used)
            generated = _prepare_generated_config(source, output, args)
            device = devices[len(specs) % len(devices)] if devices else None
            specs.append(
                RunSpec(
                    name,
                    source,
                    generated,
                    output,
                    output / "launcher.log",
                    [args.python, str(entrypoint), "--config_path", str(generated)],
                    device,
                )
            )
    args.batch_manifest = dict(
        schema_version=1,
        created_at=_timestamp(),
        status="prepared",
        runs=[_spec_record(spec) for spec in specs],
    )
    _atomic_json(root / "batch_manifest.json", args.batch_manifest)
    return specs


def _resume_specs(args: argparse.Namespace) -> list[RunSpec]:
    from src.utils.runtime import (
        implementation_hash,
        resolved_model_identity,
        validate_no_replay,
        dependency_environment,
    )
    from src.utils.checkpoint import CheckpointManager
    from src.typings import Session

    root = _resolve_path(args.resume)
    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("runs"), list
    ):
        raise ValueError("Unsupported or invalid batch manifest")
    args.batch_root, args.batch_manifest = root, manifest
    specs = []
    for record in manifest["runs"]:
        output, generated = Path(record["output_dir"]), Path(
            record["generated_config_path"]
        )
        if (
            output.resolve().parent != root
            or generated.resolve().parent != output.resolve()
        ):
            raise ValueError("Batch run paths are outside the batch directory")
        if _file_hash(generated) != record["config_sha256"]:
            raise ValueError(f"Saved batch config changed: {record['name']}")
        config = LightweightConfigLoader().load_from(generated)
        identity = resolved_model_identity(config)
        if (
            record.get("model_identity") is not None
            and identity != record["model_identity"]
        ):
            raise ValueError(
                f"Resolved model identity changed: {record['name']}; resume refused"
            )
        runtime_path = output / "runtime_manifest.json"
        if runtime_path.is_file():
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            if runtime.get("model_identity") != identity:
                raise ValueError(
                    f"Run model identity changed: {record['name']}; resume refused"
                )
            if (
                runtime.get("dependency_environment") is not None
                and runtime["dependency_environment"] != dependency_environment()
            ):
                raise ValueError(
                    f"Dependencies changed: {record['name']}; resume refused"
                )
            if runtime.get("implementation_hash") != implementation_hash(REPO_ROOT):
                raise ValueError(
                    f"Run implementation changed: {record['name']}; resume refused"
                )
        if runtime_path.is_file():
            for resource, expected_hash in (
                runtime.get("protocol", {}).get("resource_files", {}).items()
            ):
                if _file_hash(_resolve_path(resource)) != expected_hash:
                    raise ValueError(
                        f"Task resource changed: {record['name']}; resume refused"
                    )
            task = config["assignment_config"]["task"]
            environment = runtime.get("protocol", {}).get(
                "runtime_environment_identity", {}
            )
            if task in {"db_bench", "os_interaction"} and environment:
                import docker

                client = docker.from_env()
                try:
                    image = config["task_dict"][task]["parameters"].get(
                        "container_image",
                        "mysql:8.0" if task == "db_bench" else "local-os/default",
                    )
                    image_id = client.images.get(image).id
                    if environment.get("container_image") != image_id:
                        raise ValueError(
                            f"Container image changed: {record['name']}; resume refused"
                        )
                finally:
                    client.close()
        if (output / "checkpoints" / "CURRENT").exists():
            CheckpointManager(str(output)).peek_sessions()
        valid, _ = validate_output(output)
        if valid:
            record["status"] = "completed"
            print(f"Already completed: {record['name']}")
            continue
        if (output / "usage.jsonl").exists() or (
            output / "request_attempts.jsonl"
        ).exists():
            committed = CheckpointManager(str(output)).peek_sessions() or []
            validate_no_replay(
                str(output), [Session.model_validate(row) for row in committed]
            )
        record["status"] = "pending"
        specs.append(
            RunSpec(
                record["name"],
                Path(record["source_config_path"]),
                generated,
                output,
                Path(record["log_path"]),
                [
                    args.python,
                    str(REPO_ROOT / "run.py"),
                    "--config_path",
                    str(generated),
                ],
                record.get("cuda_visible_devices"),
            )
        )
    _atomic_json(root / "batch_manifest.json", manifest)
    return specs


def _model_preflight(config: dict[str, Any]) -> None:
    """Perform a completion without touching callbacks or the training ledger."""
    from src.utils.runtime import resolved_model_identity
    import httpx
    from openai import OpenAI

    identities = resolved_model_identity(config)
    for entry in config["assignment_config"]["language_model_list"]:
        alias = entry["name"]
        info = config["language_model_dict"][alias]
        if (
            info["module"]
            != "src.language_models.instance.openai_language_model.OpenaiLanguageModel"
        ):
            raise ValueError("Preflight requires an OpenAI-compatible model adapter")
        params = {**info.get("parameters", {}), **entry.get("custom_parameters", {})}

        def resolve(value):
            return (
                os.environ[value[4:]]
                if isinstance(value, str) and value.startswith("env:")
                else value
            )

        key = resolve(params.get("api_key")) or os.environ.get("OPENAI_API_KEY")
        url = resolve(params.get("base_url")) or os.environ.get("OPENAI_BASE_URL")
        timeout = min(float(params.get("timeout") or 30), 30)
        with OpenAI(
            api_key=key,
            base_url=url,
            timeout=timeout,
            max_retries=0,
            http_client=httpx.Client(trust_env=params.get("trust_env", True)),
        ) as client:
            try:
                response = client.chat.completions.create(
                    model=identities[alias]["model_name"],
                    messages=[{"role": "user", "content": "Reply with OK."}],
                    max_tokens=1,
                    temperature=0,
                )
            except Exception as error:
                raise RuntimeError(
                    f"Model completion preflight failed ({type(error).__name__})"
                ) from None
            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("Model completion preflight returned no text")
            usage = response.usage
            prompt = getattr(usage, "prompt_tokens", None)
            output = getattr(usage, "completion_tokens", None)
            if any(type(count) is not int or count < 0 for count in (prompt, output)):
                raise RuntimeError(
                    "Model completion preflight requires nonnegative integer prompt/output token usage"
                )
            if output > 1:
                raise RuntimeError(
                    "Model completion preflight output exceeds requested generation cap"
                )
            total = getattr(usage, "total_tokens", None)
            if total is None:
                total = prompt + output
            if type(total) is not int or total != prompt + output:
                raise RuntimeError(
                    "Model completion preflight returned inconsistent total token usage"
                )


def _docker_preflight(task: str, image: str) -> None:
    import docker

    client = docker.from_env()
    container = None
    try:
        client.ping()
        image = client.images.get(image).id
        if task == "os_interaction":
            client.containers.run(
                image,
                command=[
                    "bash",
                    "-lc",
                    "command -v addgroup groupadd useradd chmod chage >/dev/null",
                ],
                remove=True,
            )
        else:
            container = client.containers.run(
                image,
                command="--innodb-use-native-aio=0",
                environment={"MYSQL_ROOT_PASSWORD": "tokenworth-preflight"},
                detach=True,
            )
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                container.reload()
                if container.status == "exited":
                    raise RuntimeError("MySQL container exited during preflight")
                code, _ = container.exec_run(
                    [
                        "mysqladmin",
                        "--host=127.0.0.1",
                        "--user=root",
                        "ping",
                        "--silent",
                    ],
                    environment={"MYSQL_PWD": "tokenworth-preflight"},
                )
                if code == 0:
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("MySQL was not ready within 30 seconds")
    except Exception as error:
        raise RuntimeError(
            f"Docker preflight for {task} failed ({type(error).__name__})"
        ) from None
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
        client.close()


def _kg_preflight(url: str) -> None:
    from SPARQLWrapper import JSON, SPARQLWrapper

    query = SPARQLWrapper(url)
    query.setReturnFormat(JSON)
    query.setTimeout(30)
    query.setQuery("ASK WHERE { ?s ?p ?o }")
    try:
        result = query.query().convert()
    except Exception as error:
        raise RuntimeError(
            f"KG query preflight failed ({type(error).__name__})"
        ) from None
    if not isinstance(result, dict) or result.get("boolean") is not True:
        raise RuntimeError("KG preflight requires a nonempty queryable graph")


def _preflight(specs: list[RunSpec], args: argparse.Namespace) -> None:
    from scripts.verify_data import verify
    from src.utils.runtime import digest, resolved_model_identity

    verify()
    checked_models, checked_environments = set(), set()
    for spec in specs:
        config = LightweightConfigLoader().load_from(spec.generated_config_path)
        identity = resolved_model_identity(config)
        fingerprint = digest(identity)
        if fingerprint not in checked_models:
            _model_preflight(config)
            checked_models.add(fingerprint)
        task = config["assignment_config"]["task"]
        if task in {"db_bench", "os_interaction"}:
            image = config["task_dict"][task]["parameters"].get(
                "container_image",
                "mysql:8.0" if task == "db_bench" else "local-os/default",
            )
            if (task, image) not in checked_environments:
                _docker_preflight(task, image)
                checked_environments.add((task, image))
        elif task == "knowledge_graph":
            url = config["task_dict"][task]["parameters"]["sparql_url"]
            if (task, url) not in checked_environments:
                _kg_preflight(url)
                checked_environments.add((task, url))
        else:
            raise ValueError(f"Unsupported benchmark: {task}")
        for record in args.batch_manifest["runs"]:
            if record["output_dir"] == str(spec.output_dir):
                record["model_identity"] = identity
    args.batch_manifest["status"] = "ready"
    _atomic_json(args.batch_root / "batch_manifest.json", args.batch_manifest)
    print("Preflight passed: data, model completions, and required environments")


def _update_batch(
    args: argparse.Namespace,
    spec: RunSpec,
    status: str,
    return_code: int | None = None,
    reason: str = "",
) -> None:
    for record in args.batch_manifest["runs"]:
        if record["output_dir"] == str(spec.output_dir):
            record.update(status=status, return_code=return_code, reason=reason)
    args.batch_manifest["status"] = "running"
    _atomic_json(args.batch_root / "batch_manifest.json", args.batch_manifest)


def _print_plan(specs: list[RunSpec], jobs: int) -> None:
    print(f"Prepared {len(specs)} experiment(s), running up to {jobs} concurrently")
    for spec in specs:
        print(f"- {spec.name}: {_repo_relative_or_absolute(spec.output_dir)}")


def _run_specs(specs: list[RunSpec], args: argparse.Namespace) -> int:
    pending, active, completed = list(specs), [], []
    stop_requested = False
    stop_deadline = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested, stop_deadline
        stop_requested = True
        stop_deadline = time.monotonic() + 10
        print(f"Received signal {signum}; terminating active experiments")
        for run in active:
            if run.process.poll() is None:
                run.process.send_signal(signal.SIGINT)

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while pending or active:
            if stop_deadline is not None and time.monotonic() >= stop_deadline:
                for run in active:
                    if run.process.poll() is None:
                        run.process.kill()
            while pending and len(active) < args.jobs and not stop_requested:
                spec = pending.pop(0)
                log_file = spec.log_path.open("a", buffering=1, encoding="utf-8")
                log_file.write(
                    f"[launcher] start {_timestamp()}\n[launcher] generated_config={spec.generated_config_path}\n"
                )
                process = subprocess.Popen(
                    spec.command,
                    cwd=REPO_ROOT,
                    env=_build_env(spec.cuda_visible_devices),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                _update_batch(args, spec, "running")
                active.append(ActiveRun(spec, process, log_file, time.time()))
                print(f"Started pid={process.pid} {spec.name}")
            next_active = []
            for run in active:
                code = run.process.poll()
                if code is None:
                    next_active.append(run)
                    continue
                elapsed, reason = time.time() - run.started_at, ""
                if code == 0:
                    valid, reason = validate_output(run.spec.output_dir)
                    if not valid:
                        code = 1
                _update_batch(
                    args, run.spec, "completed" if code == 0 else "failed", code, reason
                )
                run.log_file.write(
                    f"[launcher] end {_timestamp()} return_code={code} elapsed_seconds={elapsed:.1f}\n"
                )
                if reason:
                    run.log_file.write(f"[launcher] invalid result: {reason}\n")
                run.log_file.close()
                completed.append(CompletedRun(run.spec, code, elapsed))
                print(
                    f"Finished {'OK' if code == 0 else f'FAIL({code})'} {run.spec.name}"
                )
            active = next_active
            if stop_requested and not active:
                break
            if active or pending:
                time.sleep(min(args.poll_interval, 30))
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        for run in active:
            if run.process.poll() is None:
                run.process.terminate()
            try:
                run.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                run.process.kill()
                run.process.wait(timeout=5)
            run.log_file.close()
    failures = sum(run.return_code != 0 for run in completed)
    args.batch_manifest["status"] = (
        "interrupted" if stop_requested else ("failed" if failures else "completed")
    )
    _atomic_json(args.batch_root / "batch_manifest.json", args.batch_manifest)
    return 130 if stop_requested else min(failures, 125)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume a validated lifelong-learning batch"
    )
    parser.add_argument("config_paths", nargs="*", help="Experiment YAML paths")
    parser.add_argument("-j", "--jobs", type=int, default=2)
    parser.add_argument("--output-root", help="New batch directory; it must not exist")
    parser.add_argument("--resume", help="Resume the exact existing batch directory")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda-devices")
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare configs without contacting services",
    )
    parser.add_argument("--llm-name")
    parser.add_argument("--llm-model-name")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key", help="An env: reference")
    parser.add_argument("--llm-maximum-prompt-token-count", type=int)
    parser.add_argument("--llm-timeout", type=float)
    parser.add_argument("--llm-max-retries", type=int)
    parser.add_argument("--llm-ignore-env-proxy", action="store_true")
    parser.add_argument("--kg-sparql-url")
    parser.add_argument("--db-image", help="DB Docker image reference")
    parser.add_argument("--os-image", help="OS Docker image reference")
    args = parser.parse_args()
    if args.jobs <= 0 or args.repeats <= 0 or args.poll_interval <= 0:
        parser.error("--jobs, --repeats, and --poll-interval must be positive")
    if args.llm_api_key and not args.llm_api_key.startswith("env:"):
        parser.error("--llm-api-key must be an env: reference")
    if args.resume:
        changed = (
            args.config_paths
            or args.output_root
            or args.cuda_devices
            or args.repeats != 1
            or any(
                getattr(args, name) is not None
                for name in (
                    "llm_name",
                    "llm_model_name",
                    "llm_base_url",
                    "llm_api_key",
                    "llm_maximum_prompt_token_count",
                    "llm_timeout",
                    "llm_max_retries",
                    "kg_sparql_url",
                    "db_image",
                    "os_image",
                )
            )
            or args.llm_ignore_env_proxy
        )
        if changed:
            parser.error(
                "--resume uses saved configs; do not supply configs or protocol overrides"
            )
    elif not args.config_paths:
        parser.error("Supply configs for a new batch or --resume for an existing one")
    return args


def main() -> int:
    args = parse_args()
    try:
        specs = _make_run_specs(args)
        _print_plan(specs, args.jobs)
        if args.dry_run:
            return 0
        if specs:
            _preflight(specs, args)
        result = _run_specs(specs, args)
        generate_report(args.batch_root, args.batch_root / "statistics")
        print(f"Report: {args.batch_root / 'statistics' / 'summary.md'}")
        return result
    except (OSError, ValueError, RuntimeError) as error:
        if hasattr(args, "batch_manifest"):
            args.batch_manifest["status"] = "failed"
            args.batch_manifest["failure_reason"] = str(error)
            _atomic_json(args.batch_root / "batch_manifest.json", args.batch_manifest)
        print(f"Batch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
