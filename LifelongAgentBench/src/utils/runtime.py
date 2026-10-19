"""Resolved, credential-free experiment identity and online stream validation."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import re
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit
import uuid

from src.typings import SampleStatus, Session, SessionEvaluationOutcome
from .checkpoint import atomic_json, file_hash


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def sanitize(value: Any, key: str = '') -> Any:
    if any(word in key.lower() for word in ('api_key', 'password', 'secret', 'authorization', 'access_token')):
        return '<redacted>'
    if isinstance(value, Mapping):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(v) for v in value]
    if isinstance(value, str) and ('url' in key.lower() or value.startswith(('https://', 'http://'))):
        parsed = urlsplit(value)
        # Credentials and query strings must never be persisted.
        hostname = parsed.hostname or ''
        if ':' in hostname:
            hostname = f'[{hostname}]'
        netloc = hostname + (f':{parsed.port}' if parsed.port else '')
        return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip('/'), '', ''))
    return value


def validate_sample_order(order: Sequence[Any], available: Sequence[Any]) -> None:
    if not order:
        raise ValueError('The experiment must select at least one sample.')
    if len(set(order)) != len(order):
        raise ValueError('Sample order contains duplicate IDs.')
    missing = set(order) - set(available)
    if missing:
        raise ValueError(f'Sample IDs are absent from the dataset: {sorted(missing, key=str)}')


def invalid_session_reason(session: Session, *, evaluated: bool = True) -> str | None:
    invalid = {SampleStatus.AGENT_UNKNOWN_ERROR, SampleStatus.AGENT_OUT_OF_MEMORY,
               SampleStatus.TASK_ENVIRONMENT_ERROR, SampleStatus.TASK_UNKNOWN_ERROR}
    if session.sample_status in invalid:
        return f'Sample {session.sample_index}: {session.sample_status}: {session.finish_reason or "invalid execution"}'
    if evaluated and session.evaluation_record.outcome not in {SessionEvaluationOutcome.CORRECT, SessionEvaluationOutcome.INCORRECT}:
        return f'Sample {session.sample_index}: invalid evaluation {session.evaluation_record.outcome}'
    if evaluated and session.sample_status in {SampleStatus.INITIAL, SampleStatus.RUNNING}:
        return f'Sample {session.sample_index}: unfinished session'
    return None


def validate_prefix(sessions: Sequence[Session], sample_order: Sequence[Any], task_name: Any) -> None:
    ids = [session.sample_index for session in sessions]
    if ids != list(sample_order[:len(ids)]):
        raise ValueError('Checkpoint sessions are not a unique ordered prefix of the selected stream.')
    for session in sessions:
        if session.task_name != task_name:
            raise ValueError('Checkpoint contains another benchmark.')
        reason = invalid_session_reason(session)
        if reason:
            raise ValueError(reason)


def build_runtime_manifest(raw_config: Mapping[str, Any], assignment: Any,
                           language_models: Mapping[str, Any], project_root: Path | None = None,
                           task_instance: Any = None) -> dict[str, Any]:
    root = project_root or Path.cwd()
    primary_metric = 'success_rate'
    data_manifest = root / 'data/manifest.json'
    if data_manifest.exists():
        metadata = json.loads(data_manifest.read_text())
        primary_metric = metadata.get('benchmarks', {}).get(raw_config['assignment_config']['task'], {}).get('primary_metric', primary_metric)
    model_identity = {}
    for alias, model in language_models.items():
        client = getattr(model, 'client', None)
        model_identity[alias] = sanitize({
            'adapter': f'{type(model).__module__}.{type(model).__name__}',
            'model_name': getattr(model, 'model_name', None),
            'base_url': str(client.base_url) if client is not None else None,
            'maximum_prompt_token_count': getattr(model, 'maximum_prompt_token_count', None),
        })
    resource_files = {}
    def collect(value):
        if isinstance(value, Mapping):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, str) and not value.startswith(('env:', 'http://', 'https://')):
            path = root / value
            if path.is_file():
                resource_files[value] = file_hash(path)
            elif path.is_dir() and path.is_relative_to(root / 'data'):
                for resource in path.rglob('*'):
                    if resource.is_file():
                        resource_files[str(resource.relative_to(root))] = file_hash(resource)
    task_config = raw_config.get("task_dict", {}).get(raw_config["assignment_config"]["task"])
    if task_config is None:
        task_config = assignment.task.model_dump()
    def effective(kind, entry):
        default = raw_config[kind][entry['name']]
        return {'module': default['module'], 'parameters': {**(default.get('parameters') or {}), **(entry.get('custom_parameters') or {})}}
    agent_config = effective('agent_dict', raw_config['assignment_config']['agent'])
    collect(task_config)
    protocol = sanitize({'task_config': task_config, 'agent_config': agent_config,
                         'model_identity': model_identity, 'sample_order': list(assignment.sample_order),
                         'resource_files': resource_files, 'generation_budget': assignment.generation_budget,
                         'resource_budget': raw_config['assignment_config'].get('resource_budget', {}),
                         'runtime_environment_identity': getattr(task_instance, 'runtime_environment_identity', {}),
                         'learning_protocol': 'online_prequential',
                         'primary_metric': primary_metric, 'dependency_environment': dependency_environment(root)})
    methods = raw_config['assignment_config']['callback_dict']
    method = next((entry['name'] for entry in methods.values() if entry['name'] not in {'saving', 'abnormal', 'memory_monitor'}), 'standard')
    return {'schema_version': 1, 'task': raw_config['assignment_config']['task'], 'method': method,
            'model_identity': model_identity, 'sample_order': list(assignment.sample_order),
            'protocol_hash': digest(protocol), 'protocol': protocol,
            'method_config_hash': digest(sanitize({alias: effective('callback_dict', entry) for alias, entry in methods.items()})),
            'implementation_hash': implementation_hash(root), 'learning_protocol': 'online_prequential',
            'primary_metric': primary_metric, 'dependency_environment': protocol['dependency_environment'],
            'usage_ledger': 'usage.jsonl', 'request_journal': 'request_attempts.jsonl'}


def ensure_runtime_manifest(output_dir: str, manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(output_dir) / 'runtime_manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ('schema_version', 'task', 'method', 'model_identity', 'sample_order', 'protocol_hash', 'method_config_hash', 'implementation_hash', 'primary_metric', 'dependency_environment'):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f'Runtime identity changed ({key}); resume refused. Use a new output directory.')
        return previous
    if (Path(output_dir) / 'runs.json').exists():
        raise ValueError('Legacy run lacks resolved runtime identity; use a new output directory.')
    manifest['run_id'] = uuid.uuid4().hex
    atomic_json(path, manifest)
    return manifest


def resolved_model_identity(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only launcher precheck using the same OpenAI environment precedence."""
    identity = {}
    def resolve(value):
        if isinstance(value, str) and value.startswith('env:'):
            variable = value[4:]
            if variable not in os.environ:
                raise ValueError(f'Required model environment variable {variable!r} is unset.')
            return os.environ[variable]
        return value
    for entry in raw_config['assignment_config']['language_model_list']:
        alias = entry['name']
        info = raw_config['language_model_dict'][alias]
        parameters = {**(info.get('parameters') or {}), **(entry.get('custom_parameters') or {})}
        identity[alias] = sanitize({
            'adapter': info['module'],
            'model_name': resolve(parameters.get('model_name')),
            'base_url': resolve(parameters.get('base_url')) or os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1',
            'maximum_prompt_token_count': parameters.get('maximum_prompt_token_count'),
        })
    return identity


def implementation_hash(project_root: Path | None = None) -> str:
    root = project_root or Path.cwd()
    sources = {str(path.relative_to(root)): file_hash(path)
               for path in (list((root / 'src').rglob('*.py')) + [root / 'run.py']) if path.is_file()}
    return digest(sources)


class UncommittedInferenceError(RuntimeError):
    """External execution cannot be rolled back into a once-only paper run."""


def validate_no_replay(output_dir: str, sessions: Sequence[Session]) -> None:
    committed = {str(session.sample_index) for session in sessions}
    for filename in ('usage.jsonl', 'request_attempts.jsonl'):
        ledger = Path(output_dir) / filename
        if not ledger.exists():
            continue
        with ledger.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict) or not isinstance(entry.get('attempt', 0), int):
                        raise ValueError('Invalid journal record.')
                except (ValueError, TypeError) as error:
                    raise UncommittedInferenceError(f'{filename} line {number} is incomplete; once-only execution cannot be verified.') from error
                if entry.get('attempt', 0) >= 1 and str(entry.get('sample_index')) not in committed:
                    raise UncommittedInferenceError(
                        f'Sample {entry.get("sample_index")} has an uncommitted model request. '
                        'The paper forbids task replay; resume refused. Start a new independent run in a new output directory.')


def dependency_environment(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[2]
    with (root / 'pyproject.toml').open('rb') as stream:
        project = tomllib.load(stream)['project']
    names = {re.match(r'^[A-Za-z0-9_.-]+', requirement).group(0)
             for requirement in project['dependencies']}
    # The model adapter imports httpx directly; OpenAI installs it transitively.
    names.add('httpx')
    packages = {}
    for package in sorted(names, key=str.lower):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {'python': '.'.join(str(part) for part in sys.version_info[:3]), 'packages': packages}
