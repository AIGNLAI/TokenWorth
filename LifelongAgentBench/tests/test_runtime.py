import json
import pytest
from src.typings import Session, SampleStatus, SessionEvaluationOutcome
from src.utils.runtime import ensure_runtime_manifest, sanitize, validate_sample_order, validate_prefix, invalid_session_reason, resolved_model_identity


def manifest():
    return {'schema_version': 1, 'task': 'db_bench', 'method': 'TokenWorth',
            'model_identity': {'model': {'model_name': 'resolved-a'}},
            'sample_order': ['0', '1'], 'protocol_hash': 'protocol',
            'method_config_hash': 'method', 'implementation_hash': 'code'}


@pytest.mark.parametrize('field', ['model_identity', 'sample_order', 'protocol_hash', 'method_config_hash', 'implementation_hash'])
def test_resume_rejects_changed_identity(tmp_path, field):
    original = ensure_runtime_manifest(str(tmp_path), manifest())
    modified = manifest()
    modified[field] = 'changed'
    with pytest.raises(ValueError, match=field):
        ensure_runtime_manifest(str(tmp_path), modified)
    assert json.loads((tmp_path / 'runtime_manifest.json').read_text()) == original


def test_manifest_preserves_run_id_on_resume(tmp_path):
    original = ensure_runtime_manifest(str(tmp_path), manifest())
    assert ensure_runtime_manifest(str(tmp_path), manifest()) == original


def test_secrets_and_url_credentials_removed():
    result = sanitize({'api_key': 'secret', 'nested': {'password': 'password'},
                       'base_url': 'https://user:pass@host:8000/v1/?api_key=secret#key'})
    assert result == {'api_key': '<redacted>', 'nested': {'password': '<redacted>'}, 'base_url': 'https://host:8000/v1'}


@pytest.mark.parametrize('order', [[], ['a', 'a'], ['missing']])
def test_sample_order_invalid(order):
    with pytest.raises(ValueError):
        validate_sample_order(order, ['a', 'b'])


def session(index='a', status=SampleStatus.COMPLETED, outcome=SessionEvaluationOutcome.CORRECT):
    item = Session(task_name='db_bench', sample_index=index)
    item.sample_status = status
    item.evaluation_record.outcome = outcome
    return item


@pytest.mark.parametrize('sessions', [[session('b')], [session('a'), session('a')], [session('a'), session('c')]])
def test_checkpoint_requires_exact_prefix(sessions):
    with pytest.raises(ValueError, match='prefix'):
        validate_prefix(sessions, ['a', 'b'], 'db_bench')


@pytest.mark.parametrize('status', [SampleStatus.AGENT_UNKNOWN_ERROR, SampleStatus.AGENT_OUT_OF_MEMORY, SampleStatus.TASK_ENVIRONMENT_ERROR, SampleStatus.TASK_UNKNOWN_ERROR])
def test_infrastructure_failures_are_invalid(status):
    assert invalid_session_reason(session(status=status)) is not None


def test_budget_limit_is_legal_failure():
    assert invalid_session_reason(session(status=SampleStatus.AGENT_CONTEXT_LIMIT, outcome=SessionEvaluationOutcome.INCORRECT)) is None


def test_model_env_is_resolved_for_launcher(monkeypatch):
    raw = {'assignment_config': {'language_model_list': [{'name': 'model'}]},
           'language_model_dict': {'model': {'module': 'adapter.Model', 'parameters': {'model_name': 'env:TEST_MODEL'}}}}
    monkeypatch.setenv('TEST_MODEL', 'first')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://user:pass@host/v1/?key=secret')
    assert resolved_model_identity(raw)['model']['model_name'] == 'first'
    assert resolved_model_identity(raw)['model']['base_url'] == 'https://host/v1'
    monkeypatch.setenv('TEST_MODEL', 'second')
    assert resolved_model_identity(raw)['model']['model_name'] == 'second'


def test_uncommitted_model_request_forbids_replay(tmp_path):
    from src.utils.runtime import validate_no_replay, UncommittedInferenceError
    (tmp_path / 'usage.jsonl').write_text(json.dumps({'sample_index': 'b', 'attempt': 1, 'completion_tokens': 10}) + '\n')
    with pytest.raises(UncommittedInferenceError, match='paper forbids task replay'):
        validate_no_replay(str(tmp_path), [session('a')])


def test_committed_model_requests_allow_resume(tmp_path):
    from src.utils.runtime import validate_no_replay
    (tmp_path / 'usage.jsonl').write_text(json.dumps({'sample_index': 'a', 'attempt': 1, 'completion_tokens': 10}) + '\n')
    validate_no_replay(str(tmp_path), [session('a')])


def test_no_request_budget_guard_does_not_prevent_resume(tmp_path):
    from src.utils.runtime import validate_no_replay
    (tmp_path / 'usage.jsonl').write_text(json.dumps({'sample_index': 'b', 'attempt': 0, 'status': 'budget_exhausted'}) + '\n')
    validate_no_replay(str(tmp_path), [session('a')])


def test_partial_cost_ledger_prevents_resume(tmp_path):
    from src.utils.runtime import validate_no_replay, UncommittedInferenceError
    (tmp_path / 'usage.jsonl').write_text('{"sample_index": "b"')
    with pytest.raises(UncommittedInferenceError, match='incomplete'):
        validate_no_replay(str(tmp_path), [])


def test_killed_inflight_request_journal_forbids_replay_without_usage(tmp_path):
    from src.utils.runtime import validate_no_replay, UncommittedInferenceError
    (tmp_path / 'request_attempts.jsonl').write_text(json.dumps({'sample_index': 'b', 'attempt': 1, 'status': 'request_started'}) + '\n')
    with pytest.raises(UncommittedInferenceError, match='paper forbids task replay'):
        validate_no_replay(str(tmp_path), [session('a')])


def test_dependency_identity_covers_declared_runtime_requirements():
    import re
    import tomllib
    from pathlib import Path
    from src.utils.runtime import dependency_environment
    root = Path(__file__).resolve().parents[1]
    with (root / 'pyproject.toml').open('rb') as stream:
        declared = tomllib.load(stream)['project']['dependencies']
    names = {re.match(r'^[A-Za-z0-9_.-]+', item).group(0) for item in declared}
    packages = dependency_environment(root)['packages']
    assert names | {'httpx'} == set(packages)
    assert {'mysql-connector-python', 'sqlglot', 'bashlex', 'SPARQLWrapper', 'psutil'} <= set(packages)
    assert 'pymysql' not in packages
