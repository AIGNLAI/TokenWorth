import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import run
from src.callbacks import Callback
from src.typings import LoggerConfig, PathConfig, SampleStatus, SessionEvaluationOutcome


class Factory:
    def __init__(self, value):
        self.value = value
    def model_dump(self):
        return self.value


class Memory(Callback):
    def __init__(self, root, events):
        super().__init__()
        self.set_state_dir(str(root / 'callback_state/memory'))
        self.learned = []
        self.events = events
    @classmethod
    def is_unique(cls):
        return True
    def on_task_complete(self, args):
        self.events.append(('learn', args.current_session.sample_index))
        self.learned.append(args.current_session.sample_index)
    def on_state_save(self, args):
        (Path(self.get_state_dir()) / 'state.json').write_text(json.dumps(self.learned))
    def restore_state(self):
        self.learned = json.loads((Path(self.get_state_dir()) / 'state.json').read_text())


class FakeTask:
    task_name = 'db_bench'
    def __init__(self, events, fail_evaluation=False):
        self.events = events
        self.released = 0
        self.fail_evaluation = fail_evaluation
    def reset(self, session):
        session.sample_status = SampleStatus.RUNNING
    def interact(self, session):
        session.sample_status = SampleStatus.COMPLETED
    def complete(self, session):
        self.events.append(('evaluate', session.sample_index))
        session.evaluation_record.outcome = SessionEvaluationOutcome.UNKNOWN if self.fail_evaluation else SessionEvaluationOutcome.CORRECT
    def release(self):
        self.released += 1
    def calculate_metric(self, partials):
        return {'overall': {'basic': {'session_count': len(partials)}}}


class FakeAgent:
    _inference_config_dict = {'max_tokens': 128}
    def __init__(self, events, fail_sample=None):
        self.events = events
        self.fail_sample = fail_sample
    def inference(self, session):
        self.events.append(('inference', session.sample_index))
        if session.sample_index == self.fail_sample:
            session.sample_status = SampleStatus.AGENT_UNKNOWN_ERROR
            session.finish_reason = '503 service unavailable'


def setup(tmp_path, monkeypatch, *, fail_sample=None, fail_evaluation=False, order=None):
    monkeypatch.setattr(run.coredumpy, 'patch_except', lambda **kwargs: None)
    events = []
    task = FakeTask(events, fail_evaluation)
    agent = FakeAgent(events, fail_sample)
    memory = Memory(tmp_path, events)
    model = SimpleNamespace(model_name='model-a', client=SimpleNamespace(base_url='http://host/v1'), audit_failure=None)
    task_info = {'module': 'fake.Task', 'parameters': {'max_round': 3}}
    agent_info = {'module': 'fake.Agent', 'parameters': {'language_model': 'model', 'inference_config_dict': {'max_tokens': 128}}}
    assignment = SimpleNamespace(output_dir=str(tmp_path), sample_order=order or ['0', '1'],
                                 task=Factory(task_info), agent=Factory(agent_info), generation_budget=128)
    utility = SimpleNamespace(assignment_config=assignment, language_model_dict={'model': model},
        remove_redundant_args=lambda raw: raw, construct=lambda: (task, agent, {'memory': memory}),
        postprocess=lambda *args: None, validate=lambda *args: None)
    raw = {'assignment_config': {'task': 'db_bench', 'output_dir': str(tmp_path), 'sample_order': list(assignment.sample_order),
           'agent': {'name': 'agent'}, 'callback_dict': {'memory': {'name': 'TokenWorth'}}},
           'agent_dict': {'agent': agent_info}, 'callback_dict': {'TokenWorth': {'module': 'fake.Memory', 'parameters': {}}}}
    paths = PathConfig(exception_record_file_path=str(tmp_path / 'exceptions.txt'), config_output_path=str(tmp_path / 'config.yaml'),
                       session_list_output_path=str(tmp_path / 'runs.json'), metric_output_path=str(tmp_path / 'metric.json'),
                       coredumpy_output_dir=str(tmp_path / 'coredumpy'))
    logger = LoggerConfig(level='ERROR', log_file_path=str(tmp_path / 'test.log'), logger_name='runner-test')
    return raw, utility, logger, paths, events, task, memory, model


def test_online_execution_evaluates_before_learning_and_resume_is_noop(tmp_path, monkeypatch):
    raw, utility, logger, paths, events, task, memory, _ = setup(tmp_path, monkeypatch)
    run.execute_assignment(raw, utility, logger, paths)
    assert events == [('inference', '0'), ('evaluate', '0'), ('learn', '0'), ('inference', '1'), ('evaluate', '1'), ('learn', '1')]
    assert task.released == 1
    before = list(events)
    run.execute_assignment(raw, utility, logger, paths)
    assert events == before
    assert memory.learned == ['0', '1']
    assert task.released == 2
    assert json.loads((tmp_path / 'run_status.json').read_text())['valid'] is True


def test_service_failure_never_trains_or_commits_failed_sample(tmp_path, monkeypatch):
    raw, utility, logger, paths, events, task, memory, _ = setup(tmp_path, monkeypatch, fail_sample='1')
    with pytest.raises(RuntimeError, match='503'):
        run.execute_assignment(raw, utility, logger, paths)
    assert memory.learned == ['0']
    assert ('learn', '1') not in events
    assert [record['sample_index'] for record in json.loads((tmp_path / 'runs.json').read_text())] == ['0']
    assert task.released == 1
    status = json.loads((tmp_path / 'run_status.json').read_text())
    assert status['status'] == 'failed' and not status['valid']
    assert status['completed_sample_count'] == 1


def test_evaluator_failure_prevents_learning_and_releases_task(tmp_path, monkeypatch):
    raw, utility, logger, paths, events, task, memory, _ = setup(tmp_path, monkeypatch, fail_evaluation=True)
    with pytest.raises(RuntimeError, match='invalid evaluation'):
        run.execute_assignment(raw, utility, logger, paths)
    assert memory.learned == []
    assert json.loads((tmp_path / 'runs.json').read_text()) == []
    assert task.released == 1


def test_swallowed_auxiliary_failure_prevents_checkpoint_commit(tmp_path, monkeypatch):
    raw, utility, logger, paths, events, task, memory, model = setup(tmp_path, monkeypatch)
    original = memory.on_task_complete
    def learn(args):
        original(args)
        model.audit_failure = 'auxiliary service unavailable'
    memory.on_task_complete = learn
    with pytest.raises(RuntimeError, match='Auxiliary inference failed'):
        run.execute_assignment(raw, utility, logger, paths)
    assert json.loads((tmp_path / 'runs.json').read_text()) == []
    # In-memory mutation is discarded on restore from generation zero.
    memory.restore_state()  # live state is still gen0 because save was skipped
    assert memory.learned == []
    assert task.released == 1


def test_crash_after_auxiliary_request_before_commit_refuses_replay(tmp_path, monkeypatch):
    from src.utils.runtime import UncommittedInferenceError
    raw, utility, logger, paths, events, task, memory, _ = setup(tmp_path, monkeypatch)
    original = memory.on_task_complete
    def crash(args):
        original(args)
        with (tmp_path / 'usage.jsonl').open('a') as stream:
            stream.write(json.dumps({'sample_index': args.current_session.sample_index, 'attempt': 1, 'phase': 'auxiliary', 'completion_tokens': 10}) + '\n')
        raise RuntimeError('crash after auxiliary inference')
    memory.on_task_complete = crash
    with pytest.raises(RuntimeError, match='crash after auxiliary'):
        run.execute_assignment(raw, utility, logger, paths)
    before = list(events)
    memory.on_task_complete = original
    with pytest.raises(UncommittedInferenceError, match='paper forbids task replay'):
        run.execute_assignment(raw, utility, logger, paths)
    assert events == before
    assert json.loads((tmp_path / 'runs.json').read_text()) == []
    assert (tmp_path / 'resume_rejection.json').exists()


def fail_cleanup(task):
    def release():
        task.released += 1
        def fail():
            raise RuntimeError('container removal failed')
        task._release = fail
        # Exercise the real Task wrapper: it must propagate cleanup failure.
        run.Task.release(task)
    task.release = release


def test_cleanup_failure_prevents_completed_status(tmp_path, monkeypatch):
    raw, utility, logger, paths, _, task, _, _ = setup(tmp_path, monkeypatch)
    fail_cleanup(task)
    with pytest.raises(RuntimeError, match='container removal failed'):
        run.execute_assignment(raw, utility, logger, paths)
    status = json.loads((tmp_path / 'run_status.json').read_text())
    assert status['status'] == 'failed' and not status['valid']
    assert status['completed_sample_count'] == 2
    assert 'Task cleanup failed' in status['invalid_reasons'][0]
    assert task.released == 1


def test_inference_failure_and_cleanup_failure_keep_original_error(tmp_path, monkeypatch):
    raw, utility, logger, paths, _, task, _, _ = setup(tmp_path, monkeypatch, fail_sample='1')
    fail_cleanup(task)
    with pytest.raises(RuntimeError, match='503') as captured:
        run.execute_assignment(raw, utility, logger, paths)
    assert any('container removal failed' in note for note in captured.value.__notes__)
    status = json.loads((tmp_path / 'run_status.json').read_text())
    assert len(status['invalid_reasons']) == 2
    assert '503' in status['invalid_reasons'][0]
    assert 'container removal failed' in status['invalid_reasons'][1]


def test_identity_rejection_cleanup_failure_preserves_prior_valid_result(tmp_path, monkeypatch):
    raw, utility, logger, paths, _, task, _, model = setup(tmp_path, monkeypatch)
    run.execute_assignment(raw, utility, logger, paths)
    original = (tmp_path / 'run_status.json').read_bytes()
    model.model_name = 'different-model'
    fail_cleanup(task)
    with pytest.raises(ValueError, match='model_identity') as captured:
        run.execute_assignment(raw, utility, logger, paths)
    assert (tmp_path / 'run_status.json').read_bytes() == original
    assert any('container removal failed' in note for note in captured.value.__notes__)
    reasons = json.loads((tmp_path / 'resume_rejection.json').read_text())['invalid_reasons']
    assert len(reasons) == 2


@pytest.mark.parametrize('package', ['mysql-connector-python', 'sqlglot', 'bashlex', 'SPARQLWrapper', 'psutil'])
def test_task_dependency_change_refuses_resume_before_inference(tmp_path, monkeypatch, package):
    from src.utils import runtime
    raw, utility, logger, paths, events, _, _, _ = setup(tmp_path, monkeypatch)
    run.execute_assignment(raw, utility, logger, paths)
    before = list(events)
    original_version = runtime.version
    monkeypatch.setattr(runtime, 'version', lambda name: 'changed-version' if name == package else original_version(name))
    with pytest.raises(ValueError, match='protocol_hash'):
        run.execute_assignment(raw, utility, logger, paths)
    assert events == before
    assert json.loads((tmp_path / 'run_status.json').read_text())['valid'] is True
