import json
from pathlib import Path
import pytest
from src.utils import checkpoint as module
from src.utils.checkpoint import CheckpointManager, run_lock


def write_state(root, step):
    state = root / 'callback_state' / 'memory'
    state.mkdir(parents=True, exist_ok=True)
    (state / 'memory.json').write_text(json.dumps({'step': step}))
    (state / 'alpha.json').write_text(json.dumps([step]))


def test_restore_ignores_partial_live_state(tmp_path):
    write_state(tmp_path, 1)
    manager = CheckpointManager(str(tmp_path))
    manager.commit([{'sample_index': 'a'}])
    write_state(tmp_path, 99)
    (tmp_path / 'runs.json').write_text('[{"sample_index": "b"}]')
    assert CheckpointManager(str(tmp_path)).restore() == [{'sample_index': 'a'}]
    assert json.loads((tmp_path / 'callback_state/memory/memory.json').read_text()) == {'step': 1}


@pytest.mark.parametrize('fault', ['before_directory_commit', 'before_pointer', 'after_pointer'])
def test_commit_failure_recovers_whole_generation(tmp_path, monkeypatch, fault):
    write_state(tmp_path, 1)
    manager = CheckpointManager(str(tmp_path))
    manager.commit([{'sample_index': 'a'}])
    write_state(tmp_path, 2)
    original_json = module.atomic_json
    original_replace = module.os.replace
    def atomic(path, value):
        if fault == 'before_pointer' and path.name == 'CURRENT':
            raise OSError('injected before commit')
        if fault == 'after_pointer' and path == tmp_path / 'runs.json':
            raise OSError('injected after commit')
        original_json(path, value)
    def replace(source, destination):
        if fault == 'before_directory_commit' and str(source).endswith('.pending'):
            raise OSError('injected staging failure')
        original_replace(source, destination)
    monkeypatch.setattr(module, 'atomic_json', atomic)
    monkeypatch.setattr(module.os, 'replace', replace)
    with pytest.raises(OSError):
        manager.commit([{'sample_index': 'a'}, {'sample_index': 'b'}])
    monkeypatch.undo()
    sessions = CheckpointManager(str(tmp_path)).restore()
    expected = 2 if fault == 'after_pointer' else 1
    assert len(sessions) == expected
    assert json.loads((tmp_path / 'callback_state/memory/memory.json').read_text())['step'] == expected
    assert json.loads((tmp_path / 'callback_state/memory/alpha.json').read_text()) == [expected]


def test_corrupt_committed_file_is_rejected(tmp_path):
    write_state(tmp_path, 1)
    manager = CheckpointManager(str(tmp_path))
    manager.commit([])
    path = tmp_path / 'checkpoints' / manager.generation / 'callback_state/memory/alpha.json'
    path.write_text('[]')
    with pytest.raises(ValueError, match='corrupted'):
        CheckpointManager(str(tmp_path)).restore()


def test_new_run_discards_uncommitted_state_and_rejects_legacy(tmp_path):
    write_state(tmp_path, 99)
    assert CheckpointManager(str(tmp_path)).restore() is None
    assert not (tmp_path / 'callback_state').exists()
    (tmp_path / 'runs.json').write_text('[]')
    with pytest.raises(RuntimeError, match='Legacy'):
        CheckpointManager(str(tmp_path)).restore()


def test_output_directory_lock(tmp_path):
    with run_lock(str(tmp_path)):
        with pytest.raises(RuntimeError, match='Another process'):
            with run_lock(str(tmp_path)):
                pass


def test_only_two_committed_generations_retained(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    for step in range(4):
        write_state(tmp_path, step)
        manager.commit([{}] * step)
    assert len(list((tmp_path / 'checkpoints').glob('generation-*'))) == 2
    assert len(CheckpointManager(str(tmp_path)).restore()) == 3
