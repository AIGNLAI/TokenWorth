"""Transactional checkpoints: CURRENT is the only authoritative commit marker."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from contextlib import contextmanager
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def run_lock(output_dir: str):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is running in {root}.') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class CheckpointManager:
    def __init__(self, output_dir: str):
        self.root = Path(output_dir)
        self.generations = self.root / 'checkpoints'
        self.current = self.generations / 'CURRENT'
        self.generation: str | None = None

    def peek_sessions(self) -> list[dict[str, Any]] | None:
        """Validate and read the committed generation without changing disk state."""
        if not self.current.exists():
            if (self.root / 'runs.json').exists():
                raise RuntimeError('Legacy run has no atomic checkpoint; use a new output directory.')
            return None
        pointer = json.loads(self.current.read_text())
        name = pointer['generation']
        if not isinstance(name, str) or not name.startswith('generation-') or Path(name).name != name:
            raise ValueError('Invalid checkpoint generation name.')
        committed = self.generations / name
        manifest_path = committed / 'manifest.json'
        if file_hash(manifest_path) != pointer['manifest_sha256']:
            raise ValueError('Checkpoint manifest checksum mismatch.')
        manifest = json.loads(manifest_path.read_text())
        if any(path.is_symlink() for path in committed.rglob('*')):
            raise ValueError('Checkpoint state cannot contain symlinks.')
        actual = {str(path.relative_to(committed)): file_hash(path)
                  for path in committed.rglob('*') if path.is_file() and path != manifest_path}
        if actual != manifest['files']:
            raise ValueError('Checkpoint files are missing, unexpected, or corrupted.')
        sessions = json.loads((committed / 'runs.json').read_text())
        if not isinstance(sessions, list) or len(sessions) != manifest['completed_sample_count']:
            raise ValueError('Checkpoint session count mismatch.')
        self.generation = name
        return sessions

    def restore(self) -> list[dict[str, Any]] | None:
        sessions = self.peek_sessions()
        # Working state can contain partially overwritten files from an interrupted save.
        shutil.rmtree(self.root / 'callback_state', ignore_errors=True)
        if sessions is None:
            return None
        committed = self.generations / self.generation
        if (committed / 'callback_state').exists():
            shutil.copytree(committed / 'callback_state', self.root / 'callback_state')
        atomic_json(self.root / 'runs.json', sessions)
        return sessions

    def commit(self, sessions: list[dict[str, Any]]) -> None:
        self.generations.mkdir(parents=True, exist_ok=True)
        previous_name = json.loads(self.current.read_text())['generation'] if self.current.exists() else None
        name = f'generation-{len(sessions):06d}-{uuid.uuid4().hex}'
        staging = self.generations / f'.{name}.pending'
        staging.mkdir()
        try:
            atomic_json(staging / 'runs.json', sessions)
            if (self.root / 'callback_state').exists():
                shutil.copytree(self.root / 'callback_state', staging / 'callback_state')
            files = {}
            for path in staging.rglob('*'):
                if path.is_symlink():
                    raise ValueError('Checkpoint state cannot contain symlinks.')
                if path.is_file():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
                    files[str(path.relative_to(staging))] = file_hash(path)
            atomic_json(staging / 'manifest.json', {
                'schema_version': 1, 'completed_sample_count': len(sessions), 'files': files,
            })
            for directory in sorted((p for p in staging.rglob('*') if p.is_dir()), reverse=True):
                fsync_dir(directory)
            fsync_dir(staging)
            committed = self.generations / name
            os.replace(staging, committed)
            fsync_dir(self.generations)
            atomic_json(self.current, {'generation': name, 'manifest_sha256': file_hash(committed / 'manifest.json')})
            self.generation = name
            atomic_json(self.root / 'runs.json', sessions)
            # Keep two committed generations. Orphans are never restored.
            for stale in self.generations.glob('generation-*'):
                if stale.name not in {name, previous_name}:
                    shutil.rmtree(stale)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
