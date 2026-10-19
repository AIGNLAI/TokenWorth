"""Durable identity and request accounting shared by appendix adapters."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_event(path, value):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def source_identity(adapter_root):
    adapter_root = Path(adapter_root)
    core_root = Path(__file__).resolve().parent
    paths = list(core_root.glob("*.py"))
    for directory in ("experiments", "continual_alfworld"):
        paths += list((adapter_root / directory).rglob("*.py"))
    sources = {
        (
            str(p.relative_to(adapter_root))
            if p.is_relative_to(adapter_root)
            else "tokenworth_core/" + p.name
        ): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    versions = {}
    for name in (
        "tokenworth-core",
        "numpy",
        "openai",
        "alfworld",
        "textworld",
        "PyYAML",
        "bigcodebench",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "sources": sources,
        "versions": versions,
        "python": platform.python_version(),
    }


def redacted_config(value):
    if isinstance(value, dict):
        return {
            k: redacted_config(v)
            for k, v in value.items()
            if k.lower() not in {"api_key", "output_dir", "output_root", "resume"}
        }
    if isinstance(value, list):
        return [redacted_config(v) for v in value]
    return value


class RunGuard:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity_path = self.directory / "run_identity.json"
        self.active_path = self.directory / "active_task.json"
        expected = {
            "version": "appendix-paper-v2",
            "fingerprint": fingerprint(identity),
            "identity": identity,
        }
        if self.identity_path.exists():
            if json.loads(self.identity_path.read_text()) != expected:
                raise RuntimeError("Run identity changed; use a new output directory")
        else:
            # Never adopt old states/results whose algorithm and cost contract
            # cannot be verified retroactively.
            if any(
                (self.directory / name).exists()
                for name in (
                    "method_state.json",
                    "results.jsonl",
                    "trajectories.jsonl",
                    "checkpoint.json",
                    "request_events.jsonl",
                    "active_task.json",
                    "pending_task.json",
                )
            ):
                raise RuntimeError(
                    "Legacy run has no frozen identity; use a new output directory"
                )
            atomic_json(self.identity_path, expected)

    def recover(self, completed_keys):
        if self.active_path.exists():
            active = json.loads(self.active_path.read_text())
            if active["key"] not in set(completed_keys):
                raise RuntimeError(
                    "Uncommitted task may have consumed model tokens; replay refused, use a new output directory"
                )
            self.active_path.unlink()

    def start(self, key):
        if self.active_path.exists():
            raise RuntimeError("An active task already exists")
        atomic_json(self.active_path, {"key": key})

    def finish(self, key):
        active = json.loads(self.active_path.read_text())
        if active["key"] != key:
            raise RuntimeError("Active task identity mismatch")
        self.active_path.unlink()


class CheckpointStore:
    """Atomic learner state plus references to immutable evaluated records."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "checkpoint.json"
        self.records = []
        self.references = []
        self.method = None
        if self.path.exists():
            checkpoint = json.loads(self.path.read_text(encoding="utf-8"))
            if checkpoint.get("version") != "appendix-paper-v2":
                raise RuntimeError("Incompatible appendix checkpoint")
            self.method = checkpoint["method"]
            self.references = checkpoint["records"]
            for index, reference in enumerate(self.references):
                expected = "records/{:06d}-{}.json".format(
                    index + 1, reference["sha256"]
                )
                if reference["path"] != expected:
                    raise RuntimeError("Invalid checkpoint record reference")
                data = (self.directory / expected).read_bytes()
                if hashlib.sha256(data).hexdigest() != reference["sha256"]:
                    raise RuntimeError("Checkpoint result checksum mismatch")
                self.records.append(json.loads(data))

    def commit(self, method, record):
        data = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        relative = "records/{:06d}-{}.json".format(len(self.records) + 1, digest)
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, record)
        references = [*self.references, {"path": relative, "sha256": digest}]
        atomic_json(
            self.path,
            {"version": "appendix-paper-v2", "method": method, "records": references},
        )
        self.references = references
        self.records.append(record)
        self.method = method

    def export_jsonl(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
