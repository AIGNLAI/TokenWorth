"""Verify the files that define the fixed benchmark protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data/manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record_count(path: Path, expected_count: int) -> None:
    content: Any = json.loads(path.read_text())
    if not isinstance(content, (dict, list)) or len(content) != expected_count:
        actual_count = len(content) if isinstance(content, (dict, list)) else "invalid"
        raise RuntimeError(
            f"{path.relative_to(PROJECT_ROOT)}: expected {expected_count} records, "
            f"found {actual_count}"
        )


def verify() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    errors: list[str] = []
    for relative_path, metadata in manifest["files"].items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            errors.append(f"missing: {relative_path}")
            continue
        actual_hash = sha256(path)
        if actual_hash != metadata["sha256"]:
            errors.append(
                f"checksum mismatch: {relative_path} "
                f"(expected {metadata['sha256']}, found {actual_hash})"
            )
        if "records" in metadata:
            try:
                validate_record_count(path, metadata["records"])
            except (json.JSONDecodeError, RuntimeError) as error:
                errors.append(str(error))
    if errors:
        raise SystemExit("Data verification failed:\n- " + "\n- ".join(errors))
    print(
        f"Data verification passed: {len(manifest['files'])} files match "
        f"protocol {manifest['protocol_version']}."
    )


def main() -> None:
    verify()


if __name__ == "__main__":
    main()
