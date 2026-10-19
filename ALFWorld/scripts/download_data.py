#!/usr/bin/env python3
"""Download and verify the ALFWorld text-environment data used by the release."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str
    size: int
    entries: int


ASSETS = (
    Asset(
        name="json_2.1.1_json.zip",
        url=(
            "https://github.com/alfworld/alfworld/releases/download/0.2.2/"
            "json_2.1.1_json.zip"
        ),
        sha256=("25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc"),
        size=72_018_818,
        entries=7_302,
    ),
    Asset(
        name="json_2.1.1_pddl.zip",
        url=(
            "https://github.com/alfworld/alfworld/releases/download/0.2.2/"
            "json_2.1.1_pddl.zip"
        ),
        sha256=("913942ebed06659ea0da2f8122512d98bc6add30d84961ca803132d8fbcad585"),
        size=34_881_784,
        entries=7_080,
    ),
    Asset(
        name="json_2.1.3_tw-pddl.zip",
        url=(
            "https://github.com/alfworld/alfworld/releases/download/0.4.2/"
            "json_2.1.3_tw-pddl.zip"
        ),
        sha256=("5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf"),
        size=36_507_267,
        entries=4_027,
    ),
)

EXPECTED_GAME_COUNTS = {
    "train": 3_553,
    "valid_train": 200,
    "valid_seen": 140,
    "valid_unseen": 134,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset: Asset, archive_dir: Path) -> Path:
    destination = archive_dir / asset.name
    if destination.is_file():
        if (
            destination.stat().st_size == asset.size
            and file_sha256(destination) == asset.sha256
        ):
            print(f"Verified cached asset: {destination}")
            return destination
        raise RuntimeError(
            f"Existing asset failed verification: {destination}. Remove it and retry."
        )

    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        offset = temporary.stat().st_size if temporary.is_file() else 0
        if offset >= asset.size:
            if offset == asset.size and file_sha256(temporary) == asset.sha256:
                os.replace(temporary, destination)
                return destination
            temporary.unlink()
            offset = 0
        headers = {"User-Agent": "TokenWorth"}
        if offset:
            headers["Range"] = "bytes={}-".format(offset)
        request = urllib.request.Request(asset.url, headers=headers)
        try:
            print(
                f"Downloading {asset.url} (attempt {attempt}/3, offset={offset})",
                flush=True,
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if (
                        not match
                        or int(match[1]) != offset
                        or int(match[3]) != asset.size
                        or not offset <= int(match[2]) < asset.size
                    ):
                        raise RuntimeError("Invalid range response for " + asset.name)
                    mode = "ab"
                elif response.status == 200:
                    # Some mirrors ignore Range; safely restart rather than
                    # concatenate the start of an archive onto its partial body.
                    mode = "wb"
                else:
                    raise RuntimeError("Unexpected download response status")
                with temporary.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            size = temporary.stat().st_size
            if size != asset.size:
                raise RuntimeError(
                    f"Size mismatch for {asset.name}: expected {asset.size}, got {size}"
                )
            if file_sha256(temporary) != asset.sha256:
                temporary.unlink()
                raise RuntimeError(f"Checksum mismatch for {asset.name}")
            os.replace(temporary, destination)
            return destination
        except (OSError, RuntimeError) as error:
            # Preserve a bounded partial file for verified range retries.
            # Extraction always requires final size AND pinned SHA-256.
            if temporary.is_file() and temporary.stat().st_size > asset.size:
                temporary.unlink()
            if attempt == 3:
                raise RuntimeError(
                    f"Failed to download and verify {asset.name} after 3 attempts; partial data is preserved for retry"
                ) from error
            print(f"Download failed for {asset.name}: {error}; retrying", flush=True)
            time.sleep(attempt)
    raise AssertionError("unreachable")


def extract_checked(asset: Asset, archive: Path, data_root: Path) -> None:
    root = data_root.resolve()
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) != asset.entries:
            raise RuntimeError(
                f"Unexpected entry count for {asset.name}: expected "
                f"{asset.entries}, got {len(members)}"
            )
        for member in members:
            destination = (root / member.filename).resolve()
            if os.path.commonpath((root, destination)) != str(root):
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        print(f"Extracting {archive} into {data_root}")
        zipped.extractall(data_root)


def verify_games(data_root: Path) -> None:
    dataset_root = data_root / "json_2.1.1"
    counts = Counter(
        path.relative_to(dataset_root).parts[0]
        for path in dataset_root.glob("*/*/*/game.tw-pddl")
    )
    if dict(counts) != EXPECTED_GAME_COUNTS:
        raise RuntimeError(
            f"Unexpected ALFWorld game counts: expected {EXPECTED_GAME_COUNTS}, "
            f"got {dict(counts)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="ALFWorld data root to create",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="verified archive cache (default: DATA_DIR/.downloads)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_dir.expanduser().resolve()
    archive_dir = (args.archive_dir or data_root / ".downloads").expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    for asset in ASSETS:
        archive = download(asset, archive_dir)
        extract_checked(asset, archive, data_root)

    logic_dir = data_root / "logic"
    logic_dir.mkdir(exist_ok=True)
    shutil.copy2(ALFRED_PDDL_PATH, logic_dir / "alfred.pddl")
    shutil.copy2(ALFRED_TWL2_PATH, logic_dir / "alfred.twl2")
    verify_games(data_root)
    print(f"Verified ALFWorld data root: {data_root}")
    print(f"ALFWORLD_DATA_ROOT={data_root}")


if __name__ == "__main__":
    main()
