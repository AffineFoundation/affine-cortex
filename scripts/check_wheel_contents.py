#!/usr/bin/env python3
"""Fail unless an affine-cortex wheel contains its runtime package graph."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import BadZipFile, ZipFile


REQUIRED_MEMBERS = frozenset(
    {
        "affine/core/__init__.py",
        "affine/core/environments.py",
        "affine/src/scheduler/__init__.py",
        "affine/src/scheduler/flow.py",
        "affine/src/miner/__init__.py",
        "affine/src/miner/eval.py",
        "affine/src/scorer/__init__.py",
        "affine/src/scorer/sampler.py",
        "affine/src/scorer/window_state.py",
        "affine/src/teacher/__init__.py",
        "affine/src/teacher/worker.py",
        "affine/database/__init__.py",
        "affine/database/system_config.json",
        "affine/database/dao/__init__.py",
    }
)
REMOVED_MEMBERS = frozenset(
    {
        "affine/core/instruction_gym_sampling.py",
        "affine/core/data/__init__.py",
        "affine/core/data/instruction_gym_sampling_manifest_v1.json",
    }
)
FORBIDDEN_PREFIXES = ("tests/", "scripts/")


def check_wheel(path: str | Path) -> None:
    """Raise ``ValueError`` when a built wheel is missing runtime files."""

    wheel_path = Path(path)
    if not wheel_path.is_file() or wheel_path.suffix != ".whl":
        raise ValueError(f"wheel does not exist: {wheel_path}")
    try:
        with ZipFile(wheel_path) as archive:
            members = frozenset(archive.namelist())
    except BadZipFile as exc:
        raise ValueError(f"invalid wheel archive: {wheel_path}") from exc
    missing = sorted(REQUIRED_MEMBERS - members)
    if missing:
        raise ValueError(f"wheel is missing runtime files: {', '.join(missing)}")
    removed = sorted(REMOVED_MEMBERS & members)
    if removed:
        raise ValueError(f"wheel contains removed runtime files: {', '.join(removed)}")
    forbidden = sorted(
        member for member in members if member.startswith(FORBIDDEN_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"wheel contains non-runtime files: {', '.join(forbidden)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    check_wheel(args.wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
