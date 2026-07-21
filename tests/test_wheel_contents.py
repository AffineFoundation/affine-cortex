"""Regression coverage for the affine-cortex wheel release check."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_wheel_contents.py"
)
SPEC = importlib.util.spec_from_file_location("check_wheel_contents", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_wheel(path: Path, members) -> None:
    with ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, b"")


def test_wheel_check_accepts_complete_runtime_graph(tmp_path):
    wheel = tmp_path / "affine_io-0.1.1-py3-none-any.whl"
    _write_wheel(wheel, MODULE.REQUIRED_MEMBERS)

    MODULE.check_wheel(wheel)


def test_wheel_check_rejects_removed_instruction_gym_runtime_files(tmp_path):
    wheel = tmp_path / "affine_io-0.1.1-py3-none-any.whl"
    removed = next(iter(MODULE.REMOVED_MEMBERS))
    _write_wheel(wheel, MODULE.REQUIRED_MEMBERS | {removed})

    with pytest.raises(ValueError, match="removed runtime files"):
        MODULE.check_wheel(wheel)


def test_wheel_check_reports_missing_runtime_subpackage(tmp_path):
    wheel = tmp_path / "affine_io-0.1.1-py3-none-any.whl"
    members = MODULE.REQUIRED_MEMBERS - {"affine/src/scorer/sampler.py"}
    _write_wheel(wheel, members)

    with pytest.raises(ValueError, match="affine/src/scorer/sampler.py"):
        MODULE.check_wheel(wheel)


def test_wheel_check_requires_the_runtime_config_and_eval_only_guards(tmp_path):
    wheel = tmp_path / "affine_io-0.1.1-py3-none-any.whl"
    for required in (
        "affine/database/system_config.json",
        "affine/src/miner/eval.py",
        "affine/src/teacher/worker.py",
    ):
        _write_wheel(wheel, MODULE.REQUIRED_MEMBERS - {required})
        with pytest.raises(ValueError, match=required):
            MODULE.check_wheel(wheel)


def test_wheel_check_rejects_tests_and_scripts(tmp_path):
    wheel = tmp_path / "affine_io-0.1.1-py3-none-any.whl"
    _write_wheel(wheel, MODULE.REQUIRED_MEMBERS | {"tests/test_example.py"})

    with pytest.raises(ValueError, match="tests/test_example.py"):
        MODULE.check_wheel(wheel)
