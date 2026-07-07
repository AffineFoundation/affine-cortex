from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_launcher():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / ("gpu_autoscale_wrapper.py")
    )
    module_name = f"_gpu_autoscale_wrapper_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_default_provider_starts_all_wrappers(monkeypatch):
    launcher = _load_launcher()
    monkeypatch.delenv(launcher.PROVIDER_ENV, raising=False)

    assert launcher.selected_providers() == ("lium", "targon")
    assert launcher.wrapper_script("lium").name == "lium_autoscale_wrapper.py"
    assert launcher.wrapper_script("targon").name == "targon_autoscale_wrapper.py"


def test_selects_lium_provider(monkeypatch):
    launcher = _load_launcher()
    monkeypatch.setenv(launcher.PROVIDER_ENV, " LIUM ")

    assert launcher.selected_providers() == ("lium",)
    assert launcher.wrapper_script("lium").name == "lium_autoscale_wrapper.py"


def test_selects_multiple_providers(monkeypatch):
    launcher = _load_launcher()
    monkeypatch.setenv(launcher.PROVIDER_ENV, "targon, lium, targon")

    assert launcher.selected_providers() == ("targon", "lium")


def test_invalid_provider_is_rejected(monkeypatch):
    launcher = _load_launcher()

    with pytest.raises(ValueError, match="must be one of"):
        launcher.wrapper_script("unknown")

    monkeypatch.setenv(launcher.PROVIDER_ENV, "unknown")
    with pytest.raises(ValueError, match="must be one of"):
        launcher.selected_providers()


def test_main_execs_selected_wrapper(monkeypatch):
    launcher = _load_launcher()
    calls = []

    def fake_execv(executable, argv):
        calls.append((executable, argv))
        raise SystemExit(0)

    monkeypatch.setenv(launcher.PROVIDER_ENV, "lium")
    monkeypatch.setattr(launcher.os, "execv", fake_execv)

    with pytest.raises(SystemExit):
        launcher.main()

    assert calls
    assert calls[0][0] == sys.executable
    assert calls[0][1][1].endswith("lium_autoscale_wrapper.py")


def test_main_runs_all_wrappers_by_default(monkeypatch):
    launcher = _load_launcher()
    calls = []

    def fake_run_many(providers):
        calls.append(providers)
        return 0

    monkeypatch.delenv(launcher.PROVIDER_ENV, raising=False)
    monkeypatch.setattr(launcher, "run_many", fake_run_many)

    with pytest.raises(SystemExit) as exc:
        launcher.main()

    assert exc.value.code == 0
    assert calls == [("lium", "targon")]
