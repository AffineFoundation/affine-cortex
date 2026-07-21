"""Local ``af eval`` argument, lifecycle, and result-contract tests."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from affine.src.miner import eval as eval_mod


class _FakeRemoteEnvironment:
    def __init__(self, result=None):
        self.result = result or {"score": 1.0, "success": True}
        self.evaluate_calls = []
        self.cleanup_calls = 0

    async def evaluate(self, **kwargs):
        self.evaluate_calls.append(kwargs)
        return self.result

    async def cleanup(self):
        self.cleanup_calls += 1


def test_eval_cli_forwards_local_image_seed_and_explicit_api_key(monkeypatch):
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(eval_mod, "_run", fake_run)

    result = CliRunner().invoke(
        eval_mod.eval_cmd,
        [
            "--env",
            "Instruction-Gym",
            "--base-url",
            "http://model.local/v1",
            "--model",
            "org/model",
            "--image",
            "instruction-gym:dev",
            "--task-id",
            "42",
            "--seed",
            "1234",
            "--api-key",
            "per-call-secret",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["env"] == "instruction-gym"
    assert captured["image"] == "instruction-gym:dev"
    assert captured["seed"] == 1234
    assert captured["api_key"] == "per-call-secret"


def test_load_environment_uses_override_without_pull_and_host_network(
    monkeypatch,
):
    captured = {}
    remote = _FakeRemoteEnvironment()

    def fake_load_env(**kwargs):
        captured.update(kwargs)
        return remote

    monkeypatch.setitem(
        sys.modules,
        "affinetes",
        SimpleNamespace(load_env=fake_load_env),
    )
    monkeypatch.setenv("API_KEY", "global-secret")
    monkeypatch.setenv("CHUTES_API_KEY", "chutes-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    wrapped = eval_mod._load_environment(
        "instruction-gym",
        network_host=True,
        basilica=False,
        image_override="instruction-gym:dev",
    )

    assert captured["image"] == "instruction-gym:dev"
    assert captured["pull"] is False
    assert captured["host_network"] is True
    assert "network_mode" not in captured
    assert captured["env_vars"] == {"UVICORN_WORKERS": "1"}

    asyncio.run(
        wrapped.evaluate(
            task_id=42,
            model="org/model",
            base_url="http://model.local/v1",
            seed=7,
            api_key="per-call-secret",
        )
    )
    request = remote.evaluate_calls[0]
    assert request["_timeout"] == 660
    assert request["protocol_version"] == "1.0"
    assert request["suite_id"] == "instruction_gym_ifeval_template_tasks_v1"
    assert request["universe_id"].startswith("ifeval_template_tasks_v1:")
    assert request["api_key"] == "per-call-secret"


def test_load_environment_rejects_untrusted_basilica_before_affinetes_load(
    monkeypatch,
):
    load_calls = []

    monkeypatch.setitem(
        sys.modules,
        "affinetes",
        SimpleNamespace(load_env=lambda **kwargs: load_calls.append(kwargs)),
    )
    monkeypatch.setenv("BASILICA_API_TOKEN", "test-only-token")

    with pytest.raises(ValueError, match="refuses untrusted execution mode 'basilica'"):
        eval_mod._load_environment(
            "instruction-gym",
            network_host=False,
            basilica=True,
        )

    assert load_calls == []


def test_evaluate_one_preserves_environment_failure_and_redacts_nested_secrets():
    original = {
        "score": 0.0,
        "success": False,
        "error": "case_materialization_failed",
        "extra": {
            "request": {
                "base_url": "http://private-model/v1",
                "api_key": "request-secret",
                "headers": {"Authorization": "Bearer header-secret"},
                "task_id": 9,
            }
        },
    }
    env = SimpleNamespace(evaluate=lambda **_kwargs: None)

    async def evaluate(**_kwargs):
        return original

    env.evaluate = evaluate
    wrapped = SimpleNamespace(evaluate=env.evaluate)

    result = asyncio.run(
        eval_mod._evaluate_one(
            wrapped,
            {"task_id": 9, "api_key": "caller-secret"},
            max_retries=0,
        )
    )

    assert result["success"] is False
    assert result["error"] == "case_materialization_failed"
    assert result["score"] == 0.0
    assert result["extra"]["request"] == {
        "headers": {},
        "task_id": 9,
    }
    assert "latency_seconds" not in original
    assert original["extra"]["request"]["api_key"] == "request-secret"


def test_evaluate_one_never_persists_exception_text_or_call_secrets():
    async def fail(**_kwargs):
        raise RuntimeError("upstream rejected sk-release-secret")

    result = asyncio.run(
        eval_mod._evaluate_one(
            SimpleNamespace(evaluate=fail),
            {"task_id": 9, "api_key": "sk-release-secret"},
            max_retries=0,
        )
    )

    assert result["success"] is False
    assert result["error"] == "RuntimeError: evaluation_failed"
    assert "sk-release-secret" not in str(result)


def test_evaluate_range_forwards_seed_and_explicit_api_key():
    calls = []

    async def evaluate(**kwargs):
        calls.append(kwargs)
        return {"score": 1.0, "success": True}

    wrapped = SimpleNamespace(evaluate=evaluate)
    results = asyncio.run(
        eval_mod._evaluate_range(
            wrapped,
            "http://model.local/v1",
            "org/model",
            0.0,
            0,
            5,
            7,
            0.0,
            seed=55,
            api_key="per-call-secret",
        )
    )

    assert [call["task_id"] for call in calls] == [5, 6]
    assert all(call["seed"] == 55 for call in calls)
    assert all(call["api_key"] == "per-call-secret" for call in calls)
    assert all(result["success"] is True for result in results)


def test_run_passes_seed_and_api_key_then_cleans_up(monkeypatch):
    remote = _FakeRemoteEnvironment()
    wrapper = SimpleNamespace(env=remote)
    captured = {}

    def fake_load_environment(env_name, **kwargs):
        captured["load"] = (env_name, kwargs)
        return wrapper

    async def fake_evaluate_samples(*args, **kwargs):
        captured["evaluate"] = (args, kwargs)
        return [{"score": 0.0, "success": False, "error": "expected"}]

    def fake_write_summary(*args):
        captured["summary"] = args

    monkeypatch.setattr(eval_mod, "_load_environment", fake_load_environment)
    monkeypatch.setattr(eval_mod, "_evaluate_samples", fake_evaluate_samples)
    monkeypatch.setattr(eval_mod, "_write_summary", fake_write_summary)

    asyncio.run(
        eval_mod._run(
            env="instruction-gym",
            base_url="http://model.local/v1",
            model="org/model",
            image="instruction-gym:dev",
            samples=1,
            task_id=42,
            task_id_range=None,
            temperature=0.0,
            seed=9876,
            api_key="per-call-secret",
            network_host=True,
            output="result.json",
            basilica=False,
            delay=0.0,
            max_retries=0,
        )
    )

    assert captured["load"] == (
        "instruction-gym",
        {
            "network_host": True,
            "basilica": False,
            "image_override": "instruction-gym:dev",
        },
    )
    assert captured["evaluate"][1] == {
        "seed": 9876,
        "api_key": "per-call-secret",
    }
    assert remote.cleanup_calls == 1
    assert captured["summary"][5][0]["success"] is False
    assert captured["summary"][5][0]["error"] == "expected"


def test_run_cleans_up_when_evaluation_raises(monkeypatch):
    remote = _FakeRemoteEnvironment()
    wrapper = SimpleNamespace(env=remote)

    monkeypatch.setattr(
        eval_mod, "_load_environment", lambda *_args, **_kwargs: wrapper
    )

    async def fail(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(eval_mod, "_evaluate_samples", fail)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            eval_mod._run(
                env="instruction-gym",
                base_url="http://model.local/v1",
                model="org/model",
                image=None,
                samples=1,
                task_id=42,
                task_id_range=None,
                temperature=0.0,
                seed=None,
                api_key=None,
                network_host=False,
                output=None,
                basilica=False,
                delay=0.0,
                max_retries=0,
            )
        )

    assert remote.cleanup_calls == 1
