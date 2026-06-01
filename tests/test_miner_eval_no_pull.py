from types import SimpleNamespace

import pytest

from affine.core.environments import EnvConfig, ENV_CONFIGS
from affine.src.miner import eval as eval_mod


def test_load_environment_pulls_by_default(monkeypatch):
    captured = {}
    monkeypatch.setitem(
        ENV_CONFIGS,
        "unit-memory",
        EnvConfig(name="unit-memory", docker_image="unit-memory:latest"),
    )

    def fake_load_env(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(evaluate=None)

    monkeypatch.setattr("affinetes.load_env", fake_load_env)

    wrapper = eval_mod._load_environment(
        "unit-memory",
        network_host=False,
        basilica=False,
    )

    assert wrapper.env.evaluate is None
    assert captured["image"] == "unit-memory:latest"
    assert captured["pull"] is True


def test_load_environment_can_skip_pull(monkeypatch):
    captured = {}
    monkeypatch.setitem(
        ENV_CONFIGS,
        "unit-memory",
        EnvConfig(name="unit-memory", docker_image="unit-memory:latest"),
    )

    def fake_load_env(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(evaluate=None)

    monkeypatch.setattr("affinetes.load_env", fake_load_env)

    eval_mod._load_environment(
        "unit-memory",
        network_host=False,
        basilica=False,
        pull_image=False,
    )

    assert captured["pull"] is False


@pytest.mark.asyncio
async def test_evaluate_one_preserves_environment_failure():
    class Env:
        async def evaluate(self, **kwargs):
            return {"success": False, "score": 0.0, "error": "invalid env"}

    result = await eval_mod._evaluate_one(
        Env(),
        {"task_id": 7},
        max_retries=0,
    )

    assert result["success"] is False
    assert result["error"] == "invalid env"
    assert result["task_id"] == 7
