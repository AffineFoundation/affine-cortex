"""SDKEnvironment env-var forwarding to env containers.

The post-refactor inference call routes through ``API_KEY``; env images
that predate the Chutes removal still check ``CHUTES_API_KEY`` on the
request path and 500 if it's unset. The forwarder mirrors ``API_KEY``
into ``CHUTES_API_KEY`` as a back-compat alias.

These tests stub ``os.getenv`` directly because ``affine.core.setup``
calls ``load_dotenv(override=True)`` at import time, which prevents
``patch.dict(os.environ, ..., clear=True)`` from giving us a clean
environment.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest


_INSTRUCTION_GYM_JUDGE_ENV = {
    "INSTRUCTION_GYM_JUDGE_BASE_URL": "https://judge.example/v1",
    "INSTRUCTION_GYM_JUDGE_API_KEY": "judge-secret",
    "INSTRUCTION_GYM_JUDGE_ENSEMBLE_JSON": (
        '{"aggregation_mode":"min","members":['
        '{"model_id":"judge-a","model_revision":"revision-a"},'
        '{"model_id":"judge-b","model_revision":"revision-b"}]}'
    ),
    "INSTRUCTION_GYM_APPROVED_SEMANTIC_JUDGE_ENSEMBLE_MANIFEST_SHA256": "a" * 64,
}


def _get_env_vars_with(env_lookup: dict, env_name: str = "affine:ded-v2"):
    """Run ``SDKEnvironment._get_env_vars`` for a known env, substituting
    ``os.getenv`` with a lookup against ``env_lookup``. Bypasses
    ``_load_environment`` (which spawns docker)."""
    from affine.core import environments as env_mod

    def fake_getenv(name, default=None):
        return env_lookup.get(name, default)

    with patch.object(env_mod.SDKEnvironment, "_load_environment", return_value=None):
        sdk = env_mod.SDKEnvironment(env_name)
    with patch.object(env_mod.os, "getenv", side_effect=fake_getenv):
        return sdk._get_env_vars()


def test_aliases_mirror_api_key():
    env_vars = _get_env_vars_with({"API_KEY": "k123"})

    assert env_vars["API_KEY"] == "k123"
    assert env_vars["CHUTES_API_KEY"] == "k123"
    assert env_vars["OPENAI_API_KEY"] == "k123"


def test_explicit_alias_wins_over_api_key_value():
    env_vars = _get_env_vars_with(
        {
            "API_KEY": "k123",
            "CHUTES_API_KEY": "chutes-explicit",
            "OPENAI_API_KEY": "openai-explicit",
        },
    )

    assert env_vars["API_KEY"] == "k123"
    assert env_vars["CHUTES_API_KEY"] == "chutes-explicit"
    assert env_vars["OPENAI_API_KEY"] == "openai-explicit"


def test_no_api_key_set_no_aliases_emitted():
    env_vars = _get_env_vars_with({})

    assert "API_KEY" not in env_vars
    assert "CHUTES_API_KEY" not in env_vars
    assert "OPENAI_API_KEY" not in env_vars


def test_instruction_gym_only_forwards_whitelisted_judge_credentials():
    env_vars = _get_env_vars_with(
        {
            **_INSTRUCTION_GYM_JUDGE_ENV,
            "API_KEY": "global-secret",
            "CHUTES_API_KEY": "chutes-secret",
            "OPENAI_API_KEY": "openai-secret",
            "INSTRUCTION_GYM_JUDGE_MODEL": "legacy-model",
            "INSTRUCTION_GYM_JUDGE_MODEL_REVISION": "legacy-revision",
            "INSTRUCTION_GYM_APPROVED_SEMANTIC_JUDGE_MANIFEST_SHA256": "b" * 64,
        },
        env_name="instruction-gym",
    )

    assert env_vars == {
        **_INSTRUCTION_GYM_JUDGE_ENV,
        "UVICORN_WORKERS": "1",
    }


@pytest.mark.parametrize("missing", tuple(_INSTRUCTION_GYM_JUDGE_ENV))
def test_instruction_gym_requires_complete_judge_ensemble_environment(missing):
    supplied = {
        key: value
        for key, value in _INSTRUCTION_GYM_JUDGE_ENV.items()
        if key != missing
    }

    with pytest.raises(ValueError, match=missing):
        _get_env_vars_with(supplied, env_name="instruction-gym")


def test_liveweb_requires_dashscope_api_key_and_validator_base_url():
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        _get_env_vars_with(
            {
                "API_KEY": "k123",
                "COINGECKO_API_KEY": "cg",
            },
            env_name="liveweb",
        )
    with pytest.raises(ValueError, match="VALIDATOR_BASE_URL"):
        _get_env_vars_with(
            {
                "API_KEY": "k123",
                "COINGECKO_API_KEY": "cg",
                "DASHSCOPE_API_KEY": "dashscope",
            },
            env_name="liveweb",
        )


def test_liveweb_forwards_dashscope_api_key():
    env_vars = _get_env_vars_with(
        {
            "API_KEY": "k123",
            "COINGECKO_API_KEY": "cg",
            "DASHSCOPE_API_KEY": "dashscope",
            "VALIDATOR_BASE_URL": "https://validator.example",
        },
        env_name="liveweb",
    )

    assert env_vars["COINGECKO_API_KEY"] == "cg"
    assert env_vars["DASHSCOPE_API_KEY"] == "dashscope"
    assert env_vars["VALIDATOR_BASE_URL"] == "https://validator.example"


def test_build_result_removes_all_base_url_fields():
    from affine.core import environments as env_mod

    with patch.object(env_mod.SDKEnvironment, "_load_environment", return_value=None):
        sdk = env_mod.SDKEnvironment("terminal")

    miner = SimpleNamespace(hotkey="hk", revision="rev")
    result = sdk._build_result(
        {
            "score": 1.0,
            "success": True,
            "extra": {
                "base_url": "http://10.0.0.1:8000/v1",
                "baseUrl": "http://10.0.0.2:8000/v1",
                "API_KEY": "environment-secret",
                "nested": {
                    "base-url": "http://10.0.0.3:8000/v1",
                    "headers": {
                        "Authorization": "Bearer nested-secret",
                        "X-API-Key": "nested-api-secret",
                    },
                    "ok": True,
                },
            },
        },
        miner,
        {
            "task_id": 123,
            "model": "org/model",
            "base_url": "http://10.0.0.4:8000/v1",
            "api_key": "request-secret",
        },
        time.monotonic(),
    )

    payload = json.dumps(result.extra)
    assert "base_url" not in result.extra
    assert "baseUrl" not in result.extra
    assert "base-url" not in result.extra["nested"]
    assert "headers" in result.extra["nested"]
    assert result.extra["nested"]["headers"] == {}
    assert "base_url" not in result.extra["request"]
    assert "api_key" not in result.extra["request"]
    assert "10.0.0.1" not in payload
    assert "10.0.0.2" not in payload
    assert "10.0.0.3" not in payload
    assert "10.0.0.4" not in payload
    assert "environment-secret" not in payload
    assert "nested-secret" not in payload
    assert "nested-api-secret" not in payload
    assert "request-secret" not in payload
    assert result.extra["nested"]["ok"] is True
    assert result.extra["request"]["model"] == "org/model"
