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

from unittest.mock import patch


def _get_env_vars_with(env_lookup: dict):
    """Run ``SDKEnvironment._get_env_vars`` for a known env, substituting
    ``os.getenv`` with a lookup against ``env_lookup``. Bypasses
    ``_load_environment`` (which spawns docker)."""
    from affine.core import environments as env_mod

    def fake_getenv(name, default=None):
        return env_lookup.get(name, default)

    with patch.object(env_mod.SDKEnvironment, "_load_environment", return_value=None):
        sdk = env_mod.SDKEnvironment("affine:ded-v2")
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
