"""Anti-copy worker's architecture allow-list — gated at job-claim
time so unsupported model types never reach the 60 GB weight download
or the sglang swap path.
"""

import json
from unittest.mock import patch

from affine.src.anticopy import worker as worker_module
from affine.src.anticopy.worker import (
    ALLOWED_MODEL_TYPES,
    _fetch_hf_model_type,
)


def test_default_allowlist_is_qwen36_only():
    # Pinned to current submission policy. Operators can still override
    # ANTICOPY_ALLOWED_MODEL_TYPES during emergency rollback.
    assert ALLOWED_MODEL_TYPES == {"qwen3_5_moe"}


def test_fetch_hf_model_type_returns_field(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_type": "qwen3", "hidden_size": 5120}))

    with patch.object(
        worker_module, "hf_hub_download", return_value=str(cfg),
    ):
        assert _fetch_hf_model_type("any/repo", "rev") == "qwen3"


def test_fetch_hf_model_type_handles_moe_field(tmp_path):
    # Qwen3.6-35B-A3B's flat config has model_type at the top level
    # even though the language-model fields live under text_config.
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "text_config": {"vocab_size": 248320},
    }))
    with patch.object(
        worker_module, "hf_hub_download", return_value=str(cfg),
    ):
        assert _fetch_hf_model_type("any/repo", "rev") == "qwen3_5_moe"


def test_fetch_hf_model_type_missing_field_returns_none(tmp_path):
    # Pathological config without model_type — let the caller decide
    # (current policy: fall back to "let it through").
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"hidden_size": 5120}))
    with patch.object(
        worker_module, "hf_hub_download", return_value=str(cfg),
    ):
        assert _fetch_hf_model_type("any/repo", "rev") is None


def test_fetch_hf_model_type_swallows_errors():
    # HF blip / 404 / auth fail — caller treats None as "let it through"
    # so we don't silently drop real candidates due to a transient.
    with patch.object(
        worker_module, "hf_hub_download",
        side_effect=Exception("network blip"),
    ):
        assert _fetch_hf_model_type("any/repo", "rev") is None
