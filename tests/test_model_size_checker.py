"""Model allow-list metadata."""

import asyncio

from affine.utils.model_size_checker import ModelSizeChecker, _match_allowed_model


# A valid Qwen3-32B fine-tune architecture, reused across tests.
_QWEN3_32B = {
    "model_type": "qwen3",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "intermediate_size": 25600,
    "vocab_size": 151936,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
}


def _check_config(config: dict) -> dict:
    """Drive ModelSizeChecker.check() against an in-memory config."""
    checker = ModelSizeChecker()
    checker._fetch_config = lambda model_id, revision: _async(config)
    return asyncio.run(checker.check("some/model", "rev"))


async def _async(value):
    return value


def test_qwen3_32b_allowlist_entry_matches_dense_config():
    matched_model_type, mismatch = _match_allowed_model({
        "model_type": "qwen3",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "intermediate_size": 25600,
        "vocab_size": 151936,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
    })

    assert mismatch is None
    assert matched_model_type == "qwen3"


def test_qwen3_moe_allowlist_entry_matches_moe_config():
    matched_model_type, mismatch = _match_allowed_model({
        "model_type": "qwen3_5_moe",
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "vocab_size": 248320,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
        },
    })

    assert mismatch is None
    assert matched_model_type == "qwen3_5_moe"


def test_unquantized_allowed_model_passes():
    result = _check_config(dict(_QWEN3_32B))
    assert result["pass"] is True
    assert result["model_type"] == "qwen3"


def test_quantized_allowed_model_rejected():
    config = dict(_QWEN3_32B)
    config["quantization_config"] = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
    }
    result = _check_config(config)
    assert result["pass"] is False
    assert result["reason"] == "quantized_model:bitsandbytes"


def test_quantized_config_without_method_still_rejected():
    config = dict(_QWEN3_32B)
    config["quantization_config"] = {"load_in_8bit": True}
    result = _check_config(config)
    assert result["pass"] is False
    assert result["reason"] == "quantized_model:unknown"
