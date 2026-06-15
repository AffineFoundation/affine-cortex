"""Model allow-list metadata."""

import asyncio

from affine.utils.model_size_checker import (
    ModelSizeChecker,
    _WeightShapeCheckUnavailable,
    _match_allowed_model,
    _select_vocab_tensor_names,
)


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
    checker._check_safetensors_weight_shapes = (
        lambda model_id, revision, cfg: _async(None)
    )
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


def test_safetensors_vocab_shape_mismatch_rejected():
    checker = ModelSizeChecker()
    checker._fetch_config = lambda model_id, revision: _async(dict(_QWEN3_32B))
    checker._fetch_safetensors_index_sync = lambda model_id, revision: {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00014.safetensors",
            "lm_head.weight": "model-00014-of-00014.safetensors",
        }
    }
    checker._fetch_safetensors_header_sync = lambda model_id, revision, filename: {
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [151669, 5120],
            "data_offsets": [0, 0],
        },
        "lm_head.weight": {
            "dtype": "BF16",
            "shape": [151669, 5120],
            "data_offsets": [0, 0],
        },
    }

    result = asyncio.run(checker.check("some/model", "rev-shape-mismatch"))

    assert result["pass"] is False
    assert result["reason"] == (
        "weight_shape_mismatch:model.embed_tokens.weight.dim0=151669 "
        "expected_vocab=151936"
    )
    assert result["model_type"] == "qwen3"


def test_safetensors_shape_check_unavailable_does_not_reject():
    checker = ModelSizeChecker()
    checker._fetch_config = lambda model_id, revision: _async(dict(_QWEN3_32B))
    checker._fetch_safetensors_index_sync = lambda model_id, revision: None

    def _raise(*args, **kwargs):
        raise _WeightShapeCheckUnavailable("network timeout")

    checker._fetch_safetensors_header_sync = _raise

    result = asyncio.run(checker.check("some/model", "rev-unavailable"))

    assert result["pass"] is True
    assert result["reason"] == "ok"


def test_select_vocab_tensor_names_prefers_known_language_tensors():
    names = {
        "vision_tower.embed_tokens.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    }

    assert _select_vocab_tensor_names(names) == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
