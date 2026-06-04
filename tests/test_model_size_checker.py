"""Model allow-list metadata."""

from affine.utils.model_size_checker import _match_allowed_model


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
