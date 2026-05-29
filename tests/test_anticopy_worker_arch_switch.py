"""Architecture-compatibility check that gates the ``update_weights``
vs full-restart decision in ``_ensure_sglang_running``.
"""

from affine.src.anticopy.worker import _arch_compatible


QWEN3_32B = {
    "model_type": "qwen3",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "vocab_size": 151936,
}

QWEN3_5_35B_MOE = {
    "model_type": "qwen3_5_moe",
    "text_config": {
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "vocab_size": 248320,
    },
}


def test_same_arch_is_compatible():
    # Two different fine-tunes of the same Qwen3-32B base — the bytes
    # differ but model_type matches, so update_weights_from_disk works.
    assert _arch_compatible(QWEN3_32B, QWEN3_32B)


def test_dense_to_moe_is_incompatible():
    # qwen3 (dense, vocab 151936) vs qwen3_5_moe (MoE, vocab 248320)
    # cannot share weight buffers — must kill+restart.
    assert not _arch_compatible(QWEN3_32B, QWEN3_5_35B_MOE)
    # Symmetric.
    assert not _arch_compatible(QWEN3_5_35B_MOE, QWEN3_32B)


def test_missing_current_config_is_optimistic():
    # If we can't read the running engine's config (network blip, file
    # gone), don't trigger a cold-start storm — the update_weights POST
    # will fail loudly later if there's a real mismatch.
    assert _arch_compatible(None, QWEN3_32B)
    assert _arch_compatible({}, QWEN3_32B)


def test_missing_new_config_is_optimistic():
    # Likewise on the candidate side: a half-downloaded snapshot
    # shouldn't force a 5-min restart on a transient race.
    assert _arch_compatible(QWEN3_32B, None)
    assert _arch_compatible(QWEN3_32B, {})


def test_missing_model_type_field_is_optimistic():
    # Pathological config without model_type — treat as unknown rather
    # than mismatch.
    assert _arch_compatible({"hidden_size": 5120}, QWEN3_32B)
    assert _arch_compatible(QWEN3_32B, {"hidden_size": 5120})
