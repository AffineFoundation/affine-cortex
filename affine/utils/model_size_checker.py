"""
Model Size Checker

Estimates model parameter count from HuggingFace config.json architecture fields.
Rejects models with >120B parameters or unknown architectures (fail-closed).

Key properties:
- Quantization-proof: estimates from architecture fields, not file size
- Manipulation-resistant: faking config fields breaks vLLM loading
- Fail-closed: unknown model_type → rejected
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional

from huggingface_hub import hf_hub_download


logger = logging.getLogger("affine")


MAX_MODEL_PARAMS = 120_000_000_000  # 120B


# ---------------------------------------------------------------------------
# Architecture whitelist
# ---------------------------------------------------------------------------
# Presence means the model_type is recognized.
# Values are field name overrides (empty dict = all standard field names).
# Standard defaults: hidden_size, num_hidden_layers, intermediate_size,
#   vocab_size, num_attention_heads, num_key_value_heads
#
# Override keys:
#   h  → hidden_size
#   L  → num_hidden_layers
#   i  → intermediate_size
#   V  → vocab_size
#   n_heads → num_attention_heads
#   n_kv    → num_key_value_heads

_ARCH_FIELDS: Dict[str, Dict[str, str]] = {
    # --- Common dense architectures ---
    "llama": {},
    "mistral": {},
    "qwen2": {},
    "qwen3": {},
    "gemma": {},
    "gemma2": {},
    "gemma3": {},
    "phi3": {},
    "phi": {},
    "starcoder2": {},
    "chatglm": {},
    "falcon": {},
    "gpt_neox": {},
    "command_r": {},

    # --- Common MoE architectures ---
    "qwen2_moe": {},
    "qwen3_moe": {},
    "mixtral": {},
    "deepseek_v2": {},
    "deepseek_v3": {},
    "deepseek_v32": {},
    "dbrx": {},
    "arctic": {},
    "jamba": {},

    # --- Architectures used by real Affine miners (2025-02 survey) ---
    "glm4_moe": {},
    "glm4_moe_lite": {},
    "glm_moe_dsa": {},
    "kimi_k2": {},
    "kimi_k25": {},
    "bailing_hybrid": {},
    "minimax_m2": {},
    "qwen3_next": {},
    "qwen3_5": {},
    "qwen3_5_moe": {},
}


# ---------------------------------------------------------------------------
# Multimodal config mapping
# ---------------------------------------------------------------------------
# Multimodal models nest the text model config inside a sub-key.

_MULTIMODAL_TEXT_CONFIG: Dict[str, str] = {
    "qwen3_5": "text_config",
    "qwen3_5_moe": "text_config",
    "kimi_k25": "text_config",
    "gemma3": "text_config",
}


# ---------------------------------------------------------------------------
# Parameter estimation (pure function, no I/O)
# ---------------------------------------------------------------------------

def _estimate_params(config: dict) -> Optional[int]:
    """Estimate parameter count from config.json architecture fields.

    Returns None if architecture is unknown or required fields are missing
    (fail-closed behavior).
    """
    model_type = config.get("model_type")
    if not model_type:
        return None

    # Check whitelist
    if model_type not in _ARCH_FIELDS and model_type not in _MULTIMODAL_TEXT_CONFIG:
        return None  # Unknown architecture → fail-closed

    # Extract text config for multimodal models
    if model_type in _MULTIMODAL_TEXT_CONFIG:
        text_key = _MULTIMODAL_TEXT_CONFIG[model_type]
        config = config.get(text_key, {})

    # Resolve field names (apply overrides from whitelist)
    fields = _ARCH_FIELDS.get(model_type, {})

    try:
        h = config[fields.get("h", "hidden_size")]
        L = config[fields.get("L", "num_hidden_layers")]
        V = config[fields.get("V", "vocab_size")]
        i = config.get(fields.get("i", "intermediate_size"), h * 4)
        n_heads = config[fields.get("n_heads", "num_attention_heads")]
        n_kv = config.get(fields.get("n_kv", "num_key_value_heads"), n_heads)
        head_dim = config.get("head_dim", h // n_heads)
    except (KeyError, TypeError, ZeroDivisionError):
        return None  # Missing required fields → fail-closed

    # === Embedding ===
    embed = 2 * V * h  # embedding + lm_head

    # === Attention per layer ===
    attn = 2 * h * head_dim * (n_heads + n_kv)

    # === FFN ===
    ffn_mult = 3  # SwiGLU: gate + up + down

    # MoE detection
    n_experts = (
        config.get("num_local_experts")
        or config.get("n_routed_experts")
        or config.get("num_experts")
        or 1
    )

    if n_experts > 1:
        moe_i = config.get("moe_intermediate_size") or i

        # Shared experts
        n_shared = (
            config.get("n_shared_experts")
            or config.get("num_shared_experts")
            or 0
        )
        if n_shared == 0 and (
            config.get("shared_expert_intermediate_size")
            or config.get("moe_shared_expert_intermediate_size")
        ):
            n_shared = 1
        shared_i = (
            config.get("shared_expert_intermediate_size")
            or config.get("moe_shared_expert_intermediate_size")
            or moe_i
        )

        moe_ffn = n_experts * ffn_mult * h * moe_i + n_shared * ffn_mult * h * shared_i
        dense_ffn = ffn_mult * h * i

        # Handle partial-MoE architectures
        sparse_step = config.get("decoder_sparse_step", 1)
        first_dense = config.get("first_k_dense_replace", 0)

        if sparse_step > 1:
            n_moe_layers = L // sparse_step
            n_dense_layers = L - n_moe_layers
        else:
            n_moe_layers = L - first_dense
            n_dense_layers = first_dense

        ffn_total = n_moe_layers * moe_ffn + n_dense_layers * dense_ffn
    else:
        ffn_total = L * ffn_mult * h * i

    # === Total ===
    total = embed + L * attn + ffn_total
    return total


# ---------------------------------------------------------------------------
# HuggingFace config fetcher + public API
# ---------------------------------------------------------------------------

class ModelSizeChecker:
    """Check model parameter count against the 120B limit."""

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

    async def _fetch_config(self, model_id: str, revision: str) -> Optional[dict]:
        """Fetch config.json from HuggingFace repo.

        Uses hf_hub_download which has built-in filesystem caching.
        """
        try:
            def _download():
                path = hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    revision=revision,
                    token=self.hf_token,
                )
                with open(path, "r") as f:
                    return json.load(f)

            return await asyncio.to_thread(_download)
        except Exception as e:
            logger.warning(
                f"[ModelSizeChecker] Failed to fetch config.json for "
                f"{model_id}@{revision}: {type(e).__name__}: {e}"
            )
            return None

    async def check(self, model_id: str, revision: str) -> Dict[str, Any]:
        """Check if model exceeds the parameter limit.

        Returns:
            Dict with keys:
            - pass: bool (True if model is allowed)
            - reason: str (rejection reason or "ok")
            - estimated_params: int | None
        """
        config = await self._fetch_config(model_id, revision)
        if config is None:
            return {
                "pass": False,
                "reason": "model_params_unknown",
                "estimated_params": None,
            }

        estimated = _estimate_params(config)
        if estimated is None:
            model_type = config.get("model_type", "<missing>")
            logger.info(
                f"[ModelSizeChecker] Cannot estimate params: "
                f"{model_id} model_type={model_type}"
            )
            return {
                "pass": False,
                "reason": "model_params_unknown",
                "estimated_params": None,
            }

        if estimated > MAX_MODEL_PARAMS:
            return {
                "pass": False,
                "reason": "model_too_large",
                "estimated_params": estimated,
            }

        return {
            "pass": True,
            "reason": "ok",
            "estimated_params": estimated,
        }


async def check_model_size(model_id: str, revision: str) -> Dict[str, Any]:
    """Check if a model's parameter count is within limits.

    This is the main entry point for model size checking.

    Args:
        model_id: HuggingFace model repo (e.g., "Qwen/Qwen3-32B")
        revision: Git commit hash

    Returns:
        Dict with 'pass' boolean, 'reason' string, and 'estimated_params' int|None
    """
    checker = ModelSizeChecker()
    return await checker.check(model_id, revision)
