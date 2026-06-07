"""
Model Size Checker

Validates that miners use one of the allowed model architectures.
Checks config.json architecture fields from HuggingFace.

Key properties:
- Size-proof: matches architecture fields, not file size, so a miner can't
  shrink a model (e.g. by quantizing) to dodge a size limit.
- Manipulation-resistant: faking config fields breaks vLLM loading
- Fail-closed: any mismatch → rejected
- Unquantized-only: miners submit standard-precision fine-tunes of the allowed
  architectures. Any pre-quantized config (a ``quantization_config`` block) is
  rejected — it changes serving requirements (e.g. bitsandbytes 4-bit can't be
  loaded by the sglang image and traps the scheduler in a deploy-retry loop).
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional

from huggingface_hub import hf_hub_download


logger = logging.getLogger("affine")


# ---------------------------------------------------------------------------
# Allowed model architectures
# ---------------------------------------------------------------------------
# A miner's config.json must match exactly one entry in this list.
# Fine-tunes of the same base model share these fields (weights differ,
# architecture stays the same), so this correctly permits fine-tuned variants.
#
# Keys may be dotted paths to reach nested fields (e.g. ``text_config.hidden_size``
# for multimodal configs where the language-model fields live under text_config).

ALLOWED_MODEL_CONFIGS = [
    # Qwen3-32B (dense, text-only)
    {
        "model_type": "qwen3",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "intermediate_size": 25600,
        "vocab_size": 151936,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
    },
    # Qwen3.6-35B-A3B (MoE, multimodal — language fields nested under text_config)
    {
        "model_type": "qwen3_5_moe",
        "text_config.hidden_size": 2048,
        "text_config.num_hidden_layers": 40,
        "text_config.num_attention_heads": 16,
        "text_config.num_key_value_heads": 2,
        "text_config.vocab_size": 248320,
        "text_config.num_experts": 256,
        "text_config.num_experts_per_tok": 8,
        "text_config.moe_intermediate_size": 512,
    },
]


def _lookup(config: dict, path: str) -> Any:
    """Resolve a dotted ``a.b.c`` path inside ``config``; missing nodes → None."""
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _check_against(config: dict, expected: dict) -> Optional[str]:
    """Return None if config matches ``expected``, else a mismatch description.

    Uses ``repr()`` for both values so a type mismatch (e.g. ``"5120"`` vs
    ``5120``) is visible in the rejection reason instead of printing as
    identical-looking strings.
    """
    for field, want in expected.items():
        actual = _lookup(config, field)
        if actual != want:
            return f"{field}={actual!r} (expected {want!r})"
    return None


def _match_allowed_model(config: dict) -> tuple[Optional[str], Optional[str]]:
    """Return the matching allow-list model_type, or a mismatch description.

    Picks the mismatch from the entry that agreed on ``model_type`` (best signal
    that the miner *intended* that architecture); falls back to listing the
    permitted model_types when no entry's model_type matches.
    """
    actual_type = config.get("model_type")
    type_match_mismatch: Optional[str] = None
    for expected in ALLOWED_MODEL_CONFIGS:
        mismatch = _check_against(config, expected)
        if mismatch is None:
            return str(expected.get("model_type") or ""), None
        if expected.get("model_type") == actual_type and type_match_mismatch is None:
            type_match_mismatch = mismatch

    if type_match_mismatch is not None:
        return None, type_match_mismatch
    allowed_types = ", ".join(
        sorted({str(e.get("model_type")) for e in ALLOWED_MODEL_CONFIGS})
    )
    return None, f"model_type={actual_type} (expected one of: {allowed_types})"


class ModelSizeChecker:
    """Check model architecture against the allowed model whitelist."""

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
        """Check if model matches any allowed architecture.

        Returns:
            Dict with keys:
            - pass: bool (True if model is allowed)
            - reason: str (rejection reason or "ok")
        """
        config = await self._fetch_config(model_id, revision)
        if config is None:
            return {
                "pass": False,
                "reason": "config_fetch_failed",
                "model_type": "",
            }

        matched_model_type, mismatch = _match_allowed_model(config)
        model_type = str(config.get("model_type") or "")
        if mismatch is not None:
            logger.info(
                f"[ModelSizeChecker] Model not allowed: "
                f"{model_id} model_type={model_type or '<missing>'} mismatch={mismatch}"
            )
            return {
                "pass": False,
                "reason": f"model_not_allowed:{mismatch}",
                "model_type": model_type,
            }

        quant = config.get("quantization_config")
        if isinstance(quant, dict):
            method = str(quant.get("quant_method") or "unknown")
            logger.info(
                f"[ModelSizeChecker] Quantized model rejected: {model_id} "
                f"model_type={model_type} quant_method={method}"
            )
            return {
                "pass": False,
                "reason": f"quantized_model:{method}",
                "model_type": model_type,
            }

        return {
            "pass": True,
            "reason": "ok",
            "model_type": matched_model_type or model_type,
        }


async def check_model_size(model_id: str, revision: str) -> Dict[str, Any]:
    """Check if a model matches one of the allowed architectures.

    This is the main entry point for model validation.

    Args:
        model_id: HuggingFace model repo (e.g., "Qwen/Qwen3-32B")
        revision: Git commit hash

    Returns:
        Dict with 'pass' boolean and 'reason' string
    """
    checker = ModelSizeChecker()
    return await checker.check(model_id, revision)
