"""
Model Size Checker

Validates that miners use the required model architecture (Qwen3-32B).
Checks config.json architecture fields from HuggingFace.

Key properties:
- Quantization-proof: checks architecture fields, not file size
- Manipulation-resistant: faking config fields breaks vLLM loading
- Fail-closed: any mismatch → rejected
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional

from huggingface_hub import hf_hub_download


logger = logging.getLogger("affine")


# ---------------------------------------------------------------------------
# Required model architecture
# ---------------------------------------------------------------------------
# Only models matching this exact architecture are allowed.
# Fine-tunes of the same base model share these fields (weights differ,
# architecture stays the same), so this correctly permits fine-tuned variants.

REQUIRED_MODEL_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "intermediate_size": 25600,
    "vocab_size": 151936,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
}


def _check_required_model(config: dict) -> Optional[str]:
    """Check if config matches the required model architecture.

    Returns None if config matches, or a mismatch description string.
    """
    for field, expected in REQUIRED_MODEL_CONFIG.items():
        actual = config.get(field)
        if actual != expected:
            return f"{field}={actual} (expected {expected})"
    return None


class ModelSizeChecker:
    """Check model architecture against the required model (Qwen3-32B)."""

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
        """Check if model matches the required architecture (Qwen3-32B).

        Returns:
            Dict with keys:
            - pass: bool (True if model is allowed)
            - reason: str (rejection reason or "ok")
        """
        config = await self._fetch_config(model_id, revision)
        if config is None:
            return {"pass": False, "reason": "config_fetch_failed"}

        mismatch = _check_required_model(config)
        if mismatch is not None:
            model_type = config.get("model_type", "<missing>")
            logger.info(
                f"[ModelSizeChecker] Model not allowed: "
                f"{model_id} model_type={model_type} mismatch={mismatch}"
            )
            return {"pass": False, "reason": f"model_not_allowed:{mismatch}"}

        return {"pass": True, "reason": "ok"}


async def check_model_size(model_id: str, revision: str) -> Dict[str, Any]:
    """Check if a model matches the required architecture.

    This is the main entry point for model validation.

    Args:
        model_id: HuggingFace model repo (e.g., "Qwen/Qwen3-32B")
        revision: Git commit hash

    Returns:
        Dict with 'pass' boolean and 'reason' string
    """
    checker = ModelSizeChecker()
    return await checker.check(model_id, revision)
