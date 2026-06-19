"""Preflight checks for deploy failures that are safe to treat as miner faults."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import EntryNotFoundError

from affine.core.providers.targon_client import is_qwen36
from affine.core.setup import logger


MISSING_PREPROCESSOR_CONFIG_REASON = (
    "deployment_failed:missing_preprocessor_config"
)
QWEN36_PREPROCESSOR_FILE = "preprocessor_config.json"

HFFileExistsFn = Callable[[str, str, str], Awaitable[Optional[bool]]]


@dataclass(frozen=True)
class DeployFailureClassification:
    rule_name: str
    reason: str


async def classify_deploy_preflight_failure(
    *,
    model: str,
    revision: str,
    model_type: str,
    hf_file_exists_fn: Optional[HFFileExistsFn] = None,
) -> Optional[DeployFailureClassification]:
    """Pre-classify deterministic failures before starting inference.

    Qwen3.6 repos must include ``preprocessor_config.json`` because SGLang's
    Transformers processor load fails deterministically without it. A clear
    missing-file response is a miner fault; inconclusive HF/network errors are
    ignored so the deploy path can still retry normally.
    """
    if not is_qwen36(model_type):
        return None
    exists_fn = hf_file_exists_fn or _hf_file_exists
    exists = await exists_fn(model, revision, QWEN36_PREPROCESSOR_FILE)
    if exists is False:
        return DeployFailureClassification(
            rule_name="qwen36_missing_preprocessor_config",
            reason=MISSING_PREPROCESSOR_CONFIG_REASON,
        )
    return None


async def _hf_file_exists(
    model_id: str, revision: str, filename: str,
) -> Optional[bool]:
    def _probe() -> None:
        get_hf_file_metadata(
            hf_hub_url(model_id, filename, revision=revision),
            token=os.getenv("HF_TOKEN"),
        )

    try:
        await asyncio.to_thread(_probe)
        return True
    except EntryNotFoundError:
        return False
    except Exception as e:
        logger.warning(
            f"[deploy-preflight] HF file probe inconclusive "
            f"{model_id}@{revision[:8]}:{filename}: {type(e).__name__}: {e}"
        )
        return None
