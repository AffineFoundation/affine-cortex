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
import struct
from typing import Dict, Any, Optional

import httpx
from huggingface_hub import hf_hub_download, hf_hub_url


logger = logging.getLogger("affine")

_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_WEIGHT_SHAPE_CACHE: dict[tuple[str, str], Optional[str]] = {}


class _WeightShapeCheckUnavailable(Exception):
    """Raised when a lightweight safetensors shape check cannot run."""


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


def _expected_vocab_and_hidden(config: dict) -> tuple[Optional[int], Optional[int]]:
    """Return the language vocab/hidden sizes from dense or nested configs."""
    vocab = _lookup(config, "vocab_size")
    hidden = _lookup(config, "hidden_size")
    if vocab is None:
        vocab = _lookup(config, "text_config.vocab_size")
    if hidden is None:
        hidden = _lookup(config, "text_config.hidden_size")
    return (
        vocab if isinstance(vocab, int) else None,
        hidden if isinstance(hidden, int) else None,
    )


def _select_vocab_tensor_names(names) -> list[str]:
    """Find the token embedding/lm_head tensors that must match vocab_size."""
    name_set = set(names)
    exact_candidates = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "text_model.model.embed_tokens.weight",
        "text_model.lm_head.weight",
    ]
    selected = [name for name in exact_candidates if name in name_set]
    if selected:
        return selected
    return sorted(
        name for name in name_set
        if name.endswith(".embed_tokens.weight") or name.endswith(".lm_head.weight")
    )


def _weight_shape_mismatch_reason(
    tensor_name: str,
    shape: Any,
    *,
    expected_vocab: int,
    expected_hidden: Optional[int],
) -> Optional[str]:
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not all(isinstance(dim, int) for dim in shape[:2])
    ):
        return f"weight_shape_invalid:{tensor_name}.shape={shape!r}"
    if shape[0] != expected_vocab:
        return (
            f"weight_shape_mismatch:{tensor_name}.dim0={shape[0]} "
            f"expected_vocab={expected_vocab}"
        )
    if expected_hidden is not None and shape[1] != expected_hidden:
        return (
            f"weight_shape_mismatch:{tensor_name}.dim1={shape[1]} "
            f"expected_hidden={expected_hidden}"
        )
    return None


def _decode_safetensors_header(prefix: bytes) -> dict:
    if len(prefix) < 8:
        raise _WeightShapeCheckUnavailable("safetensors header prefix too small")
    header_len = struct.unpack("<Q", prefix[:8])[0]
    if header_len <= 0 or header_len > _MAX_SAFETENSORS_HEADER_BYTES:
        raise _WeightShapeCheckUnavailable(
            f"safetensors header length out of range: {header_len}"
        )
    need = 8 + header_len
    if len(prefix) < need:
        raise _WeightShapeCheckUnavailable(
            f"safetensors header truncated: have={len(prefix)} need={need}"
        )
    try:
        return json.loads(prefix[8:need].decode("utf-8"))
    except Exception as e:
        raise _WeightShapeCheckUnavailable(
            f"safetensors header decode failed: {type(e).__name__}: {e}"
        ) from e


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

    async def _check_safetensors_weight_shapes(
        self, model_id: str, revision: str, config: dict,
    ) -> Optional[str]:
        """Return a deterministic weight/config mismatch reason, if found.

        This reads only safetensors headers over HTTP range requests. It never
        downloads full shard bodies, so it is cheap enough for monitor-side
        validation while still catching config/weight vocabulary mismatches
        before the scheduler burns a Targon deploy attempt.
        """
        expected_vocab, expected_hidden = _expected_vocab_and_hidden(config)
        if expected_vocab is None:
            return None
        cache_key = (model_id, revision)
        if cache_key in _WEIGHT_SHAPE_CACHE:
            return _WEIGHT_SHAPE_CACHE[cache_key]

        try:
            reason, checked = await asyncio.to_thread(
                self._check_safetensors_weight_shapes_sync,
                model_id,
                revision,
                expected_vocab,
                expected_hidden,
            )
        except _WeightShapeCheckUnavailable as e:
            logger.debug(
                f"[ModelSizeChecker] safetensors shape check skipped for "
                f"{model_id}@{revision[:8]}: {e}"
            )
            return None

        if checked:
            _WEIGHT_SHAPE_CACHE[cache_key] = reason
        return reason

    def _check_safetensors_weight_shapes_sync(
        self,
        model_id: str,
        revision: str,
        expected_vocab: int,
        expected_hidden: Optional[int],
    ) -> tuple[Optional[str], bool]:
        index = self._fetch_safetensors_index_sync(model_id, revision)
        if index is not None:
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                return "safetensors_index_invalid:missing_weight_map", True
            tensor_names = _select_vocab_tensor_names(weight_map.keys())
            if not tensor_names:
                return None, False
            checked = False
            for filename in sorted({str(weight_map[name]) for name in tensor_names}):
                header = self._fetch_safetensors_header_sync(
                    model_id, revision, filename,
                )
                for name in tensor_names:
                    if weight_map.get(name) != filename:
                        continue
                    meta = header.get(name)
                    if not isinstance(meta, dict):
                        return f"safetensors_index_tensor_missing:{name}", True
                    checked = True
                    reason = _weight_shape_mismatch_reason(
                        name,
                        meta.get("shape"),
                        expected_vocab=expected_vocab,
                        expected_hidden=expected_hidden,
                    )
                    if reason:
                        return reason, True
            return None, checked

        try:
            header = self._fetch_safetensors_header_sync(
                model_id, revision, "model.safetensors",
            )
        except _WeightShapeCheckUnavailable:
            return None, False
        tensor_names = _select_vocab_tensor_names(header.keys())
        checked = False
        for name in tensor_names:
            meta = header.get(name)
            if not isinstance(meta, dict):
                continue
            checked = True
            reason = _weight_shape_mismatch_reason(
                name,
                meta.get("shape"),
                expected_vocab=expected_vocab,
                expected_hidden=expected_hidden,
            )
            if reason:
                return reason, True
        return None, checked

    def _fetch_safetensors_index_sync(
        self, model_id: str, revision: str,
    ) -> Optional[dict]:
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors.index.json",
                revision=revision,
                token=self.hf_token,
            )
        except Exception:
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise _WeightShapeCheckUnavailable(
                f"safetensors index decode failed: {type(e).__name__}: {e}"
            ) from e

    def _fetch_safetensors_header_sync(
        self, model_id: str, revision: str, filename: str,
    ) -> dict:
        url = hf_hub_url(model_id, filename, revision=revision)
        first = self._fetch_url_prefix(url, 8)
        if len(first) < 8:
            raise _WeightShapeCheckUnavailable(
                f"{filename}: unable to read safetensors header length"
            )
        header_len = struct.unpack("<Q", first[:8])[0]
        if header_len <= 0 or header_len > _MAX_SAFETENSORS_HEADER_BYTES:
            raise _WeightShapeCheckUnavailable(
                f"{filename}: safetensors header length out of range: "
                f"{header_len}"
            )
        return _decode_safetensors_header(
            self._fetch_url_prefix(url, 8 + header_len)
        )

    def _fetch_url_prefix(self, url: str, size: int) -> bytes:
        headers = {
            "Range": f"bytes=0-{max(0, size - 1)}",
            "Accept-Encoding": "identity",
        }
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        chunks = []
        total = 0
        timeout = httpx.Timeout(30.0, connect=10.0)
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= size:
                            break
        except Exception as e:
            raise _WeightShapeCheckUnavailable(
                f"range fetch failed: {type(e).__name__}: {e}"
            ) from e
        return b"".join(chunks)[:size]

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

        shape_mismatch = await self._check_safetensors_weight_shapes(
            model_id, revision, config,
        )
        if shape_mismatch:
            logger.info(
                f"[ModelSizeChecker] Model weight/config mismatch rejected: "
                f"{model_id} model_type={model_type} reason={shape_mismatch}"
            )
            return {
                "pass": False,
                "reason": shape_mismatch,
                "model_type": matched_model_type or model_type,
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
