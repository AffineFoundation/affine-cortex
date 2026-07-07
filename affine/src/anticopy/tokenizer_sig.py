"""
Tokenizer signature — sha256 over the bytes of ``tokenizer.json`` (or
``tokenizer.model`` fallback) at a given HF revision. Two miners must
share the same signature for CEAC to be apple-to-apple; mismatch
auto-invalidates without burning GPU.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


# Files we'll try in order. The first one that exists at the given
# revision is taken as authoritative. ``tokenizer.json`` (fast
# tokenizer artifact) is the standard for modern HF models;
# ``tokenizer.model`` is the SentencePiece fallback some older
# checkpoints still use.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model")
TOKENIZER_SIG_REPO_INACCESSIBLE = "repo_inaccessible"
TOKENIZER_SIG_REVISION_NOT_FOUND = "revision_not_found"
TOKENIZER_SIG_MISSING_ARTIFACT = "missing_tokenizer_artifact"
TOKENIZER_SIG_PARSE_FAILED = "tokenizer_parse_failed"
TOKENIZER_SIG_TRANSIENT_FETCH_FAILED = "transient_fetch_failed"


async def compute_tokenizer_signature(
    model_id: str, revision: str, *, hf_token: Optional[str] = None
) -> Tuple[Optional[str], str, str]:
    """Return ``(sha256_hex, source_filename, reason)`` for the model's
    tokenizer at the given revision. ``reason`` is empty on success.

    Network-bound — call from inside ``asyncio.to_thread`` or use the
    helper :func:`compute_tokenizer_signature_sync` directly.
    """
    return await asyncio.to_thread(
        compute_tokenizer_signature_sync, model_id, revision, hf_token or os.getenv("HF_TOKEN")
    )


def compute_tokenizer_signature_sync(
    model_id: str, revision: str, hf_token: Optional[str] = None
) -> Tuple[Optional[str], str, str]:
    """Sync version, suitable for use outside an event loop.

    Returns a *canonical* tokenization signature: sha256 over
    ``{vocab, merges, added_tokens}`` from ``tokenizer.json``, not the
    raw file bytes. The pre-/post-processor fields (``trim_offsets``,
    ``add_prefix_space``, ``use_regex``, ...) only control how text →
    ids reverses for offsets and have **no effect on the prefill
    input_ids the worker hands to sglang** — so two miners whose
    vocab + merges are identical can be teacher-forced apple-to-apple
    even when those tuning fields disagree. (Empirically a 4-byte
    diff in those flags was knocking >50% of Qwen3 finetune miners
    out of CEAC under the old whole-file sha.)
    """
    api = HfApi(token=hf_token)
    try:
        info = api.repo_info(
            repo_id=model_id, repo_type="model", revision=revision, files_metadata=True,
        )
    except RevisionNotFoundError:
        return None, "", TOKENIZER_SIG_REVISION_NOT_FOUND
    except (GatedRepoError, RepositoryNotFoundError):
        return None, "", TOKENIZER_SIG_REPO_INACCESSIBLE
    except Exception:
        return None, "", TOKENIZER_SIG_TRANSIENT_FETCH_FAILED

    available = {
        (getattr(s, "rfilename", "") or getattr(s, "path", "") or "")
        for s in (getattr(info, "siblings", None) or [])
    }
    present = [fname for fname in TOKENIZER_FILES if fname in available]
    if not present:
        return None, "", TOKENIZER_SIG_MISSING_ARTIFACT

    saw_fetch_failure = False
    for fname in present:
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename=fname,
                revision=revision,
                token=hf_token,
            )
        except Exception:
            saw_fetch_failure = True
            continue
        sig = _canonical_tokenizer_sig(path)
        if sig is not None:
            return sig, fname, ""
    reason = (
        TOKENIZER_SIG_TRANSIENT_FETCH_FAILED
        if saw_fetch_failure
        else TOKENIZER_SIG_PARSE_FAILED
    )
    return None, "", reason


def _canonical_tokenizer_sig(path: str) -> Optional[str]:
    """Compute the canonical tokenization signature for a
    ``tokenizer.json`` (or ``tokenizer.model``) file. For JSON, hash
    only the fields that govern id assignment; for the SentencePiece
    binary fall back to the whole-file sha256 (no canonical split is
    available without parsing the proto)."""
    import json

    if path.endswith(".json"):
        try:
            with open(path, "rb") as fh:
                doc = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        canon = {
            "vocab": (doc.get("model") or {}).get("vocab"),
            "merges": (doc.get("model") or {}).get("merges"),
            "added_tokens": doc.get("added_tokens"),
        }
        blob = json.dumps(canon, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    try:
        with open(path, "rb") as fh:
            digest = hashlib.sha256()
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None
