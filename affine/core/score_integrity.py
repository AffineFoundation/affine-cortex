"""
Score Integrity — HMAC signing and verification for evaluation scores.

When an evaluation container produces a score, it must be signed with
a server-side secret so that Stage 1 (and any other consumer) can verify
the score was not tampered with.

The HMAC covers the tuple (miner_hotkey, model_revision, env, task_id, score)
to prevent cross-miner or cross-task replay as well as score inflation.

Configuration:
    Set the environment variable SCORE_HMAC_SECRET to a random secret string.
    When unset, signing is a no-op and verification emits a warning but does
    NOT reject scores — this preserves backwards compatibility with existing
    unsigned data.
"""

import os
import hmac
import hashlib
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded secret; None means "not configured".
_HMAC_SECRET: bytes | None = None
_HMAC_LOADED: bool = False


def _get_secret() -> bytes | None:
    """Return the HMAC secret, loading it once from the environment."""
    global _HMAC_SECRET, _HMAC_LOADED
    if not _HMAC_LOADED:
        raw = os.environ.get("SCORE_HMAC_SECRET", "")
        _HMAC_SECRET = raw.encode() if raw else None
        _HMAC_LOADED = True
        if _HMAC_SECRET is None:
            logger.warning(
                "SCORE_HMAC_SECRET is not set — score integrity checks are disabled. "
                "Set this variable to enable HMAC verification of evaluation scores."
            )
    return _HMAC_SECRET


def compute_score_hmac(
    miner_hotkey: str,
    model_revision: str,
    env: str,
    task_id: str | int,
    score: float,
) -> str | None:
    """Compute an HMAC-SHA256 hex digest for a score record.

    Returns the hex digest, or None if the secret is not configured.
    """
    secret = _get_secret()
    if secret is None:
        return None

    message = f"{miner_hotkey}:{model_revision}:{env}:{task_id}:{score:.6f}"
    return hmac.new(secret, message.encode(), hashlib.sha256).hexdigest()


def verify_score_hmac(
    miner_hotkey: str,
    model_revision: str,
    env: str,
    task_id: str | int,
    score: float,
    expected_hmac: str | None,
) -> bool:
    """Verify the HMAC for a score record.

    Behaviour when the secret is not configured (backwards-compat mode):
        - Always returns True (accept the score).
        - Logs a debug-level message so operators can audit.

    Behaviour when the secret IS configured:
        - If ``expected_hmac`` is None or empty, logs a WARNING and returns
          True (to avoid breaking existing unsigned records during migration).
        - Otherwise performs constant-time comparison and returns the result.
    """
    secret = _get_secret()

    # Secret not configured — backwards-compat, accept everything.
    if secret is None:
        return True

    # Secret configured but record has no HMAC — warn but accept.
    if not expected_hmac:
        logger.warning(
            "Score record missing HMAC (miner=%s env=%s task=%s). "
            "Accepting for backwards compatibility — ensure new records are signed.",
            miner_hotkey[:16],
            env,
            task_id,
        )
        return True

    # Compute and compare.
    actual = compute_score_hmac(miner_hotkey, model_revision, env, task_id, score)
    if actual is None:
        # Should not happen since we checked secret above, but be safe.
        return True

    valid = hmac.compare_digest(actual, expected_hmac)
    if not valid:
        logger.error(
            "HMAC verification FAILED for score (miner=%s env=%s task=%s score=%.6f). "
            "Possible score tampering detected!",
            miner_hotkey[:16],
            env,
            task_id,
            score,
        )
    return valid
