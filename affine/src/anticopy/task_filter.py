"""
Per-env hooks for selecting which sample_results rows can be reused as
CEAC rollouts. Centralised so the refresh service and the worker (when
verifying a rollout key) agree on what's eligible.
"""

from __future__ import annotations


def is_codex_task(env: str, task_id: int) -> bool:
    """``SWE-INFINITE`` interleaves codex and sglang rollouts: even
    ``task_id`` rows are produced by the codex CLI (not the deployed
    sglang champion), so their ``conversation[-1]`` doesn't belong to
    the champion model and we can't teacher-force a candidate against
    it. Treat those as ineligible.

    Other envs always go through the champion's sglang and are
    eligible regardless of parity.
    """
    if (env or "").upper() == "SWE-INFINITE":
        return int(task_id) % 2 == 0
    return False


def is_eligible_rollout_source(env: str, task_id: int) -> bool:
    """A sample_results row is eligible to back a CEAC rollout iff the
    response was produced by the champion's deployed sglang server."""
    return not is_codex_task(env, task_id)
