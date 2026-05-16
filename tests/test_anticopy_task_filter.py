"""Codex / sglang split inside ``SWE-INFINITE``."""

from affine.src.anticopy.task_filter import (
    is_codex_task,
    is_eligible_rollout_source,
)


def test_swe_infinite_even_is_codex():
    """Even task_id is codex (skipped); odd is sglang (eligible)."""
    assert is_codex_task("SWE-INFINITE", 0)
    assert is_codex_task("SWE-INFINITE", 42)
    assert not is_codex_task("SWE-INFINITE", 1)
    assert not is_codex_task("SWE-INFINITE", 43)


def test_other_envs_are_never_codex():
    for env in ("MTH", "CDE", "MEMORY", "TERMINAL"):
        for tid in (0, 1, 100, 999):
            assert not is_codex_task(env, tid)
            assert is_eligible_rollout_source(env, tid)


def test_case_insensitive_env_match():
    assert is_codex_task("swe-infinite", 4)
    assert not is_codex_task("swe-infinite", 5)


def test_eligibility_inverts_codex_flag():
    assert not is_eligible_rollout_source("SWE-INFINITE", 2)
    assert is_eligible_rollout_source("SWE-INFINITE", 3)
