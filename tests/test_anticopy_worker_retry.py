"""``ForwardWorker._run_job`` retry-vs-fail behaviour.

A transient infra error (HF 503, SSH gateway flake, sglang load
timeout) should *not* burn a candidate's verdict — the job is
re-queued until ``MAX_JOB_ATTEMPTS``. Only when the same job has
failed past that cap is it permanently marked ``failed``.
"""

from __future__ import annotations

import pytest

from affine.src.anticopy import worker as worker_mod
from affine.src.anticopy.worker import ForwardWorker, MAX_JOB_ATTEMPTS


class _FakeJobsDAO:
    def __init__(self):
        self.reset_calls = []
        self.mark_failed_calls = []
        self.mark_done_calls = []

    async def reset_to_pending(self, hotkey, revision):
        self.reset_calls.append((hotkey, revision))

    async def mark_failed(self, hotkey, revision, error):
        self.mark_failed_calls.append((hotkey, revision, error))

    async def mark_done(self, hotkey, revision):
        self.mark_done_calls.append((hotkey, revision))


def _patched_worker(jobs_dao):
    # All other DAOs are sentinels — the failure happens before they're
    # touched (we make ``load_anticopy_config`` itself raise).
    return ForwardWorker(
        jobs_dao=jobs_dao,
        rollouts_dao=object(),
        scores_dao=object(),
        miners_dao=object(),
        config_dao=object(),
    )


@pytest.mark.asyncio
async def test_transient_failure_under_cap_requeues(monkeypatch):
    """A first-attempt failure goes back to the queue, not to failed."""
    jobs = _FakeJobsDAO()
    worker = _patched_worker(jobs)

    async def _boom(_dao):
        raise RuntimeError("ssh gateway: Session Terminated 0")
    monkeypatch.setattr(worker_mod, "load_anticopy_config", _boom)

    await worker._run_job({
        "hotkey": "hk_xyz",
        "revision": "rev_abc",
        "model": "org/model",
        "uid": 42,
        "attempts": 1,
    })
    assert jobs.reset_calls == [("hk_xyz", "rev_abc")]
    assert jobs.mark_failed_calls == []
    assert jobs.mark_done_calls == []


@pytest.mark.asyncio
async def test_failure_past_cap_marks_failed(monkeypatch):
    """Once ``attempts >= MAX_JOB_ATTEMPTS``, stop retrying."""
    jobs = _FakeJobsDAO()
    worker = _patched_worker(jobs)

    async def _boom(_dao):
        raise RuntimeError("permanent: tokenizer load broken")
    monkeypatch.setattr(worker_mod, "load_anticopy_config", _boom)

    await worker._run_job({
        "hotkey": "hk_xyz",
        "revision": "rev_abc",
        "model": "org/model",
        "uid": 42,
        "attempts": MAX_JOB_ATTEMPTS,
    })
    assert jobs.mark_failed_calls and jobs.mark_failed_calls[0][0] == "hk_xyz"
    assert jobs.reset_calls == []
