"""Prefetcher + HF cache GC unit tests for the anticopy worker."""

from __future__ import annotations

import asyncio
import types

import pytest

import affine.src.anticopy.worker as worker_mod
from affine.src.anticopy.worker import ForwardWorker, _gc_hf_cache


# ------------------------------------------------------------ GC


class _FakeRevision:
    def __init__(self, commit_hash):
        self.commit_hash = commit_hash


class _FakeRepo:
    def __init__(self, name, last_modified, commit_hash):
        self.repo_id = name
        self.last_modified = last_modified
        self.revisions = [_FakeRevision(commit_hash)]


class _FakeStrategy:
    def __init__(self):
        self.executed = False
        self.expected_freed_size = 1024 * 1024 * 50         # 50 MB


class _FakeCacheInfo:
    def __init__(self, repos):
        self.repos = list(repos)
        self.delete_called_with = None

    def delete_revisions(self, *commit_hashes):
        self.delete_called_with = list(commit_hashes)
        strat = _FakeStrategy()
        def _execute():
            strat.executed = True
        strat.execute = _execute
        return strat


def test_gc_keeps_top_n_by_last_modified(monkeypatch):
    repos = [
        _FakeRepo("oldest",      last_modified=100, commit_hash="A"),
        _FakeRepo("middle",      last_modified=300, commit_hash="B"),
        _FakeRepo("newest",      last_modified=500, commit_hash="C"),
        _FakeRepo("really-old",  last_modified=50,  commit_hash="D"),
    ]
    fake = _FakeCacheInfo(repos)
    monkeypatch.setattr(worker_mod, "scan_cache_dir", lambda _d: fake)

    stats = _gc_hf_cache("/data", keep_recent=2)
    assert stats["kept"] == 2
    assert stats["deleted_revisions"] == 2
    # only the bottom-2 by last_modified should die
    assert set(fake.delete_called_with) == {"A", "D"}


def test_gc_noop_when_below_threshold(monkeypatch):
    fake = _FakeCacheInfo([
        _FakeRepo("only", last_modified=1, commit_hash="X"),
    ])
    monkeypatch.setattr(worker_mod, "scan_cache_dir", lambda _d: fake)
    stats = _gc_hf_cache("/data", keep_recent=3)
    assert stats["deleted_revisions"] == 0
    assert fake.delete_called_with is None


def test_gc_clamps_keep_recent_to_two(monkeypatch):
    """Caller passes 1 → we clamp up to 2 so sglang's current model
    can't be GC'd by accident."""
    repos = [
        _FakeRepo("a", last_modified=100, commit_hash="A"),
        _FakeRepo("b", last_modified=200, commit_hash="B"),
        _FakeRepo("c", last_modified=300, commit_hash="C"),
    ]
    fake = _FakeCacheInfo(repos)
    monkeypatch.setattr(worker_mod, "scan_cache_dir", lambda _d: fake)
    stats = _gc_hf_cache("/data", keep_recent=1)
    assert stats["kept"] == 2
    assert set(fake.delete_called_with) == {"A"}


def test_gc_survives_scan_error(monkeypatch):
    def _boom(_d):
        raise OSError("no cache dir")
    monkeypatch.setattr(worker_mod, "scan_cache_dir", _boom)
    assert _gc_hf_cache("/data", keep_recent=3)["deleted_revisions"] == 0


# ------------------------------------------------------------ Prefetcher


class _FakeJobsDAO:
    def __init__(self, peek_rows):
        self.peek_rows = peek_rows
        self.peek_calls = []

    async def peek_pending(self, *, limit, exclude_pk=None):
        self.peek_calls.append({"limit": limit, "exclude_pk": exclude_pk})
        return [r for r in self.peek_rows if r.get("pk") != exclude_pk][:limit]


def _patched_worker(monkeypatch, peek_rows):
    # Force the worker into local-snapshot mode for these tests: the
    # ``REMOTE_SSH_HOST`` module global may otherwise leak in from the
    # caller's shell env (e.g. an operator's prod ``.env``) and route
    # the prefetch through ``_remote_snapshot_download``, which would
    # then try to actually SSH out and break the unit test.
    monkeypatch.setattr(worker_mod, "REMOTE_SSH_HOST", "")
    worker = ForwardWorker(
        jobs_dao=_FakeJobsDAO(peek_rows),
        rollouts_dao=object(),
        scores_dao=object(),
        miners_dao=object(),
        config_dao=object(),
    )
    calls = []
    def _fake_snapshot(*args, **kwargs):
        calls.append({"repo_id": kwargs.get("repo_id"), "revision": kwargs.get("revision")})
        return f"/tmp/{kwargs.get('repo_id')}/{kwargs.get('revision')}"
    monkeypatch.setattr(worker_mod, "snapshot_download", _fake_snapshot)
    return worker, calls


@pytest.mark.asyncio
async def test_prefetcher_downloads_next_pending(monkeypatch):
    worker, calls = _patched_worker(
        monkeypatch,
        peek_rows=[
            {"pk": "JOB#a", "model": "org/model-A", "revision": "rev_A"},
            {"pk": "JOB#b", "model": "org/model-B", "revision": "rev_B"},
        ],
    )
    await worker._prefetch_next_pending(current_pk="JOB#a")
    assert calls == [{"repo_id": "org/model-B", "revision": "rev_B"}]
    assert ("org/model-B", "rev_B") in worker._prefetched


@pytest.mark.asyncio
async def test_prefetcher_dedups_same_job(monkeypatch):
    worker, calls = _patched_worker(
        monkeypatch,
        peek_rows=[{"pk": "JOB#x", "model": "org/m", "revision": "rev"}],
    )
    await worker._prefetch_next_pending(current_pk="JOB#current")
    await worker._prefetch_next_pending(current_pk="JOB#current")
    assert len(calls) == 1               # only one actual snapshot_download


@pytest.mark.asyncio
async def test_prefetcher_retries_after_failure(monkeypatch):
    worker, _ = _patched_worker(
        monkeypatch,
        peek_rows=[{"pk": "JOB#x", "model": "org/m", "revision": "rev"}],
    )

    calls = []
    def _fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "/tmp/ok"
    monkeypatch.setattr(worker_mod, "snapshot_download", _fail_once)

    await worker._prefetch_next_pending(current_pk="JOB#current")
    assert ("org/m", "rev") not in worker._prefetched   # marker dropped on fail
    await worker._prefetch_next_pending(current_pk="JOB#current")
    assert len(calls) == 2               # retried
    assert ("org/m", "rev") in worker._prefetched


@pytest.mark.asyncio
async def test_prefetcher_empty_queue_is_noop(monkeypatch):
    worker, calls = _patched_worker(monkeypatch, peek_rows=[])
    await worker._prefetch_next_pending(current_pk="JOB#current")
    assert calls == []


@pytest.mark.asyncio
async def test_prefetcher_skips_current_pk(monkeypatch):
    """If the only pending row IS the current job, prefetcher does
    nothing (current job's own _fetch_weights handles it)."""
    worker, calls = _patched_worker(
        monkeypatch,
        peek_rows=[{"pk": "JOB#current", "model": "org/m", "revision": "rev"}],
    )
    await worker._prefetch_next_pending(current_pk="JOB#current")
    assert calls == []
