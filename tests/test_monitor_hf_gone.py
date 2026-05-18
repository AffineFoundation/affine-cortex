"""miners_monitor: HF 404 / gated → permanent invalid + terminate.

Covers the two-step gate added by HF_GONE_PERMANENT_THRESHOLD:

  1. ``_get_model_info`` differentiates ``RepositoryNotFoundError`` /
     ``GatedRepoError`` from generic transients, and bumps a per-key
     consecutive-failure counter on the deterministic ones only.
  2. ``_validate_miner`` flips the row to ``permanent_invalid=True`` with
     reason ``hf_repo_not_found`` once the counter reaches the threshold,
     and ``_persist_miners`` then writes a conditional terminate into
     ``miner_stats`` (only if the row is still ``sampling``).
"""

from __future__ import annotations

import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

import affine.database.client as db_client


def _make_hf_error(cls, message: str):
    """Build an HF error without constructing the full HTTP response chain."""
    err = cls.__new__(cls)
    Exception.__init__(err, message)
    return err
from affine.src.anticopy.threshold import AntiCopyConfig
from affine.src.monitor.miners_monitor import (
    HF_GONE_PERMANENT_THRESHOLD,
    MinerInfo,
    MinersMonitor,
)


class _FakeDynamoClient:
    async def put_item(self, **kwargs):
        return {}
    async def get_item(self, **kwargs):
        return {}
    async def query(self, **kwargs):
        return {"Items": []}
    async def update_item(self, **kwargs):
        return {}


class _FakeStateDAO:
    async def get_champion_tokenizer_sig(self):
        return ""


class _FakeScoresDAO:
    async def get_score(self, hotkey, revision):
        return None


def _build_monitor():
    monitor = MinersMonitor()
    monitor.anticopy_state_dao = _FakeStateDAO()
    monitor.anticopy_scores_dao = _FakeScoresDAO()

    async def _fake_load(_dao):
        return AntiCopyConfig(enabled=False)
    monitor._safe_load_anticopy_config = (
        lambda: _fake_load(monitor.config_dao)
    )
    return monitor


# ---- _get_model_info counter behavior ---------------------------------------


@pytest.mark.asyncio
async def test_get_model_info_bumps_count_on_repo_not_found(monkeypatch):
    monitor = _build_monitor()

    class _FakeApi:
        def __init__(self, token=None):
            pass
        def repo_info(self, **kwargs):
            raise _make_hf_error(RepositoryNotFoundError, "404")
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.HfApi", _FakeApi,
    )

    for expected in (1, 2, 3):
        result = await monitor._get_model_info("org/Affine-foo", "rev1")
        assert result is None
        assert monitor._hf_gone_counts[("org/Affine-foo", "rev1")] == expected
        # Bypass the TTL cache so each call exercises the HF path.
        monitor._weights_cache.clear()


@pytest.mark.asyncio
async def test_get_model_info_bumps_count_on_gated(monkeypatch):
    monitor = _build_monitor()

    class _FakeApi:
        def __init__(self, token=None):
            pass
        def repo_info(self, **kwargs):
            raise _make_hf_error(GatedRepoError, "gated")
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.HfApi", _FakeApi,
    )

    await monitor._get_model_info("org/Affine-foo", "rev1")
    assert monitor._hf_gone_counts[("org/Affine-foo", "rev1")] == 1


@pytest.mark.asyncio
async def test_get_model_info_does_not_bump_on_transient(monkeypatch):
    monitor = _build_monitor()

    class _FakeApi:
        def __init__(self, token=None):
            pass
        def repo_info(self, **kwargs):
            raise TimeoutError("hf is having a bad day")
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.HfApi", _FakeApi,
    )

    await monitor._get_model_info("org/Affine-bar", "rev2")
    assert ("org/Affine-bar", "rev2") not in monitor._hf_gone_counts


@pytest.mark.asyncio
async def test_get_model_info_resets_count_on_success(monkeypatch):
    monitor = _build_monitor()
    monitor._hf_gone_counts[("org/Affine-baz", "rev3")] = 2

    class _Sibling:
        def __init__(self, name, sha):
            self.rfilename = name
            self.lfs = {"sha256": sha}

    class _Info:
        sha = "rev3-resolved"
        siblings = [_Sibling("model.safetensors", "abc123")]

    class _FakeApi:
        def __init__(self, token=None):
            pass
        def repo_info(self, **kwargs):
            return _Info()
        def list_repo_commits(self, **kwargs):
            return []

    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.HfApi", _FakeApi,
    )

    result = await monitor._get_model_info("org/Affine-baz", "rev3")
    assert result is not None
    assert ("org/Affine-baz", "rev3") not in monitor._hf_gone_counts


# ---- _validate_miner: threshold gate ----------------------------------------


async def _async_pass_size(*_args, **_kw):
    return {"pass": True}


async def _async_safe(*_args, **_kw):
    return {"safe": True, "reason": ""}


@pytest.mark.asyncio
async def test_validate_below_threshold_is_transient(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor()
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_model_size",
        _async_pass_size,
    )
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_template_safety",
        _async_safe,
    )

    async def _gone(_model, _rev):
        monitor._hf_gone_counts[(_model, _rev)] = (
            monitor._hf_gone_counts.get((_model, _rev), 0) + 1
        )
        return None
    monitor._get_model_info = _gone

    for _ in range(HF_GONE_PERMANENT_THRESHOLD - 1):
        info = await monitor._validate_miner(
            uid=42, hotkey="hk_gone_xyz",
            model="org/affine-model-hk_gone_xyz",
            revision="rev",
            block=99_999_999,
            commit_count=1,
        )
        assert info.is_valid is False
        assert info.permanent_invalid is False
        assert info.invalid_reason == "hf_model_fetch_failed"


@pytest.mark.asyncio
async def test_validate_at_threshold_flips_permanent(monkeypatch):
    monkeypatch.setattr(db_client, "get_client", lambda: _FakeDynamoClient())
    monitor = _build_monitor()
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_model_size",
        _async_pass_size,
    )
    monkeypatch.setattr(
        "affine.src.monitor.miners_monitor.check_template_safety",
        _async_safe,
    )

    async def _gone(model, rev):
        monitor._hf_gone_counts[(model, rev)] = (
            monitor._hf_gone_counts.get((model, rev), 0) + 1
        )
        return None
    monitor._get_model_info = _gone

    info = None
    for _ in range(HF_GONE_PERMANENT_THRESHOLD):
        info = await monitor._validate_miner(
            uid=42, hotkey="hk_gone_perm_xyz",
            model="org/affine-model-hk_gone_perm_xyz",
            revision="rev",
            block=99_999_999,
            commit_count=1,
        )
    assert info is not None
    assert info.is_valid is False
    assert info.permanent_invalid is True
    assert info.invalid_reason == "hf_repo_not_found"


# ---- _maybe_terminate_stats: conditional terminate --------------------------


class _RecordingStatsDAO:
    def __init__(self, returns=True):
        self.calls: list[dict] = []
        self._returns = returns

    async def terminate_if_sampling(self, *, hotkey, revision, reason):
        self.calls.append(
            {"hotkey": hotkey, "revision": revision, "reason": reason},
        )
        return self._returns


@pytest.mark.asyncio
async def test_maybe_terminate_fires_for_permanent_listed_reason():
    monitor = _build_monitor()
    monitor.stats_dao = _RecordingStatsDAO()
    miner = MinerInfo(
        uid=42, hotkey="hk_x", model="org/affine-model-hk_x", revision="rev",
        is_valid=False, permanent_invalid=True, invalid_reason="hf_repo_not_found",
    )
    await monitor._maybe_terminate_stats(miner)
    assert monitor.stats_dao.calls == [
        {"hotkey": "hk_x", "revision": "rev", "reason": "hf_repo_not_found"},
    ]


@pytest.mark.asyncio
async def test_maybe_terminate_skips_transient_reason():
    """A transient ``hf_model_fetch_failed`` (permanent=False) must NOT
    terminate the row — the next monitor cycle should be allowed to
    re-evaluate."""
    monitor = _build_monitor()
    monitor.stats_dao = _RecordingStatsDAO()
    miner = MinerInfo(
        uid=42, hotkey="hk_y", model="org/affine-model-hk_y", revision="rev",
        is_valid=False, permanent_invalid=False, invalid_reason="hf_model_fetch_failed",
    )
    await monitor._maybe_terminate_stats(miner)
    assert monitor.stats_dao.calls == []


@pytest.mark.asyncio
async def test_maybe_terminate_skips_permanent_unlisted_reason():
    """A reason we don't model as a terminate-worthy lifecycle event (e.g.
    a hypothetical future permanent flag) is left to the scheduler."""
    monitor = _build_monitor()
    monitor.stats_dao = _RecordingStatsDAO()
    miner = MinerInfo(
        uid=42, hotkey="hk_z", model="org/affine-model-hk_z", revision="rev",
        is_valid=False, permanent_invalid=True, invalid_reason="some_future_reason",
    )
    await monitor._maybe_terminate_stats(miner)
    assert monitor.stats_dao.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        # Sig flips back to match when the active champion changes — the
        # row must stay sampleable so the next monitor cycle can re-validate.
        "tokenizer_sig_mismatch:cand=abc123def456",
        # "Earliest committer wins" — if the origin uid deregisters, the
        # collision verdict can shift to the next miner. Don't lock in.
        "plagiarism:duplicate_of_uid=12",
        # Operators can remove a hotkey from the blacklist; row must remain
        # claimable in that case.
        "blacklisted",
    ],
)
async def test_maybe_terminate_skips_externally_reversible_permanents(reason):
    """Permanent invalid reasons whose verdict depends on something outside
    the miner (champion sig, origin uid, operator config) must NOT be
    terminated in miner_stats. monitor leaves the row sampling and lets the
    next cycle re-decide once the external state changes."""
    monitor = _build_monitor()
    monitor.stats_dao = _RecordingStatsDAO()
    miner = MinerInfo(
        uid=42, hotkey="hk_rev", model="org/affine-model-hk_rev", revision="rev",
        is_valid=False, permanent_invalid=True, invalid_reason=reason,
    )
    await monitor._maybe_terminate_stats(miner)
    assert monitor.stats_dao.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        # Commit JSON on chain is immutable; once it's malformed it stays
        # malformed forever for this (hotkey, revision).
        "invalid_json_commit",
        "multiple_commits:count=3",
        "model_name_missing_affine",
        "repo_name_not_ending_with_hotkey:repo=foo",
        "revision_mismatch:hf=abc",
        "model_check:too_big",
        "duplicate_repo:from=origin/repo",
        "malicious_template:llm_audit_skipped:reason",
    ],
)
async def test_maybe_terminate_fires_for_all_listed_permanent_reasons(reason):
    monitor = _build_monitor()
    monitor.stats_dao = _RecordingStatsDAO()
    miner = MinerInfo(
        uid=42, hotkey="hk_listed", model="org/affine-model-hk_listed",
        revision="rev",
        is_valid=False, permanent_invalid=True, invalid_reason=reason,
    )
    await monitor._maybe_terminate_stats(miner)
    assert len(monitor.stats_dao.calls) == 1
    assert monitor.stats_dao.calls[0]["reason"] == reason


@pytest.mark.asyncio
async def test_maybe_terminate_swallows_dao_exception():
    """A terminate write failing must not propagate — the row already has
    is_valid=false written; the terminate will be retried next cycle."""
    monitor = _build_monitor()

    class _Boom:
        async def terminate_if_sampling(self, **_kw):
            raise RuntimeError("ddb on fire")
    monitor.stats_dao = _Boom()

    miner = MinerInfo(
        uid=42, hotkey="hk_b", model="org/affine-model-hk_b", revision="rev",
        is_valid=False, permanent_invalid=True, invalid_reason="hf_repo_not_found",
    )
    await monitor._maybe_terminate_stats(miner)  # must not raise
