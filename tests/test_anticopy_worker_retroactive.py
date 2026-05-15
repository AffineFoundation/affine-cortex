"""``ForwardWorker._refresh_later_peer_verdicts`` retroactive update.

When candidates are scored out of commit order (priority overrides,
retries, parallel workers), the earliest-committer wins rule may
flag a peer as ``copy_of_X`` when the real origin ``Y`` only arrives
in the queue later. After Y is scored, the worker must walk forward
and refresh those stale verdicts.
"""

from __future__ import annotations

import pytest

from affine.src.anticopy.threshold import AntiCopyConfig
from affine.src.anticopy.worker import ForwardWorker


class _FakeScoresDAO:
    def __init__(self):
        self.update_calls = []

    async def update_verdict(
        self, hotkey, revision, *,
        copy_of, decision_median, decision_per_env=None,
    ):
        self.update_calls.append({
            "hotkey": hotkey, "revision": revision,
            "copy_of": copy_of,
            "decision_median": decision_median,
            "decision_per_env": decision_per_env or {},
        })


def _rollout(rollout_key, env, lps):
    """Build a per-rollout entry with explicit decision-zone logprobs.

    The decision metric only triggers on positions with lp < -0.5, so
    we plant -0.7's to populate ``decision_n``."""
    return {
        "rollout_key": rollout_key,
        "env": env,
        "resp_lp": list(lps),
        "resp_top": [],
    }


def _score(hotkey, lps):
    return {
        "schema": "ceac.score/v2",
        "hotkey": hotkey,
        "revision": "rev_" + hotkey,
        "model": "m_" + hotkey,
        "tokenizer_sig": "S" * 64,
        "computed_at": 1,
        "per_rollout": [_rollout("k1", "CDE", lps)],
    }


def _worker_with_fake_dao(scores_dao):
    return ForwardWorker(
        jobs_dao=object(),
        rollouts_dao=object(),
        scores_dao=scores_dao,
        miners_dao=object(),
        config_dao=object(),
    )


@pytest.mark.asyncio
async def test_later_peer_verdict_refreshed_when_earlier_origin_arrives():
    """Peer P committed AFTER new candidate N, and pairwise(P, N) is a
    copy. P's existing verdict must be refreshed to ``copy_of=N``."""
    cfg = AntiCopyConfig(
        nll_threshold=0.05, min_overlap=2, agreement_ratio=0.5,
    )
    dao = _FakeScoresDAO()
    worker = _worker_with_fake_dao(dao)

    # Identical logprobs (decision-position pair |Δ|=0 → copy verdict)
    peer = _score("P_later", [-0.7] * 60)
    peer["first_block"] = 2000

    new_score = _score("N_earlier", [-0.7] * 60)

    await worker._refresh_later_peer_verdicts(
        cfg=cfg,
        new_score=new_score,
        new_first_block=1000,
        peer_scores=[peer],
    )
    assert len(dao.update_calls) == 1
    call = dao.update_calls[0]
    assert call["hotkey"] == "P_later"
    assert call["copy_of"] == "N_earlier"


@pytest.mark.asyncio
async def test_earlier_peer_left_untouched():
    """A peer that committed *before* the new candidate is never the
    target of a retroactive update — only future copies are."""
    cfg = AntiCopyConfig(
        nll_threshold=0.05, min_overlap=2, agreement_ratio=0.5,
    )
    dao = _FakeScoresDAO()
    worker = _worker_with_fake_dao(dao)

    peer = _score("P_early", [-0.7] * 60)
    peer["first_block"] = 500
    new_score = _score("N_late", [-0.7] * 60)

    await worker._refresh_later_peer_verdicts(
        cfg=cfg,
        new_score=new_score,
        new_first_block=1000,
        peer_scores=[peer],
    )
    assert dao.update_calls == []


@pytest.mark.asyncio
async def test_non_copy_peer_left_untouched():
    """A peer that is genuinely different from the new candidate (above
    the threshold) is not touched — only suspected copies are
    re-evaluated."""
    cfg = AntiCopyConfig(
        nll_threshold=0.05, min_overlap=2, agreement_ratio=0.5,
    )
    dao = _FakeScoresDAO()
    worker = _worker_with_fake_dao(dao)

    peer = _score("P_later", [-0.7] * 60)
    peer["first_block"] = 2000
    # >> 0.05 logp gap on every (decision) position
    new_score = _score("N_earlier", [-2.5] * 60)

    await worker._refresh_later_peer_verdicts(
        cfg=cfg,
        new_score=new_score,
        new_first_block=1000,
        peer_scores=[peer],
    )
    assert dao.update_calls == []
