"""CEAC pairwise math + verdict tests. Pure logic — no DDB / R2."""

from __future__ import annotations

from affine.src.anticopy.pairwise import (
    compare_scores,
    detect_copies,
    is_copy_verdict,
)


def _rollout(rollout_key: str, env: str, lps, top=None):
    return {
        "rollout_key": rollout_key,
        "env": env,
        "resp_lp": list(lps),
        "resp_top": top or [],
    }


def _score(hotkey: str, rollouts, first_block: int = 0):
    return {
        "schema": "ceac.score/v1",
        "hotkey": hotkey,
        "revision": "rev_" + hotkey,
        "first_block": first_block,
        "per_rollout": list(rollouts),
    }


# ------------------------------------------------------------ pairwise math


def test_pairwise_matches_only_shared_rollout_keys():
    """No common rollout_key → no per-env entry."""
    a = _score("A", [_rollout("k1", "CDE", [0.1, 0.2])])
    b = _score("B", [_rollout("k2", "CDE", [0.1, 0.2])])
    res = compare_scores(a, b)
    assert res.n_overlap_rollouts == 0
    assert res.n_overlap_tokens == 0
    assert res.per_env == {}


def test_pairwise_identical_scores_yield_zero_nll():
    a = _score("A", [_rollout("k1", "CDE", [-0.5, -0.7, -0.1])])
    b = _score("B", [_rollout("k1", "CDE", [-0.5, -0.7, -0.1])])
    res = compare_scores(a, b)
    assert res.n_overlap_rollouts == 1
    assert res.n_overlap_tokens == 3
    assert res.per_env["CDE"].nll_median == 0.0
    assert res.per_env["CDE"].nll_mean == 0.0


def test_pairwise_skips_none_positions():
    """``None`` logprobs (e.g. masked-out tokens) drop from the diff."""
    a = _score("A", [_rollout("k1", "CDE", [-0.5, None, -0.1])])
    b = _score("B", [_rollout("k1", "CDE", [-0.5, -0.7, -0.1])])
    res = compare_scores(a, b)
    assert res.n_overlap_tokens == 2          # the None pos was dropped


def test_pairwise_top1_match_rate():
    a_top = [[[-0.3, 1]], [[-0.4, 2]]]
    b_top_same = [[[-0.3, 1]], [[-0.4, 2]]]
    b_top_one_diff = [[[-0.3, 1]], [[-0.4, 99]]]
    a = _score("A", [_rollout("k", "CDE", [0.0, 0.0], top=a_top)])
    b1 = _score("B", [_rollout("k", "CDE", [0.0, 0.0], top=b_top_same)])
    b2 = _score("C", [_rollout("k", "CDE", [0.0, 0.0], top=b_top_one_diff)])
    assert compare_scores(a, b1).per_env["CDE"].top1_match == 1.0
    assert compare_scores(a, b2).per_env["CDE"].top1_match == 0.5


# ------------------------------------------------------------ verdict


def test_verdict_below_threshold_flags_copy():
    """Single env, all logprobs identical → decision_median 0 → copy.

    Uses lp ≲ -0.5 throughout so positions register as "decision"
    candidates (the metric ignores trivial-prediction tokens where the
    reference model was 99%+ certain).
    """
    res = compare_scores(
        _score("A", [_rollout("k", "CDE", [-0.7, -0.7, -0.7])]),
        _score("B", [_rollout("k", "CDE", [-0.7, -0.7, -0.7])]),
    )
    assert is_copy_verdict(res, nll_threshold=0.004, agreement_ratio=0.5)


def test_verdict_above_threshold_independent():
    res = compare_scores(
        _score("A", [_rollout("k", "CDE", [-0.1, -0.5, -0.2])]),
        _score("B", [_rollout("k", "CDE", [-0.9, -1.5, -1.0])]),   # ~0.8 NLL
    )
    assert not is_copy_verdict(res, nll_threshold=0.004, agreement_ratio=0.5)


def test_verdict_combined_median_pools_across_envs():
    """Combined-median rule: two envs are pooled into one median. A
    low-signal env can't single-handedly flip the verdict the way it
    could under per-env voting."""
    res = compare_scores(
        _score(
            "A",
            [
                _rollout("k1", "CDE", [-0.7, -0.7]),               # 0 diff
                _rollout("k2", "MTH", [-0.7, -0.7]),               # big diff
            ],
        ),
        _score(
            "B",
            [
                _rollout("k1", "CDE", [-0.7, -0.7]),
                _rollout("k2", "MTH", [-1.5, -1.5]),
            ],
        ),
    )
    # Decision-position gaps pooled across envs: [0, 0, 0.8, 0.8]
    # → combined median = 0.4. agreement_ratio is now ignored.
    assert abs(res.decision_median_combined - 0.4) < 1e-9
    assert not is_copy_verdict(res, nll_threshold=0.04)
    # When the threshold is moved above the combined, verdict flips.
    assert is_copy_verdict(res, nll_threshold=0.5)


# ------------------------------------------------------------ detect_copies


def test_detect_copies_picks_earliest_committer():
    new = _score(
        "Z",
        [_rollout("k", "CDE", [-0.7] * 60)],
        first_block=1000,
    )
    early = _score(
        "A",
        [_rollout("k", "CDE", [-0.7] * 60)],
        first_block=500,
    )
    later = _score(
        "B",
        [_rollout("k", "CDE", [-0.7] * 60)],
        first_block=800,
    )
    dec = detect_copies(
        new, new_first_block=1000,
        all_peer_scores=[later, early],
        nll_threshold=0.004,
        min_overlap=10,
        agreement_ratio=0.5,
    )
    assert dec.copy_of_hotkey == "A"
    # winning pair was a perfect match → decision_median = 0
    assert dec.decision_median == 0.0


def test_detect_copies_below_min_overlap_returns_empty():
    new = _score(
        "Z",
        # lp ≲ -0.5 to register as decision positions; only 5 tokens so
        # the n_overlap check still fails.
        [_rollout("k", "CDE", [-0.7] * 5)],
        first_block=1000,
    )
    peer = _score(
        "A",
        [_rollout("k", "CDE", [-0.7] * 5)],
        first_block=500,
    )
    dec = detect_copies(
        new, new_first_block=1000,
        all_peer_scores=[peer],
        nll_threshold=0.004,
        min_overlap=50,
        agreement_ratio=0.5,
    )
    assert dec.copy_of_hotkey == ""           # too few tokens to commit
    # closest pair was a perfect match — diagnostic median still
    # reported so an operator can see how close the candidate was
    assert dec.decision_median == 0.0


def test_detect_copies_ignores_later_peers():
    """A peer that committed *after* the candidate can't be the origin."""
    new = _score(
        "Z",
        [_rollout("k", "CDE", [-0.5] * 100)],
        first_block=500,
    )
    later = _score(
        "L",
        [_rollout("k", "CDE", [-0.5] * 100)],
        first_block=900,
    )
    dec = detect_copies(
        new, new_first_block=500,
        all_peer_scores=[later],
        nll_threshold=0.004,
        min_overlap=10,
        agreement_ratio=0.5,
    )
    assert dec.copy_of_hotkey == ""


# --- decision-metric coverage -----------------------------------------------


def test_decision_median_ignores_trivial_positions():
    """Trivial positions (ref lp ≈ 0) dominate the all-positions median
    even when the two models diverge wildly on the few "decision" tokens.
    The decision-metric should isolate the divergent positions and NOT
    flip the verdict to copy just because most positions are easy."""
    # Reference: 9 trivial tokens (lp=-0.01, model 99%+ sure) + 1 decision
    # token (lp=-1.0). Peer matches on the trivial 9, diverges by 1.5 logp
    # on the decision token.
    a_lp = [-0.01] * 9 + [-1.0]
    b_lp = [-0.01] * 9 + [-2.5]               # 1.5 diff on the decision pos
    res = compare_scores(
        _score("A", [_rollout("k", "CDE", a_lp)]),
        _score("B", [_rollout("k", "CDE", b_lp)]),
    )
    ec = res.per_env["CDE"]
    # All-positions median is dragged to 0 by the 9 identical trivial tokens
    assert ec.nll_median == 0.0
    # But the decision-position metric isolates the diverging token
    assert ec.decision_n == 1
    assert abs(ec.decision_median - 1.5) < 1e-9
    # → not a copy under the new metric (1.5 ≫ 0.05)
    assert not is_copy_verdict(res, nll_threshold=0.05, agreement_ratio=0.5)


def test_decision_metric_no_uncertain_positions_means_no_copy():
    """When *every* position is trivial (lp ≳ -0.5), there's no signal
    to vote with; the verdict must be ``not copy`` rather than a
    default-to-copy from an empty decision set."""
    res = compare_scores(
        _score("A", [_rollout("k", "CDE", [-0.01, -0.02, -0.03])]),
        _score("B", [_rollout("k", "CDE", [-0.01, -0.02, -0.03])]),
    )
    assert res.per_env["CDE"].decision_n == 0
    assert not is_copy_verdict(res, nll_threshold=0.05, agreement_ratio=0.5)


def test_detect_copies_populates_per_env_breakdown():
    """The winning pair's per-env decision medians get surfaced on
    ``CopyDecision.decision_per_env`` so operators can diagnose which
    envs flipped the verdict without re-running pairwise."""
    new = _score(
        "Z",
        [
            _rollout("k1", "CDE", [-0.7] * 60),
            _rollout("k2", "MTH", [-0.7] * 60),
        ],
        first_block=1000,
    )
    earlier = _score(
        "A",
        [
            _rollout("k1", "CDE", [-0.7] * 60),
            _rollout("k2", "MTH", [-0.7] * 60),
        ],
        first_block=500,
    )
    dec = detect_copies(
        new, new_first_block=1000,
        all_peer_scores=[earlier],
        nll_threshold=0.05, min_overlap=10, agreement_ratio=1.0,
    )
    assert dec.copy_of_hotkey == "A"
    assert set(dec.decision_per_env.keys()) == {"CDE", "MTH"}
    for env, med in dec.decision_per_env.items():
        assert med == 0.0, f"identical lps in {env} should have med=0"


def test_detect_copies_per_env_from_closest_when_no_winner():
    """No peer crosses the threshold → still report the *closest*
    pair's per-env breakdown so an operator can see how far it was."""
    new = _score(
        "Z",
        [_rollout("k", "CDE", [-0.7] * 60)],
        first_block=1000,
    )
    far = _score(
        "A",
        [_rollout("k", "CDE", [-2.5] * 60)],     # big gap → not copy
        first_block=500,
    )
    dec = detect_copies(
        new, new_first_block=1000,
        all_peer_scores=[far],
        nll_threshold=0.05, min_overlap=10, agreement_ratio=1.0,
    )
    assert dec.copy_of_hotkey == ""
    # Closest-pair's per-env survived even without a winner.
    assert dec.decision_per_env.get("CDE", -1) > 0.05
