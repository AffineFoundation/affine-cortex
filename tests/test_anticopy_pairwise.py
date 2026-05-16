"""CEAC pairwise math + verdict tests. Pure logic — no DDB / R2."""

from __future__ import annotations

from affine.src.anticopy.pairwise import (
    compare_scores,
    detect_copies,
    is_copy_verdict,
    sparsify_rollout,
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


def test_pairwise_identical_scores_yield_zero_decision_median():
    """Sparse format: only positions below the decision cutoff (lp < -0.5)
    contribute. Position 1 (-0.7) is the only decision position; A and B
    agree there → gap 0."""
    a = _score("A", [_rollout("k1", "CDE", [-0.5, -0.7, -0.1])])
    b = _score("B", [_rollout("k1", "CDE", [-0.5, -0.7, -0.1])])
    res = compare_scores(a, b)
    assert res.n_overlap_rollouts == 1
    assert res.n_overlap_tokens == 1          # only the decision pos counts
    assert res.per_env["CDE"].decision_median == 0.0
    assert res.per_env["CDE"].decision_n == 1


def test_pairwise_union_of_decision_positions():
    """Decision positions = union of both sides' uncertain spots; the
    missing side's lp is approximated as 0 (the model was confident
    there)."""
    # A's decision pos: idx 1 (-0.7). B's decision pos: idx 2 (-0.8).
    a = _score("A", [_rollout("k", "CDE", [-0.1, -0.7, -0.1])])
    b = _score("B", [_rollout("k", "CDE", [-0.1, -0.1, -0.8])])
    res = compare_scores(a, b)
    # Union = {1, 2}. At 1: A=-0.7, B≈0 → gap=0.7. At 2: A≈0, B=-0.8 → gap=0.8
    assert res.per_env["CDE"].decision_n == 2
    assert abs(res.per_env["CDE"].decision_median - 0.75) < 1e-6   # median(0.7,0.8)


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
    assert abs(res.decision_median_combined - 0.4) < 1e-6
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


def test_detect_copies_lookback_skips_ancient_peer():
    """Peers older than ``lookback_blocks`` are out of the comparison
    window — even a perfect match doesn't flag copy."""
    new = _score(
        "Z",
        [_rollout("k", "CDE", [-0.7] * 100)],
        first_block=100_000,
    )
    ancient = _score(   # 50_000 blocks ≈ 7 days earlier, identical lps
        "A",
        [_rollout("k", "CDE", [-0.7] * 100)],
        first_block=50_000,
    )
    # lookback 5 days × 7200 blocks/day = 36000 → ancient is OUT of window
    dec = detect_copies(
        new, new_first_block=100_000,
        all_peer_scores=[ancient],
        nll_threshold=0.004, min_overlap=10, agreement_ratio=0.5,
        lookback_blocks=36000,
    )
    assert dec.copy_of_hotkey == ""
    assert dec.closest_peer_model == ""     # ancient never compared

    # Widen to 8 days → ancient IS in window, flags copy
    dec2 = detect_copies(
        new, new_first_block=100_000,
        all_peer_scores=[ancient],
        nll_threshold=0.004, min_overlap=10, agreement_ratio=0.5,
        lookback_blocks=57_600,
    )
    assert dec2.copy_of_hotkey == "A"


def test_detect_copies_lookback_zero_means_unbounded():
    """``lookback_blocks=0`` (default) keeps prior behaviour: any
    earlier peer is fair game regardless of how old it is."""
    new = _score(
        "Z",
        [_rollout("k", "CDE", [-0.7] * 100)],
        first_block=10_000_000,
    )
    very_old = _score(
        "A",
        [_rollout("k", "CDE", [-0.7] * 100)],
        first_block=1,
    )
    dec = detect_copies(
        new, new_first_block=10_000_000,
        all_peer_scores=[very_old],
        nll_threshold=0.004, min_overlap=10, agreement_ratio=0.5,
    )
    assert dec.copy_of_hotkey == "A"


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
    # Trivial positions never enter the decision metric in the first
    # place — sparse format only stores positions where lp < CUTOFF.
    assert ec.decision_n == 1
    assert abs(ec.decision_median - 1.5) < 1e-6
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
    # Neither side has decision positions; the rollout contributes no
    # gaps, so per_env stays empty and the combined median is the
    # "no evidence" sentinel.
    assert res.per_env == {}
    assert res.decision_median_combined == -1.0
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


# ------------------------------------------------------------ sparse v3 format


def test_compare_scores_accepts_v3_sparse_rollout_directly():
    """A v3 rollout (``decision_positions`` + ``decision_lps``) compares
    bit-for-bit identically to the v2 dense form it was derived from."""
    dense_a = _rollout("k", "CDE", [-0.1, -0.7, -0.1, -1.2, -0.2])
    dense_b = _rollout("k", "CDE", [-0.1, -0.7, -0.1, -1.5, -0.2])
    sparse_a = sparsify_rollout(dense_a)
    sparse_b = sparsify_rollout(dense_b)
    # v3 carries ONLY decision_positions / decision_lps for the per-rollout
    # payload — resp_lp / resp_top must be absent so the on-disk blob shrinks.
    assert "resp_lp" not in sparse_a
    assert "resp_top" not in sparse_a
    assert sparse_a["decision_positions"] == [1, 3]
    # float32 round-trip, exact equality unsafe — compare with tolerance
    assert all(
        abs(got - exp) < 1e-5
        for got, exp in zip(sparse_a["decision_lps"], [-0.7, -1.2])
    )

    a_score = _score("A", [dense_a])
    b_score = _score("B", [dense_b])
    a_score_sparse = _score("A", [sparse_a])
    b_score_sparse = _score("B", [sparse_b])

    dense_res = compare_scores(a_score, b_score)
    sparse_res = compare_scores(a_score_sparse, b_score_sparse)
    mixed_res = compare_scores(a_score_sparse, b_score)        # v3 vs v2
    assert abs(sparse_res.decision_median_combined - dense_res.decision_median_combined) < 1e-6
    assert abs(mixed_res.decision_median_combined - dense_res.decision_median_combined) < 1e-6


def test_sparsify_rollout_is_idempotent():
    rollout = _rollout("k", "CDE", [-0.1, -0.7, -0.1])
    once = sparsify_rollout(rollout)
    twice = sparsify_rollout(once)
    assert twice is once or (
        twice["decision_positions"] == once["decision_positions"]
        and twice["decision_lps"] == once["decision_lps"]
    )
