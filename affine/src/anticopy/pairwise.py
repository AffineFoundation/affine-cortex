"""
CEAC pairwise comparison and verdict logic.

Inputs are two ``score`` payloads (the per-rollout logprob bundles
produced by the forward worker). Each entry is keyed by
``rollout_key`` (``{champion_hotkey}#{env}#{task_id}``) so the
intersection is naturally apple-to-apple — different miners scored on
the same prompt.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


# A ``rollout`` inside the score blob is shaped like::
#   {
#     "rollout_key": "<hk>#<env>#<task_id>",
#     "env": "CDE",
#     "resp_lp":  [float, ...]                     # per-token logprob
#     "resp_top": [[[lp, tid], ...], ...]          # optional top-K per pos
#   }


@dataclass
class EnvCompare:
    env: str
    n_tokens: int
    nll_median: float       # median |Δlogp| over ALL positions (diagnostic)
    nll_mean: float
    top1_match: float       # 0..1, fraction of positions where top-1 token id agreed
    # ``decision_median`` is the median |Δlogp| restricted to positions
    # where the reference model's top-1 logprob was below -0.5 — i.e.
    # tokens the reference model was genuinely uncertain about. These
    # are the positions where weight differences actually surface; the
    # overwhelming mass of "trivial" positions (punctuation, BOS, format
    # tokens) where any fine-tune predicts the same thing would
    # otherwise pin ``nll_median`` to ~0 and mask real divergence.
    decision_n: int
    decision_median: float


@dataclass
class PairResult:
    n_overlap_rollouts: int
    n_overlap_tokens: int
    per_env: Dict[str, EnvCompare]


# Reference logprob below this threshold marks a "decision" position —
# the model was uncertain (top1 probability ≲ 0.6), so weight changes
# actually move which token gets picked. Trivial-prediction positions
# (lp ≈ 0, model 99%+ sure) are excluded from the decision metric.
_DECISION_LOGP_CUTOFF = -0.5


def _per_token_diffs(
    a_lp: List[float], b_lp: List[float],
    a_top: Optional[List[List[List[Any]]]] = None,
    b_top: Optional[List[List[List[Any]]]] = None,
) -> Tuple[List[float], List[int], List[float]]:
    """Return ``(all_gaps, top1_match, decision_gaps)``.

    ``decision_gaps`` is the |a-b| series restricted to positions where
    the reference (``a``) logprob is below ``_DECISION_LOGP_CUTOFF`` —
    i.e. tokens the reference model wasn't confident about. Empirically
    those positions carry the signal: trivial-prediction positions
    agree across every fine-tune and would otherwise dominate the
    overall median.
    """
    gaps: List[float] = []
    decision_gaps: List[float] = []
    top1: List[int] = []
    n = min(len(a_lp), len(b_lp))
    for i in range(n):
        a, b = a_lp[i], b_lp[i]
        if a is None or b is None:
            continue
        g = abs(float(a) - float(b))
        gaps.append(g)
        if float(a) < _DECISION_LOGP_CUTOFF:
            decision_gaps.append(g)
        if a_top is None or b_top is None:
            continue
        if i >= len(a_top) or i >= len(b_top):
            continue
        ta, tb = a_top[i], b_top[i]
        if ta and tb:
            try:
                top1.append(1 if int(ta[0][1]) == int(tb[0][1]) else 0)
            except (IndexError, TypeError, ValueError):
                pass
    return gaps, top1, decision_gaps


def compare_scores(
    score_a: Dict[str, Any], score_b: Dict[str, Any]
) -> PairResult:
    """Pairwise comparison over the intersection of rollout_keys.

    Returns one :class:`EnvCompare` per env that had at least one
    overlapping rollout. Caller is responsible for applying
    ``min_overlap`` / threshold logic."""
    by_key_b: Dict[str, Dict[str, Any]] = {}
    for r in score_b.get("per_rollout", []) or []:
        k = r.get("rollout_key")
        if k:
            by_key_b[k] = r

    per_env_gaps: Dict[str, List[float]] = defaultdict(list)
    per_env_top1: Dict[str, List[int]] = defaultdict(list)
    per_env_decision: Dict[str, List[float]] = defaultdict(list)
    n_overlap_rollouts = 0
    n_overlap_tokens = 0

    for ra in score_a.get("per_rollout", []) or []:
        k = ra.get("rollout_key")
        rb = by_key_b.get(k)
        if not rb:
            continue
        env = ra.get("env") or rb.get("env") or "?"
        gaps, top1, decision = _per_token_diffs(
            ra.get("resp_lp") or [],
            rb.get("resp_lp") or [],
            ra.get("resp_top"),
            rb.get("resp_top"),
        )
        if not gaps:
            continue
        n_overlap_rollouts += 1
        n_overlap_tokens += len(gaps)
        per_env_gaps[env].extend(gaps)
        per_env_top1[env].extend(top1)
        per_env_decision[env].extend(decision)

    per_env: Dict[str, EnvCompare] = {}
    for env, gaps in per_env_gaps.items():
        top1 = per_env_top1.get(env, [])
        decision = per_env_decision.get(env, [])
        per_env[env] = EnvCompare(
            env=env,
            n_tokens=len(gaps),
            nll_median=float(median(gaps)),
            nll_mean=float(sum(gaps) / len(gaps)),
            top1_match=(sum(top1) / len(top1)) if top1 else 0.0,
            decision_n=len(decision),
            decision_median=float(median(decision)) if decision else 0.0,
        )

    return PairResult(
        n_overlap_rollouts=n_overlap_rollouts,
        n_overlap_tokens=n_overlap_tokens,
        per_env=per_env,
    )


def is_copy_verdict(
    pair: PairResult, *, nll_threshold: float, agreement_ratio: float
) -> bool:
    """Final verdict: ``copy`` iff at least
    ``ceil(n_envs * agreement_ratio)`` envs land below ``nll_threshold``
    on the decision-position metric.

    The threshold is applied to ``decision_median`` (|Δlogp| over
    positions where the reference model was uncertain), not the
    all-positions median. Calibration runs show that fine-tunes
    overlap at lp≈0 on the bulk of "trivial" tokens, so the
    all-positions median pins to ~0 even for clearly different models
    — it is kept for diagnostics but not used here.
    """
    # Only count envs where we actually observed decision-position
    # tokens; an env with zero "uncertain" positions carries no signal
    # — its trivially-empty ``decision_median`` would otherwise vote
    # "copy" by default, which is the wrong direction for evidence.
    envs_with_signal = [ec for ec in pair.per_env.values() if ec.decision_n > 0]
    if not envs_with_signal:
        return False
    n_copy = sum(
        1 for ec in envs_with_signal if ec.decision_median < nll_threshold
    )
    needed = max(1, math.ceil(len(envs_with_signal) * agreement_ratio))
    return n_copy >= needed


# --------------------------------------------------------------- detect_copies


@dataclass
class CopyDecision:
    copy_of_hotkey: str = ""        # "" means independent / not enough data
    # Aggregate ``decision_median`` across envs (MIN of per-env values).
    # ``-1.0`` is the "no peer had evidence" sentinel. Kept for fast
    # threshold filtering (one float per row); ``decision_per_env`` is
    # the diagnostic source of truth.
    decision_median: float = -1.0
    # Per-env breakdown of the winning (or closest) peer pair.
    # ``{env_name: median |Δlogp|}`` for envs with ≥1 decision token.
    # Empty when no peer produced usable evidence.
    decision_per_env: Dict[str, float] = field(default_factory=dict)


def _per_env_decision_medians(pair: PairResult) -> Dict[str, float]:
    """Per-env ``decision_median`` snapshot. Envs with zero decision
    tokens drop out so the diagnostic only carries envs that voted."""
    return {
        env: float(ec.decision_median)
        for env, ec in pair.per_env.items()
        if ec.decision_n > 0
    }


def _aggregate_decision_median(pair: PairResult) -> float:
    """Best (smallest) ``decision_median`` across envs of a pair.

    We take the MIN per env rather than the mean because a single env
    landing under the threshold is what flips ``is_copy_verdict`` when
    ``agreement_ratio`` is low; reporting that env's median makes the
    stored number consistent with the verdict reason."""
    vals = list(_per_env_decision_medians(pair).values())
    if not vals:
        return -1.0
    return float(min(vals))


def detect_copies(
    new_score: Dict[str, Any],
    new_first_block: int,
    all_peer_scores: Iterable[Dict[str, Any]],
    *,
    nll_threshold: float,
    min_overlap: int,
    agreement_ratio: float,
) -> CopyDecision:
    """For a freshly computed score, scan every active peer that
    committed earlier (lower ``first_block``) and return the earliest
    one we flag as the same model.

    Ties on ``first_block`` resolve in favour of the lower ``hotkey``
    (lexicographic), matching ``miners_monitor._detect_plagiarism``.

    ``CopyDecision.decision_median`` is populated from the WINNING peer
    (when one exists) or from the closest peer otherwise — i.e. the
    smallest ``decision_median`` observed during the scan. Callers that
    persist verdicts to a DDB index can store this for diagnostics
    without re-running pairwise.
    """
    decision = CopyDecision()
    new_hotkey = new_score.get("hotkey", "")

    closest_pair: Optional[PairResult] = None
    closest_median = -1.0
    candidates: List[Tuple[int, str, Dict[str, Any], PairResult]] = []
    for peer in all_peer_scores:
        peer_hotkey = peer.get("hotkey", "")
        if not peer_hotkey or peer_hotkey == new_hotkey:
            continue
        peer_first_block = int(peer.get("first_block", 0) or 0)
        if (peer_first_block, peer_hotkey) >= (new_first_block, new_hotkey):
            continue                      # later or same → cannot be the origin
        pair = compare_scores(new_score, peer)
        pair_med = _aggregate_decision_median(pair)
        if pair_med >= 0 and (closest_median < 0 or pair_med < closest_median):
            closest_median = pair_med
            closest_pair = pair
        if pair.n_overlap_tokens < min_overlap:
            continue
        if not is_copy_verdict(
            pair, nll_threshold=nll_threshold, agreement_ratio=agreement_ratio
        ):
            continue
        candidates.append((peer_first_block, peer_hotkey, peer, pair))

    if not candidates:
        decision.decision_median = closest_median
        if closest_pair is not None:
            decision.decision_per_env = _per_env_decision_medians(closest_pair)
        return decision

    # Earliest committer wins (smallest (first_block, hotkey)).
    candidates.sort(key=lambda x: (x[0], x[1]))
    winning_pair = candidates[0][3]
    decision.copy_of_hotkey = candidates[0][1]
    decision.decision_median = _aggregate_decision_median(winning_pair)
    decision.decision_per_env = _per_env_decision_medians(winning_pair)
    return decision
