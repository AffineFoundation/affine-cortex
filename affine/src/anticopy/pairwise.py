"""
CEAC pairwise comparison and verdict logic.

Inputs are two ``score`` payloads (the per-rollout logprob bundles
produced by the forward worker). Each entry is keyed by
``rollout_key`` (``{champion_hotkey}#{env}#{task_id}``) so the
intersection is naturally apple-to-apple — different miners scored on
the same prompt.

Sparse format (``ceac.score/v3``):

  Per-rollout entries carry only the **decision positions** —
  positions where the scoring model's own top-1 logprob fell below
  ``DECISION_LOGP_CUTOFF`` (i.e. tokens that model was uncertain
  about). All other positions are dropped at upload time, since the
  verdict math only looks at decision-position |Δlogp| anyway.

  This drops the per-blob in-memory footprint ~20× vs storing the
  full per-position resp_lp array (~8 k uncertain positions out of
  ~137 k total in production).

  Legacy ``ceac.score/v2`` blobs carry full ``resp_lp`` lists; they
  are auto-converted to sparse on the fly via :func:`_normalize_rollout`.

The pair median is computed over the **union** of the two sides'
decision positions; positions present in only one side are treated as
``lp ≈ 0`` on the missing side (the model was confident there). The
approximation error per position is bounded by
``|DECISION_LOGP_CUTOFF| = 0.5``, which the median absorbs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# Cutoff: a token position counts as "decision" only when the scoring
# model's logprob on the ground-truth token is below this value (~ top-1
# probability ≲ 0.6). Trivial-prediction positions (lp ≈ 0, model 99%+
# sure) are excluded — every Qwen3 fine-tune agrees on them, so they
# would otherwise pin every median to ~0 and mask real divergence.
DECISION_LOGP_CUTOFF = -0.5
_DECISION_LOGP_CUTOFF = DECISION_LOGP_CUTOFF        # back-compat alias


# A v3 rollout inside the score blob is shaped like::
#   {
#     "rollout_key": "<hk>#<env>#<task_id>",
#     "env": "CDE",
#     "n_tokens": 4426,                         # original total tokens (info)
#     "decision_positions": [int, ...],         # positions where lp < cutoff
#     "decision_lps":       [float, ...],       # corresponding lps
#   }
# A v2 rollout carries the full ``resp_lp: [float, ...]`` instead;
# _normalize_rollout sparsifies it on the fly.


@dataclass
class EnvCompare:
    """Per-env pairwise summary. The verdict rule only reads
    ``decision_median``; ``decision_n`` lets operators tell apart
    "no signal" (decision_n == 0) from "low signal" (decision_n small)
    when diagnosing edge cases."""
    env: str
    decision_n: int
    decision_median: float


@dataclass
class PairResult:
    n_overlap_rollouts: int
    n_overlap_tokens: int
    per_env: Dict[str, EnvCompare]
    # Median |Δlogp| pooled over every env's decision positions
    # (i.e. one big bag rather than a per-env vote). ``-1.0`` means
    # "no env produced any decision-position evidence" — the pair
    # carries no usable signal. This is the single number
    # ``is_copy_verdict`` compares against ``nll_threshold``.
    decision_median_combined: float = -1.0
    # Argmax top-1 agreement pooled over every env's decision-position
    # **intersection** (positions both models were uncertain about).
    # ``top1_n`` is the intersection size; ``top1_agree_combined`` is
    # ``matches / top1_n`` in ``[0, 1]``, or ``-1.0`` when no rollout
    # carried aligned top-1 data on both sides (the pair yields no top-1
    # signal — the gate falls back to the |Δlogp| rule).
    top1_n: int = 0
    top1_agree_combined: float = -1.0


# --------------------------------------------------------- format helpers


def _normalize_rollout(
    rollout: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(positions, lps, top1)`` as numpy arrays containing only
    the decision positions for ``rollout``. Accepts both v4/v3 (sparse)
    and v2 (full ``resp_lp`` / ``resp_top``) rollout dicts.

    ``positions`` is int32, ``lps`` float32, ``top1`` int32 (the argmax
    token id at each decision position). ``top1`` is either the same
    length as ``positions`` (top-1 data available) or **empty** — legacy
    blobs written before the top-1 field existed carry no argmax, so the
    top-1 agreement gate simply skips that rollout.

    The caller may rely on positions being sorted ascending (v4/v3
    uploads are written sorted; v2 ``np.flatnonzero`` is naturally
    sorted); ``top1`` is aligned index-for-index with ``positions``.
    """
    pos = rollout.get("decision_positions")
    lps = rollout.get("decision_lps")
    if pos is not None and lps is not None:
        pos_arr = pos if isinstance(pos, np.ndarray) else np.asarray(pos, dtype=np.int32)
        lp_arr = lps if isinstance(lps, np.ndarray) else np.asarray(lps, dtype=np.float32)
        pos_arr = pos_arr.astype(np.int32, copy=False)
        lp_arr = lp_arr.astype(np.float32, copy=False)
        top1 = rollout.get("decision_top1")
        if top1 is not None:
            top1_arr = top1 if isinstance(top1, np.ndarray) else np.asarray(top1, dtype=np.int32)
            top1_arr = top1_arr.astype(np.int32, copy=False)
        else:
            top1_arr = np.empty(0, dtype=np.int32)
        return pos_arr, lp_arr, top1_arr

    resp = rollout.get("resp_lp")
    if not resp:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    # ``None`` entries (masked tokens) drop from the comparison; convert
    # them to a value the mask can't pick up.
    arr = np.asarray(
        [float(x) if x is not None else 0.0 for x in resp],
        dtype=np.float32,
    )
    mask = arr < DECISION_LOGP_CUTOFF
    pos_arr = np.flatnonzero(mask).astype(np.int32)
    lp_arr = arr[mask].astype(np.float32, copy=False)
    top1_arr = _extract_top1(rollout.get("resp_top"), pos_arr)
    return pos_arr, lp_arr, top1_arr


def _extract_top1(
    resp_top: Optional[List[Any]], pos_arr: np.ndarray,
) -> np.ndarray:
    """Pull the argmax token id at each decision position from a v2
    ``resp_top`` list (``resp_top[i] = [[lp, token_id], ...]`` sorted by
    descending logprob, so index 0 is the argmax).

    Returns an int32 array aligned with ``pos_arr``, or an **empty**
    array if ``resp_top`` is missing or any decision position lacks a
    top-1 entry — in that case the rollout contributes no top-1 signal
    rather than a partially-aligned one.
    """
    if not resp_top or pos_arr.size == 0:
        return np.empty(0, dtype=np.int32)
    out = np.empty(pos_arr.size, dtype=np.int32)
    for i, p in enumerate(pos_arr.tolist()):
        slot = resp_top[p] if p < len(resp_top) else None
        if not slot:
            return np.empty(0, dtype=np.int32)
        try:
            out[i] = int(slot[0][1])
        except (TypeError, ValueError, IndexError):
            return np.empty(0, dtype=np.int32)
    return out


def sparsify_rollout(rollout: Dict[str, Any]) -> Dict[str, Any]:
    """Build a v3 sparse rollout dict from a v2 one (or pass v3
    through unchanged). Used by the worker at upload time so the R2
    blob carries only decision positions instead of the full ~137 k
    per-token logprob array — typically a ~20× shrink on disk."""
    if "decision_positions" in rollout and "decision_lps" in rollout:
        return rollout
    pos, lp, top1 = _normalize_rollout(rollout)
    out = {
        "rollout_key": rollout.get("rollout_key"),
        "env": rollout.get("env", ""),
        "n_tokens": rollout.get("n_tokens", 0),
        "decision_positions": [int(x) for x in pos.tolist()],
        "decision_lps": [float(x) for x in lp.tolist()],
    }
    # ``decision_top1`` (argmax token id per decision position) is only
    # emitted when the source carried usable ``resp_top`` — legacy blobs
    # without it stay top-1-less and simply skip the top-1 gate later.
    if top1.size == pos.size and top1.size > 0:
        out["decision_top1"] = [int(x) for x in top1.tolist()]
    return out


def _sparse_decision_gaps(
    pos_a: np.ndarray, lp_a: np.ndarray,
    pos_b: np.ndarray, lp_b: np.ndarray,
) -> np.ndarray:
    """Return ``|a - b|`` over the union of ``pos_a`` and ``pos_b``.
    For positions present in only one side, the missing side's lp is
    assumed to be ``0.0`` (the model was confident there, so its true
    lp lies in ``[CUTOFF, 0]`` and the approximation error per
    position is bounded by ``|CUTOFF|``)."""
    if pos_a.size == 0 and pos_b.size == 0:
        return np.empty(0, dtype=np.float32)
    union = np.union1d(pos_a, pos_b)

    if pos_a.size > 0:
        a_idx = np.searchsorted(pos_a, union)
        a_safe = np.minimum(a_idx, pos_a.size - 1)
        a_present = pos_a[a_safe] == union
        a_vals = np.where(a_present, lp_a[a_safe], np.float32(0.0))
    else:
        a_vals = np.zeros(union.size, dtype=np.float32)

    if pos_b.size > 0:
        b_idx = np.searchsorted(pos_b, union)
        b_safe = np.minimum(b_idx, pos_b.size - 1)
        b_present = pos_b[b_safe] == union
        b_vals = np.where(b_present, lp_b[b_safe], np.float32(0.0))
    else:
        b_vals = np.zeros(union.size, dtype=np.float32)

    return np.abs(a_vals - b_vals).astype(np.float32, copy=False)


def _sparse_top1_intersection(
    pos_a: np.ndarray, top1_a: np.ndarray,
    pos_b: np.ndarray, top1_b: np.ndarray,
) -> Tuple[int, int]:
    """Return ``(n_match, n_total)`` for the argmax top-1 agreement over
    the **intersection** of the two sides' decision positions.

    The intersection — positions where *both* models were uncertain — is
    the purest discriminator: independent fine-tunes diverge most there,
    so it maximises separation between "copy" and "independent" and keeps
    the gate conservative. Positions where only one side was uncertain
    are excluded (that side's argmax equals the ground-truth token there,
    which would inflate agreement toward the easy case).

    Contributes nothing (``(0, 0)``) when either side lacks aligned
    top-1 data — e.g. a legacy blob written before the field existed.
    """
    if (
        top1_a.size == 0 or top1_b.size == 0
        or top1_a.size != pos_a.size or top1_b.size != pos_b.size
    ):
        return 0, 0
    inter = np.intersect1d(pos_a, pos_b)
    if inter.size == 0:
        return 0, 0
    # ``inter`` ⊆ pos_a and ⊆ pos_b, both sorted ascending, so
    # searchsorted yields the exact aligned index into each top1 array.
    a_tok = top1_a[np.searchsorted(pos_a, inter)]
    b_tok = top1_b[np.searchsorted(pos_b, inter)]
    n_match = int(np.count_nonzero(a_tok == b_tok))
    return n_match, int(inter.size)


# --------------------------------------------------------- compare_scores


def compare_scores(
    score_a: Dict[str, Any], score_b: Dict[str, Any],
) -> PairResult:
    """Pairwise comparison over the intersection of ``rollout_key`` s.

    Returns one :class:`EnvCompare` per env that produced at least one
    decision-position gap. Caller (``is_copy_verdict``) applies
    ``min_overlap`` and ``nll_threshold``.
    """
    by_key_b: Dict[str, Dict[str, Any]] = {}
    for r in score_b.get("per_rollout", []) or []:
        k = r.get("rollout_key")
        if k:
            by_key_b[k] = r

    per_env_decision: Dict[str, List[float]] = defaultdict(list)
    n_overlap_rollouts = 0
    n_overlap_tokens = 0
    top1_match = 0
    top1_total = 0

    for ra in score_a.get("per_rollout", []) or []:
        k = ra.get("rollout_key")
        rb = by_key_b.get(k)
        if not rb:
            continue
        env = ra.get("env") or rb.get("env") or "?"
        pos_a, lp_a, t1_a = _normalize_rollout(ra)
        pos_b, lp_b, t1_b = _normalize_rollout(rb)
        gaps = _sparse_decision_gaps(pos_a, lp_a, pos_b, lp_b)
        if gaps.size == 0:
            continue
        n_overlap_rollouts += 1
        n_overlap_tokens += int(gaps.size)
        per_env_decision[env].extend(float(g) for g in gaps.tolist())
        m, tot = _sparse_top1_intersection(pos_a, t1_a, pos_b, t1_b)
        top1_match += m
        top1_total += tot

    per_env: Dict[str, EnvCompare] = {}
    all_decision_gaps: List[float] = []
    for env, gaps in per_env_decision.items():
        per_env[env] = EnvCompare(
            env=env,
            decision_n=len(gaps),
            decision_median=float(median(gaps)) if gaps else 0.0,
        )
        all_decision_gaps.extend(gaps)

    decision_median_combined = (
        float(median(all_decision_gaps)) if all_decision_gaps else -1.0
    )
    top1_agree_combined = (
        float(top1_match / top1_total) if top1_total > 0 else -1.0
    )

    return PairResult(
        n_overlap_rollouts=n_overlap_rollouts,
        n_overlap_tokens=n_overlap_tokens,
        per_env=per_env,
        decision_median_combined=decision_median_combined,
        top1_n=top1_total,
        top1_agree_combined=top1_agree_combined,
    )


def is_copy_verdict(
    pair: PairResult,
    *,
    nll_threshold: float,
    per_env_nll_thresholds: Optional[Dict[str, float]] = None,
    per_env_min_envs: int = 3,
    agreement_ratio: float = 1.0,  # noqa: ARG001 — accepted for back-compat
    top1_threshold: Optional[float] = None,
    top1_min_overlap: int = 0,
) -> bool:
    """Final verdict logic.

    A pair is a copy if **either** gate fires (OR):

    * **Top-1 gate** (when ``top1_threshold`` is set): the argmax top-1
      agreement over the decision-position intersection is at least
      ``top1_threshold`` on at least ``top1_min_overlap`` positions. Two
      copies emit the same next token even where the model is uncertain;
      independent fine-tunes diverge there. This catches copies whose
      logprobs were perturbed enough to clear the |Δlogp| bar but whose
      argmax ranking is preserved. No-op when top-1 data is unavailable
      (``top1_agree_combined < 0``) or ``top1_threshold`` is ``None``.
    * **|Δlogp| gate** (below): the decision-median rule, in one of the
      two modes selected by whether ``per_env_nll_thresholds`` is set.

    The |Δlogp| gate has two modes, selected by whether
    ``per_env_nll_thresholds`` is set:

    * **Per-env mode** (dict non-empty): a candidate is a copy iff
      every env that has decision tokens lands below its own per-env
      threshold. A single env above the bar acquits the candidate
      ("有一个差距大就不是 copy"). Requires at least
      ``per_env_min_envs`` envs in the comparison to call a verdict;
      thinner overlaps fall back to combined-median mode below.
    * **Combined-median mode** (legacy, dict empty): the pooled
      ``decision_median_combined`` is compared against
      ``nll_threshold``. Pooling across envs lets high-signal envs
      (NAVWORLD / TERMINAL with thousands of decision tokens) anchor
      the verdict while low-signal envs (MEMORY) blend in by their
      share of the union — they can't single-handedly veto a copy
      call the way per-env voting did.

    ``agreement_ratio`` is accepted for backward-compat with callers
    that still pass it; the per-env rule already supersedes the
    earlier ``X-of-Y env votes`` style of agreement gate.
    """
    # Top-1 argmax agreement gate (OR): fires independently of the
    # |Δlogp| rule below.
    if (
        top1_threshold is not None
        and pair.top1_agree_combined >= 0
        and pair.top1_n >= top1_min_overlap
        and pair.top1_agree_combined >= top1_threshold
    ):
        return True
    if per_env_nll_thresholds:
        # Walk every configured env first so ``present`` reflects the
        # full count (early-break would skew the floor check).
        configured = []
        for env, ec in pair.per_env.items():
            if ec.decision_n <= 0:
                continue
            threshold = per_env_nll_thresholds.get(env)
            if threshold is None:
                # Env isn't in the per-env config — don't let an
                # unconfigured env veto or trigger a verdict.
                continue
            configured.append((ec.decision_median, threshold))
        if len(configured) >= per_env_min_envs:
            return all(med < t for med, t in configured)
        # Too few envs with data — fall through to combined median
        # so a thin comparison still gets a verdict rather than
        # silently defaulting to "clean".
    return (
        pair.decision_median_combined >= 0
        and pair.decision_median_combined < nll_threshold
    )


# --------------------------------------------------------------- detect_copies


@dataclass
class CopyDecision:
    # The peer the candidate is judged a copy of. ``""`` means
    # independent (no peer crossed ``nll_threshold``).
    copy_of_hotkey: str = ""
    # The peer with the smallest ``decision_median_combined`` against
    # this candidate, whether or not the threshold was crossed. Stored
    # as the HuggingFace model string (``org/repo``) rather than
    # hotkey so operators reading the row don't need to join against
    # ``miners`` to see who is similar. Empty when no peer carried
    # any decision-position evidence.
    closest_peer_model: str = ""
    # Combined cross-env decision median against the winning peer
    # (or the closest one if no peer crossed the threshold). ``-1.0``
    # is the "no peer had evidence" sentinel.
    decision_median: float = -1.0
    # Per-env breakdown of the winning (or closest) peer pair.
    # ``{env_name: median |Δlogp|}`` for envs with ≥1 decision token.
    # Empty when no peer produced usable evidence.
    decision_per_env: Dict[str, float] = field(default_factory=dict)
    # Argmax top-1 agreement (decision-position intersection) against
    # the winning (or closest) peer, in ``[0, 1]``. ``-1.0`` when no
    # peer carried aligned top-1 data. Persisted for operator triage —
    # tells apart a top-1-gated verdict from a |Δlogp|-gated one.
    top1_agreement: float = -1.0


def _per_env_decision_medians(pair: PairResult) -> Dict[str, float]:
    """Per-env ``decision_median`` snapshot. Envs with zero decision
    tokens drop out so the diagnostic only carries envs that voted."""
    return {
        env: float(ec.decision_median)
        for env, ec in pair.per_env.items()
        if ec.decision_n > 0
    }


def _aggregate_decision_median(pair: PairResult) -> float:
    """Combined ``decision_median`` for a pair — the median over the
    union of every env's decision positions (i.e. one big bag rather
    than a per-env vote). This is the number ``is_copy_verdict``
    compares against ``nll_threshold`` and the number we persist on
    the score row."""
    return float(pair.decision_median_combined)


def detect_copies(
    new_score: Dict[str, Any],
    new_first_block: int,
    all_peer_scores: Iterable[Dict[str, Any]],
    *,
    nll_threshold: float,
    min_overlap: int,
    agreement_ratio: float,
    lookback_blocks: int = 0,
    top1_threshold: Optional[float] = None,
    top1_min_overlap: int = 0,
) -> CopyDecision:
    """For a freshly computed score, scan every active peer that
    committed earlier (lower ``first_block``) and return the earliest
    one we flag as the same model.

    Ties on ``first_block`` resolve in favour of the lower ``hotkey``
    (lexicographic), matching ``miners_monitor._detect_plagiarism``.

    ``lookback_blocks`` (default 0 = unbounded) caps how far back the
    scan reaches. Peers whose ``first_block`` is more than that many
    blocks earlier than the candidate are skipped — stale "previous
    season" miners aren't realistic copy origins for a fresh upload
    and only add noise + comparison cost.

    ``CopyDecision.decision_median`` is populated from the WINNING peer
    (when one exists) or from the closest peer otherwise — i.e. the
    smallest ``decision_median`` observed during the scan. Callers that
    persist verdicts to a DDB index can store this for diagnostics
    without re-running pairwise.
    """
    decision = CopyDecision()
    new_hotkey = new_score.get("hotkey", "")

    closest_pair: Optional[PairResult] = None
    closest_peer_model: str = ""
    closest_median = -1.0
    candidates: List[Tuple[int, str, Dict[str, Any], PairResult]] = []
    for peer in all_peer_scores:
        peer_hotkey = peer.get("hotkey", "")
        if not peer_hotkey or peer_hotkey == new_hotkey:
            continue
        peer_first_block = int(peer.get("first_block", 0) or 0)
        if (peer_first_block, peer_hotkey) >= (new_first_block, new_hotkey):
            continue                      # later or same → cannot be the origin
        if (
            lookback_blocks > 0
            and peer_first_block > 0
            and (new_first_block - peer_first_block) > lookback_blocks
        ):
            continue                      # too old → not a realistic origin
        pair = compare_scores(new_score, peer)
        pair_med = _aggregate_decision_median(pair)
        if pair_med >= 0 and (closest_median < 0 or pair_med < closest_median):
            closest_median = pair_med
            closest_pair = pair
            closest_peer_model = str(peer.get("model", ""))
        if pair.n_overlap_tokens < min_overlap:
            continue
        if not is_copy_verdict(
            pair, nll_threshold=nll_threshold, agreement_ratio=agreement_ratio,
            top1_threshold=top1_threshold, top1_min_overlap=top1_min_overlap,
        ):
            continue
        candidates.append((peer_first_block, peer_hotkey, peer, pair))

    # ``closest_peer_model`` records who the candidate is most similar
    # to even when no peer crossed the threshold — useful for "this
    # almost flagged copy_of X" diagnostics.
    decision.closest_peer_model = closest_peer_model

    if not candidates:
        decision.decision_median = closest_median
        if closest_pair is not None:
            decision.decision_per_env = _per_env_decision_medians(closest_pair)
            decision.top1_agreement = float(closest_pair.top1_agree_combined)
        return decision

    # Earliest committer wins (smallest (first_block, hotkey)).
    candidates.sort(key=lambda x: (x[0], x[1]))
    winning_pair = candidates[0][3]
    winning_peer = candidates[0][2]
    decision.copy_of_hotkey = candidates[0][1]
    decision.decision_median = _aggregate_decision_median(winning_pair)
    decision.decision_per_env = _per_env_decision_medians(winning_pair)
    decision.top1_agreement = float(winning_pair.top1_agree_combined)
    # When a winner exists, the closest peer IS the winner. Reset so
    # the two fields agree.
    decision.closest_peer_model = str(winning_peer.get("model", ""))
    return decision
