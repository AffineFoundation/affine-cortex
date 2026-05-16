"""CEAC verdict backfill service.

The forward worker writes ``anticopy_scores_index`` rows with just the
R2 score blob — it does NOT compute verdicts itself. Verdict math is
CPU + R2-I/O bound and was the hot spot bringing the GPU-side worker
to its knees (~7 min per candidate when every R2 fetch happened in
the worker's own request path, plus a 4 GB cgroup that was OOMing
under the bulk peer load).

This service runs alongside :class:`RolloutRefreshService` in the
``anticopy-refresh`` container. Every tick it:

  1. Loads every score blob from R2 **once** into an in-memory dict
     (resp_top stripped) so the per-candidate compare phase is pure
     CPU on already-resident bytes.
  2. Scans ``scores_index`` for rows that don't yet have
     ``verdict_at`` populated and runs the streaming verdict for each
     against the cached peer set.
  3. Writes the resulting copy_of / decision_median /
     decision_per_env / closest_peer_model back via
     ``AntiCopyScoresIndexDAO.update_verdict``.

Memory: one full pass loads ~90 × 4 MB ≈ 360 MB (well under typical
refresh container limits). The cache is rebuilt every tick to pick up
peers added since the last pass.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from affine.core.setup import logger
from affine.database.dao.anticopy import AntiCopyScoresIndexDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.src.anticopy.pairwise import (
    CopyDecision,
    PairResult,
    _per_env_decision_medians,
    compare_scores,
    is_copy_verdict,
)
from affine.src.anticopy.r2 import AntiCopyR2
from affine.src.anticopy.threshold import (
    AntiCopyConfig,
    BLOCKS_PER_DAY,
    load_anticopy_config,
)


def _strip_resp_top(blob: Dict[str, Any]) -> Dict[str, Any]:
    """``resp_top`` is the per-position top-K matrix and is ~10× the
    resp_lp footprint per rollout. It only feeds the diagnostic
    ``top1_match`` field — the verdict rule never reads it — so
    drop it on the way in."""
    for ro in (blob.get("per_rollout") or []):
        ro.pop("resp_top", None)
    return blob


def _pick_origin(
    *,
    cfg: AntiCopyConfig,
    ref_score: Dict[str, Any],
    ref_first_block: int,
    peer_cache: Dict[Tuple[str, str], Dict[str, Any]],
) -> CopyDecision:
    """In-memory equivalent of ``detect_copies`` that early-exits on
    the earliest threshold-crossing peer.

    ``peer_cache`` is keyed by ``(hotkey, revision)`` and contains
    score blobs with ``first_block`` already attached.
    """
    lookback_blocks = cfg.verdict_lookback_days * BLOCKS_PER_DAY
    ref_hk = ref_score.get("hotkey", "")
    ref_rev = ref_score.get("revision", "")

    # Sort peers by (first_block, hotkey) so first threshold-crosser
    # is the earliest committer — i.e. the winner.
    items: List[Tuple[int, str, Dict[str, Any]]] = []
    for (hk, _rev), peer in peer_cache.items():
        if hk == ref_hk and _rev == ref_rev:
            continue
        fb = int(peer.get("first_block", 0) or 0)
        items.append((fb, hk, peer))
    items.sort(key=lambda x: (x[0], x[1]))

    closest_median = -1.0
    closest_per_env: Dict[str, float] = {}
    closest_peer_model: str = ""
    winning_pair: Optional[PairResult] = None
    winning_hk: str = ""
    winning_model: str = ""

    for fb, hk, peer in items:
        if not hk or hk == ref_hk:
            continue
        if (fb, hk) >= (ref_first_block, ref_hk):
            continue
        if (
            lookback_blocks > 0 and fb > 0
            and (ref_first_block - fb) > lookback_blocks
        ):
            continue

        pair = compare_scores(ref_score, peer)
        pair_med = pair.decision_median_combined
        peer_model = str(peer.get("model", ""))
        if pair_med >= 0 and (closest_median < 0 or pair_med < closest_median):
            closest_median = pair_med
            closest_per_env = _per_env_decision_medians(pair)
            closest_peer_model = peer_model
        if (
            pair.n_overlap_tokens >= cfg.min_overlap
            and is_copy_verdict(
                pair,
                nll_threshold=cfg.nll_threshold,
                agreement_ratio=cfg.agreement_ratio,
            )
        ):
            winning_pair = pair
            winning_hk = hk
            winning_model = peer_model
            break

    decision = CopyDecision()
    if winning_pair is not None:
        decision.copy_of_hotkey = winning_hk
        decision.decision_median = float(winning_pair.decision_median_combined)
        decision.decision_per_env = _per_env_decision_medians(winning_pair)
        decision.closest_peer_model = winning_model
    else:
        decision.decision_median = closest_median
        decision.decision_per_env = dict(closest_per_env)
        decision.closest_peer_model = closest_peer_model
    return decision


class VerdictBackfillService:
    """Periodic verdict computer for the CEAC anti-copy subsystem.

    Worker writes score rows without verdict; this service polls and
    backfills. Each tick rebuilds the peer cache (so newly-scored
    miners join the comparison set), then drains every row whose
    ``verdict_at`` is still empty.
    """

    POLL_INTERVAL_SEC = 60

    def __init__(
        self,
        *,
        scores_dao: Optional[AntiCopyScoresIndexDAO] = None,
        config_dao: Optional[SystemConfigDAO] = None,
        r2: Optional[AntiCopyR2] = None,
    ):
        self.scores_dao = scores_dao or AntiCopyScoresIndexDAO()
        self.config_dao = config_dao or SystemConfigDAO()
        self.r2 = r2 or AntiCopyR2()
        self._running = False

    def stop(self) -> None:
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("[anticopy.verdict] backfill service started")
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(
                    f"[anticopy.verdict] tick failed: {e}", exc_info=True,
                )
            await asyncio.sleep(self.POLL_INTERVAL_SEC)

    async def _tick(self) -> None:
        cfg = await load_anticopy_config(self.config_dao)
        if not cfg.enabled:
            return

        rows = await self.scores_dao.list_all()
        pending = [r for r in rows if not r.get("verdict_at")]
        if not pending:
            return

        logger.info(
            f"[anticopy.verdict] tick: {len(pending)}/{len(rows)} rows pending"
        )

        # Build peer cache (load every row's blob once). Rows without an
        # R2 key or with an unreadable blob are skipped — they can't
        # contribute to anyone's verdict anyway.
        peer_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in rows:
            r2_key = row.get("r2_key")
            if not r2_key:
                continue
            try:
                blob = await asyncio.to_thread(
                    self.r2.get_score_by_key, r2_key,
                )
            except Exception as e:
                logger.debug(
                    f"[anticopy.verdict] peer fetch failed {r2_key}: {e}"
                )
                continue
            if not blob:
                continue
            _strip_resp_top(blob)
            blob["first_block"] = int(row.get("first_block", 0) or 0)
            peer_cache[(row.get("hotkey", ""), row.get("revision", ""))] = blob

        logger.info(
            f"[anticopy.verdict] peer cache built: {len(peer_cache)} blobs"
        )

        # Process pending rows. For each, ref_score is the same blob we
        # already loaded into the cache (worker uploaded it).
        for row in pending:
            hk = row.get("hotkey", "")
            rev = row.get("revision", "")
            ref = peer_cache.get((hk, rev))
            if ref is None:
                logger.debug(
                    f"[anticopy.verdict] skip {hk[:10]}: no blob in cache"
                )
                continue
            first_block = int(row.get("first_block", 0) or 0)
            decision = _pick_origin(
                cfg=cfg,
                ref_score=ref,
                ref_first_block=first_block,
                peer_cache=peer_cache,
            )
            try:
                await self.scores_dao.update_verdict(
                    hk, rev,
                    copy_of=decision.copy_of_hotkey,
                    decision_median=decision.decision_median,
                    decision_per_env=dict(decision.decision_per_env),
                    closest_peer_model=decision.closest_peer_model,
                )
            except Exception as e:
                logger.warning(
                    f"[anticopy.verdict] update_verdict failed for "
                    f"{hk[:10]}: {e}"
                )
                continue
            if decision.copy_of_hotkey:
                logger.info(
                    f"[anticopy.verdict] {hk[:10]} verdict copy_of="
                    f"{decision.copy_of_hotkey[:10]} "
                    f"dec_med={decision.decision_median:.4f}"
                )
            else:
                logger.info(
                    f"[anticopy.verdict] {hk[:10]} verdict independent "
                    f"dec_med={decision.decision_median:.4f} "
                    f"closest={(decision.closest_peer_model or '-')[:40]}"
                )
