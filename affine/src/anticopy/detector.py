"""
Anti-Copy Detector

Two-signal voting system for detecting model copies.

Signals:
  1. hidden_states cosine  – model internal representation (independent)
  2. logprob top-3 cosine  – output distribution direction (independent)

JS divergence and token agreement are computed for diagnostics but do NOT
vote, because they are derived from the same logprob/topk data as cosine
and carry redundant information.

Decision: is_copy when ALL available signals vote copy, with at least 1 signal.
"""

import numpy as np
from typing import Dict, List

from affine.core.setup import logger
from affine.src.anticopy.metrics import js_divergence_topk, token_agreement_rate
from affine.src.anticopy.models import CopyPair, MinerLogprobs


class AntiCopyDetector:
    """Detect model copies using two independent signals.

    Args:
        hs_threshold:      hidden_states cosine >= this → vote copy
        cosine_threshold:   logprob cosine >= this → vote copy
        min_tasks:          minimum shared tasks required for comparison
    """

    def __init__(
        self,
        hs_threshold: float = 0.99,
        cosine_threshold: float = 0.99,
        min_tasks: int = 30,
    ):
        self.hs_threshold = hs_threshold
        self.cosine_threshold = cosine_threshold
        self.min_tasks = min_tasks

    @staticmethod
    def _cosine_pair(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1-D vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def detect(self, miners: Dict[int, MinerLogprobs]) -> List[CopyPair]:
        """Run detection over all miner pairs.

        Returns:
            List of CopyPair, sorted by confidence descending
        """
        if len(miners) < 2:
            return []

        uid_list = sorted(miners.keys())
        n = len(uid_list)
        logger.info(f"anti_copy: running detection on {n} miners ({n*(n-1)//2} pairs)")

        results: List[CopyPair] = []

        for i, uid_a in enumerate(uid_list):
            for uid_b in uid_list[i + 1:]:
                ma, mb = miners[uid_a], miners[uid_b]

                # ── Logprob signals ──────────────────────────────────
                lp_common = sorted(
                    t for t in set(ma.task_logprobs) & set(mb.task_logprobs)
                    if ma.task_logprobs[t].shape == mb.task_logprobs[t].shape
                )
                has_logprobs = len(lp_common) >= self.min_tasks

                med_cosine = float("nan")
                med_js = float("nan")
                med_agree = float("nan")
                task_cosine_map: Dict[int, float] = {}
                n_tasks = 0

                if has_logprobs:
                    cosines = np.array([
                        self._cosine_pair(ma.task_logprobs[t], mb.task_logprobs[t])
                        for t in lp_common
                    ])
                    med_cosine = float(np.nanmedian(cosines))
                    task_cosine_map = {
                        t: float(c) for t, c in zip(lp_common, cosines)
                    }
                    n_tasks = len(lp_common)

                    # JS divergence
                    js_vals = []
                    for tid in lp_common:
                        topk_a = ma.task_topk.get(tid, [])
                        topk_b = mb.task_topk.get(tid, [])
                        if topk_a and topk_b:
                            js_vals.append(js_divergence_topk(topk_a, topk_b))
                    if js_vals:
                        med_js = float(np.nanmedian(js_vals))

                    # Token agreement
                    agree_vals = []
                    for tid in lp_common:
                        tok_a = ma.task_tokens.get(tid, [])
                        tok_b = mb.task_tokens.get(tid, [])
                        if tok_a and tok_b:
                            agree_vals.append(token_agreement_rate(tok_a, tok_b))
                    if agree_vals:
                        med_agree = float(np.nanmedian(agree_vals))

                # ── Hidden states signal ─────────────────────────────
                hs_common = sorted(
                    t for t in set(ma.task_hidden_states) & set(mb.task_hidden_states)
                    if ma.task_hidden_states[t].shape == mb.task_hidden_states[t].shape
                )
                has_hs = len(hs_common) >= self.min_tasks
                med_hs = float("nan")

                if has_hs:
                    hs_cosines = np.array([
                        self._cosine_pair(ma.task_hidden_states[t], mb.task_hidden_states[t])
                        for t in hs_common
                    ])
                    med_hs = float(np.nanmedian(hs_cosines))
                    # Use max task count
                    n_tasks = max(n_tasks, len(hs_common))

                # Need at least some data to compare
                if not has_logprobs and not has_hs:
                    continue

                # ── Voting (2 independent signals) ─────────────────
                votes = 0
                total_votes = 0

                if has_hs:
                    total_votes += 1
                    if med_hs >= self.hs_threshold:
                        votes += 1

                if has_logprobs and not np.isnan(med_cosine):
                    total_votes += 1
                    if med_cosine >= self.cosine_threshold:
                        votes += 1

                # All available signals must agree; need at least 1
                is_copy = total_votes >= 1 and votes == total_votes

                # Confidence: fraction of votes
                confidence = votes / max(total_votes, 1)

                results.append(
                    CopyPair(
                        uid_a=uid_a,
                        uid_b=uid_b,
                        hotkey_a=ma.hotkey,
                        hotkey_b=mb.hotkey,
                        cosine_similarity=med_cosine,
                        hs_cosine=med_hs,
                        js_divergence=med_js,
                        token_agreement=med_agree,
                        n_tasks=n_tasks,
                        is_copy=is_copy,
                        confidence=confidence,
                        votes=votes,
                        total_votes=total_votes,
                        task_cosines=task_cosine_map,
                    )
                )

        copies = [r for r in results if r.is_copy]
        results.sort(key=lambda r: r.confidence, reverse=True)

        logger.info(
            f"anti_copy: {len(copies)} copy pairs detected "
            f"out of {len(results)} pairs evaluated"
        )
        return results
