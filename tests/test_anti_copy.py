"""
Tests for anti-copy detection module.

Uses synthetic logprob data that mimics real logprobs-env output.
No DB access required.
"""

import math
import numpy as np
import pytest

from affine.src.anticopy.metrics import (
    js_divergence_topk,
    token_agreement_rate,
)
from affine.src.anticopy.models import MinerLogprobs
from affine.src.anticopy.detector import AntiCopyDetector
from affine.src.anticopy.loader import _parse_tokens, MIN_TOKENS, TOP_K


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_miner(
    uid: int, task_lp: dict, noise_std: float = 0.0, with_hs: bool = False
) -> MinerLogprobs:
    """Create a MinerLogprobs with given logprob arrays, optionally noised.

    task_lp values are 20-element logprob lists (one per position).
    We expand each position to TOP_K logprobs for the extended vector.
    p2 is derived from uid+position to avoid structural similarity across miners.
    If with_hs=True, also generate synthetic hidden_states (256-dim).
    """
    m = MinerLogprobs(uid=uid, hotkey=f"hotkey_{uid}")
    for task_id, lp_list in task_lp.items():
        base = np.array(lp_list, dtype=np.float32)
        if noise_std > 0:
            rng = np.random.default_rng(uid * 1000 + task_id)
            base = base + rng.normal(0, noise_std, base.shape).astype(np.float32)

        # Synthesize top_k with uid-dependent p2 to avoid structural similarity
        topk_per_pos = []
        rng_p2 = np.random.default_rng(uid * 7 + task_id * 13)
        for i, lp in enumerate(base):
            p0 = float(np.exp(lp))
            remainder = max(1e-10, 1.0 - p0)
            split = rng_p2.uniform(0.3, 0.7)
            p1 = remainder * split
            p2 = remainder * (1 - split)
            tok_id = hash(round(float(lp), 1)) % 100000
            topk_per_pos.append([
                {"token": f"t{tok_id}", "prob": p0},
                {"token": f"t{tok_id}_alt", "prob": p1},
                {"token": f"t{tok_id}_alt2", "prob": p2},
            ])

        # Build extended logprob vector: TOP_K logprobs per position
        extended = []
        for pos_topk in topk_per_pos:
            for k in range(TOP_K):
                if k < len(pos_topk) and pos_topk[k]["prob"] > 0:
                    extended.append(np.log(pos_topk[k]["prob"]))
                else:
                    extended.append(-100.0)
        m.task_logprobs[task_id] = np.array(extended, dtype=np.float32)
        m.task_topk[task_id] = topk_per_pos
        m.task_tokens[task_id] = [topk_per_pos[i][0]["token"] for i in range(len(base))]

        if with_hs:
            # Deterministic hidden_states derived from logprobs
            rng_hs = np.random.default_rng(task_id * 100 + uid)
            hs = rng_hs.standard_normal(256).astype(np.float32)
            if noise_std > 0:
                hs = hs + rng_hs.normal(0, noise_std, 256).astype(np.float32)
            m.task_hidden_states[task_id] = hs
    return m


# Base logprob vectors for 20 tasks × 20 tokens
BASE_LPS = {
    t: list(np.random.default_rng(t).uniform(-5, 0, 20))
    for t in range(20)
}

# Base hidden_states for testing (shared by "identical" miners)
BASE_HS = {
    t: np.random.default_rng(t * 100).standard_normal(256).astype(np.float32)
    for t in range(20)
}


def make_miner_with_hs(uid: int, task_lp: dict, task_hs: dict,
                        noise_std: float = 0.0) -> MinerLogprobs:
    """Create miner with both logprob and hidden_states data."""
    m = make_miner(uid, task_lp, noise_std=noise_std)
    for task_id, hs_vec in task_hs.items():
        vec = hs_vec.copy()
        if noise_std > 0:
            rng = np.random.default_rng(uid * 2000 + task_id)
            vec = vec + rng.normal(0, noise_std, vec.shape).astype(np.float32)
        m.task_hidden_states[task_id] = vec
    return m


# ── Loader parsing tests ─────────────────────────────────────────────────────

# Sample token data matching the real logprobs-env format (3 top_k per position)
SAMPLE_TOKENS = [
    {"position": i, "token": f"tok{i}", "logprob": -0.1 * (i + 1),
     "top_k": [
         {"token": f"tok{i}", "prob": 0.9},
         {"token": f"alt{i}", "prob": 0.08},
         {"token": f"alt2_{i}", "prob": 0.02},
     ]}
    for i in range(20)
]


class TestParseTokens:
    def test_returns_correct_shapes(self):
        lp_vec, topk, greedy = _parse_tokens(SAMPLE_TOKENS)
        assert len(lp_vec) == 20 * TOP_K
        assert len(topk) == 20
        assert len(greedy) == 20

    def test_logprob_values_correct(self):
        """Logprob vector uses log(prob) from top_k, not the raw logprob field."""
        lp_vec, _, _ = _parse_tokens(SAMPLE_TOKENS)
        assert lp_vec[0] == pytest.approx(math.log(0.9), abs=1e-4)
        assert lp_vec[1] == pytest.approx(math.log(0.08), abs=1e-4)
        assert lp_vec[2] == pytest.approx(math.log(0.02), abs=1e-4)
        assert lp_vec[3] == pytest.approx(math.log(0.9), abs=1e-4)

    def test_greedy_tokens_correct(self):
        _, _, greedy = _parse_tokens(SAMPLE_TOKENS)
        assert greedy[0] == "tok0"
        assert greedy[5] == "tok5"

    def test_topk_probs_correct(self):
        _, topk, _ = _parse_tokens(SAMPLE_TOKENS)
        assert topk[0][0]["token"] == "tok0"
        assert topk[0][0]["prob"] == pytest.approx(0.9)
        assert topk[0][1]["token"] == "alt0"

    def test_positions_sorted(self):
        """Tokens provided out of order should still be sorted by position."""
        shuffled = SAMPLE_TOKENS[::-1]
        lp_vec, _, greedy = _parse_tokens(shuffled)
        assert greedy[0] == "tok0"
        assert lp_vec[0] == pytest.approx(math.log(0.9), abs=1e-4)

    def test_min_tokens_constant(self):
        assert MIN_TOKENS == 10


# ── Metric unit tests ─────────────────────────────────────────────────────────

class TestJsDivergence:
    def test_identical_distributions_returns_zero(self):
        topk = [[{"token": "a", "prob": 0.9}, {"token": "b", "prob": 0.1}]] * 5
        js = js_divergence_topk(topk, topk)
        assert js == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_distributions_returns_log2(self):
        topk_a = [[{"token": "a", "prob": 1.0}]]
        topk_b = [[{"token": "b", "prob": 1.0}]]
        js = js_divergence_topk(topk_a, topk_b)
        assert js == pytest.approx(math.log(2), rel=1e-4)

    def test_partially_overlapping(self):
        topk_a = [[{"token": "a", "prob": 0.7}, {"token": "b", "prob": 0.3}]]
        topk_b = [[{"token": "a", "prob": 0.5}, {"token": "c", "prob": 0.5}]]
        js = js_divergence_topk(topk_a, topk_b)
        assert 0.0 < js < math.log(2)


class TestTokenAgreement:
    def test_identical(self):
        toks = ["a", "b", "c"]
        assert token_agreement_rate(toks, toks) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_agreement_rate(["a", "b"], ["c", "d"]) == pytest.approx(0.0)

    def test_half_overlap(self):
        assert token_agreement_rate(["a", "b", "c", "d"], ["a", "b", "x", "y"]) == pytest.approx(0.5)

    def test_empty_inputs(self):
        assert math.isnan(token_agreement_rate([], []))


# ── Detector integration tests ────────────────────────────────────────────────

class TestAntiCopyDetector:
    def setup_method(self):
        self.detector = AntiCopyDetector(
            logprob_suspicious_threshold=0.93,
            logprob_cheat_threshold=1.1,
            min_tasks=3,
        )

    def test_identical_models_flagged_as_copy(self):
        """Two miners with exactly the same logprobs should be flagged."""
        m0 = make_miner(0, BASE_LPS)
        m1 = make_miner(1, BASE_LPS)
        pairs = self.detector.detect({0: m0, 1: m1})
        copies = [p for p in pairs if p.is_copy]
        suspicious = [p for p in pairs if p.verdict == "suspicious"]
        assert len(copies) == 0
        assert len(suspicious) == 1
        pair = suspicious[0]
        # Cosine is high but not exactly 1.0 because p2 split varies by uid
        assert pair.cosine_similarity > self.detector.logprob_suspicious_threshold
        # Only cos signal (no hs), votes 1/1
        assert pair.votes == 1
        assert pair.total_votes == 1
        assert pair.verdict == "suspicious"

    def test_independent_models_not_flagged(self):
        """Models with different logprobs should not be flagged."""
        lps_a = {t: list(np.random.default_rng(t).uniform(-5, 0, 20)) for t in range(20)}
        lps_b = {t: list(np.random.default_rng(t + 100).uniform(-5, 0, 20)) for t in range(20)}
        m0 = make_miner(0, lps_a)
        m1 = make_miner(1, lps_b)
        pairs = self.detector.detect({0: m0, 1: m1})
        copies = [p for p in pairs if p.is_copy]
        assert len(copies) == 0

    def test_slightly_noised_copy_still_flagged(self):
        """A copy with tiny noise should be flagged."""
        m0 = make_miner(0, BASE_LPS)
        m1 = make_miner(1, BASE_LPS, noise_std=1e-4)
        pairs = self.detector.detect({0: m0, 1: m1})
        suspicious = [p for p in pairs if p.verdict == "suspicious"]
        assert len(suspicious) == 1

    def test_heavily_noised_not_flagged(self):
        """A model fine-tuned enough to have different logprobs should not be flagged."""
        m0 = make_miner(0, BASE_LPS)
        m1 = make_miner(1, BASE_LPS, noise_std=2.0)
        pairs = self.detector.detect({0: m0, 1: m1})
        copies = [p for p in pairs if p.is_copy]
        assert len(copies) == 0

    def test_three_miners_one_copy(self):
        """Among 3 miners, only the copy pair should be flagged."""
        lps_c = {t: list(np.random.default_rng(t + 200).uniform(-5, 0, 20)) for t in range(20)}
        m0 = make_miner(0, BASE_LPS)
        m1 = make_miner(1, BASE_LPS)
        m2 = make_miner(2, lps_c)
        pairs = self.detector.detect({0: m0, 1: m1, 2: m2})
        suspicious = [p for p in pairs if p.verdict == "suspicious"]
        assert len(suspicious) == 1
        assert set([suspicious[0].uid_a, suspicious[0].uid_b]) == {0, 1}

    def test_insufficient_tasks_skipped(self):
        """Pairs with fewer than min_tasks shared tasks are skipped."""
        lps_2task = {0: BASE_LPS[0], 1: BASE_LPS[1]}
        m0 = make_miner(0, lps_2task)
        m1 = make_miner(1, lps_2task)
        pairs = self.detector.detect({0: m0, 1: m1})
        assert len(pairs) == 0

    def test_results_sorted_by_confidence(self):
        """Results should be sorted by confidence descending."""
        m0 = make_miner(0, BASE_LPS)
        m1 = make_miner(1, BASE_LPS)
        lps_b = {t: list(np.random.default_rng(t + 100).uniform(-5, 0, 20)) for t in range(20)}
        m2 = make_miner(2, lps_b)
        pairs = self.detector.detect({0: m0, 1: m1, 2: m2})
        for i in range(len(pairs) - 1):
            assert pairs[i].confidence >= pairs[i + 1].confidence

    def test_single_miner_returns_empty(self):
        m0 = make_miner(0, BASE_LPS)
        assert self.detector.detect({0: m0}) == []

    def test_all_pairs_returned_not_just_copies(self):
        """detect() returns all evaluated pairs, not just copies."""
        lps_b = {t: list(np.random.default_rng(t + 100).uniform(-5, 0, 20)) for t in range(20)}
        m0 = make_miner(0, BASE_LPS)
        m1 = make_miner(1, lps_b)
        pairs = self.detector.detect({0: m0, 1: m1})
        assert len(pairs) == 1
        assert pairs[0].is_copy is False
        assert pairs[0].verdict == "clean"

    def test_confidence_range(self):
        """All confidence values must be in [0, 1]."""
        lps_b = {t: list(np.random.default_rng(t + 50).uniform(-5, 0, 20)) for t in range(20)}
        m0 = make_miner(0, BASE_LPS)
        m1 = make_miner(1, BASE_LPS)
        m2 = make_miner(2, lps_b)
        for pair in self.detector.detect({0: m0, 1: m1, 2: m2}):
            assert 0.0 <= pair.confidence <= 1.0


# ── Hidden states voting tests ───────────────────────────────────────────────

class TestHiddenStatesVoting:
    def setup_method(self):
        self.detector = AntiCopyDetector(
            hs_suspicious_threshold=0.99,
            logprob_suspicious_threshold=0.93,
            min_tasks=3,
        )

    def test_both_signals_identical(self):
        """With both cos and hs identical, should get 2/2 votes."""
        detector = AntiCopyDetector(
            hs_suspicious_threshold=0.99,
            logprob_suspicious_threshold=0.93,
            hs_cheat_threshold=0.99,
            logprob_cheat_threshold=0.93,
            min_tasks=3,
        )
        m0 = make_miner_with_hs(0, BASE_LPS, BASE_HS)
        m1 = make_miner_with_hs(1, BASE_LPS, BASE_HS)
        pairs = detector.detect({0: m0, 1: m1})
        copies = [p for p in pairs if p.is_copy]
        assert len(copies) == 1
        assert copies[0].votes == 2
        assert copies[0].total_votes == 2
        assert copies[0].verdict == "cheat"

    def test_hs_only_copy(self):
        """With only hidden_states data, 1 signal is enough."""
        m0 = MinerLogprobs(uid=0, hotkey="hk0")
        m1 = MinerLogprobs(uid=1, hotkey="hk1")
        for t in range(5):
            m0.task_hidden_states[t] = BASE_HS[t]
            m1.task_hidden_states[t] = BASE_HS[t]
        pairs = self.detector.detect({0: m0, 1: m1})
        assert len(pairs) == 1
        assert pairs[0].is_copy is True
        assert pairs[0].verdict == "cheat"
        assert pairs[0].votes == 1
        assert pairs[0].total_votes == 1

    def test_cos_disagree_not_copy(self):
        """hs agrees but cos disagrees → not copy (must be unanimous)."""
        m0 = MinerLogprobs(uid=0, hotkey="hk0")
        m1 = MinerLogprobs(uid=1, hotkey="hk1")
        for t in range(5):
            m0.task_hidden_states[t] = BASE_HS[t]
            m1.task_hidden_states[t] = BASE_HS[t]
            # Same tokens so logprob signal participates, but orthogonal
            # vectors so cosine stays below threshold.
            m0.task_tokens[t] = ["x", "y"]
            m1.task_tokens[t] = ["x", "y"]
            m0.task_logprobs[t] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            m1.task_logprobs[t] = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        pairs = self.detector.detect({0: m0, 1: m1})
        copies = [p for p in pairs if p.is_copy]
        assert len(copies) == 0
        assert pairs[0].votes == 1   # only hs
        assert pairs[0].total_votes == 2

    def test_hs_disagree_not_copy(self):
        """cos agrees but hs disagrees → not copy (must be unanimous)."""
        # Different hidden states
        hs_b = {t: np.random.default_rng(t * 100 + 999).standard_normal(256).astype(np.float32)
                for t in range(20)}
        m0 = make_miner_with_hs(0, BASE_LPS, BASE_HS)
        m1 = make_miner_with_hs(1, BASE_LPS, hs_b)
        pairs = self.detector.detect({0: m0, 1: m1})
        copies = [p for p in pairs if p.is_copy]
        assert len(copies) == 0
        assert pairs[0].votes == 1   # only cos
        assert pairs[0].total_votes == 2

    def test_hs_nan_when_no_data(self):
        """hs_cosine is NaN when no hidden_states data."""
        m0 = make_miner(0, BASE_LPS)
        m1 = make_miner(1, BASE_LPS)
        pairs = self.detector.detect({0: m0, 1: m1})
        assert len(pairs) == 1
        assert math.isnan(pairs[0].hs_cosine)
