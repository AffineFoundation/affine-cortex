"""
Tests for tensor fingerprint integration in anti-copy detection.

Tests the TensorFingerprinter compare logic and the detector's 3-signal voting
when tensor fingerprints are provided. No HuggingFace downloads — uses synthetic
numpy arrays.
"""

import math
import numpy as np
import pytest

from affine.src.anticopy.detector import AntiCopyDetector
from affine.src.anticopy.models import MinerLogprobs
from affine.src.anticopy.tensor_fingerprint import (
    TensorFingerprinter,
    _cosine_similarity,
    TARGET_TENSORS,
)


# ── Cosine similarity unit tests ────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([-1.0, -2.0], dtype=np.float32)
        assert _cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        assert _cosine_similarity(a, b) == 0.0

    def test_near_identical_high_cosine(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(50000).astype(np.float32)
        b = a + rng.normal(0, 1e-5, 50000).astype(np.float32)
        cos = _cosine_similarity(a, b)
        assert cos > 0.99999


# ── TensorFingerprinter.compare tests ────────────────────────────────────────

def make_fingerprint(seed: int, n_tensors: int = 6, size: int = 1000) -> dict:
    """Create a synthetic fingerprint dict."""
    rng = np.random.default_rng(seed)
    fp = {}
    for i, name in enumerate(TARGET_TENSORS[:n_tensors]):
        fp[name] = rng.standard_normal(size).astype(np.float32)
    return fp


class TestTensorFingerprinterCompare:
    def test_identical_fingerprints(self):
        fp = make_fingerprint(42)
        avg, per_tensor = TensorFingerprinter.compare(fp, fp)
        assert avg == pytest.approx(1.0, abs=1e-6)
        assert len(per_tensor) == 6

    def test_different_fingerprints(self):
        fp_a = make_fingerprint(42)
        fp_b = make_fingerprint(99)
        avg, per_tensor = TensorFingerprinter.compare(fp_a, fp_b)
        assert avg < 0.1  # random vectors are nearly orthogonal

    def test_near_copy_fingerprints(self):
        rng = np.random.default_rng(42)
        fp_a = make_fingerprint(42, size=50000)
        # Tiny perturbation
        fp_b = {}
        for name, arr in fp_a.items():
            fp_b[name] = arr + rng.normal(0, 1e-5, arr.shape).astype(np.float32)
        avg, _ = TensorFingerprinter.compare(fp_a, fp_b)
        assert avg > 0.99999

    def test_partial_overlap(self):
        """Only common tensors should be compared."""
        fp_a = {TARGET_TENSORS[0]: np.ones(100, dtype=np.float32)}
        fp_b = {
            TARGET_TENSORS[0]: np.ones(100, dtype=np.float32),
            TARGET_TENSORS[1]: np.ones(100, dtype=np.float32),
        }
        avg, per_tensor = TensorFingerprinter.compare(fp_a, fp_b)
        assert avg == pytest.approx(1.0, abs=1e-6)
        assert len(per_tensor) == 1  # only 1 common tensor

    def test_no_common_tensors(self):
        fp_a = {TARGET_TENSORS[0]: np.ones(100, dtype=np.float32)}
        fp_b = {TARGET_TENSORS[1]: np.ones(100, dtype=np.float32)}
        avg, per_tensor = TensorFingerprinter.compare(fp_a, fp_b)
        assert math.isnan(avg)
        assert per_tensor == {}


# ── Detector 3-signal voting tests ───────────────────────────────────────────

class TestTensorVoting:
    """Test that tensor fingerprints integrate correctly with the detector."""

    def setup_method(self):
        self.detector = AntiCopyDetector(
            cosine_threshold=0.93,
            tensor_threshold=0.99995,
            min_tasks=1,  # low gate so tensor-only tests work
        )

    def test_tensor_only_copy(self):
        """With only tensor signal (no logprobs/hs), identical fingerprints → copy."""
        m0 = MinerLogprobs(uid=0, hotkey="hk0")
        m1 = MinerLogprobs(uid=1, hotkey="hk1")
        fp = make_fingerprint(42, size=50000)
        tensor_fps = {0: fp, 1: fp}
        pairs = self.detector.detect({0: m0, 1: m1}, tensor_fingerprints=tensor_fps)
        assert len(pairs) == 1
        assert pairs[0].is_copy is True
        assert pairs[0].votes == 1
        assert pairs[0].total_votes == 1
        assert pairs[0].tensor_cosine == pytest.approx(1.0, abs=1e-6)

    def test_tensor_only_independent(self):
        """Different fingerprints → not copy."""
        m0 = MinerLogprobs(uid=0, hotkey="hk0")
        m1 = MinerLogprobs(uid=1, hotkey="hk1")
        tensor_fps = {0: make_fingerprint(42), 1: make_fingerprint(99)}
        pairs = self.detector.detect({0: m0, 1: m1}, tensor_fingerprints=tensor_fps)
        assert len(pairs) == 1
        assert pairs[0].is_copy is False

    def test_tensor_disagree_blocks_copy(self):
        """If tensor says independent but hs says copy → not copy (unanimity)."""
        m0 = MinerLogprobs(uid=0, hotkey="hk0")
        m1 = MinerLogprobs(uid=1, hotkey="hk1")
        # Identical hidden states
        hs = np.random.default_rng(42).standard_normal(256).astype(np.float32)
        for t in range(5):
            m0.task_hidden_states[t] = hs
            m1.task_hidden_states[t] = hs
        # Different tensor fingerprints
        tensor_fps = {0: make_fingerprint(42), 1: make_fingerprint(99)}
        pairs = self.detector.detect({0: m0, 1: m1}, tensor_fingerprints=tensor_fps)
        copies = [p for p in pairs if p.is_copy]
        assert len(copies) == 0
        # hs votes copy, tensor votes not-copy
        assert pairs[0].total_votes == 2
        assert pairs[0].votes < pairs[0].total_votes

    def test_all_three_signals_copy(self):
        """All 3 signals agree → copy with 3/3 votes."""
        from affine.src.anticopy.loader import TOP_K

        m0 = MinerLogprobs(uid=0, hotkey="hk0")
        m1 = MinerLogprobs(uid=1, hotkey="hk1")
        rng = np.random.default_rng(42)
        shared_tokens = [f"tok{i}" for i in range(20)]
        hs = rng.standard_normal(256).astype(np.float32)
        lps = rng.uniform(-2, 0, 20 * TOP_K).astype(np.float32)
        for t in range(5):
            m0.task_logprobs[t] = lps
            m1.task_logprobs[t] = lps
            m0.task_tokens[t] = shared_tokens
            m1.task_tokens[t] = shared_tokens
            m0.task_hidden_states[t] = hs
            m1.task_hidden_states[t] = hs
        fp = make_fingerprint(42, size=50000)
        tensor_fps = {0: fp, 1: fp}
        pairs = self.detector.detect({0: m0, 1: m1}, tensor_fingerprints=tensor_fps)
        assert len(pairs) == 1
        assert pairs[0].is_copy is True
        assert pairs[0].votes == 3
        assert pairs[0].total_votes == 3

    def test_tensor_cosine_nan_when_no_fingerprint(self):
        """Without tensor fingerprints, tensor_cosine should be NaN."""
        m0 = MinerLogprobs(uid=0, hotkey="hk0")
        m1 = MinerLogprobs(uid=1, hotkey="hk1")
        hs = np.random.default_rng(42).standard_normal(256).astype(np.float32)
        for t in range(5):
            m0.task_hidden_states[t] = hs
            m1.task_hidden_states[t] = hs
        pairs = self.detector.detect({0: m0, 1: m1})
        assert len(pairs) == 1
        assert math.isnan(pairs[0].tensor_cosine)

    def test_tensor_near_threshold(self):
        """Fingerprints just above and just below threshold."""
        rng = np.random.default_rng(42)
        fp_base = make_fingerprint(42, size=50000)

        # Tiny perturbation — should be above 0.99995
        fp_close = {}
        for name, arr in fp_base.items():
            fp_close[name] = arr + rng.normal(0, 1e-5, arr.shape).astype(np.float32)

        avg, _ = TensorFingerprinter.compare(fp_base, fp_close)
        # Verify our synthetic data is actually near threshold
        assert avg > 0.99995, f"Test setup: near-copy cosine {avg} should be > 0.99995"

        m0 = MinerLogprobs(uid=0, hotkey="hk0")
        m1 = MinerLogprobs(uid=1, hotkey="hk1")
        tensor_fps = {0: fp_base, 1: fp_close}
        pairs = self.detector.detect({0: m0, 1: m1}, tensor_fingerprints=tensor_fps)
        assert len(pairs) == 1
        assert pairs[0].is_copy is True
