"""Unit tests for WindowSampler determinism and correctness."""

import pytest

from affine.src.scorer.sampler import (
    SAMPLING_MODE_LATEST,
    SAMPLING_MODE_RANDOM,
    EnvSamplingConfig,
    WindowSampler,
)


def _cfg(env: str, ranges, n: int, mode: str = SAMPLING_MODE_RANDOM) -> EnvSamplingConfig:
    return EnvSamplingConfig(env=env, dataset_range=ranges, sampling_count=n, mode=mode)


def test_same_inputs_yield_same_output():
    s = WindowSampler()
    a = s.generate(42, 1_000_000, {"x": _cfg("x", [[0, 50_000]], 1000)})
    b = s.generate(42, 1_000_000, {"x": _cfg("x", [[0, 50_000]], 1000)})
    assert a == b


def test_different_window_ids_diverge():
    s = WindowSampler()
    a = s.generate(1, 0, {"x": _cfg("x", [[0, 50_000]], 1000)})["x"]
    b = s.generate(2, 0, {"x": _cfg("x", [[0, 50_000]], 1000)})["x"]
    overlap = len(set(a) & set(b))
    # 1000 of 50k drawn twice → expected overlap ≈ 1000²/50000 = 20.
    # Allow generous slack but flag pathological overlap.
    assert overlap < 100


def test_different_envs_under_same_window_diverge():
    s = WindowSampler()
    out = s.generate(
        42,
        0,
        {
            "a": _cfg("a", [[0, 50_000]], 1000),
            "b": _cfg("b", [[0, 50_000]], 1000),
        },
    )
    overlap = len(set(out["a"]) & set(out["b"]))
    assert overlap < 100


def test_output_size_matches_sampling_count():
    s = WindowSampler()
    out = s.generate(7, 100, {"x": _cfg("x", [[0, 5000]], 250)})["x"]
    assert len(out) == 250
    assert len(set(out)) == 250  # no duplicates


def test_output_inside_dataset_range_with_multiple_intervals():
    s = WindowSampler()
    ranges = [[0, 100], [500, 600], [1000, 1100]]
    out = s.generate(99, 0, {"x": _cfg("x", ranges, 200)})["x"]
    assert len(out) == 200
    for tid in out:
        assert any(start <= tid < end for start, end in ranges), tid


def test_sample_equal_to_total_returns_all_ids():
    s = WindowSampler()
    ranges = [[0, 100]]
    out = s.generate(1, 0, {"x": _cfg("x", ranges, 100)})["x"]
    assert sorted(out) == list(range(100))


def test_oversample_raises():
    s = WindowSampler()
    try:
        s.generate(1, 0, {"x": _cfg("x", [[0, 100]], 101)})
    except ValueError as e:
        assert "exceeds available" in str(e)
    else:
        raise AssertionError("expected ValueError on oversample")


def test_block_start_affects_seed():
    s = WindowSampler()
    a = s.generate(1, 100, {"x": _cfg("x", [[0, 50_000]], 1000)})["x"]
    b = s.generate(1, 200, {"x": _cfg("x", [[0, 50_000]], 1000)})["x"]
    assert a != b


def test_sorted_output():
    s = WindowSampler()
    out = s.generate(1, 0, {"x": _cfg("x", [[0, 50_000]], 1000)})["x"]
    assert out == sorted(out)


def test_overlapping_input_ranges_merged():
    # RangeSet normalizes overlapping intervals; sampler should respect that.
    s = WindowSampler()
    ranges = [[0, 50], [40, 100]]  # merges to [[0, 100]]
    out = s.generate(1, 0, {"x": _cfg("x", ranges, 100)})["x"]
    assert sorted(out) == list(range(100))


# ---- latest mode ------------------------------------------------------------


def test_latest_takes_tail_of_single_range():
    s = WindowSampler()
    out = s.generate(
        1, 0, {"x": _cfg("x", [[0, 10000]], 5, mode=SAMPLING_MODE_LATEST)}
    )["x"]
    assert out == [9995, 9996, 9997, 9998, 9999]


def test_latest_walks_multiple_ranges_from_the_back():
    s = WindowSampler()
    # Tail interval has 3 ids (97..99); spill into the prior interval for
    # the remaining 2.
    ranges = [[0, 100], [200, 203]]
    out = s.generate(
        1, 0, {"x": _cfg("x", ranges, 5, mode=SAMPLING_MODE_LATEST)}
    )["x"]
    assert out == [98, 99, 200, 201, 202]


def test_latest_is_window_id_independent():
    """The dataset tail is deterministic in the dataset itself; latest mode
    doesn't use a seed, so different window_ids must yield identical lists."""
    s = WindowSampler()
    a = s.generate(1, 0, {"x": _cfg("x", [[0, 1000]], 50, mode=SAMPLING_MODE_LATEST)})["x"]
    b = s.generate(99, 50_000, {"x": _cfg("x", [[0, 1000]], 50, mode=SAMPLING_MODE_LATEST)})["x"]
    assert a == b


def test_latest_oversample_raises():
    s = WindowSampler()
    with pytest.raises(ValueError, match="exceeds available"):
        s.generate(1, 0, {"x": _cfg("x", [[0, 10]], 11, mode=SAMPLING_MODE_LATEST)})


def test_latest_zero_returns_empty():
    s = WindowSampler()
    out = s.generate(1, 0, {"x": _cfg("x", [[0, 10]], 0, mode=SAMPLING_MODE_LATEST)})["x"]
    assert out == []


def test_unknown_mode_raises():
    s = WindowSampler()
    with pytest.raises(ValueError, match="unknown sampling mode"):
        s.generate(1, 0, {"x": _cfg("x", [[0, 10]], 3, mode="foo")})


def test_mixed_modes_per_env():
    """Same window evaluates latest + random envs side by side."""
    s = WindowSampler()
    out = s.generate(
        1, 0,
        {
            "swe": _cfg("swe", [[0, 100]], 5, mode=SAMPLING_MODE_LATEST),
            "liveweb": _cfg("liveweb", [[0, 1000]], 5, mode=SAMPLING_MODE_RANDOM),
        },
    )
    assert out["swe"] == [95, 96, 97, 98, 99]
    assert len(out["liveweb"]) == 5
    assert len(set(out["liveweb"])) == 5


# ---- production-scale + fallback-path coverage ------------------------------


def test_random_sampling_near_total_uses_deterministic_fallback():
    """When n is close to total, the random-with-rejection loop bails out
    and the deterministic-fill fallback completes the set. The output is
    still seed-reproducible across runs."""
    s = WindowSampler()
    # Force the fallback: 99 of 100 → collision rate is very high.
    a = s.generate(1, 0, {"x": _cfg("x", [[0, 100]], 99)})["x"]
    b = s.generate(1, 0, {"x": _cfg("x", [[0, 100]], 99)})["x"]
    assert a == b
    assert len(a) == 99
    assert len(set(a)) == 99
    for tid in a:
        assert 0 <= tid < 100


def test_random_sampling_at_production_swe_scale_is_deterministic():
    """SWE/DISTILL use latest mode, but the random sampler must also
    scale to large dataset ranges. ``LIVEWEB``-style range is the
    stress test: 400 picks from 78M ids, near-zero collision rate."""
    s = WindowSampler()
    cfg = _cfg("liveweb", [[0, 78_060_000]], 400)
    out_a = s.generate(42, 1_000_000, {"liveweb": cfg})["liveweb"]
    out_b = s.generate(42, 1_000_000, {"liveweb": cfg})["liveweb"]
    assert out_a == out_b
    assert len(out_a) == 400
    assert len(set(out_a)) == 400


def test_random_sampling_with_multi_interval_dataset():
    """LOGPROBS uses two disjoint intervals. The sampler should respect
    both ranges and produce ids drawn from either."""
    s = WindowSampler()
    ranges = [[100_000_000, 300_000_000], [400_000_000, 500_000_000]]
    out = s.generate(1, 0, {"logprobs": _cfg("logprobs", ranges, 500)})["logprobs"]
    assert len(out) == 500
    assert len(set(out)) == 500
    in_range_a = sum(1 for t in out if 100_000_000 <= t < 300_000_000)
    in_range_b = sum(1 for t in out if 400_000_000 <= t < 500_000_000)
    assert in_range_a + in_range_b == 500
    # Range A is 2x larger than B → roughly twice as many picks. Allow
    # generous slack for randomness.
    assert in_range_a > in_range_b


def test_latest_mode_with_swe_scale_dataset():
    """SWE-INFINITE config: 300 latest from a ~5000-id range."""
    s = WindowSampler()
    out = s.generate(
        1, 0, {"swe": _cfg("swe", [[0, 5000]], 300, mode=SAMPLING_MODE_LATEST)}
    )["swe"]
    assert out == list(range(4700, 5000))
