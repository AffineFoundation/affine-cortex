"""Per-env verdict rule: "all envs below threshold -> copy; any env
above threshold -> clean". A floor on the minimum number of envs with
data falls back to legacy combined-median to avoid acquitting a
candidate on a single-env coincidence.
"""

from affine.src.anticopy.pairwise import (
    EnvCompare,
    PairResult,
    is_copy_verdict,
)


PER_ENV = {
    "MEMORY":       0.07,
    "TERMINAL":     0.10,
    "SWE-INFINITE": 0.12,
    "NAVWORLD":     0.06,
    "LIVEWEB":      0.06,
}


def _pair(envs: dict, combined: float = 0.05) -> PairResult:
    """envs: env -> decision_median float; decision_n defaults to 50."""
    per_env = {
        e: EnvCompare(env=e, decision_n=50, decision_median=v)
        for e, v in envs.items()
    }
    return PairResult(
        n_overlap_rollouts=10, n_overlap_tokens=10_000,
        per_env=per_env, decision_median_combined=combined,
    )


# ----- per-env rule (new) -----

def test_per_env_all_below_is_copy():
    pair = _pair({
        "MEMORY":       0.060,   # < 0.07 ✓
        "TERMINAL":     0.080,   # < 0.10 ✓
        "SWE-INFINITE": 0.110,   # < 0.12 ✓
        "NAVWORLD":     0.050,   # < 0.06 ✓
        "LIVEWEB":      0.055,   # < 0.06 ✓
    })
    assert is_copy_verdict(
        pair, nll_threshold=999.0,    # combined-mode threshold ignored
        per_env_nll_thresholds=PER_ENV, per_env_min_envs=3,
    )


def test_per_env_single_above_acquits():
    # Even one env above the bar drops the verdict to clean.
    pair = _pair({
        "MEMORY":       0.060,
        "TERMINAL":     0.080,
        "SWE-INFINITE": 0.130,   # > 0.12 ✗  ← single env above
        "NAVWORLD":     0.050,
        "LIVEWEB":      0.055,
    })
    assert not is_copy_verdict(
        pair, nll_threshold=999.0,
        per_env_nll_thresholds=PER_ENV, per_env_min_envs=3,
    )


def test_per_env_at_or_above_threshold_acquits():
    # Equality goes to the candidate (>=, not <) — sits at the bar
    # = "differs enough", not "matches".
    pair = _pair({
        "MEMORY":       0.070,   # == 0.07 → above
        "TERMINAL":     0.080,
        "SWE-INFINITE": 0.110,
        "NAVWORLD":     0.050,
    })
    assert not is_copy_verdict(
        pair, nll_threshold=999.0,
        per_env_nll_thresholds=PER_ENV, per_env_min_envs=3,
    )


def test_per_env_below_min_envs_falls_back_to_combined():
    # Only 2 envs with data — too thin for an "all envs agree" claim.
    # Falls back to combined median rule.
    pair = _pair(
        {"MEMORY": 0.060, "NAVWORLD": 0.050},   # both below per-env
        combined=0.080,                          # but combined > 0.07
    )
    # Falls through to nll_threshold=0.07 check → not a copy.
    assert not is_copy_verdict(
        pair, nll_threshold=0.07,
        per_env_nll_thresholds=PER_ENV, per_env_min_envs=3,
    )

    # Same pair but a low combined median → fallback flips to copy.
    pair2 = _pair(
        {"MEMORY": 0.060, "NAVWORLD": 0.050},
        combined=0.040,
    )
    assert is_copy_verdict(
        pair2, nll_threshold=0.07,
        per_env_nll_thresholds=PER_ENV, per_env_min_envs=3,
    )


def test_per_env_zero_decision_n_ignored():
    # An env carried in the pair but with no decision tokens shouldn't
    # count toward the floor or the "all below" check.
    pair = _pair({
        "MEMORY":       0.060,
        "TERMINAL":     0.080,
        "SWE-INFINITE": 0.110,
        "NAVWORLD":     0.050,
    })
    # Inject an empty env after construction.
    pair.per_env["LIVEWEB"] = EnvCompare("LIVEWEB", decision_n=0, decision_median=999.0)
    assert is_copy_verdict(
        pair, nll_threshold=999.0,
        per_env_nll_thresholds=PER_ENV, per_env_min_envs=3,
    )


def test_unconfigured_env_does_not_veto():
    # An env present in the pair but absent from per_env config (e.g.
    # operator hasn't tuned that env yet) is ignored, not used to
    # acquit the candidate.
    pair = _pair({
        "MEMORY":   0.060,
        "TERMINAL": 0.080,
        "NAVWORLD": 0.050,
        "DISTILL":  9.999,   # not in PER_ENV config — must be ignored
    })
    assert is_copy_verdict(
        pair, nll_threshold=999.0,
        per_env_nll_thresholds=PER_ENV, per_env_min_envs=3,
    )


# ----- legacy combined-median mode (unchanged behaviour) -----

def test_combined_mode_when_per_env_config_empty():
    pair = _pair({"MEMORY": 0.06}, combined=0.050)
    # Per-env config empty → combined-median rule with nll_threshold.
    assert is_copy_verdict(pair, nll_threshold=0.07)
    assert not is_copy_verdict(pair, nll_threshold=0.03)


def test_combined_mode_no_signal_is_clean():
    pair = _pair({}, combined=-1.0)
    assert not is_copy_verdict(pair, nll_threshold=0.07)
