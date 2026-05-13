"""Unit tests for WindowComparator (Pareto-with-tolerance)."""

from affine.src.scorer.comparator import (
    ENV_DOMINANT,
    ENV_NOT_WORSE,
    ENV_WORSE,
    EnvComparisonConfig,
    WindowComparator,
)


ENVS = ["env_a", "env_b"]


def _cfg(margin=0.01, min_n=10, tol=0.0):
    return {
        e: EnvComparisonConfig(
            env=e, margin=margin, min_tasks_per_env=min_n,
            not_worse_tolerance=tol,
        )
        for e in ENVS
    }


def _scores(values_by_env):
    """{env: [score, score, ...]} → {env: {task_id: score}}"""
    return {env: {i: v for i, v in enumerate(vals)} for env, vals in values_by_env.items()}


# ---- strict Pareto (min_dom = 0) -------------------------------------------


def test_strict_pareto_every_env_dominant_wins():
    cmp = WindowComparator()
    champ = _scores({"env_a": [0.4] * 20, "env_b": [0.5] * 20})
    chal = _scores({"env_a": [0.5] * 20, "env_b": [0.6] * 20})
    r = cmp.compare(champ, chal, _cfg(margin=0.01), min_dominant_envs=0)
    assert r.winner == "challenger"
    assert r.reason == "all_envs_dominant"
    assert r.dominant_count == 2
    assert all(e.verdict == ENV_DOMINANT for e in r.per_env)


def test_strict_pareto_tie_in_one_env_blocks():
    cmp = WindowComparator()
    champ = _scores({"env_a": [0.5] * 20, "env_b": [0.5] * 20})
    chal = _scores({"env_a": [0.6] * 20, "env_b": [0.5] * 20})  # env_b ties
    r = cmp.compare(champ, chal, _cfg(margin=0.01), min_dominant_envs=0)
    assert r.winner == "champion"
    assert r.reason == "not_all_envs_dominant"
    by_env = {e.env: e for e in r.per_env}
    assert by_env["env_a"].verdict == ENV_DOMINANT
    # env_b is within (1-0)*champ=champ, so chal_avg = champ_avg → not_worse
    assert by_env["env_b"].verdict == ENV_NOT_WORSE


def test_strict_pareto_within_margin_blocks():
    cmp = WindowComparator()
    champ = _scores({"env_a": [0.50] * 20, "env_b": [0.50] * 20})
    chal = _scores({"env_a": [0.505] * 20, "env_b": [0.55] * 20})  # env_a within margin
    r = cmp.compare(champ, chal, _cfg(margin=0.01), min_dominant_envs=0)
    assert r.winner == "champion"


# ---- partial Pareto (min_dom > 0) ------------------------------------------


def test_partial_pareto_n_dominant_others_not_worse_wins():
    """N = 1 dominant env + the other tied within tolerance → challenger wins."""
    cmp = WindowComparator()
    champ = _scores({"env_a": [0.5] * 20, "env_b": [0.5] * 20})
    chal = _scores({"env_a": [0.6] * 20, "env_b": [0.49] * 20})  # env_b regress 0.01
    # tolerance 5% → champ * (1 - 0.05) = 0.475 → 0.49 passes not_worse
    r = cmp.compare(champ, chal, _cfg(margin=0.01, tol=0.05), min_dominant_envs=1)
    assert r.winner == "challenger"
    assert r.dominant_count == 1
    assert r.not_worse_count == 2
    by_env = {e.env: e for e in r.per_env}
    assert by_env["env_a"].verdict == ENV_DOMINANT
    assert by_env["env_b"].verdict == ENV_NOT_WORSE


def test_partial_pareto_regression_beyond_tolerance_blocks():
    cmp = WindowComparator()
    champ = _scores({"env_a": [0.5] * 20, "env_b": [0.5] * 20})
    chal = _scores({"env_a": [0.6] * 20, "env_b": [0.40] * 20})  # regress 20%
    # tolerance 5% → threshold 0.475, 0.40 < 0.475 → worse
    r = cmp.compare(champ, chal, _cfg(margin=0.01, tol=0.05), min_dominant_envs=1)
    assert r.winner == "champion"
    assert "regressed_in_env:env_b" in r.reason
    by_env = {e.env: e for e in r.per_env}
    assert by_env["env_b"].verdict == ENV_WORSE


def test_partial_pareto_insufficient_dominant_count_blocks():
    """N = 2 required, only 1 env is dominant → challenger blocked even
    if every env is at least not_worse."""
    cmp = WindowComparator()
    champ = _scores({"env_a": [0.5] * 20, "env_b": [0.5] * 20})
    chal = _scores({"env_a": [0.6] * 20, "env_b": [0.5] * 20})  # 1 dom, 1 tie
    r = cmp.compare(champ, chal, _cfg(margin=0.01, tol=0.05), min_dominant_envs=2)
    assert r.winner == "champion"
    assert "insufficient_dominant_envs:1<2" in r.reason


# ---- sample-count gate -----------------------------------------------------


def test_insufficient_challenger_samples_counts_as_worse():
    cmp = WindowComparator()
    champ = _scores({"env_a": [0.5] * 20, "env_b": [0.5] * 20})
    chal = _scores({"env_a": [0.9] * 20, "env_b": [0.9] * 5})  # env_b too few
    r = cmp.compare(champ, chal, _cfg(min_n=10, tol=0.05), min_dominant_envs=1)
    assert r.winner == "champion"
    by_env = {e.env: e for e in r.per_env}
    assert by_env["env_b"].verdict == ENV_WORSE
    assert by_env["env_b"].reason == "insufficient_challenger_samples"


# ---- per-env knob override -------------------------------------------------


def test_per_env_margin_override():
    cmp = WindowComparator()
    cfg = {
        "env_a": EnvComparisonConfig(env="env_a", margin=0.05, min_tasks_per_env=10),
        "env_b": EnvComparisonConfig(env="env_b", margin=0.01, min_tasks_per_env=10),
    }
    champ = _scores({"env_a": [0.5] * 20, "env_b": [0.5] * 20})
    chal = _scores({"env_a": [0.52] * 20, "env_b": [0.6] * 20})  # env_a < 0.05 margin
    r = cmp.compare(champ, chal, cfg, min_dominant_envs=0)
    assert r.winner == "champion"


def test_per_env_tolerance_override():
    """env_a uses a tight tol (any regress = worse), env_b uses loose tol."""
    cmp = WindowComparator()
    cfg = {
        "env_a": EnvComparisonConfig(env="env_a", margin=0.01, min_tasks_per_env=10,
                                     not_worse_tolerance=0.0),
        "env_b": EnvComparisonConfig(env="env_b", margin=0.01, min_tasks_per_env=10,
                                     not_worse_tolerance=0.10),
    }
    champ = _scores({"env_a": [0.5] * 20, "env_b": [0.5] * 20})
    chal = _scores({"env_a": [0.6] * 20, "env_b": [0.46] * 20})  # env_b -8%
    # env_b tol=10% → champ*(1-0.1) = 0.45, 0.46 ≥ 0.45 → not_worse
    r = cmp.compare(champ, chal, cfg, min_dominant_envs=1)
    assert r.winner == "challenger"
    by_env = {e.env: e for e in r.per_env}
    assert by_env["env_a"].verdict == ENV_DOMINANT
    assert by_env["env_b"].verdict == ENV_NOT_WORSE


# ---- edge cases ------------------------------------------------------------


def test_champion_missing_data_treated_as_zero_baseline():
    """If champion has no samples (e.g. provider failure), use 0 baseline."""
    cmp = WindowComparator()
    champ = _scores({"env_a": [0.4] * 20, "env_b": []})  # env_b champion empty
    chal = _scores({"env_a": [0.5] * 20, "env_b": [0.3] * 20})
    r = cmp.compare(champ, chal, _cfg(margin=0.01), min_dominant_envs=0)
    by_env = {e.env: e for e in r.per_env}
    assert by_env["env_b"].challenger_avg == 0.3
    assert by_env["env_b"].delta == 0.3
    assert by_env["env_b"].verdict == ENV_DOMINANT
    assert r.winner == "challenger"


def test_empty_env_config_returns_champion():
    cmp = WindowComparator()
    r = cmp.compare(_scores({}), _scores({}), {}, min_dominant_envs=0)
    assert r.winner == "champion"
    assert r.reason == "no_envs_configured"


def test_extra_env_in_scores_but_not_in_config_is_ignored():
    cmp = WindowComparator()
    champ = _scores({"env_a": [0.4] * 20, "env_extra": [0.0] * 20})
    chal = _scores({"env_a": [0.6] * 20, "env_extra": [0.0] * 20})
    # Only env_a/env_b in cfg; chal has no env_b → insufficient → worse.
    r = cmp.compare(champ, chal, _cfg(margin=0.01), min_dominant_envs=0)
    assert r.winner == "champion"
    by_env = {e.env: e for e in r.per_env}
    assert by_env["env_b"].verdict == ENV_WORSE
