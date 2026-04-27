"""Unit tests for the fix: fairness gate must read `success` (completed
samples) not `samples` (total attempts) from env_stats.last_1hour.

If the fairness gate uses `samples`, an env whose chute is
rate-limiting most calls will appear to be "ahead" of its target rate
(because each rate-limit error increments `samples`) and get capped to
the 1-slot floor — the opposite of what fairness should do. Target
rate = `rotation_count * 3600 / rotation_interval` is set in the unit
of completed samples per hour, so the actual must be measured the same
way.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from affine.src.scheduler.sampling_scheduler import PerMinerSamplingScheduler


HK = "5Hotkey"
REV = "rev_abc"
ENVS = ["GAME", "MEMORY", "LIVEWEB"]


def make_scheduler():
    """Build a scheduler with all DAOs mocked. Caller wires up
    miner_stats_dao.get_miner_stats."""
    s = PerMinerSamplingScheduler(
        system_config_dao=MagicMock(),
        task_pool_dao=MagicMock(),
        sample_results_dao=MagicMock(),
        miners_dao=MagicMock(),
        miner_stats_dao=MagicMock(),
        scheduling_interval=10,
    )
    return s


def stats_with_env(env_data: dict) -> dict:
    """Build a miner_stats record where env_stats.last_1hour comes from env_data."""
    env_stats = {
        env: {"last_1hour": payload}
        for env, payload in env_data.items()
    }
    return {
        "hotkey": HK,
        "revision": REV,
        "env_stats": env_stats,
    }


# ─────────────────────────────────────────────────────────────────────────
# _get_miner_actual_rates: must read `success`, not `samples`
# ─────────────────────────────────────────────────────────────────────────


class TestActualRatesReadsSuccess:

    @pytest.mark.asyncio
    async def test_returns_success_field_not_samples(self):
        s = make_scheduler()
        # Mismatch on every env: samples >> success when chute rate-limits.
        s.miner_stats_dao.get_miner_stats = AsyncMock(return_value=stats_with_env({
            "GAME":    {"samples": 37, "success": 1,  "rate_limit_errors": 36},
            "MEMORY":  {"samples": 24, "success": 0,  "rate_limit_errors": 24},
            "LIVEWEB": {"samples": 17, "success": 10, "rate_limit_errors": 6},
        }))

        rates = await s._get_miner_actual_rates(HK, REV, ENVS)

        # Must report success, not samples.
        assert rates["GAME"] == 1.0
        assert rates["MEMORY"] == 0.0
        assert rates["LIVEWEB"] == 10.0
        # And critically NOT the inflated attempt counts:
        assert rates["GAME"] != 37
        assert rates["LIVEWEB"] != 17

    @pytest.mark.asyncio
    async def test_missing_record_returns_zeros(self):
        s = make_scheduler()
        s.miner_stats_dao.get_miner_stats = AsyncMock(return_value=None)

        rates = await s._get_miner_actual_rates(HK, REV, ENVS)

        assert rates == {"GAME": 0.0, "MEMORY": 0.0, "LIVEWEB": 0.0}

    @pytest.mark.asyncio
    async def test_dao_error_returns_zeros(self):
        s = make_scheduler()
        s.miner_stats_dao.get_miner_stats = AsyncMock(side_effect=RuntimeError("boom"))

        rates = await s._get_miner_actual_rates(HK, REV, ENVS)

        assert rates == {"GAME": 0.0, "MEMORY": 0.0, "LIVEWEB": 0.0}

    @pytest.mark.asyncio
    async def test_empty_window_returns_zero(self):
        s = make_scheduler()
        s.miner_stats_dao.get_miner_stats = AsyncMock(return_value=stats_with_env({
            "GAME": {},  # no success/samples fields at all
            "MEMORY": {"samples": 0, "success": 0},
            "LIVEWEB": {"samples": 5, "success": 5},
        }))

        rates = await s._get_miner_actual_rates(HK, REV, ENVS)

        assert rates["GAME"] == 0.0
        assert rates["MEMORY"] == 0.0
        assert rates["LIVEWEB"] == 5.0

    @pytest.mark.asyncio
    async def test_envs_without_stats_default_zero(self):
        """A new env not yet in env_stats → 0.0 (under-target priority)."""
        s = make_scheduler()
        s.miner_stats_dao.get_miner_stats = AsyncMock(return_value=stats_with_env({
            "LIVEWEB": {"samples": 17, "success": 10},
        }))

        rates = await s._get_miner_actual_rates(HK, REV, ENVS)

        assert rates["GAME"] == 0.0
        assert rates["MEMORY"] == 0.0
        assert rates["LIVEWEB"] == 10.0


# ─────────────────────────────────────────────────────────────────────────
# Downstream: _compute_env_completeness uses success-based rates correctly
# ─────────────────────────────────────────────────────────────────────────


class TestCompletenessFromSuccess:

    def test_rate_limited_env_looks_far_under_target(self):
        """The motivating bug. With success-based rates, an env where
        the chute returns RL on most calls correctly looks far behind
        target (low completeness → high deficit_factor → more slots).
        With the old samples-based rates, this same env would look at
        or above target."""
        s = make_scheduler()
        environments = {
            "GAME": {"sampling_config": {"rotation_count": 1, "rotation_interval": 350}},
            # target_per_hour = 1 * 3600/350 ≈ 10.29
        }
        # success = 1, samples = 37 (chute is rate-limiting)
        actual_via_success = {"GAME": 1.0}
        actual_via_samples = {"GAME": 37.0}

        comp_correct = s._compute_env_completeness(["GAME"], environments, actual_via_success)
        comp_buggy = s._compute_env_completeness(["GAME"], environments, actual_via_samples)

        # Success-based: completeness ~ 0.097 → far from 1.0, gets priority.
        assert comp_correct["GAME"] < 0.2
        # Samples-based: completeness clamps at 1.0 → starves the env.
        assert comp_buggy["GAME"] == 1.0

    def test_target_rate_units_match_success_field(self):
        """target = rotation_count * 3600 / rotation_interval is in
        completed-samples-per-hour. Verify the comparison units agree."""
        s = make_scheduler()
        environments = {
            "MEMORY": {"sampling_config": {"rotation_count": 1, "rotation_interval": 430}},
            # target_per_hour ≈ 8.37
        }
        # 8 successful samples/hour → near complete
        rates = {"MEMORY": 8.0}

        comp = s._compute_env_completeness(["MEMORY"], environments, rates)

        assert 0.9 < comp["MEMORY"] <= 1.0

    def test_rotation_disabled_env_skipped(self):
        s = make_scheduler()
        environments = {
            "TERMINAL": {"sampling_config": {"rotation_count": 0, "rotation_interval": 1730}},
        }
        comp = s._compute_env_completeness(["TERMINAL"], environments, {"TERMINAL": 5.0})
        assert "TERMINAL" not in comp


# ─────────────────────────────────────────────────────────────────────────
# Regression: simulate the UID 234 scenario
# ─────────────────────────────────────────────────────────────────────────


class TestUID234Regression:

    @pytest.mark.asyncio
    async def test_rate_limited_envs_do_not_appear_ahead(self):
        """Recreate the production data we observed for UID 234. With
        the fix, the chute-rate-limited envs (GAME, MEMORY, DISTILL,
        SWE) should NOT appear above LIVEWEB's progress, because their
        success counts are low even though their attempt counts are
        high."""
        s = make_scheduler()

        # last_1hour as observed in production for uid=234.
        env_data = {
            "LIVEWEB":      {"samples": 17, "success": 10},  # near target
            "NAVWORLD":     {"samples": 16, "success": 6},
            "DISTILL":      {"samples": 18, "success": 3, "rate_limit_errors": 15},
            "TERMINAL":     {"samples":  4, "success": 1, "rate_limit_errors": 3},
            "SWE-INFINITE": {"samples": 12, "success": 2, "rate_limit_errors": 10},
            "GAME":         {"samples": 37, "success": 1, "rate_limit_errors": 36},
            "MEMORY":       {"samples": 24, "success": 0, "rate_limit_errors": 24},
        }
        s.miner_stats_dao.get_miner_stats = AsyncMock(return_value=stats_with_env(env_data))

        envs = list(env_data.keys())
        rates = await s._get_miner_actual_rates(HK, REV, envs)

        # Targets per env (rotation_count=1; from system_config)
        environments = {
            "LIVEWEB":      {"sampling_config": {"rotation_count": 1, "rotation_interval": 350}},
            "NAVWORLD":     {"sampling_config": {"rotation_count": 1, "rotation_interval": 350}},
            "DISTILL":      {"sampling_config": {"rotation_count": 1, "rotation_interval": 580}},
            "TERMINAL":     {"sampling_config": {"rotation_count": 1, "rotation_interval": 1730}},
            "SWE-INFINITE": {"sampling_config": {"rotation_count": 1, "rotation_interval": 430}},
            "GAME":         {"sampling_config": {"rotation_count": 1, "rotation_interval": 350}},
            "MEMORY":       {"sampling_config": {"rotation_count": 1, "rotation_interval": 430}},
        }

        # Compute progress (success / target) the same way the gate does.
        progress = {}
        for env, cfg in environments.items():
            sc = cfg["sampling_config"]
            target = sc["rotation_count"] * 3600 / sc["rotation_interval"]
            progress[env] = rates[env] / target

        # LIVEWEB is the actual leader with ≈ 0.97 success rate;
        # rate-limited envs (GAME, MEMORY) sit far behind despite their
        # high attempt counts.
        assert progress["LIVEWEB"] > progress["GAME"]
        assert progress["LIVEWEB"] > progress["MEMORY"]
        assert progress["LIVEWEB"] > progress["DISTILL"]
        assert progress["LIVEWEB"] > progress["SWE-INFINITE"]

        # The fairness gate caps envs whose progress is above
        # `min(progress) + PROGRESS_LEADER_BUFFER`. The rate-limited
        # envs that were the FAULTY leaders under the samples-based
        # signal must now correctly sit at or below LIVEWEB.
        cap_threshold = (
            min(progress.values())
            + PerMinerSamplingScheduler.PROGRESS_LEADER_BUFFER
        )

        # LIVEWEB is correctly identified as ahead-of-min and would be
        # capped this tick, freeing the slot budget for laggards.
        assert progress["LIVEWEB"] > cap_threshold

        # MEMORY is the laggard and is NOT capped — it gets full
        # weighted slot share to catch up.
        assert progress["MEMORY"] <= cap_threshold

        # Sanity vs the buggy samples-based progress: under the OLD
        # signal, GAME / MEMORY would have looked far ABOVE LIVEWEB
        # (37/12.34 ≈ 3.0 and 24/8.61 ≈ 2.79 vs 17/12.34 ≈ 1.38).
        # Verify we are NOT producing those numbers anymore.
        assert progress["GAME"] < 1.0
        assert progress["MEMORY"] < 1.0
        assert progress["GAME"] < progress["LIVEWEB"]
        assert progress["MEMORY"] < progress["LIVEWEB"]
