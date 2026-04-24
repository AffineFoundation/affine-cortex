"""
End-to-end and integration tests for the scoring pipeline.

Tests the full Scorer.calculate_scores() flow with realistic data,
multi-round simulations, and mocked DB persistence.
"""

import pytest
from unittest.mock import AsyncMock
from affine.src.scorer.scorer import Scorer
from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.stage1_collector import Stage1Collector
from affine.src.scorer.utils import geometric_mean
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.system_config import SystemConfigDAO


# ── Helpers ──────────────────────────────────────────────────────────────────

ENVS = ["env_a", "env_b"]
ENV_CONFIGS = {"env_a": {}, "env_b": {}}
N_TASKS = 100
ENV_SC = {"env_a": N_TASKS, "env_b": N_TASKS}  # Window size = N_TASKS

# Tests use 2 envs — override MIN_DOMINANT_ENVS to 0 (strict Pareto)
ScorerConfig.WIN_MIN_DOMINANT_ENVS = 0


def scoring_data(miners, n_tasks=None):
    """Build API-format scoring_data from simple miner defs."""
    n = n_tasks or N_TASKS
    data = {}
    for m in miners:
        hk, rev = m["hotkey"], m.get("revision", "rev1")
        env_data = {}
        for env, score in m["envs"].items():
            env_data[env] = {
                "all_samples": [{"task_id": i, "score": score, "timestamp": 1e12 + i}
                                for i in range(n)],
                "sampling_task_ids": list(range(n)),
                "total_count": n, "completed_count": n, "completeness": 1.0,
            }
        data[f"{hk}#{rev}"] = {
            "uid": m["uid"], "hotkey": hk, "model_revision": rev,
            "model_repo": "test/model", "first_block": m.get("first_block", 100),
            "env": env_data,
        }
    return data


def run_rounds(config, miners_fn, n_rounds, initial_champion=None):
    """Run n_rounds of scoring, return list of results.
    Task count grows each round: (r+1)*N_TASKS, so CP advances naturally."""
    scorer = Scorer(config)
    champion_state = initial_champion
    challenge_states = {}
    history = []

    for r in range(n_rounds):
        n = (r + 1) * N_TASKS
        sd = scoring_data(miners_fn(r), n_tasks=n)
        result = scorer.calculate_scores(
            scoring_data=sd, environments=ENVS,
            block_number=1000 + r, champion_state=champion_state,
            prev_challenge_states=challenge_states,
            env_sampling_counts=ENV_SC, print_summary=False)
        history.append(result)

        # Persist state for next round
        challenge_states = {}
        for uid, m in result.miners.items():
            challenge_states[m.hotkey] = {
                "challenge_consecutive_wins": m.challenge_consecutive_wins,
                "challenge_total_losses": m.challenge_total_losses,
                "challenge_consecutive_losses": m.challenge_consecutive_losses,
                "challenge_checkpoints_passed": m.challenge_checkpoints_passed,
                "challenge_status": m.challenge_status,
                "revision": m.model_revision,
            }
        if result.champion_uid is not None:
            cm = result.miners[result.champion_uid]
            champion_state = {
                "hotkey": cm.hotkey, "revision": cm.model_revision,
                "uid": result.champion_uid, "since_block": 1000 + r,
            }

    return history


# ── E2E: Cold Start ──────────────────────────────────────────────────────────

class TestNoChampion:

    def test_no_champion_returns_zero_weights(self):
        """Without pre-set champion, all weights are 0."""
        config = ScorerConfig()
        sd = scoring_data([
            {"uid": 1, "hotkey": "hk1", "envs": {"env_a": 0.4, "env_b": 0.4}},
            {"uid": 2, "hotkey": "hk2", "envs": {"env_a": 0.8, "env_b": 0.7}},
        ])
        result = Scorer(config).calculate_scores(
            scoring_data=sd, environments=ENVS,
            block_number=1000, env_sampling_counts=ENV_SC, print_summary=False)
        assert result.champion_uid is None
        assert all(w == 0.0 for w in result.final_weights.values())


# ── E2E: Multi-Round ─────────────────────────────────────────────────────────

class TestMultiRound:

    def test_challenger_dethrones_after_n(self):
        config = ScorerConfig()
        config.CHAMPION_DETHRONE_MIN_CHECKPOINT = 3
        config.CHAMPION_TERMINATION_TOTAL_LOSSES = 100
        config.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES = 100

        def miners(r):
            return [{"uid": 1, "hotkey": "old", "envs": {"env_a": 0.3, "env_b": 0.3}},
                    {"uid": 2, "hotkey": "new", "envs": {"env_a": 0.8, "env_b": 0.8}}]

        champ = {"hotkey": "old", "revision": "rev1", "uid": 1, "since_block": 999}
        h = run_rounds(config, miners, 4, initial_champion=champ)
        assert h[-1].champion_uid == 2

    def test_weak_miners_terminated_after_checkpoints(self):
        """Weak miner loses at each checkpoint → terminated after M losses."""
        config = ScorerConfig()
        config.CHAMPION_TERMINATION_TOTAL_LOSSES = 2
        config.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES = 2

        # Data grows each round: round r has (r+1)*N_TASKS tasks → new checkpoint each round
        def miners(r):
            n = (r + 1) * N_TASKS
            return [{"uid": 1, "hotkey": "champ", "envs": {"env_a": 0.8, "env_b": 0.8}},
                    {"uid": 2, "hotkey": "weak", "envs": {"env_a": 0.3, "env_b": 0.3}}]

        # Override scoring_data to produce growing task counts
        scorer = Scorer(config)
        champion_state = {"hotkey": "champ", "revision": "rev1", "uid": 1, "since_block": 999}
        challenge_states = {}

        for r in range(4):
            n = (r + 1) * N_TASKS
            sd = {}
            for m in [{"uid": 1, "hotkey": "champ", "envs": {"env_a": 0.8, "env_b": 0.8}},
                      {"uid": 2, "hotkey": "weak", "envs": {"env_a": 0.3, "env_b": 0.3}}]:
                hk = m["hotkey"]
                env_data = {}
                for env, score in m["envs"].items():
                    env_data[env] = {
                        "all_samples": [{"task_id": i, "score": score, "timestamp": 1e12+i}
                                        for i in range(n)],
                        "sampling_task_ids": list(range(n)),
                        "total_count": n, "completed_count": n, "completeness": 1.0,
                    }
                sd[f"{hk}#rev1"] = {"uid": m["uid"], "hotkey": hk, "model_revision": "rev1",
                                     "model_repo": "test", "first_block": 100, "env": env_data}

            result = scorer.calculate_scores(
                scoring_data=sd, environments=ENVS,
                block_number=1000+r, champion_state=champion_state,
                prev_challenge_states=challenge_states,
                env_sampling_counts=ENV_SC, print_summary=False)

            challenge_states = {}
            for uid, m in result.miners.items():
                challenge_states[m.hotkey] = {
                    "challenge_consecutive_wins": m.challenge_consecutive_wins,
                    "challenge_total_losses": m.challenge_total_losses,
                    "challenge_consecutive_losses": m.challenge_consecutive_losses,
                    "challenge_checkpoints_passed": m.challenge_checkpoints_passed,
                    "challenge_status": m.challenge_status,
                    "revision": m.model_revision,
                }
            if result.champion_uid is not None:
                cm = result.miners[result.champion_uid]
                champion_state = {"hotkey": cm.hotkey, "revision": cm.model_revision,
                                  "uid": result.champion_uid, "since_block": 1000+r}

            if result.miners[2].challenge_status == "terminated":
                break

        assert result.miners[2].challenge_status == "terminated"

    def test_champion_absent_keeps_weight(self):
        """Champion absent for many rounds → weight stays on champion UID."""
        config = ScorerConfig()

        def miners(r):
            base = [{"uid": 2, "hotkey": "backup", "envs": {"env_a": 0.5, "env_b": 0.5}}]
            if r < 2:
                base.insert(0, {"uid": 1, "hotkey": "champ",
                                "envs": {"env_a": 0.6, "env_b": 0.6}})
            return base

        champ = {"hotkey": "champ", "revision": "rev1", "uid": 1, "since_block": 999}
        h = run_rounds(config, miners, 6, initial_champion=champ)
        # Champion (uid 1) absent after r=2, but never replaced
        assert h[-1].final_weights.get(1) == 1.0
        assert h[-1].champion_uid is None  # Not actively present

    def test_invariants_10_miners_10_rounds(self):
        config = ScorerConfig()
        config.CHAMPION_DETHRONE_MIN_CHECKPOINT = 3
        config.CHAMPION_TERMINATION_TOTAL_LOSSES = 3
        config.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES = 2

        def miners(r):
            ms = [{"uid": i, "hotkey": f"hk{i}", "revision": f"r{i}",
                   "envs": {"env_a": 0.3 + i * 0.05, "env_b": 0.3 + i * 0.04}}
                  for i in range(1, 11)]
            if r >= 5:
                ms = [m for m in ms if m["uid"] != 10]
            return ms

        champ = {"hotkey": "hk10", "revision": "r10", "uid": 10, "since_block": 999}
        for r in run_rounds(config, miners, 10, initial_champion=champ):
            champions = [uid for uid, m in r.miners.items() if m.is_champion]
            assert len(champions) <= 1
            w = sum(r.final_weights.get(uid, 0) for uid in r.miners)
            assert w == pytest.approx(1.0) or w == pytest.approx(0.0)
            if r.champion_uid and r.champion_uid in r.miners:
                assert r.miners[r.champion_uid].challenge_status != "terminated"


# ── Integration: DB Persistence ──────────────────────────────────────────────

class TestSaveResults:

    @pytest.mark.asyncio
    async def test_save_correct_fields(self):
        scorer = Scorer(ScorerConfig())
        sd = scoring_data([
            {"uid": 1, "hotkey": "hk1", "envs": {"env_a": 0.7, "env_b": 0.7}},
            {"uid": 2, "hotkey": "hk2", "envs": {"env_a": 0.3, "env_b": 0.3}},
        ])
        champ = {"hotkey": "hk1", "revision": "rev1", "uid": 1, "since_block": 4999}
        result = scorer.calculate_scores(
            scoring_data=sd, environments=ENVS, champion_state=champ,
            block_number=5000, env_sampling_counts=ENV_SC, print_summary=False)

        scores = AsyncMock(spec=ScoresDAO)
        sysconf = AsyncMock(spec=SystemConfigDAO)
        sysconf.get_param_value = AsyncMock(return_value=champ)

        await scorer.save_results(
            result=result, score_snapshots_dao=AsyncMock(spec=ScoreSnapshotsDAO),
            scores_dao=scores, miner_stats_dao=AsyncMock(spec=MinerStatsDAO),
            system_config_dao=sysconf, block_number=5000)

        assert scores.save_score.call_count == 2
        for c in scores.save_score.call_args_list:
            kw = c.kwargs
            assert "challenge_info" in kw
            assert "elo_rating" not in kw

        sysconf.set_param.assert_called_once()

    @pytest.mark.asyncio
    async def test_since_block_preserved(self):
        scorer = Scorer(ScorerConfig())
        sd = scoring_data([{"uid": 1, "hotkey": "hk1", "envs": {"env_a": 0.7, "env_b": 0.7}}])
        result = scorer.calculate_scores(
            scoring_data=sd, environments=ENVS,
            block_number=9999, env_sampling_counts=ENV_SC,
            champion_state={"hotkey": "hk1", "revision": "rev1", "uid": 1,
                            "since_block": 5000},
            print_summary=False)

        sysconf = AsyncMock(spec=SystemConfigDAO)
        sysconf.get_param_value = AsyncMock(return_value={
            "hotkey": "hk1", "revision": "rev1", "uid": 1, "since_block": 5000})

        await scorer.save_results(
            result=result, score_snapshots_dao=AsyncMock(spec=ScoreSnapshotsDAO),
            scores_dao=AsyncMock(spec=ScoresDAO),
            miner_stats_dao=AsyncMock(spec=MinerStatsDAO),
            system_config_dao=sysconf, block_number=9999)

        assert sysconf.set_param.call_args.kwargs["param_value"]["since_block"] == 5000


# ── Utils + Config ───────────────────────────────────────────────────────────

class TestUtils:
    def test_geometric_mean(self):
        assert abs(geometric_mean([4.0, 9.0]) - 6.0) < 1e-9
        assert geometric_mean([0.0, 1.0], epsilon=0.1) > 0.0
        assert geometric_mean([]) == 0.0

    def test_geometric_mean_small_values(self):
        result = geometric_mean([1e-200] * 1000)
        assert result > 0.0
        assert abs(result - 1e-200) / 1e-200 < 1e-9

    def test_geometric_mean_returns_zero_for_non_positive_values(self):
        assert geometric_mean([0.0, 1.0]) == 0.0
        assert geometric_mean([-1.0, 2.0]) == 0.0

# ── Stage1: historical_count + recent-N×window avg ──────────────────────────

class TestStage1RecentAvg:
    """`avg_score` is computed from the most recent N×window samples;
    `historical_count` is the full lifetime distinct task count."""

    def _scoring_data(self, samples_by_env):
        """Wrap raw samples into the API scoring_data shape."""
        env_data = {}
        for env, samples in samples_by_env.items():
            env_data[env] = {
                "all_samples": samples,
                "sampling_task_ids": [],   # not used by stage1 anymore
                "total_count": len(samples),
                "completed_count": len(samples),
                "completeness": 1.0,
            }
        return {
            "hk1#rev1": {
                "uid": 1, "hotkey": "hk1", "model_revision": "rev1",
                "model_repo": "test/model", "first_block": 100,
                "env": env_data,
            }
        }

    def test_recent_window_truncation(self):
        """Champion case: 2000 samples, cap=N×window=12×100=1200.
        Old 800 scored 0.0, recent 1200 scored 1.0 → avg should be 1.0."""
        config = ScorerConfig()
        config.CHAMPION_DETHRONE_MIN_CHECKPOINT = 10
        collector = Stage1Collector(config)

        samples = []
        for i in range(500):                                    # oldest, score 0
            samples.append({"task_id": i, "score": 0.0, "timestamp": 1_000_000 + i})
        for i in range(500, 1500):                              # newest, score 1
            samples.append({"task_id": i, "score": 1.0, "timestamp": 1_000_000 + i})

        sd = self._scoring_data({"env_a": samples})
        out = collector.collect(sd, ["env_a"], env_sampling_counts={"env_a": 100})

        env = out.miners[1].env_scores["env_a"]
        # cap = 10 × 100 = 1000; the 1000 newest are exactly the score-1 batch
        assert env.avg_score == pytest.approx(1.0)
        assert env.historical_count == 1500
        assert len(env.all_task_scores) == 1500   # Pareto sees full history

    def test_challenger_under_cap_uses_full_history(self):
        """Challenger case: 200 samples, cap=1000 → recent == full → avg=mean(all)."""
        config = ScorerConfig()
        config.CHAMPION_DETHRONE_MIN_CHECKPOINT = 10
        collector = Stage1Collector(config)

        samples = [{"task_id": i, "score": 0.5, "timestamp": 1_000_000 + i}
                   for i in range(200)]
        sd = self._scoring_data({"env_a": samples})
        out = collector.collect(sd, ["env_a"], env_sampling_counts={"env_a": 100})

        env = out.miners[1].env_scores["env_a"]
        assert env.avg_score == pytest.approx(0.5)
        assert env.historical_count == 200

    def test_dedup_keeps_newest_score_per_task(self):
        """Same task_id rescored later → newest timestamp wins in avg."""
        config = ScorerConfig()
        collector = Stage1Collector(config)

        samples = [
            {"task_id": 1, "score": 0.0, "timestamp": 1_000},
            {"task_id": 1, "score": 1.0, "timestamp": 2_000},   # newer
            {"task_id": 2, "score": 1.0, "timestamp": 1_500},
        ]
        sd = self._scoring_data({"env_a": samples})
        out = collector.collect(sd, ["env_a"], env_sampling_counts={"env_a": 10})

        env = out.miners[1].env_scores["env_a"]
        # Two unique tasks, both newest scores = 1.0
        assert env.avg_score == pytest.approx(1.0)
        assert env.historical_count == 2

    def test_no_window_size_uses_full_history(self):
        """If env_sampling_counts is missing for an env, use full history."""
        config = ScorerConfig()
        collector = Stage1Collector(config)

        samples = [{"task_id": i, "score": 0.7, "timestamp": 1_000 + i}
                   for i in range(50)]
        sd = self._scoring_data({"env_a": samples})
        out = collector.collect(sd, ["env_a"], env_sampling_counts={})

        env = out.miners[1].env_scores["env_a"]
        assert env.avg_score == pytest.approx(0.7)
        assert env.historical_count == 50


class TestSaveResultsHistoricalCount:
    """save_results should persist historical_count in scores_by_env."""

    @pytest.mark.asyncio
    async def test_historical_count_in_save(self):
        scorer = Scorer(ScorerConfig())
        sd = scoring_data([
            {"uid": 1, "hotkey": "hk1", "envs": {"env_a": 0.7, "env_b": 0.7}},
        ])
        champ = {"hotkey": "hk1", "revision": "rev1", "uid": 1, "since_block": 4999}
        result = scorer.calculate_scores(
            scoring_data=sd, environments=ENVS, champion_state=champ,
            block_number=5000, env_sampling_counts=ENV_SC, print_summary=False)

        scores = AsyncMock(spec=ScoresDAO)
        sysconf = AsyncMock(spec=SystemConfigDAO)
        sysconf.get_param_value = AsyncMock(return_value=champ)

        await scorer.save_results(
            result=result, score_snapshots_dao=AsyncMock(spec=ScoreSnapshotsDAO),
            scores_dao=scores, miner_stats_dao=AsyncMock(spec=MinerStatsDAO),
            system_config_dao=sysconf, block_number=5000)

        kw = scores.save_score.call_args_list[0].kwargs
        for env_name, env_data in kw["scores_by_env"].items():
            assert "historical_count" in env_data
            assert env_data["historical_count"] == N_TASKS


class TestConfig:
    def test_defaults_and_validation(self):
        c = ScorerConfig()
        assert c.CHAMPION_DETHRONE_MIN_CHECKPOINT == 10
        assert c.WIN_MARGIN_START == 0.02
        assert c.WIN_MARGIN_END == 0.03
        assert not hasattr(c, 'ELO_D')
        assert not hasattr(c, 'Z_SCORE')
        assert not hasattr(c, 'MIN_IMPROVEMENT')
        ScorerConfig.validate()
        d = ScorerConfig.to_dict()
        assert 'win_margin_start' in d
        assert 'z_score' not in d
