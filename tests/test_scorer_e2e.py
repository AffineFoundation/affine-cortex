"""
End-to-end and integration tests for the scoring pipeline.

Tests the full Scorer.calculate_scores() flow with realistic data,
multi-round simulations, and mocked DB persistence.
"""

import pytest
from unittest.mock import AsyncMock
from affine.src.scorer.scorer import Scorer
from affine.src.scorer.config import ScorerConfig
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


def scoring_data(miners):
    """Build API-format scoring_data from simple miner defs."""
    data = {}
    for m in miners:
        hk, rev = m["hotkey"], m.get("revision", "rev1")
        env_data = {}
        for env, score in m["envs"].items():
            env_data[env] = {
                "all_samples": [{"task_id": i, "score": score, "timestamp": 1e12 + i}
                                for i in range(N_TASKS)],
                "sampling_task_ids": list(range(N_TASKS)),
                "total_count": N_TASKS, "completed_count": N_TASKS, "completeness": 1.0,
            }
        data[f"{hk}#{rev}"] = {
            "uid": m["uid"], "hotkey": hk, "model_revision": rev,
            "model_repo": "test/model", "first_block": m.get("first_block", 100),
            "env": env_data,
        }
    return data


def run_rounds(config, miners_fn, n_rounds):
    """Run n_rounds of scoring, return list of results."""
    scorer = Scorer(config)
    champion_state = None
    challenge_states = {}
    history = []

    for r in range(n_rounds):
        sd = scoring_data(miners_fn(r))
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

class TestColdStart:

    def test_picks_best_and_starts_challenges(self):
        config = ScorerConfig()
        sd = scoring_data([
            {"uid": 1, "hotkey": "hk1", "envs": {"env_a": 0.4, "env_b": 0.4}},
            {"uid": 2, "hotkey": "hk2", "envs": {"env_a": 0.8, "env_b": 0.7}},
        ])
        result = Scorer(config).calculate_scores(
            scoring_data=sd, environments=ENVS,
            block_number=1000, env_sampling_counts=ENV_SC, print_summary=False)
        assert result.champion_uid == 2
        assert result.final_weights[2] == 1.0
        # Checkpoint 1 is warmup (default warmup=2) → loss not counted
        assert result.miners[1].challenge_total_losses == 0
        assert result.miners[1].challenge_checkpoints_passed == 1


# ── E2E: Multi-Round ─────────────────────────────────────────────────────────

class TestMultiRound:

    def test_challenger_dethrones_after_n(self):
        config = ScorerConfig()
        config.CHAMPION_CONSECUTIVE_WINS_REQUIRED = 3
        config.CHAMPION_TERMINATION_TOTAL_LOSSES = 100
        config.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES = 100

        def miners(r):
            return [{"uid": 1, "hotkey": "old", "envs": {"env_a": 0.3, "env_b": 0.3}},
                    {"uid": 2, "hotkey": "new", "envs": {"env_a": 0.8, "env_b": 0.8}}]

        h = run_rounds(config, miners, 4)
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
        champion_state = None
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

        h = run_rounds(config, miners, 6)
        # Champion (uid 1) absent after r=2, but never replaced
        # All later rounds should still attribute weight to uid 1
        assert h[-1].final_weights.get(1) == 1.0
        assert h[-1].champion_uid is None  # Not actively present

    def test_invariants_10_miners_10_rounds(self):
        config = ScorerConfig()
        config.CHAMPION_CONSECUTIVE_WINS_REQUIRED = 3
        config.CHAMPION_TERMINATION_TOTAL_LOSSES = 3
        config.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES = 2

        def miners(r):
            ms = [{"uid": i, "hotkey": f"hk{i}", "revision": f"r{i}",
                   "envs": {"env_a": 0.3 + i * 0.05, "env_b": 0.3 + i * 0.04}}
                  for i in range(1, 11)]
            if r >= 5:
                ms = [m for m in ms if m["uid"] != 10]
            return ms

        for r in run_rounds(config, miners, 10):
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
        result = scorer.calculate_scores(
            scoring_data=sd, environments=ENVS,
            block_number=5000, env_sampling_counts=ENV_SC, print_summary=False)

        scores = AsyncMock(spec=ScoresDAO)
        sysconf = AsyncMock(spec=SystemConfigDAO)
        sysconf.get_param_value = AsyncMock(return_value=None)

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

class TestConfig:
    def test_defaults_and_validation(self):
        c = ScorerConfig()
        assert c.CHAMPION_CONSECUTIVE_WINS_REQUIRED == 10
        assert c.PARETO_MARGIN == 0.02
        assert not hasattr(c, 'ELO_D')
        assert not hasattr(c, 'Z_SCORE')
        assert not hasattr(c, 'MIN_IMPROVEMENT')
        ScorerConfig.validate()
        d = ScorerConfig.to_dict()
        assert 'pareto_margin' in d
        assert 'z_score' not in d
