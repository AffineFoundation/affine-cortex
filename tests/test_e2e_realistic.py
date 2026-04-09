"""
Realistic E2E tests simulating production conditions with mocked DAOs.

No real DB writes — all DAOs are AsyncMock. Tests:
- Full multi-round lifecycle with state persistence (load → score → save → reload)
- Realistic checkpoint timeline with default parameters (warmup=2, N=10)
- Window rotation simulation (common tasks grow over rounds)
- Adversarial: copy attack defended by pairwise filter
- Adversarial: champion intermittent absence — weight preserved
- Multi-challenger race with deterministic resolution
"""

import pytest
from unittest.mock import AsyncMock
from affine.src.scorer.scorer import Scorer
from affine.src.scorer.config import ScorerConfig
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.system_config import SystemConfigDAO


# ── Test Harness ─────────────────────────────────────────────────────────────

ENVS = ["env_a", "env_b"]
ENV_CONFIGS = {"env_a": {}, "env_b": {}}
WINDOW = 50
ENV_SC = {"env_a": WINDOW, "env_b": WINDOW}


def build_scoring_data(miners, n_tasks):
    """Build API-format scoring_data with n_tasks per env."""
    data = {}
    for m in miners:
        hk = m["hotkey"]
        rev = m.get("revision", "rev1")
        env_data = {}
        for env, score in m["envs"].items():
            env_data[env] = {
                "all_samples": [
                    {"task_id": i, "score": score, "timestamp": 1e12 + i}
                    for i in range(n_tasks)
                ],
                "sampling_task_ids": list(range(n_tasks)),
                "total_count": n_tasks,
                "completed_count": n_tasks,
                "completeness": 1.0,
            }
        data[f"{hk}#{rev}"] = {
            "uid": m["uid"], "hotkey": hk, "model_revision": rev,
            "model_repo": "test", "first_block": m.get("first_block", 100),
            "env": env_data,
        }
    return data


class MockedDB:
    """In-memory simulation of the persistence layer.

    Mocks the four DAOs the scorer interacts with. Stores state between
    rounds so we can verify load → process → save → reload cycle.
    """

    def __init__(self):
        self.champion = None  # system_config champion record
        self.challenge_states = {}  # hotkey → state dict

        self.snapshots_dao = AsyncMock(spec=ScoreSnapshotsDAO)
        self.scores_dao = AsyncMock(spec=ScoresDAO)
        self.miner_stats_dao = AsyncMock(spec=MinerStatsDAO)
        self.system_config_dao = AsyncMock(spec=SystemConfigDAO)

        # Wire up state persistence behavior
        async def get_champion(*args, **kwargs):
            return self.champion

        async def set_champion(param_name, param_value, **kwargs):
            assert param_name == 'champion'
            self.champion = dict(param_value)

        async def update_challenge_state(hotkey, revision, consecutive_wins,
                                          total_losses, consecutive_losses,
                                          checkpoints_passed, status):
            self.challenge_states[hotkey] = {
                'challenge_consecutive_wins': consecutive_wins,
                'challenge_total_losses': total_losses,
                'challenge_consecutive_losses': consecutive_losses,
                'challenge_checkpoints_passed': checkpoints_passed,
                'challenge_status': status,
                'revision': revision,
            }

        async def get_challenge_state(hotkey, revision):
            state = self.challenge_states.get(hotkey)
            if state is None:
                return {
                    'challenge_consecutive_wins': 0,
                    'challenge_total_losses': 0,
                    'challenge_consecutive_losses': 0,
                    'challenge_checkpoints_passed': 0,
                    'challenge_status': 'sampling',
                    'revision': revision,
                }
            return dict(state)

        self.system_config_dao.get_param_value = AsyncMock(side_effect=get_champion)
        self.system_config_dao.set_param = AsyncMock(side_effect=set_champion)
        self.miner_stats_dao.update_challenge_state = AsyncMock(
            side_effect=update_challenge_state)
        self.miner_stats_dao.get_challenge_state = AsyncMock(
            side_effect=get_challenge_state)


async def run_round(scorer, db, miners_cfg, n_tasks, block_number):
    """One full scoring round with persistence cycle."""
    sd = build_scoring_data(miners_cfg, n_tasks)

    # Load state (simulating main.py)
    champion_state = await db.system_config_dao.get_param_value('champion')
    prev_states = {}
    for key, info in sd.items():
        hk = info['hotkey']
        rev = info['model_revision']
        s = await db.miner_stats_dao.get_challenge_state(hk, rev)
        s['revision'] = rev
        prev_states[hk] = s

    # Score
    result = scorer.calculate_scores(
        scoring_data=sd, environments=ENVS, env_configs=ENV_CONFIGS,
        block_number=block_number, champion_state=champion_state,
        prev_challenge_states=prev_states, env_sampling_counts=ENV_SC,
        print_summary=False)

    # Save (simulating main.py save_results)
    await scorer.save_results(
        result=result,
        score_snapshots_dao=db.snapshots_dao,
        scores_dao=db.scores_dao,
        miner_stats_dao=db.miner_stats_dao,
        system_config_dao=db.system_config_dao,
        block_number=block_number,
    )
    return result


# ── Test 1: Full lifecycle with persistence ──────────────────────────────────

class TestFullLifecycle:

    @pytest.mark.asyncio
    async def test_cold_start_persists_champion(self):
        """First round picks a champion, persists it, second round loads same champion."""
        scorer = Scorer(ScorerConfig())
        db = MockedDB()

        miners = [
            {"uid": 1, "hotkey": "hk1", "envs": {"env_a": 0.5, "env_b": 0.5}},
            {"uid": 2, "hotkey": "hk2", "envs": {"env_a": 0.8, "env_b": 0.7}},
        ]

        r1 = await run_round(scorer, db, miners, n_tasks=WINDOW, block_number=1000)
        assert r1.champion_uid == 2
        assert db.champion is not None
        assert db.champion['hotkey'] == 'hk2'
        assert db.champion['since_block'] == 1000

        r2 = await run_round(scorer, db, miners, n_tasks=WINDOW + 5, block_number=1100)
        assert r2.champion_uid == 2  # Same champion
        # since_block preserved, not overwritten to 1100
        assert db.champion['since_block'] == 1000

    @pytest.mark.asyncio
    async def test_challenge_state_persists_across_rounds(self):
        """Challenge counters survive round-trip through DB mocks."""
        config = ScorerConfig()
        config.CHAMPION_WARMUP_CHECKPOINTS = 0
        scorer = Scorer(config)
        db = MockedDB()

        miners = [
            {"uid": 1, "hotkey": "hk_weak", "envs": {"env_a": 0.3, "env_b": 0.3}},
            {"uid": 2, "hotkey": "hk_strong", "envs": {"env_a": 0.7, "env_b": 0.7}},
        ]

        # Round 1: cold start picks uid 2; uid 1 challenges, fails checkpoint 1
        r1 = await run_round(scorer, db, miners, n_tasks=WINDOW, block_number=1000)
        assert r1.champion_uid == 2

        s1 = db.challenge_states['hk_weak']
        assert s1['challenge_checkpoints_passed'] == 1
        assert s1['challenge_total_losses'] == 1
        assert s1['challenge_status'] == 'sampling'

        # Round 2: more data, checkpoint 2 fires, 2 consecutive losses → terminated
        await run_round(scorer, db, miners, n_tasks=2 * WINDOW, block_number=1100)
        s2 = db.challenge_states['hk_weak']
        assert s2['challenge_checkpoints_passed'] == 2
        assert s2['challenge_total_losses'] == 2
        assert s2['challenge_status'] == 'terminated'


# ── Test 2: Realistic dethrone timeline ──────────────────────────────────────

class TestRealisticDethrone:

    @pytest.mark.asyncio
    async def test_dethrone_takes_n_plus_warmup_checkpoints(self):
        """With default config (warmup=2, N=10), dethrone takes 12 checkpoints."""
        config = ScorerConfig()
        # Use defaults
        assert config.CHAMPION_WARMUP_CHECKPOINTS == 2
        assert config.CHAMPION_CONSECUTIVE_WINS_REQUIRED == 10
        scorer = Scorer(config)
        db = MockedDB()

        # Set initial champion (weak) via cold start round 0
        miners = [
            {"uid": 1, "hotkey": "weak", "envs": {"env_a": 0.3, "env_b": 0.3}},
        ]
        await run_round(scorer, db, miners, n_tasks=WINDOW, block_number=1000)
        assert db.champion['hotkey'] == 'weak'

        # Now add a strong challenger
        miners = [
            {"uid": 1, "hotkey": "weak", "envs": {"env_a": 0.3, "env_b": 0.3}},
            {"uid": 2, "hotkey": "strong", "envs": {"env_a": 0.9, "env_b": 0.9}},
        ]

        # Run rounds with growing data — each round adds enough data for next checkpoint
        block = 1000
        for cp in range(1, 13):  # 12 checkpoints needed (2 warmup + 10 wins)
            block += 100
            n_tasks = cp * WINDOW  # Each round, data grows by one window
            result = await run_round(scorer, db, miners, n_tasks=n_tasks, block_number=block)

        # After 12 checkpoints, strong should have dethroned weak
        assert result.champion_uid == 2
        assert db.champion['hotkey'] == 'strong'
        # since_block reflects the dethrone block
        assert db.champion['since_block'] == block

    @pytest.mark.asyncio
    async def test_dethrone_does_not_fire_during_warmup(self):
        """Strong challenger at checkpoint 2 (warmup) does not become champion."""
        config = ScorerConfig()  # warmup=2
        scorer = Scorer(config)
        db = MockedDB()

        # Initialize with weak champion
        miners_init = [
            {"uid": 1, "hotkey": "weak", "envs": {"env_a": 0.3, "env_b": 0.3}},
        ]
        await run_round(scorer, db, miners_init, n_tasks=WINDOW, block_number=1000)

        # Add strong challenger
        miners = [
            {"uid": 1, "hotkey": "weak", "envs": {"env_a": 0.3, "env_b": 0.3}},
            {"uid": 2, "hotkey": "strong", "envs": {"env_a": 0.9, "env_b": 0.9}},
        ]

        # 2 rounds, each crossing one checkpoint — both warmup
        for cp in range(1, 3):
            await run_round(scorer, db, miners, n_tasks=cp * WINDOW, block_number=1000 + 100 * cp)

        # Still weak as champion (warmup hasn't finished)
        assert db.champion['hotkey'] == 'weak'
        chal = db.challenge_states.get('strong')
        assert chal['challenge_checkpoints_passed'] == 2
        assert chal['challenge_consecutive_wins'] == 0  # Not counted during warmup


# ── Test 3: Anti-plagiarism in multi-round ───────────────────────────────────

class TestPlagiarismDefense:

    @pytest.mark.asyncio
    async def test_copier_terminated_by_pairwise(self):
        """Original miner registered first; copier with similar scores is terminated."""
        config = ScorerConfig()
        config.PAIRWISE_MIN_WINDOWS = 3
        scorer = Scorer(config)
        db = MockedDB()

        # Initialize champion (different hotkey)
        miners_init = [
            {"uid": 1, "hotkey": "champ", "first_block": 50,
             "envs": {"env_a": 0.4, "env_b": 0.4}},
        ]
        await run_round(scorer, db, miners_init, n_tasks=WINDOW, block_number=1000)

        # Add original (better than champion) and a copy (slightly worse)
        miners = [
            {"uid": 1, "hotkey": "champ", "first_block": 50,
             "envs": {"env_a": 0.4, "env_b": 0.4}},
            {"uid": 2, "hotkey": "original", "first_block": 100,
             "envs": {"env_a": 0.7, "env_b": 0.7}},
            {"uid": 3, "hotkey": "copy", "first_block": 200,
             "envs": {"env_a": 0.71, "env_b": 0.71}},
        ]

        # Run rounds until original and copy share 3 windows
        result = await run_round(scorer, db, miners, n_tasks=3 * WINDOW, block_number=1100)

        # Copy terminated by pairwise filter (older original wins)
        copy_state = db.challenge_states.get('copy')
        assert copy_state['challenge_status'] == 'terminated'

        # Original survives
        orig_state = db.challenge_states.get('original')
        assert orig_state['challenge_status'] == 'sampling'

    @pytest.mark.asyncio
    async def test_copier_cannot_become_champion(self):
        """Even if copier is faster, they can't become champion before being filtered."""
        config = ScorerConfig()
        config.CHAMPION_WARMUP_CHECKPOINTS = 0
        config.CHAMPION_CONSECUTIVE_WINS_REQUIRED = 1
        config.PAIRWISE_MIN_WINDOWS = 3
        scorer = Scorer(config)
        db = MockedDB()

        # Champion exists
        miners_init = [
            {"uid": 1, "hotkey": "champ", "first_block": 50,
             "envs": {"env_a": 0.4, "env_b": 0.4}},
        ]
        await run_round(scorer, db, miners_init, n_tasks=WINDOW, block_number=1000)

        # Original and copy both could dominate champion
        miners = [
            {"uid": 1, "hotkey": "champ", "first_block": 50,
             "envs": {"env_a": 0.4, "env_b": 0.4}},
            {"uid": 2, "hotkey": "original", "first_block": 100,
             "envs": {"env_a": 0.8, "env_b": 0.8}},
            {"uid": 3, "hotkey": "copy", "first_block": 200,
             "envs": {"env_a": 0.81, "env_b": 0.81}},
        ]

        result = await run_round(scorer, db, miners, n_tasks=3 * WINDOW, block_number=1100)
        # Pairwise filter terminates copy first; then original dethrones champion
        assert result.champion_uid == 2  # original
        assert db.challenge_states.get('copy')['challenge_status'] == 'terminated'


# ── Test 4: Champion intermittent absence ────────────────────────────────────

class TestChampionAbsence:

    @pytest.mark.asyncio
    async def test_champion_absent_then_returns(self):
        """Champion disappears for some rounds, then returns. Weight preserved throughout."""
        scorer = Scorer(ScorerConfig())
        db = MockedDB()

        full = [
            {"uid": 1, "hotkey": "champ", "envs": {"env_a": 0.5, "env_b": 0.5}},
            {"uid": 2, "hotkey": "other", "envs": {"env_a": 0.4, "env_b": 0.4}},
        ]
        only_other = [
            {"uid": 2, "hotkey": "other", "envs": {"env_a": 0.4, "env_b": 0.4}},
        ]

        # Round 1: cold start picks champ
        r1 = await run_round(scorer, db, full, n_tasks=WINDOW, block_number=1000)
        assert r1.champion_uid == 1
        original_since_block = db.champion['since_block']

        # Rounds 2-4: champion absent
        for r in range(3):
            r_result = await run_round(scorer, db, only_other,
                                        n_tasks=WINDOW + (r + 1) * 5, block_number=1100 + r * 100)
            assert r_result.champion_uid is None  # Not active this round
            assert r_result.final_weights.get(1) == 1.0  # Weight preserved
            # Champion record unchanged
            assert db.champion['hotkey'] == 'champ'
            assert db.champion['since_block'] == original_since_block

        # Round 5: champion returns
        r5 = await run_round(scorer, db, full, n_tasks=WINDOW + 30, block_number=1500)
        assert r5.champion_uid == 1  # Active again
        assert r5.final_weights[1] == 1.0
        # since_block still preserved
        assert db.champion['since_block'] == original_since_block

    @pytest.mark.asyncio
    async def test_champion_uid_reassigned_during_absence(self):
        """If a different miner takes the champion's UID slot, weight still goes there."""
        scorer = Scorer(ScorerConfig())
        db = MockedDB()

        # Round 1: cold start with hk_a at uid 1
        miners1 = [
            {"uid": 1, "hotkey": "hk_a", "envs": {"env_a": 0.7, "env_b": 0.7}},
        ]
        await run_round(scorer, db, miners1, n_tasks=WINDOW, block_number=1000)
        assert db.champion['hotkey'] == 'hk_a'
        assert db.champion['uid'] == 1

        # Round 2: hk_a gone, hk_b takes uid 1
        miners2 = [
            {"uid": 1, "hotkey": "hk_b", "envs": {"env_a": 0.6, "env_b": 0.6}},
        ]
        r2 = await run_round(scorer, db, miners2, n_tasks=WINDOW + 5, block_number=1100)
        # Old champion (hk_a) record unchanged — weight preserved on uid 1
        assert db.champion['hotkey'] == 'hk_a'
        assert r2.final_weights.get(1) == 1.0
        # hk_b at uid 1 gets the weight (since they sit on the champion's UID)
        # This is intentional: validator weight-setting uses UID, not hotkey


# ── Test 5: Multi-challenger race ────────────────────────────────────────────

class TestMultiChallengerRace:

    @pytest.mark.asyncio
    async def test_best_geo_mean_wins_simultaneous_dethrone(self):
        """Multiple challengers reach N wins same round → highest geo mean wins."""
        config = ScorerConfig()
        config.CHAMPION_WARMUP_CHECKPOINTS = 0
        config.CHAMPION_CONSECUTIVE_WINS_REQUIRED = 1
        config.PAIRWISE_MIN_WINDOWS = 100  # Disable pairwise for this test
        scorer = Scorer(config)
        db = MockedDB()

        # Initialize weak champion
        await run_round(scorer, db, [
            {"uid": 1, "hotkey": "weak", "envs": {"env_a": 0.2, "env_b": 0.2}},
        ], n_tasks=WINDOW, block_number=1000)

        # Multiple challengers, all dominating
        miners = [
            {"uid": 1, "hotkey": "weak", "envs": {"env_a": 0.2, "env_b": 0.2}},
            {"uid": 2, "hotkey": "good", "envs": {"env_a": 0.7, "env_b": 0.7}},
            {"uid": 3, "hotkey": "best", "envs": {"env_a": 0.9, "env_b": 0.9}},
        ]
        r = await run_round(scorer, db, miners, n_tasks=WINDOW + 10, block_number=1100)
        # uid 3 has highest geo mean → wins
        assert r.champion_uid == 3
