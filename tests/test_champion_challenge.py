"""
Unit tests for ChampionChallenge with checkpoint-gated comparisons.

A comparison only fires when common_tasks crosses a window-size boundary.
First WARMUP checkpoints are logged but don't count toward wins/losses.
"""

import pytest
from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.models import MinerData, EnvScore
from affine.src.scorer.champion_challenge import ChampionChallenge

ENVS = ["env_a", "env_b"]
WINDOW = 50
ENV_SC = {"env_a": WINDOW, "env_b": WINDOW}
# Most tests use warmup=0 to test core logic directly
NO_WARMUP = {"CHAMPION_WARMUP_CHECKPOINTS": 0}


def cfg(**kw):
    c = ScorerConfig()
    merged = {**NO_WARMUP, **kw}
    for k, v in merged.items():
        setattr(c, k, v)
    return c


def miner(uid, hotkey, revision="rev1", first_block=100, scores=None, n_tasks=WINDOW):
    m = MinerData(uid=uid, hotkey=hotkey, model_revision=revision,
                  model_repo="test/model", first_block=first_block)
    if scores:
        m.env_scores = {env: _env(s, n_tasks) for env, s in zip(ENVS, scores)}
    return m


def _env(avg, n):
    tasks = {i: avg for i in range(n)}
    return EnvScore(avg_score=avg, sample_count=n, completeness=1.0,
                    all_task_scores=tasks)


def cs(hotkey="hk_champ", revision="rev1", uid=1):
    return {"hotkey": hotkey, "revision": revision, "uid": uid, "since_block": 100}


def prev(hotkey, wins=0, tl=0, cl=0, cp=0, status="sampling", revision="rev1"):
    return {hotkey: {"challenge_consecutive_wins": wins,
                     "challenge_total_losses": tl,
                     "challenge_consecutive_losses": cl,
                     "challenge_checkpoints_passed": cp,
                     "challenge_status": status, "revision": revision}}


def run(cc, miners, champion_state=None, prev_states=None):
    return cc.run(miners, ENVS, ENV_SC, champion_state, prev_states or {})


# ── Checkpoint Gate ──────────────────────────────────────────────────────────

class TestCheckpointGate:

    def test_first_checkpoint_fires(self):
        cc = ChampionChallenge(cfg(CHAMPION_CONSECUTIVE_WINS_REQUIRED=3))
        miners = {1: miner(1, "hk_champ", scores=[0.3, 0.3]),
                  2: miner(2, "hk_chal", scores=[0.6, 0.6])}
        r = run(cc, miners, cs())
        assert miners[2].challenge_consecutive_wins == 1
        assert miners[2].challenge_checkpoints_passed == 1
        assert len(r.comparisons) == 1

    def test_below_window_no_comparison(self):
        cc = ChampionChallenge(cfg())
        miners = {1: miner(1, "hk_champ", scores=[0.3, 0.3], n_tasks=WINDOW),
                  2: miner(2, "hk_chal", scores=[0.6, 0.6], n_tasks=WINDOW - 1)}
        r = run(cc, miners, cs())
        assert len(r.comparisons) == 0

    def test_same_checkpoint_not_repeated(self):
        cc = ChampionChallenge(cfg())
        miners = {1: miner(1, "hk_champ", scores=[0.3, 0.3]),
                  2: miner(2, "hk_chal", scores=[0.6, 0.6])}
        r = run(cc, miners, cs(), prev("hk_chal", cp=1))
        assert len(r.comparisons) == 0

    def test_second_checkpoint(self):
        cc = ChampionChallenge(cfg())
        miners = {1: miner(1, "hk_champ", scores=[0.3, 0.3], n_tasks=2 * WINDOW),
                  2: miner(2, "hk_chal", scores=[0.6, 0.6], n_tasks=2 * WINDOW)}
        r = run(cc, miners, cs(), prev("hk_chal", wins=1, cp=1))
        assert miners[2].challenge_consecutive_wins == 2
        assert miners[2].challenge_checkpoints_passed == 2


# ── Warmup ───────────────────────────────────────────────────────────────────

class TestWarmup:

    def test_warmup_checkpoints_not_counted(self):
        """During warmup, comparisons happen but wins/losses stay 0."""
        cc = ChampionChallenge(cfg(CHAMPION_WARMUP_CHECKPOINTS=2))
        miners = {1: miner(1, "hk_champ", scores=[0.3, 0.3]),
                  2: miner(2, "hk_chal", scores=[0.6, 0.6])}
        r = run(cc, miners, cs())
        # Checkpoint 1 fires but is warmup → no win counted
        assert miners[2].challenge_checkpoints_passed == 1
        assert miners[2].challenge_consecutive_wins == 0
        assert miners[2].challenge_total_losses == 0
        assert len(r.comparisons) == 1  # Comparison still happens (for logging)

    def test_warmup_checkpoint2_still_not_counted(self):
        """Checkpoint 2 is still warmup with warmup=2."""
        cc = ChampionChallenge(cfg(CHAMPION_WARMUP_CHECKPOINTS=2))
        miners = {1: miner(1, "hk_champ", scores=[0.3, 0.3], n_tasks=2 * WINDOW),
                  2: miner(2, "hk_chal", scores=[0.6, 0.6], n_tasks=2 * WINDOW)}
        r = run(cc, miners, cs(), prev("hk_chal", cp=1))
        assert miners[2].challenge_checkpoints_passed == 2
        assert miners[2].challenge_consecutive_wins == 0  # Still warmup

    def test_checkpoint3_counts_after_warmup2(self):
        """Checkpoint 3 is the first counted checkpoint with warmup=2."""
        cc = ChampionChallenge(cfg(CHAMPION_WARMUP_CHECKPOINTS=2))
        miners = {1: miner(1, "hk_champ", scores=[0.3, 0.3], n_tasks=3 * WINDOW),
                  2: miner(2, "hk_chal", scores=[0.6, 0.6], n_tasks=3 * WINDOW)}
        r = run(cc, miners, cs(), prev("hk_chal", cp=2))
        assert miners[2].challenge_checkpoints_passed == 3
        assert miners[2].challenge_consecutive_wins == 1  # First real win

    def test_warmup_loss_not_counted(self):
        """Warmup checkpoint loss doesn't increment total_losses."""
        cc = ChampionChallenge(cfg(CHAMPION_WARMUP_CHECKPOINTS=2,
                                   CHAMPION_TERMINATION_TOTAL_LOSSES=1))
        # Challenger within threshold → would be a loss
        miners = {1: miner(1, "hk_champ", scores=[0.5, 0.5]),
                  2: miner(2, "hk_chal", scores=[0.51, 0.51])}
        r = run(cc, miners, cs())
        # Loss at warmup checkpoint → not counted → not terminated
        assert miners[2].challenge_total_losses == 0
        assert miners[2].challenge_status == "sampling"

    def test_dethrone_requires_n_post_warmup_wins(self):
        """N wins must be post-warmup. Warmup checkpoint wins don't count."""
        N = 2
        cc = ChampionChallenge(cfg(CHAMPION_WARMUP_CHECKPOINTS=2,
                                   CHAMPION_CONSECUTIVE_WINS_REQUIRED=N))
        # At checkpoint 4 (post-warmup=2), prev post-warmup wins=1
        miners = {1: miner(1, "hk_champ", scores=[0.3, 0.3], n_tasks=4 * WINDOW),
                  2: miner(2, "hk_chal", scores=[0.6, 0.6], n_tasks=4 * WINDOW)}
        r = run(cc, miners, cs(), prev("hk_chal", wins=N - 1, cp=3))
        assert r.champion_uid == 2  # N post-warmup wins reached


# ── Champion Change ──────────────────────────────────────────────────────────

class TestChampionChange:

    def test_dethrone_after_n_checkpoints(self):
        N = 3
        cc = ChampionChallenge(cfg(CHAMPION_CONSECUTIVE_WINS_REQUIRED=N))
        miners = {1: miner(1, "hk_champ", scores=[0.3, 0.3], n_tasks=N * WINDOW),
                  2: miner(2, "hk_chal", scores=[0.6, 0.6], n_tasks=N * WINDOW)}
        r = run(cc, miners, cs(), prev("hk_chal", wins=N - 1, cp=N - 1))
        assert r.champion_uid == 2

    def test_loss_resets_win_streak(self):
        cc = ChampionChallenge(cfg(CHAMPION_CONSECUTIVE_WINS_REQUIRED=5,
                                   CHAMPION_TERMINATION_TOTAL_LOSSES=100,
                                   CHAMPION_TERMINATION_CONSECUTIVE_LOSSES=100))
        miners = {1: miner(1, "hk_champ", scores=[0.5, 0.5], n_tasks=5 * WINDOW),
                  2: miner(2, "hk_chal", scores=[0.51, 0.51], n_tasks=5 * WINDOW)}
        r = run(cc, miners, cs(), prev("hk_chal", wins=4, cp=4))
        assert r.champion_uid == 1
        assert miners[2].challenge_consecutive_wins == 0

    def test_multiple_qualified_picks_best_geo_mean(self):
        cc = ChampionChallenge(cfg(CHAMPION_CONSECUTIVE_WINS_REQUIRED=1))
        miners = {1: miner(1, "hk_champ", scores=[0.2, 0.2]),
                  3: miner(3, "hk_3", scores=[0.7, 0.7]),
                  5: miner(5, "hk_5", scores=[0.9, 0.9])}
        r = run(cc, miners, cs())
        assert r.champion_uid == 5


# ── Termination ──────────────────────────────────────────────────────────────

class TestTermination:

    def test_total_losses(self):
        cc = ChampionChallenge(cfg(CHAMPION_TERMINATION_TOTAL_LOSSES=3,
                                   CHAMPION_TERMINATION_CONSECUTIVE_LOSSES=100))
        miners = {1: miner(1, "hk_champ", scores=[0.5, 0.5], n_tasks=3 * WINDOW),
                  2: miner(2, "hk_chal", scores=[0.51, 0.51], n_tasks=3 * WINDOW)}
        r = run(cc, miners, cs(), prev("hk_chal", tl=2, cp=2))
        assert miners[2].challenge_status == "terminated"

    def test_terminated_not_compared(self):
        cc = ChampionChallenge(cfg())
        miners = {1: miner(1, "hk_champ", scores=[0.3, 0.3]),
                  2: miner(2, "hk_chal", scores=[0.6, 0.6])}
        r = run(cc, miners, cs(),
                prev("hk_chal", status="terminated", tl=5, cl=5, cp=5))
        assert len(r.comparisons) == 0


# ── Champion Stability / Cold Start ──────────────────────────────────────────

class TestChampionStability:

    def test_cold_start_picks_best(self):
        cc = ChampionChallenge(cfg())
        miners = {1: miner(1, "hk_a", scores=[0.4, 0.4]),
                  2: miner(2, "hk_b", scores=[0.8, 0.7])}
        r = run(cc, miners)
        assert r.champion_uid == 2

    def test_champion_absent_weight_preserved(self):
        """Champion not present this round → weight stays on champion UID, no replacement."""
        cc = ChampionChallenge(cfg())
        miners = {2: miner(2, "hk_chal", scores=[0.5, 0.5])}
        r = run(cc, miners, cs())
        # Champion (uid 1) absent → no active champion this round
        assert r.champion_uid is None
        # Weight stays on champion's UID via grace mechanism
        assert r.final_weights.get(1) == 1.0
        assert r.final_weights[2] == 0.0

    def test_champion_absent_never_replaced(self):
        """Even after many rounds offline, champion is never replaced automatically."""
        cc = ChampionChallenge(cfg())
        miners = {2: miner(2, "hk_chal", scores=[0.9, 0.9])}
        # Run multiple times — no auto-replacement should happen
        for _ in range(20):
            r = run(cc, miners, cs())
            assert r.champion_uid is None
            assert r.final_weights.get(1) == 1.0


# ── Refactor: PARETO_MARGIN, low completeness participation ──────────────────

class TestRefactorBehavior:

    def test_terminated_cannot_dethrone(self):
        """Bug 1 regression: a terminated miner with N wins cannot dethrone."""
        N = 3
        cc = ChampionChallenge(cfg(CHAMPION_CONSECUTIVE_WINS_REQUIRED=N))
        miners = {
            1: miner(1, "hk_champ", scores=[0.3, 0.3]),
            2: miner(2, "hk_chal", scores=[0.7, 0.7]),
        }
        # Stale state: challenger has N wins but is terminated
        r = run(cc, miners, cs(),
                prev("hk_chal", wins=N, status="terminated", cp=N))
        assert r.champion_uid == 1  # Champion unchanged
        assert miners[2].challenge_status == "terminated"

    def test_low_completeness_miner_participates(self):
        """A miner with low completeness still participates in scoring
        (no validity gate). The checkpoint mechanism handles data sufficiency."""
        cc = ChampionChallenge(cfg())
        # Build challenger with low completeness but sufficient task data
        m_chal = MinerData(uid=2, hotkey="hk_chal", model_revision="r2",
                           model_repo="m", first_block=200)
        tasks = {i: 0.7 for i in range(WINDOW)}
        m_chal.env_scores = {
            env: EnvScore(avg_score=0.7, sample_count=WINDOW,
                          completeness=0.5,  # Low completeness, but data exists
                          all_task_scores=tasks)
            for env in ENVS
        }
        miners = {
            1: miner(1, "hk_champ", scores=[0.3, 0.3]),
            2: m_chal,
        }
        r = run(cc, miners, cs())
        # Challenger reaches checkpoint 1 despite low completeness
        assert miners[2].challenge_checkpoints_passed == 1
        assert len(r.comparisons) == 1

    def test_cold_start_skips_no_data_miners(self):
        """Cold start filters out miners with sample_count=0 in any env."""
        cc = ChampionChallenge(cfg())
        # Miner with no data
        m_empty = MinerData(uid=1, hotkey="hk_empty", model_revision="r",
                            model_repo="m", first_block=50)
        m_empty.env_scores = {env: EnvScore(0.0, 0, 0.0) for env in ENVS}
        # Miner with data
        m_good = miner(2, "hk_good", scores=[0.5, 0.5])
        miners = {1: m_empty, 2: m_good}
        r = run(cc, miners)  # cold start
        assert r.champion_uid == 2

    def test_pareto_margin_strict(self):
        """Challenger must beat by strictly more than PARETO_MARGIN."""
        cc = ChampionChallenge(cfg())
        # Champion 0.5, challenger 0.521 (margin=0.02 → threshold=0.52, 0.521 > 0.52)
        miners = {
            1: miner(1, "hk_champ", scores=[0.5, 0.5]),
            2: miner(2, "hk_chal", scores=[0.521, 0.521]),
        }
        r = run(cc, miners, cs())
        assert miners[2].challenge_consecutive_wins == 1

    def test_pareto_margin_below_loses(self):
        """Challenger below margin loses."""
        cc = ChampionChallenge(cfg())
        # Challenger 0.519 < 0.52 threshold
        miners = {
            1: miner(1, "hk_champ", scores=[0.5, 0.5]),
            2: miner(2, "hk_chal", scores=[0.519, 0.519]),
        }
        r = run(cc, miners, cs())
        assert miners[2].challenge_total_losses == 1
        assert miners[2].challenge_consecutive_wins == 0


# ── Pairwise Filter ──────────────────────────────────────────────────────────

class TestPairwiseFilter:

    def test_older_terminates_copy(self):
        """A copy with similar scores is terminated by the older original."""
        cc = ChampionChallenge(cfg(PAIRWISE_MIN_WINDOWS=3))
        n = 3 * WINDOW
        miners = {
            1: miner(1, "hk_champ", first_block=50, scores=[0.5, 0.5], n_tasks=n),
            2: miner(2, "hk_orig", first_block=100, scores=[0.7, 0.7], n_tasks=n),
            3: miner(3, "hk_copy", first_block=200, scores=[0.71, 0.71], n_tasks=n),
        }
        run(cc, miners, cs())
        assert miners[3].challenge_status == "terminated"
        assert miners[2].challenge_status == "sampling"

    def test_genuine_improvement_terminates_older(self):
        """Significantly better newer miner terminates the older one."""
        cc = ChampionChallenge(cfg(PAIRWISE_MIN_WINDOWS=3))
        n = 3 * WINDOW
        miners = {
            1: miner(1, "hk_champ", first_block=50, scores=[0.4, 0.4], n_tasks=n),
            2: miner(2, "hk_old", first_block=100, scores=[0.5, 0.5], n_tasks=n),
            3: miner(3, "hk_new", first_block=200, scores=[0.9, 0.9], n_tasks=n),
        }
        run(cc, miners, cs())
        assert miners[2].challenge_status == "terminated"
        assert miners[3].challenge_status == "sampling"

    def test_below_threshold_no_filter(self):
        """Pair with insufficient common tasks is not compared."""
        cc = ChampionChallenge(cfg(PAIRWISE_MIN_WINDOWS=3))
        n = 2 * WINDOW  # Below 3-window threshold
        miners = {
            1: miner(1, "hk_champ", first_block=50, scores=[0.5, 0.5], n_tasks=n),
            2: miner(2, "hk_orig", first_block=100, scores=[0.7, 0.7], n_tasks=n),
            3: miner(3, "hk_copy", first_block=200, scores=[0.71, 0.71], n_tasks=n),
        }
        run(cc, miners, cs())
        assert miners[2].challenge_status == "sampling"
        assert miners[3].challenge_status == "sampling"

    def test_champion_excluded_from_filter(self):
        """Champion is never terminated by pairwise filter."""
        cc = ChampionChallenge(cfg(PAIRWISE_MIN_WINDOWS=3))
        n = 3 * WINDOW
        miners = {
            1: miner(1, "hk_champ", first_block=50, scores=[0.3, 0.3], n_tasks=n),
            2: miner(2, "hk_chal", first_block=100, scores=[0.9, 0.9], n_tasks=n),
        }
        run(cc, miners, cs())
        # Champion not terminated even though uid 2 dominates them
        assert miners[1].challenge_status == "sampling"

    def test_terminated_peer_does_not_filter(self):
        """A terminated miner cannot be the basis for terminating others."""
        cc = ChampionChallenge(cfg(PAIRWISE_MIN_WINDOWS=3))
        n = 3 * WINDOW
        miners = {
            1: miner(1, "hk_champ", first_block=50, scores=[0.5, 0.5], n_tasks=n),
            2: miner(2, "hk_term", first_block=100, scores=[0.7, 0.7], n_tasks=n),
            3: miner(3, "hk_copy", first_block=200, scores=[0.71, 0.71], n_tasks=n),
        }
        # uid 2 already terminated → should not filter uid 3
        run(cc, miners, cs(),
            prev("hk_term", status="terminated", tl=5, cl=5, cp=5))
        assert miners[2].challenge_status == "terminated"
        assert miners[3].challenge_status == "sampling"


# ── Weights ──────────────────────────────────────────────────────────────────

class TestWeights:

    def test_champion_full_weight(self):
        cc = ChampionChallenge(cfg())
        miners = {1: miner(1, "hk_champ", scores=[0.5, 0.5]),
                  2: miner(2, "hk_chal", scores=[0.3, 0.3])}
        r = run(cc, miners, cs())
        assert r.final_weights[1] == 1.0
        assert r.final_weights[2] == 0.0

    def test_champion_with_partial_env_data_still_active(self):
        """Champion missing some env data is still active (no validity gate).
        Comparisons against them naturally fail to advance because min_common=0
        for the missing env. Champion keeps weight."""
        cc = ChampionChallenge(cfg())
        m1 = MinerData(uid=1, hotkey="hk_champ", model_revision="rev1",
                       model_repo="m", first_block=50)
        m1.env_scores = {"env_a": _env(0.5, WINDOW)}  # Missing env_b
        miners = {
            1: m1,
            2: miner(2, "hk_chal", scores=[0.7, 0.7]),
        }
        r = run(cc, miners, cs())
        assert r.champion_uid == 1
        assert r.final_weights[1] == 1.0
        assert r.final_weights[2] == 0.0
        # Challenger can't advance checkpoint (env_b common = 0)
        assert miners[2].challenge_checkpoints_passed == 0


# ── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_uid_zero_as_champion(self):
        """UID 0 should be a valid champion (avoids `if champion_uid` truthiness bug)."""
        cc = ChampionChallenge(cfg())
        miners = {0: miner(0, "hk_zero", scores=[0.8, 0.8]),
                  1: miner(1, "hk_other", scores=[0.5, 0.5])}
        r = run(cc, miners)  # cold start
        assert r.champion_uid == 0
        assert r.final_weights[0] == 1.0
        assert r.final_weights[1] == 0.0

    def test_empty_environments(self):
        cc = ChampionChallenge(cfg())
        miners = {1: miner(1, "hk", scores=[0.5, 0.5])}
        r = cc.run(miners, environments=[], env_sampling_counts={},
                   champion_state=None, prev_challenge_states={})
        assert r.champion_uid is None
        assert r.final_weights[1] == 0.0

    def test_empty_miners(self):
        cc = ChampionChallenge(cfg())
        r = cc.run({}, ENVS, ENV_SC, None, {})
        assert r.champion_uid is None
        assert r.final_weights == {}

    def test_window_size_zero(self):
        """No sampling_count configured → no challenges, but champion still resolves."""
        cc = ChampionChallenge(cfg())
        miners = {1: miner(1, "hk_champ", scores=[0.5, 0.5]),
                  2: miner(2, "hk_chal", scores=[0.8, 0.8])}
        r = cc.run(miners, ENVS, env_sampling_counts={}, champion_state=cs(),
                   prev_challenge_states={})
        assert r.champion_uid == 1
        assert len(r.comparisons) == 0
        # No challenge happened
        assert miners[2].challenge_checkpoints_passed == 0

    def test_dethrone_does_not_revive_terminated(self):
        """Termination is permanent. Dethrone resets active miners but
        terminated miners stay terminated forever."""
        cc = ChampionChallenge(cfg(CHAMPION_CONSECUTIVE_WINS_REQUIRED=1))
        miners = {1: miner(1, "hk_champ", scores=[0.2, 0.2]),
                  2: miner(2, "hk_strong", scores=[0.9, 0.9]),
                  3: miner(3, "hk_dead", scores=[0.3, 0.3])}
        r = run(cc, miners, cs(),
                prev("hk_dead", status="terminated", tl=5, cl=5, cp=5))
        # uid 2 dethrones uid 1; uid 3 stays terminated
        assert r.champion_uid == 2
        assert miners[3].challenge_status == "terminated"

    def test_dethrone_resets_then_termination_skipped(self):
        """After dethrone, all states reset, so terminations don't fire on same round."""
        cc = ChampionChallenge(cfg(CHAMPION_CONSECUTIVE_WINS_REQUIRED=1,
                                   CHAMPION_TERMINATION_TOTAL_LOSSES=1))
        miners = {1: miner(1, "hk_champ", scores=[0.2, 0.2]),
                  2: miner(2, "hk_strong", scores=[0.9, 0.9]),
                  3: miner(3, "hk_weak", scores=[0.3, 0.3])}
        r = run(cc, miners, cs(),
                prev("hk_weak", tl=0))  # weak about to lose
        # uid 2 dethrones uid 1; reset → weak shouldn't be terminated
        assert r.champion_uid == 2
        assert miners[3].challenge_status == "sampling"

    def test_terminated_miner_excluded_from_pairwise_basis(self):
        """A terminated miner cannot terminate others via pairwise."""
        cc = ChampionChallenge(cfg(PAIRWISE_MIN_WINDOWS=3))
        n = 3 * WINDOW
        miners = {
            1: miner(1, "hk_champ", first_block=50, scores=[0.5, 0.5], n_tasks=n),
            2: miner(2, "hk_dead", first_block=100, scores=[0.7, 0.7], n_tasks=n),
            3: miner(3, "hk_copy", first_block=200, scores=[0.71, 0.71], n_tasks=n),
        }
        # uid 2 already terminated → uid 3 should not be filtered by uid 2
        run(cc, miners, cs(),
            prev("hk_dead", status="terminated", tl=5, cl=5, cp=5))
        assert miners[2].challenge_status == "terminated"
        assert miners[3].challenge_status == "sampling"

