"""
Champion Challenge Scoring — Winner-takes-all.

Rules:
- Only the champion gets 100% weight
- Pairwise Pareto filter terminates copies (older miner has incumbent advantage)
- Challengers must dominate champion at N consecutive checkpoints to take the crown
- Each checkpoint requires K × window_size new common tasks
- First WARMUP checkpoints don't count (data too sparse)
- Champion is never replaced by being temporarily absent — weight preserved
"""

from typing import Dict, List, Optional, Tuple

from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.models import MinerData, ParetoComparison, ChampionChallengeOutput
from affine.src.scorer.stage2_pareto import Stage2ParetoFilter
from affine.src.scorer.utils import geometric_mean
from affine.core.setup import logger


class ChampionChallenge:

    def __init__(self, config: ScorerConfig = ScorerConfig):
        self.config = config
        self.pareto = Stage2ParetoFilter(config)

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        env_sampling_counts: Dict[str, int],
        champion_state: Optional[Dict],
        prev_challenge_states: Dict[str, Dict],
    ) -> ChampionChallengeOutput:
        if not environments or not miners:
            return self._empty_output(miners)

        self._load_states(miners, prev_challenge_states)

        # Resolve champion: who is active, who holds weight
        champion_uid, champion_miner, weight_uid, champion_changed = \
            self._resolve_champion(miners, environments, champion_state)

        window_size = self._window_size(environments, env_sampling_counts)

        # Pairwise anti-plagiarism filter
        self._pairwise_filter(miners, environments, window_size, champion_uid)

        # Champion challenge
        comparisons = self._run_challenges(
            miners, environments, window_size, champion_uid, champion_miner)

        # Dethrone check
        new_uid, new_miner = self._check_dethrone(miners, environments, champion_uid)
        if new_uid is not None:
            if champion_miner:
                champion_miner.is_champion = False
            new_miner.is_champion = True
            self._reset_all_states(miners)
            logger.info(f"DETHRONED: UID {new_uid} ({new_miner.hotkey[:8]}...) replaces UID {champion_uid}")
            champion_uid = new_uid
            champion_miner = new_miner
            weight_uid = new_uid
            champion_changed = True

        # Termination check (after dethrone, since dethrone resets states)
        self._check_terminations(miners, champion_uid)

        final_weights = self._assign_weights(miners, weight_uid)
        self._log_summary(miners, champion_uid, champion_changed)

        return ChampionChallengeOutput(
            miners=miners,
            comparisons=comparisons,
            champion_uid=champion_uid,
            champion_hotkey=champion_miner.hotkey if champion_miner else None,
            champion_changed=champion_changed,
            final_weights=final_weights,
        )

    # ── Phase 1: Load previous state ─────────────────────────────────────────

    def _load_states(self, miners: Dict[int, MinerData], prev: Dict[str, Dict]):
        """Load each miner's persisted challenge state. Identity is hotkey
        (revision is fixed per hotkey by upstream constraint, so no
        revision-change recovery path exists)."""
        for miner in miners.values():
            p = prev.get(miner.hotkey, {})
            miner.challenge_consecutive_wins = p.get('challenge_consecutive_wins', 0)
            miner.challenge_total_losses = p.get('challenge_total_losses', 0)
            miner.challenge_consecutive_losses = p.get('challenge_consecutive_losses', 0)
            miner.challenge_checkpoints_passed = p.get('challenge_checkpoints_passed', 0)
            miner.challenge_status = p.get('challenge_status', 'sampling')

    # ── Phase 2: Resolve champion ────────────────────────────────────────────

    def _resolve_champion(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        champion_state: Optional[Dict],
    ) -> Tuple[Optional[int], Optional[MinerData], Optional[int], bool]:
        """Returns (active_uid, active_miner, weight_uid, changed).

        - active_uid/miner: in-round champion that participates in challenges
          (None if champion is absent or invalid this round)
        - weight_uid: UID receiving 1.0 weight — always set when a champion
          identity exists, even if not active this round
        - changed: True only on cold start
        """
        if not champion_state:
            # Cold start: pick by geometric mean
            uid, miner = self._best_by_geo_mean(miners, environments)
            if miner:
                miner.is_champion = True
                self._reset_all_states(miners)
                logger.info(f"New champion (cold start): UID {uid} ({miner.hotkey[:8]}...)")
            return uid, miner, uid, miner is not None

        hk = champion_state.get('hotkey')
        rev = champion_state.get('revision')

        # Look for champion by identity (hotkey + revision)
        for uid, miner in miners.items():
            if miner.hotkey == hk and miner.model_revision == rev:
                if self._has_all_valid_envs(miner, environments):
                    miner.is_champion = True
                    return uid, miner, uid, False
                # Present but data incomplete → weight stays on this UID
                logger.warning(f"Champion {hk[:8]}... incomplete data, weight preserved on UID {uid}")
                return None, None, uid, False

        # Champion completely absent → weight stays on stored UID
        stored_uid = champion_state.get('uid')
        if hk:
            logger.warning(f"Champion {hk[:8]}... not present, weight preserved on UID {stored_uid}")
        return None, None, stored_uid, False

    # ── Phase 3a: Pairwise Pareto filter (anti-plagiarism) ───────────────────

    def _pairwise_filter(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        window_size: int,
        champion_uid: Optional[int],
    ):
        """Compare all non-champion miner pairs once they share PAIRWISE_MIN_WINDOWS
        × window_size common tasks. Older miner is the incumbent. Dominated miner
        is terminated. Filters plagiarized models that can't beat the original's
        threshold.
        """
        if window_size <= 0:
            return

        threshold = self.config.PAIRWISE_MIN_WINDOWS * window_size
        eligible = sorted(
            (
                (uid, m) for uid, m in miners.items()
                if uid != champion_uid
                and m.challenge_status != 'terminated'
                and self._has_all_valid_envs(m, environments)
            ),
            key=lambda x: (x[1].first_block, x[0]),
        )

        for i, (uid_a, miner_a) in enumerate(eligible):
            if miner_a.challenge_status == 'terminated':
                continue
            for uid_b, miner_b in eligible[i + 1:]:
                if miner_b.challenge_status == 'terminated':
                    continue
                if self._min_common_tasks(miner_a, miner_b, environments) < threshold:
                    continue

                cmp = self.pareto._compare_miners(miner_a, miner_b, environments, "pairwise")
                if cmp.a_dominates_b:
                    miner_b.challenge_status = 'terminated'
                    logger.info(f"PAIRWISE: UID {uid_b} terminated, dominated by older UID {uid_a}")
                elif cmp.b_dominates_a:
                    miner_a.challenge_status = 'terminated'
                    logger.info(f"PAIRWISE: UID {uid_a} terminated, dominated by newer UID {uid_b}")
                    break

    # ── Phase 3b: Run challenges (checkpoint-gated) ──────────────────────────

    def _run_challenges(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        window_size: int,
        champion_uid: Optional[int],
        champion_miner: Optional[MinerData],
    ) -> List[ParetoComparison]:
        if window_size <= 0:
            logger.warning("window_size=0 (missing sampling_config?), no challenges")
            return []
        if not champion_miner:
            return []

        warmup = self.config.CHAMPION_WARMUP_CHECKPOINTS
        N = self.config.CHAMPION_CONSECUTIVE_WINS_REQUIRED
        M = self.config.CHAMPION_TERMINATION_TOTAL_LOSSES
        comparisons = []

        for uid, miner in miners.items():
            if uid == champion_uid or miner.challenge_status == 'terminated':
                continue
            if not self._has_all_valid_envs(miner, environments):
                continue

            # Checkpoint gate: Kth comparison requires K × window_size common tasks
            min_common = self._min_common_tasks(champion_miner, miner, environments)
            if min_common < (miner.challenge_checkpoints_passed + 1) * window_size:
                continue

            miner.challenge_checkpoints_passed += 1
            cp = miner.challenge_checkpoints_passed

            cmp = self.pareto._compare_miners(
                champion_miner, miner, environments, "champion_challenge")
            comparisons.append(cmp)

            if cp <= warmup:
                result = "dominates" if cmp.b_dominates_a else "fails"
                logger.info(f"UID {uid} {result} at warmup checkpoint {cp}/{warmup}")
                continue

            if cmp.b_dominates_a:
                miner.challenge_consecutive_wins += 1
                miner.challenge_consecutive_losses = 0
                logger.info(f"UID {uid} dominates at CP {cp} "
                            f"(wins: {miner.challenge_consecutive_wins}/{N})")
            else:
                miner.challenge_total_losses += 1
                miner.challenge_consecutive_losses += 1
                miner.challenge_consecutive_wins = 0
                logger.info(f"UID {uid} fails at CP {cp} "
                            f"(losses: {miner.challenge_total_losses}/{M})")

        return comparisons

    # ── Phase 4: Dethrone ────────────────────────────────────────────────────

    def _check_dethrone(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        champion_uid: Optional[int],
    ) -> Tuple[Optional[int], Optional[MinerData]]:
        """Find a challenger qualified to take the crown. Returns (None, None)
        if no one qualifies. If multiple qualify, picks the one with highest
        geometric mean (tiebreaker: earlier first_block)."""
        N = self.config.CHAMPION_CONSECUTIVE_WINS_REQUIRED
        qualified = [
            (uid, self._geo_mean(miners[uid], environments), miners[uid].first_block)
            for uid, m in miners.items()
            if uid != champion_uid and m.challenge_consecutive_wins >= N
        ]
        if not qualified:
            return None, None
        qualified.sort(key=lambda x: (-x[1], x[2]))
        new_uid = qualified[0][0]
        return new_uid, miners[new_uid]

    # ── Phase 5: Terminate ───────────────────────────────────────────────────

    def _check_terminations(self, miners: Dict[int, MinerData], champion_uid: Optional[int]):
        M = self.config.CHAMPION_TERMINATION_TOTAL_LOSSES
        M_con = self.config.CHAMPION_TERMINATION_CONSECUTIVE_LOSSES
        for uid, miner in miners.items():
            if uid == champion_uid or miner.challenge_status == 'terminated':
                continue
            if (miner.challenge_total_losses >= M
                    or miner.challenge_consecutive_losses >= M_con):
                miner.challenge_status = 'terminated'

    # ── Phase 6: Weights ─────────────────────────────────────────────────────

    def _assign_weights(
        self, miners: Dict[int, MinerData], weight_uid: Optional[int]
    ) -> Dict[int, float]:
        weights = {}
        for uid, miner in miners.items():
            w = 1.0 if uid == weight_uid else 0.0
            miner.normalized_weight = w
            weights[uid] = w
        # Champion's UID may be absent from miners dict — still record their weight
        if weight_uid is not None and weight_uid not in weights:
            weights[weight_uid] = 1.0
        return weights

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _has_all_valid_envs(self, miner: MinerData, environments: List[str]) -> bool:
        return all(env in miner.env_scores and miner.env_scores[env].is_valid
                   for env in environments)

    def _geo_mean(self, miner: MinerData, environments: List[str]) -> float:
        scores = [miner.env_scores[env].avg_score for env in environments]
        return geometric_mean(scores, epsilon=self.config.GEOMETRIC_MEAN_EPSILON)

    def _best_by_geo_mean(
        self, miners: Dict[int, MinerData], environments: List[str]
    ) -> Tuple[Optional[int], Optional[MinerData]]:
        best_uid, best_miner, best_score = None, None, -1.0
        for uid, miner in miners.items():
            if not self._has_all_valid_envs(miner, environments):
                continue
            score = self._geo_mean(miner, environments)
            if (score > best_score
                    or (score == best_score and best_miner
                        and miner.first_block < best_miner.first_block)):
                best_uid, best_miner, best_score = uid, miner, score
        return best_uid, best_miner

    def _min_common_tasks(
        self, a: MinerData, b: MinerData, environments: List[str]
    ) -> int:
        return min(
            len(set(a.env_scores[env].all_task_scores) & set(b.env_scores[env].all_task_scores))
            for env in environments
        )

    def _window_size(
        self, environments: List[str], env_sampling_counts: Dict[str, int]
    ) -> int:
        counts = [env_sampling_counts.get(env, 0) for env in environments]
        return min(counts) if counts else 0

    def _reset_state(self, miner: MinerData):
        miner.challenge_consecutive_wins = 0
        miner.challenge_total_losses = 0
        miner.challenge_consecutive_losses = 0
        miner.challenge_checkpoints_passed = 0
        miner.challenge_status = 'sampling'

    def _reset_all_states(self, miners: Dict[int, MinerData]):
        """Reset counters for non-terminated miners on champion change.
        Termination is permanent — terminated miners are never revived."""
        for miner in miners.values():
            if miner.challenge_status == 'terminated':
                continue
            self._reset_state(miner)

    def _empty_output(self, miners: Dict[int, MinerData]) -> ChampionChallengeOutput:
        return ChampionChallengeOutput(
            miners=miners,
            comparisons=[],
            champion_uid=None,
            champion_hotkey=None,
            champion_changed=False,
            final_weights={uid: 0.0 for uid in miners},
        )

    def _log_summary(
        self, miners: Dict[int, MinerData], champion_uid: Optional[int], changed: bool
    ):
        if champion_uid is not None and champion_uid in miners:
            hk = miners[champion_uid].hotkey[:8]
        else:
            hk = "absent"
        terminated = sum(1 for m in miners.values() if m.challenge_status == 'terminated')
        logger.info(f"Champion: UID {champion_uid} ({hk}...) | "
                     f"Changed: {changed} | Terminated: {terminated}")
