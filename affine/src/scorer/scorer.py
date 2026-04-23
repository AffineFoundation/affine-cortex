"""
Main Scorer Orchestrator

Coordinates the champion challenge scoring algorithm and manages result persistence.
"""

import time
from typing import Dict, Any, Optional
from .config import ScorerConfig
from .models import ScoringResult
from .stage1_collector import Stage1Collector
from .champion_challenge import ChampionChallenge

from affine.core.setup import logger


# First-challenge slots bonus — +10 slots the first time a miner's CP
# crosses CHAMPION_WARMUP_CHECKPOINTS+1 within a reign. Capped at 30;
# miners already at/above the cap get no DB write (no-op, not retried
# later because CP monotonicity blocks re-triggering within the reign).
FIRST_CHALLENGE_SLOTS_BONUS = 10
FIRST_CHALLENGE_SLOTS_CAP = 30
FIRST_CHALLENGE_DEFAULT_SLOTS = 15  # Matches MinerStatsDAO's init default


class Scorer:
    """Main scorer orchestrator.

    Coordinates the two-stage champion challenge scoring algorithm:
    1. Data Collection: Collect and validate sample data
    2. Champion Challenge: Compare all miners against champion via Pareto dominance

    The champion gets 100% weight. All other miners get 0%.
    """

    def __init__(self, config: ScorerConfig = ScorerConfig):
        self.config = config
        self.stage1 = Stage1Collector(config)
        self.champion_challenge = ChampionChallenge(config)

    def calculate_scores(
        self,
        scoring_data: Dict[str, Any],
        environments: list,
        block_number: int,
        env_sampling_counts: Optional[Dict[str, int]] = None,
        champion_state: Optional[Dict[str, Any]] = None,
        prev_challenge_states: Optional[Dict[str, Dict[str, Any]]] = None,
        anticopy_records: Optional[Dict[str, Dict[str, Any]]] = None,
        print_summary: bool = True,
    ) -> ScoringResult:
        """Run one scoring round.

        Args:
            scoring_data: API /samples/scoring response
            environments: enabled scoring environments
            block_number: current Bittensor block
            env_sampling_counts: {env_name: window_size} per environment
            champion_state: persisted champion info, or None for cold start
            prev_challenge_states: persisted per-miner challenge state (keyed by hotkey)
            print_summary: print the per-miner summary table
        """
        start_time = time.time()
        logger.info(f"Total Miners: {len(scoring_data)}")

        # Stage 1: Data Collection
        stage1_output = self.stage1.collect(
            scoring_data, environments, env_sampling_counts or {})

        # Stage 2: Champion Challenge
        challenge_output = self.champion_challenge.run(
            miners=stage1_output.miners,
            environments=environments,
            env_sampling_counts=env_sampling_counts or {},
            champion_state=champion_state,
            prev_challenge_states=prev_challenge_states or {},
            anticopy_records=anticopy_records or {},
        )

        result = ScoringResult(
            block_number=block_number,
            calculated_at=int(time.time()),
            environments=environments,
            config=self.config.to_dict(),
            miners=challenge_output.miners,
            final_weights=challenge_output.final_weights,
            champion_uid=challenge_output.champion_uid,
            champion_hotkey=challenge_output.champion_hotkey,
            total_miners=len(scoring_data),
        )

        elapsed_time = time.time() - start_time
        logger.info(f"SCORING COMPLETED - Time: {elapsed_time:.2f}s, Champion: UID {result.champion_uid}")

        # Print detailed summary
        if print_summary:
            self._print_summary(challenge_output.miners, environments, challenge_output.champion_uid)

        return result

    def _print_summary(self, miners: Dict[int, Any], environments: list, champion_uid: Optional[int]):
        """Print a summary table of all miners and their challenge status."""
        logger.info("=" * 100)
        logger.info(f"{'UID':>4} | {'Hotkey':12} | {'Status':11} | {'CP':>3} | {'Wins':>4} | {'TotL':>4} | {'Weight':>7} | Env Scores")
        logger.info("-" * 100)

        for uid in sorted(miners.keys()):
            miner = miners[uid]
            hotkey_short = f"{miner.hotkey[:8]}..."
            if uid == champion_uid:
                status = "CHAMPION"
            elif miner.challenge_status == 'terminated':
                status = "TERMINATED"
            else:
                status = "sampling"

            env_strs = []
            for env in sorted(environments):
                if env in miner.env_scores and miner.env_scores[env].sample_count > 0:
                    env_strs.append(f"{miner.env_scores[env].avg_score:.3f}")
                else:
                    env_strs.append("  N/A")

            logger.info(
                f"{uid:4d} | {hotkey_short:12} | {status:11} | "
                f"{miner.challenge_checkpoints_passed:3d} | "
                f"{miner.challenge_consecutive_wins:4d} | "
                f"{miner.challenge_total_losses:4d} | "
                f"{miner.normalized_weight:7.4f} | "
                f"{' '.join(env_strs)}"
            )

        logger.info("=" * 100)

    async def save_results(
        self,
        result: ScoringResult,
        score_snapshots_dao=None,
        scores_dao=None,
        miner_stats_dao=None,
        system_config_dao=None,
        block_number: Optional[int] = None,
    ):
        """Save scoring results to database.

        Args:
            result: ScoringResult to save
            score_snapshots_dao: ScoreSnapshotsDAO instance (optional)
            scores_dao: ScoresDAO instance (optional)
            miner_stats_dao: MinerStatsDAO instance (optional)
            system_config_dao: SystemConfigDAO instance for champion state (optional)
            block_number: Block number override (defaults to result.block_number)
        """
        if not score_snapshots_dao or not scores_dao:
            logger.warning("DAO instances not provided, skipping database save")
            return

        block = block_number or result.block_number
        logger.info(f"Saving scoring results to database (block {block})")

        # Save snapshot metadata
        statistics = {
            "total_miners": result.total_miners,
            "champion_uid": result.champion_uid,
            "champion_hotkey": result.champion_hotkey,
            "miner_final_scores": {
                str(uid): weight
                for uid, weight in result.final_weights.items()
            }
        }

        await score_snapshots_dao.save_snapshot(
            block_number=block,
            scorer_hotkey="scorer_service",
            config=result.config,
            statistics=statistics
        )

        # Save per-miner scores
        for uid, miner in result.miners.items():
            total_samples = sum(
                env_score.sample_count
                for env_score in miner.env_scores.values()
            )

            scores_by_env = {
                env: {
                    "score": score.avg_score,
                    "sample_count": score.sample_count,
                    "historical_count": score.historical_count,
                    "completeness": score.completeness,
                    # Real per-challenger vs-champion numbers, populated by
                    # champion_challenge. Absent when miner is champion or
                    # shares no tasks with champion in this env.
                    **miner.vs_champion_per_env.get(env, {}),
                }
                for env, score in miner.env_scores.items()
            }

            overall_score = miner.normalized_weight
            # Average only over envs that have actual data
            data_scores = [
                d["score"] for d in scores_by_env.values() if d["sample_count"] > 0
            ]
            average_score = sum(data_scores) / len(data_scores) if data_scores else 0.0

            challenge_info = {
                "status": miner.challenge_status,
                "consecutive_wins": miner.challenge_consecutive_wins,
                "total_losses": miner.challenge_total_losses,
                "consecutive_losses": miner.challenge_consecutive_losses,
                "checkpoints_passed": miner.challenge_checkpoints_passed,
                "is_champion": miner.is_champion,
                "termination_reason": miner.termination_reason,
            }

            await scores_dao.save_score(
                block_number=block,
                miner_hotkey=miner.hotkey,
                uid=uid,
                model_revision=miner.model_revision,
                model=miner.model_repo,
                first_block=miner.first_block,
                overall_score=overall_score,
                average_score=average_score,
                scores_by_env=scores_by_env,
                total_samples=total_samples,
                challenge_info=challenge_info,
            )

        # Save challenge state for each miner
        if miner_stats_dao:
            for uid, miner in result.miners.items():
                await miner_stats_dao.update_challenge_state(
                    hotkey=miner.hotkey,
                    revision=miner.model_revision,
                    consecutive_wins=miner.challenge_consecutive_wins,
                    total_losses=miner.challenge_total_losses,
                    consecutive_losses=miner.challenge_consecutive_losses,
                    checkpoints_passed=miner.challenge_checkpoints_passed,
                    status=miner.challenge_status,
                    termination_reason=miner.termination_reason,
                )

            # First-challenge slots bonus: one-time boost when a miner's
            # CP first crosses warmup this reign. Flag was set in
            # _run_challenges at the exact transition; we skip champion
            # (promoted same round) and terminated miners as a safety
            # belt. Miners already at or above the cap need no DB write.
            now = int(time.time())
            for uid, miner in result.miners.items():
                if not miner.should_grant_first_challenge_bonus:
                    continue
                if miner.is_champion or miner.challenge_status == 'terminated':
                    continue

                current_slots = await miner_stats_dao.get_miner_slots(
                    miner.hotkey, miner.model_revision
                )
                if current_slots is None:
                    current_slots = FIRST_CHALLENGE_DEFAULT_SLOTS
                if current_slots >= FIRST_CHALLENGE_SLOTS_CAP:
                    continue  # Already at/above cap — leave alone

                new_slots = min(
                    current_slots + FIRST_CHALLENGE_SLOTS_BONUS,
                    FIRST_CHALLENGE_SLOTS_CAP,
                )
                ok = await miner_stats_dao.update_sampling_slots(
                    hotkey=miner.hotkey,
                    revision=miner.model_revision,
                    slots=new_slots,
                    adjusted_at=now,
                )
                if ok:
                    logger.info(
                        f"First-challenge bonus: UID {uid} ({miner.hotkey[:8]}...) "
                        f"slots {current_slots} -> {new_slots} "
                        f"(CP={miner.challenge_checkpoints_passed})"
                    )

        # Save champion state to system_config (only when champion is present this round)
        # Champion offline does NOT update DB — original record stays intact
        if system_config_dao and result.champion_uid is not None:
            champion_miner = result.miners.get(result.champion_uid)
            if champion_miner:
                existing = await system_config_dao.get_param_value('champion')
                # Preserve original since_block if champion identity unchanged
                if (existing
                        and existing.get('hotkey') == champion_miner.hotkey
                        and existing.get('revision') == champion_miner.model_revision):
                    since_block = existing.get('since_block', block)
                else:
                    since_block = block

                await system_config_dao.set_param(
                    param_name='champion',
                    param_value={
                        'hotkey': champion_miner.hotkey,
                        'revision': champion_miner.model_revision,
                        'uid': result.champion_uid,
                        'since_block': since_block,
                    },
                    param_type='dict',
                    updated_by='scorer_service',
                )

        logger.info(f"Successfully saved scoring results for {len(result.miners)} miners")


def create_scorer(config: Optional[ScorerConfig] = None) -> Scorer:
    """Factory function to create a Scorer instance."""
    if config is None:
        config = ScorerConfig()
    return Scorer(config)
