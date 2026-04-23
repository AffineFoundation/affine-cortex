"""
Champion Challenge Scoring — Winner-takes-all.

Pipeline (per scoring round):
  1. Load persisted challenge state for each miner
  2. Resolve champion (cold start by geo mean / locate by hotkey+revision)
  3. Pairwise Pareto filter (Pareto dominance): older incumbent terminates copies
  4. Run challenges: each non-champion miner vs champion, gated by checkpoints
  5. Dethrone check: challenger reached min CP and dominates champion?
  6. Termination check: M total or M-1 consecutive losses → terminated
  7. Assign weights: champion = 1.0, others = 0.0

Dethrone requires:
  - checkpoint_passed >= CHAMPION_DETHRONE_MIN_CHECKPOINT (data depth)
  - The most recent comparison is a win (dominates champion)
  Consecutive wins are NOT required. Each comparison uses the full common
  task history, so a single comparison at high CP is statistically decisive.
  The termination mechanism (M total / M-1 consecutive losses) handles
  early pruning of hopeless challengers.

Invariants:
- The champion is never replaced for being temporarily absent.
- Termination is permanent.
- Margin scales linearly from WIN_MARGIN_START to WIN_MARGIN_END as CP grows.
- Champion must be pre-set via `af db set-champion` before first run.
"""

from typing import Any, Dict, List, Optional, Tuple

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
        anticopy_records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ChampionChallengeOutput:
        if not environments or not miners:
            return self._empty_output(miners)

        self._load_anticopy_records(miners, anticopy_records or {})
        self._load_states(miners, prev_challenge_states)

        champion_uid, champion_miner, weight_uid, champion_changed = \
            self._resolve_champion(miners, environments, champion_state)

        # Log config validity; CP & pairwise thresholds now use per-env counts
        # so heterogeneous sampling_counts are handled correctly.
        self._assert_env_counts(environments, env_sampling_counts)

        # Pareto dominance: terminate dominated non-champion miners
        self._pairwise_filter(
            miners, environments, env_sampling_counts, champion_uid, champion_miner)

        # Champion challenge (checkpoint-gated)
        comparisons = self._run_challenges(
            miners, environments, env_sampling_counts, champion_uid, champion_miner)

        # Dethrone
        new_uid, new_miner = self._check_dethrone(miners, environments, champion_uid)
        if new_uid is not None:
            if champion_miner:
                champion_miner.is_champion = False
            new_miner.is_champion = True
            self._reset_all_states(miners)
            logger.info(f"DETHRONED: UID {new_uid} ({new_miner.hotkey[:8]}...) "
                         f"replaces UID {champion_uid}")
            champion_uid = new_uid
            champion_miner = new_miner
            weight_uid = new_uid
            champion_changed = True

        # Termination check (after dethrone — its reset wipes counters first)
        self._check_terminations(miners, champion_uid)

        # Per-challenger vs-champion snapshot for UI/diagnostics. Populates
        # miner.vs_champion_per_env for every non-champion miner with the
        # real common-task data that Pareto would use. Not CP-gated — every
        # round, so rank display always has a current threshold.
        self._populate_vs_champion_per_env(miners, environments, champion_miner)

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

    # ── Phase 1: Load state ──────────────────────────────────────────────────

    def _load_states(self, miners: Dict[int, MinerData], prev: Dict[str, Dict]):
        """Load persisted challenge state for each miner. Identity is hotkey
        (revision is fixed per hotkey by upstream constraint)."""
        for miner in miners.values():
            p = prev.get(miner.hotkey, {})
            miner.challenge_consecutive_wins = p.get('challenge_consecutive_wins', 0)
            miner.challenge_total_losses = p.get('challenge_total_losses', 0)
            miner.challenge_consecutive_losses = p.get('challenge_consecutive_losses', 0)
            miner.challenge_checkpoints_passed = p.get('challenge_checkpoints_passed', 0)
            miner.challenge_status = p.get('challenge_status', 'sampling')
            miner.termination_reason = p.get('termination_reason', '')

    def _load_anticopy_records(self, miners: Dict[int, MinerData], records: Dict[str, Dict[str, Any]]):
        for miner in miners.values():
            record = records.get(f"{miner.model_repo}#{miner.model_revision}", {})
            miner.anticopy_status = record.get('status', 'clean')
            miner.anticopy_target_uid = record.get('copy_of_uid')

    # ── Phase 2: Resolve champion ────────────────────────────────────────────

    def _resolve_champion(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        champion_state: Optional[Dict],
    ) -> Tuple[Optional[int], Optional[MinerData], Optional[int], bool]:
        """Returns (champion_uid, champion_miner, weight_uid, changed).

        - champion_uid/miner: in-round active champion (None if absent)
        - weight_uid: UID receiving 1.0 weight (always set when champion identity exists)
        - changed: True only on cold start
        """
        if not champion_state:
            logger.error(
                "No champion in system_config. "
                "Use `af db set-champion` to set one before starting the scorer."
            )
            return None, None, None, False

        hk = champion_state.get('hotkey')
        rev = champion_state.get('revision')

        for uid, miner in miners.items():
            if miner.hotkey == hk and miner.model_revision == rev:
                miner.is_champion = True
                return uid, miner, uid, False

        # Champion identity not in current scoring data → preserve weight on stored UID
        stored_uid = champion_state.get('uid')
        if hk:
            logger.warning(f"Champion {hk[:8]}... not present, weight preserved on UID {stored_uid}")
        return None, None, stored_uid, False

    # ── Phase 3a: Pairwise Pareto filter ────────────────────────────────────

    def _pairwise_filter(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        env_sampling_counts: Dict[str, int],
        champion_uid: Optional[int],
        champion_miner: Optional[MinerData] = None,
    ):
        """Compare non-champion miner pairs with sufficient common data.
        The earlier-registered miner is the incumbent. Dominated → terminated.

        Threshold is per-env: every env must have at least
        PARETO_MIN_WINDOWS × env_sampling_count common tasks. Using a
        single global window_size would mix min_common from one env with
        the window of another and give an inflated measure.
        """
        min_windows = self.config.PARETO_MIN_WINDOWS

        # Phase 3a-1: Champion strict-dominates check.
        # If champion exceeds a miner by margin in ALL envs, terminate
        # immediately — no point continuing to challenge.
        # Compare with miner as A (incumbent) and champion as B (challenger),
        # so b_dominates_a means "champion beats miner by margin in every env".
        if champion_miner and champion_uid is not None:
            for uid, m in miners.items():
                if uid == champion_uid or m.challenge_status == 'terminated':
                    continue
                if not self._meets_per_env_threshold(
                        champion_miner, m, environments,
                        env_sampling_counts, min_windows):
                    continue
                cmp = self.pareto._compare_miners(
                    m, champion_miner, environments, "champion_dominance",
                    min_dominant_envs=0)  # strict: ALL envs
                if cmp.b_dominates_a:  # champion actively beats miner in every env
                    m.challenge_status = 'terminated'
                    detail = ','.join(
                        f'{e}:{d.get("a_score",0):.3f}vs{d.get("b_score",0):.3f}{"✗" if d.get("winner")!="A" else "✓"}'
                        for e, d in cmp.env_comparisons.items() if d.get("winner"))
                    m.termination_reason = f'dominated_by_champion:{champion_miner.hotkey[:10]}|{detail}'
                    logger.info(f"CHAMPION DOMINANCE: UID {uid} terminated "
                                f"(champion exceeds by margin in all envs)")

        # Phase 3a-2: Non-champion pairwise Pareto dominance.
        eligible = sorted(
            (
                (uid, m) for uid, m in miners.items()
                if uid != champion_uid and m.challenge_status != 'terminated'
            ),
            key=lambda x: (x[1].first_block, x[0]),
        )

        for i, (uid_a, miner_a) in enumerate(eligible):
            if miner_a.challenge_status == 'terminated':
                continue
            for uid_b, miner_b in eligible[i + 1:]:
                if miner_b.challenge_status == 'terminated':
                    continue
                if not self._meets_per_env_threshold(
                        miner_a, miner_b, environments,
                        env_sampling_counts, min_windows):
                    continue

                cmp = self.pareto._compare_miners(
                    miner_a, miner_b, environments, "pairwise",
                    min_dominant_envs=self.config.PARETO_MIN_DOMINANT_ENVS)
                if cmp.a_dominates_b:
                    miner_b.challenge_status = 'terminated'
                    detail = ','.join(
                        f'{e}:{d.get("b_score",0):.3f}vs{d.get("a_score",0):.3f}{"✗" if d.get("winner")=="A" else "✓"}'
                        for e, d in cmp.env_comparisons.items() if d.get("winner"))
                    miner_b.termination_reason = f'dominated_by:{miner_a.hotkey[:10]}|{detail}'
                    logger.info(f"PAIRWISE: UID {uid_b} terminated by older UID {uid_a}")
                elif cmp.b_dominates_a:
                    miner_a.challenge_status = 'terminated'
                    detail = ','.join(
                        f'{e}:{d.get("a_score",0):.3f}vs{d.get("b_score",0):.3f}{"✗" if d.get("winner")=="B" else "✓"}'
                        for e, d in cmp.env_comparisons.items() if d.get("winner"))
                    miner_a.termination_reason = f'dominated_by:{miner_b.hotkey[:10]}|{detail}'
                    logger.info(f"PAIRWISE: UID {uid_a} terminated by newer UID {uid_b}")
                    break

    # ── Phase 3b: Champion challenges ────────────────────────────────────────

    def _run_challenges(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        env_sampling_counts: Dict[str, int],
        champion_uid: Optional[int],
        champion_miner: Optional[MinerData],
    ) -> List[ParetoComparison]:
        if not env_sampling_counts or any(
                env_sampling_counts.get(e, 0) <= 0 for e in environments):
            logger.warning(
                "env_sampling_counts missing/zero for some env, no challenges")
            return []
        if not champion_miner:
            return []

        warmup = self.config.CHAMPION_WARMUP_CHECKPOINTS
        dethrone_cp = self.config.CHAMPION_DETHRONE_MIN_CHECKPOINT
        M = self.config.CHAMPION_TERMINATION_TOTAL_LOSSES
        comparisons = []

        for uid, miner in miners.items():
            if uid == champion_uid or miner.challenge_status == 'terminated':
                continue

            # Checkpoint gate: per-env CP = common[env] // sampling_count[env],
            # overall CP = min across envs. This honors each env's window size
            # independently; mixing min_common from one env with the divisor
            # of another would inflate CP when sampling_counts differ.
            new_cp = self._per_env_cp(
                champion_miner, miner, environments, env_sampling_counts)
            if new_cp <= miner.challenge_checkpoints_passed:
                continue  # No new checkpoint reached

            prev_cp = miner.challenge_checkpoints_passed
            miner.challenge_checkpoints_passed = new_cp
            cp = new_cp

            # First-challenge slots bonus trigger: the moment a miner's CP
            # crosses warmup for the first time this reign (prev<warmup+1<=new).
            # CP is monotonic within a reign, so this fires at most once
            # per (miner, reign). Champion is already filtered above.
            first_real_cp = warmup + 1
            if prev_cp < first_real_cp <= new_cp:
                miner.should_grant_first_challenge_bonus = True

            # Only one comparison per round per miner — uses all available data.
            cmp = self.pareto._compare_miners(
                champion_miner, miner, environments, "champion_challenge",
                checkpoint=cp)
            comparisons.append(cmp)

            if cp <= warmup:
                result = "dominates" if cmp.b_dominates_a else "fails"
                logger.info(f"UID {uid} {result} at warmup CP {cp}/{warmup}")
                continue

            if cmp.b_dominates_a:
                miner.challenge_consecutive_wins += 1
                miner.challenge_consecutive_losses = 0
                logger.info(f"UID {uid} dominates at CP {cp} "
                            f"(wins: {miner.challenge_consecutive_wins})")
            else:
                miner.challenge_total_losses += 1
                miner.challenge_consecutive_losses += 1
                miner.challenge_consecutive_wins = 0
                # Always record latest loss detail so termination has context
                detail = ','.join(
                    f'{e}:{d.get("b_score",0):.3f}vs{d.get("a_score",0):.3f}{"✗" if d.get("winner")=="A" else "✓"}'
                    for e, d in cmp.env_comparisons.items() if d.get("winner"))
                miner.termination_reason = f'lost_to_champion:{champion_miner.hotkey[:10]}|{detail}'
                # At dethrone CP or beyond, losing is decisive — terminate immediately
                if cp >= dethrone_cp:
                    miner.challenge_status = 'terminated'
                    logger.info(f"UID {uid} fails at dethrone CP {cp} → terminated")
                else:
                    logger.info(f"UID {uid} fails at CP {cp} "
                                f"(losses: {miner.challenge_total_losses}/{M})")

        return comparisons

    # ── Phase 4: Dethrone check ──────────────────────────────────────────────

    def _check_dethrone(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        champion_uid: Optional[int],
    ) -> Tuple[Optional[int], Optional[MinerData]]:
        """Pick a qualified challenger to take the crown.

        A challenger qualifies when it has reached CHAMPION_DETHRONE_MIN_CHECKPOINT
        and dominates the champion (consecutive_wins > 0 means the most recent
        comparison was a win). Excludes terminated miners. If multiple qualify,
        picks the earliest registered (tiebreaker: highest geometric mean)."""
        dethrone_cp = self.config.CHAMPION_DETHRONE_MIN_CHECKPOINT
        qualified = [
            (uid, self._geo_mean(miners[uid], environments), miners[uid].first_block)
            for uid, m in miners.items()
            if uid != champion_uid
            and m.challenge_status != 'terminated'
            and m.challenge_checkpoints_passed >= dethrone_cp
            and m.challenge_consecutive_wins > 0
        ]
        if not qualified:
            return None, None
        qualified.sort(key=lambda x: (x[2], -x[1]))
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
                # termination_reason already set by _run_challenges with last loss detail

    # ── Phase 5b: Per-challenger vs-champion snapshot ────────────────────────

    def _populate_vs_champion_per_env(
        self,
        miners: Dict[int, MinerData],
        environments: List[str],
        champion_miner: Optional[MinerData],
    ) -> None:
        """Fill miner.vs_champion_per_env for every non-champion miner.

        For each (challenger, env) this records the exact numbers the Pareto
        comparison uses: champion's avg on common tasks, challenger's avg on
        common tasks, and the dethrone threshold (champion_on_common +
        WIN_MARGIN_END). Runs regardless of checkpoint so downstream UIs
        always have the current real threshold, not a historical-avg proxy.
        """
        if not champion_miner:
            return

        margin = self.config.WIN_MARGIN_END
        not_worse_tol = self.config.WIN_NOT_WORSE_TOLERANCE
        for uid, miner in miners.items():
            if miner.is_champion:
                continue
            per_env: Dict[str, Dict[str, Any]] = {}
            for env in environments:
                es_champ = champion_miner.env_scores.get(env)
                es_chall = miner.env_scores.get(env)
                if not es_champ or not es_chall:
                    continue
                common = set(es_champ.all_task_scores) & set(es_chall.all_task_scores)
                if not common:
                    continue
                champ_on_common = sum(
                    es_champ.all_task_scores[t] for t in common
                ) / len(common)
                chall_on_common = sum(
                    es_chall.all_task_scores[t] for t in common
                ) / len(common)
                per_env[env] = {
                    "common_tasks": len(common),
                    "score_on_common": chall_on_common,
                    "champion_score_on_common": champ_on_common,
                    # Upper bound: if challenger's score-on-common exceeds
                    # this, they WIN this env in Pareto comparison.
                    "dethrone_threshold": champ_on_common + margin,
                    # Lower bound: if challenger's score-on-common drops
                    # below this, they LOSE this env (fails "not worse"
                    # check, blocking dethrone). Between the two → tie.
                    "not_worse_threshold": champ_on_common * (1 - not_worse_tol),
                }
            miner.vs_champion_per_env = per_env

    # ── Phase 6: Assign weights ──────────────────────────────────────────────

    def _assign_weights(
        self, miners: Dict[int, MinerData], weight_uid: Optional[int]
    ) -> Dict[int, float]:
        weights = {}
        for uid, miner in miners.items():
            w = 1.0 if uid == weight_uid else 0.0
            miner.normalized_weight = w
            weights[uid] = w
        # Champion's UID may be absent from miners dict (champion offline)
        if weight_uid is not None and weight_uid not in weights:
            weights[weight_uid] = 1.0
        return weights

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _geo_mean(self, miner: MinerData, environments: List[str]) -> float:
        scores = [miner.env_scores[env].avg_score for env in environments
                  if env in miner.env_scores]
        if len(scores) != len(environments):
            return 0.0
        return geometric_mean(scores, epsilon=self.config.GEOMETRIC_MEAN_EPSILON)

    def _per_env_cp(
        self,
        a: MinerData,
        b: MinerData,
        environments: List[str],
        env_sampling_counts: Dict[str, int],
    ) -> int:
        """Overall CP = min over envs of (common[env] // sampling_count[env]).

        Each env's CP is measured against its own window size, so heterogeneous
        sampling_counts are handled correctly. Returns 0 if any env is missing
        from either miner or has a non-positive sampling_count.
        """
        per_env_cp = []
        for env in environments:
            es_a = a.env_scores.get(env)
            es_b = b.env_scores.get(env)
            if not es_a or not es_b:
                return 0
            count = env_sampling_counts.get(env, 0)
            if count <= 0:
                return 0
            common = len(set(es_a.all_task_scores) & set(es_b.all_task_scores))
            per_env_cp.append(common // count)
        return min(per_env_cp) if per_env_cp else 0

    def _meets_per_env_threshold(
        self,
        a: MinerData,
        b: MinerData,
        environments: List[str],
        env_sampling_counts: Dict[str, int],
        min_windows: int,
    ) -> bool:
        """True iff common[env] >= min_windows × sampling_count[env] for every env.

        Per-env threshold (instead of a single global min×count) so an env
        with fewer common tasks can't be masked by another env with many.
        """
        for env in environments:
            es_a = a.env_scores.get(env)
            es_b = b.env_scores.get(env)
            if not es_a or not es_b:
                return False
            count = env_sampling_counts.get(env, 0)
            if count <= 0:
                return False
            common = len(set(es_a.all_task_scores) & set(es_b.all_task_scores))
            if common < min_windows * count:
                return False
        return True

    def _assert_env_counts(
        self, environments: List[str], env_sampling_counts: Dict[str, int]
    ) -> None:
        """Log once if any env is missing from the sampling counts map."""
        missing = [env for env in environments if not env_sampling_counts.get(env)]
        if missing:
            logger.warning(
                f"Missing env_sampling_counts for {missing}; challenges disabled. "
                f"Check environments config."
            )

    def _reset_state(self, miner: MinerData):
        miner.challenge_consecutive_wins = 0
        miner.challenge_total_losses = 0
        miner.challenge_consecutive_losses = 0
        miner.challenge_checkpoints_passed = 0
        miner.challenge_status = 'sampling'
        miner.termination_reason = ''

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
