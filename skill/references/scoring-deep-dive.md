# Scoring Deep Dive — Champion Challenge System

> Verified against affine-cortex@e3a9fce (2026-06). **The old 4-stage ELO/Pareto pipeline (`stage1_collector.py` … `stage4_weights.py`, `elo.py`) no longer exists.** Scoring is now a single-champion contest. ELO and OpenSkill were fully removed.

## Concept

One **champion** holds the entire subnet. Challengers fight one-at-a-time to dethrone it. Weights are **winner-takes-all** (champion = 1.0, all others 0.0). There is no rank-decay distribution, no geometric mean, no per-env weight multiplier.

Code:
- `affine/src/scheduler/flow.py` — orchestrator (challenger selection, GPU deploy, battle decide, weight write, recovery)
- `affine/src/scorer/comparator.py` — the dethrone rule
- `affine/src/scorer/challenger_queue.py` — queue eligibility/lifecycle
- `affine/src/scorer/weight_writer.py` — winner-takes-all writes
- `affine/src/scorer/window_state.py` — champion/battle/env state; `get_scoring_environments`
- `affine/src/scorer/sampling_thresholds.py` — pool buffer + completion ratio
- `affine/api/routers/scores.py` — `/scores/weights/latest` (winner pass-through + optional past-N split)

## Challenger queue

Materialized (no separate table) from online miners joined with `miner_stats.challenge_status`:
- Eligible iff `is_valid == "true"`, `challenge_status` ∈ {missing, `"sampling"`}, `uid != champion_uid`.
- Ordered by `(first_block ASC, uid ASC)` — **earliest committer first**.
- `pick_next` atomically claims one: `sampling → in_progress`.
- Lifecycle: `sampling → in_progress →` (win) `champion` / (loss/fail) `terminated`. Terminated hotkeys never re-enter.
- Queued challengers may be pre-deployed on spare endpoints to accrue baseline samples in parallel; only one is the active battle.

## Battle window

- **One battle per window**, `WINDOW_BLOCKS = 7200` (~24h), overridable via `SCHEDULER_TASK_POOL_REFRESH_BLOCKS`.
- Both sides sampled on the **same per-env task pool**.
- **Pool size** = `ceil(sampling_count × 1.1)` (`SAMPLE_BUFFER_RATIO = 0.1`).
- **Champion completion** at 95% of pool (`CHAMPION_COMPLETION_RATIO = 0.95`); bottom 5% long tail abandoned. Challenger then samples the champion's done-set, early-stopping per env at `sampling_count` overlap.
- **Comparison restricted to the intersection** of task_ids both sides actually sampled (asymmetric coverage neutralized).

## The dethrone rule (`comparator.py`)

For each scoring env, classify the challenger vs the champion mean:

| Bucket | Condition |
|--------|-----------|
| **dominant** | `chal_avg > champ_avg + margin` (`margin = DEFAULT_MARGIN = 0.03`) |
| **not_worse** | `chal_avg ≥ champ_avg × (1 − not_worse_tolerance)`, `tolerance = 0.02` → keep ≥ 98% |
| **worse** | below the not-worse bound |

- **Sign-crossing envs** (`ADDITIVE_MARGIN_ENVS = {"DISTILL-V2"}`) use an **additive** band `champ_avg − DEFAULT_ADDITIVE_MARGIN (0.02)` instead of the multiplicative one, because a % band is meaningless when scores cross zero.
- **Sample-count gate**: an env with `chal_n < min_tasks_per_env` (= `sampling_count` in prod) is forced to **worse** ("insufficient_challenger_samples").
- Champion missing an env → treated as mean `0.0` (can't block a sample-sufficient challenger).

**Decision** (`WIN_MIN_DOMINANT_ENVS = 1`, *partial Pareto*):

> Challenger wins iff **dominant_count ≥ 1 AND every evaluated env is at least not_worse** (no env may be `worse`).

(`min_dominant_envs == 0` would be strict Pareto — dominant in *every* env — but that path isn't used in production.)

**Early-regression short-circuit** (`flow.py`): since every env must be not-worse, a single env where the challenger falls below the not-worse threshold (once that env reaches decide depth) is a terminal loss; the battle ends early without waiting for slower envs. Thresholds match the full decide pass exactly.

## Weight emission

**Scheduler write** (`weight_writer.py`): winner-takes-all. One `scores` row per miner (champion 1.0, rest 0.0) + one `score_snapshots` row capturing winner uid/hotkey/revision/model and `final_weights`.

**Weights endpoint** (`/scores/weights/latest`, what the validator sets on-chain):
- *Pre-activation (default)*: winner-takes-all pass-through; the 1.0 share resolved by hotkey to its current valid uid. Deregistered/invalid → `BURN_UID = -1` → folded into UID 0 burn by the validator.
- *Post-activation (opt-in)*: weight split **evenly** across the most recent **N distinct champion hotkeys**, each `1/N` (`_DEFAULT_SPLIT_CHAMPION_COUNT = 5`). Gated by `weights_split_after_block`. **Both split params are absent from the shipped `system_config.json` → split OFF by default.**

## Active scoring environments

`get_scoring_environments` returns envs with both `enabled_for_sampling` and `enabled_for_scoring` true:

| Env | sampling_count | mode |
|-----|----------------|------|
| SWE-INFINITE | 200 | latest |
| NAVWORLD | 300 | random |
| MEMORY | 100 | random |
| TERMINAL | 300 | random |

DISTILL-V2 is `sampling` only (shadow) — data accrues but it's excluded from the dethrone decision. DISTILL, LIVEWEB, LOGPROBS are fully off.

## Key parameters

| Parameter | Value | File |
|-----------|-------|------|
| `DEFAULT_MARGIN` | 0.03 | `comparator.py` |
| `DEFAULT_NOT_WORSE_TOLERANCE` | 0.02 | `comparator.py` |
| `WIN_MIN_DOMINANT_ENVS` | 1 | `comparator.py` |
| `DEFAULT_ADDITIVE_MARGIN` | 0.02 | `comparator.py` |
| `ADDITIVE_MARGIN_ENVS` | `{DISTILL-V2}` | `comparator.py` |
| `SAMPLE_BUFFER_RATIO` | 0.1 | `sampling_thresholds.py` |
| `CHAMPION_COMPLETION_RATIO` | 0.95 | `sampling_thresholds.py` |
| `WINDOW_BLOCKS` | 7200 | `flow.py` |
| `_DEFAULT_SPLIT_CHAMPION_COUNT` | 5 | `scores.py` |
| `BURN_UID` | -1 | `scores.py` |
| `validator_burn_percentage` | 0.0 | `system_config.json` |

## Doc drift note

`docs/FAQ.md` still describes a stricter "all-envs-better" rule; the code is partial Pareto (`WIN_MIN_DOMINANT_ENVS=1`). Trust the code.
