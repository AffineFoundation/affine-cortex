# Affine Cortex Project Notes

Affine miners submit HuggingFace model snapshots. The validator-side
scheduler deploys submitted models, samples them against the current champion,
and writes winner-takes-all weights.

## Current Miner Flow

1. Train or fine-tune a model.
2. Upload weights to a public HuggingFace repo.
3. Commit `{model, revision}` on chain with `af commit`.
4. Watch public status with `af get-rank` and `af get-miner`.

## Model Requirements

- Exactly one of two architectures, matched field-by-field against
  `affine/utils/model_size_checker.py`:
  - Qwen3-32B (dense, `model_type qwen3`).
  - Qwen3.6-35B-A3B (MoE, `model_type qwen3_5_moe`).
- A block gate (`AFFINE_QWEN36_ONLY_ENFORCE_BLOCK`) can restrict new
  submissions to Qwen3.6 only.
- Rejected at admission: quantized models, gated/private repos, names not
  containing `affine`, repos not ending with the hotkey.
- One commit per hotkey, ever. A new attempt needs a fresh hotkey.

## Scoring Model (Champion Challenge)

- One champion holds the subnet; weights are winner-takes-all.
- Challengers are ordered by `first_block` (earliest first) and fight one at a
  time. A challenger dethrones the champion only if it is dominant in at least
  one scoring env and not-worse (>=98%) in every scoring env.
- Comparison is restricted to the task IDs both sides sampled; a single
  regressing env ends the battle early as a loss.
- Margins live in `affine/src/scorer/comparator.py`
  (`DEFAULT_MARGIN` 0.03, `DEFAULT_NOT_WORSE_TOLERANCE` 0.02,
  `WIN_MIN_DOMINANT_ENVS` 1). No ELO/geometric-mean/per-env weights.
- Optional flat 1/N split across the past N champion hotkeys is off by default.

## Active Environments

- Scoring envs: `SWE-INFINITE`, `NAVWORLD`, `MEMORY`, `TERMINAL`.
- `DISTILL-V2` is sampled only (shadow) and excluded from the dethrone
  decision. `DISTILL`, `LIVEWEB`, `LOGPROBS` are off.
- The registry (`affine/core/environments.py`) still builds older images
  (print/lgc/game/cde/ded/abd/arc/swe-pro), but they are absent from
  `system_config.json` and inactive.

## Anti-Copy (CEAC)

- CEAC (Champion-Echo Anti-Copy, `affine/src/anticopy/`) replaced the old
  hidden-state/logprob-cosine detector entirely.
- It teacher-forces each candidate over the champion's rollouts and compares
  per-token logprob gaps at "decision positions" (champion top-1 logprob
  below -0.5). Copy iff the combined median gap is below `nll_threshold`
  (0.05). The current champion is exempt.
- Services: `anticopy-refresh` (champion rollout pool + verdict backfill) and
  `anticopy-worker` (SSH to a GPU host, remote `snapshot_download`, sglang
  teacher-force). A copy verdict is consumed by the monitor, which sets the
  miner invalid (non-permanent).
- A separate `model_hash` (safetensors SHA256) check catches exact clones.

## Current Public API

- `GET /api/v1/rank/current`: aggregate rank/status payload for `af get-rank`.
- `GET /api/v1/miners/uid/{uid}` and `/miners/hotkey/{hotkey}`: public miner metadata.
- `GET /api/v1/scores/latest` and `/scores/uid/{uid}`: score snapshots.
- `GET /api/v1/scores/weights/latest`: latest normalized weights.
- `GET /api/v1/config`: public validator config keys only.

## Scheduler Model

- State lives in `system_config`.
- Inference machines are represented by `inference_endpoints`.
- SSH endpoints are operator-managed and manually registered.
- Targon endpoints can be configured for automatic workload lifecycle.
- The scheduler supports multiple endpoints for one miner and different
  endpoints for concurrent miners.

## Sampling Model

- Task pools refresh every 7200 blocks.
- SWE and DISTILL use the latest configured task IDs for the refresh.
- Other environments sample deterministic random IDs from their configured
  ranges for the refresh.
- Executors write sample results; the comparator decides whether a challenger
  dethrones the champion.

## Reference Docs

- `skill/references/scoring-deep-dive.md`: full Champion Challenge mechanics.
- `skill/references/database-schema.md`: DynamoDB tables (incl. CEAC +
  inference endpoints).
- `skill/references/rl-training-guide.md`: miner training notes.
- `skill/references/api-reference.md`: API endpoints + auth.
- See also `docs/MINER.md`, `docs/VALIDATOR.md`, `docs/FAQ.md`.
