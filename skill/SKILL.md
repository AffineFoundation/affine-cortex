# Affine Cortex Project Notes

Affine miners submit HuggingFace model snapshots. The validator-side
scheduler deploys submitted models, samples them against the current champion,
and writes winner-takes-all weights.

## Current Miner Flow

1. Train or fine-tune a model.
2. Upload weights to a public HuggingFace repo.
3. Commit `{model, revision}` on chain with `af commit`.
4. Watch public status with `af get-rank` and `af get-miner`.

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
