# DynamoDB Schema Reference

> Verified against affine-cortex@e3a9fce (2026-06). 11 tables, PAY_PER_REQUEST. **Changed since the ELO era:** the old `task_pool` and `anti_copy_results` tables are gone; `inference_endpoints` and three `anticopy_*` tables were added (for self-hosted serving + CEAC).

## Tables (`affine/database/schema.py`)

| Table | Purpose |
|-------|---------|
| **sample_results** | Completed evaluation samples (TTL ~30d). Keyed by miner+revision+env / task_id |
| **execution_logs** | Execution history (TTL ~7d) |
| **scores** | Per-snapshot per-uid scores (winner-takes-all: champion 1.0, rest 0.0) |
| **score_snapshots** | Snapshot metadata — winner uid/hotkey/revision/model, `final_weights`, full battle context |
| **miners** | Miner registrations (commit `{model, revision}`, model_hash, is_valid) |
| **miner_stats** | Per-miner lifecycle: `challenge_status` (sampling/in_progress/champion/terminated), wins/losses, termination_reason, per-env sampling stats |
| **system_config** | System configuration — env gates, champion record, contest params, anticopy config |
| **inference_endpoints** | Serving endpoints registry — `kind: "ssh" \| "targon"`, base_url, autoscale_managed, lease info |
| **anticopy_rollouts** | CEAC: champion rollout pool index (R2-backed decision-position blobs) |
| **anticopy_scores_index** | CEAC: per-candidate forward-pass score blobs (done-marker; no verdict) |
| **anticopy_state** | CEAC: refresh/verdict service state |

(Large blobs — rollouts, CEAC score vectors — live in Cloudflare **R2**, not DynamoDB; the tables hold indexes/pointers.)

## Key access patterns

- **Challenger queue** is *materialized*, not a table: online `miners` joined with `miner_stats.challenge_status`, ordered `(first_block ASC, uid ASC)`.
- **Champion** is a record in `system_config` (`ChampionRecord`).
- **Weights**: `/scores/weights/latest` reads `scores` + `score_snapshots` (winner share, optional past-N-champion split).
- **CEAC verdict**: written to `anticopy_scores_index` / via DAO `update_verdict`; consumed by the monitor (`verdict_copy_of` → `mark_invalid`).
- **Serving**: scheduler reads/writes `inference_endpoints` to deploy SGLang on SSH/Targon hosts.

## Key files

- `affine/database/schema.py` — table definitions (`get_table_name(...)`)
- `affine/database/tables.py` — table init + TTL
- `affine/database/dao/*.py` — DAOs: `anticopy.py`, `execution_logs.py`, `inference_endpoints.py`, `miner_stats.py`, `miners.py`, `sample_results.py`, `score_snapshots.py`, `scores.py`, `system_config.py`
