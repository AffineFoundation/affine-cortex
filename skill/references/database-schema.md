# DynamoDB Schema Reference

9 tables, all PAY_PER_REQUEST billing.

## Tables

| Table | PK | SK | TTL | Purpose |
|-------|----|----|-----|---------|
| **sample_results** | `MINER#{hotkey}#REV#{rev}#ENV#{env}` | `TASK#{task_id}` | 30 days | Completed evaluation samples |
| **task_pool** | `MINER#{hotkey}#REV#{rev}` | `ENV#{env}#STATUS#{status}#TASK_ID#{id}` | - | Pending/assigned tasks |
| **scores** | `SNAPSHOT#{id}` | `UID#{uid}` | - | Score snapshots |
| **miners** | `MINER#{hotkey}` | `REVISION#{rev}` | - | Miner registrations |
| **miner_stats** | `MINER#{hotkey}#REV#{rev}` | `ENV#{env}` | - | Per-env sampling stats |
| **execution_logs** | `EXEC#{id}` | `TASK#{task_id}` | 7 days | Execution history |
| **score_snapshots** | `SNAPSHOT#{id}` | `META` | - | Score snapshot metadata |
| **system_config** | `CONFIG` | `KEY#{key}` | - | System configuration |
| **anti_copy_results** | `PAIR#{a}#{b}` | `TIMESTAMP#{ts}` | 30 days | Copy detection results |

## Key Access Patterns

- **Task fetch**: Query task_pool by PK (miner+rev), filter by env+status=PENDING
- **Sample storage**: Put to sample_results after evaluation
- **Score query**: Query scores by snapshot ID (latest), or scan for top-N
- **Miner lookup**: Get miner by hotkey, query revisions
- **Stats aggregation**: Query miner_stats for completeness checks

## Key Files

- `affine/database/schema.py` -- Table definitions
- `affine/database/tables.py` -- Table initialization + TTL management
- `affine/database/dao/*.py` -- Data access objects for each table
