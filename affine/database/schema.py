"""
DynamoDB table schema definitions

Defines table structures with partition keys, sort keys, and indexes.
"""

from typing import Dict, Any, List


def get_table_name(base_name: str) -> str:
    """Get full table name with prefix."""
    from affine.database.client import get_table_prefix
    return f"{get_table_prefix()}_{base_name}"


# Sample Results Table
#
# Design Philosophy:
# - PK combines the 3 most frequent query dimensions: hotkey + revision + env
# - SK uses task_id for natural ordering
# - uid removed (mutable, should query via bittensor metadata -> hotkey first)
# - GSI for efficient timestamp range queries (incremental updates)
# - block_number stored but not indexed (no block query requirement)
#
# Query Patterns:
# 1. Get samples by hotkey+revision+env -> Query by PK
# 2. Get samples by hotkey+revision (all envs) -> Query with PK prefix + filter
# 3. Get samples by hotkey (all revisions) -> Scan with hotkey prefix + filter
# 4. Get samples by timestamp range -> Use timestamp-index GSI (gsi_partition='SAMPLE' AND timestamp > :since)
# 5. Get samples by uid -> Query bittensor metadata first to get hotkey+revision, then query here
#
# GSI Design:
# - gsi_partition: Fixed value "SAMPLE" for all records (partition key)
# - timestamp: Milliseconds since epoch (range key, supports > < BETWEEN)
# - This design enables efficient Query operations for incremental updates
SAMPLE_RESULTS_SCHEMA = {
    "TableName": get_table_name("sample_results"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # MINER#{hotkey}#REV#{revision}#ENV#{env}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # TASK#{task_id}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "gsi_partition", "AttributeType": "S"},
        {"AttributeName": "timestamp", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "timestamp-index",
            "KeySchema": [
                {"AttributeName": "gsi_partition", "KeyType": "HASH"},   # Fixed "SAMPLE"
                {"AttributeName": "timestamp", "KeyType": "RANGE"},      # Sortable timestamp
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings for sample_results (30 days retention)
SAMPLE_RESULTS_TTL = {
    "AttributeName": "ttl",
}


# Task Pool Table
#
# Design Philosophy:
# - PK: MINER#{hotkey}#REV#{revision} - partition by miner for efficient cleanup
# - SK: ENV#{env}#STATUS#{status}#TASK_ID#{task_id} - composite sort key with business semantics
# - GSI1: env-status-index for weighted random task selection
#
# Query Patterns:
# 1. Weighted random task selection (by TaskPoolManager):
#    - Query GSI1 by ENV#{env}#STATUS#pending
#    - SK sorted by MINER, enabling efficient grouping and counting
#    - Weighted random select miner (probability ∝ task count)
#    - Randomly select one task from chosen miner
# 2. Miner task cleanup (by Scheduler):
#    - Query main table by PK=MINER#{hotkey}#REV#{revision}
#    - Batch delete all tasks for invalid miners (36x faster)
# 3. Check miner pending tasks (by Scheduler):
#    - Query main table by PK with env filter
#    - Direct query, no GSI needed
# 4. Pool statistics:
#    - Query GSI1 by ENV#{env}#STATUS#{status} with Select=COUNT
#
# Design Rationale:
# - No UUID: task_id has business semantics, easier to debug
# - MINER partition: enables O(m) cleanup instead of O(n) individual deletes
# - GSI1 SK by MINER: supports efficient weighted counting
# - Fairness: new miners don't wait for old miners (weighted random, not FIFO)
TASK_POOL_SCHEMA = {
    "TableName": get_table_name("task_pool"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # MINER#{hotkey}#REV#{revision}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # ENV#{env}#STATUS#{status}#TASK_ID#{task_id}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "gsi1_pk", "AttributeType": "S"},
        {"AttributeName": "gsi1_sk", "AttributeType": "S"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "env-status-index",
            "KeySchema": [
                {"AttributeName": "gsi1_pk", "KeyType": "HASH"},   # ENV#{env}#STATUS#{status}
                {"AttributeName": "gsi1_sk", "KeyType": "RANGE"},  # MINER#{hotkey}#REV#{revision}#TASK_ID#{task_id}
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# Legacy name for compatibility during transition
TASK_QUEUE_SCHEMA = TASK_POOL_SCHEMA


# Execution Logs Table
EXECUTION_LOGS_SCHEMA = {
    "TableName": get_table_name("execution_logs"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings (applied after table creation)
EXECUTION_LOGS_TTL = {
    "AttributeName": "ttl",
}


# Scores Table
SCORES_SCHEMA = {
    "TableName": get_table_name("scores"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "latest_marker", "AttributeType": "S"},
        {"AttributeName": "block_number", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "latest-block-index",
            "KeySchema": [
                {"AttributeName": "latest_marker", "KeyType": "HASH"},
                {"AttributeName": "block_number", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings (applied after table creation)
SCORES_TTL = {
    "AttributeName": "ttl",
}


# System Config Table
SYSTEM_CONFIG_SCHEMA = {
    "TableName": get_table_name("system_config"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Miners Table
# Schema design:
# - PK: UID#{uid} - unique primary key, each UID has only one record
# - No SK needed - single record per UID
# - GSI1: is-valid-index for querying valid/invalid miners
# - GSI2: hotkey-index for querying miner by hotkey
#
# Query patterns:
# 1. Get miner by UID: Direct get by PK
# 2. Get all valid miners: Query GSI1 with is_valid=true
# 3. Get miner by hotkey: Query GSI2 with hotkey
# 4. Get miners by model hash: Scan with filter (for anti-plagiarism)
MINERS_SCHEMA = {
    "TableName": get_table_name("miners"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "is_valid", "AttributeType": "S"},
        {"AttributeName": "hotkey", "AttributeType": "S"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "is-valid-index",
            "KeySchema": [
                {"AttributeName": "is_valid", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
        {
            "IndexName": "hotkey-index",
            "KeySchema": [
                {"AttributeName": "hotkey", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}



# Score Snapshots Table
# Stores metadata for each scoring calculation
SCORE_SNAPSHOTS_SCHEMA = {
    "TableName": get_table_name("score_snapshots"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # BLOCK#{block_number}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # TIME#{timestamp}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
        {"AttributeName": "latest_marker", "AttributeType": "S"},
        {"AttributeName": "timestamp", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "latest-index",
            "KeySchema": [
                {"AttributeName": "latest_marker", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

# TTL settings for score_snapshots
SCORE_SNAPSHOTS_TTL = {
    "AttributeName": "ttl",
}


# Miner Stats Table
# Schema design:
# - PK: HOTKEY#{hotkey} - partition by hotkey
# - SK: REV#{revision} - each revision is a separate record
#
# Query patterns:
# 1. Get miner stats: Direct query by hotkey + revision
# 2. Get all revisions for a hotkey: Query by PK prefix
# 3. Get all historical miners: Full table scan
# 4. Cleanup inactive miners: Full table scan with filter
#
# Design rationale:
# - Permanent storage of all miner metadata (not just current 256)
# - Real-time sampling statistics via sliding windows
# - No GSI needed (cleanup uses full scan, which is efficient for small tables)
MINER_STATS_SCHEMA = {
    "TableName": get_table_name("miner_stats"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},   # HOTKEY#{hotkey}
        {"AttributeName": "sk", "KeyType": "RANGE"},  # REV#{revision}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# All table schemas
ALL_SCHEMAS = [
    SAMPLE_RESULTS_SCHEMA,
    TASK_POOL_SCHEMA,
    EXECUTION_LOGS_SCHEMA,
    SCORES_SCHEMA,
    SYSTEM_CONFIG_SCHEMA,
    MINERS_SCHEMA,
    SCORE_SNAPSHOTS_SCHEMA,
    MINER_STATS_SCHEMA,
]