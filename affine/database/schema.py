"""
DynamoDB table schema definitions

Defines table structures with partition keys, sort keys, and indexes.
"""

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
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "sk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}







# Inference Endpoints Table (Stage AI)
#
# Provider-agnostic registry for inference deployments. Replaces the older
# Targon-specific ``targon_deployments`` table — operator can register
# multiple endpoints (SSH/sglang on b300, Targon hosted, future B300) here
# and providers read connection details from this table instead of env
# vars.
#
# Schema:
#   PK: ENDPOINT#{name}   — unique label per endpoint (operator-chosen)
#
# Non-key attributes (sparse — fields are provider-specific):
#   kind                   "ssh" | "targon"
#   active                 bool — false rows are ignored at startup
#   public_inference_url   the URL env containers actually connect to
#   notes                  free-form
#   assigned_uid           miner currently assigned to this endpoint
#   assigned_hotkey
#   assigned_model
#   assigned_revision
#   deployment_id          provider-specific live deployment id
#   base_url               URL executor should use for this assignment
#   assignment_role        "champion" | "challenger" | "active"
#   assigned_at            assignment timestamp
#
#   # ssh-kind extras
#   ssh_url                "ssh://user@host[:port]"
#   ssh_key_path           optional path to private key
#   sglang_port            port sglang listens on inside the docker container
#   sglang_dp              data-parallel size
#   sglang_image           docker image (default lmsysorg/sglang:latest)
#   sglang_cache_dir       host mount for HF cache
#   sglang_context_len     context length passed to sglang
#   sglang_mem_fraction    GPU memory fraction passed to sglang
#   sglang_chunked_prefill chunked-prefill size passed to sglang
#   sglang_tool_call_parser tool-call parser name, "none" to omit
#   ready_timeout_sec      readiness probe timeout
#   poll_interval_sec      readiness probe interval
#
#   # targon-kind extras (most config is still env-based for back-compat)
#   targon_api_url
#
# Query patterns:
# 1. Get one endpoint by name → GetItem on pk
# 2. List all active endpoints of a kind → full table scan + filter
#    (table has at most a handful of rows, scan is fine)
INFERENCE_ENDPOINTS_SCHEMA = {
    "TableName": get_table_name("inference_endpoints"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


# Anti-Copy (CEAC) — three sibling tables backing the lazy plagiarism
# detector described in ``devlog/ceac_design.md``.
#
# ``anticopy_rollouts`` indexes the rolling pool of champion-generated
# rollouts. Each row points at an R2 blob that holds the actual
# tokenized prompt + response. The 7-day rolling window is enforced by
# ``ttl``; refresh recomputes the daily slice.
#   PK: ROLLOUT#{champion_hotkey}#{env}#{task_id}
#   non-key:
#     champion_hotkey, champion_revision, env, task_id, day
#     tokenizer_sig, r2_key, response_len, created_at, ttl
ANTICOPY_ROLLOUTS_SCHEMA = {
    "TableName": get_table_name("anticopy_rollouts"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
        {"AttributeName": "tokenizer_sig", "AttributeType": "S"},
        {"AttributeName": "created_at", "AttributeType": "N"},
    ],
    "GlobalSecondaryIndexes": [
        {
            # All current rollouts with a given tokenizer signature,
            # newest first. Used by ``forward_worker`` to enumerate the
            # rollout set a candidate should be teacher-forced against.
            "IndexName": "tokenizer-created-index",
            "KeySchema": [
                {"AttributeName": "tokenizer_sig", "KeyType": "HASH"},
                {"AttributeName": "created_at", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        },
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

ANTICOPY_ROLLOUTS_TTL = {"AttributeName": "ttl"}


# ``anticopy_scores_index`` — one row per (miner_hotkey, revision). The
# heavy per-rollout logprob blob lives in R2; this DDB row is a small
# index so the verdict pass can scan all active candidates cheaply.
#   PK: SCORE#{hotkey}#{revision}
#   non-key:
#     hotkey, revision, tokenizer_sig, computed_at, r2_key
#     rollout_keys: list of "{champion_hotkey}#{env}#{task_id}" the
#                   score covers (used by pairwise intersection)
#     verdict_copy_of: hotkey of the earliest miner this one copies, or
#                     "" if independent / not yet evaluated.
#     n_overlap_max:   the largest overlap observed against any peer
#                     (debug aid; not used for verdict).
ANTICOPY_SCORES_INDEX_SCHEMA = {
    "TableName": get_table_name("anticopy_scores_index"),
    "KeySchema": [
        {"AttributeName": "pk", "KeyType": "HASH"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "pk", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


ANTICOPY_STATE_SCHEMA = {
    # Machine-managed metadata for the CEAC subsystem. Holds runtime
    # state the ``anticopy-refresh`` service writes on every daily tick
    # (active champion uid + tokenizer signature of the rollout pool)
    # — values that change too often to live in ``system_config``,
    # which is the human-tunable settings table.
    "TableName": get_table_name("anticopy_state"),
    "KeySchema": [
        {"AttributeName": "key", "KeyType": "HASH"},
    ],
    "AttributeDefinitions": [
        {"AttributeName": "key", "AttributeType": "S"},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}


