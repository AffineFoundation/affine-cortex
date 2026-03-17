# API Reference

FastAPI server at `affine/api/server.py`, prefix `/api/v1`.

## Authentication

Executor auth via Bittensor keypair signatures:
- Headers: `X-Hotkey` (SS58), `X-Signature` (hex-encoded), `X-Message` (unix timestamp)
- Timestamp must be within 60 seconds
- Verified via `bittensor.Keypair.verify()`
- Non-strict mode: all hotkeys allowed (development)
- Strict mode: only registered validators

## Task Endpoints (`/api/v1/tasks`)

### POST /tasks/fetch
Fetch pending tasks for evaluation.
- Request: `X-Hotkey`, `X-Signature`, `X-Message` headers
- Response: `TaskFetchResponse` with enriched tasks (miner_uid, chute_slug, model)
- Internal: Gets ALL pending tasks, random shuffle, takes batch_size

### POST /tasks/submit
Submit evaluation results.
- Request: `SampleSubmission` with executor signature
- Response: `SampleSubmitResponse`
- Internal: Verify signature, background save sample + log + delete task

### GET /tasks/pool/stats
Task pool statistics.

## Sample Endpoints (`/api/v1/samples`)

### GET /samples/{hotkey}/{env}/{task_id}
Get specific sample result.

### GET /samples/scoring
Scoring data for all miners (used by scorer service).
- Cached with 5-minute refresh

### GET /samples/pool/uid/{uid}/{env}
Task pool for specific miner+env.

## Miner Endpoints (`/api/v1/miners`)

### GET /miners/uid/{uid}
Miner information by UID.

### GET /miners/uid/{uid}/stats
Miner sampling statistics.

## Score Endpoints (`/api/v1/scores`)

### GET /scores/latest
Latest score snapshot.

### GET /scores/uid/{uid}
Score for specific miner.

### GET /scores/weights/latest
Latest normalized weights (consumed by validators).

## Scoring Cache

- Proactive cache with background refresh every 5 minutes
- State machine: EMPTY -> WARMING -> READY <-> REFRESHING
- Two modes: `scoring` (enabled_for_scoring) and `sampling` (enabled_for_sampling)
- File: `affine/api/services/scoring_cache.py`
