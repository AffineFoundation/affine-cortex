# API Reference

FastAPI server at `affine/api/server.py`, prefix `/api/v1`.

## Public Endpoints

### GET /rank/current

Aggregated rank/status payload consumed by `af get-rank`.

Response sections:
- `window`: current champion, in-flight battle, task refresh block, aggregate sample counts.
- `queue`: pending challenger queue head.
- `scores`: latest score snapshot with per-environment metrics.

### GET /miners/uid/{uid}

Public miner metadata by UID.

### GET /miners/hotkey/{hotkey}

Public miner metadata by hotkey.

### GET /scores/latest

Latest score snapshot.

### GET /scores/uid/{uid}

Score row for a specific miner UID.

### GET /scores/weights/latest

Latest normalized weights consumed by validator weight setting.

### GET /config

Public validator config keys only.

## Internal Endpoints

Internal diagnostics are mounted only when `INTERNAL_ENDPOINTS_ENABLED` is set.
