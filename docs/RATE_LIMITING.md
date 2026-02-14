# Sampling Rate Limiting Module

This document describes the design and implementation of the rate limiting module in the sampling scheduler.

## Table of Contents

- [Overview](#overview)
- [Design Goals](#design-goals)
- [Core Components](#core-components)
- [Rate Calculation Formula](#rate-calculation-formula)
- [Edge Cases](#edge-cases)
- [Known Limitations](#known-limitations)

## Overview

The rate limiting module prevents answer memorization attacks by limiting how frequently a miner can receive new sampling tasks. This ensures miners cannot rapidly sample the entire dataset to memorize answers before the sampling list rotates.

**Key Principle**: Rate limiting operates **independently of `rotation_enabled`**. Even when rotation is paused, miners are still rate-limited based on the configured rotation parameters. This prevents miners from exploiting paused rotation periods.

## Design Goals

| Goal | Implementation |
|------|----------------|
| Prevent memorization attacks | Limit sampling rate to slightly above rotation rate |
| Ensure timely completion | Minimum rate guarantee: complete sampling within 48 hours |
| Real-time enforcement | Sliding window counter avoids 5-minute stats delay |
| System model exemption | UID 0 and UID > 1000 are not rate limited |

## Core Components

### Data Structures

| Component | Description |
|-----------|-------------|
| `_allocation_timestamps` | In-memory storage, key=`{hotkey}#{revision}#{env}`, value=list of timestamps |
| `_get_allocation_count()` | Get allocation count in sliding window, auto-cleanup expired entries |
| `_record_allocations()` | Record new allocation timestamps after task creation |
| `_should_skip_env_for_miner()` | Core rate limiting decision logic |

### Flow

```
1. Scheduler calls _schedule_miner()
2. For each environment:
   a. Get allocation count via _get_allocation_count() (cleans expired timestamps)
   b. Call _should_skip_env_for_miner() to check rate limit
   c. If rate limited, skip this environment
3. After creating tasks, call _record_allocations() to track new allocations
```

## Rate Calculation Formula

```python
# Rotation-based rate (with 20% margin)
rotation_rate = rotation_count * (3600 / rotation_interval) * 1.2

# Minimum rate guarantee (complete in 48 hours)
min_rate = sampling_count / 48

# Use the higher of the two
allowed_per_hour = math.ceil(max(rotation_rate, min_rate))
```

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `rotation_count` | Tasks rotated per interval | 5 |
| `rotation_interval` | Rotation interval in seconds | 3600 |
| `sampling_count` | Total tasks in sampling list | 200 |
| `RATE_MARGIN` | Safety margin multiplier | 1.2 (20%) |

## Edge Cases

### Case 1: Normal Rotation Environment

```
rotation_count=5, rotation_interval=3600, sampling_count=200

rotation_rate = 5 * (3600/3600) * 1.2 = 6
min_rate = 200 / 48 = 4.17
allowed = ceil(max(6, 4.17)) = 6/hour
```

**Result**: Miner can allocate up to 6 tasks per hour per environment.

### Case 2: Slow Rotation Environment

```
rotation_count=1, rotation_interval=3600, sampling_count=200

rotation_rate = 1 * 1 * 1.2 = 1.2
min_rate = 200 / 48 = 4.17
allowed = ceil(max(1.2, 4.17)) = 5/hour
```

**Result**: Minimum rate guarantee kicks in, allowing 5 tasks/hour to ensure 48-hour completion.

### Case 3: Rotation Paused (rotation_enabled=false)

```
rotation_count=5, sampling_count=200, rotation_enabled=false

rotation_rate = 5 * 1 * 1.2 = 6
min_rate = 200 / 48 = 4.17
allowed = ceil(max(6, 4.17)) = 6/hour
```

**Result**: Rate limiting still applies (independent of `rotation_enabled` flag).

### Case 4: rotation_count=0

```
rotation_count=0, sampling_count=200

rotation_rate = 0
min_rate = 200 / 48 = 4.17
allowed = ceil(max(0, 4.17)) = 5/hour
```

**Result**: Only minimum rate guarantee applies.

### Case 5: No Configuration

```
rotation_count=0, sampling_count=0

rotation_rate = 0
min_rate = 0
allowed = 0 → No rate limiting
```

**Result**: No rate limiting when both parameters are zero.

### Case 6: System Models

```
uid=0 or uid>1000 → Skip rate limiting entirely
```

**Result**: System models (UID 0, UID > 1000) are never rate limited.

## Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| In-memory storage | Counter resets on process restart | Low: temporary over-allocation, infrequent restarts |
| No cleanup on miner removal | `_allocation_timestamps` accumulates | Low: ~KB per key, can add periodic cleanup |
| Counter resets on revision change | New revision = new counter | Low: revision changes are costly for miners |

## Implementation Checklist

| # | Item | Status |
|---|------|--------|
| 1 | Sliding window avoids 5-minute stats delay | ✅ |
| 2 | Minimum rate ensures 48-hour completion | ✅ |
| 3 | Independent of `rotation_enabled` | ✅ |
| 4 | `math.ceil()` makes fractional rates effective | ✅ |
| 5 | `rotation_count=0` uses minimum rate | ✅ |
| 6 | System models (uid==0 or >1000) exempt | ✅ |
| 7 | Records allocations immediately after task creation | ✅ |

## Related Documentation

- [Validator Guide](VALIDATOR.md) - Validator setup and operation
- [Miner Guide](MINER.md) - Miner setup and operation
- [FAQ](FAQ.md) - Frequently Asked Questions
