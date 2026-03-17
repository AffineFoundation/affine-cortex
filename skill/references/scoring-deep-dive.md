# Scoring Algorithm Deep Dive

Detailed reference for Affine's 4-stage scoring pipeline.

## Pareto Frontier Algorithm

**File**: `affine-cortex/affine/src/scorer/stage2_pareto.py`

The Pareto filter removes "dominated" miners (likely copies). For each environment subset:

1. Sort miners by `first_block` (earliest commit wins ties)
2. For each miner pair (A earlier, B later):
   - Align scores on **common tasks** (only tasks both miners completed)
   - Calculate threshold: `threshold = prior_score + gap`
   - B "wins" an environment if `score_b > threshold + epsilon` (epsilon = 1e-9)
   - A dominates B only if A wins in **ALL** environments in the subset
3. Dominated miners are filtered from that subset

**Threshold Formula** (`utils.py:160-223`):
```
SE = sqrt(p * (1 - p) / n)
gap = z * SE
gap = clamp(gap, MIN_IMPROVEMENT, MAX_IMPROVEMENT)
threshold = min(prior_score + gap, 1.0)
```
- `p` = prior score (0-1), `n` = sample count, `z` = z_score (default 1.5 ~ 87% confidence)
- Floor: 2% minimum improvement required; Ceiling: 10% max

**Examples**: prior=0.5 n=100 -> threshold=0.575; prior=0.5 n=500 -> threshold=0.534

## Geometric Mean Scoring

**File**: `affine-cortex/affine/src/scorer/utils.py:120-157`

```
Standard:  GM = (v1 * v2 * ... * vn)^(1/n)   -- returns 0 if ANY value <= 0
Smoothed:  GM = ((v1+e) * (v2+e) * ... * (vn+e))^(1/n) - e
```
- Smoothing epsilon = 0.01 (prevents single zero from collapsing all scores)
- Applied per-subset in Stage 3 to combine environment scores

## Subset Scoring (Stage 3)

**File**: `affine-cortex/affine/src/scorer/stage3_subset.py`

For each subset, rank miners by geometric mean score, then:
- Layer weights: `2^(layer-1)` -- larger subsets weighted exponentially more
- Rank-based decay: position k receives `score * 0.5^(k-1)`
- Currently MAX_LAYERS=1, only full set is used

## Weight Distribution (Stage 4)

**File**: `affine-cortex/affine/src/scorer/stage4_weights.py`

- Sum all subset weights per miner
- Remove miners below 1% threshold
- Normalize remaining to sum = 1.0
- Sub-threshold weights redistributed to UID 0

## Parameters Reference

All in `affine/src/scorer/config.py`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `Z_SCORE` | 1.5 | Statistical confidence (~87%) |
| `MIN_IMPROVEMENT` | 0.02 | 2% minimum gap for dominance |
| `MAX_IMPROVEMENT` | 0.10 | 10% cap on dominance threshold |
| `DECAY_FACTOR` | 0.5 | Rank decay -- 50% per position |
| `GEOMETRIC_MEAN_EPSILON` | 0.01 | Smoothing to prevent zero collapse |
| `MIN_WEIGHT_THRESHOLD` | 0.01 | 1% minimum weight |
| `MIN_COMPLETENESS` | 0.9 | 90% completeness required |
