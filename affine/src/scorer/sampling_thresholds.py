"""Shared sampling-threshold policy for scheduler + executor.

These knobs decide:
  - how big the per-env task_id pool is (over-sampled relative to the
    base ``sampling_count`` so the scheduler can absorb a long tail of
    failed/slow tasks)
  - how complete a champion must be before a battle is allowed to start
    (i.e. how much of the pool must have a sample row)

Both numbers used to be split awkwardly: the scheduler oversampled the
pool by 10% and then required champion_done ≥ ``sampling_count``
(allowing 10% missing). The executor capped challenger work at
``sampling_count`` too. Math (see plan): with 10% missing on champion,
expected (champion ∩ challenger) ≈ 182 < 200 so ``_battle_overlap_ready``
could never trigger and the long tail blocked every model swap.

New policy: pool stays at ``sampling_count × 1.1`` but champion is
"done" at ``95% of pool`` (eg 209/220). The remaining 5% is the
deliberately-abandoned long tail. Challenger then dispatches
champion's full done-set with no cap, early-stopping per-env once
``overlap ≥ sampling_count`` so the math closes.
"""

from __future__ import annotations

import math


SAMPLE_BUFFER_RATIO = 0.1
"""Pool size = ``ceil(sampling_count × (1 + SAMPLE_BUFFER_RATIO))``.

Generated once per refresh by ``WindowSampler.generate``. The buffer
exists so that the slow / errored 5% tail doesn't block the contest."""


CHAMPION_COMPLETION_RATIO = 0.95
"""Fraction of the pool a champion must successfully sample for its
sampling phase to be considered done. The remaining 5% is the
operator-acknowledged abandonment threshold — a permanently-failing
task at this rate is a signal for human intervention, not a reason
to stall the entire battle pipeline."""


def champion_completion_threshold(sampling_count: int) -> int:
    """Per-env minimum champion successful-sample count for
    ``_samples_complete`` (scheduler) and the executor's per-env
    champion early-stop check."""
    pool = math.ceil(sampling_count * (1 + SAMPLE_BUFFER_RATIO))
    return math.ceil(pool * CHAMPION_COMPLETION_RATIO)
