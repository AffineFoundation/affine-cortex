"""Executor configuration.

The real concurrency gate is a single cross-process
``multiprocessing.BoundedSemaphore`` shared by every worker subprocess.
It's sized to the inference backend's saturation point (b300: 8 GPUs ×
60 in-flight per card) — the goal is to keep that semaphore exhausted
so the inference cluster stays fully utilized while we sample.

Cross-env priority emerges naturally from the shared sem: a worker with
more remaining task_ids spawns more coroutines, makes more acquire
attempts per unit time, and so wins a proportionally larger share of
slots than envs with little remaining work. No explicit priority queue
is needed.

Per-worker ``max_concurrent`` is a defensive floor — large enough not
to gate anything, just so a runaway pending list can't schedule tens of
thousands of coroutines into asyncio at once.
"""

# Total executor in-flight budget. The b300 inference cluster saturates
# around 480 (8 × 60 per-card), and that's still the bottleneck for envs
# that hit it (SWE-INFINITE, MEMORY, NAVWORLD, TERMINAL). 600 adds
# headroom for envs that don't go through b300 — LIVEWEB runs against
# its own SSH hosts for web scraping, so its in-flight share doesn't
# consume GPU capacity. Pair with per-env ``max_concurrent`` caps
# (EnvConfig.max_concurrent) to keep any single env from monopolising
# the shared pool.
GLOBAL_DISPATCH_BUDGET = 600

# Per-worker asyncio-side safety floor. Never the real gate — the
# global sem above is. Set high so dispatch never blocks on local
# semaphore acquisition.
DEFAULT_MAX_CONCURRENT = 500


def get_max_concurrent(_env: str) -> int:
    """Return the per-worker dispatch cap. Argument retained for callsite
    stability; the cap is intentionally uniform across envs."""
    return DEFAULT_MAX_CONCURRENT
