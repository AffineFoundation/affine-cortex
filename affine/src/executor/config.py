"""Executor configuration.

The real concurrency gate is a single cross-process
``multiprocessing.BoundedSemaphore`` shared by every worker subprocess.
It's sized to the inference backend's saturation point (b300: 8 GPUs ×
60 in-flight per card) — the goal is to keep that semaphore exhausted
so the inference cluster stays fully utilized while we sample.

Cross-env priority is adjusted by the manager from live progress:
backlogged envs that are using their share receive more slots, while
envs that cannot consume their current cap release capacity but keep a
small probe floor so they cannot starve.

Per-worker ``max_concurrent`` is a defensive floor — large enough not
to gate anything, just so a runaway pending list can't schedule tens of
thousands of coroutines into asyncio at once.
"""

# Per-host in-flight cap. ``ExecutorManager`` sizes the global sem to
# ``PER_HOST_DISPATCH_BUDGET × len(active_ssh_endpoints)`` and gates
# every dispatch against the host its sglang lives on.
PER_HOST_DISPATCH_BUDGET = 600

GLOBAL_DISPATCH_BUDGET = PER_HOST_DISPATCH_BUDGET

# Per-worker asyncio-side safety floor. Never the real gate — the
# global sem above is. Set high so dispatch never blocks on local
# semaphore acquisition.
DEFAULT_MAX_CONCURRENT = 500


def get_max_concurrent(_env: str) -> int:
    """Return the per-worker dispatch cap. Argument retained for callsite
    stability; the cap is intentionally uniform across envs."""
    return DEFAULT_MAX_CONCURRENT
