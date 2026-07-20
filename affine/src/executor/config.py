"""Executor configuration.

The real concurrency gate is a single cross-process
``multiprocessing.BoundedSemaphore`` shared by every worker subprocess.
It is sized to the inference backend's measured saturation point; the
default is 400 in-flight evaluations per endpoint and can be overridden
for a different GPU profile.

Cross-env priority is adjusted by the manager from live progress:
backlogged envs that are using their share receive more slots, while
envs that cannot consume their current cap release capacity but keep a
small probe floor so they cannot starve.

Per-worker ``max_concurrent`` is a defensive floor — large enough not
to gate anything, just so a runaway pending list can't schedule tens of
thousands of coroutines into asyncio at once.
"""

import os


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a positive integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


# Per-host in-flight cap. ``ExecutorManager`` sizes the global sem to
# ``PER_HOST_DISPATCH_BUDGET × len(active_ssh_endpoints)`` and gates
# every dispatch against the host its sglang lives on.
PER_HOST_DISPATCH_BUDGET = _positive_int_env(
    "AFFINE_EXECUTOR_PER_HOST_DISPATCH_BUDGET",
    400,
)

GLOBAL_DISPATCH_BUDGET = PER_HOST_DISPATCH_BUDGET

# Per-worker asyncio-side safety floor. Never the real gate — the
# global sem above is. Set high so dispatch never blocks on local
# semaphore acquisition.
DEFAULT_MAX_CONCURRENT = 500


def get_max_concurrent(_env: str) -> int:
    """Return the per-worker dispatch cap. Argument retained for callsite
    stability; the cap is intentionally uniform across envs."""
    return DEFAULT_MAX_CONCURRENT
