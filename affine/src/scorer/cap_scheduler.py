"""Cross-env priority dispatch caps.

The plan calls for the dispatcher to favor envs with the most remaining
samples so all envs converge to "done" at roughly the same time. Each
worker subprocess owns its own asyncio semaphore — keeping that
architecture (one env per process, per-env paramiko transport, isolated
crash blast radius), we share priority via a single small DynamoDB key:

  - ``ExecutorManager`` (parent process) reads remaining counts every
    ``CAP_REFRESH_INTERVAL_SEC`` and writes ``executor_caps`` into
    ``system_config``.
  - Each worker reads that key at the top of every ``_tick`` and resizes
    its dispatch semaphore. Stale-by-a-few-seconds is fine; the budget
    only needs to redistribute over minutes as envs finish.

This module owns only the math. The publisher/reader live in
``executor/main.py`` and ``executor/worker.py`` respectively so they
can be tested independently from DB plumbing.
"""

from __future__ import annotations

from typing import Dict


DEFAULT_GLOBAL_BUDGET = 480  # 8 GPUs × 60 per-card target concurrency
DEFAULT_MIN_CAP = 5          # keep idle/done envs reactive to late retries
DEFAULT_MAX_CAP_PER_ENV = 200  # one env can never monopolize dispatch


def compute_caps(
    remaining: Dict[str, int],
    *,
    global_budget: int = DEFAULT_GLOBAL_BUDGET,
    min_cap: int = DEFAULT_MIN_CAP,
    max_cap_per_env: int = DEFAULT_MAX_CAP_PER_ENV,
) -> Dict[str, int]:
    """Allocate ``global_budget`` across envs in proportion to remaining
    work.

    Rules:

      - Active envs (``remaining > 0``) get a share of the budget
        proportional to their remaining count.
      - Done envs (``remaining <= 0``) get exactly ``min_cap`` so they
        can still pick up late-arriving retries or new task_ids without
        having to wait for the next cap refresh.
      - No env's cap exceeds ``max_cap_per_env``. Past that point a
        single env's env containers can't physically absorb more
        concurrency and over-dispatching just queues inside the
        container.
      - No env's cap goes below ``min_cap``. Reserves a small floor so
        an active env always has dispatch headroom even if its share
        rounds to zero.

    Empty input returns an empty dict.
    """
    if not remaining:
        return {}

    active_remaining = sum(max(0, r) for r in remaining.values())
    caps: Dict[str, int] = {}

    for env, r in remaining.items():
        if r <= 0:
            caps[env] = min_cap
            continue
        if active_remaining <= 0:
            caps[env] = min_cap
            continue
        share = int(global_budget * r / active_remaining)
        caps[env] = max(min_cap, min(max_cap_per_env, share))

    return caps
