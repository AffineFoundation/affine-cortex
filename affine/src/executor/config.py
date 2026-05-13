"""
Executor configuration for different environments
"""

# Max concurrent tasks for each environment.
#
# Per-env affinetes pool typically has 2-4 instances (env containers).
# Each container can realistically handle ~5 concurrent evaluations
# before the HTTP server queues up requests and the client sees
# ReadError. The numbers below target ~5 concurrent per container.
#
# Tuning history: the old executor was throttled by the scheduler's HTTP
# task fan-out (/tasks/fetch + rate limiter). The new DB-driven executor
# scans the full task pool at once, so explicit per-env concurrency caps
# replace that throttle.
ENV_MAX_CONCURRENT = {
    "LIVEWEB": 100,
    "NAVWORLD": 100,
    "SWE-INFINITE": 100,
    "MEMORY": 100,
    "DISTILL": 100,
    "TERMINAL": 100,
    "LOGPROBS": 100,       # currently disabled in environments.enabled
}

# Default for any env not listed above.
DEFAULT_MAX_CONCURRENT = 100


def get_max_concurrent(env: str) -> int:
    """Get max concurrent tasks for a specific environment.
    
    Args:
        env: Environment name (e.g., "affine:sat", "agentgym:webshop")
        
    Returns:
        Max concurrent tasks for the environment
    """
    return ENV_MAX_CONCURRENT.get(env, DEFAULT_MAX_CONCURRENT)