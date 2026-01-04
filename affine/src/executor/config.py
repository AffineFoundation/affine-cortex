"""
Executor configuration for different environments
"""

# Max concurrent tasks for each environment
ENV_MAX_CONCURRENT = {
    "LGC": 90,
    "GAME": 180,
    "LGC-v2": 120,
}

# Default max concurrent tasks if environment not found in config
DEFAULT_MAX_CONCURRENT = 60


def get_max_concurrent(env: str) -> int:
    """Get max concurrent tasks for a specific environment.
    
    Args:
        env: Environment name (e.g., "affine:sat", "agentgym:webshop")
        
    Returns:
        Max concurrent tasks for the environment
    """
    return ENV_MAX_CONCURRENT.get(env, DEFAULT_MAX_CONCURRENT)