"""Low-level provider clients.

Only ``targon_client`` survives the queue-window refactor — it's the SDK
wrapper used by :class:`affine.src.scorer.providers.targon.TargonProvider`.
The richer provider abstraction (lifecycle, role binding, pool) lives in
``affine.src.scorer.providers``.
"""

from affine.core.providers.targon_client import TargonClient, get_targon_client

__all__ = ["TargonClient", "get_targon_client"]
