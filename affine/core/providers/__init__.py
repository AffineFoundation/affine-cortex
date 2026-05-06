"""Inference provider abstraction layer.

Unifies Chutes and Targon (and future providers) behind a single interface so
that task routing (sampling -> executor) is provider-agnostic.
"""

from affine.core.providers.base import BaseProvider, ProviderInstanceInfo
from affine.core.providers.chutes import ChutesProvider
from affine.core.providers.targon import TargonProvider

__all__ = [
    "BaseProvider",
    "ProviderInstanceInfo",
    "ChutesProvider",
    "TargonProvider",
]
