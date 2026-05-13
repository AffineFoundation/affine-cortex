"""Scoring + state library shared by scheduler, executor, and API readers.

Despite the directory name, there is no longer an ``af servers scorer``
process — comparator + weight_writer are invoked synchronously by the
flow scheduler. Modules here are stateless / DAO-free and can be
imported by any service.
"""

from .challenger_queue import ChallengerQueue, MinerCandidate
from .comparator import (
    ComparisonResult,
    EnvComparison,
    EnvComparisonConfig,
    WindowComparator,
)
from .sampler import EnvSamplingConfig, WindowSampler
from .weight_writer import WeightSubject, WeightWriter
from .window_state import (
    BattleRecord,
    ChampionRecord,
    EnvConfig,
    InMemoryConfigStore,
    MinerSnapshot,
    StateStore,
    SystemConfigKVAdapter,
    TaskIdState,
)

__all__ = [
    "ChallengerQueue",
    "MinerCandidate",
    "ComparisonResult",
    "EnvComparison",
    "EnvComparisonConfig",
    "WindowComparator",
    "EnvSamplingConfig",
    "WindowSampler",
    "WeightSubject",
    "WeightWriter",
    "BattleRecord",
    "ChampionRecord",
    "EnvConfig",
    "InMemoryConfigStore",
    "MinerSnapshot",
    "StateStore",
    "SystemConfigKVAdapter",
    "TaskIdState",
]
