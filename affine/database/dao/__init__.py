"""
DAO implementations for all tables

Provides high-level data access interfaces.
"""

from affine.database.dao.execution_logs import ExecutionLogsDAO
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.system_config import SystemConfigDAO

__all__ = [
    "ExecutionLogsDAO",
    "MinerStatsDAO",
    "MinersDAO",
    "SampleResultsDAO",
    "ScoreSnapshotsDAO",
    "ScoresDAO",
    "SystemConfigDAO",
]
