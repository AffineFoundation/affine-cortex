"""
Local Statistics Store

Persists 5-minute granularity sampling statistics to local files.
Enables fast recovery after restarts without database queries.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from collections import defaultdict
from affine.core.setup import logger


class LocalStatsStore:
    """Local storage for 5-minute granularity sampling statistics"""
    
    def __init__(self, base_dir: str = "/var/lib/affine/sampling_stats"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Current 5-minute window statistics (in-memory buffer)
        self._current_window_start = self._get_window_start(int(time.time()))
        self._current_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "samples": 0,
            "success": 0,
            "rate_limit_errors": 0,
            "timeout_errors": 0,
            "other_errors": 0
        })
    
    def _get_window_start(self, timestamp: int) -> int:
        """Get 5-minute window start for a timestamp
        
        Args:
            timestamp: Unix timestamp (seconds)
            
        Returns:
            Window start timestamp (rounded down to 5-minute boundary)
        """
        return (timestamp // 300) * 300
    
    def _get_window_filename(self, window_start: int) -> Path:
        """Get filename for a time window
        
        Args:
            window_start: Window start timestamp
            
        Returns:
            Path to stats file
        """
        dt = datetime.fromtimestamp(window_start)
        filename = dt.strftime("stats_%Y-%m-%d_%H-%M.json")
        return self.base_dir / filename
    
    def record_sample(self, hotkey: str, revision: str, env: str,
                     success: bool, error_type: str):
        """Record a sampling event (accumulate to current window)
        
        Args:
            hotkey: Miner hotkey
            revision: Model revision
            env: Environment name
            success: Whether sample succeeded
            error_type: Error type ('rate_limit', 'timeout', 'other', or None)
        """
        current_time = int(time.time())
        window_start = self._get_window_start(current_time)
        
        # Check if window rotation is needed
        if window_start > self._current_window_start:
            self.flush()
            self._current_window_start = window_start
        
        # Accumulate statistics
        key = f"{hotkey}#{revision}#{env}"
        stats = self._current_stats[key]
        stats["samples"] += 1
        
        if success:
            stats["success"] += 1
        elif error_type == "rate_limit":
            stats["rate_limit_errors"] += 1
        elif error_type == "timeout":
            stats["timeout_errors"] += 1
        elif error_type == "other":
            stats["other_errors"] += 1
    
    def flush(self):
        """Flush current window statistics to file"""
        if not self._current_stats:
            return
        
        window_file = self._get_window_filename(self._current_window_start)
        
        data = {
            "timestamp": self._current_window_start,
            "window_start": self._current_window_start,
            "window_end": self._current_window_start + 300,
            "miners": {k: dict(v) for k, v in self._current_stats.items()}
        }
        
        with open(window_file, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        
        dt = datetime.fromtimestamp(self._current_window_start)
        logger.info(f"Flushed stats for window {dt.strftime('%Y-%m-%d %H:%M')}")
        
        # Reset to new defaultdict to avoid lingering references
        self._current_stats = defaultdict(lambda: {
            "samples": 0,
            "success": 0,
            "rate_limit_errors": 0,
            "timeout_errors": 0,
            "other_errors": 0
        })
    
    def load_aggregated_stats(self, hours: float) -> Dict[str, Dict[str, int]]:
        """Load and aggregate statistics for the last N hours
        
        Args:
            hours: Number of hours to load
            
        Returns:
            Aggregated statistics by miner key
            {
                "hotkey#revision#env": {
                    "samples": 1000,
                    "success": 950,
                    "rate_limit_errors": 30,
                    "timeout_errors": 10,
                    "other_errors": 10
                }
            }
        """
        cutoff_time = int(time.time()) - int(hours * 3600)
        aggregated: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "samples": 0,
            "success": 0,
            "rate_limit_errors": 0,
            "timeout_errors": 0,
            "other_errors": 0
        })
        
        # Iterate through all stats files
        for stats_file in sorted(self.base_dir.glob("stats_*.json")):
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                
                # Time filter
                if data["timestamp"] < cutoff_time:
                    continue
                
                # Aggregate each miner's statistics
                for miner_key, stats in data["miners"].items():
                    agg = aggregated[miner_key]
                    agg["samples"] += stats["samples"]
                    agg["success"] += stats["success"]
                    agg["rate_limit_errors"] += stats["rate_limit_errors"]
                    agg["timeout_errors"] += stats.get("timeout_errors", 0)
                    agg["other_errors"] += stats["other_errors"]
            
            except Exception as e:
                logger.error(f"Failed to load {stats_file}: {e}")
        
        return {k: dict(v) for k, v in aggregated.items()}
    
    def cleanup_old_files(self, keep_hours: int = 25):
        """Delete statistics files older than N hours
        
        Args:
            keep_hours: Number of hours to keep (default: 25 to ensure 24h coverage)
        """
        cutoff_time = int(time.time()) - (keep_hours * 3600)
        deleted = 0
        
        for stats_file in self.base_dir.glob("stats_*.json"):
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                
                if data["timestamp"] < cutoff_time:
                    stats_file.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup {stats_file}: {e}")
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old stats files")