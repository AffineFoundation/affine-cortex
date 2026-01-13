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
        
        # Multi-window buffer: {window_start: {miner_key: stats}}
        # Each window's data is tracked separately
        self._window_buffers: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {
                "samples": 0,
                "success": 0,
                "rate_limit_errors": 0,
                "timeout_errors": 0,
                "other_errors": 0
            })
        )
    
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
        """Record a sampling event (accumulate to window buffer)
        
        Args:
            hotkey: Miner hotkey
            revision: Model revision
            env: Environment name
            success: Whether sample succeeded
            error_type: Error type ('rate_limit', 'timeout', 'other', or None)
        
        Note:
            Data is accumulated to the appropriate time window buffer
            and will be flushed periodically by the sync loop.
        """
        # Determine which window this sample belongs to
        current_time = int(time.time())
        window_start = self._get_window_start(current_time)
        
        # Accumulate to the appropriate window buffer
        key = f"{hotkey}#{revision}#{env}"
        stats = self._window_buffers[window_start][key]
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
        """Flush all accumulated window buffers to files
        
        This method:
        1. Iterates through all window buffers
        2. For each window: reads existing file, accumulates buffer, writes back
        3. Clears all flushed buffers
        
        Called periodically by sync loop (every 5 minutes).
        """
        if not self._window_buffers:
            return
        
        flushed_windows = []
        
        # Flush each window's buffer
        for window_start, window_buffer in self._window_buffers.items():
            if not window_buffer:
                continue
            
            window_file = self._get_window_filename(window_start)
            
            # Step 1: Read existing data from file
            existing_miners = {}
            if window_file.exists():
                try:
                    with open(window_file, 'r') as f:
                        existing_data = json.load(f)
                        existing_miners = existing_data.get("miners", {})
                except Exception as e:
                    logger.warning(f"Failed to load existing stats from {window_file}: {e}")
            
            # Step 2: Accumulate buffer to existing data
            for miner_key, buffer_stats in window_buffer.items():
                if miner_key not in existing_miners:
                    existing_miners[miner_key] = {
                        "samples": 0,
                        "success": 0,
                        "rate_limit_errors": 0,
                        "timeout_errors": 0,
                        "other_errors": 0
                    }
                
                # Accumulate all metrics
                existing_miners[miner_key]["samples"] += buffer_stats["samples"]
                existing_miners[miner_key]["success"] += buffer_stats["success"]
                existing_miners[miner_key]["rate_limit_errors"] += buffer_stats["rate_limit_errors"]
                existing_miners[miner_key]["timeout_errors"] += buffer_stats["timeout_errors"]
                existing_miners[miner_key]["other_errors"] += buffer_stats["other_errors"]
            
            # Step 3: Write accumulated data to file
            data = {
                "timestamp": window_start,
                "window_start": window_start,
                "window_end": window_start + 300,
                "miners": existing_miners
            }
            
            with open(window_file, 'w') as f:
                json.dump(data, f, separators=(',', ':'))
            
            dt = datetime.fromtimestamp(window_start)
            logger.info(f"Flushed stats for window {dt.strftime('%Y-%m-%d %H:%M')}")
            flushed_windows.append(window_start)
        
        # Step 4: Clear flushed buffers
        for window_start in flushed_windows:
            del self._window_buffers[window_start]
    
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