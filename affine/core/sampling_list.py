"""
Sampling List Manager

Manages dynamic sampling list generation and rotation logic.
"""

import random
from typing import List, Tuple, Set, Dict, Any

from affine.core.setup import logger
from affine.core.range_set import RangeSet


def get_task_id_set_from_config(env_config: Dict[str, Any]) -> Set[int]:
    """Get task ID set from environment configuration.
    
    Prioritizes sampling_list from sampling_config.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        Set of task IDs
    """
    sampling_config = env_config.get('sampling_config', {})
    
    # Use sampling_list
    sampling_list = sampling_config.get('sampling_list')
    if sampling_list:
        return set(sampling_list)
    
    # If no sampling_list, return empty set (should not happen in normal operation)
    logger.warning(
        f"No sampling_list found in sampling_config, returning empty set. "
        f"Config: {sampling_config}"
    )
    return set()


class SamplingListManager:
    """Sampling list manager for dynamic task rotation."""
    
    async def initialize_sampling_list(
        self,
        env: str,
        initial_range: List[List[int]],
        sampling_size: int
    ) -> List[int]:
        """Initialize sampling list from initial range.
        
        Uses RangeSet for efficient handling of large ranges.
        
        Args:
            env: Environment name
            initial_range: Initial range in [[start, end], ...] format
            sampling_size: Target sampling list size
            
        Returns:
            Initialized task ID list (may be smaller than sampling_size if insufficient IDs)
        """
        # Use RangeSet to avoid expanding large ranges
        range_set = RangeSet(initial_range)
        total_ids = range_set.size()
        
        # Randomly sample up to sampling_size
        actual_size = min(total_ids, sampling_size)
        sampling_list = range_set.random_sample(actual_size)
        
        logger.info(
            f"Initialized sampling list for {env}: "
            f"sampled {actual_size} from {total_ids} initial IDs (target={sampling_size})"
        )
        
        return sorted(sampling_list)
    
    async def rotate_sampling_list(
        self,
        env: str,
        current_list: List[int],
        dataset_range: List[List[int]],
        sampling_count: int,
        rotation_count: int,
        prioritize_new: bool = False
    ) -> Tuple[List[int], List[int], List[int]]:
        """Rotate sampling list while maintaining sampling_count size.

        Rotation strategy:
        1. Use RangeSet to avoid expanding large dataset_range
        2. Calculate available = dataset - current (using RangeSet)
        3. Determine removal/addition counts based on current vs target size
        4. Remove from the FRONT of the list (old data first)
        5. Add new data to the END (no sorting to preserve order)

        Args:
            env: Environment name
            current_list: Current sampling list (order matters)
            dataset_range: Dataset range in [[start, end], ...] format (supports multiple ranges)
            sampling_count: Target sampling list size
            rotation_count: Number of IDs to rotate per cycle
            prioritize_new: If True, prioritize sampling from later range segments
                (for dynamic-range environments where newer segments = newer data)

        Returns:
            (new_list, removed_ids, added_ids)
        """
        if rotation_count < 0:
            logger.warning(f"Invalid rotation_count for {env}: {rotation_count}")
            return current_list, [], []
        
        # rotation_count=0 is valid: only adjust size, no rotation
        
        # Use RangeSet to avoid expanding large ranges
        dataset_rangeset = RangeSet(dataset_range)
        dataset_size = dataset_rangeset.size()
        current_set = set(current_list)
        
        # Calculate available IDs using RangeSet subtraction
        available_rangeset = dataset_rangeset.subtract_ids(current_set)
        available_count = available_rangeset.size()
        
        # Safety check: Skip if would use > 80% of dataset
        if sampling_count + rotation_count > dataset_size * 0.8:
            logger.warning(
                f"Skipping rotation for {env}: safety check failed - "
                f"sampling_count ({sampling_count}) + rotation_count ({rotation_count}) "
                f"> 80% of dataset ({dataset_size * 0.8:.0f})"
            )
            return current_list, [], []
        
        current_size = len(current_list)
        
        # Determine removal and addition counts
        if current_size < sampling_count:
            # Fill mode: Only add to reach target
            to_remove = 0
            to_add = sampling_count - current_size
        elif current_size > sampling_count:
            # Shrink+Rotate mode: Remove surplus + rotation_count, add rotation_count
            surplus = current_size - sampling_count
            to_remove = surplus + rotation_count
            to_add = rotation_count
        else:
            # Standard rotation: Remove N, add N
            to_remove = rotation_count
            to_add = rotation_count
        
        # Skip if not enough available IDs
        if available_count < to_add:
            logger.warning(
                f"Skipping rotation for {env}: insufficient available IDs - "
                f"need={to_add}, available={available_count}"
            )
            return current_list, [], []
        
        # Execute removal: Remove from FRONT (old data first)
        if to_remove > 0:
            to_remove = min(to_remove, current_size)
            removed_ids = current_list[:to_remove]  # Remove from front
            remaining_list = current_list[to_remove:]  # Keep rest in order
        else:
            removed_ids = []
            remaining_list = current_list[:]
        
        # Execute addition: Add to END (recalculate available after removal)
        remaining_set = set(remaining_list)
        if prioritize_new:
            # Iterate raw segments in reverse (newest first) to preserve
            # segment ordering that RangeSet normalization would merge away
            added_ids = []
            needed = to_add
            for segment in reversed(dataset_range):
                if needed <= 0:
                    break
                seg_available = RangeSet([segment]).subtract_ids(remaining_set)
                seg_count = min(needed, seg_available.size())
                if seg_count > 0:
                    sampled = seg_available.random_sample(seg_count)
                    added_ids.extend(sampled)
                    remaining_set.update(sampled)
                    needed -= seg_count
        else:
            available_for_add = dataset_rangeset.subtract_ids(remaining_set)
            added_ids = available_for_add.random_sample(to_add)
        
        # Merge: Keep order - remaining list + new additions at end
        new_list = remaining_list + added_ids
        
        logger.info(
            f"Rotated {env}: removed={len(removed_ids)} from front, added={len(added_ids)} to end, "
            f"size: {current_size} -> {len(new_list)} (target={sampling_count})"
        )
        
        return new_list, removed_ids, added_ids