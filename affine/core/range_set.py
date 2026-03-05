"""
Range Set - Efficient interval-based set operations

Handles large ranges without expanding all IDs into memory.
Supports operations like subtraction and random sampling on intervals.
"""

import random
from typing import List, Set, Tuple


class RangeSet:
    """Efficient interval-based set representation.
    
    Stores data as a list of non-overlapping, sorted intervals [start, end)
    where end is exclusive. This avoids expanding billions of IDs into memory.
    """
    
    def __init__(self, ranges: List[List[int]]):
        """Initialize from a list of [start, end) intervals.
        
        Args:
            ranges: List of [start, end) pairs where end is exclusive
        """
        self.ranges = self._normalize_ranges(ranges)
    
    @classmethod
    def from_ids(cls, ids: List[int]) -> 'RangeSet':
        """Create a RangeSet from a list of individual IDs.
        
        Efficiently converts a list of IDs into merged intervals.
        For example, [1, 2, 3, 7, 8, 10] becomes [[1, 4), [7, 9), [10, 11)].
        
        Args:
            ids: List of integer IDs (need not be sorted or unique)
            
        Returns:
            RangeSet covering exactly the given IDs
        """
        if not ids:
            return cls([])
        
        sorted_ids = sorted(set(ids))
        ranges = []
        start = sorted_ids[0]
        prev = start
        
        for id_val in sorted_ids[1:]:
            if id_val == prev + 1:
                prev = id_val
            else:
                ranges.append([start, prev + 1])
                start = id_val
                prev = id_val
        
        ranges.append([start, prev + 1])
        return cls(ranges)
    
    def _normalize_ranges(self, ranges: List[List[int]]) -> List[Tuple[int, int]]:
        """Normalize ranges: merge overlapping intervals and sort.
        
        Args:
            ranges: List of [start, end) pairs
            
        Returns:
            List of non-overlapping, sorted (start, end) tuples
        """
        if not ranges:
            return []
        
        # Convert to tuples, filter out zero-width ranges, and sort by start position
        intervals = sorted((r[0], r[1]) for r in ranges if r[1] > r[0])
        
        if not intervals:
            return []
        
        # Merge overlapping intervals
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            
            if start <= last_end:
                # Overlapping or adjacent - merge
                merged[-1] = (last_start, max(last_end, end))
            else:
                # Non-overlapping - add new interval
                merged.append((start, end))
        
        return merged
    
    def size(self) -> int:
        """Calculate total number of IDs in all ranges.
        
        Returns:
            Total count of IDs
        """
        return sum(end - start for start, end in self.ranges)
    
    def subtract_ids(self, ids: Set[int]) -> 'RangeSet':
        """Subtract a set of IDs from this RangeSet.
        
        This splits intervals at points where IDs are removed.
        
        Args:
            ids: Set of IDs to remove
            
        Returns:
            New RangeSet with IDs removed
        """
        if not ids:
            return RangeSet([[s, e] for s, e in self.ranges])
        
        new_ranges = []
        
        for start, end in self.ranges:
            # Find IDs within this range
            ids_in_range = sorted(id for id in ids if start <= id < end)
            
            if not ids_in_range:
                # No IDs to remove in this range
                new_ranges.append([start, end])
                continue
            
            # Split range around removed IDs
            current_start = start
            
            for id_to_remove in ids_in_range:
                if current_start < id_to_remove:
                    # Add segment before this ID
                    new_ranges.append([current_start, id_to_remove])
                
                # Skip the removed ID
                current_start = id_to_remove + 1
            
            # Add remaining segment after last removed ID
            if current_start < end:
                new_ranges.append([current_start, end])
        
        return RangeSet(new_ranges)
    
    def random_sample(self, n: int) -> List[int]:
        """Randomly sample n IDs from the ranges.
        
        Uses weighted random selection: probability of selecting from a range
        is proportional to its size.
        
        Args:
            n: Number of IDs to sample
            
        Returns:
            List of randomly selected IDs
            
        Raises:
            ValueError: If n > total available IDs
        """
        total_size = self.size()
        
        if n > total_size:
            raise ValueError(
                f"Cannot sample {n} IDs from RangeSet with only {total_size} IDs"
            )
        
        if n == 0:
            return []
        
        # Strategy: Use weighted random selection
        # Each range has weight = its size
        
        samples = []
        range_weights = [end - start for start, end in self.ranges]
        
        # Sample with replacement first (faster), then deduplicate
        # Over-sample to account for potential duplicates
        oversample_factor = 1.5 if n < total_size * 0.1 else 1.2
        target_samples = min(int(n * oversample_factor), total_size)
        
        attempts = 0
        max_attempts = 10
        
        while len(samples) < n and attempts < max_attempts:
            needed = n - len(samples)
            batch_size = min(int(needed * oversample_factor), total_size - len(samples))
            
            for _ in range(batch_size):
                # Choose a range weighted by size
                range_idx = random.choices(
                    range(len(self.ranges)),
                    weights=range_weights,
                    k=1
                )[0]
                
                start, end = self.ranges[range_idx]
                # Choose random ID within that range
                sample_id = random.randint(start, end - 1)
                samples.append(sample_id)
            
            # Deduplicate
            samples = list(set(samples))
            attempts += 1
        
        # Final fallback: if still not enough, use sequential sampling
        if len(samples) < n:
            # Convert to set for O(1) lookup
            samples_set = set(samples)
            
            # Iterate through ranges to fill remaining
            for start, end in self.ranges:
                if len(samples) >= n:
                    break
                
                for id in range(start, end):
                    if id not in samples_set:
                        samples.append(id)
                        samples_set.add(id)
                        
                        if len(samples) >= n:
                            break
        
        return samples[:n]
    
    def prioritized_sample(self, n: int) -> List[int]:
        """Sample n IDs, prioritizing later ranges (newest data first).

        Iterates ranges from last to first, randomly sampling from each
        segment until n IDs are collected. This ensures newly added
        range segments are sampled before older ones.

        Args:
            n: Number of IDs to sample

        Returns:
            List of randomly selected IDs

        Raises:
            ValueError: If n > total available IDs
        """
        total_size = self.size()

        if n > total_size:
            raise ValueError(
                f"Cannot sample {n} IDs from RangeSet with only {total_size} IDs"
            )

        if n == 0:
            return []

        samples: set = set()

        for start, end in reversed(self.ranges):
            if len(samples) >= n:
                break

            segment_size = end - start
            needed = n - len(samples)
            sample_count = min(needed, segment_size)

            if sample_count >= segment_size:
                # Take all IDs from this segment
                samples.update(range(start, end))
            else:
                # Randomly sample from this segment
                segment_samples: set = set()
                while len(segment_samples) < sample_count:
                    segment_samples.add(random.randint(start, end - 1))
                samples.update(segment_samples)

        return list(samples)[:n]

    def to_list(self) -> List[List[int]]:
        """Convert to list of [start, end) pairs.
        
        Returns:
            List of [start, end) intervals
        """
        return [[start, end] for start, end in self.ranges]
    
    def __contains__(self, item: int) -> bool:
        """Check if an ID is contained in any range using binary search.
        
        Args:
            item: Integer ID to check
            
        Returns:
            True if item is in any range
        """
        import bisect
        
        if not self.ranges:
            return False
        
        # Binary search for the rightmost range whose start <= item
        starts = [r[0] for r in self.ranges]
        idx = bisect.bisect_right(starts, item) - 1
        
        if idx < 0:
            return False
        
        start, end = self.ranges[idx]
        return start <= item < end
    
    def __len__(self) -> int:
        """Return the total number of IDs in all ranges."""
        return self.size()
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another RangeSet."""
        if not isinstance(other, RangeSet):
            return NotImplemented
        return self.ranges == other.ranges
    
    def intersection(self, other: 'RangeSet') -> 'RangeSet':
        """Compute the intersection of two RangeSets.
        
        Args:
            other: Another RangeSet
            
        Returns:
            New RangeSet containing only IDs present in both
        """
        result = []
        i, j = 0, 0
        
        while i < len(self.ranges) and j < len(other.ranges):
            a_start, a_end = self.ranges[i]
            b_start, b_end = other.ranges[j]
            
            # Find overlap
            overlap_start = max(a_start, b_start)
            overlap_end = min(a_end, b_end)
            
            if overlap_start < overlap_end:
                result.append([overlap_start, overlap_end])
            
            # Advance the range that ends first
            if a_end < b_end:
                i += 1
            else:
                j += 1
        
        return RangeSet(result)
    
    def union(self, other: 'RangeSet') -> 'RangeSet':
        """Compute the union of two RangeSets.
        
        Args:
            other: Another RangeSet
            
        Returns:
            New RangeSet containing IDs from either set
        """
        combined = self.to_list() + other.to_list()
        return RangeSet(combined)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"RangeSet({self.to_list()}, size={self.size()})"