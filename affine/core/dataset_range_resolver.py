"""
Dataset Range Resolver

Resolves dynamic dataset_range from remote metadata sources.
Environments can declare a `dataset_range_source` in their sampling_config
to fetch the range from a remote URL instead of hardcoding it.

Example config:
    "dataset_range_source": {
        "url": "https://example.com/metadata.json",
        "field": "tasks.completed_up_to",
        "range_type": "zero_to_value"
    }
"""

import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


def _extract_field(data: Dict[str, Any], field_path: str) -> Any:
    """Extract a value from nested dict using dot-notation path.

    Args:
        data: JSON-parsed dictionary
        field_path: Dot-separated path, e.g. "tasks.completed_up_to"

    Returns:
        The extracted value

    Raises:
        KeyError: If the path does not exist
    """
    current = data
    for key in field_path.split("."):
        current = current[key]
    return current


def _build_range(value: int, range_type: str) -> List[List[int]]:
    """Build dataset_range from extracted value and range_type.

    Supported range_types:
        - "zero_to_value": [[0, value - 1]]  (0-indexed inclusive range)

    Args:
        value: The extracted integer value
        range_type: How to interpret the value

    Returns:
        dataset_range in [[start, end], ...] format
    """
    if range_type == "zero_to_value":
        if value <= 0:
            return [[0, 0]]
        return [[0, value - 1]]

    raise ValueError(f"Unknown range_type: {range_type}")


def expand_dataset_range(
    old_range: List[List[int]],
    new_value: int,
    range_type: str = "zero_to_value",
) -> Optional[List[List[int]]]:
    """Expand dataset_range by appending the new portion as a separate segment.

    Instead of replacing the whole range, this preserves existing segments
    and appends only the delta. This allows rotation logic to distinguish
    newer data from older data via segment order.

    Example:
        old_range=[[0, 141]], new_value=200, range_type="zero_to_value"
        -> [[0, 141], [141, 199]]  (new segment [141, 199) appended)

    Args:
        old_range: Current dataset_range segments
        new_value: New value from remote metadata
        range_type: How to interpret the value

    Returns:
        Expanded range with new segment appended, or None if no expansion needed
    """
    if range_type == "zero_to_value":
        new_end = new_value - 1
        if new_end <= 0:
            return None

        if not old_range:
            return [[0, new_end]]

        # Find the max end across all existing segments
        current_max_end = max(seg[1] for seg in old_range)

        if new_end <= current_max_end:
            return None  # No expansion needed

        # Append delta as a new segment: [current_max_end, new_end)
        return old_range + [[current_max_end, new_end]]

    raise ValueError(f"Unknown range_type: {range_type}")


async def resolve_dataset_range_source(
    source: Dict[str, str],
    old_range: Optional[List[List[int]]] = None,
    timeout: float = 10.0,
) -> Optional[List[List[int]]]:
    """Resolve dataset_range from a remote metadata source.

    If old_range is provided, expands by appending a new segment for the
    delta (so rotation can prioritize newer data). If old_range is None,
    builds a fresh single-segment range.

    Args:
        source: Dict with keys: url, field, range_type
        old_range: Current dataset_range to expand (None for fresh build)
        timeout: HTTP request timeout in seconds

    Returns:
        Resolved dataset_range, or None if resolution fails / no change
    """
    url = source.get("url")
    field_path = source.get("field")
    range_type = source.get("range_type", "zero_to_value")

    if not url or not field_path:
        logger.error(f"dataset_range_source missing required keys (url, field): {source}")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    logger.error(
                        f"Failed to fetch dataset_range_source: "
                        f"HTTP {resp.status} from {url}"
                    )
                    return None
                data = await resp.json()

        value = _extract_field(data, field_path)
        value = int(value)

        if old_range:
            resolved_range = expand_dataset_range(old_range, value, range_type)
        else:
            resolved_range = _build_range(value, range_type)

        if resolved_range is not None:
            logger.info(
                f"Resolved dataset_range_source: {url} -> "
                f"{field_path}={value} -> range={resolved_range}"
            )
        return resolved_range

    except (aiohttp.ClientError, TimeoutError) as e:
        logger.error(f"HTTP error resolving dataset_range_source from {url}: {e}")
        return None
    except (KeyError, TypeError) as e:
        logger.error(f"Failed to extract field '{field_path}' from {url}: {e}")
        return None
    except (ValueError, OverflowError) as e:
        logger.error(f"Invalid value for dataset_range_source from {url}: {e}")
        return None
