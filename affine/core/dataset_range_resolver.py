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


async def resolve_dataset_range_source(
    source: Dict[str, str],
    timeout: float = 10.0,
) -> Optional[List[List[int]]]:
    """Resolve dataset_range from a remote metadata source.

    Args:
        source: Dict with keys: url, field, range_type
        timeout: HTTP request timeout in seconds

    Returns:
        Resolved dataset_range, or None if resolution fails
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
        resolved_range = _build_range(value, range_type)

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
