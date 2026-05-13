"""Resolve an env's ``dataset_range`` from a remote metadata URL.

Some envs (SWE-INFINITE, DISTILL) accumulate new task_ids over time as
new bugs / examples land in their public dataset. The dataset publishes
a small metadata JSON exposing the current top index; we fetch that
once per window refresh and produce a ``[[0, latest - 1]]`` range so
``mode='latest'`` picks the freshest task_ids.

Falls back to ``None`` on any failure — callers should keep the static
``dataset_range`` already in ``system_config`` when this returns ``None``.

Config shape on the env (under ``sampling``):

    "dataset_range_source": {
        "url": "https://example.com/metadata.json",
        "field": "tasks.completed_up_to",
        "range_type": "zero_to_value"
    }

Only ``range_type='zero_to_value'`` is supported today; if a future env
needs a different shape (e.g. ``offset_window``), add the case in
:func:`resolve_dataset_range` rather than letting callers handle the
fallback.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp

from affine.core.setup import logger


def _extract(data: Dict[str, Any], path: str) -> Any:
    """Read a value out of nested-dict JSON by dot-notation path."""
    cur = data
    for key in path.split("."):
        cur = cur[key]
    return cur


async def resolve_dataset_range(
    source: Dict[str, Any], *, timeout: float = 10.0,
) -> Optional[List[List[int]]]:
    """Fetch ``source.url``, read ``source.field`` (dot path), return
    ``[[0, value - 1]]``. Returns ``None`` on any failure so callers
    fall back to the static range already on the env config."""
    url = source.get("url")
    field = source.get("field")
    range_type = source.get("range_type", "zero_to_value")
    if not url or not field:
        return None
    if range_type != "zero_to_value":
        logger.error(
            f"dataset_range_source: unsupported range_type={range_type!r} "
            f"(only 'zero_to_value' is implemented)"
        )
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        f"dataset_range_source {url}: HTTP {resp.status}"
                    )
                    return None
                data = await resp.json(content_type=None)
        value = int(_extract(data, field))
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"dataset_range_source fetch failed for {url}: {e}")
        return None
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(
            f"dataset_range_source parse failed for {url} field={field!r}: {e}"
        )
        return None
    if value <= 0:
        return [[0, 0]]
    return [[0, value - 1]]
