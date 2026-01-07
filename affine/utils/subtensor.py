#!/usr/bin/env python3
"""
Subtensor wrapper with automatic reconnection on failure.
"""

import os
import asyncio
import logging
import threading
from typing import Optional, Any
import bittensor as bt
from affine.core.setup import logger

_affine_logger = logging.getLogger("affine")
if _affine_logger.level >= logging.CRITICAL:
    _root = logging.getLogger()
    if _root.handlers:
        _affine_logger.setLevel(_root.level)


class SubtensorWrapper:
    """
    Wrapper for bittensor async_subtensor with automatic reconnection on failure.
    """

    def __init__(self, endpoint: Optional[str] = None, fallback: Optional[str] = None):
        self._endpoint = endpoint or os.getenv("SUBTENSOR_ENDPOINT", "finney")
        self._fallback = fallback or os.getenv(
            "SUBTENSOR_FALLBACK", "wss://lite.sub.latent.to:443"
        )
        self._subtensor: Optional[bt.AsyncSubtensor] = None
        self._lock = asyncio.Lock()

    async def _create_connection(self) -> bt.AsyncSubtensor:
        """Create and initialize a new subtensor connection."""
        try:
            logger.debug(f"Attempting to connect to primary endpoint: {self._endpoint}")
            subtensor = bt.AsyncSubtensor(self._endpoint)
            await subtensor.initialize()
            logger.info(f"Successfully connected to primary endpoint: {self._endpoint}")
            return subtensor
        except Exception as e:
            logger.warning(
                f"Failed to connect to primary endpoint {self._endpoint}"
            )
            if self._fallback:
                logger.info(f"Attempting fallback connection to: {self._fallback}")
                try:
                    subtensor = bt.AsyncSubtensor(self._fallback)
                    await subtensor.initialize()
                    logger.info(f"Successfully connected to fallback: {self._fallback}")
                    return subtensor
                except Exception as fallback_error:
                    logger.error(
                        f"Failed to connect to fallback {self._fallback}: {fallback_error}"
                    )
                    raise
            raise

    async def ensure_connected(self):
        """Ensure we have a valid connection."""
        async with self._lock:
            if self._subtensor is None:
                self._subtensor = await self._create_connection()
            return self._subtensor

    def __getattr__(self, name: str) -> Any:
        """
        Proxy all attribute access to the underlying subtensor.
        Automatically reconnects on failure.
        """

        async def wrapper(*args, **kwargs):
            try:
                subtensor = await self.ensure_connected()
                method = getattr(subtensor, name)

                result = method(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                else:
                    return result
            except BaseException as e:
                logger.debug(f"Method {name} failed, attempting reconnection: {e}")

                async with self._lock:
                    if self._subtensor:
                        try:
                            await self._subtensor.close()
                        except:
                            pass
                        self._subtensor = None

                    self._subtensor = await self._create_connection()

                method = getattr(self._subtensor, name)
                result = method(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                else:
                    return result

        return wrapper

    async def close(self):
        """Close the connection."""
        async with self._lock:
            if self._subtensor:
                try:
                    await self._subtensor.close()
                except Exception as e:
                    logger.debug(f"Error closing subtensor: {e}")
                finally:
                    self._subtensor = None


# Global instance
_GLOBAL_SUBTENSOR: Optional[SubtensorWrapper] = None
_GLOBAL_LOCK = threading.Lock()


def get_global_subtensor() -> SubtensorWrapper:
    """
    Get or create the global SubtensorWrapper instance.

    Returns:
        SubtensorWrapper: The global subtensor wrapper instance.
    """
    global _GLOBAL_SUBTENSOR

    with _GLOBAL_LOCK:
        if _GLOBAL_SUBTENSOR is None:
            _GLOBAL_SUBTENSOR = SubtensorWrapper()
        return _GLOBAL_SUBTENSOR


async def get_subtensor() -> SubtensorWrapper:
    """
    Get the global SubtensorWrapper instance (async version).
    Ensures the connection is established before returning.

    Returns:
        SubtensorWrapper: The connected subtensor wrapper.
    """
    wrapper = get_global_subtensor()
    await wrapper.ensure_connected()
    return wrapper
