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


def _patch_bittensor_commit_decoder() -> None:
    """Fix bittensor 10.x revealed-commitment decoding.

    Why: bittensor's decode_revealed_commitment assumes com_hex is a 0x-hex
    string, but the chain returns a raw str whose codepoints are the bytes
    (e.g. '5\\x02{"model": ...}'). bytes.fromhex() then fails on every
    commit, so the entire batch (all miners) never decodes.

    Re-implement the decoder by treating the str as latin-1 bytes and
    stripping the SCALE compact-length prefix, matching the pre-10.x behavior.
    """
    try:
        from bittensor.core.chain_data import utils as _bt_utils
        from bittensor.core import async_subtensor as _bt_async
    except Exception as e:
        logger.debug(f"Skipping bittensor commit-decoder patch: {e}")
        return

    if getattr(_bt_utils, "_affine_safe_decode_patched", False):
        return

    def _scale_offset(first_byte: int) -> int:
        mode = first_byte & 0b11
        if mode == 0:
            return 1
        if mode == 1:
            return 2
        return 4

    def _to_bytes(com) -> bytes:
        if isinstance(com, (bytes, bytearray)):
            return bytes(com)
        if isinstance(com, str):
            stripped = com.removeprefix("0x")
            try:
                return bytes.fromhex(stripped)
            except ValueError:
                return com.encode("latin-1", errors="replace")
        return bytes(com)

    def _safe_decode(encoded_data):
        com, revealed_block = encoded_data
        com_bytes = _to_bytes(com)
        offset = _scale_offset(com_bytes[0]) if com_bytes else 0
        return revealed_block, com_bytes[offset:].decode("utf-8", errors="ignore")

    def _safe_decode_with_hotkey(encoded_data):
        key, data = encoded_data
        ss58_address = key
        decoded = []
        for p in data:
            try:
                decoded.append(_safe_decode(p))
            except Exception as exc:
                logger.warning(
                    f"Skipping malformed revealed commit for {ss58_address}: {exc}"
                )
        return ss58_address, tuple(decoded)

    _bt_utils.decode_revealed_commitment = _safe_decode
    _bt_utils.decode_revealed_commitment_with_hotkey = _safe_decode_with_hotkey
    _bt_async.decode_revealed_commitment_with_hotkey = _safe_decode_with_hotkey
    _bt_utils._affine_safe_decode_patched = True


_patch_bittensor_commit_decoder()


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
