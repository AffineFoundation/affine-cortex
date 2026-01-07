#!/usr/bin/env python3
"""
Validator Service

Fetches weights from API and sets them on-chain using Bittensor.
"""

import asyncio
import os
import sys
import signal
import time
import click
import bittensor as bt
from typing import Dict, Optional

from affine.core.setup import logger, setup_logging
from affine.utils.api_client import create_api_client
from affine.utils.subtensor import get_subtensor
from affine.utils.errors import NetworkError
from affine.src.validator.weight_setter import WeightSetter


class ValidatorService:
    """
    Validator Service
    
    Core workflow:
    1. Wait for next weight submission window (every 180 blocks)
    2. Fetch latest weights from backend API
    3. Get burn percentage from API
    4. Set weights on chain using bittensor
    5. Verify weights were set successfully
    """
    
    def __init__(
        self,
        wallet_name: str,
        hotkey_name: str,
        netuid: int,
        network: str = "finney",
        watchdog_timeout: int = 600,
    ):
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.netuid = netuid
        self.network = network
        self.watchdog_timeout = watchdog_timeout
        
        # Load wallet
        try:
            self.wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
            logger.info(f"Wallet: {self.wallet}")
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Wallet file not found or invalid: {e}")
            raise RuntimeError(f"Failed to load wallet '{wallet_name}/{hotkey_name}': {e}") from e
        except Exception as e:
            logger.error(f"Failed to load wallet: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error loading wallet: {e}") from e

        self.api_client = None
        self.running = False
        self.weight_setter = WeightSetter(self.wallet, self.netuid)
        
        # Watchdog state
        self.last_block_update_time = time.time()
        self.last_block_number = None
        self.watchdog_task = None
        
    async def fetch_weights_from_api(self, max_retries: int = 12, retry_interval: int = 3) -> Optional[Dict]:
        """Fetch latest weights from backend API with retry logic
        
        Args:
            max_retries: Maximum number of retry attempts (default: 12)
            retry_interval: Seconds to wait between retries (default: 5)
        
        Returns:
            Weights data dict or None if all retries failed
        """
        if self.api_client is None:
            self.api_client = await create_api_client()
        
        for attempt in range(1, max_retries + 1):
            self.update_watchdog(f"fetching weights attempt {attempt}")
            try:
                response = await self.api_client.get("/scores/weights/latest")
                
                if not isinstance(response, dict) or not response.get("weights"):
                    logger.warning(f"Invalid or empty weights from API (attempt {attempt}/{max_retries})")
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_interval}s...")
                        self.update_watchdog("weights fetch retry wait")
                        await asyncio.sleep(retry_interval)
                        continue
                    return None
                
                weights_dict = response["weights"]
                block_number = response.get("block_number", "unknown")
                
                if attempt > 1:
                    logger.info(f"Successfully fetched weights on attempt {attempt}/{max_retries}")
                logger.info(f"Fetched {len(weights_dict)} weights (block={block_number})")
                return response
            
            except (NetworkError, ConnectionError, TimeoutError) as e:
                logger.error(f"Network error fetching weights (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_interval}s...")
                    self.update_watchdog("weights fetch network error retry wait")
                    await asyncio.sleep(retry_interval)
                else:
                    logger.error(f"Failed to fetch weights after {max_retries} attempts due to network errors")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error fetching weights (attempt {attempt}/{max_retries}): {e}", exc_info=True)
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_interval}s...")
                    self.update_watchdog("weights fetch error retry wait")
                    await asyncio.sleep(retry_interval)
                else:
                    logger.error(f"Failed to fetch weights after {max_retries} attempts")
                    return None
        
        return None

    async def fetch_config_from_api(self, max_retries: int = 12, retry_interval: int = 3) -> Optional[Dict]:
        """Fetch configuration from backend API with retry logic
        
        Args:
            max_retries: Maximum number of retry attempts (default: 12)
            retry_interval: Seconds to wait between retries (default: 3)
        
        Returns:
            Config data dict or None if all retries failed
        """
        if self.api_client is None:
            self.api_client = await create_api_client()
        
        for attempt in range(1, max_retries + 1):
            self.update_watchdog(f"fetching config attempt {attempt}")
            try:
                response = await self.api_client.get("/config")
                
                if not isinstance(response, dict) or not response.get("configs"):
                    logger.warning(f"Invalid or empty config from API (attempt {attempt}/{max_retries})")
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_interval}s...")
                        self.update_watchdog("config fetch retry wait")
                        await asyncio.sleep(retry_interval)
                        continue
                    return None
                
                configs = response["configs"]
                
                if attempt > 1:
                    logger.info(f"Successfully fetched config on attempt {attempt}/{max_retries}")
                logger.info(f"Fetched {len(configs)} config parameters")
                return configs
            
            except (NetworkError, ConnectionError, TimeoutError) as e:
                logger.error(f"Network error fetching config (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_interval}s...")
                    self.update_watchdog("config fetch network error retry wait")
                    await asyncio.sleep(retry_interval)
                else:
                    logger.error(f"Failed to fetch config after {max_retries} attempts due to network errors")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error fetching config (attempt {attempt}/{max_retries}): {e}", exc_info=True)
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_interval}s...")
                    self.update_watchdog("config fetch error retry wait")
                    await asyncio.sleep(retry_interval)
                else:
                    logger.error(f"Failed to fetch config after {max_retries} attempts")
                    return None
        
        return None

    async def watchdog_monitor(self):
        """Monitor block updates and restart if stuck"""
        logger.info(f"Watchdog started with timeout: {self.watchdog_timeout}s")
        
        while self.running:
            await asyncio.sleep(60)
            
            time_since_update = time.time() - self.last_block_update_time
            
            if time_since_update > self.watchdog_timeout:
                logger.error(
                    f"Watchdog timeout! No block update for {time_since_update:.0f}s "
                    f"(last block: {self.last_block_number})"
                )
                logger.error("Forcing restart...")
                os.kill(os.getpid(), signal.SIGTERM)
                await asyncio.sleep(5)
                sys.exit(1)
            
            logger.debug(
                f"Watchdog: {time_since_update:.0f}s since last block update "
                f"(block: {self.last_block_number})"
            )

    def update_block_progress(self, block_number: int):
        """Update watchdog state when block progresses"""
        if self.last_block_number != block_number:
            self.last_block_update_time = time.time()
            self.last_block_number = block_number
            logger.debug(f"Block progress updated: {block_number}")
    
    def update_watchdog(self, operation: str = ""):
        """Update watchdog timestamp to indicate ongoing activity"""
        self.last_block_update_time = time.time()
        if operation:
            logger.debug(f"Watchdog updated: {operation}")

    async def wait_for_next_window(self, subtensor, interval_blocks: int):
        """Wait for the next weight submission window"""
        current_block = await subtensor.get_current_block()
        self.update_block_progress(current_block)
        
        # Calculate next window
        current_epoch = current_block // interval_blocks
        next_window_start = (current_epoch + 1) * interval_blocks
        blocks_remaining = next_window_start - current_block
        
        if blocks_remaining <= 0:
            logger.info(f"In submission window at block {current_block}")
            return current_block
        
        logger.info(f"Waiting for block {next_window_start} ({blocks_remaining} blocks remaining)")
        
        # Wait block by block until we reach the target
        while self.running and current_block < next_window_start:
            try:
                logger.info(f"Current block: {current_block}, target: {next_window_start}, remaining: {next_window_start - current_block}")
                
                # Wait for next block (watchdog will handle timeout)
                await subtensor.wait_for_block(current_block + 1)
                
                # Update current block and watchdog
                current_block = await subtensor.get_current_block()
                self.update_block_progress(current_block)
                
            except Exception as e:
                logger.error(f"Error in wait_for_next_window: {e}")
                await asyncio.sleep(30)
                # Re-fetch current block after error
                current_block = await subtensor.get_current_block()
                self.update_block_progress(current_block)
        
        return next_window_start

    async def run_iteration(self):
        """Run one iteration of weight setting"""
        # 1. Fetch weights
        self.update_watchdog("fetching weights")
        weights_data = await self.fetch_weights_from_api()
        if not weights_data:
            return

        # 2. Get config from API
        self.update_watchdog("fetching config")
        config = await self.fetch_config_from_api()
        if not config:
            logger.warning("Failed to fetch config, using default burn_percentage=0.0")
            burn_percentage = 0.0
        else:
            burn_percentage = config.get("validator_burn_percentage", 0.0)
            burn_percentage = float(burn_percentage)

        # 3. Set weights using WeightSetter with timeout
        self.update_watchdog("setting weights")
        try:
            # Timeout set to 8 minutes (less than 10min watchdog timeout)
            await asyncio.wait_for(
                self.weight_setter.set_weights(
                    weights_data.get("weights", {}),
                    burn_percentage
                ),
                timeout=480
            )
            self.update_watchdog("weights set completed")
        except asyncio.TimeoutError:
            logger.error("set_weights timed out after 480s")
            self.update_watchdog("weights set timeout")

    async def start(self):
        """Start the validator service"""
        logger.info("Starting ValidatorService...")
        self.running = True
        
        # Start watchdog
        self.watchdog_task = asyncio.create_task(self.watchdog_monitor())
        
        try:
            while self.running:
                try:
                    interval_blocks = int(os.getenv("WEIGHT_SET_INTERVAL_BLOCKS", "180"))
                    subtensor = await get_subtensor()
                    await self.wait_for_next_window(subtensor, interval_blocks)
                    
                    await self.run_iteration()
                    
                    # Sleep a bit to avoid tight loop if something goes wrong with window calculation
                    await asyncio.sleep(10)

                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    self.update_watchdog("main loop error recovery")
                    await asyncio.sleep(60)
        finally:
            if self.watchdog_task:
                self.watchdog_task.cancel()
                try:
                    await self.watchdog_task
                except asyncio.CancelledError:
                    pass

    async def stop(self):
        self.running = False
        logger.info("Stopping ValidatorService...")
        if self.watchdog_task:
            self.watchdog_task.cancel()


@click.command()
@click.option("--netuid", type=int, default=120)
@click.option("--wallet-name", type=str, default=os.getenv("BT_WALLET_COLD"))
@click.option("--hotkey-name", type=str, default=os.getenv("BT_WALLET_HOT"))
@click.option("--network", type=str, default="finney")
@click.option("--verbosity", type=str, default="1")
@click.option("--watchdog-timeout", type=int, default=600, help="Watchdog timeout in seconds (default: 600s/10min)")
def main(netuid, wallet_name, hotkey_name, network, verbosity, watchdog_timeout):
    setup_logging(int(verbosity))
    
    if not wallet_name or not hotkey_name:
        logger.error("Wallet name and hotkey name are required")
        return

    async def run_service():
        service = ValidatorService(
            wallet_name,
            hotkey_name,
            netuid,
            network,
            watchdog_timeout=watchdog_timeout
        )
        task = asyncio.create_task(service.start())
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_service())

if __name__ == "__main__":
    main()