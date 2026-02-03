"""
Weight Setter

Handles weight processing and setting on chain.

This module provides functionality to process, normalize, and set miner weights
on the Bittensor blockchain. It includes support for burn percentage mechanisms
and robust error handling with retry logic.
"""

import bittensor as bt
from typing import List, Tuple, Dict, Optional
import numpy as np
import asyncio

from affine.core.setup import logger
from affine.utils.subtensor import get_subtensor
from affine.utils.errors import AffineError, NetworkError


class WeightProcessingError(AffineError):
    """Raised when weight processing fails (e.g., invalid data, normalization errors)."""
    pass


class WeightSettingError(AffineError):
    """Raised when weight setting on chain fails."""
    pass


class WeightSetter:
    """
    Handles weight processing and on-chain weight setting for validators.
    
    This class processes weights from the API, normalizes them, applies burn
    percentage if configured, and sets them on-chain with retry logic.
    
    Attributes:
        wallet: Bittensor wallet for signing transactions
        netuid: Network UID for the subnet
    """
    
    def __init__(self, wallet: bt.Wallet, netuid: int) -> None:
        """Initialize WeightSetter.
        
        Args:
            wallet: Bittensor wallet instance for signing transactions
            netuid: Network UID of the subnet
        """
        self.wallet = wallet
        self.netuid = netuid

    async def process_weights(
        self,
        api_weights: Dict[str, Dict],
        burn_percentage: float = 0.0
    ) -> Tuple[List[int], List[float]]:
        """
        Process and normalize weights, applying burn if specified.
        
        This method:
        1. Parses UIDs and weights from API response
        2. Filters out invalid entries (negative UIDs, zero/negative weights)
        3. Normalizes weights to sum to 1.0
        4. Applies burn percentage if configured (scales all weights, adds burn to UID 0)
        
        Args:
            api_weights: Dictionary mapping UID strings to weight data dicts.
                Expected format: {"uid": {"weight": float_value}}
            burn_percentage: Percentage of weights to burn (0.0-1.0). Burned weights
                are allocated to UID 0. Default: 0.0
        
        Returns:
            Tuple of (uids, weights) where:
            - uids: List of miner UIDs (integers)
            - weights: List of normalized weight values (floats, sum to 1.0)
        
        Raises:
            WeightProcessingError: If weight processing fails (e.g., all weights invalid,
                normalization error, invalid burn percentage)
        """
        uids: List[int] = []
        weights: List[float] = []
        
        # Validate burn_percentage
        if burn_percentage < 0.0 or burn_percentage > 1.0:
            raise WeightProcessingError(
                f"Invalid burn_percentage: {burn_percentage}. Must be between 0.0 and 1.0"
            )
        
        # Parse and filter valid weights
        for uid_str, weight_data in api_weights.items():
            try:
                uid = int(uid_str)
                weight = float(weight_data.get("weight", 0.0))
                if uid >= 0 and weight > 0:
                    uids.append(uid)
                    weights.append(weight)
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping invalid weight entry: uid={uid_str}, error={e}")
                continue
                
        if not uids:
            logger.warning("No valid weights found in API response")
            return [], []

        # Normalize to sum = 1.0
        try:
            weights_array = np.array(weights, dtype=np.float64)
            total = weights_array.sum()
            
            if total <= 0:
                raise WeightProcessingError("Sum of weights is zero or negative, cannot normalize")
            
            weights_array = weights_array / total
        except (ValueError, ZeroDivisionError, RuntimeError) as e:
            raise WeightProcessingError(f"Failed to normalize weights: {e}") from e
        
        # Apply burn: scale all by (1 - burn%), then UID 0 += burn%
        if burn_percentage > 0 and burn_percentage <= 1.0:
            weights_array *= (1.0 - burn_percentage)
            
            if 0 in uids:
                uid_0_index = uids.index(0)
                weights_array[uid_0_index] += burn_percentage
            else:
                uids = [0] + uids
                weights_array = np.concatenate([[burn_percentage], weights_array])
                
        return uids, weights_array.tolist()

    async def set_weights(
        self,
        api_weights: Dict[str, Dict],
        burn_percentage: float = 0.0,
        max_retries: int = 3
    ) -> bool:
        """
        Set weights on chain with retry logic.
        
        This method processes weights, validates them, and attempts to set them
        on-chain. It includes retry logic for transient failures and provides
        detailed logging throughout the process.
        
        Args:
            api_weights: Dictionary mapping UID strings to weight data dicts
            burn_percentage: Percentage of weights to burn (0.0-1.0). Default: 0.0
            max_retries: Maximum number of retry attempts. Default: 3
        
        Returns:
            True if weights were successfully set on-chain, False otherwise
        
        Raises:
            WeightProcessingError: If weight processing fails
            WeightSettingError: If all retry attempts fail
        """
        try:
            subtensor = await get_subtensor()
            uids, weights = await self.process_weights(api_weights, burn_percentage)
        except WeightProcessingError as e:
            logger.error(f"Failed to process weights: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during weight processing: {e}")
            raise WeightProcessingError(f"Unexpected error: {e}") from e
        
        if not uids:
            logger.warning("No valid weights to set")
            return False

        logger.info(f"Setting weights for {len(uids)} miners (burn={burn_percentage:.1%})")
        if burn_percentage > 0 and 0 in uids:
            uid_0_index = uids.index(0)
            logger.info(f"  UID 0 (burn): {weights[uid_0_index]:.6f}")
            
        # Print uid:weight mapping
        logger.info("Weights to be set:")
        for uid, weight in zip(uids, weights):
            logger.info(f"  UID {uid:3d}: {weight:.6f}")

        last_error: Optional[Exception] = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")
                
                current_block = await subtensor.get_current_block()
                logger.info(f"Current block: {current_block}")
                
                success = await subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.netuid,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=True,
                    wait_for_finalization=True,
                )
                
                if success:
                    logger.info("✅ Weights set successfully (chain confirmed)")
                    return True
                else:
                    logger.error(f"❌ Chain rejected weight setting on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        logger.info("Retrying weight setting in 60 seconds...")
                        await asyncio.sleep(60)
                        continue
                    else:
                        logger.error("❌ All attempts failed")
                        return False

            except (NetworkError, ConnectionError, TimeoutError) as e:
                last_error = e
                logger.error(f"Network error setting weights on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying after network error in 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                else:
                    logger.error("❌ All attempts failed due to network errors")
                    raise WeightSettingError(
                        f"Failed to set weights after {max_retries} attempts due to network errors"
                    ) from e
            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected error setting weights on attempt {attempt + 1}: {e}",
                    exc_info=True
                )
                if attempt < max_retries - 1:
                    logger.info("Retrying after error in 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                else:
                    logger.error("❌ All attempts failed")
                    raise WeightSettingError(
                        f"Failed to set weights after {max_retries} attempts: {e}"
                    ) from e
        
        # This should not be reached, but included for type safety
        if last_error:
            raise WeightSettingError(
                f"Failed to set weights after {max_retries} attempts"
            ) from last_error
        return False