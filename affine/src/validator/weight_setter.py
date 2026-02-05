"""
Weight Setter

Handles weight processing and setting on chain.
"""

import bittensor as bt
from typing import List, Tuple, Dict
import numpy as np
import asyncio

from affine.core.setup import logger
from affine.utils.subtensor import get_subtensor

class WeightSetter:
    def __init__(self, wallet: bt.Wallet, netuid: int):
        self.wallet = wallet
        self.netuid = netuid

    async def process_weights(
        self,
        api_weights: Dict[str, Dict],
        burn_percentage: float = 0.0
    ) -> Tuple[List[int], List[float]]:
        """Process and normalize weights, applying burn and system miner weights.

        System miners (uid < 0) participate in scoring but don't receive actual
        rewards on chain. Their weights are accumulated and allocated to UID 0
        (validator) along with burn percentage.

        Args:
            api_weights: Dict mapping uid string to weight data
            burn_percentage: Percentage of total weight to burn (allocate to UID 0)

        Returns:
            Tuple of (uids, weights) for chain setting (only uid >= 0)
        """
        uids = []
        weights = []
        system_weight_total = 0.0  # Accumulate system miner weights

        # Parse weights, separating regular miners from system miners
        for uid_str, weight_data in api_weights.items():
            try:
                uid = int(uid_str)
                weight = float(weight_data.get("weight", 0.0))

                if weight <= 0:
                    continue

                if uid < 0:
                    # System miner: accumulate weight for UID 0
                    system_weight_total += weight
                else:
                    # Regular miner: add to chain weights
                    uids.append(uid)
                    weights.append(weight)
            except (ValueError, TypeError):
                continue

        if not uids and system_weight_total == 0:
            return [], []

        # Calculate total weight (including system miners) for normalization
        total_weight = sum(weights) + system_weight_total

        if total_weight == 0:
            return [], []

        # Normalize regular miner weights
        weights_array = np.array(weights, dtype=np.float64)
        weights_array = weights_array / total_weight

        # Normalize system weight
        normalized_system_weight = system_weight_total / total_weight

        # Calculate extra weight for UID 0 (burn + system miners)
        extra_weight = 0.0

        # Apply burn: scale all by (1 - burn%), add burn% to extra
        if burn_percentage > 0 and burn_percentage <= 1.0:
            weights_array *= (1.0 - burn_percentage)
            normalized_system_weight *= (1.0 - burn_percentage)
            extra_weight += burn_percentage

        # Add system miner weights to extra
        extra_weight += normalized_system_weight

        # Add extra weight to UID 0
        if extra_weight > 0:
            if 0 in uids:
                weights_array[uids.index(0)] += extra_weight
            else:
                uids = [0] + uids
                weights_array = np.concatenate([[extra_weight], weights_array])

        return uids, weights_array.tolist()

    async def set_weights(
        self,
        api_weights: Dict[str, Dict],
        burn_percentage: float = 0.0,
        max_retries: int = 3
    ) -> bool:
        """Set weights on chain with retry logic."""
        subtensor = await get_subtensor()
        uids, weights = await self.process_weights(api_weights, burn_percentage)
        
        if not uids:
            logger.warning("No valid weights to set")
            return False

        logger.info(f"Setting weights for {len(uids)} miners (burn={burn_percentage:.1%})")
        if 0 in uids:
            logger.info(f"  UID 0 (burn + system miners): {weights[uids.index(0)]:.6f}")

        # Print uid:weight mapping
        logger.info("Weights to be set:")
        for uid, weight in zip(uids, weights):
            logger.info(f"  UID {uid:3d}: {weight:.6f}")

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

            except Exception as e:
                logger.error(f"Error setting weights on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying after error in 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                else:
                    return False
        
        return False