"""
Authentication Service

Handles signature verification for executor authentication.
Uses Bittensor wallet signatures to verify that executors are authorized validators.
"""

import logging
import time
from typing import Optional, Tuple, List, Set
from dataclasses import dataclass
from bittensor import Keypair

from affine.core.setup import logger


@dataclass
class AuthConfig:
    """Configuration for authentication."""
    # List of validator hotkeys that are authorized to execute tasks
    # In production, this would be fetched from chain
    authorized_validators: Set[str]
    
    # Signature expiry time in seconds
    signature_expiry_seconds: int = 300  # 5 minutes
    
    # Whether to enforce strict signature validation
    strict_mode: bool = True


class AuthService:
    """
    Service for authenticating executors.
    
    Verifies that:
    1. The hotkey is a registered validator
    2. The signature is valid
    3. The request hasn't expired
    """
    
    def __init__(
        self,
        authorized_validators: Optional[Set[str]] = None,
        signature_expiry_seconds: int = 300,
        strict_mode: bool = True
    ):
        """
        Initialize AuthService.

        Args:
            authorized_validators: Set of authorized validator hotkeys
            signature_expiry_seconds: How long signatures are valid
            strict_mode: Whether to enforce strict validation
        """
        self.config = AuthConfig(
            authorized_validators=authorized_validators or set(),
            signature_expiry_seconds=signature_expiry_seconds,
            strict_mode=strict_mode
        )
    
    def is_validator(self, hotkey: str) -> bool:
        """
        Check if a hotkey is a registered validator.
        
        Args:
            hotkey: Hotkey to check
            
        Returns:
            True if hotkey is a validator
        """
        if not self.config.strict_mode:
            # In non-strict mode, allow all hotkeys
            return True
        
        return hotkey in self.config.authorized_validators
    
    def is_authorized_validator(self, hotkey: str) -> bool:
        """
        Check if a hotkey is an authorized validator (alias for is_validator).
        
        Args:
            hotkey: Hotkey to check
            
        Returns:
            True if hotkey is an authorized validator
        """
        return self.is_validator(hotkey)
    
    def verify_signature(
        self,
        message: str,
        signature: str,
        hotkey: str
    ) -> bool:
        """
        Verify a Bittensor wallet signature.
        
        Args:
            message: Original message that was signed
            signature: Signature to verify (hex string or bytes)
            hotkey: Expected signer's hotkey
            
        Returns:
            True if signature is valid
        """
        try:
            keypair = Keypair(ss58_address=hotkey)
            
            # Convert signature from hex if needed
            if isinstance(signature, str):
                # Remove 0x prefix if present
                signature = signature.replace("0x", "")
                # Convert hex string to bytes
                signature_bytes = bytes.fromhex(signature)
            else:
                signature_bytes = signature
            
            # Verify signature
            is_valid = keypair.verify(message.encode() if isinstance(message, str) else message, signature_bytes)
            
            if not is_valid:
                logger.warning(f"Invalid signature for hotkey {hotkey[:8]}...")
            
            return is_valid
        except Exception as e:
            logger.error(f"Error verifying signature: {signature}, error: {e}")
            return False
    
    def verify_request_signature(
        self,
        hotkey: str,
        timestamp: int,
        nonce: str,
        signature: str,
        additional_data: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Verify a complete request signature.
        
        The message format is: {hotkey}:{timestamp}:{nonce}[:additional_data]
        
        Args:
            hotkey: Executor's hotkey
            timestamp: Unix timestamp of the request
            nonce: Random nonce to prevent replay attacks
            signature: Signature of the message
            additional_data: Optional additional data included in signature
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if hotkey is authorized
        if not self.is_validator(hotkey):
            return False, f"Hotkey {hotkey[:8]}... is not an authorized validator"
        
        # Check timestamp expiry
        current_time = int(time.time())
        time_diff = abs(current_time - timestamp)
        
        if time_diff > self.config.signature_expiry_seconds:
            return False, f"Request timestamp expired (diff={time_diff}s, max={self.config.signature_expiry_seconds}s)"
        
        # Construct message
        if additional_data:
            message = f"{hotkey}:{timestamp}:{nonce}:{additional_data}"
        else:
            message = f"{hotkey}:{timestamp}:{nonce}"
        
        # Verify signature
        if not self.verify_signature(message, signature, hotkey):
            return False, "Invalid signature"
        
        return True, ""
    
    def add_validator(self, hotkey: str):
        """
        Add a validator to the authorized list.
        
        Args:
            hotkey: Validator's hotkey
        """
        self.config.authorized_validators.add(hotkey)
        logger.info(f"Added validator {hotkey[:8]}... to authorized list")
    
    def remove_validator(self, hotkey: str):
        """
        Remove a validator from the authorized list.
        
        Args:
            hotkey: Validator's hotkey
        """
        self.config.authorized_validators.discard(hotkey)
        logger.info(f"Removed validator {hotkey[:8]}... from authorized list")
    
    def update_validators(self, validators: List[str]):
        """
        Update the entire list of authorized validators.
        
        Args:
            validators: List of validator hotkeys
        """
        self.config.authorized_validators = set(validators)
        logger.info(f"Updated authorized validators: {len(validators)} validators")
    
    def generate_signing_message(
        self,
        hotkey: str,
        nonce: str,
        additional_data: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Generate the message that should be signed by the executor.
        
        Args:
            hotkey: Executor's hotkey
            nonce: Random nonce
            additional_data: Optional additional data
            
        Returns:
            Tuple of (message, timestamp)
        """
        timestamp = int(time.time())
        
        if additional_data:
            message = f"{hotkey}:{timestamp}:{nonce}:{additional_data}"
        else:
            message = f"{hotkey}:{timestamp}:{nonce}"
        
        return message, timestamp


async def create_auth_service_from_chain(netuid: int = 1) -> AuthService:
    """
    Create an AuthService with validators fetched from chain.
    
    Args:
        netuid: Subnet UID to fetch validators from
        
    Returns:
        Configured AuthService
    """
    try:
        from affine.utils.subtensor import get_subtensor
        
        subtensor = await get_subtensor()
        metagraph = await subtensor.metagraph(netuid)
        
        # Get validators (nodes with validator_permit)
        validators = set()
        for uid in range(len(metagraph.hotkeys)):
            if metagraph.validator_permit[uid]:
                validators.add(metagraph.hotkeys[uid])
        
        logger.info(f"Loaded {len(validators)} validators from chain (netuid={netuid})")
        
        return AuthService(authorized_validators=validators)
        
    except ImportError:
        logger.error("bittensor not installed, cannot fetch validators from chain")
        raise RuntimeError("Cannot create auth service: bittensor not installed")
    except Exception as e:
        logger.error(f"Error fetching validators from chain: {e}")
        raise RuntimeError(f"Cannot create auth service: {e}")