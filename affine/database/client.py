"""
DynamoDB client management

Provides singleton client instance and connection pooling.
"""

import os
import aiobotocore.session
from botocore.config import Config

_client = None
_session = None


def get_region() -> str:
    """Get AWS region from environment."""
    return os.getenv("AWS_REGION", "us-east-1")


def get_table_prefix() -> str:
    """Get table name prefix from environment."""
    return os.getenv("DYNAMODB_TABLE_PREFIX", "affine")


async def init_client():
    """Initialize DynamoDB client.
    
    Creates a singleton client instance with connection pooling.
    """
    global _client, _session
    
    if _client is not None:
        return _client
    
    _session = aiobotocore.session.get_session()
    
    # Create client with connection pooling
    _client = await _session.create_client(
        'dynamodb',
        region_name=get_region(),
        config=Config(
            max_pool_connections=100,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
    ).__aenter__()
    
    return _client


async def close_client():
    """Close DynamoDB client."""
    global _client
    
    if _client is not None:
        await _client.__aexit__(None, None, None)
        _client = None


def get_client():
    """Get current DynamoDB client instance.
    
    Returns:
        DynamoDB client
        
    Raises:
        RuntimeError: If client not initialized
    """
    if _client is None:
        raise RuntimeError("DynamoDB client not initialized. Call init_client() first.")
    
    return _client