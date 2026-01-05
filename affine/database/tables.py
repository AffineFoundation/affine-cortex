"""
Table initialization and management

Handles table creation and verification.
"""

import asyncio
from typing import List
from affine.database.client import get_client
from affine.database.schema import ALL_SCHEMAS


async def table_exists(table_name: str) -> bool:
    """Check if table exists.
    
    Args:
        table_name: Name of the table
        
    Returns:
        True if table exists, False otherwise
    """
    client = get_client()
    
    try:
        await client.describe_table(TableName=table_name)
        return True
    except client.exceptions.ResourceNotFoundException:
        return False


async def enable_ttl(table_name: str, ttl_attribute: str):
    """Enable TTL on a table.
    
    Args:
        table_name: Name of the table
        ttl_attribute: Name of the TTL attribute
    """
    client = get_client()
    
    try:
        await client.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={
                'Enabled': True,
                'AttributeName': ttl_attribute
            }
        )
        print(f"Enabled TTL on {table_name} (attribute: {ttl_attribute})")
    except Exception as e:
        print(f"Warning: Failed to enable TTL on {table_name}: {e}")


async def create_table(schema: dict, ttl_attribute: str = None):
    """Create a DynamoDB table from schema.
    
    Args:
        schema: Table schema definition
        ttl_attribute: Optional TTL attribute name to enable after creation
    """
    client = get_client()
    table_name = schema["TableName"]
    
    if await table_exists(table_name):
        print(f"Table {table_name} already exists, skipping creation")
        return
    
    print(f"Creating table {table_name}...")
    
    await client.create_table(**schema)
    
    # Wait for table to be active
    waiter = client.get_waiter('table_exists')
    await waiter.wait(TableName=table_name)
    
    print(f"Table {table_name} created successfully")
    
    # Enable TTL if specified
    if ttl_attribute:
        await enable_ttl(table_name, ttl_attribute)


async def init_tables():
    """Initialize all DynamoDB tables.
    
    Creates all tables defined in schemas if they don't exist.
    """
    from affine.database.schema import (
        SAMPLE_RESULTS_SCHEMA,
        TASK_QUEUE_SCHEMA,
        EXECUTION_LOGS_SCHEMA, EXECUTION_LOGS_TTL,
        SCORES_SCHEMA, SCORES_TTL,
        SYSTEM_CONFIG_SCHEMA,
        MINERS_SCHEMA,
        SCORE_SNAPSHOTS_SCHEMA, SCORE_SNAPSHOTS_TTL,
        MINER_STATS_SCHEMA,
    )
    
    print("Initializing DynamoDB tables...")
    
    # Create tables with TTL configuration
    await asyncio.gather(
        create_table(SAMPLE_RESULTS_SCHEMA),
        create_table(TASK_QUEUE_SCHEMA),
        create_table(EXECUTION_LOGS_SCHEMA, ttl_attribute=EXECUTION_LOGS_TTL["AttributeName"]),
        create_table(SCORES_SCHEMA, ttl_attribute=SCORES_TTL["AttributeName"]),
        create_table(SYSTEM_CONFIG_SCHEMA),
        create_table(MINERS_SCHEMA),
        create_table(SCORE_SNAPSHOTS_SCHEMA, ttl_attribute=SCORE_SNAPSHOTS_TTL["AttributeName"]),
        create_table(MINER_STATS_SCHEMA),
    )
    
    print("All tables initialized successfully")


async def list_tables() -> List[str]:
    """List all tables with the configured prefix.
    
    Returns:
        List of table names
    """
    from affine.database.client import get_table_prefix
    
    client = get_client()
    prefix = get_table_prefix()
    
    response = await client.list_tables()
    all_tables = response.get('TableNames', [])
    
    # Filter tables with our prefix
    return [t for t in all_tables if t.startswith(prefix)]


async def delete_table(table_name: str):
    """Delete a table.
    
    WARNING: This permanently deletes all data in the table.
    
    Args:
        table_name: Name of the table to delete
    """
    client = get_client()
    
    if not await table_exists(table_name):
        print(f"Table {table_name} does not exist")
        return
    
    print(f"Deleting table {table_name}...")
    
    await client.delete_table(TableName=table_name)
    
    # Wait for table to be deleted
    waiter = client.get_waiter('table_not_exists')
    await waiter.wait(TableName=table_name)
    
    print(f"Table {table_name} deleted successfully")


async def reset_tables():
    """Delete and recreate all tables.
    
    WARNING: This permanently deletes all data.
    """
    print("Resetting all tables...")
    
    # Delete all tables
    tables = await list_tables()
    delete_tasks = [delete_table(t) for t in tables]
    await asyncio.gather(*delete_tasks)
    
    # Recreate tables
    await init_tables()
    
    print("All tables reset successfully")