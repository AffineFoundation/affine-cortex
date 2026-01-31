"""
Database CLI tool for management and testing

Provides commands for initializing, testing, and managing DynamoDB tables.
"""

import asyncio
import sys
from typing import Optional
import os
import click

from affine.database import init_client, close_client, init_tables
from affine.database.tables import list_tables, reset_tables, delete_table
from affine.database.dao import (
    SampleResultsDAO,
    TaskPoolDAO,
    ExecutionLogsDAO,
    ScoresDAO,
    SystemConfigDAO,
    MinersDAO,
)


async def cmd_init():
    """Initialize all DynamoDB tables."""
    print("Initializing DynamoDB tables...")
    await init_client()
    
    try:
        await init_tables()
        print("✓ Tables initialized successfully")
    finally:
        await close_client()


async def cmd_list():
    """List all tables."""
    await init_client()
    
    try:
        tables = await list_tables()
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
    finally:
        await close_client()


async def cmd_reset():
    """Reset all tables (delete and recreate)."""
    confirm = input("WARNING: This will delete all data. Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted")
        return
    
    await init_client()
    
    try:
        await reset_tables()
        print("✓ Tables reset successfully")
    finally:
        await close_client()


async def cmd_reset_table(table_name: str):
    """Reset a single table (delete and recreate)."""
    from affine.database.schema import get_table_name
    
    # Get full table name with environment prefix
    full_table_name = get_table_name(table_name)
    
    confirm = input(f"WARNING: This will delete all data in '{full_table_name}'. Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted")
        return
    
    await init_client()
    
    try:
        print(f"Deleting table '{full_table_name}'...")
        await delete_table(full_table_name)
        
        print(f"Recreating table '{full_table_name}'...")
        await init_tables()
        
        print(f"✓ Table '{full_table_name}' reset successfully")
    except Exception as e:
        print(f"✗ Failed to reset table: {e}")
        sys.exit(1)
    finally:
        await close_client()


async def cmd_migrate(tail: int, max_results: Optional[int]):
    """Run migration from R2 to DynamoDB."""
    from affine.database.migrate import run_migration
    
    print(f"Starting migration (tail={tail}, max_results={max_results or 'all'})")
    await run_migration(tail_blocks=tail, max_results=max_results)


async def cmd_load_config(json_file: str):
    """Load system configuration from JSON file.
    
    Supports smooth transition with optional initial_range:
    - If initial_range exists: Use it to initialize sampling_list
    - If initial_range missing: Keep existing sampling_list in database (no override)
    """
    import json
    import os
    import time
    
    print(f"Loading configuration from {json_file}...")
    
    # Check file exists
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    
    # Load JSON
    try:
        with open(json_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)
    
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        # Load validator burn percentage if present
        if 'validator_burn_percentage' in config:
            burn_percentage = float(config['validator_burn_percentage'])
            
            await config_dao.set_param(
                param_name='validator_burn_percentage',
                param_value=burn_percentage,
                param_type='float',
                description='Percentage of weight to burn (allocate to UID 0)',
                updated_by='cli_load_config'
            )
            
            print(f"✓ Loaded validator burn percentage: {burn_percentage:.1%}")
        
        # Load environments configuration
        if 'environments' in config:
            from affine.core.sampling_list import SamplingListManager
            import random
            
            environments = config['environments']
            existing_envs = await config_dao.get_param_value('environments', default={})
            manager = SamplingListManager()
            
            from affine.core.dataset_range_resolver import resolve_dataset_range_source

            for env_name, env_config in environments.items():
                sampling_config = env_config.get('sampling_config')
                if not sampling_config:
                    print(f"  Warning: {env_name} missing sampling_config, skipping")
                    continue

                # Resolve dynamic dataset_range from remote source if configured
                range_source = sampling_config.get('dataset_range_source')
                if range_source:
                    resolved_range = await resolve_dataset_range_source(range_source)
                    if resolved_range is not None:
                        sampling_config['dataset_range'] = resolved_range
                        print(f"  {env_name}: Resolved dataset_range from remote source: {resolved_range}")
                    else:
                        print(f"  {env_name}: Failed to resolve dataset_range_source, using fallback: {sampling_config.get('dataset_range')}")

                # Check if initial_range is provided
                if 'initial_range' in sampling_config:
                    # Use initial_range to generate sampling_list
                    initial_range = sampling_config['initial_range']
                    sampling_count = sampling_config.get('sampling_count', 0)
                    
                    # Generate sampling_list from initial_range
                    sampling_list = await manager.initialize_sampling_list(
                        env=env_name,
                        initial_range=initial_range,
                        sampling_size=sampling_count
                    )
                    
                    sampling_config['sampling_list'] = sampling_list
                    sampling_config['last_rotation_at'] = int(time.time())
                    
                    # Remove initial_range after use (keep config clean)
                    del sampling_config['initial_range']
                    
                    print(f"  {env_name}: Initialized sampling_list from initial_range (size={len(sampling_list)})")
                
                else:
                    # No initial_range: preserve existing sampling_list from database
                    existing_env = existing_envs.get(env_name, {})
                    existing_config = existing_env.get('sampling_config', {})
                    existing_list = existing_config.get('sampling_list')
                    
                    if existing_list:
                        sampling_config['sampling_list'] = existing_list
                        sampling_config['last_rotation_at'] = existing_config.get('last_rotation_at', int(time.time()))
                        print(f"  {env_name}: Preserved existing sampling_list from database (size={len(existing_list)})")
                    else:
                        # No existing list: generate from dataset_range using RangeSet-based manager
                        dataset_range = sampling_config.get('dataset_range', [[0, 0]])
                        sampling_count = sampling_config.get('sampling_count', 0)
                        
                        # Use SamplingListManager which uses RangeSet for efficient handling
                        sampling_list = await manager.initialize_sampling_list(
                            env=env_name,
                            initial_range=dataset_range,
                            sampling_size=sampling_count
                        )

                        sampling_config['sampling_list'] = sampling_list
                        sampling_config['last_rotation_at'] = int(time.time())
                        
                        print(f"  {env_name}: Generated new sampling_list from dataset_range (size={len(sampling_list)})")
            
            # Save to database
            await config_dao.set_param(
                param_name='environments',
                param_value=environments,
                param_type='dict',
                description='Environment configurations with dynamic sampling',
                updated_by='cli_load_config'
            )
            
            print(f"\n✓ Loaded configuration for {len(environments)} environments:")
            
            for env_name, env_config in environments.items():
                enabled_sampling = env_config.get('enabled_for_sampling', False)
                enabled_scoring = env_config.get('enabled_for_scoring', False)
                
                sampling_config = env_config.get('sampling_config')
                if sampling_config:
                    sampling_list = sampling_config.get('sampling_list', [])
                    rotation_count = sampling_config.get('rotation_count', 0)
                    status = f"sampling_list={len(sampling_list)} tasks"
                    if rotation_count > 0:
                        status += f", rotation={rotation_count} tasks/hour"
                    else:
                        status += ", rotation=disabled"
                else:
                    status = "no sampling_config"
                
                flags = []
                if enabled_sampling:
                    flags.append("sampling")
                if enabled_scoring:
                    flags.append("scoring")
                flags_str = "+".join(flags) if flags else "disabled"
                
                print(f"  {env_name} [{flags_str}]: {status}")
        
        print("\n✓ Configuration loaded successfully!")
        
    finally:
        await close_client()


async def cmd_blacklist_list():
    """List all blacklisted hotkeys."""
    print("Fetching blacklist...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        blacklist = await config_dao.get_blacklist()
        
        if not blacklist:
            print("Blacklist is empty")
        else:
            print(f"Blacklist contains {len(blacklist)} hotkey(s):")
            for i, hotkey in enumerate(blacklist, 1):
                print(f"  {i}. {hotkey}")
    
    finally:
        await close_client()


async def cmd_blacklist_add(hotkeys: list):
    """Add hotkeys to blacklist."""
    print(f"Adding {len(hotkeys)} hotkey(s) to blacklist...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        # Get current blacklist
        current = await config_dao.get_blacklist()
        print(f"Current blacklist size: {len(current)}")
        
        # Add new hotkeys
        result = await config_dao.add_to_blacklist(
            hotkeys=hotkeys,
            updated_by='cli_blacklist_add'
        )
        
        new_blacklist = result.get('param_value', [])
        print(f"✓ Updated blacklist size: {len(new_blacklist)}")
        print(f"  Added: {len(new_blacklist) - len(current)} new hotkey(s)")
        
    finally:
        await close_client()


async def cmd_blacklist_remove(hotkeys: list):
    """Remove hotkeys from blacklist."""
    print(f"Removing {len(hotkeys)} hotkey(s) from blacklist...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        # Get current blacklist
        current = await config_dao.get_blacklist()
        print(f"Current blacklist size: {len(current)}")
        
        # Remove hotkeys
        result = await config_dao.remove_from_blacklist(
            hotkeys=hotkeys,
            updated_by='cli_blacklist_remove'
        )
        
        new_blacklist = result.get('param_value', [])
        print(f"✓ Updated blacklist size: {len(new_blacklist)}")
        print(f"  Removed: {len(current) - len(new_blacklist)} hotkey(s)")
        
    finally:
        await close_client()


async def cmd_blacklist_clear():
    """Clear all hotkeys from blacklist."""
    confirm = input("WARNING: This will clear the entire blacklist. Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted")
        return
    
    print("Clearing blacklist...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        result = await config_dao.set_blacklist(
            hotkeys=[],
            updated_by='cli_blacklist_clear'
        )
        
        print("✓ Blacklist cleared successfully")
        
    finally:
        await close_client()


async def cmd_set_burn_percentage(burn_percentage: float):
    """Set validator burn percentage."""
    if burn_percentage < 0 or burn_percentage > 1:
        print(f"Error: Burn percentage must be between 0 and 1 (got {burn_percentage})")
        sys.exit(1)
    
    print(f"Setting burn percentage to {burn_percentage:.1%}...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        result = await config_dao.set_param(
            param_name='validator_burn_percentage',
            param_value=burn_percentage,
            param_type='float',
            description='Percentage of weight to burn (allocate to UID 0)',
            updated_by='cli_set_burn_percentage'
        )
        
        print(f"✓ Burn percentage set to {burn_percentage:.1%}")
        
    finally:
        await close_client()


async def cmd_get_burn_percentage():
    """Get current validator burn percentage."""
    print("Fetching burn percentage...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        config = await config_dao.get_param('validator_burn_percentage')
        
        if not config:
            print("Burn percentage not set (default: 0.0)")
        else:
            burn_percentage = config.get('param_value', 0.0)
            print(f"Current burn percentage: {burn_percentage:.1%}")
            print(f"Last updated: {config.get('updated_at', 'unknown')}")
            print(f"Updated by: {config.get('updated_by', 'unknown')}")
    
    finally:
        await close_client()


async def cmd_get_config():
    """Get and print current system configuration."""
    import json
    
    print("Fetching system configuration...\n")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        # Fetch all configuration parameters
        burn_config = await config_dao.get_param('validator_burn_percentage')
        environments = await config_dao.get_param_value('environments', default={})
        blacklist = await config_dao.get_blacklist()
        
        # Print burn percentage
        print("=" * 80)
        print("VALIDATOR BURN PERCENTAGE")
        print("=" * 80)
        if burn_config:
            burn_percentage = burn_config.get('param_value', 0.0)
            print(f"Value: {burn_percentage:.1%}")
            print(f"Updated: {burn_config.get('updated_at', 'unknown')}")
            print(f"Updated by: {burn_config.get('updated_by', 'unknown')}")
        else:
            print("Not set (default: 0.0)")
        
        # Print blacklist
        print("\n" + "=" * 80)
        print("BLACKLIST")
        print("=" * 80)
        if blacklist:
            print(f"Count: {len(blacklist)} hotkey(s)")
            for i, hotkey in enumerate(blacklist, 1):
                print(f"  {i}. {hotkey}")
        else:
            print("Empty")
        
        # Print environments configuration
        print("\n" + "=" * 80)
        print("ENVIRONMENTS CONFIGURATION")
        print("=" * 80)
        if not environments:
            print("No environments configured")
        else:
            print(f"Total environments: {len(environments)}\n")
            
            for env_name, env_config in environments.items():
                print(f"{'─' * 80}")
                print(f"Environment: {env_name}")
                print(f"{'─' * 80}")
                
                # Status flags
                enabled_sampling = env_config.get('enabled_for_sampling', False)
                enabled_scoring = env_config.get('enabled_for_scoring', False)
                flags = []
                if enabled_sampling:
                    flags.append("sampling")
                if enabled_scoring:
                    flags.append("scoring")
                status = "+".join(flags) if flags else "disabled"
                print(f"Status: [{status}]")
                
                # Sampling configuration
                sampling_config = env_config.get('sampling_config')
                if sampling_config:
                    print(f"\nSampling Configuration:")
                    
                    # Dataset range
                    dataset_range = sampling_config.get('dataset_range', [])
                    print(f"  Dataset range: {dataset_range}")
                    
                    # Sampling count
                    sampling_count = sampling_config.get('sampling_count', 0)
                    print(f"  Sampling count: {sampling_count}")
                    
                    # Sampling list
                    sampling_list = sampling_config.get('sampling_list', [])
                    print(f"  Sampling list: {len(sampling_list)} tasks")
                    if sampling_list:
                        # Show first and last few items
                        if len(sampling_list) <= 10:
                            print(f"    Tasks: {sampling_list}")
                        else:
                            preview = sampling_list[:5] + ["..."] + sampling_list[-5:]
                            print(f"    Tasks: {preview}")
                    
                    # Rotation settings
                    rotation_enabled = sampling_config.get('rotation_enabled', False)
                    rotation_count = sampling_config.get('rotation_count', 0)
                    rotation_interval = sampling_config.get('rotation_interval', 3600)
                    
                    print(f"  Rotation enabled: {rotation_enabled}")
                    if rotation_enabled:
                        print(f"  Rotation count: {rotation_count} tasks/rotation")
                        print(f"  Rotation interval: {rotation_interval}s ({rotation_interval/3600:.1f} hours)")
                    
                    # Last rotation
                    last_rotation = sampling_config.get('last_rotation_at')
                    if last_rotation:
                        import time
                        elapsed = int(time.time()) - last_rotation
                        print(f"  Last rotation: {elapsed}s ago ({elapsed/3600:.1f} hours)")
                
                # Scoring configuration
                scoring_config = env_config.get('scoring_config')
                if scoring_config:
                    print(f"\nScoring Configuration:")
                    weights = scoring_config.get('weights', {})
                    print(f"  Weights: {json.dumps(weights, indent=4)}")
                
                print()  # Blank line between environments
        
        print("=" * 80)
        print("✓ Configuration printed successfully")
        
    finally:
        await close_client()


async def cmd_delete_samples_by_range(
    hotkey: Optional[str],
    revision: Optional[str],
    env: str,
    start_task_id: int,
    end_task_id: int
):
    """Delete samples within a task_id range.
    
    If hotkey and revision are provided, deletes samples for that specific miner.
    If they are not provided, deletes all samples in the environment and range.
    """
    if hotkey and revision:
        print(f"Deleting samples for hotkey={hotkey[:12]}..., revision={revision[:8]}..., env={env}, task_id range=[{start_task_id}, {end_task_id})...")
        confirm = input(f"WARNING: This will delete samples for specific miner in range [{start_task_id}, {end_task_id}). Type 'yes' to confirm: ")
    else:
        print(f"Deleting ALL samples for env={env}, task_id range=[{start_task_id}, {end_task_id})...")
        confirm = input(f"WARNING: This will delete ALL samples across all miners/revisions for env={env} in range [{start_task_id}, {end_task_id}). Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted")
        return
    
    await init_client()
    
    try:
        sample_dao = SampleResultsDAO()
        
        if hotkey and revision:
            # Delete for specific miner
            deleted_count = await sample_dao.delete_samples_by_task_range(
                miner_hotkey=hotkey,
                model_revision=revision,
                env=env,
                start_task_id=start_task_id,
                end_task_id=end_task_id
            )
        else:
            # Delete for all miners in the environment
            deleted_count = await sample_dao.delete_all_samples_by_task_range(
                env=env,
                start_task_id=start_task_id,
                end_task_id=end_task_id
            )
        
        print(f"✓ Deleted {deleted_count} samples")
    
    except Exception as e:
        print(f"✗ Failed to delete samples: {e}")
        sys.exit(1)
    finally:
        await close_client()


async def cmd_delete_samples_empty_conversation():
    """Delete all samples with empty conversation across the entire database."""
    print("Scanning entire sample database for invalid samples (empty conversation)...")
    
    confirm = input("WARNING: This will scan and delete ALL samples with empty conversation in the database. Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted")
        return
    
    await init_client()
    
    try:
        sample_dao = SampleResultsDAO()
        deleted_count = await sample_dao.delete_all_samples_with_empty_conversation()
        
        print(f"\n✓ Scan complete. Deleted {deleted_count} samples with empty conversation")
    
    except Exception as e:
        print(f"\n✗ Failed to delete samples: {e}")
        sys.exit(1)
    finally:
        await close_client()


async def cmd_cleanup_inactive_miners(days: int):
    """Cleanup long-inactive miners.
    
    Finds miners that haven't been updated for the specified number of days
    and have never had any weight (best_weight == 0).
    """
    from affine.database.dao.miner_stats import MinerStatsDAO
    from datetime import datetime
    
    await init_client()
    
    try:
        dao = MinerStatsDAO()
        
        # Get inactive miners (dry-run)
        inactive_miners = await dao.cleanup_inactive_miners(
            inactive_days=days,
            dry_run=True
        )
        
        if not inactive_miners:
            print("No inactive miners found.")
            return
        
        # Display results
        print(f"\nFound {len(inactive_miners)} inactive miners (>{days} days, zero weight):\n")
        
        for i, miner in enumerate(inactive_miners[:10], 1):
            last_updated = datetime.fromtimestamp(miner['last_updated_at'])
            print(
                f"{i}. {miner['hotkey'][:16]}...#{miner['revision'][:8]}... "
                f"(last_updated: {last_updated.strftime('%Y-%m-%d')}, "
                f"weight: {miner['best_weight']})"
            )
        
        if len(inactive_miners) > 10:
            print(f"... and {len(inactive_miners) - 10} more")
        
        # Confirm deletion
        confirm = input(f"\nWARNING: Delete {len(inactive_miners)} miners? Type 'yes' to confirm: ")
        
        if confirm.lower() == 'yes':
            await dao.cleanup_inactive_miners(inactive_days=days, dry_run=False)
            print(f"✓ Cleanup completed: {len(inactive_miners)} miners deleted")
        else:
            print("Cleanup cancelled")
    
    finally:
        await close_client()


async def cmd_update_miners():
    """Initialize/update miner_stats from sample_results using batch scan.
    
    Uses batch scanning to handle millions of samples efficiently:
    - Scans in batches to avoid memory overflow
    - Updates miner_stats incrementally per batch
    - Only updates updatable fields (first_seen_at, last_updated_at)
    - Skips immutable fields like best_rank (requires online data)
    """
    from affine.database.dao.sample_results import SampleResultsDAO
    from affine.database.dao.miner_stats import MinerStatsDAO
    from affine.database.tables import table_exists, create_table
    from affine.database.schema import MINER_STATS_SCHEMA, get_table_name
    from datetime import datetime
    
    print("Initializing miner_stats from sample_results...")
    print("Note: Only updates timestamps (first_seen_at, last_updated_at)")
    print("      Historical best_rank is not updated (requires online data)\n")
    
    await init_client()
    
    try:
        # Check if miner_stats table exists
        miner_stats_table = get_table_name("miner_stats")
        if not await table_exists(miner_stats_table):
            print(f"Table '{miner_stats_table}' does not exist. Creating it...")
            await create_table(MINER_STATS_SCHEMA)
            print(f"✓ Table '{miner_stats_table}' created successfully\n")
    
        sample_dao = SampleResultsDAO()
        miner_dao = MinerStatsDAO()
        
        from affine.database.client import get_client
        client = get_client()
        
        # Batch scan configuration
        batch_size = 1000  # Process 1000 samples per batch
        total_samples = 0
        total_batches = 0
        miners_created = 0
        miners_updated = 0
        
        # Track miners across batches to detect new ones
        seen_miners = set()
        
        print("Scanning sample_results table in batches...")
        
        params = {
            'TableName': sample_dao.table_name,
            'Limit': batch_size,
            'ProjectionExpression': 'miner_hotkey, model_revision, #model, #ts',
            'ExpressionAttributeNames': {
                '#model': 'model',
                '#ts': 'timestamp'
            }
        }
        
        while True:
            response = await client.scan(**params)
            items = response.get('Items', [])
            
            if not items:
                break
            
            # Process this batch
            batch_miners = {}  # {(hotkey, revision): {model, first_seen, last_seen}}
            
            for item in items:
                sample = sample_dao._deserialize(item)
                hotkey = sample.get('miner_hotkey')
                revision = sample.get('model_revision')
                model = sample.get('model', '')
                timestamp = sample.get('timestamp', 0)
                
                if not hotkey or not revision:
                    continue
                
                key = (hotkey, revision)
                if key not in batch_miners:
                    batch_miners[key] = {
                        'model': model,
                        'first_seen': timestamp,
                        'last_seen': timestamp
                    }
                else:
                    batch_miners[key]['first_seen'] = min(batch_miners[key]['first_seen'], timestamp)
                    batch_miners[key]['last_seen'] = max(batch_miners[key]['last_seen'], timestamp)
            
            # Update miner_stats for this batch
            for (hotkey, revision), info in batch_miners.items():
                existing = await miner_dao.get_miner_stats(hotkey, revision)
                
                if existing:
                    # Update timestamps only (don't touch best_rank/best_weight)
                    first_seen_ms = info['first_seen']
                    last_seen_ms = info['last_seen']
                    
                    # Convert milliseconds to seconds
                    first_seen_sec = first_seen_ms // 1000 if first_seen_ms > 1e10 else first_seen_ms
                    last_seen_sec = last_seen_ms // 1000 if last_seen_ms > 1e10 else last_seen_ms
                    
                    # Update first_seen_at if this sample is older
                    needs_update = False
                    updates = {}
                    
                    current_first_seen = existing.get('first_seen_at', float('inf'))
                    if first_seen_sec < current_first_seen:
                        updates['first_seen_at'] = first_seen_sec
                        needs_update = True
                    
                    current_last_updated = existing.get('last_updated_at', 0)
                    if last_seen_sec > current_last_updated:
                        updates['last_updated_at'] = last_seen_sec
                        needs_update = True
                    
                    if needs_update:
                        # Atomic update of timestamps only
                        pk = miner_dao._make_pk(hotkey)
                        sk = miner_dao._make_sk(revision)
                        
                        update_parts = []
                        expr_names = {}
                        expr_values = {}
                        
                        if 'first_seen_at' in updates:
                            update_parts.append('#first_seen = :first_seen')
                            expr_names['#first_seen'] = 'first_seen_at'
                            expr_values[':first_seen'] = {'N': str(updates['first_seen_at'])}
                        
                        if 'last_updated_at' in updates:
                            update_parts.append('#last_updated = :last_updated')
                            expr_names['#last_updated'] = 'last_updated_at'
                            expr_values[':last_updated'] = {'N': str(updates['last_updated_at'])}
                        
                        await client.update_item(
                            TableName=miner_dao.table_name,
                            Key={'pk': {'S': pk}, 'sk': {'S': sk}},
                            UpdateExpression=f"SET {', '.join(update_parts)}",
                            ExpressionAttributeNames=expr_names,
                            ExpressionAttributeValues=expr_values
                        )
                        miners_updated += 1
                else:
                    # Create new record
                    first_seen_ms = info['first_seen']
                    last_seen_ms = info['last_seen']
                    
                    # Convert milliseconds to seconds
                    first_seen_sec = first_seen_ms // 1000 if first_seen_ms > 1e10 else first_seen_ms
                    last_seen_sec = last_seen_ms // 1000 if last_seen_ms > 1e10 else last_seen_ms
                    
                    await miner_dao.update_miner_info(
                        hotkey=hotkey,
                        revision=revision,
                        model=info['model'],
                        rank=None,
                        weight=None,
                        is_online=False
                    )
                    
                    # Update timestamps to historical values
                    pk = miner_dao._make_pk(hotkey)
                    sk = miner_dao._make_sk(revision)
                    
                    await client.update_item(
                        TableName=miner_dao.table_name,
                        Key={'pk': {'S': pk}, 'sk': {'S': sk}},
                        UpdateExpression='SET #first_seen = :first_seen, #last_updated = :last_updated',
                        ExpressionAttributeNames={
                            '#first_seen': 'first_seen_at',
                            '#last_updated': 'last_updated_at'
                        },
                        ExpressionAttributeValues={
                            ':first_seen': {'N': str(first_seen_sec)},
                            ':last_updated': {'N': str(last_seen_sec)}
                        }
                    )
                    
                    miners_created += 1
                
                seen_miners.add((hotkey, revision))
            
            total_samples += len(items)
            total_batches += 1
            
            # Progress update
            print(
                f"  Batch {total_batches}: Processed {len(items)} samples "
                f"(total: {total_samples}, unique miners: {len(seen_miners)}, "
                f"created: {miners_created}, updated: {miners_updated})"
            )
            
            # Check for next page
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
            
            params['ExclusiveStartKey'] = last_key
        
        print(f"\n✓ Initialization complete:")
        print(f"  - Total samples scanned: {total_samples}")
        print(f"  - Total batches: {total_batches}")
        print(f"  - Unique miners found: {len(seen_miners)}")
        print(f"  - New records created: {miners_created}")
        print(f"  - Existing records updated: {miners_updated}")
        print(f"\nNote: Only timestamps were updated. Historical best_rank requires online miner data.")
    
    except Exception as e:
        print(f"\n✗ Failed to initialize miner_stats: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await close_client()


@click.group()
def db():
    """Database management commands."""
    pass


@db.command()
def init():
    """Initialize all DynamoDB tables."""
    asyncio.run(cmd_init())


@db.command("list")
def list_cmd():
    """List all tables."""
    asyncio.run(cmd_list())


@db.command()
def reset():
    """Reset all tables (delete and recreate)."""
    asyncio.run(cmd_reset())


@db.command("reset-table")
@click.option("--table", required=True, help="Table name to reset (e.g., task_queue, sample_results)")
def reset_table(table):
    """Reset a single table (delete and recreate)."""
    asyncio.run(cmd_reset_table(table))


@db.command()
@click.option("--tail", type=int, default=100000, help="Number of blocks to look back")
@click.option("--max-results", type=int, default=None, help="Maximum results to migrate")
def migrate(tail, max_results):
    """Migrate data from R2."""
    asyncio.run(cmd_migrate(tail, max_results))


@db.command("load-config")
@click.option(
    "--json-file",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_config.json"),
    help="Path to JSON configuration file"
)
def load_config(json_file):
    """Load system configuration from JSON file."""
    asyncio.run(cmd_load_config(json_file))


@db.group()
def blacklist():
    """Manage miner blacklist."""
    pass


@blacklist.command("list")
def blacklist_list():
    """List all blacklisted hotkeys."""
    asyncio.run(cmd_blacklist_list())


@blacklist.command()
@click.argument("hotkeys", nargs=-1, required=True)
def add(hotkeys):
    """Add hotkeys to blacklist."""
    asyncio.run(cmd_blacklist_add(list(hotkeys)))


@blacklist.command()
@click.argument("hotkeys", nargs=-1, required=True)
def remove(hotkeys):
    """Remove hotkeys from blacklist."""
    asyncio.run(cmd_blacklist_remove(list(hotkeys)))


@blacklist.command()
def clear():
    """Clear all hotkeys from blacklist."""
    asyncio.run(cmd_blacklist_clear())


@db.command("set-burn")
@click.argument("percentage", type=float)
def set_burn(percentage):
    """Set validator burn percentage (0.0 to 1.0)."""
    asyncio.run(cmd_set_burn_percentage(percentage))


@db.command("get-burn")
def get_burn():
    """Get current validator burn percentage."""
    asyncio.run(cmd_get_burn_percentage())


@db.command("get-config")
def get_config():
    """Get and print current system configuration."""
    asyncio.run(cmd_get_config())


@db.command("delete-samples-by-range")
@click.option("--hotkey", default=None, help="Miner's hotkey (optional, if not provided will delete for all miners)")
@click.option("--revision", default=None, help="Model revision hash (optional, if not provided will delete for all revisions)")
@click.option("--env", required=True, help="Environment name (e.g., agentgym:alfworld)")
@click.option("--start-task-id", required=True, type=int, help="Start task_id (inclusive)")
@click.option("--end-task-id", required=True, type=int, help="End task_id (exclusive)")
def delete_samples_by_range(hotkey, revision, env, start_task_id, end_task_id):
    """Delete samples within a task_id range for a specific miner and environment.
    
    If --hotkey and --revision are provided, deletes samples for that specific miner.
    If they are omitted, deletes all samples in the environment and range across all miners.
    
    Examples:
        # Delete for specific miner
        af db delete-samples-by-range --hotkey 5C5... --revision abc123 --env agentgym:alfworld --start-task-id 0 --end-task-id 100
        
        # Delete for all miners in environment
        af db delete-samples-by-range --env agentgym:alfworld --start-task-id 0 --end-task-id 100
    """
    # Validate that both hotkey and revision are provided together or both omitted
    if (hotkey is None) != (revision is None):
        print("Error: --hotkey and --revision must be provided together or both omitted")
        sys.exit(1)
    
    asyncio.run(cmd_delete_samples_by_range(hotkey, revision, env, start_task_id, end_task_id))


@db.command("delete-samples-empty-conversation")
def delete_samples_empty_conversation():
    """Delete all samples with empty conversation across the entire database.
    
    This command will scan the entire sample_results table and delete any samples
    where the conversation field is empty or null. Progress will be logged during execution.
    
    Example:
        af db delete-samples-empty-conversation
    """
    asyncio.run(cmd_delete_samples_empty_conversation())


@db.command("cleanup-inactive-miners")
@click.option('--days', default=30, help='Inactive days threshold')
def cleanup_inactive_miners(days: int):
    """Cleanup long-inactive miners from miner_stats table.
    
    Finds miners that haven't been updated for the specified number of days
    and have never had any weight (best_weight == 0), then prompts for deletion.
    
    Examples:
        af db cleanup-inactive-miners --days 30
        af db cleanup-inactive-miners --days 90
    """
    asyncio.run(cmd_cleanup_inactive_miners(days))


@db.command("update-miners")
def update_miners():
    """Initialize miner_stats table from existing sample_results data.
    
    Scans sample_results table and creates miner_stats records for all
    unique (hotkey, revision) combinations found. Skips existing records.
    
    This is useful for:
    - Initial setup after adding miner_stats table
    - Backfilling historical miner metadata
    
    Example:
        af db update-miners
    """
    asyncio.run(cmd_update_miners())


async def cmd_get_stats():
    """Get total sampling statistics for each environment.
    
    Aggregates sampling statistics across all miners for each environment.
    """
    print("Fetching environment sampling statistics...\n")
    await init_client()
    
    try:
        from affine.database.dao.miner_stats import MinerStatsDAO
        
        dao = MinerStatsDAO()
        all_miners = await dao.get_all_historical_miners()
        
        # Aggregate stats by environment
        env_aggregates = {}
        
        for miner in all_miners:
            env_stats = miner.get('env_stats', {})
            
            for env, stats in env_stats.items():
                if env not in env_aggregates:
                    env_aggregates[env] = {
                        'last_15min': {'samples': 0, 'success': 0, 'rate_limit_errors': 0, 'timeout_errors': 0, 'other_errors': 0},
                        'last_1hour': {'samples': 0, 'success': 0, 'rate_limit_errors': 0, 'timeout_errors': 0, 'other_errors': 0},
                        'last_6hours': {'samples': 0, 'success': 0, 'rate_limit_errors': 0, 'timeout_errors': 0, 'other_errors': 0},
                        'last_24hours': {'samples': 0, 'success': 0, 'rate_limit_errors': 0, 'timeout_errors': 0, 'other_errors': 0}
                    }
                
                for window in ['last_15min', 'last_1hour', 'last_6hours', 'last_24hours']:
                    if window in stats:
                        wstats = stats[window]
                        env_aggregates[env][window]['samples'] += wstats.get('samples', 0)
                        env_aggregates[env][window]['success'] += wstats.get('success', 0)
                        env_aggregates[env][window]['rate_limit_errors'] += wstats.get('rate_limit_errors', 0)
                        env_aggregates[env][window]['timeout_errors'] += wstats.get('timeout_errors', 0)
                        env_aggregates[env][window]['other_errors'] += wstats.get('other_errors', 0)
        
        # Print results
        if not env_aggregates:
            print("No sampling statistics found.")
            return
        
        # Calculate global totals
        global_totals = {
            'last_15min': {'samples': 0, 'success': 0, 'rate_limit_errors': 0, 'timeout_errors': 0, 'other_errors': 0},
            'last_1hour': {'samples': 0, 'success': 0, 'rate_limit_errors': 0, 'timeout_errors': 0, 'other_errors': 0},
            'last_6hours': {'samples': 0, 'success': 0, 'rate_limit_errors': 0, 'timeout_errors': 0, 'other_errors': 0},
            'last_24hours': {'samples': 0, 'success': 0, 'rate_limit_errors': 0, 'timeout_errors': 0, 'other_errors': 0}
        }
        
        for env_stats in env_aggregates.values():
            for window in ['last_15min', 'last_1hour', 'last_6hours', 'last_24hours']:
                global_totals[window]['samples'] += env_stats[window]['samples']
                global_totals[window]['success'] += env_stats[window]['success']
                global_totals[window]['rate_limit_errors'] += env_stats[window]['rate_limit_errors']
                global_totals[window]['timeout_errors'] += env_stats[window]['timeout_errors']
                global_totals[window]['other_errors'] += env_stats[window]['other_errors']
        
        # Print global totals first
        print("="*80)
        print("GLOBAL SAMPLING STATISTICS (ALL ENVIRONMENTS)")
        print("="*80)
        
        window_minutes = {
            'last_15min': 15,
            'last_1hour': 60,
            'last_6hours': 360,
            'last_24hours': 1440
        }
        
        for window in ['last_15min', 'last_1hour', 'last_6hours', 'last_24hours']:
            wstats = global_totals[window]
            samples = wstats['samples']
            success = wstats['success']
            success_rate = (success / samples * 100) if samples > 0 else 0
            samples_per_min = samples / window_minutes[window] if samples > 0 else 0
            
            print(f"\n{window}:")
            print(f"  Total samples: {samples}")
            print(f"  Success: {success} ({success_rate:.1f}%)")
            print(f"  Rate limit errors: {wstats['rate_limit_errors']}")
            print(f"  Timeout errors: {wstats['timeout_errors']}")
            print(f"  Other errors: {wstats['other_errors']}")
            print(f"  Samples/min: {samples_per_min:.2f}")
        
        # Print per-environment statistics
        print("\n" + "="*80)
        print("PER-ENVIRONMENT SAMPLING STATISTICS")
        print("="*80)
        
        for env in sorted(env_aggregates.keys()):
            print(f"\n{env}:")
            print("-"*80)
            
            for window in ['last_15min', 'last_1hour', 'last_6hours', 'last_24hours']:
                wstats = env_aggregates[env][window]
                samples = wstats['samples']
                success = wstats['success']
                success_rate = (success / samples * 100) if samples > 0 else 0
                samples_per_min = samples / window_minutes[window] if samples > 0 else 0
                
                print(f"\n  {window}:")
                print(f"    Total samples: {samples}")
                print(f"    Success: {success} ({success_rate:.1f}%)")
                print(f"    Rate limit errors: {wstats['rate_limit_errors']}")
                print(f"    Timeout errors: {wstats['timeout_errors']}")
                print(f"    Other errors: {wstats['other_errors']}")
                print(f"    Samples/min: {samples_per_min:.2f}")
        
        print("\n" + "="*80)
    
    finally:
        await close_client()


@db.command("get-stats")
def get_stats():
    """Get total sampling statistics for each environment.
    
    Aggregates and displays sampling statistics across all miners,
    grouped by environment. Shows statistics for different time windows
    (15min, 1hour, 6hours, 24hours).
    
    Example:
        af db get-stats
    """
    asyncio.run(cmd_get_stats())


async def cmd_get_pool():
    """Get task pool statistics: task count per miner per environment.
    
    Shows how many tasks each miner has in the sampling pool for each environment,
    including pending, assigned, and paused tasks, plus sampling statistics and missing tasks.
    Only displays environments where enabled_for_sampling=True.
    """
    print("Fetching task pool statistics...\n")
    await init_client()
    
    try:
        from affine.database.dao.miner_stats import MinerStatsDAO
        from affine.database.dao.sample_results import SampleResultsDAO
        
        task_dao = TaskPoolDAO()
        miners_dao = MinersDAO()
        miner_stats_dao = MinerStatsDAO()
        config_dao = SystemConfigDAO()
        sample_dao = SampleResultsDAO()
        
        # Get only valid miners for hotkey lookup
        all_miners = await miners_dao.get_valid_miners()
        uid_by_hotkey = {m['hotkey']: m['uid'] for m in all_miners}
        
        # Get all miner stats for sampling statistics
        all_miner_stats = await miner_stats_dao.get_all_historical_miners()
        # For global view: use sampling_stats (global aggregated)
        # For per-env view: use env_stats (per-environment breakdown)
        global_stats_by_miner = {
            (m['hotkey'], m['revision']): m.get('sampling_stats', {})
            for m in all_miner_stats
        }
        env_stats_by_miner = {
            (m['hotkey'], m['revision']): m.get('env_stats', {})
            for m in all_miner_stats
        }
        # Get slots for each miner
        slots_by_miner = {
            (m['hotkey'], m['revision']): m.get('sampling_slots', 6)
            for m in all_miner_stats
        }
        
        # Get enabled environments from system config
        environments = await config_dao.get_param_value('environments', default={})
        active_envs = sorted([
            env_name for env_name, env_config in environments.items()
            if env_config.get('enabled_for_sampling', False)
        ])
        
        if not active_envs:
            print("No environments configured.")
            return
        
        # Build miner list from valid miners only (no historical miners)
        # Each valid miner has: uid, hotkey, revision from miners table
        all_miners_keys = set()
        for m in all_miners:
            hotkey = m['hotkey']
            revision = m['revision']
            all_miners_keys.add((hotkey, revision))
        
        # Aggregate statistics across all environments
        global_stats = {
            'pending': {},  # {(hotkey, revision): count}
            'assigned': {},
            'paused': {},
            'missing': {}  # {(hotkey, revision): count}
        }
        
        env_stats = {}  # {env: {status: {(hotkey, revision): count}}}
        env_missing_stats = {}  # {env: {(hotkey, revision): count}}
        
        for env in active_envs:
            env_stats[env] = {}
            env_missing_stats[env] = {}
            
            # Get sampling_list for this environment
            env_config = environments.get(env, {})
            sampling_config = env_config.get('sampling_config', {})
            sampling_list = set(sampling_config.get('sampling_list', []))
            
            for status in ['pending', 'assigned', 'paused']:
                counts = await task_dao.get_miner_task_counts(env, status)
                env_stats[env][status] = counts
                
                # Aggregate to global
                for miner_key, count in counts.items():
                    parts = miner_key.split('#')
                    if len(parts) >= 2:
                        hotkey, revision = parts[0], parts[1]
                        key = (hotkey, revision)
                        global_stats[status][key] = global_stats[status].get(key, 0) + count
            
            # Calculate missing tasks for each miner in this environment
            # Use all valid miners (not just those with tasks in pool)
            all_miner_keys_in_env = set()
            for hotkey, revision in all_miners_keys:
                all_miner_keys_in_env.add(f"{hotkey}#{revision}")
            
            # Prepare concurrent tasks for all miners
            async def process_miner_missing_tasks(miner_key: str):
                """Process missing tasks calculation for a single miner."""
                parts = miner_key.split('#')
                if len(parts) < 2:
                    return None
                
                hotkey, revision = parts[0], parts[1]
                
                # Get completed task IDs and pool task IDs concurrently
                completed_taskids, pool_taskids = await asyncio.gather(
                    sample_dao.get_completed_task_ids(hotkey, revision, env),
                    task_dao.get_pending_task_ids_for_miner(hotkey, revision, env, include_paused=True)
                )
                
                # Calculate missing tasks
                missing_taskids = sampling_list - completed_taskids - pool_taskids
                missing_count = len(missing_taskids)
                
                return miner_key, hotkey, revision, missing_count
            
            # Execute all miner processing concurrently
            results = await asyncio.gather(
                *[process_miner_missing_tasks(miner_key) for miner_key in all_miner_keys_in_env]
            )
            
            # Aggregate results
            for result in results:
                if result is not None:
                    miner_key, hotkey, revision, missing_count = result
                    
                    # Store in env-specific stats
                    env_missing_stats[env][miner_key] = missing_count
                    
                    # Aggregate to global
                    key = (hotkey, revision)
                    global_stats['missing'][key] = global_stats['missing'].get(key, 0) + missing_count
        
        # Print global summary with sampling stats
        print("=" * 200)
        print("GLOBAL TASK POOL SUMMARY")
        print("=" * 200)
        
        if not all_miners_keys:
            print("No miners found.")
        else:
            # Sort by total tasks descending
            miner_totals = []
            for hotkey, revision in all_miners_keys:
                pending = global_stats['pending'].get((hotkey, revision), 0)
                assigned = global_stats['assigned'].get((hotkey, revision), 0)
                paused = global_stats['paused'].get((hotkey, revision), 0)
                missing = global_stats['missing'].get((hotkey, revision), 0)
                total = pending + assigned + paused
                uid = uid_by_hotkey.get(hotkey, '?')
                
                # Get global sampling stats
                sampling_stats = global_stats_by_miner.get((hotkey, revision), {})
                stats_15m = sampling_stats.get('last_15min', {})
                stats_1h = sampling_stats.get('last_1hour', {})
                stats_6h = sampling_stats.get('last_6hours', {})
                stats_24h = sampling_stats.get('last_24hours', {})
                
                # Get slots
                slots = slots_by_miner.get((hotkey, revision), 6)
                
                miner_totals.append((
                    uid, hotkey, revision, pending, assigned, paused, missing, total,
                    stats_15m, stats_1h, stats_6h, stats_24h, slots
                ))
            
            # Sort by (UID, hotkey, revision) to ensure unique ordering
            # Use (hotkey, revision) as primary sort key since it's the unique identifier
            miner_totals.sort(key=lambda x: (
                int(x[0]) if str(x[0]).isdigit() else 99999,  # UID (numeric)
                x[1],  # hotkey
                x[2]   # revision
            ))
            
            # Print header
            print(
                f"\n{'UID':<5} {'Hotkey':<19} {'Rev':<11} "
                f"{'Pend':<6} {'Assg':<6} {'Paus':<6} {'Miss':<6} "
                f"{'Slots':<6} "
                f"{'15m':<12} {'1h':<12} {'6h':<12} {'24h':<12}"
            )
            print("-" * 200)
            
            total_pending = sum(global_stats['pending'].values())
            total_assigned = sum(global_stats['assigned'].values())
            total_paused = sum(global_stats['paused'].values())
            total_missing = sum(global_stats['missing'].values())
            
            for uid, hotkey, revision, pending, assigned, paused, missing, total, s15m, s1h, s6h, s24h, slots in miner_totals:
                # Format sampling stats as "samples/succ"
                def format_stats(stats):
                    if not stats or stats.get('samples', 0) == 0:
                        return '-'
                    samples = stats.get('samples', 0)
                    success = stats.get('success', 0)
                    return f"{samples}/{success}"
                
                print(
                    f"{str(uid):<5} {hotkey[:16]+'...':<18} {revision[:8]+'...':<10} "
                    f"{pending:<6} {assigned:<6} {paused:<6} {missing:<6} "
                    f"{slots:<6} "
                    f"{format_stats(s15m):<12} {format_stats(s1h):<12} "
                    f"{format_stats(s6h):<12} {format_stats(s24h):<12}"
                )
            
            print("-" * 200)
            print(
                f"{'TOTAL':<5} {'':<18} {'':<10} "
                f"{total_pending:<6} {total_assigned:<6} {total_paused:<6} {total_missing:<6}"
            )
        
        # Print per-environment breakdown
        print("\n" + "=" * 200)
        print("PER-ENVIRONMENT TASK POOL BREAKDOWN")
        print("=" * 200)
        
        for env in active_envs:
            print(f"\n{env}:")
            print("-" * 200)
            
            # Use ALL valid miners for consistency (not just those with tasks in this env)
            # Aggregate counts for this environment
            env_totals = []
            for hotkey, revision in all_miners_keys:
                miner_key = f"{hotkey}#{revision}"
                pending = env_stats[env]['pending'].get(miner_key, 0)
                assigned = env_stats[env]['assigned'].get(miner_key, 0)
                paused = env_stats[env]['paused'].get(miner_key, 0)
                missing = env_missing_stats[env].get(miner_key, 0)
                total = pending + assigned + paused
                uid = uid_by_hotkey.get(hotkey, '?')
                
                # Get sampling stats for this specific environment
                miner_env_stats = env_stats_by_miner.get((hotkey, revision), {})
                env_specific_stats = miner_env_stats.get(env, {})
                stats_15m = env_specific_stats.get('last_15min', {})
                stats_1h = env_specific_stats.get('last_1hour', {})
                stats_6h = env_specific_stats.get('last_6hours', {})
                stats_24h = env_specific_stats.get('last_24hours', {})
                
                # Get slots
                slots = slots_by_miner.get((hotkey, revision), 6)
                
                env_totals.append((
                    uid, hotkey, revision, pending, assigned, paused, missing, total,
                    stats_15m, stats_1h, stats_6h, stats_24h, slots
                ))
            
            # Sort by (UID, hotkey, revision) to ensure unique ordering
            env_totals.sort(key=lambda x: (
                int(x[0]) if str(x[0]).isdigit() else 99999,  # UID (numeric)
                x[1],  # hotkey
                x[2]   # revision
            ))
            
            print(
                f"  {'UID':<5} {'Hotkey':<19} {'Rev':<11} "
                f"{'Pend':<6} {'Assg':<6} {'Paus':<6} {'Miss':<6} "
                f"{'Slots':<6} "
                f"{'15m':<12} {'1h':<12} {'6h':<12} {'24h':<12}"
            )
            print("  " + "-" * 196)
            
            env_total_pending = sum(env_stats[env]['pending'].values())
            env_total_assigned = sum(env_stats[env]['assigned'].values())
            env_total_paused = sum(env_stats[env]['paused'].values())
            env_total_missing = sum(env_missing_stats[env].values())
            
            for uid, hotkey, revision, pending, assigned, paused, missing, total, s15m, s1h, s6h, s24h, slots in env_totals:
                # Format sampling stats as "samples/succ"
                def format_stats(stats):
                    if not stats or stats.get('samples', 0) == 0:
                        return '-'
                    samples = stats.get('samples', 0)
                    success = stats.get('success', 0)
                    return f"{samples}/{success}"
                
                print(
                    f"  {str(uid):<5} {hotkey[:16]+'...':<18} {revision[:8]+'...':<10} "
                    f"{pending:<6} {assigned:<6} {paused:<6} {missing:<6} "
                    f"{slots:<6} "
                    f"{format_stats(s15m):<12} {format_stats(s1h):<12} "
                    f"{format_stats(s6h):<12} {format_stats(s24h):<12}"
                )
            
            print("  " + "-" * 196)
            print(
                f"  {'TOTAL':<5} {'':<18} {'':<10} "
                f"{env_total_pending:<6} {env_total_assigned:<6} {env_total_paused:<6} {env_total_missing:<6}"
            )
        
        print("\n" + "=" * 206)
    
    finally:
        await close_client()


@db.command("get-pool")
def get_pool():
    """Get task pool statistics per miner per environment.
    
    Shows the number of tasks each miner has in the sampling pool,
    broken down by environment and status (pending/assigned/paused).
    
    Example:
        af db get-pool
    """
    asyncio.run(cmd_get_pool())


async def cmd_get_miner(hotkey: str, revision: Optional[str]):
    """Get detailed performance data for a specific miner.
    
    Shows all performance metrics including:
    - Basic info (model, timestamps, rank, weight)
    - Sampling slots and adjustment history
    - Global sampling statistics (aggregated across all envs)
    - Per-environment sampling statistics
    
    Args:
        hotkey: Miner's hotkey (full or prefix)
        revision: Model revision (optional, if not provided shows all revisions)
    """
    import json
    import time
    from datetime import datetime
    
    print(f"Fetching miner stats for hotkey={hotkey}...\n")
    await init_client()
    
    try:
        from affine.database.dao.miner_stats import MinerStatsDAO
        from affine.database.dao.miners import MinersDAO
        
        miner_stats_dao = MinerStatsDAO()
        miners_dao = MinersDAO()
        
        # Get all miner stats
        all_stats = await miner_stats_dao.get_all_historical_miners()
        
        # Filter by hotkey (support prefix matching)
        matching_stats = [
            s for s in all_stats
            if s['hotkey'].startswith(hotkey) or hotkey in s['hotkey']
        ]
        
        if not matching_stats:
            print(f"No miner found with hotkey matching '{hotkey}'")
            return
        
        # Further filter by revision if provided
        if revision:
            matching_stats = [
                s for s in matching_stats
                if s['revision'].startswith(revision) or revision in s['revision']
            ]
            
            if not matching_stats:
                print(f"No miner found with revision matching '{revision}'")
                return
        
        # Get online miners for UID lookup
        online_miners = await miners_dao.get_valid_miners()
        uid_by_hotkey = {m['hotkey']: m['uid'] for m in online_miners}
        
        # Print each matching miner
        for i, stats in enumerate(matching_stats, 1):
            if i > 1:
                print("\n" + "=" * 120 + "\n")
            
            hotkey_val = stats['hotkey']
            revision_val = stats['revision']
            uid = uid_by_hotkey.get(hotkey_val, 'offline')
            
            print("=" * 120)
            print(f"MINER #{i}: {hotkey_val}")
            print("=" * 120)
            
            # Basic Info
            print("\n[BASIC INFO]")
            print(f"  UID: {uid}")
            print(f"  Hotkey: {hotkey_val}")
            print(f"  Revision: {revision_val}")
            print(f"  Model: {stats.get('model', 'N/A')}")
            
            first_seen = stats.get('first_seen_at', 0)
            last_updated = stats.get('last_updated_at', 0)
            if first_seen > 0:
                first_seen_dt = datetime.fromtimestamp(first_seen)
                print(f"  First seen: {first_seen_dt.strftime('%Y-%m-%d %H:%M:%S')} ({first_seen})")
            else:
                print(f"  First seen: N/A")
            
            if last_updated > 0:
                last_updated_dt = datetime.fromtimestamp(last_updated)
                elapsed = int(time.time()) - last_updated
                print(f"  Last updated: {last_updated_dt.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed}s ago)")
            else:
                print(f"  Last updated: N/A")
            
            print(f"  Best rank: {stats.get('best_rank', 'N/A')}")
            print(f"  Best weight: {stats.get('best_weight', 0.0):.6f}")
            print(f"  Currently online: {stats.get('is_currently_online', False)}")
            
            # Slots Info
            print("\n[SAMPLING SLOTS]")
            slots = stats.get('sampling_slots', 6)
            slots_last_adjusted = stats.get('slots_last_adjusted_at', 0)
            print(f"  Current slots: {slots}")
            
            if slots_last_adjusted > 0:
                adjusted_dt = datetime.fromtimestamp(slots_last_adjusted)
                elapsed = int(time.time()) - slots_last_adjusted
                print(f"  Last adjusted: {adjusted_dt.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed}s ago)")
            else:
                print(f"  Last adjusted: Never")
            
            # Global Sampling Stats
            print("\n[GLOBAL SAMPLING STATISTICS]")
            sampling_stats = stats.get('sampling_stats', {})
            
            if not sampling_stats:
                print("  No global statistics available")
            else:
                windows = ['last_15min', 'last_1hour', 'last_6hours', 'last_24hours']
                window_labels = {
                    'last_15min': '15min',
                    'last_1hour': '1hour',
                    'last_6hours': '6hours',
                    'last_24hours': '24hours'
                }
                
                print(f"\n  {'Window':<10} {'Samples':<10} {'Success':<10} {'Rate':<10} {'Slots':<8} {'RateLimit':<12} {'Timeout':<10} {'Other':<10} {'Samp/min':<10}")
                print("  " + "-" * 118)
                
                for window in windows:
                    wstats = sampling_stats.get(window, {})
                    if not wstats:
                        continue
                    
                    samples = wstats.get('samples', 0)
                    success = wstats.get('success', 0)
                    success_rate = wstats.get('success_rate', 0.0)
                    rate_limit = wstats.get('rate_limit_errors', 0)
                    timeout = wstats.get('timeout_errors', 0)
                    other = wstats.get('other_errors', 0)
                    samp_per_min = wstats.get('samples_per_min', 0.0)
                    
                    # Show slots in 6hours row (used by adjuster)
                    slots_display = str(slots) if window == 'last_6hours' else '-'
                    
                    print(
                        f"  {window_labels[window]:<10} {samples:<10} {success:<10} "
                        f"{success_rate:<10.2%} {slots_display:<8} {rate_limit:<12} {timeout:<10} {other:<10} {samp_per_min:<10.2f}"
                    )
            
            # Per-Environment Stats
            print("\n[PER-ENVIRONMENT STATISTICS]")
            env_stats = stats.get('env_stats', {})
            
            if not env_stats:
                print("  No per-environment statistics available")
            else:
                for env_name in sorted(env_stats.keys()):
                    env_data = env_stats[env_name]
                    print(f"\n  Environment: {env_name}")
                    print("  " + "-" * 110)
                    
                    windows = ['last_15min', 'last_1hour', 'last_6hours', 'last_24hours']
                    window_labels = {
                        'last_15min': '15min',
                        'last_1hour': '1hour',
                        'last_6hours': '6hours',
                        'last_24hours': '24hours'
                    }
                    
                    print(f"    {'Window':<10} {'Samples':<10} {'Success':<10} {'Rate':<10} {'Slots':<8} {'RateLimit':<12} {'Timeout':<10} {'Other':<10} {'Samp/min':<10}")
                    print("    " + "-" * 116)
                    
                    for window in windows:
                        wstats = env_data.get(window, {})
                        if not wstats:
                            continue
                        
                        samples = wstats.get('samples', 0)
                        success = wstats.get('success', 0)
                        success_rate = wstats.get('success_rate', 0.0)
                        rate_limit = wstats.get('rate_limit_errors', 0)
                        timeout = wstats.get('timeout_errors', 0)
                        other = wstats.get('other_errors', 0)
                        samp_per_min = wstats.get('samples_per_min', 0.0)
                        
                        # Show slots in 6hours row (used by adjuster)
                        slots_display = str(slots) if window == 'last_6hours' else '-'
                        
                        print(
                            f"    {window_labels[window]:<10} {samples:<10} {success:<10} "
                            f"{success_rate:<10.2%} {slots_display:<8} {rate_limit:<12} {timeout:<10} {other:<10} {samp_per_min:<10.2f}"
                        )
            
            print("\n" + "=" * 120)
        
        print(f"\n✓ Found {len(matching_stats)} matching miner(s)")
    
    finally:
        await close_client()


@db.command("get-miner")
@click.option("--hotkey", required=True, help="Miner's hotkey (full or prefix)")
@click.option("--revision", default=None, help="Model revision (optional)")
def get_miner(hotkey: str, revision: Optional[str]):
    """Get detailed performance data for a specific miner.
    
    Shows comprehensive statistics including sampling performance, slots,
    and adjustment history for the specified miner.
    
    Examples:
        # Get stats for all revisions of a hotkey
        af db get-miner --hotkey 5DJHkQEio6qSayH3
        
        # Get stats for specific revision
        af db get-miner --hotkey 5DJHkQEio6qSayH3 --revision 7b4a3a20
    """
    asyncio.run(cmd_get_miner(hotkey, revision))


def main():
    """Main CLI entry point."""
    db()


if __name__ == "__main__":
    main()