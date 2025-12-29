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
    DataRetentionDAO,
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


async def cmd_reset(yes: bool = False):
    """Reset all tables (delete and recreate)."""
    if not yes:
        if not click.confirm("WARNING: This will delete all data. Continue?"):
            click.echo("Aborted")
            return

    await init_client()

    try:
        await reset_tables()
        click.echo("✓ Tables reset successfully")
    finally:
        await close_client()


async def cmd_reset_table(table_name: str, yes: bool = False):
    """Reset a single table (delete and recreate)."""
    from affine.database.schema import get_table_name

    # Get full table name with environment prefix
    full_table_name = get_table_name(table_name)

    if not yes:
        if not click.confirm(f"WARNING: This will delete all data in '{full_table_name}'. Continue?"):
            click.echo("Aborted")
            return

    await init_client()

    try:
        click.echo(f"Deleting table '{full_table_name}'...")
        await delete_table(full_table_name)

        click.echo(f"Recreating table '{full_table_name}'...")
        await init_tables()

        click.echo(f"✓ Table '{full_table_name}' reset successfully")
    except Exception as e:
        click.echo(f"✗ Failed to reset table: {e}")
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
            
            for env_name, env_config in environments.items():
                sampling_config = env_config.get('sampling_config')
                if not sampling_config:
                    print(f"  Warning: {env_name} missing sampling_config, skipping")
                    continue
                
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


async def cmd_blacklist_clear(yes: bool = False):
    """Clear all hotkeys from blacklist."""
    if not yes:
        if not click.confirm("WARNING: This will clear the entire blacklist. Continue?"):
            click.echo("Aborted")
            return

    click.echo("Clearing blacklist...")
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
    end_task_id: int,
    yes: bool = False
):
    """Delete samples within a task_id range.
    
    If hotkey and revision are provided, deletes samples for that specific miner.
    If they are not provided, deletes all samples in the environment and range.
    """
    # interactive confirmation unless `yes` flag is provided
    if hotkey and revision:
        click.echo(f"Deleting samples for hotkey={hotkey[:12]}..., revision={revision[:8]}..., env={env}, task_id range=[{start_task_id}, {end_task_id})...")
        if not yes and not click.confirm(f"WARNING: This will delete samples for specific miner in range [{start_task_id}, {end_task_id}). Continue?"):
            click.echo("Aborted")
            return
    else:
        click.echo(f"Deleting ALL samples for env={env}, task_id range=[{start_task_id}, {end_task_id})...")
        if not yes and not click.confirm(f"WARNING: This will delete ALL samples across all miners/revisions for env={env} in range [{start_task_id}, {end_task_id}). Continue?"):
            click.echo("Aborted")
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


async def cmd_delete_samples_empty_conversation(yes: bool = False):
    """Delete all samples with empty conversation across the entire database."""
    click.echo("Scanning entire sample database for invalid samples (empty conversation)...")

    if not yes and not click.confirm("WARNING: This will scan and delete ALL samples with empty conversation in the database. Continue?"):
        click.echo("Aborted")
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
@click.option("--yes", "-y", is_flag=True, help="Assume yes for confirmation prompts")
@click.option("--force", "-f", is_flag=True, help="Alias for --yes")
def reset(yes, force):
    """Reset all tables (delete and recreate)."""
    asyncio.run(cmd_reset(yes=yes or force))


@db.command("reset-table")
@click.option("--table", required=True, help="Table name to reset (e.g., task_queue, sample_results)")
@click.option("--yes", "-y", is_flag=True, help="Assume yes for confirmation prompts")
@click.option("--force", "-f", is_flag=True, help="Alias for --yes")
def reset_table(table, yes, force):
    """Reset a single table (delete and recreate)."""
    asyncio.run(cmd_reset_table(table, yes=yes or force))


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
@click.option("--yes", "-y", is_flag=True, help="Assume yes for confirmation prompts")
@click.option("--force", "-f", is_flag=True, help="Alias for --yes")
def clear(yes, force):
    """Clear all hotkeys from blacklist."""
    asyncio.run(cmd_blacklist_clear(yes=yes or force))


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
@click.option("--yes", "-y", is_flag=True, help="Assume yes for confirmation prompts")
@click.option("--force", "-f", is_flag=True, help="Alias for --yes")
def delete_samples_by_range(hotkey, revision, env, start_task_id, end_task_id, yes, force):
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
        click.echo("Error: --hotkey and --revision must be provided together or both omitted")
        sys.exit(1)

    asyncio.run(cmd_delete_samples_by_range(hotkey, revision, env, start_task_id, end_task_id, yes=(yes or force)))


@db.command("delete-samples-empty-conversation")
@click.option("--yes", "-y", is_flag=True, help="Assume yes for confirmation prompts")
@click.option("--force", "-f", is_flag=True, help="Alias for --yes")
def delete_samples_empty_conversation(yes, force):
    """Delete all samples with empty conversation across the entire database.
    
    This command will scan the entire sample_results table and delete any samples
    where the conversation field is empty or null. Progress will be logged during execution.
    
    Example:
        af db delete-samples-empty-conversation
    """
    asyncio.run(cmd_delete_samples_empty_conversation(yes=(yes or force)))


def main():
    """Main CLI entry point."""
    db()


if __name__ == "__main__":
    main()