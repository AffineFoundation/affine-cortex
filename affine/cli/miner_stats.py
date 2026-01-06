"""
Miner Stats CLI Commands

Provides commands for managing historical miner statistics and cleanup.
"""

import asyncio
import click
from datetime import datetime
from affine.database import init_client, close_client
from affine.core.setup import logger


@click.group()
def miner_stats():
    """Query miner historical statistics."""
    pass


@miner_stats.command()
@click.argument('hotkey')
@click.argument('revision')
def get_stats(hotkey: str, revision: str):
    """Get statistics for a specific miner.
    
    Examples:
        af miner-stats get-stats 5GrwvaEF5zXb... abc123def
    """
    async def _get_stats():
        await init_client()
        
        try:
            from affine.database.dao.miner_stats import MinerStatsDAO
            
            dao = MinerStatsDAO()
            stats = await dao.get_miner_stats(hotkey, revision)
            
            if not stats:
                click.echo(f"No stats found for {hotkey}#{revision}")
                return
            
            # Display basic info
            click.echo(f"\n=== Miner Stats ===")
            click.echo(f"Hotkey: {stats['hotkey']}")
            click.echo(f"Revision: {stats['revision']}")
            click.echo(f"Model: {stats.get('model', 'N/A')}")
            click.echo(f"First seen: {datetime.fromtimestamp(stats['first_seen_at']).strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"Last updated: {datetime.fromtimestamp(stats['last_updated_at']).strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"Best rank: {stats.get('best_rank', 'N/A')}")
            click.echo(f"Best weight: {stats.get('best_weight', 0.0):.4f}")
            click.echo(f"Currently online: {stats.get('is_currently_online', False)}")
            
            # Display global statistics
            if 'sampling_stats' in stats and stats['sampling_stats']:
                click.echo("\n=== Global Sampling Statistics ===")
                for window in ['last_15min', 'last_1hour', 'last_6hours', 'last_24hours']:
                    if window in stats['sampling_stats']:
                        wstats = stats['sampling_stats'][window]
                        click.echo(f"\n{window}:")
                        click.echo(f"  Samples: {wstats.get('samples', 0)}")
                        click.echo(f"  Success: {wstats.get('success', 0)}")
                        click.echo(f"  Success rate: {wstats.get('success_rate', 0.0):.2%}")
                        click.echo(f"  Samples/min: {wstats.get('samples_per_min', 0.0):.2f}")
                        click.echo(f"  Rate limit errors: {wstats.get('rate_limit_errors', 0)}")
                        click.echo(f"  Timeout errors: {wstats.get('timeout_errors', 0)}")
                        click.echo(f"  Other errors: {wstats.get('other_errors', 0)}")
            
            # Display per-environment statistics
            if 'env_stats' in stats and stats['env_stats']:
                click.echo("\n=== Per-Environment Statistics ===")
                for env, env_data in stats['env_stats'].items():
                    click.echo(f"\n{env}:")
                    for window in ['last_15min', 'last_1hour', 'last_6hours', 'last_24hours']:
                        if window in env_data:
                            wstats = env_data[window]
                            click.echo(f"  {window}:")
                            click.echo(f"    Samples: {wstats.get('samples', 0)}")
                            click.echo(f"    Success rate: {wstats.get('success_rate', 0.0):.2%}")
                            click.echo(f"    Samples/min: {wstats.get('samples_per_min', 0.0):.2f}")
        
        finally:
            await close_client()
    
    asyncio.run(_get_stats())


@miner_stats.command()
@click.option('--limit', default=20, help='Number of miners to show')
def list_all(limit: int):
    """List all historical miners.
    
    Examples:
        af miner-stats list-all
        af miner-stats list-all --limit 50
    """
    async def _list_all():
        await init_client()
        
        try:
            from affine.database.dao.miner_stats import MinerStatsDAO
            
            dao = MinerStatsDAO()
            all_miners = await dao.get_all_historical_miners()
            
            click.echo(f"\nTotal historical miners: {len(all_miners)}\n")
            
            # Sort by last_updated_at (most recent first)
            sorted_miners = sorted(
                all_miners,
                key=lambda m: m.get('last_updated_at', 0),
                reverse=True
            )
            
            for i, miner in enumerate(sorted_miners[:limit], 1):
                last_updated = datetime.fromtimestamp(miner['last_updated_at'])
                online = "🟢" if miner.get('is_currently_online', False) else "🔴"
                
                click.echo(
                    f"{i:2d}. {online} {miner['hotkey'][:16]}...#{miner['revision'][:8]}... "
                    f"| rank: {miner.get('best_rank', 'N/A'):3} "
                    f"| weight: {miner.get('best_weight', 0.0):6.4f} "
                    f"| updated: {last_updated.strftime('%Y-%m-%d %H:%M')}"
                )
            
            if len(sorted_miners) > limit:
                click.echo(f"\n... and {len(sorted_miners) - limit} more")
                click.echo(f"Use --limit to show more miners")
        
        finally:
            await close_client()
    
    asyncio.run(_list_all())