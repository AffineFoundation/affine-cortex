"""
``af servers scheduler`` — block-driven flow scheduler entry point.

Polls subtensor for new blocks; on each block tick calls
``FlowScheduler.tick(block)`` which walks the 9-step flow:

  task_ids refresh → champion validate → cold start → champion deploy
  → champion samples wait → start battle → challenger samples wait
  → decide → cleanup

All state lives in ``system_config``; this process is restart-safe via
deterministic re-derivation (sampler is seeded; Targon adopts existing
workloads by naming convention).
"""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Dict, List, Optional

import click

from affine.core.providers.targon_client import get_targon_client
from affine.core.setup import logger, setup_logging
from affine.database import close_client, init_client
from affine.utils.subtensor import get_subtensor
from affine.database.dao.miners import MinersDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.system_config import SystemConfigDAO

from affine.src.scorer.challenger_queue import ChallengerQueue
from affine.src.scorer.comparator import WindowComparator
from affine.src.scorer.dao_adapters import (
    MinersQueueAdapter,
    SampleResultsAdapter,
)
from affine.src.scorer.sampler import WindowSampler
from affine.src.scorer.weight_writer import WeightWriter
from affine.src.scorer.window_state import (
    StateStore,
    SystemConfigKVAdapter,
)

from . import ssh as ssh_lifecycle
from . import targon as targon_lifecycle
from .flow import FlowConfig, FlowScheduler


PROVIDER_KIND = os.getenv("AFFINE_PROVIDER_KIND", "targon").lower()
"""Which inference provider to use. ``targon`` (default) routes through
the hosted Targon API; multi-instance, deployments are independent.
``ssh`` SSHs into an operator-managed host and runs sglang via docker —
single-instance, model swap on each contest."""

NETUID = int(os.getenv("NETUID", "120"))
SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK", "finney")
TICK_INTERVAL_SEC = float(os.getenv("SCHEDULER_TICK_INTERVAL_SEC", "12"))
ORPHAN_SWEEP_INTERVAL_SEC = int(
    os.getenv("SCHEDULER_ORPHAN_SWEEP_INTERVAL_SEC", "600")
)


def _endpoint_matches_target(endpoint, target) -> bool:
    return (
        endpoint.assigned_uid == target.uid
        and endpoint.assigned_hotkey == target.hotkey
        and endpoint.assigned_model == target.model
        and endpoint.assigned_revision == target.revision
    )


def _select_ssh_endpoints(endpoints: List[object], target, *, role: str) -> List[object]:
    """Pick SSH machines for one miner.

    Policy is intentionally simple:
    - reuse endpoints already assigned to the same miner (restart/adopt);
    - champion gets up to N-1 free endpoints, leaving capacity for a challenger;
    - challenger gets all remaining free endpoints;
    - with one machine, challenger may reuse it sequentially.
    """
    if not endpoints:
        raise RuntimeError("no active ssh endpoints")
    matching = [ep for ep in endpoints if _endpoint_matches_target(ep, target)]
    if matching:
        return matching

    free = [ep for ep in endpoints if ep.assigned_uid is None]
    if role == "champion":
        if len(endpoints) == 1:
            return [endpoints[0]]
        desired = max(1, len(endpoints) - 1)
        selected = free[:desired]
    elif role == "challenger":
        selected = free or ([endpoints[0]] if len(endpoints) == 1 else [])
    else:
        selected = free or endpoints

    if not selected:
        raise RuntimeError(
            f"no ssh endpoints available for role={role} "
            f"target_uid={target.uid}"
        )
    return selected


async def _run() -> None:
    await init_client()

    miners_dao = MinersDAO()
    samples_dao = SampleResultsDAO()
    scores_dao = ScoresDAO()
    snapshots_dao = ScoreSnapshotsDAO()
    config_dao = SystemConfigDAO()

    kv_adapter = SystemConfigKVAdapter(config_dao, updated_by="scheduler")
    state_store = StateStore(kv_adapter)

    queue = ChallengerQueue(MinersQueueAdapter(miners_dao))
    sampler = WindowSampler()
    comparator = WindowComparator()
    weight_writer = WeightWriter(scores_dao, snapshots_dao)
    samples_adapter = SampleResultsAdapter(
        dao=samples_dao, validator_hotkey="scheduler",
    )

    async def sample_count_fn(
        hotkey: str, revision: str, env: str, task_ids: List[int],
        refresh_block: int,
    ) -> int:
        return await samples_adapter.count_samples_for_tasks(
            hotkey, revision, env, task_ids, refresh_block=refresh_block,
        )

    async def scores_reader(
        hotkey: str, revision: str, env: str, task_ids: List[int],
        refresh_block: int,
    ) -> Dict[int, float]:
        return await samples_adapter.read_scores_for_tasks(
            hotkey, revision, env, task_ids, refresh_block=refresh_block,
        )

    async def list_valid_miners_fn():
        return await miners_dao.get_valid_miners()

    # Dispatch provider per AFFINE_PROVIDER_KIND. ``targon`` (default) is
    # multi-instance — independent deployments per role. ``ssh`` is
    # single-instance — one model on the operator's GPU host at a time;
    # the FlowConfig flag tells the flow to invalidate champion.deployment_id
    # at the right moments (refresh + post-loss teardown) so step 5 fresh-
    # deploys when needed.
    if PROVIDER_KIND == "ssh":
        # SSH endpoints are configured in DynamoDB. There is intentionally
        # no env-var fallback for the host URL/key/public URL; otherwise a
        # stale container env could silently override the operator-managed
        # endpoint registry.
        from affine.database.dao.inference_endpoints import InferenceEndpointsDAO
        endpoints_dao = InferenceEndpointsDAO()
        active = await endpoints_dao.list_active(kind="ssh")
        if not active:
            raise RuntimeError(
                "scheduler: provider=ssh but no active ssh endpoint is "
                "registered; configure one with `af db set-endpoint "
                "--kind ssh --ssh-url ssh://user@host[:port] ...`"
            )
        active = sorted(active, key=lambda ep: ep.name)
        ssh_configs = {
            ep.name: ssh_lifecycle.SSHConfig.from_endpoint(ep)
            for ep in active
        }
        logger.info(
            f"scheduler: provider=ssh endpoints={[ep.name for ep in active]!r} "
            f"(from inference_endpoints table)"
        )

        async def deploy_fn(target, role: str = "active"):
            endpoints = sorted(
                await endpoints_dao.list_active(kind="ssh"),
                key=lambda ep: ep.name,
            )
            selected = _select_ssh_endpoints(endpoints, target, role=role)
            deployments = []
            for ep in selected:
                cfg = ssh_configs.get(ep.name) or ssh_lifecycle.SSHConfig.from_endpoint(ep)
                result = await ssh_lifecycle.deploy(cfg, target)
                await endpoints_dao.set_assignment(
                    ep.name,
                    uid=target.uid,
                    hotkey=target.hotkey,
                    model=target.model,
                    revision=target.revision,
                    deployment_id=result.deployment_id,
                    base_url=result.base_url,
                    role=role,
                )
                deployments.append(
                    targon_lifecycle.MachineDeployment(
                        endpoint_name=ep.name,
                        deployment_id=result.deployment_id,
                        base_url=result.base_url,
                    )
                )
            primary = deployments[0]
            return targon_lifecycle.DeployResult(
                deployment_id=primary.deployment_id,
                base_url=primary.base_url,
                deployments=deployments,
            )

        async def teardown_fn(deployment_id):
            if not deployment_id:
                return
            for name, cfg in ssh_configs.items():
                if deployment_id == cfg.deployment_id():
                    await ssh_lifecycle.teardown(cfg, deployment_id)
                    await endpoints_dao.clear_assignment(name)
                    return
            logger.warning(
                f"scheduler: no ssh endpoint owns deployment_id={deployment_id!r}"
            )

        flow_config = FlowConfig(single_instance_provider=(len(active) == 1))
        targon_client = None
        logger.info(
            "scheduler: ssh control plane → "
            + ", ".join(f"{cfg.host}:{cfg.port}" for cfg in ssh_configs.values())
        )
    else:
        targon_client = get_targon_client()
        targon_endpoint_count = int(
            await config_dao.get_param_value("targon_endpoint_count", default=0) or 0
        )
        if targon_endpoint_count > 0:
            logger.warning(
                "scheduler: targon_endpoint_count>0 requested, but automatic "
                "Targon endpoint provisioning is not implemented in this flow; "
                "falling back to direct Targon workload lifecycle."
            )

        async def deploy_fn(target, role: str = "active"):
            return await targon_lifecycle.deploy(targon_client, target)

        async def teardown_fn(deployment_id):
            await targon_lifecycle.teardown(targon_client, deployment_id)

        flow_config = FlowConfig()
        logger.info("scheduler: provider=targon")

    scheduler = FlowScheduler(
        config=flow_config,
        state=state_store,
        queue=queue,
        sampler=sampler,
        comparator=comparator,
        weight_writer=weight_writer,
        deploy_fn=deploy_fn,
        teardown_fn=teardown_fn,
        sample_count_fn=sample_count_fn,
        scores_reader=scores_reader,
        list_valid_miners_fn=list_valid_miners_fn,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    # Use the project's async-safe SubtensorWrapper (re-connects on failure).
    # Hard-coding ``bittensor.subtensor`` (sync) doesn't exist on the version
    # of bittensor shipped in the deploy image.
    subtensor = await get_subtensor()
    logger.info(f"scheduler started: netuid={NETUID}")

    last_block: Optional[int] = None
    last_orphan_sweep = 0.0
    try:
        while not stop_event.is_set():
            try:
                block = await subtensor.get_current_block()
            except Exception as e:
                logger.warning(f"scheduler: get_current_block raised: {e}")
                await asyncio.sleep(TICK_INTERVAL_SEC)
                continue

            if block != last_block:
                last_block = block
                try:
                    await scheduler.tick(block)
                except Exception as e:
                    logger.error(
                        f"scheduler.tick({block}) raised: "
                        f"{type(e).__name__}: {e}", exc_info=True,
                    )

            now = asyncio.get_event_loop().time()
            if (
                targon_client is not None
                and now - last_orphan_sweep >= ORPHAN_SWEEP_INTERVAL_SEC
            ):
                last_orphan_sweep = now
                try:
                    await _orphan_sweep_tick(state_store, targon_client)
                except Exception as e:
                    logger.warning(f"scheduler orphan_sweep raised: {e}")

            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=TICK_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                pass
    finally:
        await close_client()


async def _orphan_sweep_tick(state_store, targon_client) -> None:
    """Collect known deployment_ids (champion + in-flight battle) and ask
    Targon to drop any affine-prefixed workload outside the known set."""
    known = set()
    champ = await state_store.get_champion()
    if champ and champ.deployment_id:
        known.add(champ.deployment_id)
    battle = await state_store.get_battle()
    if battle and battle.deployment_id:
        known.add(battle.deployment_id)
    await targon_lifecycle.orphan_sweep(targon_client, known)


@click.command()
@click.option("-v", "--verbose", count=True, default=1)
def main(verbose: int) -> None:
    """Run the flow scheduler service."""
    setup_logging(verbosity=verbose, component="scheduler")
    asyncio.run(_run())


if __name__ == "__main__":
    main()
