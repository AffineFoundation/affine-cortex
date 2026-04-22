"""Targon deployer reconciliation loop.

Closed-loop auto-deployment over a small set of "targets" drawn from the
current champion plus winning challengers:

    targets = { champion } ∪ top-N challengers by checkpoints_passed
              where challenge_status == 'sampling' AND consecutive_wins > 0
    |targets| ≤ MAX_TARGON_DEPLOYMENTS   (default 4, env-configurable)

Each loop iteration:

    1. Resolve targets from system_config.champion + miner_stats.
    2. For every target: ensure a Targon deployment exists (adopt matching
       live workload if present; otherwise create).
    3. Tear down any deployment whose (hotkey, revision) is no longer a
       target — covers champion change, challenger termination, challenger
       losing its only win, etc. Teardown is immediate (no 24h grace).
    4. Health sweep: probe every active/deploying deployment. On 2
       consecutive failed probes (debounces state-aggregator lag) we delete
       the Targon workload + mark the DB row deleted. The miner re-enters
       the queue next cycle — if it's still the top candidate it redeploys,
       otherwise a higher-priority waiter takes the slot.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from affine.core.providers.targon_client import TargonClient, get_targon_client
from affine.core.setup import logger
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.database.dao.targon_deployments import TargonDeploymentsDAO


DEFAULT_POLL_INTERVAL = int(os.getenv("TARGON_POLL_INTERVAL_SEC", "30"))
DEFAULT_MAX_DEPLOYMENTS = int(os.getenv("TARGON_MAX_DEPLOYMENTS", "4"))


class TargonDeployerService:
    def __init__(
        self,
        targon_dao: Optional[TargonDeploymentsDAO] = None,
        config_dao: Optional[SystemConfigDAO] = None,
        miners_dao: Optional[MinersDAO] = None,
        miner_stats_dao: Optional[MinerStatsDAO] = None,
        client: Optional[TargonClient] = None,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        max_deployments: int = DEFAULT_MAX_DEPLOYMENTS,
    ):
        self.dao = targon_dao or TargonDeploymentsDAO()
        self.config_dao = config_dao or SystemConfigDAO()
        self.miners_dao = miners_dao or MinersDAO()
        self.miner_stats_dao = miner_stats_dao or MinerStatsDAO()
        self.client = client or get_targon_client()
        self.poll_interval = poll_interval
        self.max_deployments = max(1, max_deployments)
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        if not self.client.configured:
            logger.warning(
                "TargonDeployerService: TARGON_API_KEY/URL not set — reconciler "
                "idle. Set env vars to activate."
            )
        logger.info(
            f"TargonDeployerService starting (interval={self.poll_interval}s, "
            f"max_deployments={self.max_deployments})"
        )
        while not self._stop_event.is_set():
            try:
                await self._reconcile_targets()
                await self._health_sweep()
            except Exception as e:
                logger.error(f"TargonDeployerService loop error: {e}", exc_info=True)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.poll_interval
                )
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    async def _resolve_targets(self) -> List[Tuple[str, str, str]]:
        """Compute the target set for this cycle.

        Selection order (first fills first, up to ``max_deployments``):
          1. Current champion (if one is configured)
          2. **Incumbents** — winners whose deployment already exists in the
             targon_deployments table (status active/deploying/rebuilding) and
             still qualify (sampling + consecutive_wins > 0). Giving them
             priority turns the cap into a **FIFO queue**: once a miner gets a
             slot, it keeps it until it drops out voluntarily (terminated /
             wins→0 / champion change).
          3. **Newcomers** — eligible winners not yet deployed. Ordered by
             checkpoints_passed desc, wins desc. They queue up behind
             incumbents and take any freed slots in the next cycle.

        Eligibility filter for challengers:
          - challenge_status == 'sampling'   (terminated miners excluded)
          - consecutive_wins > 0             (only current winners deploy)
          - (hotkey, revision) != champion's (avoid double counting)

        Returns list of (hotkey, revision, role) where role ∈ {'champion','winner'}.
        """
        targets: List[Tuple[str, str, str]] = []
        seen: Set[Tuple[str, str]] = set()

        champion = await self.config_dao.get_param_value("champion", default=None)
        if champion and champion.get("hotkey") and champion.get("revision"):
            champ_key = (champion["hotkey"], champion["revision"])
            targets.append((champ_key[0], champ_key[1], "champion"))
            seen.add(champ_key)

        if self.max_deployments <= 1:
            return targets

        all_stats = await self.miner_stats_dao.get_all_historical_miners()

        # (hotkey, revision) -> (cp, wins) for every eligible challenger.
        eligible: Dict[Tuple[str, str], Tuple[int, int]] = {}
        for s in all_stats:
            if s.get("challenge_status") != "sampling":
                continue
            wins = int(s.get("challenge_consecutive_wins", 0) or 0)
            if wins <= 0:
                continue
            hotkey = s.get("hotkey")
            revision = s.get("revision")
            if not hotkey or not revision:
                continue
            if (hotkey, revision) in seen:
                continue
            cp = int(s.get("challenge_checkpoints_passed", 0) or 0)
            eligible[(hotkey, revision)] = (cp, wins)

        # Pass 1: incumbents (already deployed → keep their slot).
        incumbents: List[Tuple[int, int, str, str]] = []
        incumbent_seen: Set[Tuple[str, str]] = set()
        for status in ("active", "deploying"):
            for d in await self.dao.list_by_status(status):
                key = (d.get("hotkey"), d.get("revision"))
                if key in seen or key in incumbent_seen or key not in eligible:
                    continue
                cp, wins = eligible[key]
                incumbents.append((cp, wins, key[0], key[1]))
                incumbent_seen.add(key)

        # Pass 2: newcomers (eligible but not yet deployed).
        newcomers: List[Tuple[int, int, str, str]] = []
        for key, (cp, wins) in eligible.items():
            if key in seen or key in incumbent_seen:
                continue
            newcomers.append((cp, wins, key[0], key[1]))

        # Within each group, sort by CP desc then wins desc for deterministic
        # ordering. Incumbents are admitted before any newcomer regardless of
        # ranking — that's what makes this a queue, not a priority reshuffle.
        incumbents.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
        newcomers.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
        ordered = incumbents + newcomers

        slots_left = self.max_deployments - len(targets)
        for cp, wins, hotkey, revision in ordered[:slots_left]:
            targets.append((hotkey, revision, "winner"))
            seen.add((hotkey, revision))

        if len(ordered) > slots_left:
            queued = ordered[slots_left:]
            logger.info(
                f"_resolve_targets: {len(queued)} eligible winner(s) queued "
                f"(waiting for slot): "
                + ", ".join(f"{hk[:8]}..@{rev[:6]} cp={cp} w={w}"
                            for cp, w, hk, rev in queued[:5])
            )
        return targets

    async def _reconcile_targets(self) -> None:
        """Ensure current target set is deployed and nothing else is."""
        targets = await self._resolve_targets()
        target_keys: Set[Tuple[str, str]] = {(hk, rev) for hk, rev, _ in targets}

        # Teardown FIRST so newly-freed slots can be re-used this cycle.
        await self._teardown_non_targets(target_keys)

        if not self.client.configured:
            return
        for hotkey, revision, role in targets:
            try:
                await self._ensure_target(hotkey, revision, role)
            except Exception as e:
                logger.error(
                    f"_reconcile_targets: ensure {role} {hotkey[:12]}...@{revision[:8]} "
                    f"failed: {e}",
                    exc_info=True,
                )

    async def _teardown_non_targets(
        self, target_keys: Set[Tuple[str, str]]
    ) -> None:
        """Delete any Targon deployment whose (hotkey, revision) is not in
        the current target set. Covers champion change, challenger
        termination, wins→0, and cap overflow.
        """
        if not self.client.configured:
            return
        for status in ("active", "deploying"):
            for d in await self.dao.list_by_status(status):
                key = (d.get("hotkey"), d.get("revision"))
                if key in target_keys:
                    continue
                deployment_id = d["deployment_id"]
                await self.client.delete_deployment(deployment_id)
                await self.dao.mark_deleted(deployment_id)
                logger.info(
                    f"Auto-release: deleted {deployment_id} "
                    f"hotkey={(key[0] or '')[:12]}.. rev={(key[1] or '')[:8]}.. "
                    f"(no longer a target)"
                )

    async def _ensure_target(
        self, hotkey: str, revision: str, role: str
    ) -> None:
        existing = await self.dao.get_by_hotkey_revision(hotkey, revision)
        if existing and existing.get("status") != "deleted":
            return

        miner = await self.miners_dao.get_miner_by_hotkey(hotkey)
        if not miner or not miner.get("model"):
            logger.warning(
                f"TargonDeployerService: {role} {hotkey[:12]}.. has no miner/model row"
            )
            return

        from affine.core.providers.targon_client import external_url
        model_hf_repo = miner["model"]
        uid = miner.get("uid")

        # Dedupe: if a Targon workload with our deterministic name already
        # exists (manual `af targon deploy`, or orphaned by a crashed writer),
        # adopt it instead of paying for a second GPU.
        adopted_id = await self._find_existing_on_targon(
            model_hf_repo, revision, uid=uid, hotkey=hotkey,
        )
        deployment_id = adopted_id or await self.client.create_deployment(
            model_hf_repo=model_hf_repo, revision=revision,
        )
        if not deployment_id:
            logger.warning(
                f"TargonDeployerService: no deployment_id for "
                f"{model_hf_repo}@{revision[:8]}"
            )
            return

        # RENTAL URL template is deterministic from deployment_id+port,
        # so we persist it immediately — no blind window between "workload
        # ready" and "router picks up the URL".
        base_url = external_url(deployment_id) + "/v1"
        await self.dao.upsert_deployment(
            deployment_id=deployment_id, hotkey=hotkey, revision=revision,
            model_hf_repo=model_hf_repo, image=self.client.default_image,
            base_url=base_url, instance_count=0, status="deploying",
            mount_path=self.client.data_volume_mount,
        )
        logger.info(
            f"{'Adopted' if adopted_id else 'Created'} Targon {deployment_id} "
            f"for {role} {hotkey[:12]}..@{revision[:8]}"
        )

    async def _find_existing_on_targon(
        self, model_hf_repo: str, revision: str,
        *, uid: Optional[int] = None, hotkey: Optional[str] = None,
    ) -> Optional[str]:
        """Look up a live Targon workload that matches our naming convention."""
        expected_name = self.client._workload_name(
            model_hf_repo, revision, uid=uid, hotkey=hotkey,
        )
        wls = await self.client.list_workloads(limit=100)
        for w in ((wls or {}).get("items", []) or []):
            if w.get("name") == expected_name:
                state = (w.get("state") or {}).get("status", "").lower()
                if state in {"running", "provisioning", "deploying", "rebuilding"}:
                    return w.get("uid")
        return None

    async def _health_sweep(self) -> None:
        """Poll every live deployment; release any that's no longer healthy.

        Targon may silently rebuild a container, which wipes the /data volume
        (weights lost). Rather than try to restart/rebuild in place and trust
        stale cache, we just delete dead deployments and let the next
        _reconcile_targets cycle decide what to redeploy. The affected miner
        re-enters the queue — if it's still the top candidate it redeploys
        immediately; if a higher-priority winner is waiting, that one takes
        the slot instead.
        """
        if not self.client.configured:
            return
        active = await self.dao.list_by_status("active")
        deploying = await self.dao.list_by_status("deploying")
        for d in list(active) + list(deploying):
            await self._check_and_release(d)

    # Consecutive failed probes before we conclude a deployment is truly dead.
    # Debounces transient Targon state-aggregator lag (~30s) without waiting
    # long enough to matter to the miner. With poll_interval=30s, release
    # happens ~60s after the first failed probe.
    UNHEALTHY_FAIL_THRESHOLD = 2

    async def _check_and_release(self, deployment: Dict[str, Any]) -> None:
        deployment_id = deployment["deployment_id"]
        status = await self.client.get_deployment_status(deployment_id)
        if status is None:
            await self._on_failed_probe(deployment, reason="status=None")
            return

        running = int(status.get("running_instances", 0) or 0)
        healthy = bool(status.get("healthy", False))
        base_url = status.get("base_url")

        if running > 0 and healthy:
            # Resets consecutive_failures = 0 via the DAO.
            await self.dao.update_health(
                deployment_id, instance_count=running, healthy=True, base_url=base_url,
            )
            return

        # Grace for deployments that are still initializing. We never release
        # a freshly-created row on its first few probes — Targon's state
        # aggregator needs time to reflect "running".
        age = int(time.time()) - int(deployment.get("created_at", 0) or 0)
        if deployment.get("status") == "deploying" and age < 600:
            return

        await self._on_failed_probe(
            deployment, reason=f"running={running} healthy={healthy} age={age}s",
        )

    async def _on_failed_probe(
        self, deployment: Dict[str, Any], *, reason: str
    ) -> None:
        deployment_id = deployment["deployment_id"]
        next_failures = int(deployment.get("consecutive_failures", 0) or 0) + 1
        if next_failures >= self.UNHEALTHY_FAIL_THRESHOLD:
            await self._release_dead(
                deployment,
                reason=f"{reason} (failures={next_failures}/{self.UNHEALTHY_FAIL_THRESHOLD})",
            )
            return
        await self.dao.increment_failure(deployment_id)
        logger.info(
            f"Targon {deployment_id} probe failed "
            f"{next_failures}/{self.UNHEALTHY_FAIL_THRESHOLD} — {reason}"
        )

    async def _release_dead(
        self, deployment: Dict[str, Any], *, reason: str
    ) -> None:
        deployment_id = deployment["deployment_id"]
        await self.client.delete_deployment(deployment_id)
        await self.dao.mark_deleted(deployment_id)
        logger.warning(
            f"Released Targon {deployment_id} "
            f"hotkey={(deployment.get('hotkey') or '')[:12]}.. "
            f"rev={(deployment.get('revision') or '')[:8]}.. — {reason}; "
            f"miner re-enters queue next cycle"
        )

