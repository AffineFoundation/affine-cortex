"""Targon deployer reconciliation loop.

Stage-1 target: the validator-selected champion. The loop runs forever:

    1. Read champion from system_config.
    2. Ensure a Targon deployment exists for that hotkey+revision. If missing,
       create one (retrying with exponential backoff on failure).
    3. Sweep all active deployments and refresh their health in the DB. If a
       deployment becomes unhealthy, attempt restart_container; if that fails,
       fall back to delete+recreate (which re-uses /data, so weights stay
       cached).

Stage-2 will change `_reconcile_targets` to accept a list of targets rather
than just the champion, leaving the rest of the loop untouched.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from affine.core.providers.targon_client import TargonClient, get_targon_client
from affine.core.setup import logger
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.database.dao.targon_deployments import TargonDeploymentsDAO


DEFAULT_POLL_INTERVAL = int(os.getenv("TARGON_POLL_INTERVAL_SEC", "30"))
DEFAULT_MAX_REBUILD_RETRIES = int(os.getenv("TARGON_MAX_REBUILD_RETRIES", "5"))
DEFAULT_OLD_CHAMPION_TTL_HOURS = float(os.getenv("TARGON_OLD_CHAMPION_TTL_HOURS", "24"))


class TargonDeployerService:
    def __init__(
        self,
        targon_dao: Optional[TargonDeploymentsDAO] = None,
        config_dao: Optional[SystemConfigDAO] = None,
        miners_dao: Optional[MinersDAO] = None,
        client: Optional[TargonClient] = None,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        max_rebuild_retries: int = DEFAULT_MAX_REBUILD_RETRIES,
        old_champion_ttl_hours: float = DEFAULT_OLD_CHAMPION_TTL_HOURS,
    ):
        self.dao = targon_dao or TargonDeploymentsDAO()
        self.config_dao = config_dao or SystemConfigDAO()
        self.miners_dao = miners_dao or MinersDAO()
        self.client = client or get_targon_client()
        self.poll_interval = poll_interval
        self.max_rebuild_retries = max_rebuild_retries
        self.old_champion_ttl_seconds = int(old_champion_ttl_hours * 3600)
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        if not self.client.configured:
            logger.warning(
                "TargonDeployerService: TARGON_API_KEY/URL not set — reconciler "
                "idle. Set env vars to activate."
            )
        logger.info(
            f"TargonDeployerService starting (interval={self.poll_interval}s, "
            f"max_rebuild_retries={self.max_rebuild_retries})"
        )
        while not self._stop_event.is_set():
            try:
                champion = await self.config_dao.get_param_value("champion", default=None)
                if champion:
                    await self._reconcile_target(champion)
                await self._health_sweep()
                await self._gc_old_deployments(champion)
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

    async def _reconcile_target(self, champion: Dict[str, Any]) -> None:
        hotkey = champion.get("hotkey")
        revision = champion.get("revision")
        if not hotkey or not revision:
            return

        existing = await self.dao.get_by_hotkey_revision(hotkey, revision)
        if existing and existing.get("status") not in {"failed", "deleted"}:
            return
        if existing and existing.get("status") == "failed":
            next_retry = int(existing.get("next_retry_at", 0) or 0)
            if time.time() < next_retry:
                return
            await self._rebuild(existing)
            return

        miner = await self.miners_dao.get_miner_by_hotkey(hotkey)
        if not miner:
            logger.warning(
                f"TargonDeployerService: champion hotkey {hotkey[:12]}... not in miners table"
            )
            return
        model_hf_repo = miner.get("model")
        if not model_hf_repo:
            logger.warning(f"TargonDeployerService: champion miner has no model")
            return

        if not self.client.configured:
            return

        # Dedupe across restarts: if a Targon workload with our deterministic
        # name already exists (e.g. created manually via `af targon deploy-uid`
        # or left over from a prior reconciler run whose DB write failed),
        # adopt it instead of paying for a second GPU.
        uid = champion.get("uid")
        adopted_id = await self._find_existing_on_targon(
            model_hf_repo, revision, uid=uid, hotkey=hotkey,
        )
        if adopted_id:
            from affine.core.providers.targon_client import external_url
            base_url = external_url(adopted_id) + "/v1"
            await self.dao.upsert_deployment(
                deployment_id=adopted_id, hotkey=hotkey, revision=revision,
                model_hf_repo=model_hf_repo, image=self.client.default_image,
                base_url=base_url, instance_count=0, status="deploying",
                mount_path=self.client.data_volume_mount,
            )
            logger.info(f"Adopted existing Targon workload {adopted_id} for champion")
            return

        try:
            deployment_id = await self.client.create_deployment(
                model_hf_repo=model_hf_repo, revision=revision,
            )
            if not deployment_id:
                logger.warning(
                    f"TargonDeployerService: create_deployment returned no id for "
                    f"{model_hf_repo}@{revision[:8]}"
                )
                return
            # RENTAL URL template is deterministic from deployment_id+port,
            # so we can persist base_url immediately and avoid a blind window
            # between "workload ready" and "router picks up the URL".
            from affine.core.providers.targon_client import external_url
            base_url = external_url(deployment_id) + "/v1"
            await self.dao.upsert_deployment(
                deployment_id=deployment_id, hotkey=hotkey, revision=revision,
                model_hf_repo=model_hf_repo, image=self.client.default_image,
                base_url=base_url,
                instance_count=0, status="deploying",
                mount_path=self.client.data_volume_mount,
            )
            logger.info(
                f"Created Targon deployment {deployment_id} for champion "
                f"{hotkey[:12]}...@{revision[:8]} base_url={base_url}"
            )
        except Exception as e:
            logger.error(f"TargonDeployerService: deploy failed: {e}", exc_info=True)

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
        if not self.client.configured:
            return
        active = await self.dao.list_by_status("active")
        deploying = await self.dao.list_by_status("deploying")
        rebuilding = await self.dao.list_by_status("rebuilding")
        for d in list(active) + list(deploying) + list(rebuilding):
            await self._refresh_one(d)

    async def _refresh_one(self, deployment: Dict[str, Any]) -> None:
        deployment_id = deployment["deployment_id"]
        status = await self.client.get_deployment_status(deployment_id)
        if status is None:
            await self._handle_unhealthy(deployment)
            return

        running = int(status.get("running_instances", 0) or 0)
        healthy = bool(status.get("healthy", False))
        base_url = status.get("base_url")

        if running > 0 and healthy:
            await self.dao.update_health(
                deployment_id, instance_count=running, healthy=True, base_url=base_url,
            )
            return

        await self._handle_unhealthy(deployment)

    async def _handle_unhealthy(self, deployment: Dict[str, Any]) -> None:
        deployment_id = deployment["deployment_id"]
        failures = int(deployment.get("consecutive_failures", 0) or 0)
        if failures >= self.max_rebuild_retries:
            await self.dao.set_status(deployment_id, "failed")
            logger.error(
                f"Targon deployment {deployment_id} failed after "
                f"{failures} attempts; router will fall back to Chutes."
            )
            return
        await self._rebuild(deployment)

    async def _rebuild(self, deployment: Dict[str, Any]) -> None:
        deployment_id = deployment["deployment_id"]
        await self.dao.set_status(deployment_id, "rebuilding")
        try:
            ok = await self.client.restart_container(deployment_id)
            if not ok:
                logger.warning(
                    f"Targon restart_container failed for {deployment_id}, "
                    f"falling back to delete+recreate"
                )
                await self.client.delete_deployment(deployment_id)
                new_id = await self.client.create_deployment(
                    model_hf_repo=deployment["model_hf_repo"],
                    revision=deployment["revision"],
                )
                if new_id and new_id != deployment_id:
                    await self.dao.mark_deleted(deployment_id)
                    await self.dao.upsert_deployment(
                        deployment_id=new_id,
                        hotkey=deployment["hotkey"],
                        revision=deployment["revision"],
                        model_hf_repo=deployment["model_hf_repo"],
                        image=deployment.get("image"),
                        instance_count=0,
                        status="deploying",
                        mount_path=deployment.get(
                            "mount_path", self.client.data_volume_mount
                        ),
                    )
                    return
            await self.dao.increment_failure(deployment_id)
        except Exception as e:
            logger.error(f"Rebuild failed for {deployment_id}: {e}", exc_info=True)
            await self.dao.increment_failure(deployment_id)

    async def _gc_old_deployments(self, current_champion: Optional[Dict[str, Any]]) -> None:
        """Tear down deployments that no longer match the current champion.

        Stage-1 rule: any active/rebuilding deployment whose (hotkey, revision)
        differs from the current champion and whose updated_at is older than
        TARGON_OLD_CHAMPION_TTL_HOURS gets deleted to free GPU hours.
        """
        if not self.client.configured:
            return
        champ_key = (
            (current_champion.get("hotkey"), current_champion.get("revision"))
            if current_champion else (None, None)
        )
        cutoff = int(time.time()) - self.old_champion_ttl_seconds
        for status in ("active", "rebuilding", "failed"):
            for d in await self.dao.list_by_status(status):
                if (d.get("hotkey"), d.get("revision")) == champ_key:
                    continue
                if int(d.get("updated_at", 0) or 0) > cutoff:
                    continue
                try:
                    await self.client.delete_deployment(d["deployment_id"])
                    await self.dao.mark_deleted(d["deployment_id"])
                    logger.info(
                        f"GC: deleted stale Targon deployment {d['deployment_id']} "
                        f"(hotkey={d.get('hotkey', '')[:12]}..., revision={d.get('revision','')[:8]}...)"
                    )
                except Exception as e:
                    logger.error(f"GC delete failed for {d['deployment_id']}: {e}")
