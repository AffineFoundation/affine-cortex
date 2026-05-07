"""Targon deployer reconciliation loop.

Closed-loop auto-deployment over a small set of "targets" drawn from the
current champion plus the strongest currently-sampling challengers:

    targets = { champion } ∪ top-N challengers ranked by
                  (consecutive_wins > 0  DESC,   # winners first
                   first_block           ASC)    # within tier, earliest-submitted first
              where challenge_status == 'sampling'
    |targets| ≤ MAX_TARGON_DEPLOYMENTS   (default 8, env-configurable)

Each loop iteration:

    1. Resolve targets from system_config.champion + miner_stats.
    2. For every target: ensure a Targon deployment exists (adopt matching
       live workload if present; otherwise create).
    3. Tear down any deployment whose (hotkey, revision) is no longer a
       target — covers champion change, challenger termination, challenger
       losing its only win, etc. Teardown is immediate (no 24h grace).
    4. Health sweep: probe every active/deploying deployment. The
       authoritative signal is an active OpenAI /v1/models probe against
       the workload — Targon's container-level state lags vLLM in both
       directions (ready_replicas can stay 0 for a minute after the model
       has actually loaded, and stays 1 after a rebuild has wiped /data
       while vLLM is re-downloading), so trusting it would either miss
       a serving deployment or keep routing to a broken one. /v1/models
       is exactly what task fetch sees when it routes traffic.

       Release is time-based, not count-based: a deployment must be
       continuously unhealthy for at least TARGON_UNHEALTHY_RELEASE_SEC
       (default 600s = 10 min) before we delete it. Targon's CDN has
       occasional minute-long network blips and miner-side vLLM can pause
       for a few minutes during prefill — a faster trigger would churn
       perfectly fine deployments. Once released, the miner re-enters the
       queue next cycle.

       Freshly-created deployments get TARGON_INITIAL_LOAD_GRACE_SEC
       (default 1800s) to complete their first model download before the
       failure clock even starts — long enough for typical Affine miner
       models (~30B params), short enough that a stuck container doesn't
       waste a whole day.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

from affine.core.providers.targon_client import TargonClient, get_targon_client
from affine.core.setup import logger
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.miners import MinersDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.database.dao.targon_deployments import TargonDeploymentsDAO


DEFAULT_POLL_INTERVAL = int(os.getenv("TARGON_POLL_INTERVAL_SEC", "30"))
DEFAULT_MAX_DEPLOYMENTS = int(os.getenv("TARGON_MAX_DEPLOYMENTS", "8"))
# How often to scan all Targon-side workloads for orphans (workloads that
# match our naming convention but have no DB row). Cheaper than per-cycle
# because it's bounded by Targon's list endpoint size.
DEFAULT_ORPHAN_SWEEP_INTERVAL = int(os.getenv("TARGON_ORPHAN_SWEEP_SEC", "300"))

# How long a deployment may stay in 'deploying' state before the failure
# counter starts ticking. Sized for first-time HF weight download on a
# fresh /data volume — typical Affine fine-tunes (~30B params, fp16) take
# 10-25 min over Targon's network. After this window a still-loading
# deployment is released so we don't sit on a stuck container indefinitely.
DEFAULT_INITIAL_LOAD_GRACE = int(os.getenv("TARGON_INITIAL_LOAD_GRACE_SEC", "1800"))
# Timeout for the OpenAI /v1/models probe. Steady-state is sub-100ms; the
# generous default just covers TLS setup on the first probe of a new URL.
DEFAULT_MODEL_PROBE_TIMEOUT = float(os.getenv("TARGON_MODEL_PROBE_TIMEOUT_SEC", "5"))
# A deployment must be continuously unhealthy for at least this long before
# we tear it down. Targon's network reaches us through a CDN that has its
# own occasional blips (and miner-side vLLM can pause for a few minutes
# during prefill on a long context); a too-eager release would churn
# perfectly fine deployments. 10 min is well past every transient we've
# observed but far short of a real rebuild's recovery time.
DEFAULT_UNHEALTHY_RELEASE_SEC = int(os.getenv("TARGON_UNHEALTHY_RELEASE_SEC", "600"))


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
        self.orphan_sweep_interval = DEFAULT_ORPHAN_SWEEP_INTERVAL
        self._last_orphan_sweep_at = 0
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
                await self._orphan_sweep()
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

        Champion gets slot 0; remaining slots are filled from sampling
        miners ranked by:

            (has_wins   DESC,   # consecutive_wins > 0 → True before False
             first_block ASC)   # within a tier, earliest submitter first

        Eligibility filter (applies to *both* champion and challengers):
          - is_valid == 'true'                   (validator admits it)
          - chute_status == 'hot'                (Chutes is serving)
          - challenge_status == 'sampling'       (challengers only)
          - (hotkey, revision) != champion's     (no double counting)

        Why is_valid is required, not just chute_hot: the chute_hot
        filter catches the *common* invalidation paths because most
        of them clear chute_status as a side effect (chute_fetch_failed,
        chute_slug_empty, chute_not_hot). But anticopy, model_mismatch,
        repo-name violations, revision_mismatch, hf_model_fetch_failed,
        and multiple_commits all leave chute_status='hot' while flipping
        is_valid='false'. Sampling scheduler also gates on is_valid via
        get_valid_miners(), so without the same gate here the deployer
        burns GPU on miners no one is actually sampling. UID 187's
        anticopy case is the canonical example.

        Why is_valid applies to champion too: dethrone-protection is
        meant to ride out a *transient Chutes blip* — short hiccups
        in the network/CDN. validator-side invalidation (anticopy,
        repo-name) is deterministic, not transient, and sampling
        scheduler stops sampling the champion anyway. Keeping the
        deployment costs GPU-hours for nothing. Once is_valid flips
        back to 'true', the deployer recreates the slot on the next
        reconcile.

        Why Chutes-hot is a hard filter for challengers: a cold Chutes
        is the cheapest "miner has effectively dropped offline" signal
        we have. Spending Targon GPU-hours on those slots throws money
        away — the miner is likely to churn soon. We'd rather leave
        the slot empty than waste it.

        Side effect: a previously-eligible incumbent that goes
        invalid/cold falls out of the target set and gets torn down on
        this reconcile, freeing the slot for the next queued candidate.

        Returns list of (hotkey, revision, role) where role ∈ {'champion','winner'}.
        """
        targets: List[Tuple[str, str, str]] = []
        seen: Set[Tuple[str, str]] = set()

        # Per-hotkey (first_block, chute_status, is_valid) from the
        # miners table. Loaded up front because both the champion check
        # and the challenger eligibility loop need is_valid + chute_hot.
        # Bounded to ~256 rows → one scan is cheaper than per-hotkey lookups.
        miners = await self.miners_dao.get_all_miners()
        first_block_by_hotkey: Dict[str, int] = {}
        chute_hot_by_hotkey: Dict[str, bool] = {}
        is_valid_by_hotkey: Dict[str, bool] = {}
        for m in miners:
            hk = m.get("hotkey")
            if not hk:
                continue
            try:
                first_block_by_hotkey[hk] = int(m.get("first_block") or 0)
            except (TypeError, ValueError):
                pass
            chute_hot_by_hotkey[hk] = (m.get("chute_status") or "").lower() == "hot"
            # is_valid is a string column ('true' / 'false') because it
            # backs a GSI partition key; coerce defensively.
            is_valid_by_hotkey[hk] = str(m.get("is_valid") or "").lower() == "true"

        champion = await self.config_dao.get_param_value("champion", default=None)
        if (
            champion
            and champion.get("hotkey")
            and champion.get("revision")
            and is_valid_by_hotkey.get(champion["hotkey"], False)
        ):
            champ_key = (champion["hotkey"], champion["revision"])
            targets.append((champ_key[0], champ_key[1], "champion"))
            seen.add(champ_key)

        if self.max_deployments <= 1:
            return targets

        all_stats = await self.miner_stats_dao.get_all_historical_miners()

        # (hotkey, revision) -> (has_wins, first_block) for hot, valid,
        # sampling miners. Cold/invalid miners are filtered out entirely
        # (not just deprioritized) so they release their existing slot.
        eligible: Dict[Tuple[str, str], Tuple[bool, int]] = {}
        for s in all_stats:
            if s.get("challenge_status") != "sampling":
                continue
            hotkey = s.get("hotkey")
            revision = s.get("revision")
            if not hotkey or not revision:
                continue
            if (hotkey, revision) in seen:
                continue
            if not chute_hot_by_hotkey.get(hotkey, False):
                continue
            if not is_valid_by_hotkey.get(hotkey, False):
                continue
            wins = int(s.get("challenge_consecutive_wins", 0) or 0)
            has_wins = wins > 0
            first_block = first_block_by_hotkey.get(hotkey)
            if first_block is None or first_block <= 0:
                first_block = 10**12  # sentinel: unknown sorts last
            eligible[(hotkey, revision)] = (has_wins, first_block)

        # Pass 1: incumbents (already deployed → keep their slot).
        # Giving them admission priority turns the cap into a FIFO queue:
        # once a hot, sampling miner gets a slot, it keeps it until it
        # drops out voluntarily (terminated / chute went cold / champion
        # change / cap shrunk).
        incumbents: List[Tuple[bool, int, str, str]] = []
        incumbent_seen: Set[Tuple[str, str]] = set()
        for status in ("active", "deploying"):
            for d in await self.dao.list_by_status(status):
                key = (d.get("hotkey"), d.get("revision"))
                if key in seen or key in incumbent_seen or key not in eligible:
                    continue
                has_wins, first_block = eligible[key]
                incumbents.append((has_wins, first_block, key[0], key[1]))
                incumbent_seen.add(key)

        # Pass 2: newcomers (eligible but not yet deployed).
        newcomers: List[Tuple[bool, int, str, str]] = []
        for key, (has_wins, first_block) in eligible.items():
            if key in seen or key in incumbent_seen:
                continue
            newcomers.append((has_wins, first_block, key[0], key[1]))

        # Composite sort: has_wins first (True before False), then
        # first_block ascending. Hotkey/revision suffixes break further
        # ties deterministically.
        sort_key = lambda t: (0 if t[0] else 1, t[1], t[2], t[3])
        incumbents.sort(key=sort_key)
        newcomers.sort(key=sort_key)
        ordered = incumbents + newcomers

        slots_left = self.max_deployments - len(targets)
        for has_wins, first_block, hotkey, revision in ordered[:slots_left]:
            targets.append((hotkey, revision, "winner"))
            seen.add((hotkey, revision))

        if len(ordered) > slots_left:
            queued = ordered[slots_left:]
            logger.info(
                f"_resolve_targets: {len(queued)} hot+sampling miner(s) queued "
                f"(waiting for slot): "
                + ", ".join(
                    f"{hk[:8]}..@{rev[:6]} wins={'y' if hw else 'n'} fb={fb}"
                    for hw, fb, hk, rev in queued[:5]
                )
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
        uid = miner.get("uid")
        if uid is None:
            # Without uid we can't build a unique workload name — bail
            # rather than silently fall back to a model-only name that
            # would collide whenever two miners share the same model.
            logger.warning(
                f"TargonDeployerService: {role} {hotkey[:12]}.. miner row missing uid"
            )
            return

        from affine.core.providers.targon_client import (
            external_url, derive_deployment_args_from_chute, fixed_gpu_count,
            resource_name_for, TARGON_GPU_TYPE,
        )
        from affine.utils.api_client import get_chute_info
        model_hf_repo = miner["model"]
        chute_id = miner.get("chute_id")

        # Mirror the miner's Chutes config (gpu_count, engine, resource tier)
        # on Targon so tensor parallelism and runtime match what the miner
        # actually built for. Launch-args not exposed by the Chutes API
        # (max-model-len, mem-fraction, …) stay at env defaults.
        chute_args = derive_deployment_args_from_chute(
            await get_chute_info(chute_id) if chute_id else None
        )

        # Operator override: when TARGON_FIXED_GPU_COUNT is set we ignore the
        # chute's gpu_count and pin every workload to the configured size, so
        # a fixed Targon pool can be split evenly across N miners regardless
        # of how each miner happened to size their chute.
        forced = fixed_gpu_count()
        if forced is not None:
            chute_args["gpu_count"] = forced
            resolved = resource_name_for(TARGON_GPU_TYPE, forced)
            if resolved:
                chute_args["resource_name"] = resolved

        # Dedupe: if a Targon workload with our deterministic name already
        # exists (manual `af targon deploy`, or orphaned by a crashed writer),
        # adopt it instead of paying for a second GPU.
        adopted_id = await self._find_existing_on_targon(
            model_hf_repo, revision, uid=uid, hotkey=hotkey,
        )
        deployment_id = adopted_id or await self.client.create_deployment(
            model_hf_repo=model_hf_repo, revision=revision,
            uid=uid, hotkey=hotkey, **chute_args,
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
        *, uid: int, hotkey: str,
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

        Skips the entire sweep when the Targon API itself is unreachable —
        otherwise a transient platform outage would cause every deployment
        to return ``status=None`` simultaneously and trip the failure
        threshold, mass-releasing the whole pool.
        """
        if not self.client.configured:
            return
        if not await self._targon_api_healthy():
            logger.warning(
                "Targon API probe failed; skipping health sweep this cycle "
                "(no deployments will be released until the API recovers)"
            )
            return
        active = await self.dao.list_by_status("active")
        deploying = await self.dao.list_by_status("deploying")
        for d in list(active) + list(deploying):
            await self._check_and_release(d)

    async def _targon_api_healthy(self) -> bool:
        """Lightweight probe: list 1 workload. Returns False on any
        exception or None response — those mean the API as a whole is
        unreachable and per-deployment status calls would just repeat the
        same false signal.
        """
        try:
            result = await self.client.list_workloads(limit=1)
        except Exception as e:
            logger.warning(f"Targon API probe raised: {e}")
            return False
        return result is not None

    async def _orphan_sweep(self) -> None:
        """Delete Targon workloads that match our naming prefix but have
        no row in ``targon_deployments``.

        Catches the cases where ``create_deployment`` succeeded on Targon
        but ``upsert_deployment`` failed locally (so the workload runs
        forever billing nobody), where an operator hand-deleted a DB row,
        or where a manual ``af targon deploy`` was abandoned without a
        matching target.

        Throttled by ``self.orphan_sweep_interval`` (default 5 min) so we
        don't pull a 200-item list every poll. We never delete a workload
        whose name doesn't start with the affine prefix, so this is safe
        in shared Targon accounts.
        """
        if not self.client.configured:
            return
        now = int(time.time())
        if now - self._last_orphan_sweep_at < self.orphan_sweep_interval:
            return

        try:
            listing = await self.client.list_workloads(limit=200)
        except Exception as e:
            logger.warning(f"Orphan-sweep: list_workloads raised: {e}")
            return
        if listing is None:
            # API unhealthy — try again next interval, don't update timestamp
            # so we don't skip a real sweep.
            return
        self._last_orphan_sweep_at = now

        items = (listing.get("items") if isinstance(listing, dict) else listing) or []
        if not items:
            return

        db_uids: Set[str] = set()
        for status in ("active", "deploying"):
            for d in await self.dao.list_by_status(status):
                db_uids.add(d["deployment_id"])

        prefix = self.client.WORKLOAD_NAME_PREFIX + "-"
        deleted = 0
        skipped_unmanaged = 0
        for w in items:
            wid = w.get("uid") or w.get("id")
            name = (w.get("name") or "")
            if not wid:
                continue
            if not name.startswith(prefix):
                skipped_unmanaged += 1
                continue
            if wid in db_uids:
                continue
            try:
                ok = await self.client.delete_deployment(wid)
                if ok:
                    deleted += 1
                    logger.warning(
                        f"Orphan-sweep: deleted untracked Targon workload "
                        f"{wid} (name={name})"
                    )
                else:
                    logger.warning(
                        f"Orphan-sweep: delete returned false for {wid} (name={name})"
                    )
            except Exception as e:
                logger.error(f"Orphan-sweep: failed to delete {wid}: {e}")

        if deleted or skipped_unmanaged:
            logger.info(
                f"Orphan-sweep: scanned {len(items)} workload(s), "
                f"deleted {deleted}, skipped {skipped_unmanaged} non-affine"
            )

    # Floor on the number of failed probes before release becomes possible.
    # The release decision is primarily *time*-based (see
    # DEFAULT_UNHEALTHY_RELEASE_SEC); this counter just guards against the
    # edge case where last_health_check_at is already old when we observe
    # a single transient blip — without it, one bad probe after a long
    # deployer outage would tear down an otherwise-fine deployment.
    UNHEALTHY_FAIL_FLOOR = 2

    async def _probe_model_ready(self, base_url: Optional[str]) -> bool:
        """Active inference-layer probe — true iff /v1/models lists ≥1 model.

        Targon's container-level "healthy" only verifies port connectivity,
        so a freshly-rebuilt workload with /data wiped passes Targon's check
        while vLLM/sglang is still downloading weights from HF. We need a
        real signal from the OpenAI server itself to tell "warming up" or
        "rebuild in progress" apart from "ready to serve".

        Any non-200 response, network error, timeout, or empty data list
        returns False — callers feed that into the same failure counter
        the container probe uses.
        """
        if not base_url:
            return False
        url = base_url.rstrip("/") + "/models"
        try:
            timeout = aiohttp.ClientTimeout(total=DEFAULT_MODEL_PROBE_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return False
                    payload = await resp.json(content_type=None)
        except Exception as e:
            logger.debug(f"_probe_model_ready({url}): {type(e).__name__}: {e}")
            return False
        if not isinstance(payload, dict):
            return False
        data = payload.get("data")
        return isinstance(data, list) and len(data) > 0

    async def _check_and_release(self, deployment: Dict[str, Any]) -> None:
        deployment_id = deployment["deployment_id"]
        status = await self.client.get_deployment_status(deployment_id)
        if status is None:
            await self._on_failed_probe(deployment, reason="status=None")
            return

        running = int(status.get("running_instances", 0) or 0)
        healthy = bool(status.get("healthy", False))
        base_url = status.get("base_url") or deployment.get("base_url")

        # Active inference probe is the authoritative readiness signal.
        # Targon's ready_replicas can lag behind vLLM in *either* direction:
        # it stays 0 for a minute after the model has actually loaded
        # (so trusting Targon means tasks miss a serving deployment), and
        # it can also stay 1 after a rebuild has wiped /data (the rebuild
        # case our probe was added for). Either way, /v1/models is the
        # signal that matches what task fetch will actually see when it
        # routes traffic, so we make it the only signal that matters.
        model_ready = await self._probe_model_ready(base_url)

        if model_ready:
            # Probe success is sufficient — synthesize at least 1 instance
            # so update_health flips status to 'active' even when Targon's
            # bookkeeping is still catching up.
            await self.dao.update_health(
                deployment_id,
                instance_count=max(running, 1),
                healthy=True,
                base_url=base_url,
            )
            return

        age = int(time.time()) - int(deployment.get("created_at", 0) or 0)
        db_status = deployment.get("status", "")

        # Initial-load grace — a brand-new deployment hasn't yet successfully
        # served any inference, so its row is still 'deploying'. We hold off
        # on the failure counter until the grace window expires so a slow
        # first-time HF download doesn't get released mid-pull.
        if db_status == "deploying" and age < DEFAULT_INITIAL_LOAD_GRACE:
            return

        await self._on_failed_probe(
            deployment,
            reason=(
                f"running={running} healthy={healthy} model_ready=False "
                f"age={age}s db_status={db_status}"
            ),
        )

    async def _on_failed_probe(
        self, deployment: Dict[str, Any], *, reason: str
    ) -> None:
        """Time-based release decision.

        ``last_health_check_at`` is bumped only by update_health() in the
        green-on-both-signals path, so it doubles as "last time this
        deployment was confirmed serving". A deployment that has never
        been confirmed serving falls back to ``created_at`` so post-grace
        stuck-loading workloads are still eligible for release.

        Release happens iff the deployment has been continuously unhealthy
        for at least DEFAULT_UNHEALTHY_RELEASE_SEC *and* the failure floor
        is met (avoids releasing on a single blip after a long deployer
        downtime, where the timestamp gap is misleading).
        """
        deployment_id = deployment["deployment_id"]
        now = int(time.time())
        last_healthy_at = (
            int(deployment.get("last_health_check_at", 0) or 0)
            or int(deployment.get("created_at", 0) or 0)
            or now
        )
        unhealthy_for = max(0, now - last_healthy_at)
        next_failures = int(deployment.get("consecutive_failures", 0) or 0) + 1

        if (
            unhealthy_for >= DEFAULT_UNHEALTHY_RELEASE_SEC
            and next_failures >= self.UNHEALTHY_FAIL_FLOOR
        ):
            await self._release_dead(
                deployment,
                reason=(
                    f"{reason} "
                    f"(unhealthy_for={unhealthy_for}s/"
                    f"{DEFAULT_UNHEALTHY_RELEASE_SEC}s, failures={next_failures})"
                ),
            )
            return
        await self.dao.increment_failure(deployment_id)
        logger.info(
            f"Targon {deployment_id} probe failed "
            f"(failures={next_failures}, unhealthy_for={unhealthy_for}s/"
            f"{DEFAULT_UNHEALTHY_RELEASE_SEC}s) — {reason}"
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

