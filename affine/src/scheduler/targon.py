"""
Targon workload lifecycle for the queue-window scheduler.

Replaces the old long-running ``targon_deployer`` reconciler. In the new
window model the scheduler knows up front exactly which two miners need
inference (champion + challenger), so we just adopt-or-create the two
matching workloads at INIT and delete them at WEIGHT_SET. No top-N pool,
no health-sweep reconciler, no ``targon_deployments`` DAO.

Helpers preserved from the old service (logic intact, dependencies stripped):
  - ``find_existing_workload`` — adopt a live workload by naming convention
  - ``probe_model_ready``      — true iff /v1/models lists ≥1 model
  - ``wait_for_ready``         — block until ready or timeout
  - ``orphan_sweep``           — drop workloads we no longer track
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import aiohttp

from affine.core.providers.targon_client import (
    TargonClient,
    external_url,
    fixed_gpu_count,
)
from affine.core.setup import logger


# Default timings — overridable per-call.
DEFAULT_PROBE_TIMEOUT_SEC = 5.0
DEFAULT_READY_POLL_INTERVAL_SEC = 15.0
DEFAULT_READY_DEADLINE_SEC = 1800  # 30 min from first probe to "ready"


@dataclass(frozen=True)
class DeployTarget:
    """One miner that needs inference for a window."""
    uid: int
    hotkey: str
    model: str
    revision: str


@dataclass
class MachineDeployment:
    endpoint_name: str
    deployment_id: str
    base_url: str


@dataclass
class DeployResult:
    """What ``deploy`` returns. Stored in ``ChampionRecord.deployment_id``
    or ``BattleRecord.deployment_id`` so a recovery run can re-adopt
    without restarting the workload."""
    deployment_id: str
    base_url: str
    deployments: List[MachineDeployment] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.deployments and self.deployment_id and self.base_url:
            self.deployments = [
                MachineDeployment(
                    endpoint_name="",
                    deployment_id=self.deployment_id,
                    base_url=self.base_url,
                )
            ]


async def find_existing_workload(
    client: TargonClient, target: DeployTarget
) -> Optional[str]:
    """Return the live workload id matching ``target``'s naming convention,
    or None if no matching live workload exists.

    A "live" workload is one whose state is running / provisioning /
    deploying / rebuilding — i.e. not terminated or in error.
    """
    try:
        expected_name = client._workload_name(
            target.model, target.revision, uid=target.uid, hotkey=target.hotkey,
        )
    except Exception as e:
        logger.warning(f"targon.find_existing_workload: name derive failed: {e}")
        return None
    try:
        wls = await client.list_workloads(limit=200)
    except Exception as e:
        logger.warning(f"targon.find_existing_workload: list_workloads raised: {e}")
        return None
    for w in ((wls or {}).get("items", []) or []):
        if w.get("name") != expected_name:
            continue
        state = (w.get("state") or {}).get("status", "").lower()
        if state in {"running", "provisioning", "deploying", "rebuilding"}:
            return w.get("uid")
    return None


async def probe_model_ready(
    base_url: Optional[str], *, timeout_sec: float = DEFAULT_PROBE_TIMEOUT_SEC,
) -> bool:
    """True iff ``GET base_url/models`` returns 200 with a non-empty ``data``.

    Targon's container-level health only verifies port connectivity, so a
    freshly-rebuilt workload with weights wiped passes that check while
    vLLM/sglang is still downloading. We need a signal from the OpenAI
    layer itself.
    """
    if not base_url:
        return False
    url = base_url.rstrip("/") + "/models"
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_sec)
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return False
                payload = await resp.json(content_type=None)
    except Exception as e:
        logger.debug(f"targon.probe_model_ready({url}): {type(e).__name__}: {e}")
        return False
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    return isinstance(data, list) and len(data) > 0


async def wait_for_ready(
    base_url: str,
    *,
    deadline_sec: int = DEFAULT_READY_DEADLINE_SEC,
    poll_interval_sec: float = DEFAULT_READY_POLL_INTERVAL_SEC,
    probe_timeout_sec: float = DEFAULT_PROBE_TIMEOUT_SEC,
) -> None:
    """Block until ``probe_model_ready(base_url)`` returns True or deadline
    elapses. Raises ``TimeoutError`` on deadline."""
    deadline = time.monotonic() + deadline_sec
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if await probe_model_ready(base_url, timeout_sec=probe_timeout_sec):
            logger.info(
                f"targon: model ready after {attempt} probes ({base_url})"
            )
            return
        await asyncio.sleep(poll_interval_sec)
    raise TimeoutError(
        f"targon: model at {base_url} did not become ready in {deadline_sec}s"
    )


async def deploy(
    client: TargonClient,
    target: DeployTarget,
    *,
    gpu_count: Optional[int] = None,
    ready_deadline_sec: int = DEFAULT_READY_DEADLINE_SEC,
) -> DeployResult:
    """Adopt the existing workload if one matches, else create. Block until
    ``/v1/models`` reports the model ready."""
    if not client.configured:
        raise RuntimeError(
            "TargonClient is not configured (TARGON_API_URL / TARGON_API_KEY)"
        )
    gpus = gpu_count if gpu_count is not None else (fixed_gpu_count() or 1)

    deployment_id = await find_existing_workload(client, target)
    if deployment_id is not None:
        logger.info(
            f"targon: adopted existing workload {deployment_id} for "
            f"{target.model}@{target.revision[:8]}"
        )
    else:
        created = await client.create_deployment(
            target.model,
            target.revision,
            uid=target.uid,
            hotkey=target.hotkey,
            gpu_count=gpus,
        )
        if not created:
            raise RuntimeError(
                f"targon: create_deployment returned falsy for "
                f"{target.model}@{target.revision[:8]}"
            )
        deployment_id = created
        logger.info(
            f"targon: created workload {deployment_id} for "
            f"{target.model}@{target.revision[:8]}"
        )

    base_url = external_url(deployment_id)
    await wait_for_ready(base_url, deadline_sec=ready_deadline_sec)
    return DeployResult(deployment_id=deployment_id, base_url=base_url)


async def teardown(client: TargonClient, deployment_id: Optional[str]) -> None:
    """Delete a workload. Swallow errors — a window finalize must succeed
    even if Targon temporarily refuses; the next ``orphan_sweep`` will
    catch what we couldn't drop here."""
    if not deployment_id:
        return
    try:
        ok = await client.delete_deployment(deployment_id)
        if not ok:
            logger.warning(
                f"targon.teardown: delete_deployment returned falsy "
                f"for {deployment_id}"
            )
    except Exception as e:
        logger.warning(
            f"targon.teardown({deployment_id}) error: {type(e).__name__}: {e}"
        )


async def orphan_sweep(
    client: TargonClient, known_workload_ids: Iterable[str],
) -> int:
    """Drop affine-prefixed Targon workloads not in ``known_workload_ids``.

    The scheduler tracks live deployments via the ``champion`` and
    ``current_battle`` records in system_config. Pass the union of those
    deployment_ids here; anything affine-prefixed but unknown is an orphan
    (failed create→record handoff, manual hand-deletion, abandoned manual
    deploy) and gets dropped. Workloads whose name doesn't start with the
    affine prefix are never touched — safe in shared Targon accounts.

    Returns the number of workloads deleted.
    """
    if not client.configured:
        return 0
    try:
        listing = await client.list_workloads(limit=200)
    except Exception as e:
        logger.warning(f"targon.orphan_sweep: list_workloads raised: {e}")
        return 0
    if listing is None:
        return 0
    items = (listing.get("items") if isinstance(listing, dict) else listing) or []
    if not items:
        return 0

    known = set(known_workload_ids)
    prefix = client.WORKLOAD_NAME_PREFIX + "-"
    deleted = 0
    for w in items:
        wid = w.get("uid") or w.get("id")
        name = w.get("name") or ""
        if not wid or wid in known or not name.startswith(prefix):
            continue
        try:
            ok = await client.delete_deployment(wid)
            if ok:
                deleted += 1
                logger.warning(
                    f"targon.orphan_sweep: deleted untracked workload "
                    f"{wid} (name={name})"
                )
        except Exception as e:
            logger.error(f"targon.orphan_sweep: delete {wid} raised: {e}")
    return deleted
