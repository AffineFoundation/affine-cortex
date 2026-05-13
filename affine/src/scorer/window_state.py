"""
Flow-scheduler state model.

Lives under ``affine/src/scorer/`` for legacy reasons (the directory used
to host a scorer service; it is now a library shared by scheduler,
executor, and the API readers). The data model is the bare minimum the
flow scheduler needs:

  ``champion``                    — who currently holds the crown
  ``current_battle``              — who is being evaluated against them
  ``current_task_ids[env]``       — the task_id pool shared by both
  ``last_task_refresh_block``     — when the pool was last regenerated

Everything else (window phases, archives, status enums, window_id) is
gone. State machine "transitions" are now implicit in the read state:

  champion=None                 → cold start
  champion=X, deployment=None   → champion needs Targon
  champion=X, no battle, samples incomplete → executor still warming up X
  champion=X, no battle, samples complete   → ready to start a battle
  champion=X, battle=Y                      → contest in flight
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol


# ---- dataclasses ----------------------------------------------------------


@dataclass
class MinerSnapshot:
    """Identity tuple for a miner — what the scheduler / executor / scorer
    need to reach the right HF model and the right ``sample_results`` rows.
    """
    uid: int
    hotkey: str
    revision: str
    model: str


@dataclass
class DeploymentRecord:
    """One inference machine serving a miner."""
    endpoint_name: str
    deployment_id: str
    base_url: str


@dataclass
class ChampionRecord:
    """The current champion. Targon workload is owned for the champion's
    full reign — only torn down when a challenger wins."""
    uid: int
    hotkey: str
    revision: str
    model: str
    deployment_id: Optional[str] = None
    base_url: Optional[str] = None
    deployments: List[DeploymentRecord] = field(default_factory=list)
    since_block: int = 0


@dataclass
class BattleRecord:
    """An in-flight contest. The challenger's Targon workload only lives
    as long as this record does."""
    challenger: MinerSnapshot
    deployment_id: str
    base_url: str
    started_at_block: int
    deployments: List[DeploymentRecord] = field(default_factory=list)


@dataclass
class TaskIdState:
    """Per-env task_id pool the executor evaluates against. Regenerated
    every ~7200 blocks (always between battles)."""
    task_ids: Dict[str, List[int]] = field(default_factory=dict)
    refreshed_at_block: int = 0


@dataclass
class EnvConfig:
    """Per-env runtime config from ``system_config['environments']``.

    Two independent gates:

      - ``enabled_for_sampling`` — include this env in the sampler /
        executor pipeline (materialise task_ids, run evaluate()).
      - ``enabled_for_scoring`` — include this env in the DECIDE Pareto
        comparison. Strictly narrower: an env that's
        ``enabled_for_sampling=true, enabled_for_scoring=false`` keeps
        accumulating data while DECIDE ignores it — used for onboarding
        a new env so its possibly-broken scoring can't poison contest
        outcomes during the bake-in period.
    """
    display_name: str
    enabled_for_sampling: bool
    sampling_count: int
    dataset_range: List[List[int]]
    sampling_mode: str = "random"  # 'random' | 'latest'
    enabled_for_scoring: bool = True
    # Optional remote source for the dataset's current top index. When
    # set, the scheduler resolves ``dataset_range`` from this URL at
    # each window refresh so envs whose dataset accumulates new
    # task_ids (SWE-INFINITE, DISTILL) always sample the freshest tail.
    # Shape: ``{"url": str, "field": str, "range_type": "zero_to_value"}``.
    dataset_range_source: Optional[Dict[str, Any]] = None


# ---- store protocol -------------------------------------------------------


class ConfigKVStore(Protocol):
    """Tiny KV protocol mapped onto ``SystemConfigDAO``."""

    async def get(self, key: str, default: Any = None) -> Any: ...
    async def set(self, key: str, value: Any) -> None: ...
    async def delete(self, key: str) -> bool: ...


# ---- typed accessor -------------------------------------------------------


class StateStore:
    """Typed accessor over the simplified ``system_config`` keys."""

    KEY_CHAMPION = "champion"
    KEY_BATTLE = "current_battle"
    KEY_TASK_IDS = "current_task_ids"
    KEY_ENVIRONMENTS = "environments"

    def __init__(self, kv: ConfigKVStore):
        self._kv = kv

    # -- champion -----------------------------------------------------------

    async def get_champion(self) -> Optional[ChampionRecord]:
        raw = await self._kv.get(self.KEY_CHAMPION)
        if not raw:
            return None
        return _champion_from_dict(raw)

    async def set_champion(self, champ: ChampionRecord) -> None:
        await self._kv.set(self.KEY_CHAMPION, asdict(champ))

    async def clear_champion(self) -> None:
        await self._kv.delete(self.KEY_CHAMPION)

    # -- battle -------------------------------------------------------------

    async def get_battle(self) -> Optional[BattleRecord]:
        raw = await self._kv.get(self.KEY_BATTLE)
        if not raw:
            return None
        return _battle_from_dict(raw)

    async def set_battle(self, battle: BattleRecord) -> None:
        await self._kv.set(self.KEY_BATTLE, _battle_to_dict(battle))

    async def clear_battle(self) -> None:
        await self._kv.delete(self.KEY_BATTLE)

    # -- task ids -----------------------------------------------------------

    async def get_task_state(self) -> Optional[TaskIdState]:
        raw = await self._kv.get(self.KEY_TASK_IDS)
        if not raw:
            return None
        return TaskIdState(
            task_ids={str(k): [int(x) for x in v] for k, v in (raw.get("task_ids") or {}).items()},
            refreshed_at_block=int(raw.get("refreshed_at_block", 0)),
        )

    async def set_task_state(self, state: TaskIdState) -> None:
        await self._kv.set(self.KEY_TASK_IDS, asdict(state))

    # -- env config ---------------------------------------------------------

    async def get_environments(self) -> Dict[str, EnvConfig]:
        """Envs the worker should sample for. Filtered to
        ``enabled_for_sampling=true`` regardless of their
        ``enabled_for_scoring`` flag — sampling-only envs still need
        their task_ids materialised so data accumulates while DECIDE
        excludes them."""
        raw = await self._kv.get(self.KEY_ENVIRONMENTS, default={}) or {}
        out: Dict[str, EnvConfig] = {}
        for env, cfg in raw.items():
            coerced = _env_from_payload(cfg)
            if coerced.enabled_for_sampling:
                out[env] = coerced
        return out

    async def get_scoring_environments(self) -> Dict[str, EnvConfig]:
        """Envs the comparator should consider at DECIDE. Stricter
        than :meth:`get_environments`: requires both
        ``enabled_for_sampling=true`` AND ``enabled_for_scoring=true``
        so a new env can be sampled for data collection without its
        (possibly broken) scores poisoning the Pareto verdict."""
        return {
            env: cfg
            for env, cfg in (await self.get_environments()).items()
            if cfg.enabled_for_scoring
        }


# ---- in-memory test fake --------------------------------------------------


class InMemoryConfigStore:
    """Drop-in fake for ``SystemConfigKVAdapter`` in tests. Plain dict."""

    def __init__(self):
        self.data: Dict[str, Any] = {}

    async def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    async def delete(self, key: str) -> bool:
        return self.data.pop(key, None) is not None


# ---- production adapter ---------------------------------------------------


class SystemConfigKVAdapter:
    """Wraps ``SystemConfigDAO`` to satisfy ``ConfigKVStore``."""

    def __init__(self, dao, *, updated_by: str = "scheduler"):
        self._dao = dao
        self._updated_by = updated_by

    async def get(self, key: str, default: Any = None) -> Any:
        return await self._dao.get_param_value(key, default=default)

    async def set(self, key: str, value: Any) -> None:
        # Infer ``param_type`` from the value so callers don't have to.
        # SystemConfigDAO uses this column as metadata only (no validation),
        # but it's still required positional.
        if isinstance(value, bool):
            ptype = "bool"
        elif isinstance(value, int):
            ptype = "int"
        elif isinstance(value, float):
            ptype = "float"
        elif isinstance(value, str):
            ptype = "str"
        elif isinstance(value, list):
            ptype = "list"
        else:
            ptype = "dict"
        await self._dao.set_param(
            param_name=key,
            param_value=value,
            param_type=ptype,
            updated_by=self._updated_by,
        )

    async def delete(self, key: str) -> bool:
        return await self._dao.delete_param(key)


# ---- internal codecs ------------------------------------------------------


def _from_dict(cls, raw: Dict[str, Any]):
    fields = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in raw.items() if k in fields})


def _deployment_list(raw: Dict[str, Any]) -> List[DeploymentRecord]:
    deployments = []
    for item in raw.get("deployments") or []:
        if not isinstance(item, dict):
            continue
        deployment_id = str(item.get("deployment_id") or "")
        base_url = str(item.get("base_url") or "")
        if not deployment_id or not base_url:
            continue
        deployments.append(
            DeploymentRecord(
                endpoint_name=str(item.get("endpoint_name") or ""),
                deployment_id=deployment_id,
                base_url=base_url,
            )
        )
    if (
        "deployments" not in raw
        and not deployments
        and raw.get("deployment_id")
        and raw.get("base_url")
    ):
        deployments.append(
            DeploymentRecord(
                endpoint_name=str(raw.get("endpoint_name") or ""),
                deployment_id=str(raw.get("deployment_id")),
                base_url=str(raw.get("base_url")),
            )
        )
    return deployments


def _primary_deployment(deployments: List[DeploymentRecord]) -> tuple[Optional[str], Optional[str]]:
    if not deployments:
        return None, None
    first = deployments[0]
    return first.deployment_id, first.base_url


def _champion_from_dict(raw: Dict[str, Any]) -> ChampionRecord:
    deployments = _deployment_list(raw)
    deployment_id = raw.get("deployment_id")
    base_url = raw.get("base_url")
    if deployments and (not deployment_id or not base_url):
        deployment_id, base_url = _primary_deployment(deployments)
    return ChampionRecord(
        uid=int(raw["uid"]),
        hotkey=str(raw["hotkey"]),
        revision=str(raw["revision"]),
        model=str(raw["model"]),
        deployment_id=str(deployment_id) if deployment_id else None,
        base_url=str(base_url) if base_url else None,
        deployments=deployments,
        since_block=int(raw.get("since_block", 0)),
    )


def _battle_to_dict(b: BattleRecord) -> Dict[str, Any]:
    return {
        "challenger": asdict(b.challenger),
        "deployment_id": b.deployment_id,
        "base_url": b.base_url,
        "started_at_block": b.started_at_block,
        "deployments": [asdict(d) for d in b.deployments],
    }


def _battle_from_dict(raw: Dict[str, Any]) -> BattleRecord:
    chal_raw = raw.get("challenger") or {}
    deployments = _deployment_list(raw)
    deployment_id = str(raw.get("deployment_id") or "")
    base_url = str(raw.get("base_url") or "")
    if deployments and (not deployment_id or not base_url):
        primary_id, primary_url = _primary_deployment(deployments)
        deployment_id = primary_id or ""
        base_url = primary_url or ""
    return BattleRecord(
        challenger=MinerSnapshot(**{
            k: chal_raw[k] for k in ("uid", "hotkey", "revision", "model")
            if k in chal_raw
        }),
        deployment_id=deployment_id,
        base_url=base_url,
        started_at_block=int(raw.get("started_at_block", 0)),
        deployments=deployments,
    )


def _env_from_payload(payload: Any) -> EnvConfig:
    if not isinstance(payload, dict):
        return EnvConfig(
            display_name="", enabled_for_sampling=False,
            sampling_count=0, dataset_range=[], sampling_mode="random",
        )
    sampling = payload.get("sampling") or payload.get("window_config") or {}
    src = sampling.get("dataset_range_source")
    return EnvConfig(
        display_name=str(payload.get("display_name", "")),
        enabled_for_sampling=bool(payload.get("enabled_for_sampling", False)),
        # Default True so existing envs keep their pre-flag behavior
        # (sampling and scoring both on). New envs opt out of scoring
        # explicitly by setting ``enabled_for_scoring: false``.
        enabled_for_scoring=bool(payload.get("enabled_for_scoring", True)),
        sampling_count=int(sampling.get("sampling_count", 0)),
        dataset_range=list(sampling.get("dataset_range", []) or []),
        sampling_mode=str(sampling.get("sampling_mode", "random")),
        dataset_range_source=src if isinstance(src, dict) else None,
    )
