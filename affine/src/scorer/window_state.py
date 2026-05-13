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
class ChampionRecord:
    """The current champion. Targon workload is owned for the champion's
    full reign — only torn down when a challenger wins."""
    uid: int
    hotkey: str
    revision: str
    model: str
    deployment_id: Optional[str] = None
    base_url: Optional[str] = None
    since_block: int = 0


@dataclass
class BattleRecord:
    """An in-flight contest. The challenger's Targon workload only lives
    as long as this record does."""
    challenger: MinerSnapshot
    deployment_id: str
    base_url: str
    started_at_block: int


@dataclass
class TaskIdState:
    """Per-env task_id pool the executor evaluates against. Regenerated
    every ~7200 blocks (always between battles)."""
    task_ids: Dict[str, List[int]] = field(default_factory=dict)
    refreshed_at_block: int = 0


@dataclass
class EnvConfig:
    """Per-env runtime config from ``system_config['environments']``."""
    display_name: str
    enabled: bool
    sampling_count: int
    dataset_range: List[List[int]]
    sampling_mode: str = "random"  # 'random' | 'latest'


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
        return _from_dict(ChampionRecord, raw)

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
        raw = await self._kv.get(self.KEY_ENVIRONMENTS, default={}) or {}
        out: Dict[str, EnvConfig] = {}
        for env, cfg in raw.items():
            coerced = _env_from_payload(cfg)
            if coerced.enabled:
                out[env] = coerced
        return out


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
        await self._dao.set_param(
            key=key, value=value, updated_by=self._updated_by,
        )

    async def delete(self, key: str) -> bool:
        return await self._dao.delete_param(key)


# ---- internal codecs ------------------------------------------------------


def _from_dict(cls, raw: Dict[str, Any]):
    fields = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in raw.items() if k in fields})


def _battle_to_dict(b: BattleRecord) -> Dict[str, Any]:
    return {
        "challenger": asdict(b.challenger),
        "deployment_id": b.deployment_id,
        "base_url": b.base_url,
        "started_at_block": b.started_at_block,
    }


def _battle_from_dict(raw: Dict[str, Any]) -> BattleRecord:
    chal_raw = raw.get("challenger") or {}
    return BattleRecord(
        challenger=MinerSnapshot(**{
            k: chal_raw[k] for k in ("uid", "hotkey", "revision", "model")
            if k in chal_raw
        }),
        deployment_id=str(raw.get("deployment_id") or ""),
        base_url=str(raw.get("base_url") or ""),
        started_at_block=int(raw.get("started_at_block", 0)),
    )


def _env_from_payload(payload: Any) -> EnvConfig:
    if not isinstance(payload, dict):
        return EnvConfig(
            display_name="", enabled=False,
            sampling_count=0, dataset_range=[], sampling_mode="random",
        )
    sampling = payload.get("sampling") or payload.get("window_config") or {}
    return EnvConfig(
        display_name=str(payload.get("display_name", "")),
        enabled=bool(payload.get("enabled", True)),
        sampling_count=int(sampling.get("sampling_count", 0)),
        dataset_range=list(sampling.get("dataset_range", []) or []),
        sampling_mode=str(sampling.get("sampling_mode", "random")),
    )
