"""
Anti-Copy (CEAC) DAOs — three sibling tables backing the lazy
plagiarism detector described in ``devlog/ceac_design.md``.

The DAOs only handle metadata. The heavy payload (per-token logprobs,
tokenized prompts) lives in R2 and is referenced via ``r2_key``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from affine.database.base_dao import BaseDAO
from affine.database.client import get_client
from affine.database.schema import get_table_name


# ---------------------------------------------------------------------- rollouts


class AntiCopyRolloutsDAO(BaseDAO):
    """Index over the rolling rollout pool. Each row points at an R2
    blob containing the tokenized prompt + response."""

    table_name_base = "anticopy_rollouts"

    # Default retention for a rollout pool entry. Matches the design
    # default of ~7 days; callers can pass an explicit ``ttl`` to
    # override.
    DEFAULT_TTL_DAYS = 7

    def __init__(self):
        self.table_name = get_table_name(self.table_name_base)
        super().__init__()

    @staticmethod
    def make_rollout_key(champion_hotkey: str, env: str, task_id: int) -> str:
        """Stable id used both as DDB pk and as the cross-score join key."""
        return f"{champion_hotkey}#{env}#{task_id}"

    @staticmethod
    def _make_pk(rollout_key: str) -> str:
        return f"ROLLOUT#{rollout_key}"

    async def upsert(
        self,
        *,
        champion_hotkey: str,
        champion_revision: str,
        env: str,
        task_id: int,
        day: str,                       # "YYYY-MM-DD"
        tokenizer_sig: str,
        r2_key: str,
        response_len: int,
        prompt_len: int,
        ttl_days: int = DEFAULT_TTL_DAYS,
    ) -> Dict[str, Any]:
        rollout_key = self.make_rollout_key(champion_hotkey, env, task_id)
        now = int(time.time())
        item = {
            "pk": self._make_pk(rollout_key),
            "rollout_key": rollout_key,
            "champion_hotkey": champion_hotkey,
            "champion_revision": champion_revision,
            "env": env,
            "task_id": task_id,
            "day": day,
            "tokenizer_sig": tokenizer_sig,
            "r2_key": r2_key,
            "response_len": response_len,
            "prompt_len": prompt_len,
            "created_at": now,
            "ttl": now + ttl_days * 86400,
        }
        return await self.put(item)

    async def list_by_tokenizer(
        self, tokenizer_sig: str, *, max_age_days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return all current rollouts whose tokenizer matches.

        ``max_age_days`` filters by ``created_at`` client-side (DDB TTL
        is best-effort, so a short window may include expired rows).
        """
        client = get_client()
        params: Dict[str, Any] = {
            "TableName": self.table_name,
            "IndexName": "tokenizer-created-index",
            "KeyConditionExpression": "tokenizer_sig = :sig",
            "ExpressionAttributeValues": {":sig": {"S": tokenizer_sig}},
            "ScanIndexForward": False,        # newest first
        }
        items: List[Dict[str, Any]] = []
        last_key: Optional[Dict[str, Any]] = None
        while True:
            if last_key:
                params["ExclusiveStartKey"] = last_key
            resp = await client.query(**params)
            for raw in resp.get("Items", []):
                items.append(self._deserialize(raw))
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break

        if max_age_days is not None:
            cutoff = int(time.time()) - max_age_days * 86400
            items = [it for it in items if int(it.get("created_at", 0)) >= cutoff]
        return items

    async def delete_rollout(self, rollout_key: str) -> None:
        client = get_client()
        await client.delete_item(
            TableName=self.table_name,
            Key={"pk": {"S": self._make_pk(rollout_key)}},
        )


# ---------------------------------------------------------------------- scores index


class AntiCopyScoresIndexDAO(BaseDAO):
    """Index over per-(hotkey, revision) score blobs in R2. Verdict +
    overlap stats are recorded on this row so the verdict pass can
    work entirely off DDB."""

    table_name_base = "anticopy_scores_index"

    def __init__(self):
        self.table_name = get_table_name(self.table_name_base)
        super().__init__()

    @staticmethod
    def _make_pk(hotkey: str, revision: str) -> str:
        return f"SCORE#{hotkey}#{revision}"

    async def upsert(
        self,
        *,
        hotkey: str,
        revision: str,
        tokenizer_sig: str,
        r2_key: str,
        rollout_keys: List[str],
        first_block: int,
        verdict_copy_of: str = "",
        decision_median: float = -1.0,
    ) -> Dict[str, Any]:
        """Write the score-index row. ``decision_median`` is the
        |Δlogp| median on decision positions against the WINNING (or
        closest) peer — the same number used to compute the verdict.
        We store it instead of ``n_overlap_max`` because the verdict
        already implies overlap >= ``min_overlap``; the actual median
        is what an operator needs to sanity-check a flag."""
        now = int(time.time())
        item = {
            "pk": self._make_pk(hotkey, revision),
            "hotkey": hotkey,
            "revision": revision,
            "tokenizer_sig": tokenizer_sig,
            "r2_key": r2_key,
            "rollout_keys": rollout_keys,
            "first_block": int(first_block),
            "computed_at": now,
            "verdict_copy_of": verdict_copy_of,
            "decision_median": float(decision_median),
        }
        return await self.put(item)

    async def get_score(self, hotkey: str, revision: str) -> Optional[Dict[str, Any]]:
        return await self.get(self._make_pk(hotkey, revision))

    async def update_verdict(
        self,
        hotkey: str,
        revision: str,
        *,
        copy_of: str,
        decision_median: float,
    ) -> None:
        """Refresh just the verdict + its diagnostic ``decision_median``
        without rewriting the whole row. Called both at first-eval and
        from the retroactive pass when a newly-arrived earlier-committer
        flips an existing peer's verdict."""
        client = get_client()
        await client.update_item(
            TableName=self.table_name,
            Key={"pk": {"S": self._make_pk(hotkey, revision)}},
            UpdateExpression=(
                "SET verdict_copy_of = :copy_of, "
                "decision_median = :dec, verdict_at = :now"
            ),
            ExpressionAttributeValues={
                ":copy_of": {"S": copy_of},
                ":dec": {"N": str(float(decision_median))},
                ":now": {"N": str(int(time.time()))},
            },
        )

    async def list_all(self) -> List[Dict[str, Any]]:
        """Full scan. The table size is bounded by # active miners
        (~256), so a scan is fine and avoids GSI maintenance."""
        client = get_client()
        out: List[Dict[str, Any]] = []
        last_key: Optional[Dict[str, Any]] = None
        params: Dict[str, Any] = {"TableName": self.table_name}
        while True:
            if last_key:
                params["ExclusiveStartKey"] = last_key
            resp = await client.scan(**params)
            for raw in resp.get("Items", []):
                out.append(self._deserialize(raw))
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
        return out


# ---------------------------------------------------------------------- jobs queue


class AntiCopyJobsDAO(BaseDAO):
    """Pull queue. ``miners_monitor`` enqueues new candidates;
    ``forward_worker`` polls + dequeues."""

    table_name_base = "anticopy_jobs"

    STATE_PENDING = "pending"
    STATE_RUNNING = "running"
    STATE_DONE = "done"
    STATE_FAILED = "failed"

    def __init__(self):
        self.table_name = get_table_name(self.table_name_base)
        super().__init__()

    @staticmethod
    def _make_pk(hotkey: str, revision: str) -> str:
        return f"JOB#{hotkey}#{revision}"

    async def enqueue(
        self, *, hotkey: str, revision: str, model: str, uid: int
    ) -> Dict[str, Any]:
        """Idempotent: re-enqueueing an existing job is a no-op (kept
        for crash-recovery / retry semantics).

        ``enqueued_at`` is stored as ms-since-epoch so the queue's
        ``state-enqueued-index`` orders jobs strictly by submit time,
        not by the per-second tie-breaker DDB falls back on (which
        sorts by base-table pk alphabetically — surprising for callers
        that expect FIFO by submission order).
        """
        client = get_client()
        now_ms = int(time.time() * 1000)
        try:
            await client.put_item(
                TableName=self.table_name,
                Item=self._serialize({
                    "pk": self._make_pk(hotkey, revision),
                    "hotkey": hotkey,
                    "revision": revision,
                    "model": model,
                    "uid": int(uid),
                    "state": self.STATE_PENDING,
                    "enqueued_at": now_ms,
                    "attempts": 0,
                    "last_error": "",
                }),
                ConditionExpression="attribute_not_exists(pk)",
            )
        except Exception as e:
            # ConditionalCheckFailedException is expected for the
            # already-enqueued case; anything else propagates.
            err_code = getattr(getattr(e, "response", None), "get", lambda *_: {})(
                "Error", {}
            ).get("Code")
            if err_code != "ConditionalCheckFailedException":
                raise
        return {"hotkey": hotkey, "revision": revision}

    async def peek_pending(
        self, *, limit: int = 5, exclude_pk: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read up to ``limit`` oldest pending jobs WITHOUT claiming.

        Used by the worker's prefetcher to look at the next job(s) it
        will run so it can stage their weights in the background.
        """
        client = get_client()
        params = {
            "TableName": self.table_name,
            "IndexName": "state-enqueued-index",
            "KeyConditionExpression": "#st = :st",
            "ExpressionAttributeNames": {"#st": "state"},
            "ExpressionAttributeValues": {":st": {"S": self.STATE_PENDING}},
            "ScanIndexForward": True,
            "Limit": int(limit),
        }
        resp = await client.query(**params)
        out: List[Dict[str, Any]] = []
        for raw in resp.get("Items", []):
            row = self._deserialize(raw)
            if exclude_pk and row.get("pk") == exclude_pk:
                continue
            out.append(row)
        return out

    async def claim_next(self) -> Optional[Dict[str, Any]]:
        """Pop the oldest pending job and mark it running. Returns the
        job row or None when the queue is empty.

        Implementation: query the GSI for the oldest pending row, then
        conditional-update to flip state to running. If the conditional
        fails (another worker raced us), retry with the next oldest.
        """
        client = get_client()
        params = {
            "TableName": self.table_name,
            "IndexName": "state-enqueued-index",
            "KeyConditionExpression": "#st = :st",
            "ExpressionAttributeNames": {"#st": "state"},
            "ExpressionAttributeValues": {":st": {"S": self.STATE_PENDING}},
            "ScanIndexForward": True,                # oldest first
            "Limit": 10,
        }
        resp = await client.query(**params)
        for raw in resp.get("Items", []):
            row = self._deserialize(raw)
            try:
                await client.update_item(
                    TableName=self.table_name,
                    Key={"pk": {"S": row["pk"]}},
                    UpdateExpression=(
                        "SET #st = :running, started_at = :now, "
                        "attempts = if_not_exists(attempts, :zero) + :one"
                    ),
                    ConditionExpression="#st = :pending",
                    ExpressionAttributeNames={"#st": "state"},
                    ExpressionAttributeValues={
                        ":running": {"S": self.STATE_RUNNING},
                        ":pending": {"S": self.STATE_PENDING},
                        ":now": {"N": str(int(time.time()))},
                        ":zero": {"N": "0"},
                        ":one": {"N": "1"},
                    },
                )
                row["state"] = self.STATE_RUNNING
                return row
            except Exception as e:
                err_code = getattr(getattr(e, "response", None), "get", lambda *_: {})(
                    "Error", {}
                ).get("Code")
                if err_code == "ConditionalCheckFailedException":
                    continue
                raise
        return None

    async def mark_done(self, hotkey: str, revision: str) -> None:
        client = get_client()
        await client.update_item(
            TableName=self.table_name,
            Key={"pk": {"S": self._make_pk(hotkey, revision)}},
            UpdateExpression="SET #st = :done, finished_at = :now REMOVE last_error",
            ExpressionAttributeNames={"#st": "state"},
            ExpressionAttributeValues={
                ":done": {"S": self.STATE_DONE},
                ":now": {"N": str(int(time.time()))},
            },
        )

    async def mark_failed(self, hotkey: str, revision: str, error: str) -> None:
        client = get_client()
        await client.update_item(
            TableName=self.table_name,
            Key={"pk": {"S": self._make_pk(hotkey, revision)}},
            UpdateExpression=(
                "SET #st = :failed, last_error = :err, finished_at = :now"
            ),
            ExpressionAttributeNames={"#st": "state"},
            ExpressionAttributeValues={
                ":failed": {"S": self.STATE_FAILED},
                ":err": {"S": (error or "")[:500]},
                ":now": {"N": str(int(time.time()))},
            },
        )

    async def reset_to_pending(self, hotkey: str, revision: str) -> None:
        """Crash-recovery: put a stuck ``running`` (or ``failed``) job back
        in queue. ``enqueued_at`` is written as ms-since-epoch to match
        :meth:`enqueue` — the ``state-enqueued-index`` GSI's sort key is
        numeric, so mixing seconds and ms would corrupt the FIFO order."""
        client = get_client()
        await client.update_item(
            TableName=self.table_name,
            Key={"pk": {"S": self._make_pk(hotkey, revision)}},
            UpdateExpression="SET #st = :pending, enqueued_at = :now",
            ExpressionAttributeNames={"#st": "state"},
            ExpressionAttributeValues={
                ":pending": {"S": self.STATE_PENDING},
                ":now": {"N": str(int(time.time() * 1000))},
            },
        )

    async def get_job(self, hotkey: str, revision: str) -> Optional[Dict[str, Any]]:
        return await self.get(self._make_pk(hotkey, revision))


# ---------------------------------------------------------------------- state


class AntiCopyStateDAO(BaseDAO):
    """Machine-managed metadata for the CEAC subsystem.

    Holds runtime state that the ``anticopy-refresh`` service writes on
    every daily tick — anchored champion uid + the tokenizer signature
    used when materialising the active rollout pool. These values are
    NOT operator settings; they update too often to live in
    ``system_config`` (the human-tunable table). They are also kept
    separate from the per-rollout ``anticopy_rollouts`` index so its
    queries (filtered by tokenizer_sig + day) don't have to step over
    metadata rows.

    A single fixed pk (``CURRENT``) holds the whole bundle as
    individual attributes — there's only ever one "active state" entry,
    and grouping the fields lets a caller read both with one
    ``get_item`` call.
    """

    table_name_base = "anticopy_state"
    PK_CURRENT = "CURRENT"

    def __init__(self):
        self.table_name = get_table_name(self.table_name_base)
        super().__init__()

    async def get_state(self) -> Dict[str, Any]:
        """Return the active state dict (may be empty if refresh hasn't
        run yet). Bypasses ``BaseDAO.get`` because this table's primary
        key attribute is named ``key`` rather than the standard ``pk``."""
        client = get_client()
        resp = await client.get_item(
            TableName=self.table_name,
            Key={"key": {"S": self.PK_CURRENT}},
        )
        item = resp.get("Item")
        return self._deserialize(item) if item else {}

    async def get_active_champion(self) -> Optional[int]:
        st = await self.get_state()
        uid = st.get("active_champion_uid")
        try:
            return int(uid) if uid is not None else None
        except (TypeError, ValueError):
            return None

    async def get_champion_tokenizer_sig(self) -> str:
        st = await self.get_state()
        return str(st.get("champion_tokenizer_sig") or "")

    # Deployment-config fields stored alongside the daily anchor.
    # These are operator-set on first deploy; refresh service never
    # touches them. Putting them here (vs ``system_config``) keeps all
    # CEAC-specific runtime config in one row so an operator can dump
    # the whole subsystem state with a single ``get_state`` call.
    #
    # ``mapping`` is ``state_field_name -> env_var_name``. Used both by
    # :meth:`get_deployment_config` to materialise effective values and
    # by :func:`hydrate_env_from_state` (worker bootstrap helper) to
    # propagate state into ``os.environ`` before the worker module
    # constants are first evaluated.
    DEPLOYMENT_FIELD_TO_ENV: Dict[str, str] = {
        "rollouts_bucket": "R2_ANTICOPY_ROLLOUTS_BUCKET",
        "scores_bucket": "R2_ANTICOPY_SCORES_BUCKET",
        "remote_ssh_host": "ANTICOPY_REMOTE_SSH_HOST",
        "remote_ssh_port": "ANTICOPY_REMOTE_SSH_PORT",
        "remote_ssh_key": "ANTICOPY_REMOTE_SSH_KEY",
        "remote_python": "ANTICOPY_REMOTE_PYTHON",
        "remote_sglang_port": "ANTICOPY_REMOTE_SGLANG_PORT",
        "sglang_url": "ANTICOPY_SGLANG_URL",
        "sglang_base_gpu_id": "ANTICOPY_BASE_GPU_ID",
        "sglang_extra_args": "ANTICOPY_SGLANG_EXTRA_ARGS",
        "hf_cache": "ANTICOPY_HF_CACHE",
    }

    async def get_deployment_config(self) -> Dict[str, str]:
        """Resolved deployment config — DDB state takes precedence over
        env vars; missing-everywhere returns empty string. Returned dict
        is keyed by the state-field name (not the env-var name)."""
        import os
        state = await self.get_state()
        out: Dict[str, str] = {}
        for field, env_name in self.DEPLOYMENT_FIELD_TO_ENV.items():
            val = state.get(field)
            if val is None or val == "":
                val = os.getenv(env_name, "")
            out[field] = str(val) if val is not None else ""
        return out

    async def hydrate_env(self) -> None:
        """Materialise the deployment config into ``os.environ`` for
        every field that has a value in DDB state.

        Used by the worker / refresh entry points right after
        ``init_client`` so the downstream module-level ``os.getenv(...)``
        constants (and any third-party libs that read env at import
        time) see the operator's DDB-driven config. Fields not set in
        DDB are left untouched so the existing ``.env`` keeps working
        as a fallback.
        """
        import os
        state = await self.get_state()
        for field, env_name in self.DEPLOYMENT_FIELD_TO_ENV.items():
            val = state.get(field)
            if val is not None and val != "":
                os.environ[env_name] = str(val)

    async def set_state(self, **fields: Any) -> None:
        """Partial update — pass any subset of state fields as kwargs.

        Reserved field names: ``key`` (the row's primary key) and
        ``updated_at`` (always bumped here, callers should not pass it).
        Field values must be JSON-serialisable scalars/lists/dicts;
        :meth:`BaseDAO._serialize` handles the DynamoDB type tags.

        Known fields written by the refresh service:
          * ``active_champion_uid`` (int)
          * ``active_champion_hotkey`` (str)
          * ``active_champion_revision`` (str)
          * ``active_champion_model`` (str)
          * ``active_champion_day`` (str, YYYY-MM-DD)
          * ``champion_tokenizer_sig`` (str)
          * ``last_refresh_date`` (str, YYYY-MM-DD)
        """
        if not fields:
            return
        # Strip reserved keys to avoid collisions with key/updated_at.
        fields = {k: v for k, v in fields.items() if k not in ("key", "updated_at")}
        client = get_client()
        # We update in place rather than ``put_item``-overwriting so a
        # caller that only knows one field doesn't wipe the others by
        # accident.
        updates: List[str] = ["#updated_at = :updated_at"]
        names: Dict[str, str] = {"#updated_at": "updated_at"}
        values: Dict[str, Any] = {
            ":updated_at": {"N": str(int(time.time()))},
        }
        # Reuse _serialize for the value side: it converts each Python
        # value to the right ``{S/N/M/L/...}`` envelope.
        serialised = self._serialize(fields)
        for i, (field_name, ddb_value) in enumerate(serialised.items()):
            name_alias = f"#f{i}"
            value_alias = f":v{i}"
            updates.append(f"{name_alias} = {value_alias}")
            names[name_alias] = field_name
            values[value_alias] = ddb_value
        await client.update_item(
            TableName=self.table_name,
            Key={"key": {"S": self.PK_CURRENT}},
            UpdateExpression="SET " + ", ".join(updates),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

