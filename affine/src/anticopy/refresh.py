"""
CEAC rollout pool refresh service. Wakes up roughly once a day at the
configured UTC hour, looks at the current champion, and converts the
champion's most-recent ``rollouts_per_env`` ``sample_results`` rows per
enabled env into CEAC rollout entries (R2 blob + DDB index row).

We **don't** re-run any environment. The samples are already there
from the regular scoring pipeline — refresh is a pure transform:

    sample_results.extra.conversation  →  tokenize → R2 + DDB index

so the only inference cost CEAC adds is the candidate-side teacher-
forcing in :mod:`affine.src.anticopy.worker`.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from affine.core.setup import logger
from affine.database.dao.anticopy import AntiCopyRolloutsDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.anticopy import AntiCopyStateDAO
from affine.database.dao.system_config import SystemConfigDAO

from affine.src.anticopy.r2 import AntiCopyR2
from affine.src.anticopy.task_filter import is_eligible_rollout_source
from affine.src.anticopy.threshold import AntiCopyConfig, load_anticopy_config
from affine.src.anticopy.tokenizer_sig import compute_tokenizer_signature


# Bookkeeping key — last UTC date we ran a refresh tick, persisted in
# ``anticopy_state`` so a service restart doesn't double-refresh on the
# same day. (Operator-tunable values stay in ``system_config``; this is
# machine-managed and lives next to the other ``anticopy_state``
# fields.)
_LAST_REFRESH_FIELD = "last_refresh_date"

# Max ``prompt_ids + response_ids`` per rollout. sglang's serving
# context (currently 32K) must accommodate this; we leave headroom
# for the one ``max_new_tokens=1`` sampled token + a small safety
# margin. When the raw conversation exceeds the cap, we truncate
# from the FRONT of the prompt (keeping the response intact and the
# most recent prompt turns) — see ``_build_rollout_payload``.
_MAX_TOTAL_TOKENS = 30000


class RolloutRefreshService:
    """Daily cron-style refresher. Tick every 5 min, do real work only
    when the UTC date has changed and we've crossed ``refresh_utc_hour``.
    """

    POLL_INTERVAL_SEC = 300

    def __init__(
        self,
        *,
        rollouts_dao: Optional[AntiCopyRolloutsDAO] = None,
        sample_dao: Optional[SampleResultsDAO] = None,
        config_dao: Optional[SystemConfigDAO] = None,
        state_dao: Optional[AntiCopyStateDAO] = None,
        r2: Optional[AntiCopyR2] = None,
    ):
        self.rollouts_dao = rollouts_dao or AntiCopyRolloutsDAO()
        self.sample_dao = sample_dao or SampleResultsDAO()
        # ``config_dao`` is kept for operator-tunable settings only
        # (``anticopy`` block + the live ``champion`` snapshot). The
        # machine-managed runtime state (anchored champion of the day,
        # tokenizer sig, last-refresh date) lives in ``state_dao``.
        self.config_dao = config_dao or SystemConfigDAO()
        self.state_dao = state_dao or AntiCopyStateDAO()
        self.r2 = r2 or AntiCopyR2()
        self._running = False

    # ---- main loop ---------------------------------------------------

    async def run(self) -> None:
        self._running = True
        logger.info("[anticopy.refresh] service started")
        while self._running:
            try:
                await self._maybe_tick()
            except Exception as e:
                logger.error(f"[anticopy.refresh] tick failed: {e}", exc_info=True)
            await asyncio.sleep(self.POLL_INTERVAL_SEC)

    def stop(self) -> None:
        self._running = False

    async def _maybe_tick(self) -> None:
        cfg = await load_anticopy_config(self.config_dao)
        if not cfg.enabled:
            return

        now = dt.datetime.now(dt.timezone.utc)
        if now.hour < cfg.refresh_utc_hour:
            return
        today = now.strftime("%Y-%m-%d")

        state = await self.state_dao.get_state()
        if state.get(_LAST_REFRESH_FIELD) == today:
            return

        await self.tick(cfg, day=today)

        await self.state_dao.set_state(**{_LAST_REFRESH_FIELD: today})

    # ---- one tick ----------------------------------------------------

    async def tick(self, cfg: AntiCopyConfig, *, day: str) -> Dict[str, int]:
        """Refresh the rollout pool for ``day``. Returns
        ``{env: rollouts_promoted}`` for logging / tests.

        Anchoring: the first refresh on a given day snapshots the
        live ``champion`` into ``anticopy_active_champion`` and uses
        that snapshot. Later refreshes the same day (e.g. after an
        operator wipes the pool) keep using the snapshot, even if
        prod scheduler has since dethroned the original champion.
        This keeps ``rollout_key`` (= ``{champion_hk}#{env}#{tid}``)
        stable across same-day refreshes so candidate score sets
        always intersect with peers' score sets.
        """
        state = await self.state_dao.get_state()
        if state.get("active_champion_day") != day:
            # New day (or never set): take a fresh snapshot from the
            # currently live champion (operator-managed, stays in
            # ``system_config``).
            current = await self.config_dao.get_param_value(
                "champion", default=None,
            )
            if not current or not current.get("hotkey") or not current.get("revision"):
                logger.warning(
                    "[anticopy.refresh] no champion configured, skipping tick"
                )
                return {}
            champ_uid = current.get("uid")
            champ_hotkey = str(current["hotkey"])
            champ_revision = str(current["revision"])
            champ_model = str(current.get("model", ""))
            await self.state_dao.set_state(
                active_champion_uid=int(champ_uid) if champ_uid is not None else None,
                active_champion_hotkey=champ_hotkey,
                active_champion_revision=champ_revision,
                active_champion_model=champ_model,
                active_champion_day=day,
            )
            logger.info(
                f"[anticopy.refresh] anchored champion for {day}: "
                f"uid={champ_uid} hk={champ_hotkey[:12]} "
                f"rev={champ_revision[:8]}"
            )
        else:
            champ_hotkey = str(state.get("active_champion_hotkey", ""))
            champ_revision = str(state.get("active_champion_revision", ""))
            champ_model = str(state.get("active_champion_model", ""))
            logger.info(
                f"[anticopy.refresh] reusing anchored champion for {day}: "
                f"uid={state.get('active_champion_uid')} "
                f"hk={champ_hotkey[:12]} rev={champ_revision[:8]}"
            )

        # Compute champion's tokenizer signature once for the whole tick.
        tokenizer_sig, _src = await compute_tokenizer_signature(
            champ_model, champ_revision,
        )
        if not tokenizer_sig:
            logger.warning(
                f"[anticopy.refresh] could not compute tokenizer_sig for "
                f"{champ_model}@{champ_revision[:8]} — abort tick"
            )
            return {}

        # Pin the active tokenizer signature so miners_monitor can use
        # it as the early-reject reference (step 8.0).
        await self.state_dao.set_state(champion_tokenizer_sig=tokenizer_sig)

        # Load tokenizer once per tick (slow import — keep lazy).
        from transformers import AutoTokenizer
        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, champ_model, revision=champ_revision
        )

        promoted: Dict[str, int] = {}
        for env in cfg.enabled_envs:
            try:
                n = await self._refresh_env(
                    cfg=cfg,
                    day=day,
                    env=env,
                    champion_hotkey=champ_hotkey,
                    champion_revision=champ_revision,
                    tokenizer_sig=tokenizer_sig,
                    tokenizer=tokenizer,
                )
                promoted[env] = n
            except Exception as e:
                logger.error(
                    f"[anticopy.refresh] env={env} promote failed: {e}",
                    exc_info=True,
                )
                promoted[env] = 0
        logger.info(
            f"[anticopy.refresh] day={day} promoted={promoted} "
            f"tokenizer_sig={tokenizer_sig[:12]}"
        )
        return promoted

    async def _refresh_env(
        self,
        *,
        cfg: AntiCopyConfig,
        day: str,
        env: str,
        champion_hotkey: str,
        champion_revision: str,
        tokenizer_sig: str,
        tokenizer: Any,
    ) -> int:
        # 1) enumerate the champion's completed task_ids in this env
        task_ids = await self.sample_dao.get_completed_task_ids(
            champion_hotkey, champion_revision, env,
        )
        # 2) filter out codex-produced rows (see task_filter docstring)
        eligible = sorted(
            (t for t in task_ids if is_eligible_rollout_source(env, t)),
            reverse=True,
        )
        if not eligible:
            logger.info(
                f"[anticopy.refresh] env={env} no eligible champion samples"
            )
            return 0

        targets = eligible[: cfg.rollouts_per_env]
        n_done = 0
        for tid in targets:
            try:
                sample = await self.sample_dao.get_sample_by_task_id(
                    champion_hotkey, champion_revision, env, str(tid),
                    include_extra=True,
                )
            except Exception as e:
                logger.debug(
                    f"[anticopy.refresh] sample fetch failed env={env} task={tid}: {e}"
                )
                continue
            if not sample:
                continue

            try:
                rollout_payload = self._build_rollout_payload(
                    sample=sample,
                    env=env,
                    task_id=int(tid),
                    champion_hotkey=champion_hotkey,
                    champion_revision=champion_revision,
                    tokenizer=tokenizer,
                    tokenizer_sig=tokenizer_sig,
                    day=day,
                )
            except _SkipSample as skip:
                logger.debug(
                    f"[anticopy.refresh] skip env={env} task={tid}: {skip}"
                )
                continue

            r2_key = await asyncio.to_thread(
                self.r2.put_rollout,
                champion_hotkey=champion_hotkey,
                env=env,
                task_id=int(tid),
                payload=rollout_payload,
            )

            tokens = rollout_payload.get("tokens") or []
            mask = rollout_payload.get("assistant_mask") or []
            await self.rollouts_dao.upsert(
                champion_hotkey=champion_hotkey,
                champion_revision=champion_revision,
                env=env,
                task_id=int(tid),
                day=day,
                tokenizer_sig=tokenizer_sig,
                r2_key=r2_key,
                # v2 schema: ``response_len`` is the count of
                # assistant-positioned tokens (what the worker
                # actually teacher-forces), ``prompt_len`` is the
                # remainder = framing + non-assistant content.
                response_len=sum(1 for b in mask if b),
                prompt_len=len(tokens) - sum(1 for b in mask if b),
                ttl_days=cfg.pool_days,
            )
            n_done += 1
        return n_done

    @staticmethod
    def _build_rollout_payload(
        *,
        sample: Dict[str, Any],
        env: str,
        task_id: int,
        champion_hotkey: str,
        champion_revision: str,
        tokenizer: Any,
        tokenizer_sig: str,
        day: str,
    ) -> Dict[str, Any]:
        """Schema v2: encode the *entire* multi-turn trajectory + an
        ``assistant_mask`` flagging which tokens were produced by the
        champion model. The worker teacher-forces the whole sequence
        and averages NLL/top1 over the mask=True positions only.

        Why not just the last assistant turn (v1):
        agent-loop envs in this subnet (NAVWORLD/TERMINAL/MEMORY)
        end with a short template-y "final answer" turn. All Qwen3-
        derived finetunes converge to nearly the same token + logprob
        on those short final outputs, so the v1 last-turn signal
        couldn't distinguish independent finetune from real copy
        (caveat 4: independent training on the same base). SWE's
        agent reasoning *did* discriminate (top1 99% copy vs 28%
        independent), so the fix is to feed every assistant turn —
        reasoning included — into the pairwise comparison.
        """
        extra = sample.get("extra") or {}
        conv = extra.get("conversation") or []
        if not isinstance(conv, list) or len(conv) < 2:
            raise _SkipSample("conversation missing or too short")
        # At least one assistant turn with content is required.
        has_assistant_content = any(
            isinstance(m, dict) and m.get("role") == "assistant" and (m.get("content") or "")
            for m in conv
        )
        if not has_assistant_content:
            raise _SkipSample("no assistant turn with content found")

        try:
            tokens, assistant_mask = _tokenize_trajectory_with_mask(
                tokenizer, conv,
            )
        except Exception as e:
            raise _SkipSample(f"tokenize trajectory failed: {e}")
        if not tokens:
            raise _SkipSample("trajectory tokenized to empty")
        if not any(assistant_mask):
            raise _SkipSample("no assistant tokens after tokenization")

        # Truncate from the FRONT if total exceeds the serving budget.
        # We keep the tail because trailing assistant turns are the
        # part the model has to actually predict; the dropped prefix
        # is shared between candidates so any KV-cache "mismatch"
        # cancels out in pairwise NLL.
        if len(tokens) > _MAX_TOTAL_TOKENS:
            cut = len(tokens) - _MAX_TOTAL_TOKENS
            tokens = tokens[cut:]
            assistant_mask = assistant_mask[cut:]
            if not any(assistant_mask):
                raise _SkipSample(
                    "truncation removed all assistant tokens"
                )

        return {
            "schema": "ceac.rollout/v2",
            "champion": {
                "hotkey": champion_hotkey,
                "revision": champion_revision,
            },
            "env": env,
            "task_id": int(task_id),
            "day": day,
            "tokenizer_sig": tokenizer_sig,
            "tokens": list(tokens),
            "assistant_mask": [bool(b) for b in assistant_mask],
            # Keep the raw conv for debugging only — small vs the
            # token array.
            "messages": conv,
            "created_at": int(time.time()),
        }


def _tokenize_trajectory_with_mask(tokenizer, conv):
    """Render the conv via the model's chat template, tokenize the
    text (with offset mappings), and build an ``assistant_mask``
    that is True for tokens which fall inside an assistant turn's
    content range.

    The mask is constructed off the rendered text rather than the
    raw message list because the tokenizer may add chat-template
    framing (``<|im_start|>...<|im_end|>``) whose token boundaries
    don't align with message boundaries. Using offset_mapping is
    the only robust way to attribute each input_id to a source
    span.
    """
    import re

    # Render the WHOLE conversation as text (no generation prompt;
    # we want the trajectory exactly as the champion produced it).
    text = tokenizer.apply_chat_template(
        conv, add_generation_prompt=False, tokenize=False,
    )

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = list(enc.input_ids)
    offsets = list(enc.offset_mapping)

    # Find every assistant content span. Standard Qwen / ChatML uses
    # ``<|im_start|>assistant\n<CONTENT><|im_end|>`` — capture group 1
    # is the content body, which is what we want to mark.
    spans = []
    pattern = re.compile(
        r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL,
    )
    for m in pattern.finditer(text):
        spans.append((m.start(1), m.end(1)))

    if not spans:
        # Non-ChatML templates fall back to an empty mask — caller
        # raises ``_SkipSample``.
        return tokens, [False] * len(tokens)

    mask = []
    for off in offsets:
        if not off:
            mask.append(False)
            continue
        a, b = off
        is_asst = any(a < s_end and b > s_start for s_start, s_end in spans)
        mask.append(is_asst)
    return tokens, mask


class _SkipSample(Exception):
    """Internal — sample row isn't usable as a CEAC rollout."""
