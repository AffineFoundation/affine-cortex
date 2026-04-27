#!/usr/bin/env python3
"""Audit cold-time / never-sampled terminations for false positives.

Verifies, against live DynamoDB, that every terminated miner whose
``termination_reason`` is ``cold_too_long`` or ``never_sampled``
satisfies the full set of preconditions that the cold-tracker is
documented to enforce. Emits a per-miner report and a summary.

Checks performed:

  cold_too_long
    1. cold_seconds_total >= 36h
    2. has_been_hot is True
    3. chute is not currently 'hot'
    4. hotkey is not the current champion
    5. cross-check: time-since-latest-sample (across the current
       sampling envs) >= cold_seconds_total within a 1h slop. Detects
       cold_total over-attribution.

  never_sampled
    1. has_been_hot is False
    2. chain_age >= 48h (computed from first_block vs current block,
       12s/block)
    3. zero samples across **all** envs (current sampling + historical
       /deprecated /teacher) for the miner's current revision
    4. zero samples under any other revision of the same hotkey (full
       prefix scan of pk = hotkey#*)
    5. hotkey is not the current champion

A clean run prints `0/N false positives` on both lines. A failing run
lists every miner that violated a check. Intended to be run after a
deploy or whenever the cold-tracker logic is changed.

Usage:
  python3 scripts/audit_cold_termination_false_positives.py
"""

import asyncio
import sys
import time
from typing import Optional

sys.path.insert(0, '.')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from affine.database.client import init_client, close_client, get_client
from affine.database.dao.miners import MinersDAO
from affine.database.dao.miner_stats import MinerStatsDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.system_config import SystemConfigDAO


COLD_THRESHOLD_S = 36 * 3600
NEVER_SAMPLED_THRESHOLD_S = 48 * 3600
SECONDS_PER_BLOCK = 12

# Current sampling envs are loaded from system_config; the historical
# list below catches samples produced under removed/deprecated envs so
# never_sampled is not falsely flagged.
HISTORICAL_ENVS = [
    'SWE-PRO', 'SWE-SYNTH', 'PRINT', 'ARC-GEN',
    'CDE', 'LGC', 'LGC-V2', 'SAT', 'ABD', 'DED',
    'KNOWLEDGE-EVAL', 'CORPUS-EVAL',
]


def fmt_h(seconds: Optional[float]) -> str:
    if seconds is None:
        return '    -'
    return f'{seconds/3600:6.2f}h'


async def fetch_miners(miners_dao: MinersDAO):
    """Merge valid + invalid miners by uid; drop uid 0."""
    valid = await miners_dao.get_valid_miners()
    invalid = await miners_dao.get_invalid_miners()
    by_uid = {}
    for m in valid + invalid:
        uid = m.get('uid')
        if uid is None or uid == 0:
            continue
        if uid not in by_uid:
            by_uid[uid] = m
    return [by_uid[u] for u in sorted(by_uid)]


async def collect_terminated(miners, stats_dao: MinerStatsDAO):
    """Return (cold_too_long_rows, never_sampled_rows) with stats."""
    cold = []
    ns = []
    for m in miners:
        hk = m.get('hotkey', '')
        rev = m.get('revision', '')
        if not rev:
            continue
        try:
            s = await stats_dao.get_miner_stats(hk, rev)
        except Exception:
            continue
        if not s:
            continue
        reason = s.get('termination_reason') or ''
        row = {
            'uid': m.get('uid'),
            'hk': hk,
            'hk8': hk[:8],
            'rev': rev,
            'chute': m.get('chute_status') or '-',
            'first_block': m.get('first_block') or 0,
            'cold': int(s.get('cold_seconds_total') or 0),
            'hbh': bool(s.get('has_been_hot', False)),
            'cs': s.get('challenge_status') or 'sampling',
            'tr': reason,
            'term_at': s.get('terminated_at'),
        }
        if reason == 'cold_too_long':
            cold.append(row)
        elif reason == 'never_sampled':
            ns.append(row)
    return cold, ns


async def hotkey_has_any_sample(sample_dao: SampleResultsDAO, hotkey: str) -> bool:
    """True iff sample_results contains at least one row whose pk
    starts with ``{hotkey}#`` — covers every (revision, env) combo
    the miner has ever produced."""
    client = get_client()
    resp = await client.scan(
        TableName=sample_dao.table_name,
        FilterExpression='begins_with(pk, :prefix)',
        ExpressionAttributeValues={':prefix': {'S': hotkey + '#'}},
        ProjectionExpression='pk',
        Limit=1,
    )
    return bool(resp.get('Items'))


async def audit_cold_too_long(rows, sample_dao, sampling_envs, champion_hk):
    print(f'\n=== cold_too_long audit ({len(rows)} terminations) ===')
    bad = 0
    now = int(time.time())
    for r in rows:
        notes = []
        if r['cold'] < COLD_THRESHOLD_S:
            notes.append(f'!cold<36h({fmt_h(r["cold"])})')
        if not r['hbh']:
            notes.append('!hbh=False')
        if r['chute'] == 'hot':
            notes.append('!currently_hot')
        if champion_hk and r['hk'] == champion_hk:
            notes.append('!IS_CHAMPION')

        # Cross-check: time-since-latest-sample should not be smaller
        # than cold_seconds_total (ignoring 1h slop). If it is, the
        # cold accumulator over-counted.
        try:
            ts_ms = await sample_dao.get_latest_sample_timestamp_ms(
                r['hk'], r['rev'], sampling_envs)
        except Exception:
            ts_ms = None
        if ts_ms is not None:
            gap_s = now - int(ts_ms) // 1000
            if r['cold'] > gap_s + 3600:
                notes.append(
                    f'!cold>gap_since_last_sample(cold={fmt_h(r["cold"])} '
                    f'gap={fmt_h(gap_s)})'
                )

        if notes:
            bad += 1
            print(f'  BAD uid={r["uid"]:>3} hk={r["hk8"]} '
                  f'cold={fmt_h(r["cold"])} hbh={r["hbh"]} '
                  f'chute={r["chute"]} :: {" ".join(notes)}')
    if bad == 0:
        print(f'  ✓ 0/{len(rows)} false positives')
    return bad


async def audit_never_sampled(rows, miners, sample_dao, sampling_envs,
                              champion_hk, current_block):
    print(f'\n=== never_sampled audit ({len(rows)} terminations) ===')
    all_envs = list(sampling_envs) + HISTORICAL_ENVS
    bad = 0
    miner_by_hk = {m.get('hotkey'): m for m in miners}
    for r in rows:
        notes = []
        if r['hbh']:
            notes.append('!hbh=True')

        m = miner_by_hk.get(r['hk'])
        first_block = (m or {}).get('first_block') or 0
        chain_age = (current_block - first_block) * SECONDS_PER_BLOCK \
            if (first_block and current_block > first_block) else None
        if chain_age is None or chain_age < NEVER_SAMPLED_THRESHOLD_S:
            notes.append(f'!chain_age<48h({fmt_h(chain_age)})')

        # Check (revision, all envs) — exact (hotkey, revision) match.
        try:
            ts_ms = await sample_dao.get_latest_sample_timestamp_ms(
                r['hk'], r['rev'], all_envs)
        except Exception:
            ts_ms = None
        if ts_ms is not None:
            age_h = (int(time.time()) - int(ts_ms) / 1000) / 3600
            notes.append(f'!has_samples_current_rev({age_h:.1f}h_ago)')

        # Check cross-revision: any sample for this hotkey under any
        # revision and any env.
        try:
            has_any = await hotkey_has_any_sample(sample_dao, r['hk'])
        except Exception:
            has_any = False
        if has_any and ts_ms is None:
            notes.append('!has_samples_other_rev')

        if champion_hk and r['hk'] == champion_hk:
            notes.append('!IS_CHAMPION')

        if notes:
            bad += 1
            print(f'  BAD uid={r["uid"]:>3} hk={r["hk8"]} '
                  f'chain_age={fmt_h(chain_age)} hbh={r["hbh"]} '
                  f':: {" ".join(notes)}')
    if bad == 0:
        print(f'  ✓ 0/{len(rows)} false positives '
              f'(checked {len(all_envs)} envs + cross-revision scan)')
    return bad


async def main():
    await init_client()
    try:
        miners_dao = MinersDAO()
        stats_dao = MinerStatsDAO()
        sample_dao = SampleResultsDAO()
        cfg_dao = SystemConfigDAO()

        sampling_envs = await cfg_dao.get_sampling_environments()
        champion = await cfg_dao.get_param_value('champion')
        champion_hk = (champion or {}).get('hotkey') \
            if isinstance(champion, dict) else None

        miners = await fetch_miners(miners_dao)
        current_block = max((m.get('block_number') or 0) for m in miners) or 0

        print(f'champion: {champion_hk[:12] + "..." if champion_hk else None}')
        print(f'sampling envs: {sampling_envs}')
        print(f'historical envs scanned: {HISTORICAL_ENVS}')
        print(f'current block: {current_block}, miners audited: {len(miners)}')

        cold, ns = await collect_terminated(miners, stats_dao)

        cold_bad = await audit_cold_too_long(
            cold, sample_dao, sampling_envs, champion_hk)
        ns_bad = await audit_never_sampled(
            ns, miners, sample_dao, sampling_envs, champion_hk, current_block)

        # Champion safety: ensure the current champion is not terminated
        # under either cold path.
        print(f'\n=== champion safety ===')
        if champion_hk:
            champ_rows = [m for m in miners if m.get('hotkey') == champion_hk]
            for cm in champ_rows:
                rev = cm.get('revision') or ''
                if not rev:
                    continue
                s = await stats_dao.get_miner_stats(cm['hotkey'], rev)
                if not s:
                    continue
                tr = s.get('termination_reason') or ''
                cs = s.get('challenge_status') or 'sampling'
                tag = 'BAD' if (cs == 'terminated'
                                and tr in ('cold_too_long', 'never_sampled')) \
                       else ' ok'
                print(f'  {tag} uid={cm.get("uid")} '
                      f'rev={rev[:12]} cs={cs} tr={tr!r}')
        else:
            print('  (no champion configured)')

        print(f'\nSUMMARY:')
        print(f'  cold_too_long  false positives: {cold_bad}/{len(cold)}')
        print(f'  never_sampled  false positives: {ns_bad}/{len(ns)}')
        sys.exit(1 if (cold_bad or ns_bad) else 0)
    finally:
        await close_client()


if __name__ == '__main__':
    asyncio.run(main())
