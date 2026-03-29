"""
OpenSkill Matches DAO

Stores match records for idempotency and audit trail.

PK: ENV#{env}
SK: TASK#{task_id}
TTL: 30 days
"""

import time
from typing import Dict, Any, List, Set, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class OpenSkillMatchesDAO(BaseDAO):

    def __init__(self):
        self.table_name = get_table_name("openskill_matches")
        super().__init__()

    def _make_pk(self, env: str) -> str:
        return f"ENV#{env}"

    def _make_sk(self, task_id: int) -> str:
        return f"TASK#{task_id}"

    async def is_task_processed(self, env: str, task_id: int) -> bool:
        pk = self._make_pk(env)
        sk = self._make_sk(task_id)
        item = await self.get(pk, sk)
        return item is not None

    async def get_processed_task_ids(self, env: str) -> Set[int]:
        """Get all processed task IDs for an environment."""
        from affine.database.client import get_client
        client = get_client()

        pk = self._make_pk(env)
        params = {
            'TableName': self.table_name,
            'KeyConditionExpression': 'pk = :pk',
            'ExpressionAttributeValues': {':pk': {'S': pk}},
            'ProjectionExpression': 'task_id',
        }

        task_ids = set()
        while True:
            response = await client.query(**params)
            for item in response.get('Items', []):
                tid = item.get('task_id', {})
                if 'N' in tid:
                    task_ids.add(int(tid['N']))
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
            params['ExclusiveStartKey'] = last_key

        return task_ids

    async def save_match(
        self,
        env: str,
        task_id: int,
        participants: List[Dict[str, Any]],
        n_participants: int,
        skipped: bool = False,
        skip_reason: Optional[str] = None,
        ttl_days: int = 30,
    ) -> Dict[str, Any]:
        item = {
            'pk': self._make_pk(env),
            'sk': self._make_sk(task_id),
            'env': env,
            'task_id': task_id,
            'processed_at': int(time.time()),
            'participants': participants,
            'n_participants': n_participants,
            'skipped': skipped,
            'ttl': self.get_ttl(ttl_days),
        }
        if skip_reason:
            item['skip_reason'] = skip_reason
        return await self.put(item)

    async def batch_save_matches(self, matches: List[Dict[str, Any]]):
        if matches:
            await self.batch_write(matches)
