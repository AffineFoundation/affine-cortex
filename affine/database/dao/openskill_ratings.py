"""
OpenSkill Ratings DAO

Stores per-(miner, env) OpenSkill ratings (mu, sigma).

PK: MINER#{hotkey}#REV#{revision}
SK: ENV#{env}
"""

import time
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class OpenSkillRatingsDAO(BaseDAO):

    def __init__(self):
        self.table_name = get_table_name("openskill_ratings")
        super().__init__()

    def _make_pk(self, hotkey: str, revision: str) -> str:
        return f"MINER#{hotkey}#REV#{revision}"

    def _make_sk(self, env: str) -> str:
        return f"ENV#{env}"

    async def get_rating(
        self, hotkey: str, revision: str, env: str
    ) -> Optional[Dict[str, Any]]:
        pk = self._make_pk(hotkey, revision)
        sk = self._make_sk(env)
        return await self.get(pk, sk)

    async def get_all_ratings_for_miner(
        self, hotkey: str, revision: str
    ) -> List[Dict[str, Any]]:
        pk = self._make_pk(hotkey, revision)
        return await self.query(pk)

    async def save_rating(
        self, hotkey: str, revision: str, env: str,
        mu: float, sigma: float
    ) -> Dict[str, Any]:
        item = {
            'pk': self._make_pk(hotkey, revision),
            'sk': self._make_sk(env),
            'hotkey': hotkey,
            'revision': revision,
            'env': env,
            'mu': mu,
            'sigma': sigma,
            'updated_at': int(time.time()),
        }
        return await self.put(item)

    async def batch_save_ratings(self, ratings: List[Dict[str, Any]]):
        items = []
        now = int(time.time())
        for r in ratings:
            items.append({
                'pk': self._make_pk(r['hotkey'], r['revision']),
                'sk': self._make_sk(r['env']),
                'hotkey': r['hotkey'],
                'revision': r['revision'],
                'env': r['env'],
                'mu': r['mu'],
                'sigma': r['sigma'],
                'updated_at': now,
            })
        if items:
            await self.batch_write(items)

    async def get_all_ratings(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Scan all ratings. Returns {hotkey#revision: {env: {mu, sigma}}}."""
        from affine.database.client import get_client
        client = get_client()

        all_items = []
        params = {'TableName': self.table_name}
        while True:
            response = await client.scan(**params)
            all_items.extend(
                self._deserialize(item) for item in response.get('Items', [])
            )
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
            params['ExclusiveStartKey'] = last_key

        result = {}
        for item in all_items:
            key = f"{item['hotkey']}#{item['revision']}"
            env = item['env']
            if key not in result:
                result[key] = {}
            result[key][env] = {'mu': item['mu'], 'sigma': item['sigma']}

        return result
