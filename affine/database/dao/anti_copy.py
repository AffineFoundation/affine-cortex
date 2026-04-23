"""
Anti-Copy Results DAO

Persists copy detection results to DynamoDB.
Each model gets one record per detection round.
"""

import time
from typing import Dict, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class AntiCopyDAO(BaseDAO):

    def __init__(self):
        self.table_name = get_table_name("anti_copy_results")
        super().__init__()

    def _make_pk(self, model: str, revision: str) -> str:
        return f"{model}#{revision}"

    def _make_sk(self, timestamp: int) -> str:
        return f"ROUND#{timestamp}"

    @staticmethod
    def _normalize_result(result: Optional[Dict]) -> Optional[Dict]:
        if not result:
            return result
        if "status" not in result:
            result = result.copy()
            result["status"] = "cheat" if result.get("is_copy") else "clean"
        return result

    async def save_round(self, results: List[Dict], round_timestamp: int = None):
        """Save one round of detection results.

        Args:
            results: List of dicts, one per model:
                {uid, hotkey, model, revision, block, status, is_copy,
                 copy_of: [{uid, hotkey, model, logprobs_cosine, hs_cosine,
                            js_div, n_tasks}]}
            round_timestamp: Unix timestamp for this round (default: now)
        """
        ts = round_timestamp or int(time.time())
        ttl = self.get_ttl(30)

        items = []
        for r in results:
            item = {
                "pk": self._make_pk(r["model"], r["revision"]),
                "sk": self._make_sk(ts),
                "uid": r["uid"],
                "hotkey": r["hotkey"],
                "model": r["model"],
                "revision": r["revision"],
                "block": r["block"],
                "status": r.get("status", "cheat" if r.get("is_copy") else "clean"),
                "is_copy": r["is_copy"],
                "copy_of": r.get("copy_of", []),
                "timestamp": ts,
                "ttl": ttl,
            }
            items.append(item)

        if items:
            await self.batch_write(items)

    async def get_latest(self, model: str, revision: str) -> Optional[Dict]:
        """Get the latest detection result for a model.

        Returns:
            Latest result dict, or None
        """
        results = await self.query(
            pk=self._make_pk(model, revision),
            limit=1,
            reverse=True,
        )
        return self._normalize_result(results[0]) if results else None

    async def get_history(self, model: str, revision: str) -> List[Dict]:
        """Get all detection results for a model."""
        return [
            self._normalize_result(result)
            for result in await self.query(pk=self._make_pk(model, revision))
        ]
