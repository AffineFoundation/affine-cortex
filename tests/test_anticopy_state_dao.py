"""``AntiCopyStateDAO`` partial-update + readback behaviour.

These tests run against an in-memory fake DDB client that supports
just the operations the DAO uses (``update_item`` with ``SET ...``
and ``get_item``). The goal is to verify:
 - set_state() does NOT clobber unrelated fields
 - get_state() round-trips the data unchanged
 - get_champion_tokenizer_sig / get_active_champion convenience methods
"""

from __future__ import annotations

import pytest

import affine.database.dao.anticopy as anticopy_dao_mod
from affine.database.dao.anticopy import AntiCopyStateDAO


class _FakeDynamoClient:
    """Minimal in-memory DDB stub. Only handles single-row update_item +
    get_item against ``Key={"key": {"S": ...}}`` because that's the
    shape AntiCopyStateDAO uses."""

    def __init__(self):
        self.items = {}                # key_str -> {attr: {type: val}}

    async def update_item(
        self, *, TableName, Key, UpdateExpression,
        ExpressionAttributeNames, ExpressionAttributeValues,
    ):
        k = Key["key"]["S"]
        row = self.items.setdefault(k, {"key": {"S": k}})
        # Parse "SET #a = :a, #b = :b, ..."
        if not UpdateExpression.startswith("SET "):
            raise NotImplementedError(UpdateExpression)
        for assignment in UpdateExpression[4:].split(", "):
            lhs, rhs = [s.strip() for s in assignment.split("=")]
            attr = ExpressionAttributeNames[lhs]
            value = ExpressionAttributeValues[rhs]
            row[attr] = value

    async def get_item(self, *, TableName, Key):
        k = Key["key"]["S"]
        if k in self.items:
            return {"Item": self.items[k]}
        return {}


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeDynamoClient()
    # ``AntiCopyStateDAO`` does ``from affine.database.client import
    # get_client``, so the binding inside the DAO's module namespace is
    # the one we need to override, not the source module's.
    monkeypatch.setattr(anticopy_dao_mod, "get_client", lambda: client)
    return client


@pytest.mark.asyncio
async def test_partial_update_preserves_other_fields(fake_client):
    dao = AntiCopyStateDAO()
    # initial bulk write
    await dao.set_state(
        active_champion_uid=48,
        active_champion_hotkey="5E7eCacd...",
        active_champion_revision="8ffc0655",
        active_champion_day="2026-05-15",
        champion_tokenizer_sig="A" * 64,
    )
    # later partial update — only the tokenizer sig changes
    await dao.set_state(champion_tokenizer_sig="B" * 64)

    state = await dao.get_state()
    assert state["active_champion_uid"] == 48
    assert state["active_champion_hotkey"] == "5E7eCacd..."
    assert state["active_champion_day"] == "2026-05-15"
    assert state["champion_tokenizer_sig"] == "B" * 64
    assert "updated_at" in state


@pytest.mark.asyncio
async def test_convenience_getters(fake_client):
    dao = AntiCopyStateDAO()
    # empty state — getters return safe defaults
    assert await dao.get_active_champion() is None
    assert await dao.get_champion_tokenizer_sig() == ""
    assert await dao.get_state() == {}

    await dao.set_state(active_champion_uid=14, champion_tokenizer_sig="sig123")
    assert await dao.get_active_champion() == 14
    assert await dao.get_champion_tokenizer_sig() == "sig123"


@pytest.mark.asyncio
async def test_empty_set_state_noop(fake_client):
    dao = AntiCopyStateDAO()
    await dao.set_state()   # nothing to write
    assert fake_client.items == {}   # didn't create a row
