"""Recovery restores active descendants without replaying finished work."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from omnigent.db.utils import generate_agent_id
from omnigent.entities import Conversation
from omnigent.server.child_session_recovery import restore_active_children
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore


@pytest.fixture
def recovery_tree(
    db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    SqlAlchemyConversationStore, Conversation, Callable[..., Conversation], Mock, AsyncMock
]:
    from omnigent.server.routes import sessions

    store = SqlAlchemyConversationStore(db_uri)
    agent = SqlAlchemyAgentStore(db_uri).create(generate_agent_id(), "test", "bundle")
    parent = store.create_conversation(runner_id="new", agent_id=agent.id)
    relay, recovered = Mock(), AsyncMock()
    monkeypatch.setattr(sessions, "_ensure_runner_relay", relay)
    monkeypatch.setattr(sessions, "_publish_runner_recovered_status", recovered)
    monkeypatch.setattr("omnigent.runtime.get_runner_router", lambda: None)

    def child(
        status: str = "running", *, owner: Conversation = parent, **kwargs: Any
    ) -> Conversation:
        row = store.create_conversation(
            kind="sub_agent",
            parent_conversation_id=owner.id,
            agent_id=agent.id,
            runner_id="old",
            **kwargs,
        )
        store.set_session_live_status(row.id, status)
        return store.get_conversation(row.id)  # type: ignore[return-value]

    return store, parent, child, relay, recovered


@pytest.mark.asyncio
async def test_restore_active_descendants_and_idle_ancestor(recovery_tree: Any) -> None:
    store, parent, child, relay, recovered = recovery_tree
    active = child()
    waiting = child("waiting")
    disconnected = child("failed")
    store.set_labels(
        disconnected.id,
        {
            "omnigent.last_task_error_code": "runner_disconnected",
            "omnigent.last_task_error_message": "Disconnected",
        },
    )
    idle_ancestor = child("idle")
    nested = child(owner=idle_ancestor)
    untouched = [child("idle"), child("failed")]
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        assert store.get_conversation(body["session_id"]).runner_id == "new"
        return httpx.Response(201)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://runner"
    ) as client:
        await restore_active_children(parent, client, store)

    by_id = {call["session_id"]: call["session_init"] for call in calls}
    assert set(by_id) == {active.id, waiting.id, disconnected.id, idle_ancestor.id, nested.id}
    assert {sid for sid, envelope in by_id.items() if envelope["suppress_recovery_turn"]} == {
        idle_ancestor.id
    }
    assert {sid for sid, envelope in by_id.items() if envelope["resume_interrupted_turn"]} == {
        active.id,
        waiting.id,
        disconnected.id,
        nested.id,
    }
    ids = list(by_id)
    assert ids.index(idle_ancestor.id) < ids.index(nested.id)
    assert relay.call_count == 5
    assert recovered.await_count == 1
    assert all(store.get_conversation(row.id).runner_id == "old" for row in untouched)


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusion", ["closed", "archived", "stopped", "hosted", "live_runner"])
async def test_do_not_restore_excluded_children(
    recovery_tree: Any, monkeypatch: pytest.MonkeyPatch, exclusion: str
) -> None:
    from omnigent.server.routes._sessions.common import _intentional_stop_sessions

    store, parent, child, relay, _ = recovery_tree
    row = child()
    if exclusion == "closed":
        store.set_labels(row.id, {"omnigent.closed": "true"})
    elif exclusion == "archived":
        store.update_conversation(row.id, archived=True)
    elif exclusion == "stopped":
        _intentional_stop_sessions.add(row.id)
    elif exclusion == "hosted":
        store.set_host_id(row.id, "a" * 32, workspace="/tmp")
    else:
        monkeypatch.setattr(
            "omnigent.runtime.get_runner_router", lambda: Mock(runner_is_online=lambda _: True)
        )
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: pytest.fail("unexpected init"))
        ) as client:
            await restore_active_children(parent, client, store)
        assert store.get_conversation(row.id).runner_id == "old"
        relay.assert_not_called()
    finally:
        _intentional_stop_sessions.discard(row.id)


@pytest.mark.asyncio
async def test_mirror_rebinds_without_independent_terminal_or_success_status(
    recovery_tree: Any,
) -> None:
    store, parent, child, relay, recovered = recovery_tree
    row = child()
    store.set_labels(row.id, {"omnigent.wrapper": "codex-native-ui-subagent"})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("mirror initialized"))
    ) as client:
        await restore_active_children(parent, client, store)
    assert store.get_conversation(row.id).runner_id == "new"
    relay.assert_called_once()
    recovered.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_child_init_does_not_recover_its_descendants_or_block_siblings(
    recovery_tree: Any,
) -> None:
    store, parent, child, relay, recovered = recovery_tree
    failed = child()
    nested = child(owner=failed)
    sibling = child()
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        session_id = json.loads(request.content)["session_id"]
        calls.append(session_id)
        return httpx.Response(503 if session_id == failed.id else 201)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://runner"
    ) as client:
        await restore_active_children(parent, client, store)
    assert set(calls) == {failed.id, sibling.id}
    assert store.get_conversation(nested.id).runner_id == "old"
    assert relay.call_count == 1
    recovered.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_rebind_is_preserved(
    recovery_tree: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, parent, child, relay, _ = recovery_tree
    row = child()
    replace = store.replace_runner_id

    def competing_rebind(session_id: str, runner_id: str, **kwargs: Any) -> Conversation:
        replace(session_id, "manual")
        return replace(session_id, runner_id, **kwargs)

    monkeypatch.setattr(store, "replace_runner_id", competing_rebind)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("stale init"))
    ) as client:
        await restore_active_children(parent, client, store)
    assert store.get_conversation(row.id).runner_id == "manual"
    relay.assert_not_called()


@pytest.mark.asyncio
async def test_child_finishing_after_scan_is_not_restored(
    recovery_tree: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omnigent.server.routes._sessions.common import _session_status_cache

    store, parent, child, relay, _ = recovery_tree
    row = child()
    get = store.get_conversation

    def finish_before_recheck(session_id: str) -> Conversation | None:
        if session_id == row.id:
            store.set_session_live_status(row.id, "idle")
            _session_status_cache[row.id] = "idle"
        return get(session_id)

    monkeypatch.setattr(store, "get_conversation", finish_before_recheck)
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: pytest.fail("finished child initialized"))
        ) as client:
            await restore_active_children(parent, client, store)
        assert get(row.id).runner_id == "old"
        relay.assert_not_called()
    finally:
        _session_status_cache.pop(row.id, None)
