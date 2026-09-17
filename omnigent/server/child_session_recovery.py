"""Restore interrupted child sessions after their parent has initialized."""

from __future__ import annotations

import asyncio
import logging

import httpx

from omnigent.entities import Conversation
from omnigent.harness_plugins import native_agents
from omnigent.runner.session_init_protocol import build_runner_session_init_payload
from omnigent.server.runner_session_init import RunnerSessionInitializer
from omnigent.stores.conversation_store import ConversationNotFoundError, ConversationStore
from omnigent.util.session_lifecycle import is_session_closed
from omnigent.version import VERSION

_logger = logging.getLogger(__name__)


def is_parent_owned_subagent(conv: Conversation) -> bool:
    """Native mirrors belong to their parent's runtime, not a separate terminal."""
    from omnigent.server.routes._sessions.common import (
        _ACP_SUBAGENT_ID_LABEL_KEY,
        _ANTIGRAVITY_NATIVE_SUBAGENT_WRAPPER_LABEL_VALUE,
        _CLAUDE_NATIVE_WRAPPER_LABEL_KEY,
    )

    wrapper = conv.labels.get(_CLAUDE_NATIVE_WRAPPER_LABEL_KEY)
    return conv.kind == "sub_agent" and (
        bool(conv.labels.get(_ACP_SUBAGENT_ID_LABEL_KEY))
        or wrapper == _ANTIGRAVITY_NATIVE_SUBAGENT_WRAPPER_LABEL_VALUE
        or (
            wrapper is not None
            and any(wrapper == agent.subagent_wrapper_label for agent in native_agents())
        )
    )


def _restorable(conv: Conversation) -> bool:
    from omnigent.server.routes._sessions.common import (
        _intentional_stop_sessions,
        _interrupt_fenced_sessions,
    )

    return (
        conv.agent_id is not None
        and not conv.archived
        and not is_session_closed(conv.labels, conv.title)
        and conv.id not in _intentional_stop_sessions
        and conv.id not in _interrupt_fenced_sessions
    )


def _interrupted(conv: Conversation) -> bool:
    from omnigent.server.routes._sessions.common import _session_status_cache
    from omnigent.server.routes._sessions.helpers import _last_task_error_from_labels

    status = _session_status_cache.get(conv.id, conv.live_status)
    error = _last_task_error_from_labels(conv.labels)
    return status in {"running", "waiting"} or (
        status == "failed"
        and error is not None
        and error.get("code") in {"runner_disconnected", "runner_failed_to_start"}
    )


async def restore_active_children(
    parent: Conversation,
    client: httpx.AsyncClient,
    store: ConversationStore,
    initializer: RunnerSessionInitializer | None = None,
) -> None:
    """Rebind and initialize interrupted descendants on their recovered parent's runner."""
    from omnigent.runtime import get_runner_router
    from omnigent.server.routes.sessions import (
        _ensure_runner_relay,
        _publish_runner_recovered_status,
    )

    if parent.runner_id is None or not _restorable(parent):
        return
    router = get_runner_router()
    # Include idle ancestors only when needed to host an interrupted descendant.
    tree: dict[str, Conversation] = {parent.id: parent}
    frontier = [parent.id]
    while frontier:
        children = await asyncio.to_thread(store.list_child_conversation_ids_by_parent, frontier)
        rows = await asyncio.to_thread(
            store.get_conversations, [child for ids in children.values() for child in ids]
        )
        frontier = []
        for row in rows.values():
            if (
                row.id in tree
                or row.host_id is not None
                or row.runner_id is None
                or row.parent_conversation_id not in tree
                or (
                    row.runner_id != parent.runner_id
                    and router is not None
                    and router.runner_is_online(row.runner_id)
                )
                or not _restorable(row)
            ):
                continue
            tree[row.id] = row
            frontier.append(row.id)
    active = {row.id for row in tree.values() if row.id != parent.id and _interrupted(row)}
    needed = set(active)
    for row in reversed(list(tree.values())):
        if row.id in needed and row.parent_conversation_id in tree:
            needed.add(row.parent_conversation_id)

    restored = {parent.id}
    for snapshot in tree.values():
        if snapshot.id == parent.id or snapshot.id not in needed:
            continue
        assert snapshot.parent_conversation_id is not None
        owner = await asyncio.to_thread(store.get_conversation, snapshot.parent_conversation_id)
        child = await asyncio.to_thread(store.get_conversation, snapshot.id)
        if (
            owner is None
            or owner.id not in restored
            or owner.runner_id != parent.runner_id
            or not _restorable(owner)
            or child is None
            or child.runner_id != snapshot.runner_id
            or child.parent_conversation_id != owner.id
            or child.host_id is not None
            or not _restorable(child)
            or (snapshot.id in active and not _interrupted(child))
        ):
            continue
        try:
            if child.runner_id != parent.runner_id:
                child = await asyncio.to_thread(
                    store.replace_runner_id,
                    child.id,
                    parent.runner_id,
                    expected_runner_id=child.runner_id,
                )
                if child.runner_id != parent.runner_id:
                    continue
            mirrored = is_parent_owned_subagent(child)
            if not mirrored:
                if initializer is not None:
                    response = await initializer.initialize(
                        child,
                        client,
                        timeout=10.0,
                        suppress_recovery_turn=not _interrupted(child),
                        resume_interrupted_turn=_interrupted(child),
                    )
                else:
                    response = await client.post(
                        "/v1/sessions",
                        json=build_runner_session_init_payload(
                            child,
                            server_version=VERSION,
                            suppress_recovery_turn=not _interrupted(child),
                            resume_interrupted_turn=_interrupted(child),
                        ),
                        timeout=10.0,
                    )
                response.raise_for_status()
            _ensure_runner_relay(child.id, parent.runner_id, client, store)
            if not mirrored:
                from omnigent.server.routes._sessions.helpers import _last_task_error_from_labels

                fresh = await asyncio.to_thread(store.get_conversation, child.id)
                error = _last_task_error_from_labels(fresh.labels) if fresh is not None else None
                if error and error.get("code") in {
                    "runner_disconnected",
                    "runner_failed_to_start",
                }:
                    await _publish_runner_recovered_status(child.id, store)
            restored.add(child.id)
        except (httpx.HTTPError, ConnectionError, ConversationNotFoundError):
            _logger.warning("Failed to restore child session %s", snapshot.id, exc_info=True)
