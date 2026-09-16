"""Automatic recovery of a confirmed crash on the runner's original host."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from omnigent.db.db_models import current_workspace_id
from omnigent.entities import Conversation
from omnigent.server import shutdown_state
from omnigent.server.host_registry import HostRegistry
from omnigent.stores.conversation_store import ConversationNotFoundError, ConversationStore
from omnigent.util.session_lifecycle import is_session_closed

_logger = logging.getLogger(__name__)

RECOVERY_STOPPED_LABEL = "omnigent.runner_recovery.stopped"
RECOVERY_MODE_LABEL = "omnigent.runner_recovery.mode"
RECOVERY_ATTEMPT_LABEL = "omnigent.runner_recovery.attempted_at"
RECOVERY_COOLDOWN_S = 60


def is_parent_owned_subagent(conv: Conversation) -> bool:
    """Identify vendor subagent mirrors, which have no independent session runtime."""
    from omnigent.harness_plugins import native_agents
    from omnigent.server.routes._sessions.common import (
        _ACP_SUBAGENT_ID_LABEL_KEY,
        _ANTIGRAVITY_NATIVE_SUBAGENT_WRAPPER_LABEL_VALUE,
        _CLAUDE_NATIVE_WRAPPER_LABEL_KEY,
    )

    if conv.kind != "sub_agent":
        return False
    wrapper = conv.labels.get(_CLAUDE_NATIVE_WRAPPER_LABEL_KEY)
    return bool(conv.labels.get(_ACP_SUBAGENT_ID_LABEL_KEY)) or (
        wrapper is not None
        and (
            wrapper == _ANTIGRAVITY_NATIVE_SUBAGENT_WRAPPER_LABEL_VALUE
            or any(wrapper == agent.subagent_wrapper_label for agent in native_agents())
        )
    )


def can_restore_session(conv: Conversation) -> bool:
    """Whether recovery may restore a session without new user input."""
    return (
        conv.agent_id is not None
        and not conv.archived
        and not is_session_closed(conv.labels, conv.title)
        and conv.labels.get(RECOVERY_STOPPED_LABEL) != "true"
    )


def was_interrupted(conv: Conversation, *, reconciled: bool = False) -> bool:
    """Recognize in-flight work even when the disconnect relay failed it first."""
    from omnigent.server.routes._sessions.common import (
        _intentional_stop_sessions,
        _interrupt_fenced_sessions,
        _session_status_cache,
    )
    from omnigent.server.routes._sessions.helpers import _last_task_error_from_labels
    from omnigent.server.routes._sessions.orchestration import _MID_TURN_STATUSES

    if (
        not can_restore_session(conv)
        or conv.id in _intentional_stop_sessions
        or conv.id in _interrupt_fenced_sessions
    ):
        return False
    status = _session_status_cache.get(conv.id, conv.live_status)
    if status in _MID_TURN_STATUSES:
        return True
    error = _last_task_error_from_labels(conv.labels)
    codes = {"runner_disconnected"}
    if reconciled:
        codes.add("runner_failed_to_start")
    return status == "failed" and error is not None and error.get("code") in codes


def recovery_suppresses_turn(conv: Conversation) -> bool:
    """A parent restored only to host its children must not replay old input."""
    return conv.labels.get(RECOVERY_MODE_LABEL) == f"{conv.runner_id}:restore"


async def may_initialize_session(conv: Conversation, store: ConversationStore) -> bool:
    """Initialize only independent sessions whose lifecycle still allows recovery."""
    # The parent's harness owns these children, including their recovery status.
    # Initializing the mirror would create an unrelated, empty native terminal.
    if is_parent_owned_subagent(conv):
        return False
    seen: set[str] = set()
    check_ancestors = conv.labels.get(RECOVERY_MODE_LABEL, "").startswith(f"{conv.runner_id}:")
    while True:
        if (
            conv.archived
            or is_session_closed(conv.labels, conv.title)
            or conv.labels.get(RECOVERY_STOPPED_LABEL) == "true"
        ):
            return False
        seen.add(conv.id)
        if not check_ancestors or not conv.parent_conversation_id:
            return True
        if conv.parent_conversation_id in seen:
            return False
        parent = await asyncio.to_thread(store.get_conversation, conv.parent_conversation_id)
        if parent is None:
            return False
        conv = parent


async def prepare_recovery_bindings(
    root: Conversation,
    runner_id: str,
    interrupted: Sequence[Conversation],
    store: ConversationStore,
) -> bool:
    """Restore the affected group before the replacement runner can connect.

    The host-launch lock owns this operation. Conditional writes preserve a
    child's binding when a concurrent user action has already moved it.
    """
    if root.runner_id is None or not can_restore_session(root):
        return False
    previous_runner_id = root.runner_id
    eligible: list[Conversation] = []
    for snapshot in interrupted:
        fresh = await asyncio.to_thread(store.get_conversation, snapshot.id)
        if (
            fresh is not None
            and fresh.runner_id == root.runner_id
            and fresh.host_id == snapshot.host_id
            and fresh.workspace == snapshot.workspace
            and fresh.parent_conversation_id == snapshot.parent_conversation_id
            and fresh.agent_id == snapshot.agent_id
            and was_interrupted(fresh, reconciled=True)
        ):
            eligible.append(fresh)
    if not eligible:
        return False
    resume_ids = {conv.id for conv in eligible}
    await asyncio.to_thread(store.set_labels, root.id, {RECOVERY_ATTEMPT_LABEL: str(time.time())})
    for conv in [root, *(c for c in eligible if c.id != root.id)]:
        if is_parent_owned_subagent(conv):
            mode = "parent"
        else:
            mode = "resume" if conv.id in resume_ids else "restore"
        try:
            await asyncio.to_thread(
                store.set_labels, conv.id, {RECOVERY_MODE_LABEL: f"{runner_id}:{mode}"}
            )
            rebound = await asyncio.to_thread(
                store.replace_runner_id, conv.id, runner_id, expected_runner_id=previous_runner_id
            )
        except ConversationNotFoundError:
            if conv.id == root.id:
                return False
            continue
        if conv.id == root.id and rebound.runner_id != runner_id:
            return False
    return True


class HostRunnerRecovery:
    """Schedule one recovery attempt without blocking the host frame reader."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        host_registry: HostRegistry,
    ) -> None:
        self._store = conversation_store
        self._hosts = host_registry
        self._tasks: dict[tuple[int, str], asyncio.Task[None]] = {}

    def schedule(
        self,
        runner_id: str,
        affected: Sequence[Conversation],
        interrupted: Sequence[Conversation],
    ) -> None:
        """Recover only a host-owned group with work interrupted by this crash.

        Call with snapshots taken before crash reconciliation fails idle roots.
        Tunnel loss alone is not proof of an unexpected process exit.
        """
        key = (current_workspace_id(), runner_id)
        if shutdown_state.server_shutting_down() or key in self._tasks:
            return
        roots = [c for c in affected if c.host_id is not None and can_restore_session(c)]
        if len(roots) != 1:
            return
        root = roots[0]
        members = {root.id}
        while True:
            descendants = {
                c.id for c in affected if c.parent_conversation_id in members and c.host_id is None
            }
            if descendants <= members:
                break
            members.update(descendants)
        interrupted = [c for c in interrupted if c.id in members]
        if not interrupted:
            return
        task = asyncio.create_task(
            self._recover(root, runner_id, interrupted), name=f"recover-host-runner-{runner_id}"
        )
        self._tasks[key] = task
        task.add_done_callback(lambda _: self._tasks.pop(key, None))

    async def _recover(
        self, root: Conversation, runner_id: str, interrupted: Sequence[Conversation]
    ) -> None:
        from omnigent.server.routes._sessions.helpers import _query_host_runner_status
        from omnigent.server.routes.sessions import _launch_runner_on_host

        try:
            if root.host_id is None:
                return
            host = self._hosts.get(root.host_id)
            if host is None:
                return
            # Recheck with the original supervisor: a reconnect or explicit
            # stop can overtake the crash report while it is being delivered.
            if await _query_host_runner_status(host, self._hosts, runner_id) != "dead":
                return
            fresh = await asyncio.to_thread(self._store.get_conversation, root.id)
            if (
                fresh is None
                or fresh.runner_id != runner_id
                or fresh.host_id != root.host_id
                or not can_restore_session(fresh)
                or shutdown_state.server_shutting_down()
            ):
                return
            try:
                last_attempt = float(fresh.labels.get(RECOVERY_ATTEMPT_LABEL, "0"))
            except ValueError:
                return
            if time.time() - last_attempt < RECOVERY_COOLDOWN_S:
                return
            attempt = await _launch_runner_on_host(
                fresh, self._store, self._hosts, host, recovery_sessions=interrupted
            )
            if attempt.error_code is not None:
                _logger.warning(
                    "Automatic runner recovery refused for %s: %s", root.id, attempt.error_code
                )
        except Exception:
            _logger.exception("Automatic runner recovery failed for %s", root.id)

    async def shutdown(self) -> None:
        """Cancel attempts before tearing down the server's transports."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
