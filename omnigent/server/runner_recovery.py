"""Automatic recovery of a confirmed crash on the runner's original host."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Sequence
from contextlib import suppress

from omnigent.db.db_models import current_workspace_id
from omnigent.entities import Conversation
from omnigent.server import shutdown_state
from omnigent.server.host_registry import HostRegistry
from omnigent.stores.conversation_store import ConversationNotFoundError, ConversationStore
from omnigent.util.session_lifecycle import is_session_closed

_logger = logging.getLogger(__name__)

RECOVERY_LABEL_NAMESPACE = "omnigent.runner_recovery."
RECOVERY_STOPPED_LABEL = f"{RECOVERY_LABEL_NAMESPACE}stopped"
RECOVERY_MODE_LABEL = f"{RECOVERY_LABEL_NAMESPACE}mode"
RECOVERY_ATTEMPT_LABEL = f"{RECOVERY_LABEL_NAMESPACE}attempted_at"
RECOVERY_ATTEMPT_RUNNER_LABEL = f"{RECOVERY_LABEL_NAMESPACE}attempted_runner"
RECOVERY_COOLDOWN_S = 60
RECOVERY_RESULT_GRACE_S = 60


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


def recovery_waits_for_parent(conv: Conversation) -> bool:
    """A replacement runner alone cannot confirm a mirrored child's recovery."""
    return conv.labels.get(RECOVERY_MODE_LABEL) == f"{conv.runner_id}:parent"


def _may_initialize_snapshot(conv: Conversation) -> bool:
    if (
        not can_restore_session(conv)
        or is_parent_owned_subagent(conv)
        or recovery_waits_for_parent(conv)
    ):
        return False
    return conv.labels.get(RECOVERY_MODE_LABEL) != f"{conv.runner_id}:resume" or was_interrupted(
        conv, reconciled=True
    )


async def may_initialize_session(conv: Conversation, store: ConversationStore) -> bool:
    """Recheck recovery ownership and lifecycle before sending initialization."""
    from omnigent.server.routes._sessions.common import (
        _intentional_stop_sessions,
        _interrupt_fenced_sessions,
    )

    if not _may_initialize_snapshot(conv):
        return False
    target = conv
    seen: set[str] = set()
    check_ancestors = conv.labels.get(RECOVERY_MODE_LABEL, "").startswith(f"{conv.runner_id}:")
    while True:
        if not can_restore_session(conv):
            return False
        seen.add(conv.id)
        if not check_ancestors or not conv.parent_conversation_id:
            break
        if conv.parent_conversation_id in seen:
            return False
        parent = await asyncio.to_thread(store.get_conversation, conv.parent_conversation_id)
        if parent is None or parent.runner_id != target.runner_id:
            return False
        conv = parent
    if len(seen) > 1:
        fresh = await asyncio.to_thread(store.get_conversation, target.id)
        if (
            fresh is None
            or fresh.runner_id != target.runner_id
            or not _same_recovery_location(fresh, target)
            or not _may_initialize_snapshot(fresh)
        ):
            return False
    # An ancestor read can yield to Stop on this replica; check its fences
    # again after the final await, even before durable labels catch up.
    return not any(
        sid in _intentional_stop_sessions or sid in _interrupt_fenced_sessions for sid in seen
    )


def _same_recovery_location(fresh: Conversation, snapshot: Conversation) -> bool:
    return (
        fresh.host_id == snapshot.host_id
        and fresh.workspace == snapshot.workspace
        and fresh.parent_conversation_id == snapshot.parent_conversation_id
        and fresh.agent_id == snapshot.agent_id
    )


async def rollback_recovery_bindings(
    runner_id: str, previous_runner_id: str, store: ConversationStore
) -> None:
    """Drain rollback writes even when shutdown cancels the launch observer."""
    task = asyncio.create_task(
        asyncio.to_thread(_rollback_recovery_bindings, runner_id, previous_runner_id, store)
    )
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def _rollback_recovery_bindings(
    runner_id: str, previous_runner_id: str, store: ConversationStore
) -> None:
    """Undo only this unlaunched attempt; preserve concurrent user rebindings."""
    for row in store.list_conversations_by_runner_id(runner_id):
        try:
            rebound = store.replace_runner_id(
                row.id, previous_runner_id, expected_runner_id=runner_id
            )
            if rebound.runner_id == previous_runner_id and row.host_id is not None:
                store.set_labels(row.id, {RECOVERY_ATTEMPT_RUNNER_LABEL: previous_runner_id})
        except ConversationNotFoundError:
            continue


async def prepare_recovery_bindings(
    root: Conversation,
    runner_id: str,
    interrupted: Sequence[Conversation],
    store: ConversationStore,
) -> bool:
    """Finish in-flight store writes before undoing a cancelled/failed claim."""
    task = asyncio.create_task(_prepare_recovery_bindings(root, runner_id, interrupted, store))
    try:
        return await asyncio.shield(task)
    except (Exception, asyncio.CancelledError):
        await asyncio.gather(task, return_exceptions=True)
        if root.runner_id is not None:
            await rollback_recovery_bindings(runner_id, root.runner_id, store)
        raise


async def _prepare_recovery_bindings(
    root: Conversation,
    runner_id: str,
    interrupted: Sequence[Conversation],
    store: ConversationStore,
) -> bool:
    """Claim interrupted work and its ancestor chain under the host-launch lock.

    Lifecycle labels and bindings can live in separate databases. Recheck after
    claiming and undo stale claims; initialization is the final lifecycle gate.
    """
    previous_runner_id = root.runner_id
    if previous_runner_id is None or not can_restore_session(root):
        return False
    plan: dict[str, Conversation] = {root.id: root}
    resume_ids: set[str] = set()
    for snapshot in interrupted:
        fresh = await asyncio.to_thread(store.get_conversation, snapshot.id)
        if (
            fresh is None
            or fresh.runner_id != previous_runner_id
            or not _same_recovery_location(fresh, snapshot)
            or not was_interrupted(fresh, reconciled=True)
        ):
            continue
        chain = [fresh]
        seen = {fresh.id}
        while chain[-1].id != root.id:
            parent_id = chain[-1].parent_conversation_id
            parent = (
                await asyncio.to_thread(store.get_conversation, parent_id) if parent_id else None
            )
            if (
                parent is None
                or parent.id in seen
                or parent.runner_id != previous_runner_id
                or not can_restore_session(parent)
                or (parent.id != root.id and parent.host_id is not None)
                or (parent.id == root.id and not _same_recovery_location(parent, root))
            ):
                break
            seen.add(parent.id)
            chain.append(parent)
        else:
            for ancestor in reversed(chain):
                plan.setdefault(ancestor.id, ancestor)
            resume_ids.add(fresh.id)
    if not resume_ids:
        return False

    claimed: set[str] = set()
    for snapshot in plan.values():
        fresh = await asyncio.to_thread(store.get_conversation, snapshot.id)
        if (
            fresh is None
            or fresh.runner_id != previous_runner_id
            or not _same_recovery_location(fresh, snapshot)
            or not can_restore_session(fresh)
            or (fresh.id in resume_ids and not was_interrupted(fresh, reconciled=True))
            or (fresh.id != root.id and fresh.parent_conversation_id not in claimed)
        ):
            continue
        mode = (
            "parent"
            if is_parent_owned_subagent(fresh)
            else ("resume" if fresh.id in resume_ids else "restore")
        )
        try:
            # Connect must see the mode with the new binding. Lost/undone
            # claims leave it inert because consumers match the runner id.
            await asyncio.to_thread(
                store.set_labels, fresh.id, {RECOVERY_MODE_LABEL: f"{runner_id}:{mode}"}
            )
            rebound = await asyncio.to_thread(
                store.replace_runner_id, fresh.id, runner_id, expected_runner_id=previous_runner_id
            )
        except ConversationNotFoundError:
            continue
        if rebound.runner_id == runner_id:
            claimed.add(fresh.id)

    # A Stop, completion, or user rebind can land during any claim above.
    valid: set[str] = set()
    for snapshot in plan.values():
        if snapshot.id not in claimed:
            continue
        fresh = await asyncio.to_thread(store.get_conversation, snapshot.id)
        if (
            fresh is not None
            and fresh.runner_id == runner_id
            and _same_recovery_location(fresh, snapshot)
            and can_restore_session(fresh)
            and (fresh.id not in resume_ids or was_interrupted(fresh, reconciled=True))
            and (fresh.id == root.id or fresh.parent_conversation_id in valid)
        ):
            valid.add(fresh.id)
        else:
            with suppress(ConversationNotFoundError):
                await asyncio.to_thread(
                    store.replace_runner_id,
                    snapshot.id,
                    previous_runner_id,
                    expected_runner_id=runner_id,
                )
    fresh_root = await asyncio.to_thread(store.get_conversation, root.id)
    if (
        not (resume_ids & valid)
        or fresh_root is None
        or fresh_root.runner_id != runner_id
        or not _same_recovery_location(fresh_root, root)
        or not can_restore_session(fresh_root)
    ):
        await rollback_recovery_bindings(runner_id, previous_runner_id, store)
        return False
    await asyncio.to_thread(
        store.set_labels,
        root.id,
        {
            RECOVERY_ATTEMPT_LABEL: str(time.time()),
            RECOVERY_ATTEMPT_RUNNER_LABEL: runner_id,
        },
    )
    return True


async def reconcile_recovery_launch(
    root: Conversation,
    replacement_runner_id: str,
    store: ConversationStore,
    hosts: HostRegistry,
) -> bool:
    """Reap a confirmed launch whose root moved while the host was spawning."""
    from omnigent.server.routes._sessions.helpers import _spawn_superseded_runner_stop

    fresh = await asyncio.to_thread(store.get_conversation, root.id)
    if (
        fresh is not None
        and fresh.runner_id == replacement_runner_id
        and _same_recovery_location(fresh, root)
        and can_restore_session(fresh)
    ):
        return True
    assert root.runner_id is not None and root.host_id is not None
    await rollback_recovery_bindings(replacement_runner_id, root.runner_id, store)
    _spawn_superseded_runner_stop(root.id, root.host_id, replacement_runner_id, hosts)
    return False


class HostRunnerRecovery:
    """Schedule one recovery attempt without blocking the host frame reader."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        host_registry: HostRegistry,
    ) -> None:
        self._closed = False
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
        if self._closed or shutdown_state.server_shutting_down() or key in self._tasks:
            return
        roots = [c for c in affected if c.host_id is not None and can_restore_session(c)]
        if len(roots) != 1:
            log = _logger.warning if len(roots) > 1 else _logger.debug
            log(
                "Skipping recovery for runner %s: expected one host root, found %d",
                runner_id,
                len(roots),
            )
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
            attempt_runner = fresh.labels.get(RECOVERY_ATTEMPT_RUNNER_LABEL)
            if attempt_runner == runner_id or (
                attempt_runner is None
                and fresh.labels.get(RECOVERY_MODE_LABEL, "").startswith(f"{runner_id}:")
            ):
                try:
                    last_attempt = float(fresh.labels.get(RECOVERY_ATTEMPT_LABEL, "0"))
                except ValueError:
                    last_attempt = float("nan")
                now = time.time()
                if not math.isfinite(last_attempt) or last_attempt > now:
                    _logger.warning("Ignoring invalid recovery timestamp for %s", root.id)
                    last_attempt = 0
                if now - last_attempt < RECOVERY_COOLDOWN_S:
                    return
            attempt = await _launch_runner_on_host(
                fresh, self._store, self._hosts, host, recovery_sessions=interrupted
            )
            if attempt.pending_launch is not None:
                request_id, result_future = attempt.pending_launch
                try:
                    result = await asyncio.wait_for(result_future, RECOVERY_RESULT_GRACE_S)
                    if result.get("status") == "failed":
                        await rollback_recovery_bindings(attempt.runner_id, runner_id, self._store)
                        attempt.error_code = result.get("error_code") or "runner_launch_failed"
                        attempt.error = result.get("error")
                    elif not await reconcile_recovery_launch(
                        fresh, attempt.runner_id, self._store, self._hosts
                    ):
                        attempt.error_code = "recovery_superseded"
                except asyncio.TimeoutError:
                    _logger.warning(
                        "Recovery launch outcome unknown for %s; retaining bindings; "
                        "use explicit Retry if the session remains disconnected",
                        root.id,
                    )
                finally:
                    # Timeout/shutdown does not prove launch failed. A runner
                    # already started by the host must retain its bindings.
                    host.pending_launches.pop(request_id, None)
                    attempt.pending_launch = None
                    if not result_future.done():
                        result_future.cancel()
            if attempt.error_code is not None:
                _logger.warning(
                    "Automatic runner recovery refused for %s: %s; "
                    "use explicit Retry if the session remains disconnected",
                    root.id,
                    attempt.error_code,
                )
        except Exception:
            _logger.exception("Automatic runner recovery failed for %s", root.id)

    async def shutdown(self) -> None:
        """Cancel attempts before tearing down the server's transports."""
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
