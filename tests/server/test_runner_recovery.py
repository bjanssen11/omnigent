"""Recovery stays with the original host and restores only interrupted work."""

from __future__ import annotations

import asyncio
import copy
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from omnigent.entities import Conversation
from omnigent.server import runner_recovery as recovery
from omnigent.server.routes import sessions
from omnigent.server.routes._sessions import common, helpers


def _conv(session_id: str, **kwargs: object) -> Conversation:
    return Conversation(
        id=session_id,
        created_at=1,
        updated_at=1,
        root_conversation_id="parent",
        agent_id="agent",
        runner_id="old",
        live_status="running",
        **kwargs,
    )


class Store:
    def __init__(self, *rows: Conversation) -> None:
        self.rows = {c.id: copy.deepcopy(c) for c in rows}

    def get_conversation(self, session_id: str) -> Conversation | None:
        return copy.deepcopy(self.rows.get(session_id))

    def list_conversations_by_runner_id(self, runner_id: str) -> list[Conversation]:
        return [copy.deepcopy(c) for c in self.rows.values() if c.runner_id == runner_id]

    def set_labels(self, session_id: str, labels: dict[str, str]) -> None:
        self.rows[session_id].labels.update(labels)

    def replace_runner_id(
        self, session_id: str, runner_id: str, *, expected_runner_id: str | None = None
    ) -> Conversation:
        row = self.rows[session_id]
        if expected_runner_id is None or row.runner_id == expected_runner_id:
            row.runner_id = runner_id
        return copy.deepcopy(row)


@pytest.fixture
def group(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(common, "_session_status_cache", {})
    monkeypatch.setattr(common, "_intentional_stop_sessions", set())
    monkeypatch.setattr(common, "_interrupt_fenced_sessions", set())
    monkeypatch.setattr(recovery.shutdown_state, "server_shutting_down", lambda: False)
    parent = _conv("parent", host_id="original-host", workspace="/original/worktree")
    child = _conv("child", kind="sub_agent", parent_conversation_id=parent.id)
    return parent, child, Store(parent, child)


@pytest.mark.asyncio
@pytest.mark.parametrize("host_status", ["dead", "alive", "unknown", None])
async def test_only_confirmed_death_reuses_original_host(group, monkeypatch, host_status):
    parent, child, store = group
    host = SimpleNamespace(host_id="original-host")
    hosts = SimpleNamespace(get=lambda host_id: host if host_id == parent.host_id else None)
    monkeypatch.setattr(helpers, "_query_host_runner_status", AsyncMock(return_value=host_status))
    launch = AsyncMock(return_value=SimpleNamespace(error_code=None))
    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    coordinator = recovery.HostRunnerRecovery(store, hosts)
    coordinator.schedule("old", [parent, child], [parent, child])
    coordinator.schedule("old", [parent, child], [parent, child])
    await asyncio.gather(*coordinator._tasks.values())
    if host_status == "dead":
        launch.assert_awaited_once()
        args, kwargs = launch.call_args
        assert args[0].workspace == "/original/worktree"
        assert args[3] is host
        assert [c.id for c in kwargs["recovery_sessions"]] == ["parent", "child"]
    else:
        launch.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["stopped", "closed", "rebound", "cooldown", "offline"])
async def test_recovery_rechecks_lifecycle_before_launch(group, monkeypatch, change):
    parent, child, store = group
    host = object()
    hosts = SimpleNamespace(get=lambda _: None if change == "offline" else host)

    async def status(*_):
        fresh = store.rows[parent.id]
        if change == "stopped":
            fresh.labels[recovery.RECOVERY_STOPPED_LABEL] = "true"
        elif change == "closed":
            fresh.archived = True
        elif change == "rebound":
            fresh.runner_id = "user-replacement"
        elif change == "cooldown":
            fresh.labels[recovery.RECOVERY_ATTEMPT_LABEL] = str(time.time())
            fresh.labels[recovery.RECOVERY_MODE_LABEL] = "old:resume"
        return "dead"

    monkeypatch.setattr(helpers, "_query_host_runner_status", status)
    launch = AsyncMock()
    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    coordinator = recovery.HostRunnerRecovery(store, hosts)
    coordinator.schedule("old", [parent, child], [parent, child])
    await asyncio.gather(*coordinator._tasks.values())
    launch.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_adopt_children_without_original_host(group, monkeypatch):
    parent, child, store = group
    parent.host_id = None
    launch = AsyncMock()
    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    coordinator = recovery.HostRunnerRecovery(store, SimpleNamespace(get=lambda _: object()))
    coordinator.schedule("old", [parent, child], [parent, child])
    assert not coordinator._tasks
    launch.assert_not_awaited()


@pytest.mark.asyncio
async def test_bindings_restored_before_launch_and_completed_child_left_alone(group, monkeypatch):
    parent, child, store = group
    completed = _conv("completed", kind="sub_agent", parent_conversation_id=parent.id)
    completed.live_status = "idle"
    store.rows[completed.id] = completed
    host = SimpleNamespace(pending_launches={})
    monkeypatch.setattr(helpers, "_spawn_superseded_runner_stop", lambda *_: None)
    monkeypatch.setattr(helpers, "_resolve_harness", lambda _: "openai-agents")

    def send(_host, _frame):
        new_runner_id = store.rows[parent.id].runner_id
        assert new_runner_id != "old"
        assert store.rows[child.id].runner_id == new_runner_id
        assert store.rows[completed.id].runner_id == "old"
        for future in host.pending_launches.values():
            future.set_result({"status": "launched"})

    attempt = await helpers._launch_runner_on_host_impl(
        parent,
        store,
        SimpleNamespace(send_text=send),
        host,
        recovery_sessions=[parent, child],
    )
    assert attempt.error_code is None
    assert (
        store.rows[child.id].labels[recovery.RECOVERY_MODE_LABEL] == f"{attempt.runner_id}:resume"
    )


@pytest.mark.asyncio
async def test_idle_parent_is_restored_without_replaying_its_old_turn(group):
    parent, child, store = group
    parent.live_status = "idle"
    store.rows[parent.id].live_status = "idle"
    assert await recovery.prepare_recovery_bindings(parent, "new", [child], store)
    assert recovery.recovery_suppresses_turn(store.rows[parent.id])
    assert not recovery.recovery_suppresses_turn(store.rows[child.id])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "labels",
    [
        {"omnigent.wrapper": "claude-code-native-ui-subagent"},
        {"omnigent.wrapper": "codex-native-ui-subagent"},
        {"omnigent.wrapper": "opencode-native-ui-subagent"},
        {"omnigent.wrapper": "antigravity-native-ui-subagent"},
        {"omnigent.acp.subagent_id": "vendor-child"},
    ],
)
async def test_mirrored_child_recovers_through_parent_without_standalone_init(group, labels):
    parent, child, store = group
    store.rows[parent.id].live_status = parent.live_status = "idle"
    child.labels.update(labels)
    store.rows[child.id].labels.update(labels)

    assert await recovery.prepare_recovery_bindings(parent, "new", [child], store)
    restored_parent, restored_child = store.rows[parent.id], store.rows[child.id]
    assert restored_parent.runner_id == restored_child.runner_id == "new"
    assert recovery.recovery_suppresses_turn(restored_parent)
    assert restored_child.labels[recovery.RECOVERY_MODE_LABEL] == "new:parent"
    assert await recovery.may_initialize_session(restored_parent, store)
    assert not await recovery.may_initialize_session(restored_child, store)


@pytest.mark.asyncio
async def test_independent_native_child_still_initializes_after_recovery(group):
    parent, child, store = group
    child.labels["omnigent.wrapper"] = "claude-code-native-ui"
    store.rows[child.id].labels.update(child.labels)
    assert await recovery.prepare_recovery_bindings(parent, "new", [child], store)
    assert store.rows[child.id].labels[recovery.RECOVERY_MODE_LABEL] == "new:resume"
    assert await recovery.may_initialize_session(store.rows[child.id], store)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["stopped", "closed", "rebound", "finished", "moved_host"])
async def test_child_changed_while_recovery_waited_is_not_restarted(group, change):
    parent, child, store = group
    fresh = store.rows[child.id]
    if change == "stopped":
        fresh.labels[recovery.RECOVERY_STOPPED_LABEL] = "true"
    elif change == "closed":
        fresh.labels["omnigent.closed"] = "true"
    elif change == "rebound":
        fresh.runner_id = "user-replacement"
    elif change == "moved_host":
        fresh.host_id = "user-selected-host"
    else:
        fresh.live_status = "idle"
    assert await recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    assert store.rows[child.id].runner_id != "new"


def test_interruption_evidence_excludes_completed_failed_and_cancelled(group):
    parent, _, _ = group
    assert recovery.was_interrupted(parent)
    parent.live_status = "idle"
    assert not recovery.was_interrupted(parent)
    parent.live_status = "failed"
    assert not recovery.was_interrupted(parent)
    common._interrupt_fenced_sessions.add(parent.id)
    parent.live_status = "running"
    assert not recovery.was_interrupted(parent)


@pytest.mark.asyncio
async def test_stop_after_group_rebind_prevents_child_initialization(group):
    parent, child, store = group
    assert await recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    store.rows[parent.id].labels[recovery.RECOVERY_STOPPED_LABEL] = "true"
    assert not await recovery.may_initialize_session(store.rows[child.id], store)


@pytest.mark.asyncio
async def test_manual_child_rebind_during_recovery_is_preserved(group):
    parent, child, store = group
    original_replace = store.replace_runner_id

    def racing_replace(session_id, runner_id, **kwargs):
        if session_id == child.id:
            store.rows[child.id].runner_id = "user-replacement"
        return original_replace(session_id, runner_id, **kwargs)

    store.replace_runner_id = racing_replace
    assert await recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    assert store.rows[child.id].runner_id == "user-replacement"


@pytest.mark.asyncio
async def test_deleted_child_does_not_abort_parent_recovery(group):
    parent, child, store = group
    original_replace = store.replace_runner_id

    def racing_replace(session_id, runner_id, **kwargs):
        if session_id == child.id:
            raise recovery.ConversationNotFoundError(session_id)
        return original_replace(session_id, runner_id, **kwargs)

    store.replace_runner_id = racing_replace
    assert await recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    assert store.rows[parent.id].runner_id == "new"


@pytest.mark.asyncio
async def test_lost_root_cas_does_not_cool_down_user_selected_runner(group, monkeypatch):
    parent, child, store = group
    original_replace = store.replace_runner_id

    def racing_replace(session_id, runner_id, **kwargs):
        if session_id == parent.id:
            store.rows[parent.id].runner_id = "user-replacement"
        return original_replace(session_id, runner_id, **kwargs)

    store.replace_runner_id = racing_replace
    assert not await recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    assert recovery.RECOVERY_ATTEMPT_LABEL not in store.rows[parent.id].labels
    assert store.rows[child.id].runner_id == "old"

    # Even a previous attempt's timestamp must not cool down a user's new binding.
    store.rows[parent.id].labels[recovery.RECOVERY_ATTEMPT_LABEL] = str(time.time())
    fresh = store.rows[parent.id]
    monkeypatch.setattr(helpers, "_query_host_runner_status", AsyncMock(return_value="dead"))
    launch = AsyncMock(return_value=SimpleNamespace(error_code=None))
    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    coordinator = recovery.HostRunnerRecovery(store, SimpleNamespace(get=lambda _: object()))
    coordinator.schedule("user-replacement", [fresh], [fresh])
    await asyncio.gather(*coordinator._tasks.values())
    launch.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["stopped", "archived", "closed"])
async def test_lifecycle_change_during_root_cas_undoes_claims(group, lifecycle):
    parent, child, store = group
    original_replace = store.replace_runner_id

    def racing_replace(session_id, runner_id, **kwargs):
        if session_id == parent.id:
            if lifecycle == "archived":
                store.rows[parent.id].archived = True
            else:
                label = (
                    recovery.RECOVERY_STOPPED_LABEL
                    if lifecycle == "stopped"
                    else "omnigent.closed"
                )
                store.rows[parent.id].labels[label] = "true"
        return original_replace(session_id, runner_id, **kwargs)

    store.replace_runner_id = racing_replace
    assert not await recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    for session_id in [parent.id, child.id]:
        assert store.rows[session_id].runner_id == "old"


@pytest.mark.asyncio
async def test_launch_send_failure_rolls_back_bindings_and_logs_retry(group, monkeypatch, caplog):
    parent, child, store = group
    host = SimpleNamespace(pending_launches={})

    def send(*_):
        raise ConnectionError("host disconnected")

    hosts = SimpleNamespace(get=lambda _: host, send_text=send)
    monkeypatch.setattr(helpers, "_query_host_runner_status", AsyncMock(return_value="dead"))
    monkeypatch.setattr(helpers, "_spawn_superseded_runner_stop", lambda *_: None)
    monkeypatch.setattr(helpers, "_resolve_harness", lambda _: "openai-agents")
    launch = AsyncMock(wraps=helpers._launch_runner_on_host_impl)
    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    coordinator = recovery.HostRunnerRecovery(store, hosts)
    coordinator.schedule("old", [parent, child], [parent, child])
    await asyncio.gather(*coordinator._tasks.values())
    launch.assert_awaited_once()
    assert store.rows[parent.id].runner_id == store.rows[child.id].runner_id == "old"
    assert not host.pending_launches
    assert "host_disconnected" in caplog.text
    assert "explicit Retry" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_closes_scheduling_before_draining_tasks(group, monkeypatch):
    parent, child, store = group
    coordinator = recovery.HostRunnerRecovery(store, SimpleNamespace(get=lambda _: object()))
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def recover(*_):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    monkeypatch.setattr(coordinator, "_recover", recover)
    coordinator.schedule("old", [parent, child], [parent, child])
    await started.wait()
    drain = asyncio.create_task(coordinator.shutdown())
    await cancelled.wait()
    coordinator.schedule("new", [parent, child], [parent, child])
    assert len(coordinator._tasks) == 1
    release.set()
    await drain
    coordinator.schedule("new", [parent, child], [parent, child])
    assert not coordinator._tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp", ["bad", "nan", "inf", "99999999999"])
async def test_invalid_cooldown_does_not_strand_recovery(group, monkeypatch, caplog, timestamp):
    parent, child, store = group
    store.rows[parent.id].labels.update(
        {
            recovery.RECOVERY_MODE_LABEL: "old:resume",
            recovery.RECOVERY_ATTEMPT_LABEL: timestamp,
        }
    )
    monkeypatch.setattr(helpers, "_query_host_runner_status", AsyncMock(return_value="dead"))
    launch = AsyncMock(return_value=SimpleNamespace(error_code=None))
    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    coordinator = recovery.HostRunnerRecovery(store, SimpleNamespace(get=lambda _: object()))
    coordinator.schedule("old", [parent, child], [parent, child])
    await asyncio.gather(*coordinator._tasks.values())
    launch.assert_awaited_once()
    assert "Ignoring invalid recovery timestamp" in caplog.text


@pytest.mark.asyncio
async def test_cancelled_launch_removes_pending_request(group, monkeypatch):
    parent, child, store = group
    host = SimpleNamespace(pending_launches={})
    sent = asyncio.Event()
    monkeypatch.setattr(helpers, "_spawn_superseded_runner_stop", lambda *_: None)
    monkeypatch.setattr(helpers, "_resolve_harness", lambda _: "openai-agents")
    launch = asyncio.create_task(
        helpers._launch_runner_on_host_impl(
            parent,
            store,
            SimpleNamespace(send_text=lambda *_: sent.set()),
            host,
            recovery_sessions=[parent, child],
        )
    )
    await sent.wait()
    assert len(host.pending_launches) == 1
    launch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await launch
    assert not host.pending_launches


@pytest.mark.asyncio
@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("change", ["stopped", "closed", "archived", "completed"])
async def test_child_lifecycle_change_during_claim_is_undone(group, mirrored, change):
    parent, child, store = group
    if mirrored:
        store.rows[child.id].labels["omnigent.wrapper"] = "claude-code-native-ui-subagent"
    original_replace = store.replace_runner_id

    def racing_replace(sid, rid, **kwargs):
        result = original_replace(sid, rid, **kwargs)
        if sid == child.id and rid == "new":
            if change == "completed":
                store.rows[sid].live_status = "idle"
            elif change == "archived":
                store.rows[sid].archived = True
            else:
                label = (
                    recovery.RECOVERY_STOPPED_LABEL if change == "stopped" else "omnigent.closed"
                )
                store.rows[sid].labels[label] = "true"
        return result

    store.replace_runner_id = racing_replace
    assert await recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    assert store.rows[parent.id].runner_id == "new"
    assert store.rows[child.id].runner_id == "old"


@pytest.mark.asyncio
@pytest.mark.parametrize("mirrored", [False, True])
async def test_nested_idle_ancestors_are_restored_without_turn_replay(group, mirrored):
    parent, child, store = group
    store.rows[parent.id].live_status = parent.live_status = "idle"
    intermediate = _conv("intermediate", kind="sub_agent", parent_conversation_id=parent.id)
    intermediate.live_status = "idle"
    store.rows[intermediate.id] = intermediate
    store.rows[child.id].parent_conversation_id = child.parent_conversation_id = intermediate.id
    if mirrored:
        for sid in [intermediate.id, child.id]:
            store.rows[sid].labels["omnigent.wrapper"] = "claude-code-native-ui-subagent"
    assert await recovery.prepare_recovery_bindings(parent, "new", [child], store)
    for sid in [parent.id, intermediate.id, child.id]:
        assert store.rows[sid].runner_id == "new"
    assert recovery.recovery_suppresses_turn(store.rows[parent.id])
    assert await recovery.may_initialize_session(store.rows[child.id], store) is not mirrored
    if mirrored:
        assert recovery.recovery_waits_for_parent(store.rows[intermediate.id])
        assert not await recovery.may_initialize_session(store.rows[intermediate.id], store)
    else:
        assert recovery.recovery_suppresses_turn(store.rows[intermediate.id])


@pytest.mark.asyncio
async def test_stopped_intermediate_blocks_its_descendants(group):
    parent, child, store = group
    intermediate = _conv("intermediate", kind="sub_agent", parent_conversation_id=parent.id)
    intermediate.labels[recovery.RECOVERY_STOPPED_LABEL] = "true"
    store.rows[intermediate.id] = intermediate
    store.rows[child.id].parent_conversation_id = child.parent_conversation_id = intermediate.id
    assert not await recovery.prepare_recovery_bindings(parent, "new", [child], store)
    assert all(c.runner_id == "old" for c in store.rows.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["send", "refused"])
async def test_failed_launch_rollback_preserves_concurrent_user_binding(
    group, monkeypatch, failure
):
    parent, child, store = group
    host = SimpleNamespace(pending_launches={})
    stop = AsyncMock()
    monkeypatch.setattr(helpers, "_spawn_superseded_runner_stop", stop)
    monkeypatch.setattr(helpers, "_resolve_harness", lambda _: "openai-agents")

    def send(*_):
        store.rows[child.id].runner_id = "user-selected"
        if failure == "send":
            raise ConnectionError("offline")
        for future in host.pending_launches.values():
            future.set_result({"status": "failed", "error_code": "unconfigured"})

    attempt = await helpers._launch_runner_on_host_impl(
        parent, store, SimpleNamespace(send_text=send), host, recovery_sessions=[parent, child]
    )
    assert attempt.error_code == ("host_disconnected" if failure == "send" else "unconfigured")
    assert store.rows[parent.id].runner_id == "old"
    assert store.rows[child.id].runner_id == "user-selected"
    stop.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_binding_preparation_drains_writes_before_rollback(group, monkeypatch):
    parent, child, store = group
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(*_):
        store.rows[parent.id].runner_id = "new"
        entered.set()
        await release.wait()
        store.rows[child.id].runner_id = "new"
        return True

    monkeypatch.setattr(recovery, "_prepare_recovery_bindings", prepare)
    task = asyncio.create_task(
        recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    )
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.rows[parent.id].runner_id == store.rows[child.id].runner_id == "old"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["completed", "ancestor_rebound", "ownership_removed"])
async def test_connect_rechecks_recovery_lifecycle_and_ownership(group, change):
    parent, child, store = group
    if change == "ownership_removed":
        store.rows[child.id].labels["omnigent.wrapper"] = "claude-code-native-ui-subagent"
    assert await recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    if change == "completed":
        store.rows[child.id].live_status = "idle"
    elif change == "ancestor_rebound":
        store.rows[parent.id].runner_id = "user-selected"
    else:
        store.rows[child.id].labels.pop("omnigent.wrapper")
    assert not await recovery.may_initialize_session(store.rows[child.id], store)


@pytest.mark.asyncio
async def test_rolled_back_attempt_cools_down_duplicate_original_exit(group, monkeypatch):
    parent, child, store = group
    host = SimpleNamespace(pending_launches={})

    def send(*_):
        raise ConnectionError("host disconnected before launch")

    hosts = SimpleNamespace(get=lambda _: host, send_text=send)
    monkeypatch.setattr(helpers, "_query_host_runner_status", AsyncMock(return_value="dead"))
    monkeypatch.setattr(helpers, "_resolve_harness", lambda _: "openai-agents")
    launch = AsyncMock(wraps=helpers._launch_runner_on_host_impl)
    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    for _ in range(2):
        coordinator = recovery.HostRunnerRecovery(store, hosts)
        fresh = store.get_conversation(parent.id)
        coordinator.schedule("old", [fresh, child], [fresh, child])
        await asyncio.gather(*coordinator._tasks.values())
    launch.assert_awaited_once()
    assert store.rows[parent.id].runner_id == "old"
    assert store.rows[parent.id].labels[recovery.RECOVERY_ATTEMPT_RUNNER_LABEL] == "old"
