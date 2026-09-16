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
    host = SimpleNamespace(
        hello=SimpleNamespace(supports_runner_recovery=True), host_id="original-host"
    )
    hosts = SimpleNamespace(get=lambda host_id: host if host_id == parent.host_id else None)
    monkeypatch.setattr(helpers, "_query_host_runner_status", AsyncMock(return_value=host_status))
    launch = AsyncMock(return_value=helpers._HostLaunchAttempt(runner_id="new"))
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
    host = SimpleNamespace(hello=SimpleNamespace(supports_runner_recovery=True))
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
    coordinator = recovery.HostRunnerRecovery(
        store,
        SimpleNamespace(
            get=lambda _: SimpleNamespace(hello=SimpleNamespace(supports_runner_recovery=True))
        ),
    )
    coordinator.schedule("old", [parent, child], [parent, child])
    assert not coordinator._tasks
    launch.assert_not_awaited()


@pytest.mark.asyncio
async def test_bindings_restored_before_launch_and_completed_child_left_alone(group, monkeypatch):
    parent, child, store = group
    completed = _conv("completed", kind="sub_agent", parent_conversation_id=parent.id)
    completed.live_status = "idle"
    store.rows[completed.id] = completed
    host = SimpleNamespace(
        hello=SimpleNamespace(supports_runner_recovery=True), pending_launches={}
    )
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
    launch = AsyncMock(return_value=helpers._HostLaunchAttempt(runner_id="new"))
    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    coordinator = recovery.HostRunnerRecovery(
        store,
        SimpleNamespace(
            get=lambda _: SimpleNamespace(hello=SimpleNamespace(supports_runner_recovery=True))
        ),
    )
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
    host = SimpleNamespace(
        hello=SimpleNamespace(supports_runner_recovery=True), pending_launches={}
    )

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
    coordinator = recovery.HostRunnerRecovery(
        store,
        SimpleNamespace(
            get=lambda _: SimpleNamespace(hello=SimpleNamespace(supports_runner_recovery=True))
        ),
    )
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
    launch = AsyncMock(return_value=helpers._HostLaunchAttempt(runner_id="new"))
    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    coordinator = recovery.HostRunnerRecovery(
        store,
        SimpleNamespace(
            get=lambda _: SimpleNamespace(hello=SimpleNamespace(supports_runner_recovery=True))
        ),
    )
    coordinator.schedule("old", [parent, child], [parent, child])
    await asyncio.gather(*coordinator._tasks.values())
    launch.assert_awaited_once()
    assert "Ignoring invalid recovery timestamp" in caplog.text


@pytest.mark.asyncio
async def test_cancelled_launch_removes_pending_request(group, monkeypatch):
    parent, child, store = group
    host = SimpleNamespace(
        hello=SimpleNamespace(supports_runner_recovery=True), pending_launches={}
    )
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
    host = SimpleNamespace(
        hello=SimpleNamespace(supports_runner_recovery=True), pending_launches={}
    )
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
        store.rows[parent.id].labels[recovery.RECOVERY_MODE_LABEL] = "new:resume"
        entered.set()
        await release.wait()
        store.rows[child.id].runner_id = "new"
        store.rows[child.id].labels[recovery.RECOVERY_MODE_LABEL] = "new:resume"
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
    host = SimpleNamespace(
        hello=SimpleNamespace(supports_runner_recovery=True), pending_launches={}
    )

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


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failed", "launched", "unknown", "shutdown"])
async def test_recovery_tracks_late_launch_result_without_holding_launch_lock(
    group, monkeypatch, outcome
):
    parent, child, store = group
    host = SimpleNamespace(
        hello=SimpleNamespace(supports_runner_recovery=True), pending_launches={}
    )
    hosts = SimpleNamespace(get=lambda _: host, send_text=lambda *_: None)
    monkeypatch.setattr(helpers, "_query_host_runner_status", AsyncMock(return_value="dead"))
    monkeypatch.setattr(helpers, "_resolve_harness", lambda _: "openai-agents")
    monkeypatch.setattr(helpers, "_HOST_LAUNCH_RESULT_TIMEOUT_S", 0.01)
    monkeypatch.setattr(recovery, "RECOVERY_RESULT_GRACE_S", 0.1)
    timed_out = asyncio.Event()

    async def launch(*args, **kwargs):
        attempt = await helpers._launch_runner_on_host_impl(*args, **kwargs)
        assert attempt.pending_launch is not None
        assert not helpers._relaunch_locks.get(parent.id, asyncio.Lock()).locked()
        timed_out.set()
        return attempt

    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    coordinator = recovery.HostRunnerRecovery(store, hosts)
    coordinator.schedule("old", [parent, child], [parent, child])
    await timed_out.wait()
    future = next(iter(host.pending_launches.values()))
    assert not future.cancelled()
    if outcome in {"failed", "launched"}:
        future.set_result(
            {"status": outcome, "error_code": "unconfigured" if outcome == "failed" else None}
        )
    if outcome == "shutdown":
        await coordinator.shutdown()
    else:
        await asyncio.gather(*coordinator._tasks.values())
    assert not host.pending_launches
    assert future.done()
    assert (store.rows[parent.id].runner_id == "old") is (outcome == "failed")
    assert store.rows[child.id].runner_id == store.rows[parent.id].runner_id


@pytest.mark.asyncio
async def test_cancelled_rollback_drains_store_writes(group):
    import threading

    parent, child, store = group
    for row in store.rows.values():
        row.runner_id = "new"
        row.labels[recovery.RECOVERY_MODE_LABEL] = "new:resume"
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original_replace = store.replace_runner_id

    def replace(sid, rid, **kwargs):
        if sid == parent.id:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(timeout=5)
        return original_replace(sid, rid, **kwargs)

    store.replace_runner_id = replace
    task = asyncio.create_task(recovery.rollback_recovery_bindings("new", "old", store))
    try:
        await entered.wait()
        task.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.rows[parent.id].runner_id == store.rows[child.id].runner_id == "old"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["harness", "encode", "send"])
async def test_launch_preparation_errors_do_not_strand_claims(group, monkeypatch, stage):
    from omnigent.host import frames

    parent, child, store = group
    host = SimpleNamespace(
        hello=SimpleNamespace(supports_runner_recovery=True), pending_launches={}
    )

    def fail(*_):
        raise RuntimeError("launch preparation failed")

    monkeypatch.setattr(
        helpers, "_resolve_harness", fail if stage == "harness" else lambda _: "openai-agents"
    )
    if stage == "encode":
        monkeypatch.setattr(frames, "encode_host_frame", fail)
    launch = helpers._launch_runner_on_host_impl(
        parent, store, SimpleNamespace(send_text=fail), host, recovery_sessions=[parent, child]
    )
    if stage == "send":
        attempt = await launch
        assert attempt.error_code == "host_launch_failed"
    else:
        with pytest.raises(RuntimeError, match="launch preparation failed"):
            await launch
    assert store.rows[parent.id].runner_id == store.rows[child.id].runner_id == "old"
    assert not host.pending_launches


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["stopped", "closed", "rebound", "fenced"])
async def test_initialization_rechecks_target_after_ancestor_read(group, change):
    parent, child, store = group
    assert await recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    original_get = store.get_conversation

    def racing_get(sid):
        if sid == parent.id:
            if change == "rebound":
                store.rows[child.id].runner_id = "user-selected"
            elif change == "fenced":
                common._interrupt_fenced_sessions.add(child.id)
            else:
                label = (
                    recovery.RECOVERY_STOPPED_LABEL if change == "stopped" else "omnigent.closed"
                )
                store.rows[child.id].labels[label] = "true"
        return original_get(sid)

    store.get_conversation = racing_get
    assert not await recovery.may_initialize_session(copy.deepcopy(store.rows[child.id]), store)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["immediate", "late", "unknown"])
async def test_recovery_reaps_launch_when_root_moves_during_send(group, monkeypatch, outcome):
    import json

    parent, child, store = group
    host = SimpleNamespace(
        hello=SimpleNamespace(supports_runner_recovery=True), pending_launches={}
    )
    replacement = None
    sent = asyncio.Event()

    def send(_, payload):
        nonlocal replacement
        frame = json.loads(payload)
        assert frame["recovery_of_runner_id"] == "old"
        replacement = store.rows[parent.id].runner_id
        store.rows[parent.id].runner_id = "user-selected"
        sent.set()
        if outcome == "immediate":
            host.pending_launches[frame["request_id"]].set_result({"status": "launched"})

    hosts = SimpleNamespace(get=lambda _: host, send_text=send)
    stop = []
    monkeypatch.setattr(helpers, "_spawn_superseded_runner_stop", lambda *args: stop.append(args))
    monkeypatch.setattr(helpers, "_query_host_runner_status", AsyncMock(return_value="dead"))
    monkeypatch.setattr(helpers, "_resolve_harness", lambda _: "openai-agents")
    monkeypatch.setattr(helpers, "_HOST_LAUNCH_RESULT_TIMEOUT_S", 0.01)
    monkeypatch.setattr(recovery, "RECOVERY_RESULT_GRACE_S", 0.05)
    monkeypatch.setattr(sessions, "_launch_runner_on_host", helpers._launch_runner_on_host_impl)
    coordinator = recovery.HostRunnerRecovery(store, hosts)
    coordinator.schedule("old", [parent, child], [parent, child])
    await sent.wait()
    if outcome == "late":
        await asyncio.sleep(0.03)
        next(iter(host.pending_launches.values())).set_result({"status": "launched"})
    await asyncio.gather(*coordinator._tasks.values())
    assert store.rows[parent.id].runner_id == "user-selected"
    assert store.rows[child.id].runner_id == "old"
    assert len(stop) == 1
    assert stop[0][2] == replacement


@pytest.mark.asyncio
async def test_late_cleanup_preserves_unclaimed_child_and_its_runner(group, monkeypatch):
    parent, child, store = group
    assert await recovery.prepare_recovery_bindings(parent, "new", [parent, child], store)
    newcomer = _conv("new-child", kind="sub_agent", parent_conversation_id=parent.id)
    newcomer.runner_id = "new"
    store.rows[newcomer.id] = newcomer
    store.rows[parent.id].runner_id = "user-selected"
    stopped = []
    monkeypatch.setattr(
        helpers, "_spawn_superseded_runner_stop", lambda *args: stopped.append(args)
    )
    assert not await recovery.reconcile_recovery_launch(parent, "new", store, SimpleNamespace())
    assert store.rows[parent.id].runner_id == "user-selected"
    assert store.rows[child.id].runner_id == "old"
    assert store.rows[newcomer.id].runner_id == "new"
    assert not stopped


@pytest.mark.asyncio
async def test_launch_rider_receives_snapshot_without_pending_result_ownership(group):
    parent, _, store = group
    store.rows[parent.id].runner_id = "new"
    future = asyncio.get_running_loop().create_future()
    original = helpers._HostLaunchAttempt(runner_id="new", pending_launch=("request", future))
    helpers._relaunch_last_attempt[parent.id] = original
    try:
        rider = await helpers._launch_runner_on_host_impl(
            parent, store, SimpleNamespace(), SimpleNamespace()
        )
        original.error_code = "late_failure"
        assert rider is not original
        assert rider.runner_id == "new"
        assert rider.error_code is None
        assert rider.pending_launch is None
        assert not future.done()
    finally:
        future.cancel()
        helpers._relaunch_last_attempt.pop(parent.id, None)


@pytest.mark.asyncio
async def test_deferred_cleanup_rechecks_bindings_before_stop(group, monkeypatch):
    from omnigent.server.runner_session_init import runner_lifecycle_lock

    parent, _, store = group
    stop = AsyncMock(return_value=True)
    monkeypatch.setattr(helpers, "_stop_session_host_runner", stop)
    async with runner_lifecycle_lock("new"):
        helpers._spawn_superseded_runner_stop(
            parent.id, parent.host_id, "new", SimpleNamespace(), store
        )
        store.rows[parent.id].runner_id = "new"
    await asyncio.gather(*helpers._detached_supersede_stops)
    stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_older_host_is_ineligible_before_recovery_claims(group, monkeypatch):
    parent, child, store = group
    host = SimpleNamespace(hello=SimpleNamespace(supports_runner_recovery=False))
    hosts = SimpleNamespace(get=lambda _: host)
    launch = AsyncMock()
    monkeypatch.setattr(sessions, "_launch_runner_on_host", launch)
    coordinator = recovery.HostRunnerRecovery(store, hosts)
    coordinator.schedule("old", [parent, child], [parent, child])
    await asyncio.gather(*coordinator._tasks.values())
    launch.assert_not_awaited()
    attempt = await helpers._launch_runner_on_host_impl(
        parent, store, hosts, host, recovery_sessions=[parent, child]
    )
    assert attempt.error_code == "host_recovery_unsupported"
    assert store.rows[parent.id].runner_id == store.rows[child.id].runner_id == "old"
    assert not store.rows[parent.id].labels


@pytest.mark.asyncio
async def test_concurrent_resume_can_reuse_already_cleared_stop(group):
    parent, _, store = group
    parent.labels[recovery.RECOVERY_STOPPED_LABEL] = "true:observed"
    store.rows[parent.id].labels[recovery.RECOVERY_STOPPED_LABEL] = ""
    store.compare_and_set_label = lambda *_: False
    await recovery.clear_observed_recovery_stop(parent, store)
    assert not recovery.recovery_is_stopped(parent)
    assert store.rows[parent.id].labels[recovery.RECOVERY_STOPPED_LABEL] == ""
