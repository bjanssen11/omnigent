"""Revocation ordering against real database transactions (including the PG lane)."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest

from omnigent.db.account_authority import account_authority_scope
from omnigent.errors import OmnigentError
from omnigent.server import accounts_store as accounts_module
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.server.device_grant_store import DeviceGrantStore
from omnigent.stores.host_store import HostStore
from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore
from omnigent.stores.scheduled_task_store.sqlalchemy_store import SqlAlchemyScheduledTaskStore


def test_account_generation_survives_delete_and_rejects_old_writes(db_uri: str) -> None:
    accounts = SqlAlchemyAccountStore(db_uri)
    permissions = SqlAlchemyPermissionStore(db_uri)
    old = accounts.create_user_with_password("alice", "test-password-hash")
    assert old.account_generation is not None
    assert accounts.delete_user("alice") is True
    assert accounts.get_user("alice") is None
    with pytest.raises(OmnigentError, match="revoked"):
        permissions.ensure_user("alice")
    new = accounts.create_user_with_password("alice", "replacement-password-hash")
    assert new.account_generation and new.account_generation != old.account_generation
    with account_authority_scope("alice", old.account_generation):
        for write in (
            lambda: permissions.ensure_user("alice"),
            lambda: permissions.grant("alice", uuid.uuid4().hex, 3),
            lambda: HostStore(db_uri).upsert_on_connect(uuid.uuid4().hex, "laptop", "alice"),
            lambda: DeviceGrantStore(db_uri).create_redeemed_grant(
                "late-grant",
                user_id="alice",
                client_id="omnigent-cli",
                refresh_token_hash="hash",
                created_at=int(time.time()),
            ),
            lambda: SqlAlchemyScheduledTaskStore(db_uri).create(
                uuid.uuid4().hex, "job", "hello", "FREQ=DAILY", "alice", uuid.uuid4().hex, "UTC"
            ),
        ):
            with pytest.raises(OmnigentError, match="revoked"):
                write()
    with account_authority_scope("alice", new.account_generation):
        permissions.ensure_user("alice")
        assert HostStore(db_uri).upsert_on_connect(uuid.uuid4().hex, "laptop", "alice")


def test_host_replacement_serializes_with_account_deletion(
    db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    accounts = SqlAlchemyAccountStore(db_uri)
    hosts = HostStore(db_uri)
    accounts.create_user_with_password("alice", "test-password-hash")
    host_id = uuid.uuid4().hex
    host = hosts.register_managed_host(
        host_id=host_id,
        name="managed",
        user_id="alice",
        token="old-token",
        provider="modal",
        sandbox_id="old-sandbox",
        token_expires_at=int(time.time()) + 3600,
    )
    assert hosts.detach_stale_managed_sandbox(
        host_id, sandbox_id="old-sandbox", expected_updated_at=host.updated_at
    )
    assert hosts.mark_sandbox_terminated(host_id, sandbox_id="old-sandbox")
    reached, resume, replacement_started = Event(), Event(), Event()
    cleanup = accounts_module._revoke_durable_authority

    def pause_cleanup(*args, **kwargs):
        reached.set()
        assert resume.wait(10)
        return cleanup(*args, **kwargs)

    def replace():
        replacement_started.set()
        return hosts.replace_managed_host_sandbox(
            host_id=host_id,
            user_id="alice",
            token="replacement",
            provider="modal",
            sandbox_id="new-sandbox",
            token_expires_at=int(time.time()) + 3600,
        )

    monkeypatch.setattr(accounts_module, "_revoke_durable_authority", pause_cleanup)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(accounts.delete_user, "alice")
        try:
            assert reached.wait(10)
            replacing = pool.submit(replace)
            assert replacement_started.wait(10)
            with pytest.raises(TimeoutError):
                replacing.result(timeout=0.1)
        finally:
            resume.set()
        assert deleting.result(timeout=10) is True
        with pytest.raises(OmnigentError, match="revoked"):
            replacing.result(timeout=10)
    assert hosts.get_host(host_id) is None
    assert hosts.resolve_launch_token(host_id, "replacement") is None


def test_deletion_retains_sandbox_cleanup_and_invalidates_magic_links(db_uri: str) -> None:
    accounts = SqlAlchemyAccountStore(db_uri)
    hosts = HostStore(db_uri)
    accounts.create_user_with_password("alice", "test-password-hash")
    host_id = uuid.uuid4().hex
    hosts.register_managed_host(
        host_id=host_id,
        name="managed",
        user_id="alice",
        token="token",
        provider="modal",
        sandbox_id="sandbox",
        token_expires_at=int(time.time()) + 3600,
    )
    accounts.create_token(
        "magic",
        kind="magic",
        user_id="alice",
        created_by=None,
        created_at=0,
        expires_at=2000000000,
    )
    assert accounts.delete_user("alice") is True
    assert hosts.get_host(host_id) is None
    assert hosts.resolve_launch_token(host_id, "token") is None
    assert hosts.mark_sandbox_terminated(host_id, sandbox_id="sandbox")
    accounts.create_user_with_password("alice", "new-password-hash")
    assert accounts.redeem_token("magic", kind="magic", now_epoch_seconds=1) is None


def test_username_reuse_does_not_inherit_connections_or_projects(db_uri: str) -> None:
    from omnigent.stores.project_store.sqlalchemy_store import SqlAlchemyProjectStore
    from tests.server.test_credential_store import _store

    accounts = SqlAlchemyAccountStore(db_uri)
    projects = SqlAlchemyProjectStore(db_uri)
    credentials = _store(db_uri)
    old = accounts.create_user_with_password("alice", "test-password-hash")
    project = projects.create(uuid.uuid4().hex, "private", "alice")
    projects.save_order([project.id], user_id="alice")
    credentials.upsert("alice", "github", secret={"access_token": "local-fake-token"}, metadata={})
    assert accounts.delete_user("alice")
    accounts.create_user_with_password("alice", "replacement-password")
    assert projects.list(user_id="alice") == []
    assert projects.get_order(user_id="alice") is None
    assert credentials.get("alice", "github", with_secret=True) is None
    with account_authority_scope("alice", old.account_generation):
        with pytest.raises(OmnigentError, match="revoked"):
            credentials.upsert(
                "alice", "github", secret={"access_token": "late-token"}, metadata={}
            )
        with pytest.raises(OmnigentError, match="revoked"):
            projects.create(uuid.uuid4().hex, "late-project", "alice")


def test_cleanup_failure_rolls_back_revocation(
    db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omnigent.stores import host_store as hosts_module

    accounts = SqlAlchemyAccountStore(db_uri)
    permissions = SqlAlchemyPermissionStore(db_uri)
    hosts = HostStore(db_uri)
    account = accounts.create_user_with_password("alice", "test-password-hash")
    session_id, host_id = uuid.uuid4().hex, uuid.uuid4().hex
    permissions.grant("alice", session_id, 3)
    hosts.upsert_on_connect(host_id, "laptop", "alice")

    def fail_cleanup(*args):
        raise RuntimeError("test cleanup failure")

    monkeypatch.setattr(hosts_module, "delete_host_in_session", fail_cleanup)
    with pytest.raises(RuntimeError, match="test cleanup failure"):
        accounts.delete_user("alice")
    assert accounts.get_user("alice") == account
    assert permissions.get("alice", session_id) is not None
    assert hosts.get_host(host_id) is not None


def test_unbound_saved_authority_cannot_acquire_new_account(db_uri: str) -> None:
    accounts = SqlAlchemyAccountStore(db_uri)
    permissions = SqlAlchemyPermissionStore(db_uri)
    accounts.create_user_with_password("alice", "test-password-hash")
    with account_authority_scope("alice", None):
        with pytest.raises(OmnigentError, match="revoked"):
            permissions.ensure_user("alice")


def test_host_identity_changes_preserve_account_generation(db_uri: str) -> None:
    accounts = SqlAlchemyAccountStore(db_uri)
    hosts = HostStore(db_uri)
    account = accounts.create_user_with_password("alice", "test-password-hash")
    original_id, rotated_id = uuid.uuid4().hex, uuid.uuid4().hex
    hosts.upsert_on_connect(original_id, "laptop", "local")
    with account_authority_scope("alice", account.account_generation):
        claimed = hosts.upsert_on_connect(original_id, "laptop", "alice", allow_host_id_reown=True)
        assert claimed.account_generation == account.account_generation
        rotated = hosts.upsert_on_connect(rotated_id, "laptop", "alice")
    assert rotated.account_generation == account.account_generation
    assert hosts.get_host(rotated_id).account_generation == account.account_generation


def test_runner_issuance_serializes_with_deletion(db_uri: str) -> None:
    from omnigent.db.account_authority import account_generation
    from omnigent.server.auth import LEVEL_OWNER
    from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore

    accounts = SqlAlchemyAccountStore(db_uri)
    account = accounts.create_user_with_password("alice", "test-password-hash")
    host_id = uuid.uuid4().hex
    HostStore(db_uri).upsert_on_connect(host_id, "laptop", "alice")
    runner_id = uuid.uuid4().hex
    conv = SqlAlchemyConversationStore(db_uri).create_conversation(
        host_id=host_id, workspace="/tmp/workspace", runner_id=runner_id
    )
    SqlAlchemyPermissionStore(db_uri).grant("alice", conv.id, LEVEL_OWNER)
    reached, resume, deletion_started = Event(), Event(), Event()

    def issue(owner):
        assert owner == "alice"
        assert account_generation(owner) == account.account_generation
        reached.set()
        assert resume.wait(10)
        return "issued-under-lock"

    def delete():
        deletion_started.set()
        return accounts.delete_user("alice")

    with ThreadPoolExecutor(max_workers=2) as pool:
        issuance = pool.submit(accounts.with_runner_authority, runner_id, issue)
        try:
            assert reached.wait(10)
            deletion = pool.submit(delete)
            assert deletion_started.wait(10)
            with pytest.raises(TimeoutError):
                deletion.result(timeout=0.2)
        finally:
            resume.set()
        assert issuance.result(timeout=10) == "issued-under-lock"
        assert deletion.result(timeout=10) is True
    assert accounts.with_runner_authority(runner_id, issue) is None


def test_concurrent_admin_deletions_preserve_an_active_admin(db_uri: str) -> None:
    from threading import Barrier

    accounts = SqlAlchemyAccountStore(db_uri)
    alice = accounts.create_user_with_password("alice", "hash", is_admin=True)
    bob = accounts.create_user_with_password("bob", "hash", is_admin=True)
    barrier = Barrier(2)

    def delete(actor, target):
        with account_authority_scope(actor.id, actor.account_generation):
            barrier.wait(timeout=10)
            try:
                return accounts.delete_user(target.id)
            except OmnigentError:
                return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(delete, alice, bob)
        b = pool.submit(delete, bob, alice)
        assert sorted([a.result(timeout=10), b.result(timeout=10)]) == [False, True]
    assert sum(accounts.is_admin(user) for user in ("alice", "bob")) == 1


def test_launch_admission_requires_binding_unless_initial_bind(db_uri: str) -> None:
    from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore

    accounts = SqlAlchemyAccountStore(db_uri)
    account = accounts.create_user_with_password("alice", "test-password-hash")
    hosts = HostStore(db_uri)
    host_id = uuid.uuid4().hex
    hosts.upsert_on_connect(host_id, "laptop", "alice")
    conversations = SqlAlchemyConversationStore(db_uri)
    conv = conversations.create_conversation()
    with pytest.raises(OmnigentError, match="bound"):
        hosts.admit_launch(host_id, conv.id, "alice", account.account_generation)
    hosts.admit_launch(host_id, conv.id, "alice", account.account_generation, allow_unbound=True)
    conversations.set_host_id(conv.id, host_id, workspace="/tmp/workspace")
    hosts.admit_launch(host_id, conv.id, "alice", account.account_generation)
    assert accounts.delete_user("alice") is True
    with pytest.raises(OmnigentError, match="revoked"):
        hosts.admit_launch(
            host_id, conv.id, "alice", account.account_generation, allow_unbound=True
        )


def test_transfer_admission_checks_source_and_saved_generation(db_uri: str) -> None:
    from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore

    accounts = SqlAlchemyAccountStore(db_uri)
    account = accounts.create_user_with_password("alice", "test-password-hash")
    hosts = HostStore(db_uri)
    source, destination, other = (uuid.uuid4().hex for _ in range(3))
    for host_id in (source, destination, other):
        hosts.upsert_on_connect(host_id, host_id, "alice")
    conversations = SqlAlchemyConversationStore(db_uri)
    conv = conversations.create_conversation()
    conversations.set_host_id(conv.id, source, workspace="/tmp/source")
    with pytest.raises(OmnigentError, match="bound"):
        hosts.admit_launch(destination, conv.id, "alice", account.account_generation)
    with pytest.raises(OmnigentError, match="bound"):
        hosts.admit_launch(
            destination, conv.id, "alice", account.account_generation, allow_unbound=True
        )

    def admit_transfer():
        hosts.admit_launch(
            destination,
            conv.id,
            "alice",
            account.account_generation,
            allow_unbound=True,
            transfer_from_host_id=source,
        )

    admit_transfer()
    assert conversations.get_conversation(conv.id).host_id == source
    conversations.set_host_id(conv.id, other)
    with pytest.raises(OmnigentError, match="bound"):
        admit_transfer()
    conversations.set_host_id(conv.id, source)
    assert accounts.delete_user("alice") is True
    accounts.create_user_with_password("alice", "replacement-password-hash")
    hosts.upsert_on_connect(destination, "new-laptop", "alice")
    with pytest.raises(OmnigentError, match="revoked"):
        admit_transfer()


def test_transfer_admission_waits_for_account_deletion(db_uri, monkeypatch):
    from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore

    accounts = SqlAlchemyAccountStore(db_uri)
    account = accounts.create_user_with_password("alice", "test-password-hash")
    hosts = HostStore(db_uri)
    source, destination = uuid.uuid4().hex, uuid.uuid4().hex
    for host_id in (source, destination):
        hosts.upsert_on_connect(host_id, host_id, "alice")
    conversations = SqlAlchemyConversationStore(db_uri)
    conv = conversations.create_conversation()
    conversations.set_host_id(conv.id, source, workspace="/tmp/source")
    reached, resume, admission_started = Event(), Event(), Event()
    cleanup = accounts_module._revoke_durable_authority

    def pause_cleanup(*args, **kwargs):
        reached.set()
        assert resume.wait(10)
        return cleanup(*args, **kwargs)

    def admit_transfer():
        admission_started.set()
        hosts.admit_launch(
            destination,
            conv.id,
            "alice",
            account.account_generation,
            transfer_from_host_id=source,
        )

    monkeypatch.setattr(accounts_module, "_revoke_durable_authority", pause_cleanup)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(accounts.delete_user, "alice")
        try:
            assert reached.wait(10)
            admitting = pool.submit(admit_transfer)
            assert admission_started.wait(10)
            with pytest.raises(TimeoutError):
                admitting.result(timeout=0.1)
        finally:
            resume.set()
        assert deleting.result(timeout=10) is True
        with pytest.raises(OmnigentError, match="revoked"):
            admitting.result(timeout=10)
    assert conversations.get_conversation(conv.id).runner_id is None


@pytest.mark.parametrize("revoked_user", ["actor", "target"])
def test_target_scope_preserves_actor_and_target_checks(db_uri, revoked_user):
    from omnigent.db.account_authority import target_account_scope

    accounts = SqlAlchemyAccountStore(db_uri)
    permissions = SqlAlchemyPermissionStore(db_uri)
    actor = accounts.create_user_with_password("actor", "actor-hash")
    target = accounts.create_user_with_password("target", "target-hash")
    with account_authority_scope(actor.id, actor.account_generation):
        with target_account_scope(target.id, target.account_generation):
            permissions.ensure_user(target.id)
            assert accounts.delete_user(revoked_user) is True
            accounts.create_user_with_password(revoked_user, "replacement-hash")
            with pytest.raises(OmnigentError, match="revoked"):
                permissions.grant(target.id, uuid.uuid4().hex, 1)
    with account_authority_scope("actor", accounts.get_user("actor").account_generation):
        with target_account_scope("target", accounts.get_user("target").account_generation):
            permissions.ensure_user("target")


@pytest.mark.parametrize("existing_external_user", [False, True])
def test_target_without_registration_cannot_acquire_new_account(db_uri, existing_external_user):
    from omnigent.db.account_authority import target_account_scope

    accounts = SqlAlchemyAccountStore(db_uri)
    permissions = SqlAlchemyPermissionStore(db_uri)
    actor = accounts.create_user_with_password("actor", "actor-hash")
    if existing_external_user:
        permissions.ensure_user("target")
    target = permissions.get_user("target")
    with account_authority_scope(actor.id, actor.account_generation):
        with target_account_scope("target", target.account_generation if target else None):
            if existing_external_user:
                assert accounts.delete_user("target") is True
            accounts.create_user_with_password("target", "new-password-hash")
            with pytest.raises(OmnigentError, match="revoked"):
                permissions.ensure_user("target")
            with pytest.raises(OmnigentError, match="revoked"):
                permissions.grant("target", uuid.uuid4().hex, 1)
