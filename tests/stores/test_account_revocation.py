"""Account registration identity across host persistence and ID changes."""

import uuid

from omnigent.db.account_authority import account_authority_scope
from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.stores.host_store import HostStore


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
