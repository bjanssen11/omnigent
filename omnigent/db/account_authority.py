"""Account registration identities carried through requests and durable work."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from omnigent.db.db_models import current_workspace_id


@dataclass(frozen=True)
class AccountAuthority:
    user_id: str
    generation: str | None
    workspace_id: int


_authority: ContextVar[AccountAuthority | None] = ContextVar("account_authority", default=None)
_targets: ContextVar[tuple[AccountAuthority, ...]] = ContextVar("account_targets", default=())


def bind_account_authority(user_id: str, generation: str) -> None:
    """Capture the identity authenticated by this request or durable job."""
    _authority.set(AccountAuthority(user_id, generation, current_workspace_id()))


def clear_account_authority() -> None:
    """Start authentication without borrowing an earlier identity."""
    _authority.set(None)
    _targets.set(())


def current_account_user() -> str | None:
    authority = _authority.get()
    if authority and authority.workspace_id == current_workspace_id():
        return authority.user_id
    return None


def account_generation(user_id: str) -> str | None:
    authority = _authority.get()
    if (
        authority
        and authority.workspace_id == current_workspace_id()
        and authority.user_id == user_id
    ):
        return authority.generation
    return None


@contextmanager
def account_authority_scope(user_id: str | None, generation: str | None) -> Iterator[None]:
    authority = (
        AccountAuthority(user_id, generation, current_workspace_id())
        if user_id is not None
        else None
    )
    token = _authority.set(authority)
    try:
        yield
    finally:
        _authority.reset(token)


@contextmanager
def target_account_scope(user_id: str, generation: str | None) -> Iterator[None]:
    """Pin a separately looked-up target without replacing the acting identity."""
    target = AccountAuthority(user_id, generation, current_workspace_id())
    token = _targets.set((*_targets.get(), target))
    try:
        yield
    finally:
        _targets.reset(token)
