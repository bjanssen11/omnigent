"""Account-generation checks shared by authentication and authority writes.

An authenticated request captures a generation once. Background tasks inherit
that context; durable jobs must restore their saved generation before running.
Writers lock the account before their resource rows, using an immediate
transaction on SQLite. Deletion uses the same ordering.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select
from sqlalchemy.orm import Session

from omnigent.db.db_models import SqlUser, current_workspace_id
from omnigent.errors import ErrorCode, OmnigentError


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


def lock_account(session: Session, user_id: str) -> SqlUser | None:
    """Lock before touching grants, hosts, or other owned resources."""
    return session.execute(
        select(SqlUser)
        .where(SqlUser.workspace_id == current_workspace_id(), SqlUser.id == user_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()


class _GenerationDefault(Enum):
    CONTEXT = "context"


def require_active_account(
    session: Session,
    user_id: str | None,
    *,
    generation: str | None | _GenerationDefault = _GenerationDefault.CONTEXT,
) -> str | None:
    """Validate the captured generation under the writer's account lock.

    Identities without an accounts row keep the header/OIDC/machine behavior.
    An accounts request cannot create a missing row or cross a re-registration.
    """
    authority = _authority.get()
    checks: dict[str, list[str | None]] = {}
    if authority is not None and authority.workspace_id == current_workspace_id():
        checks[authority.user_id] = [authority.generation]
    for target in _targets.get():
        if target.workspace_id == current_workspace_id():
            checks.setdefault(target.user_id, []).append(target.generation)
    if user_id is not None:
        checks.setdefault(user_id, [])
        if isinstance(generation, str) or generation is None:
            checks[user_id].append(generation)
    current_generation = None
    for checked_id in sorted(checks):
        row = lock_account(session, checked_id)
        if (row is not None and row.deleted_at is not None) or any(
            (row.account_generation if row else None) != expected
            for expected in checks[checked_id]
        ):
            raise OmnigentError("account authority has been revoked", code=ErrorCode.UNAUTHORIZED)
        if checked_id == user_id and row is not None:
            current_generation = row.account_generation
    return current_generation
