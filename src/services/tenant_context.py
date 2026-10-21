"""Which customer ("tenant") the current work is for.

Tenant isolation (backlog 1.19, D-37) puts a ``tenant_id`` label on every customer-owned row and,
as a second wall, row-level security that compares it with ``app.tenant_id`` on the database
connection. This module is the one place the application holds that value.

  * The tenant comes ONLY from the verified sign-in token's ``custom:subscription_id`` claim --
    never from a request body, header or URL.
  * A request that carries no valid claim has NO tenant. That is deliberate: once the second wall is
    on, a connection with no tenant sees zero rows, so a missing claim fails closed.
  * Background work that legitimately spans customers must say so with ``system()``. It is never
    entered from a request.
  * ``get_conn()`` passes the value to each new connection. Until the database wall is applied this
    changes nothing for any query.

The value lives in a ``ContextVar``, so it follows the request through ``await`` and into the
threadpool FastAPI runs ``def`` endpoints on, and does not leak to other requests.
"""
from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

# The application is imported both as ``services.*`` and ``src.services.*``. Two module objects would
# mean two ContextVars, and a tenant set through one name would be invisible through the other.
_ME = sys.modules[__name__]
for _alias in ("services.tenant_context", "src.services.tenant_context"):
    sys.modules.setdefault(_alias, _ME)

CLAIM = "custom:subscription_id"
_VALID = re.compile(r"[A-Za-z0-9_.-]{1,64}")

_tenant: ContextVar[Optional[str]] = ContextVar("bp_tenant", default=None)
_system: ContextVar[bool] = ContextVar("bp_system", default=False)


def clean(value: Any) -> Optional[str]:
    """The label if it is a plain short identifier, else None (never a repaired guess)."""
    if not isinstance(value, str):
        return None
    return value if _VALID.fullmatch(value) else None


def from_claims(claims: Any) -> Optional[str]:
    return clean(claims.get(CLAIM)) if isinstance(claims, dict) else None


def from_principal(principal: Any) -> Optional[str]:
    return from_claims(getattr(principal, "claims", None))


def current() -> Optional[str]:
    return _tenant.get()


def is_system() -> bool:
    return _system.get()


@contextmanager
def bound(tenant: Optional[str]) -> Iterator[None]:
    """Run with ``tenant`` as the current one (None clears it); restores the previous on exit."""
    token = _tenant.set(clean(tenant))
    try:
        yield
    finally:
        _tenant.reset(token)


@contextmanager
def system() -> Iterator[None]:
    """Run as platform work that spans customers (scheduler, listeners). Never from a request."""
    token = _system.set(True)
    try:
        yield
    finally:
        _system.reset(token)


def set_for_request(tenant: Optional[str]):
    """Set the tenant and return a token for ``reset``. For the request middleware only."""
    return _tenant.set(clean(tenant))


def reset(token) -> None:
    _tenant.reset(token)


def connection_settings() -> Optional[tuple]:
    """(tenant, system) to apply to a new connection, or None when there is nothing to say."""
    tenant, system_on = _tenant.get(), _system.get()
    if tenant is None and not system_on:
        return None
    return (tenant or "", "on" if system_on else "off")
