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

import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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


def enter_system():
    """Start running as platform work without a ``with`` block; returns a token for ``leave_system``.
    For process entry points (the web app's startup, command-line tools)."""
    return _system.set(True)


def leave_system(token) -> None:
    _system.reset(token)


def as_system(fn):
    """``fn`` wrapped so it always runs as platform work. For a shared worker that a request may happen to
    start first (a log writer, a queue) but that serves every customer."""
    def run(*args, **kwargs):
        with system():
            return fn(*args, **kwargs)
    run.__name__ = getattr(fn, "__name__", "run")
    return run


def set_for_request(tenant: Optional[str]):
    """Set the tenant and return a token for ``reset``. For the request middleware only."""
    return _tenant.set(clean(tenant))


def reset(token) -> None:
    _tenant.reset(token)


PLATFORM_LABEL_ENV = "PLATFORM_TENANT_LABEL"


def platform_label() -> Optional[str]:
    """The one company label that platform work with no company of its own writes under (D-49).

    Valid only while exactly one customer exists. Before a second customer is added, each job must bind
    the company of the record it processes, and this setting must be removed. Unset (or not a plain
    identifier) means platform writes name no company and are refused once the label is mandatory.
    """
    return clean(os.environ.get(PLATFORM_LABEL_ENV, "").strip())


LEGACY_LABEL = "default"   # what the data written before tenant labels carries (D-37 Q1)


def company_label() -> str:
    """The company label for code that writes or searches by company and has no better source (D-49).

    While exactly one customer exists this is the configured platform label, else the legacy label that
    all existing rows carry. Reads only the environment, never the request, so it behaves the same under
    either import name and does not change what a running service does today. Before a second customer
    is added, callers must take the company from the record or request they serve instead.
    """
    return platform_label() or LEGACY_LABEL


def connection_settings() -> Optional[tuple]:
    """(tenant, system) to apply to a new connection, or None when there is nothing to say."""
    tenant, system_on = _tenant.get(), _system.get()
    if tenant is None and not system_on:
        return None
    if tenant is None:
        tenant = platform_label()
    return (tenant or "", "on" if system_on else "off")


# --- carrying the tenant into threads --------------------------------------------------------------
# asyncio tasks, ``asyncio.to_thread`` and Starlette's threadpool copy the ContextVars; a bare
# ``threading.Thread`` and ``ThreadPoolExecutor.submit`` do NOT. Work a request starts (a report job, a
# workflow run) would then lose its tenant and, with row-level security on, see no rows. So both are
# made to carry the two values of the code that started them -- once, here, for every current and
# future call site, instead of editing each one.
#
#   * a thread/task started while a tenant (or system mode) is bound runs with the same;
#   * a pool task ALWAYS sets both values explicitly (even to "none"), so a pooled thread can never run
#     one caller's task under another caller's tenant;
#   * a pool's own worker threads are not given a tenant of their own.


@contextmanager
def _rebind(tenant: Optional[str], system_on: bool) -> Iterator[None]:
    t = _tenant.set(tenant)
    s = _system.set(system_on)
    try:
        yield
    finally:
        _system.reset(s)
        _tenant.reset(t)


def _carry(fn):
    tenant, system_on = _tenant.get(), _system.get()

    def run(*args, **kwargs):
        with _rebind(tenant, system_on):
            return fn(*args, **kwargs)
    return run


def install_propagation() -> None:
    """Idempotent. Patches ``Thread.start`` and ``ThreadPoolExecutor.submit`` to carry the tenant."""
    if getattr(threading.Thread, "_bp_tenant_carry", False):
        return
    original_start = threading.Thread.start
    original_submit = ThreadPoolExecutor.submit

    def start(self):
        owner = getattr(self._target, "__module__", "") if getattr(self, "_target", None) else ""
        if owner != "concurrent.futures.thread" and (_tenant.get() is not None or _system.get()):
            self.run = _carry(self.run)
        return original_start(self)

    def submit(self, fn, /, *args, **kwargs):
        return original_submit(self, _carry(fn), *args, **kwargs)

    threading.Thread.start = start
    ThreadPoolExecutor.submit = submit
    threading.Thread._bp_tenant_carry = True


install_propagation()


# --- telling the database ----------------------------------------------------------------------------
# One place for every connection, whichever code makes it: the driver's connect() is wrapped, so
# get_conn(), the ~16 direct psycopg2.connect() callers and the scripts all get it. A pooled
# connection is reused by many callers, so the pool also re-applies on every checkout.

_SET_SQL = "SELECT set_config('app.tenant_id', %s, false), set_config('app.system', %s, false)"


def tell_connection(conn, *, always: bool = False) -> None:
    """Set ``app.tenant_id`` / ``app.system`` on ``conn`` from the current context.

    Does nothing when nothing is bound, unless ``always`` (a reused connection must be cleared). The
    setting is committed on its own: a later rollback by the caller must not undo it.
    """
    say = connection_settings()
    if say is None:
        if not always:
            return
        say = ("", "off")
    cur = conn.cursor()
    try:
        cur.execute(_SET_SQL, say)
    finally:
        cur.close()
    if not getattr(conn, "autocommit", False):
        conn.commit()


def install_driver_patch(module=None) -> None:
    """Wrap ``module.connect`` (default: psycopg2) so each new connection is told its tenant."""
    if module is None:
        try:
            import psycopg2 as module  # type: ignore
        except Exception:  # noqa: BLE001 - no driver, nothing to patch
            return
    if getattr(module, "_bp_tenant_connect", False):
        return
    original = module.connect

    def connect(*args, **kwargs):
        conn = original(*args, **kwargs)
        try:
            tell_connection(conn)
        except Exception:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        return conn

    module.connect = connect
    module._bp_tenant_connect = True


def install_pool_hook() -> None:
    """SQLAlchemy pools reuse connections: re-apply (or clear) the tenant each time one is checked out."""
    try:
        from sqlalchemy import event
        from sqlalchemy.engine import Engine
    except Exception:  # noqa: BLE001
        return
    if getattr(Engine, "_bp_tenant_checkout", False):
        return

    @event.listens_for(Engine, "checkout")
    def _checkout(dbapi_connection, connection_record, connection_proxy):
        tell_connection(dbapi_connection, always=True)

    Engine._bp_tenant_checkout = True


install_driver_patch()
install_pool_hook()
