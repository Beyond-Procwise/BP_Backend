"""Database connections for the assurance layer: read through one door, write through another.

Reads (facts, tone inputs) use ``reader``; the capture and learning writes use ``writer``. Each uses a
DEDICATED role when its credentials are configured, and otherwise falls back to an INTERIM control:

    reader   no dedicated role   the application's existing login, with the SESSION set read-only
    writer   no dedicated role   the application's existing login, unchanged

The interim control is a guardrail, not a boundary: a session can switch ``default_transaction_read_only``
back off, and the existing login still holds write privileges. Only the dedicated reader role makes a write
impossible (deploy/sql/2026-10-09_email_agent_roles.sql). ``read_control()`` reports which is in force and is
stored with every assurance record, so an audit can see what each draft was read under.

Dedicated credentials come from the environment, never the repo:
    EMAIL_AGENT_RO_USER / EMAIL_AGENT_RO_PASSWORD     members of email_agent_reader
    EMAIL_AGENT_RW_USER / EMAIL_AGENT_RW_PASSWORD     members of email_agent_writer
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

DEDICATED = "dedicated_role"
INTERIM = "interim_readonly_session"
UNENFORCED = "unenforced"      # the interim control could not be applied to this connection
_warned = set()


def _creds(prefix: str):
    user, pw = os.environ.get(f"EMAIL_AGENT_{prefix}_USER"), os.environ.get(f"EMAIL_AGENT_{prefix}_PASSWORD")
    return (user, pw) if user and pw else None


def read_control() -> str:
    """Which control the reader is under right now."""
    return DEDICATED if _creds("RO") else INTERIM


def _connect_as(settings: Any, user: str, password: str):
    import psycopg2

    return psycopg2.connect(host=getattr(settings, "db_host"), dbname=getattr(settings, "db_name"),
                            port=getattr(settings, "db_port", 5432), user=user, password=password)


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


@contextmanager
def _borrow(agent_nick: Any) -> Iterator[Any]:
    """The agent's own connection, closed afterwards if it handed us a bare connection."""
    cm = agent_nick.get_db_connection()
    with cm as conn:
        yield conn
    if cm is conn and hasattr(conn, "close"):
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def reader(agent_nick: Any, state: Optional[dict] = None) -> Iterator[Any]:
    """A connection that only reads. ``state['control']`` is set to the control that actually applied."""

    state = state if state is not None else {}
    creds = _creds("RO")
    if creds:
        conn = _connect_as(agent_nick.settings, *creds)
        try:
            conn.set_session(readonly=True, autocommit=True)     # belt and braces on top of the role
            state["control"] = DEDICATED
            yield conn
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        return
    _warn_once("interim", "email assurance reads run under the INTERIM control (a read-only session on the "
                          "application login): set EMAIL_AGENT_RO_USER/PASSWORD to use the dedicated role")
    with _borrow(agent_nick) as conn:
        prior = None
        state["control"] = UNENFORCED
        setter = getattr(conn, "set_session", None)
        if callable(setter):
            try:
                prior = (conn.readonly, conn.autocommit)
                setter(readonly=True, autocommit=True)
                state["control"] = INTERIM
            except Exception:  # noqa: BLE001 - reads are still reads; say so rather than fail the draft
                _warn_once("unenforced", "could not set the session read-only; continuing without the interim control")
        try:
            yield conn
        finally:
            if prior is not None:                                # a caller that reuses this connection gets it back as it was
                try:
                    # psycopg2: readonly=None means "leave it alone", NOT "back to the default" - 'default' is
                    # the value that resets it. Getting this wrong leaves a reused connection read-only.
                    setter(readonly="default" if prior[0] is None else prior[0], autocommit=prior[1])
                except Exception:  # noqa: BLE001
                    _warn_once("restore", "could not restore the connection's read-only setting")


@contextmanager
def writer(agent_nick: Any = None) -> Iterator[Any]:
    """A connection for the email_agent tables. Commits on success, rolls back on error."""

    creds = _creds("RW")
    if creds and agent_nick is not None:
        conn = _connect_as(agent_nick.settings, *creds)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        return
    if agent_nick is not None:
        with _borrow(agent_nick) as conn:
            yield conn
        return
    from src.services.db import get_conn

    with get_conn() as conn:
        yield conn
