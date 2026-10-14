"""Presentation mode: who may turn it on, for how long, and the trail it leaves.

  * Only an Admin. Checked on every request that asks for presentation data, not by hiding a toggle.
  * Per SESSION. An activation lives as long as the sign-in it was made on (the token's expiry) and
    is ended by sign-out; it is never a user or tenant default.
  * Being Admin is necessary, not sufficient: the session must also have an active activation.
  * Every activation, deactivation and presentation export is appended to proc.bp_presentation_log.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

from src.services import rbac

from .scope import ADMIN

logger = logging.getLogger(__name__)


def session_id(principal: Any) -> str:
    """One sign-in. A new sign-in has a new token (new auth_time/iat/jti), so a new session id."""
    claims = getattr(principal, "claims", None) or {}
    basis = "|".join(str(x) for x in (getattr(principal, "subject", ""), claims.get("jti"),
                                      claims.get("auth_time") or claims.get("iat")))
    return hashlib.sha256(basis.encode()).hexdigest()[:32]


def session_expiry(principal: Any) -> Optional[dt.datetime]:
    exp = (getattr(principal, "claims", None) or {}).get("exp")
    try:
        return dt.datetime.fromtimestamp(int(exp), dt.timezone.utc) if exp else None
    except (TypeError, ValueError, OverflowError):
        return None


def is_admin(principal: Any) -> bool:
    return rbac.effective_role(principal) == ADMIN


def _log(subject: str, sid: str, event: str, detail: Optional[Dict[str, Any]] = None) -> None:
    from src.services.db import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_presentation_log (subject, session_id, event, detail, expires_at) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s)",
                    (subject, sid, event, json.dumps(detail or {}), (detail or {}).get("expires_at")))


def activate(principal: Any) -> Dict[str, Any]:
    if not is_admin(principal):
        raise PermissionError("only an Admin can use presentation data")
    exp = session_expiry(principal)
    if exp is not None and exp <= dt.datetime.now(dt.timezone.utc):
        raise PermissionError("the session has expired")
    sid = session_id(principal)
    _log(principal.subject, sid, "activate", {"expires_at": exp.isoformat() if exp else None})
    return {"active": True, "session_id": sid}


def deactivate(principal: Any) -> Dict[str, Any]:
    sid = session_id(principal)
    _log(getattr(principal, "subject", "") or "", sid, "deactivate")
    return {"active": False}


def is_active(principal: Any) -> bool:
    """Admin AND an activation on THIS session that has not been ended and has not expired."""
    if not is_admin(principal):
        return False
    exp = session_expiry(principal)
    if exp is not None and exp <= dt.datetime.now(dt.timezone.utc):
        return False
    from src.services.db import get_conn
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT event FROM proc.bp_presentation_log WHERE subject = %s AND session_id = %s "
                        "AND event IN ('activate','deactivate') ORDER BY log_id DESC LIMIT 1",
                        (principal.subject, session_id(principal)))
            row = cur.fetchone()
            return bool(row and row[0] == "activate")
    except Exception:
        logger.error("presentation mode: could not read the activation log; treating as off", exc_info=True)
        return False


def log_export(principal: Any, detail: Dict[str, Any]) -> None:
    _log(getattr(principal, "subject", "") or "", session_id(principal), "export", detail)
