"""Admin -> Integrations -> Email: connect Microsoft 365 and choose shared mailboxes.

Every route except the callback is `mailbox.bind` (configure -- Admin only), checked by the
server; the screen never decides on its own who may connect a company mailbox. The
callback is called by Microsoft's redirect of the admin's browser and carries no bearer
token: it trusts only the signed, single-use state (services/mail_m365/consent.py), and it
never puts anything Microsoft sent into the redirect except a code from a fixed set.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from services.db import get_conn
from services.mail_m365 import admin, consent
from services.mail_m365.config import load_settings, read_integration
from services.mail_m365.consent import load_state_key
from services.mail_m365.graph_client import Throttled
from services.mail_m365.setup_script import render as render_script
from services.style.graph_source import MailboxAccessDenied, MailboxUnreachable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/integrations/mail", tags=["Mail integration"])
callback_router = APIRouter(prefix="/admin/integrations/mail", tags=["Mail integration"])

_AGENT = "MailIntegrationRouter"
_STATUS = {"not_connected": 409, "duplicate": 409, "same_as_control": 422,
           "cannot_read": 422, "over_scoped": 422, "inconclusive": 503, "not_found": 404}
# The only codes the callback may put in the redirect; anything else becomes consent_failed.
_STATE_CODES = {"malformed", "bad_signature", "expired", "reused"}
_CONSENT_CODES = {"not_granted", "bad_tenant", "tenant_mismatch", "access_denied"}


def _conn():
    return get_conn()


def _actor(p: Any) -> str:
    return str(getattr(p, "email", None) or getattr(p, "username", None)
               or getattr(p, "subject", None) or "unknown")


def _settings_or_503():
    s = load_settings()
    if s is None:
        raise HTTPException(503, "Microsoft 365 is not configured on this server.")
    return s


def _refused(exc: admin.MailboxRefused) -> HTTPException:
    return HTTPException(_STATUS.get(exc.code, 422), {"code": exc.code, "message": exc.message})


class AddBody(BaseModel):
    address: str = Field(..., min_length=3)
    control_mailbox: str = Field(..., min_length=3)


class VerifyBody(BaseModel):
    control_mailbox: str = Field(..., min_length=3)


class DefaultBody(BaseModel):
    is_default: bool


@router.get("", summary="Microsoft 365 connection and shared mailboxes")
def get_status(principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "read"})
    with _conn() as conn:
        return {**admin.status(conn, settings=load_settings()), "canConfigure": True}


@router.post("/m365/connect", summary="Start Microsoft's admin approval")
def connect(principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "connect"})
    s = _settings_or_503()
    with _conn() as conn:
        return {"authorization_url": consent.begin(conn, s, load_state_key(s), _actor(principal))}


@callback_router.get("/m365/callback", include_in_schema=False)
def callback(state: str = "", tenant: Optional[str] = None,
             admin_consent: Optional[str] = None, error: Optional[str] = None):
    s = _settings_or_503()
    back = f"{s.public_base_url.rstrip('/')}/?view=settings"
    try:
        with _conn() as conn:
            consent.complete(conn, s, load_state_key(s), state=state, tenant=tenant,
                             admin_consent=admin_consent, error=error)
    except consent.StateRejected as exc:
        logger.warning("mail consent callback refused: %s", exc.code)
        code = exc.code if exc.code in _STATE_CODES else "consent_failed"
        return RedirectResponse(f"{back}&mail_error={code}", status_code=302)
    except consent.ConsentNotGiven as exc:
        code = exc.code if exc.code in _CONSENT_CODES else "consent_failed"
        logger.warning("mail consent not given: %s", code)
        return RedirectResponse(f"{back}&mail_error={code}", status_code=302)
    return RedirectResponse(f"{back}&mail=connected", status_code=302)


@router.get("/m365/setup-script", response_class=PlainTextResponse,
            summary="PowerShell that restricts Procwise to the chosen mailboxes")
def setup_script(principal: Any = Depends(require_user),
                 service_principal_object_id: Optional[str] = None) -> str:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "script"})
    s = _settings_or_503()
    with _conn() as conn:
        tenant = read_integration(conn).get("tenant_id") or "<your tenant id>"
    try:
        return render_script(tenant, s.client_id, service_principal_object_id)
    except ValueError:
        raise HTTPException(400, "That is not a valid Object ID. Copy it from Microsoft Entra "
                                 "-> Enterprise applications -> Procwise Mail, or connect "
                                 "Microsoft 365 first.")


@router.post("/mailboxes", summary="Add a shared mailbox and prove access is restricted")
def add(body: AddBody, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "add", "address": body.address})
    s = _settings_or_503()
    try:
        with _conn() as conn:
            return admin.add_mailbox(conn, body.address, body.control_mailbox,
                                     actor=_actor(principal), settings=s)
    except admin.MailboxRefused as exc:
        raise _refused(exc)


@router.post("/mailboxes/{binding_id}/verify", summary="Re-run the access proof")
def verify(binding_id: int, body: VerifyBody, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "verify", "binding_id": binding_id})
    s = _settings_or_503()
    try:
        with _conn() as conn:
            return admin.verify_mailbox(conn, binding_id, body.control_mailbox,
                                        actor=_actor(principal), settings=s)
    except admin.MailboxRefused as exc:
        raise _refused(exc)


@router.post("/mailboxes/{binding_id}/test-send", summary="Send a test email to yourself")
def test_send(binding_id: int, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "test_send", "binding_id": binding_id})
    s = _settings_or_503()
    to = getattr(principal, "email", None)
    if not to:
        raise HTTPException(422, "Your sign-in has no email address to send the test to.")
    try:
        with _conn() as conn:
            return admin.test_send(conn, binding_id, to, actor=_actor(principal), settings=s)
    except admin.MailboxRefused as exc:
        raise _refused(exc)
    except Throttled:
        raise HTTPException(503, "Microsoft is limiting requests right now. Try again in a minute.")
    except MailboxAccessDenied:
        raise HTTPException(403, "Microsoft refused to let Procwise send from this mailbox. "
                                 "Check the mailbox is in the “Procwise Mailboxes” group; "
                                 "changes can take up to 30 minutes.")
    except MailboxUnreachable:
        raise HTTPException(502, "Microsoft did not answer. Nothing was sent; try again shortly.")


@router.patch("/mailboxes/{binding_id}", summary="Make this the default sending mailbox")
def set_default(binding_id: int, body: DefaultBody, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "default", "binding_id": binding_id})
    if not body.is_default:
        raise HTTPException(422, "Choose another mailbox as the default instead.")
    try:
        with _conn() as conn:
            return admin.make_default(conn, binding_id, actor=_actor(principal))
    except admin.MailboxRefused as exc:
        raise _refused(exc)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.delete("/mailboxes/{binding_id}", summary="Stop using this mailbox")
def remove(binding_id: int, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "remove", "binding_id": binding_id})
    try:
        with _conn() as conn:
            return admin.remove_mailbox(conn, binding_id, actor=_actor(principal))
    except admin.MailboxRefused as exc:
        raise _refused(exc)


@router.delete("", summary="Disconnect Microsoft 365")
def disconnect(principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "disconnect"})
    with _conn() as conn:
        return admin.disconnect(conn, actor=_actor(principal))
