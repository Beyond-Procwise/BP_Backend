"""Admin -> Integrations -> Email: connect Microsoft 365 and choose shared mailboxes.

Every route except the callback is `mailbox.bind` (configure -- Admin only), checked by the
server; the screen never decides on its own who may connect a company mailbox. The
callback is called by Microsoft's redirect of the admin's browser and carries no bearer
token: it trusts only the signed, single-use state and a Microsoft sign-in whose id_token
it verifies itself (services/mail_m365/consent.py), and it never puts anything Microsoft
sent into the redirect except a code from a fixed set.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from services.db import get_conn
from services import mail_policy, rbac
from services.actions import action_class
from services.mail_m365 import admin, consent, sending
from services.mail_m365.config import load_settings, read_integration, settings_page_url
from services.mail_m365.consent import load_state_key
from services.mail_m365.graph_client import Throttled
from services.mail_m365.setup_script import render as render_script
from services.style.graph_source import MailboxAccessDenied, MailboxUnreachable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/integrations/mail", tags=["Mail integration"])
callback_router = APIRouter(prefix="/admin/integrations/mail", tags=["Mail integration"])
# Any signed-in user may ask which address supplier email leaves from; no endpoint gate.
sender_router = APIRouter(prefix="/mail", tags=["Mail integration"])

_AGENT = "MailIntegrationRouter"
_STATUS = {"not_connected": 409, "duplicate": 409, "same_as_control": 422,
           "cannot_read": 422, "over_scoped": 422, "inconclusive": 503, "not_found": 404}
# The only codes the callback may put in the redirect; anything else becomes consent_failed.
_STATE_CODES = {"malformed", "bad_signature", "expired", "reused"}
_CONSENT_CODES = {"not_granted", "bad_tenant", "tenant_mismatch", "access_denied",
                  "wrong_account", "signin_failed"}


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


def _refused(exc: admin.MailboxRefused) -> JSONResponse:
    """A structured body as a real JSON response: the app's global HTTPException handler
    flattens non-string details to a Python repr, which would hide code and message."""
    return JSONResponse(status_code=_STATUS.get(exc.code, 422),
                        content={"detail": {"code": exc.code, "message": exc.message}})


class AddBody(BaseModel):
    address: str = Field(..., min_length=3)
    control_mailbox: str = Field(..., min_length=3)


class VerifyBody(BaseModel):
    control_mailbox: str = Field(..., min_length=3)


class DefaultBody(BaseModel):
    is_default: bool


@router.get("", summary="Microsoft 365 connection and shared mailboxes")
def get_status(principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.view", principal, agent=_AGENT, context={"op": "read"})
    with _conn() as conn:
        return {**admin.status(conn, settings=load_settings()),
                "canConfigure": _can_configure(principal)}


def _can_configure(principal: Any) -> bool:
    """May this caller change the settings? Asked of the role table directly: the gate
    would write a refused-attempt audit row on every read of the page."""
    try:
        return rbac.may(rbac.effective_role(principal), action_class("mailbox.bind"))
    except Exception:  # noqa: BLE001 - when unsure, show no controls (the server still decides)
        logger.exception("could not work out whether the caller may configure email")
        return False


def _policy_view(p: Any) -> dict:
    return {"policy": {"version": p.version, "internal_send": p.internal_send,
                       "company_domains": list(p.company_domains),
                       "external_recipients": p.external_recipients,
                       "read_scope": p.read_scope, "on_disconnect": p.on_disconnect,
                       "access_level": p.access_level,
                       "notices_from_mailbox": p.notices_from_mailbox,
                       "poll_seconds": p.poll_seconds,
                       "restriction_group": p.restriction_group},
            "history": list(p.history)}


class PolicyBody(BaseModel):
    changes: dict
    version: int


def _detail(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": {"code": code, "message": message}})


@router.get("/policy", summary="The email rules for this company")
def get_policy(principal: Any = Depends(require_user)) -> Any:
    gate("mailbox.view", principal, agent=_AGENT, context={"op": "policy_read"})
    try:
        policy = mail_policy.load()
    except mail_policy.PolicyUnavailable:
        logger.exception("email policy unavailable")
        return _detail(503, "policy_unavailable",
                       "The email settings cannot be read right now. Nothing has been changed.")
    return {**_policy_view(policy), "canConfigure": _can_configure(principal)}


@router.patch("/policy", summary="Change the email rules")
def patch_policy(body: PolicyBody, principal: Any = Depends(require_user)) -> Any:
    gate("mailbox.bind", principal, agent=_AGENT,
         context={"op": "policy_save", "fields": sorted(body.changes)})
    try:
        policy = mail_policy.save(body.changes, by=_actor(principal), expected_version=body.version)
    except mail_policy.PolicyConflict as exc:
        return _detail(409, "conflict", str(exc))
    except ValueError as exc:
        return _detail(422, "invalid", str(exc))
    except mail_policy.PolicyUnavailable:
        logger.exception("email policy unavailable on save")
        return _detail(503, "policy_unavailable",
                       "The email settings cannot be saved right now. Nothing has been changed.")
    return _policy_view(policy)


@router.post("/m365/connect", summary="Start Microsoft's admin approval")
def connect(principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "connect"})
    s = _settings_or_503()
    with _conn() as conn:
        return {"authorization_url": consent.begin(conn, s, load_state_key(s), _actor(principal))}


def _audit_refusal(stage: str, code: str) -> None:
    """A refused callback is a security event: someone's browser (or a forged link) came
    back with a state or an approval we would not accept. Only our fixed code is kept,
    never the query text."""
    try:
        from services.agent_actions import record_action
        record_action(phase="authorize", action_type="mailbox.consent_callback",
                      agent=_AGENT, status="denied",
                      summary=f"Microsoft 365 connection refused at the callback: {code}",
                      details={"stage": stage, "code": code})
    except Exception:  # noqa: BLE001 - audit must never break the redirect
        logger.debug("consent callback audit failed", exc_info=True)


@callback_router.get("/m365/callback", include_in_schema=False)
def callback(state: str = "", tenant: Optional[str] = None,
             admin_consent: Optional[str] = None, error: Optional[str] = None,
             code: Optional[str] = None):
    """Both of Microsoft's returns land here: the admin-consent page (tenant,
    admin_consent) and then the sign-in (code). The signed state says which; the tenant
    is stored only from the sign-in's verified id_token (services/mail_m365/consent.py)."""
    s = _settings_or_503()
    back = settings_page_url(s)
    try:
        with _conn() as conn:
            nxt = consent.handle_callback(conn, s, load_state_key(s), state=state,
                                          tenant=tenant, admin_consent=admin_consent,
                                          error=error, code=code)
    except consent.StateRejected as exc:
        logger.warning("mail consent callback refused: %s", exc.code)
        code_out = exc.code if exc.code in _STATE_CODES else "consent_failed"
        _audit_refusal("state", code_out)
        return RedirectResponse(f"{back}&mail_error={code_out}", status_code=302)
    except consent.ConsentNotGiven as exc:
        code_out = exc.code if exc.code in _CONSENT_CODES else "consent_failed"
        logger.warning("mail consent not given: %s %s", code_out, exc.detail)
        _audit_refusal("consent", code_out)
        return RedirectResponse(f"{back}&mail_error={code_out}", status_code=302)
    except Exception:  # noqa: BLE001 - the admin's browser must never land on a JSON 500
        logger.exception("mail consent callback failed")
        _audit_refusal("unexpected", "consent_failed")
        return RedirectResponse(f"{back}&mail_error=consent_failed", status_code=302)
    if nxt:   # consent recorded by Microsoft; now the sign-in that proves which tenant
        return RedirectResponse(nxt, status_code=302)
    return RedirectResponse(f"{back}&mail=connected", status_code=302)


@router.get("/m365/setup-script", response_class=PlainTextResponse,
            summary="PowerShell that restricts Procwise to the chosen mailboxes")
def setup_script(principal: Any = Depends(require_user),
                 service_principal_object_id: Optional[str] = None) -> str:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "script"})
    s = _settings_or_503()
    try:
        policy = mail_policy.load()
    except mail_policy.PolicyUnavailable:
        raise HTTPException(503, "The email settings cannot be read right now, so the "
                                 "script was not made.")
    with _conn() as conn:
        tenant = read_integration(conn).get("tenant_id") or "<your tenant id>"
    try:
        return render_script(tenant, s.client_id, service_principal_object_id,
                             group=policy.restriction_group, access_level=policy.access_level)
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
        return _refused(exc)


@router.post("/mailboxes/{binding_id}/verify", summary="Re-run the access proof")
def verify(binding_id: int, body: VerifyBody, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "verify", "binding_id": binding_id})
    s = _settings_or_503()
    try:
        with _conn() as conn:
            return admin.verify_mailbox(conn, binding_id, body.control_mailbox,
                                        actor=_actor(principal), settings=s)
    except admin.MailboxRefused as exc:
        return _refused(exc)


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
        return _refused(exc)
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
        return _refused(exc)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.delete("/mailboxes/{binding_id}", summary="Stop using this mailbox")
def remove(binding_id: int, principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "remove", "binding_id": binding_id})
    try:
        with _conn() as conn:
            return admin.remove_mailbox(conn, binding_id, actor=_actor(principal))
    except admin.MailboxRefused as exc:
        return _refused(exc)


@router.delete("", summary="Disconnect Microsoft 365")
def disconnect(principal: Any = Depends(require_user)) -> dict:
    gate("mailbox.bind", principal, agent=_AGENT, context={"op": "disconnect"})
    with _conn() as conn:
        return admin.disconnect(conn, actor=_actor(principal))


@sender_router.get("/sender", summary="Which address supplier emails go out from, and can they")
def get_sender(principal: Any = Depends(require_user)) -> dict:
    """A read for every signed-in user (the compose window needs it), so it is not gated
    and writes no refused-attempt audit row. Exposes only mode, address, can_send, reason
    and max_attachment_bytes (0 = no cap Procwise knows of; compose blocks above it)."""
    try:
        with _conn() as conn:
            d = sending.decide(conn)
    except Exception:  # noqa: BLE001 - compose must get an answer, never a 500
        logger.exception("mail sender status failed")
        return {"mode": "held", "address": None, "can_send": False,
                "reason": sending.HELD_UNCHECKED, "max_attachment_bytes": 0}
    address = d.binding.mailbox_address if d.binding is not None else None
    if d.kind == "ses":
        return {"mode": "ses", "address": None, "can_send": True, "reason": None,
                "max_attachment_bytes": 0}
    if d.kind == "graph":
        return {"mode": "m365", "address": address, "can_send": True, "reason": None,
                "max_attachment_bytes": sending.MAX_MIME_BYTES}
    return {"mode": "held", "address": address, "can_send": False, "reason": d.reason,
            "max_attachment_bytes": 0}
