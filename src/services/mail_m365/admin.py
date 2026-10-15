"""What an admin does on Integrations -> Email, as plain functions the router calls.

The order inside add_mailbox is the product claim: read the mailbox first (otherwise a
binding row would exist for a mailbox we cannot use), then create it inactive, then prove
the control mailbox is denied - only that activates it.
"""
from __future__ import annotations

from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import Any, Callable, Dict, Optional

from services.mail_m365.config import (M365Settings, merge_integration,
                                       read_integration)
from services.mail_m365.graph_client import M365Client
from services.style.graph_source import MailboxUnreachable, resolve_credentials
from services.style.mailbox import (ORG_USER_REF, PROVIDER_GRAPH, ROLE_SEND_RECEIVE,
                                    BindingNotFound, MailboxBinding,
                                    MailboxBindingRepository, ScopeVerificationFailed,
                                    normalise_address)

ClientFactory = Callable[[str], M365Client]

_MESSAGES = {
    "over_scoped": ("We could also read {control}, so access is not restricted to your chosen "
                    "mailboxes yet. Run the commands above, wait up to 30 minutes for Microsoft "
                    "to apply them, then try again."),
    "cannot_read": ("We could not read {address}. Check it is in the “Procwise Mailboxes” "
                    "group; changes can take up to 30 minutes."),
    "duplicate": "{address} is already connected.",
    "same_as_control": "The control mailbox must be a different mailbox — normally your own.",
    "inconclusive": ("Microsoft did not give a clear answer ({detail}). Nothing was changed; "
                     "try again shortly."),
    "not_connected": "Connect Microsoft 365 first.",
    "not_found": "That mailbox is not connected.",
}


class MailboxRefused(Exception):
    def __init__(self, code: str, **fmt: Any) -> None:
        self.code = code
        self.message = _MESSAGES[code].format(**fmt)
        super().__init__(self.message)


def _default_factory(settings: M365Settings) -> ClientFactory:
    def build(tenant_id: str) -> M365Client:
        return M365Client(resolve_credentials(settings.secret_arn, tenant_id=tenant_id))
    return build


def _tenant(conn: Any) -> str:
    integration = read_integration(conn)
    if integration.get("status") != "connected" or not integration.get("tenant_id"):
        raise MailboxRefused("not_connected")
    return integration["tenant_id"]


def _view(b: MailboxBinding) -> Dict[str, Any]:
    return {"binding_id": b.binding_id, "address": b.mailbox_address, "active": b.is_active,
            "is_default": b.is_default, "health_state": b.health_state,
            "verified_at": b.scope_verified_at, "last_test_send_at": b.last_test_send_at,
            "can_be_default": bool(b.is_active and b.last_test_send_at and not b.is_default)}


def _org_binding(repo: MailboxBindingRepository, binding_id: int, *,
                 active_only: bool = False) -> MailboxBinding:
    b = repo.get(binding_id)
    if b is None or b.user_ref != ORG_USER_REF or (active_only and not b.is_active):
        raise MailboxRefused("not_found")
    return b


def status(conn: Any, *, settings: Optional[M365Settings]) -> Dict[str, Any]:
    integration = read_integration(conn) if conn is not None else {}
    connected = integration.get("status") == "connected"
    mailboxes = [_view(b) for b in MailboxBindingRepository(conn).list_org()] if connected else []
    return {"configured": settings is not None, "connected": connected,
            "tenant_id": integration.get("tenant_id"),
            "consented_by": integration.get("consented_by"),
            "consented_at": integration.get("consented_at"), "mailboxes": mailboxes}


def add_mailbox(conn: Any, address: str, control_mailbox: str, *, actor: str,
                settings: M365Settings, client_factory: Optional[ClientFactory] = None
                ) -> Dict[str, Any]:
    address, control = normalise_address(address), normalise_address(control_mailbox)
    tenant = _tenant(conn)
    if address == control:
        raise MailboxRefused("same_as_control")
    repo = MailboxBindingRepository(conn)
    if any(normalise_address(b.mailbox_address) == address for b in repo.list_org()):
        raise MailboxRefused("duplicate", address=address)
    client = (client_factory or _default_factory(settings))(tenant)
    try:
        readable, _detail = client.can_read(address)
    except MailboxUnreachable as exc:
        raise MailboxRefused("inconclusive", detail=str(exc))
    if not readable:
        raise MailboxRefused("cannot_read", address=address)
    binding = repo.create(user_ref=ORG_USER_REF, provider=PROVIDER_GRAPH,
                          mailbox_address=address, role=ROLE_SEND_RECEIVE,
                          credential_ref=settings.secret_arn)
    return {"ok": True, "mailbox": _view(_prove(repo, binding.binding_id, control, client))}


def _prove(repo: MailboxBindingRepository, binding_id: int, control: str,
           client: M365Client) -> MailboxBinding:
    try:
        return repo.verify_scope(binding_id, control_mailbox=control, probe=client.probe_control)
    except ScopeVerificationFailed:
        raise MailboxRefused("over_scoped", control=control)
    except MailboxUnreachable as exc:
        raise MailboxRefused("inconclusive", detail=str(exc))


def verify_mailbox(conn: Any, binding_id: int, control_mailbox: str, *, actor: str,
                   settings: M365Settings, client_factory: Optional[ClientFactory] = None
                   ) -> Dict[str, Any]:
    tenant = _tenant(conn)
    repo = MailboxBindingRepository(conn)
    b = _org_binding(repo, binding_id)
    control = normalise_address(control_mailbox)
    if control == normalise_address(b.mailbox_address):
        raise MailboxRefused("same_as_control")
    client = (client_factory or _default_factory(settings))(tenant)
    return {"ok": True, "mailbox": _view(_prove(repo, binding_id, control, client))}


def _test_message(sender: str, to: str) -> bytes:
    msg = EmailMessage()
    msg["From"], msg["To"] = sender, to
    msg["Subject"] = "Procwise test email"
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=sender.partition("@")[2] or None)
    msg.set_content(f"This is a test from Procwise. It was sent from {sender} through "
                    "Microsoft 365, and a copy should be in that mailbox's Sent Items.")
    return bytes(msg)


def test_send(conn: Any, binding_id: int, to_address: str, *, actor: str,
              settings: M365Settings, client_factory: Optional[ClientFactory] = None
              ) -> Dict[str, Any]:
    tenant = _tenant(conn)
    repo = MailboxBindingRepository(conn)
    b = _org_binding(repo, binding_id, active_only=True)
    client = (client_factory or _default_factory(settings))(tenant)
    client.send_mime(b.mailbox_address, _test_message(b.mailbox_address,
                                                      normalise_address(to_address)))
    return {"ok": True, "mailbox": _view(repo.record_test_send(binding_id))}


def make_default(conn: Any, binding_id: int, *, actor: str) -> Dict[str, Any]:
    repo = MailboxBindingRepository(conn)
    _org_binding(repo, binding_id)
    try:
        return {"ok": True, "mailbox": _view(repo.set_default(binding_id))}
    except BindingNotFound:
        raise MailboxRefused("not_found")
    # ValueError (not verified / no test send) propagates: the router shows its text as a 422.


def remove_mailbox(conn: Any, binding_id: int, *, actor: str) -> Dict[str, Any]:
    repo = MailboxBindingRepository(conn)
    _org_binding(repo, binding_id)
    return {"ok": True, "mailbox": _view(repo.deactivate(binding_id, reason=f"removed by {actor}"))}


def disconnect(conn: Any, *, actor: str) -> Dict[str, Any]:
    repo = MailboxBindingRepository(conn)
    for b in repo.list_org():
        repo.deactivate(b.binding_id, reason=f"Microsoft 365 disconnected by {actor}")
    # merge, never a whole-row write: a stale read must not resurrect spent consent nonces.
    merge_integration(conn, {"status": "disconnected", "disconnected_by": actor}, actor=actor)
    return {"ok": True, "note": ("Procwise has stopped using your mailboxes. To remove its "
                                 "access entirely, delete “Procwise Mail” under Microsoft Entra "
                                 "→ Enterprise applications.")}
