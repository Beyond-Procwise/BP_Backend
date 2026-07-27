"""Put the finished draft in the customer's Drafts folder.

This is the last step, and the one where the product claim is most exposed. Everything
before this reads; this writes into someone's mailbox, and a write needs
``Mail.ReadWrite`` — a broader grant than the ``Mail.Read`` the exemplar path uses. The
neighbouring permission is ``Mail.Send``, and the difference between a subsystem that
drafts and one that sends is one entry in an app registration.

So the claim is checked rather than asserted, in two independent ways:

* **No send-capable code exists.** The only endpoint this module knows is
  ``POST /users/{mailbox}/messages``, which Graph documents as creating a draft. There is
  no call to ``/sendMail`` and none to ``/messages/{id}/send``. A test parses this module
  and asserts neither string appears in the code.
* **The granted permissions are inspected.** ``verify_no_send_permission`` decodes the
  access token's ``roles`` claim and refuses to proceed if a send permission is present.
  That is proof about the actual grant, not about our intentions — and it is a read of a
  token we already hold, with no side effect.

The obvious alternative — try to send something and check it fails — is not implemented
and should not be. A probe that succeeds has sent a real email to a real person.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.style.graph_source import (
    GRAPH_BASE,
    GraphExemplarSource,
    MailboxAccessDenied,
    MailboxUnreachable,
)
from services.style.mailbox import MailboxBinding

logger = logging.getLogger(__name__)

# Every Graph application permission that can put mail on the wire. Presence of any of
# these in a token means the credential can send, whatever this code does or does not call.
SEND_PERMISSIONS = {
    "mail.send",
    "mail.send.shared",
    "mail.readwrite.shared",  # can send on behalf of a shared mailbox
}


class SendPermissionGranted(Exception):
    """The credential can send mail.

    Not a warning. If a send permission is present, "this platform cannot send on your
    behalf" is false at the credential level, and no amount of care in this module makes
    it true again.
    """


class DraftWriteFailed(Exception):
    """The draft could not be created in the mailbox."""


@dataclass(frozen=True)
class WrittenDraft:
    """Where the draft ended up."""

    external_draft_ref: str
    web_link: Optional[str] = None


def decode_token_roles(token: str) -> List[str]:
    """The ``roles`` claim of a JWT access token.

    The signature is deliberately not verified. This is not authentication — the token
    was just issued to us by the identity provider over TLS. It is an inspection of what
    we were granted, and a forged token would only mislead us about our own permissions,
    which is not a threat model that makes sense here.
    """

    try:
        payload_segment = token.split(".")[1]
        padding = "=" * (-len(payload_segment) % 4)
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        claims = json.loads(decoded)
    except Exception as exc:
        raise ValueError(f"could not decode the access token: {exc}") from exc

    roles = claims.get("roles") or claims.get("scp") or []
    if isinstance(roles, str):
        roles = roles.split()
    return [str(r) for r in roles]


def verify_no_send_permission(token: str) -> List[str]:
    """Raise if the credential holds any permission that can send mail.

    Returns the granted roles when it does not, so a caller can record what was actually
    checked rather than just that a check passed.
    """

    roles = decode_token_roles(token)
    granted_send = sorted(
        {r for r in roles if r.strip().lower() in SEND_PERMISSIONS}
    )
    if granted_send:
        raise SendPermissionGranted(
            "the mailbox credential holds "
            + ", ".join(granted_send)
            + " — it can send mail. Remove the permission from the application "
            "registration before enabling draft write-back; the platform's claim that it "
            "cannot send on the customer's behalf is false while this is granted."
        )
    return roles


class GraphDraftWriter:
    """Creates drafts in a bound mailbox. Creates only — it cannot send."""

    def __init__(
        self,
        binding: MailboxBinding,
        *,
        source: Optional[GraphExemplarSource] = None,
        transport: Optional[Any] = None,
        credentials: Optional[Any] = None,
        enforce_no_send: bool = True,
    ) -> None:
        self.binding = binding
        # Reuses the read adapter's auth rather than opening a second credential path.
        self._source = source or GraphExemplarSource(
            binding, transport=transport, credentials=credentials
        )
        self.enforce_no_send = enforce_no_send

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _token(self) -> str:
        token = self._source.token()
        if self.enforce_no_send:
            # Checked on every write, not once at binding time. A permission added to the
            # app registration next month would otherwise go unnoticed until someone
            # asked why an email had been sent.
            verify_no_send_permission(token)
        return token

    def write_draft(
        self,
        *,
        subject: Optional[str],
        body: str,
        to: Optional[List[str]] = None,
    ) -> WrittenDraft:
        """Create a draft in the bound mailbox's Drafts folder.

        ``POST /users/{mailbox}/messages`` creates an unsent message. Nothing here sets a
        send flag, and Graph has no way to send as a side effect of creation.
        """

        if not self.binding.can_receive_drafts:
            raise DraftWriteFailed(
                f"binding {self.binding.binding_id} has role {self.binding.role!r} and is "
                "not a draft target"
            )
        if not self.binding.usable:
            raise DraftWriteFailed(
                f"binding {self.binding.binding_id} is inactive or revoked"
            )
        if not (body or "").strip():
            raise DraftWriteFailed("refusing to create an empty draft")

        message: Dict[str, Any] = {
            "subject": subject or "",
            "body": {"contentType": "Text", "content": body},
        }
        if to:
            message["toRecipients"] = [
                {"emailAddress": {"address": address}} for address in to if address
            ]

        # Headers are built OUTSIDE the try, deliberately. Building them runs the
        # permission check, and SendPermissionGranted must never be caught by the
        # generic handler below: a caller reading it as DraftWriteFailed would treat
        # "this credential can send mail" as "the draft didn't save" and retry.
        headers = self._headers()

        url = f"{GRAPH_BASE}/users/{self.binding.mailbox_address}/messages"
        try:
            status, payload = self._source.transport.post_json(url, headers, message)
        except MailboxUnreachable:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise DraftWriteFailed(str(exc)) from exc

        if status in (401, 403, 404):
            raise MailboxAccessDenied(f"draft creation denied with HTTP {status}")
        if not 200 <= status < 300:
            raise DraftWriteFailed(f"draft creation returned HTTP {status}")

        ref = (payload or {}).get("id")
        if not ref:
            raise DraftWriteFailed("the provider returned no draft identifier")

        logger.info(
            "Created draft %s in %s (binding %s)",
            ref, self.binding.mailbox_address, self.binding.binding_id,
        )
        return WrittenDraft(external_draft_ref=ref, web_link=(payload or {}).get("webLink"))
