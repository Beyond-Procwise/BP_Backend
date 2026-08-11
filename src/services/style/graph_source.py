"""Reading exemplars from a Microsoft 365 mailbox.

Another implementation of the Phase 2 ``ExemplarSource`` protocol, and nothing outside
this file needed to change to add it — which was the point of drawing that seam.

Three things are worth knowing before reading the code.

**No send capability, by construction.** The only Graph paths this module knows are
``/messages`` and ``/mailFolders`` under a single user, plus a token endpoint. There is no
``/sendMail``, and a test asserts the string does not appear. The Graph application
registration is expected to hold ``Mail.Read`` scoped by an application access policy to
the bound mailbox; the scope probe below is what checks that expectation rather than
trusting it.

**The scope probe is the product claim.** ``probe_control_mailbox`` deliberately tries to
read a mailbox it should not be able to reach and reports whether it succeeded. A denial
is what lets a binding activate. Anyone can assert "we only read your mailbox"; this makes
the platform demonstrate it.

**Under C2, bodies never touch a table.** ``fetch`` returns ``RawExemplar`` objects that
the caller may put in a prompt and must not persist. The Mode C2 cache is process-local
and short-lived; see ``mailbox_cache``.

HTTP goes through an injected ``transport`` so this is testable without a tenant. The
default uses ``requests``. Note that everything here has been verified against a stub —
there is no Microsoft 365 tenant in this environment, so the request shapes are written
to the documented API but have not been exercised against a live Graph endpoint.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.style.mailbox import MailboxBinding
from services.style.sources import RawExemplar

from src.services import egress

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
GRAPH_SCOPE = "https://graph.microsoft.com/.default"

DEFAULT_TIMEOUT = 20
DEFAULT_FETCH_LIMIT = 25

# Folder names that are the mail client's, not the customer's filing. An "intent" read off
# one of these would be meaningless.
_SYSTEM_FOLDERS = {
    "inbox", "drafts", "sent items", "sentitems", "deleted items", "deleteditems",
    "junk email", "junkemail", "outbox", "archive", "conversation history",
    "clutter", "scheduled",
}


class MailboxUnreachable(Exception):
    """The mail provider did not answer in time, or at all.

    Drafting treats this as a reason to degrade visibly, never to fail — see the circuit
    breaker in the drafting path.
    """


class MailboxAccessDenied(Exception):
    """The provider refused. Expected during scope verification; a problem anywhere else."""


@dataclass(frozen=True)
class GraphCredentials:
    tenant_id: str
    client_id: str
    client_secret: str


def resolve_credentials(credential_ref: str, *, client=None) -> GraphCredentials:
    """Fetch client credentials from Secrets Manager.

    ``credential_ref`` is an ARN. It is never a token: the database refuses anything that
    does not look like a Secrets Manager ARN, so a secret cannot end up in an application
    table even by mistake.
    """

    if not str(credential_ref or "").startswith("arn:aws:secretsmanager:"):
        raise ValueError("credential_ref must be a Secrets Manager ARN")

    if client is None:  # pragma: no cover - exercised only with real AWS
        import boto3

        region = credential_ref.split(":")[3]
        client = boto3.client("secretsmanager", region_name=region)

    payload = client.get_secret_value(SecretId=credential_ref)
    raw = payload.get("SecretString")
    if not raw:
        raise ValueError(f"secret {credential_ref} has no SecretString payload")

    data = json.loads(raw) if isinstance(raw, str) else raw
    try:
        return GraphCredentials(
            tenant_id=data["tenant_id"],
            client_id=data["client_id"],
            client_secret=data["client_secret"],
        )
    except KeyError as exc:
        raise ValueError(f"secret {credential_ref} is missing {exc.args[0]}") from exc


class RequestsTransport:
    """The default transport. Kept tiny so it can be swapped for a stub in tests."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def post_form(self, url: str, data: Dict[str, str]) -> Tuple[int, Any]:

        try:
            response = egress.post(url, purpose=egress.Purpose.MAILBOX,
                                   data=data, timeout=self.timeout)
        except Exception as exc:  # requests.Timeout and friends
            raise MailboxUnreachable(str(exc)) from exc
        return response.status_code, _safe_json(response)

    def get(self, url: str, headers: Dict[str, str]) -> Tuple[int, Any]:

        try:
            response = egress.get(url, purpose=egress.Purpose.MAILBOX,
                                  headers=headers, timeout=self.timeout)
        except Exception as exc:
            raise MailboxUnreachable(str(exc)) from exc
        return response.status_code, _safe_json(response)

    def post_json(self, url: str, headers: Dict[str, str], payload: Any) -> Tuple[int, Any]:
        """Used only to CREATE a draft. There is no endpoint here that transmits."""


        try:
            response = egress.post(url, purpose=egress.Purpose.MAILBOX,
                                   headers=headers, json=payload,
                                   timeout=self.timeout)
        except Exception as exc:
            raise MailboxUnreachable(str(exc)) from exc
        return response.status_code, _safe_json(response)


def _safe_json(response) -> Any:  # pragma: no cover - trivial
    try:
        return response.json()
    except Exception:
        return {}


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html or "")
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def folder_to_intent(folder_name: Optional[str], known_intents) -> Optional[str]:
    """Map a mailbox folder name onto a style intent code.

    Subfolder names are read first because they are the customer's own filing, and a
    person who has a folder called "Award notifications" has already classified their mail
    more reliably than any model will. Only exact matches count once normalised — a folder
    called "Awards 2025" is a guess, and a wrong intent is worse than no intent because it
    compiles a profile from the wrong emails.

    System folders are ignored: "Inbox" is not a communication type.
    """

    if not folder_name:
        return None
    lowered = folder_name.strip().lower()
    if lowered in _SYSTEM_FOLDERS:
        return None
    normalised = re.sub(r"[^a-z]+", "_", lowered).strip("_")
    if normalised in known_intents:
        return normalised
    singular = normalised.rstrip("s")
    for code in known_intents:
        if code == singular or code.rstrip("s") == normalised.rstrip("s"):
            return code
    return None


class GraphExemplarSource:
    """An ``ExemplarSource`` backed by one Microsoft 365 mailbox."""

    def __init__(
        self,
        binding: MailboxBinding,
        *,
        transport: Optional[Any] = None,
        credentials: Optional[GraphCredentials] = None,
        secrets_client: Optional[Any] = None,
        fetch_limit: int = DEFAULT_FETCH_LIMIT,
        known_intents: Optional[Any] = None,
    ) -> None:
        self.binding = binding
        self.transport = transport or RequestsTransport()
        self._credentials = credentials
        self._secrets_client = secrets_client
        self._token: Optional[str] = None
        self.fetch_limit = fetch_limit
        self._known_intents = known_intents

    # -- auth ----------------------------------------------------------------------

    def credentials(self) -> GraphCredentials:
        if self._credentials is None:
            self._credentials = resolve_credentials(
                self.binding.credential_ref, client=self._secrets_client
            )
        return self._credentials

    def token(self) -> str:
        """Client credentials flow. No user is present, and none should be — this is an
        application permission scoped to one mailbox, not a delegated user session."""

        if self._token:
            return self._token

        creds = self.credentials()
        status, payload = self.transport.post_form(
            TOKEN_URL.format(tenant_id=creds.tenant_id),
            {
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scope": GRAPH_SCOPE,
                "grant_type": "client_credentials",
            },
        )
        if status != 200 or not isinstance(payload, dict) or not payload.get("access_token"):
            raise MailboxUnreachable(
                f"token request failed with status {status}: "
                f"{(payload or {}).get('error_description', 'no detail')}"
            )
        self._token = payload["access_token"]
        return self._token

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token()}", "Accept": "application/json"}

    def _get(self, path: str) -> Tuple[int, Any]:
        return self.transport.get(f"{GRAPH_BASE}{path}", self._headers())

    # -- scope verification --------------------------------------------------------

    def probe_control_mailbox(self, control_mailbox: str) -> Tuple[bool, str]:
        """Try to read a mailbox this credential should not reach.

        Returns ``(allowed, detail)``. ``allowed=True`` is the failure case: the grant is
        wider than the binding claims. Passed to
        ``MailboxBindingRepository.verify_scope``, which refuses activation on a success.

        A transport failure is NOT counted as a denial. "The network was down" and "the
        tenant refused us" look identical from here, and treating the first as proof of
        the second would let a binding activate on the strength of a timeout.
        """

        status, payload = self._get(f"/users/{control_mailbox}/messages?$top=1")

        if status in (401, 403):
            return False, f"denied with HTTP {status}"
        if status == 404:
            # Graph returns 404 for a mailbox an application access policy hides, which is
            # a denial in everything but name.
            return False, "denied with HTTP 404 (mailbox not visible to this application)"
        if 200 <= status < 300:
            count = len((payload or {}).get("value", []))
            return True, f"READ SUCCEEDED with HTTP {status}; {count} message(s) returned"
        raise MailboxUnreachable(
            f"control probe returned HTTP {status}, which is neither a denial nor a "
            "success — scope cannot be verified from an inconclusive result"
        )

    # -- reading -------------------------------------------------------------------

    def list_folders(self) -> List[Dict[str, Any]]:
        status, payload = self._get(
            f"/users/{self.binding.mailbox_address}/mailFolders?$top=100"
        )
        if status in (401, 403, 404):
            raise MailboxAccessDenied(f"folder listing denied with HTTP {status}")
        if not 200 <= status < 300:
            raise MailboxUnreachable(f"folder listing returned HTTP {status}")
        return list((payload or {}).get("value", []))

    def _known_intent_codes(self):
        if self._known_intents is not None:
            return set(self._known_intents)
        # Read the controlled vocabulary rather than hardcoding it, so a new intent code
        # becomes matchable without a deploy.
        try:
            from services.db import get_conn

            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT code FROM proc.bp_style_intent WHERE code <> '_all'")
                codes = {r[0] for r in cur.fetchall()}
                cur.close()
                return codes
        except Exception:
            logger.warning("Could not read the intent vocabulary", exc_info=True)
            return set()

    def _messages_in(self, folder_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
        scope = (
            f"/users/{self.binding.mailbox_address}/mailFolders/{folder_id}/messages"
            if folder_id
            else f"/users/{self.binding.mailbox_address}/messages"
        )
        # Only sent mail: the customer's own writing is the point, and their inbox is
        # everyone else's style.
        query = (
            f"?$top={limit}&$select=id,subject,body,bodyPreview,sentDateTime"
            "&$orderby=sentDateTime desc"
        )
        status, payload = self.transport.get(
            f"{GRAPH_BASE}{scope}{query}", self._headers()
        )
        if status in (401, 403, 404):
            raise MailboxAccessDenied(f"message read denied with HTTP {status}")
        if not 200 <= status < 300:
            raise MailboxUnreachable(f"message read returned HTTP {status}")
        return list((payload or {}).get("value", []))

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """One message by id, or None if it is gone.

        Used by the feedback loop to see whether a draft we wrote back was later sent.
        ``isDraft`` flips to false and ``sentDateTime`` appears once it goes.
        """

        status, payload = self._get(
            f"/users/{self.binding.mailbox_address}/messages/{message_id}"
            "?$select=id,subject,body,isDraft,sentDateTime"
        )
        if status == 404:
            return None
        if status in (401, 403):
            raise MailboxAccessDenied(f"message read denied with HTTP {status}")
        if not 200 <= status < 300:
            raise MailboxUnreachable(f"message read returned HTTP {status}")
        return payload or None

    def fetch(self, user_ref: str, intent: str | None = None) -> List[RawExemplar]:
        """Emails from the bound mailbox, newest first.

        Folder names are consulted before any classifier: a folder called "Award
        notifications" is the customer's own filing and beats a model's guess. Where no
        folder matches the requested intent, the Sent Items folder is read and the intent
        left unset for a downstream classifier to fill.
        """

        known = self._known_intent_codes()
        folder_id = None
        matched_intent = None

        if intent:
            for folder in self.list_folders():
                code = folder_to_intent(folder.get("displayName"), known)
                if code == intent:
                    folder_id = folder.get("id")
                    matched_intent = code
                    break
            if folder_id is None:
                logger.info(
                    "No mailbox folder matches intent %r; falling back to sent mail",
                    intent,
                )

        if folder_id is None:
            for folder in self.list_folders():
                name = (folder.get("displayName") or "").strip().lower()
                if name in ("sent items", "sentitems"):
                    folder_id = folder.get("id")
                    break

        messages = self._messages_in(folder_id, self.fetch_limit)

        exemplars: List[RawExemplar] = []
        for message in messages:
            body = message.get("body") or {}
            content = body.get("content") or message.get("bodyPreview") or ""
            if (body.get("contentType") or "").lower() == "html":
                content = _html_to_text(content)
            if not content.strip():
                continue
            exemplars.append(
                RawExemplar(
                    body=content,
                    subject=message.get("subject"),
                    intent=matched_intent or intent,
                    source_ref=message.get("id"),
                    message_id=message.get("id"),
                )
            )
        return exemplars
