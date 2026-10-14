"""Turn a proc.draft_rfq_emails row into one draft shape, everywhere.

C2 happened because the approval endpoint and the pending list read raw
columns while the send path (``EmailDispatchService._hydrate_draft``) started
from the JSON ``payload`` blob and only fell back to columns via
``setdefault``. Payload wins for ``subject``/``body``/``recipients``/
``attachments`` on the send side; the approval side never applied that same
rule, so the two sides hashed different content for the identical row and a
genuinely-approved draft could never pass its own content-binding check.

This module is the one place that hydration happens now. It is a leaf module
(no imports of ``email_dispatch_service``, ``approval_store`` or the
``approvals`` router) so every one of those can import it without creating a
cycle -- ``email_dispatch_guard`` already imports ``approval_store`` at
module scope, so a hydration helper living in either of those would trap the
others in a cycle the moment they needed it too.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable as IterableABC
from typing import Any, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

# The exact column order EmailDispatchService._fetch_latest_draft selects in
# (see email_dispatch_service.py). Both approvals.py's _load_draft and
# approval_store.py's list_pending_dispatch_approvals must select this same
# set (order does not matter for them, since they use a dict cursor) so
# hydrate_draft has every column it needs -- a partial SELECT silently
# hydrates an incomplete draft, which is exactly how C2 happened.
DRAFT_COLUMNS = (
    "id",
    "rfq_id",
    "supplier_id",
    "supplier_name",
    "subject",
    "body",
    "sent",
    "recipient_email",
    "contact_level",
    "thread_index",
    "payload",
    "sender",
    "sent_on",
    "workflow_id",
    "run_id",
    "unique_id",
    "mailbox",
    # Who asked for this draft, when a person did (P3). The approvals surface
    # compares it with the approver to refuse self-approval, so it has to
    # survive the hydration that every read of a draft goes through.
    "requested_by",
    "dispatch_run_id",
    "dispatched_at",
    "attachments",
)


def normalise_recipients(recipients: Optional[Iterable[Any]]) -> list:
    """Dedupe case-insensitively, keeping the first-seen casing and order."""

    if recipients is None:
        return []
    if isinstance(recipients, str):
        items: Iterable[Any] = [recipients]
    else:
        items = recipients

    out: list = []
    seen_lower: set = set()
    for value in items:
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen_lower:
            continue
        seen_lower.add(lowered)
        out.append(candidate)
    return out


def hydrate_draft(row: Mapping[str, Any], *, default_sender: Optional[str] = None) -> Dict[str, Any]:
    """The same draft dict ``EmailDispatchService._hydrate_draft`` builds.

    ``row`` is a mapping keyed by :data:`DRAFT_COLUMNS` (a ``RealDictCursor``
    row, or any dict with those keys). Payload wins for keys it carries;
    columns fill in only what payload does not -- that precedence, applied
    consistently, is the whole fix for C2.
    """

    payload = row.get("payload")
    hydrated: Dict[str, Any]
    if isinstance(payload, dict):
        hydrated = dict(payload)
    else:
        try:
            hydrated = json.loads(payload) if payload else {}
        except Exception:  # noqa: BLE001 - a corrupt payload hydrates as empty, not a crash
            hydrated = {}

    recipient_email = row.get("recipient_email")
    sent_on = row.get("sent_on")

    defaults = {
        "id": row.get("id"),
        "rfq_id": row.get("rfq_id"),
        "supplier_id": row.get("supplier_id"),
        "supplier_name": row.get("supplier_name"),
        "subject": row.get("subject"),
        "body": row.get("body"),
        "sent_status": bool(row.get("sent")),
        "receiver": recipient_email,
        "contact_level": row.get("contact_level"),
        "thread_index": row.get("thread_index"),
        "sender": row.get("sender"),
        "recipients": hydrated.get("recipients") or ([recipient_email] if recipient_email else []),
        "workflow_id": row.get("workflow_id"),
        "run_id": row.get("run_id"),
        "unique_id": row.get("unique_id"),
        "mailbox": row.get("mailbox"),
        "dispatch_run_id": row.get("dispatch_run_id"),
        "dispatched_at": row.get("dispatched_at"),
        "attachments": row.get("attachments"),
    }
    for key, value in defaults.items():
        hydrated.setdefault(key, value)

    # The ONE field where the column beats the payload, and deliberately so.
    # Everything above is payload-over-columns because the send path transmits
    # the payload (C2). `requested_by` is not transmitted — it decides an
    # authorization outcome, whether the approver is the person who asked — so
    # it is read from the column the migration guarantees. A payload claiming a
    # different requester would otherwise be able to talk its way past the
    # self-approval bar by naming somebody else.
    hydrated["requested_by"] = row.get("requested_by")

    if sent_on and "sent_on" not in hydrated:
        hydrated["sent_on"] = (
            sent_on if isinstance(sent_on, str) else getattr(sent_on, "isoformat", lambda: sent_on)()
        )

    recipients_value = hydrated.get("recipients")
    if isinstance(recipients_value, str):
        hydrated["recipients"] = normalise_recipients([recipients_value])
    elif isinstance(recipients_value, IterableABC):
        hydrated["recipients"] = normalise_recipients(recipients_value)
    else:
        hydrated["recipients"] = []

    if not hydrated.get("sender") and default_sender:
        hydrated["sender"] = default_sender

    return hydrated


def resolve_effective_content(
    draft: Mapping[str, Any],
    *,
    recipients: Optional[Iterable[Any]] = None,
    subject_override: Optional[str] = None,
    body_override: Optional[str] = None,
    attachments_override: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """What ``EmailDispatchService.send_draft`` would actually transmit for
    ``draft``, given these (optional) overrides.

    C1 was hashing the *stored* draft while the send path could transmit a
    caller-supplied ``subject_override``/``body_override`` straight from an
    HTTP request -- an approved draft's hash never moved, so an edited body
    sailed through under someone else's approval. The fix is to hash this
    resolved material instead, on both sides:

    * The send path calls this with the request's real overrides.
    * The approval endpoint and the pending list call it with none, which
      must compute to the *same* thing the send path would compute for a
      send with no overrides right now -- otherwise C1's fix reintroduces
      C2's mismatch from the other direction. That is why this one function
      is shared rather than each side approximating the same arithmetic.
    """

    from src.services.email_dispatch_guard import resolve_recipients

    resolved_recipients = resolve_recipients(dict(draft), recipients)

    if subject_override is not None:
        subject_candidate = subject_override
    else:
        subject_candidate = draft.get("subject")
    subject_str = str(subject_candidate).strip() if subject_candidate else ""
    unique_id = draft.get("unique_id")
    # Mirrors EmailDispatchService.send_draft's own fallback verbatim: a
    # draft with a genuinely empty subject is sent under a synthetic one, so
    # the hash of "what will be sent" must reflect that too, not the blank
    # stored value.
    subject = subject_str or (f"{unique_id} – Request for Quotation" if unique_id else "")

    body_source = body_override if body_override is not None else draft.get("body")
    body = str(body_source).strip() if body_source else ""

    attachments = (
        attachments_override if attachments_override is not None else draft.get("attachments")
    )

    return {
        "recipients": resolved_recipients,
        "subject": subject,
        "body": body,
        "attachments": attachments,
    }
