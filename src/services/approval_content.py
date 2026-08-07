"""Hash what an approver saw, so an edited draft cannot ride an old approval.

One helper, called by both the approval endpoint and the send path. They must
never compute this differently -- if they drift, an approval starts covering
content nobody approved, which is the whole failure this exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _normalised(draft: Mapping) -> Dict[str, Any]:
    """Bridge the stored column names to what resolve_recipients reads.

    proc.draft_rfq_emails stores recipient_email (singular); resolve_recipients
    reads recipients/receiver. Every caller hashing a raw database row would
    otherwise resolve to an empty recipient set -- and since the approval and
    the send hash in different places, a mapping applied at only some call
    sites makes the two disagree silently. It belongs here, once.
    """

    out = dict(draft)
    # Presence, not truthiness: a caller that resolved recipients and got none
    # passes recipients=[], which is a real answer. Overriding it with a column
    # the resolver ignores would hash against a recipient the send path will
    # never use -- the divergence this helper exists to prevent.
    if "recipients" not in out and "receiver" not in out:
        single = out.get("recipient_email")
        if single:
            out["receiver"] = single
    return out


def _attachment_identities(draft: Mapping[str, Any]) -> list:
    """Attachment names, sorted. Content is not hashed -- the identity of what
    was attached is what an approver actually reviewed."""

    out = []
    raw = draft.get("attachments")
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("filename") or item.get("name")
            else:
                name = item
            text = str(name or "").strip()
            if text:
                out.append(text)
    elif raw is not None:
        # Attachments present but not in expected shape
        logger.warning(
            "approval_content: attachments is %s, not list; treating as empty",
            type(raw).__name__,
        )
    return sorted(out)


def content_hash(draft: Any) -> str:
    """A stable digest of the recipients, subject, body and attachments.

    Recipients come from ``resolve_recipients``, which is what the send path
    actually sends to -- ``proc.draft_rfq_emails`` stores ``recipient_email``
    (singular) plus a payload blob, so reading a column directly would let the
    approved set and the sent set diverge.

    Never raises. A draft that cannot be read hashes as empty, so the
    comparison fails to match and the send is denied. Crashing here would
    take down a send that should merely have been refused.
    """

    if not isinstance(draft, Mapping):
        logger.error(
            "approval_content: draft is %s, not a mapping; hashing as empty",
            type(draft).__name__,
        )
        draft = {}

    draft = _normalised(draft)

    try:
        from src.services.email_dispatch_guard import resolve_recipients

        recipients = sorted(r.casefold() for r in resolve_recipients(draft, None))
        subject = str(draft.get("subject") or "").strip()
        body = str(draft.get("body") or "").strip()
        attachments = _attachment_identities(draft)
    except Exception as exc:  # noqa: BLE001 - a hash that cannot be computed must not crash a send
        logger.error("approval_content: hashing failed: %s", exc)
        recipients = []
        subject = ""
        body = ""
        attachments = []

    payload = {
        "recipients": recipients,
        "subject": subject,
        "body": body,
        "attachments": attachments,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
