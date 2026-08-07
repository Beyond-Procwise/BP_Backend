"""Hash what an approver saw, so an edited draft cannot ride an old approval.

One helper, called by both the approval endpoint and the send path. They must
never compute this differently -- if they drift, an approval starts covering
content nobody approved, which is the whole failure this exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _attachment_identities(draft: Dict[str, Any]) -> list:
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
    return sorted(out)


def content_hash(draft: Dict[str, Any]) -> str:
    """A stable digest of the recipients, subject, body and attachments.

    Recipients come from ``resolve_recipients``, which is what the send path
    actually sends to -- ``proc.draft_rfq_emails`` stores ``recipient_email``
    (singular) plus a payload blob, so reading a column directly would let the
    approved set and the sent set diverge.

    Never raises: an unhashable draft yields a digest of what could be read,
    and the comparison then simply fails to match, which denies.
    """

    try:
        from src.services.email_dispatch_guard import resolve_recipients

        recipients = sorted(r.casefold() for r in resolve_recipients(draft, None))
    except Exception as exc:  # noqa: BLE001 - a hash that cannot be computed must not crash a send
        logger.error("approval_content: recipient resolution failed: %s", exc)
        recipients = []

    payload = {
        "recipients": recipients,
        "subject": str(draft.get("subject") or "").strip(),
        "body": str(draft.get("body") or "").strip(),
        "attachments": _attachment_identities(draft),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
