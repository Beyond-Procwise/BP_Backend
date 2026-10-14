"""How we reach a supplier, resolved from the supplier master and nowhere else.

There were four answers to "what address does this supplier have", and three of
them could be steered by data moving through a workflow:

  * ``email_dispatch_guard`` queried ``proc.bp_supplier`` by ``supplier_id`` --
    correct, and the only one that was.
  * ``EmailDraftingAgent._resolve_receiver`` took ``supplier["contact_email"]``,
    which ``SupplierRankingAgent`` had republished off the supplier frame.
  * ...then ``contact_email_1`` / ``_2`` off that same carried dict.
  * ...then ``profile["contacts"][*]["email"]`` from the supplier profile.

Policy #674 (``EmailRecipientAllowlistPolicy``) already states the rule the first
one follows: recipients come from ``bp_supplier.contact_email_1/2``, matched
exactly, and never from anything that arrived with the content. The other three
were a second route to the same fact, and a second route is the one nobody
re-checks.

So this is the one lookup, and the draft path and the send path both use it.
The send path re-checks at dispatch regardless -- that is not redundancy to be
optimised away, it is the guarantee. What changes is that a draft is now
addressed from the master in the first place, rather than being addressed from
a payload and caught later.

Resolving to nothing is a legitimate answer. It means the draft is held with no
recipient, which is the safe outcome; every caller must treat an empty result as
"we cannot address this", never as "fall back to what we were handed".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_SQL = (
    "SELECT contact_email_1, contact_email_2, contact_name_1 "
    "FROM proc.bp_supplier WHERE supplier_id = %s"
)


@dataclass(frozen=True)
class SupplierContact:
    """What the master holds for one supplier. Empty means we cannot reach them."""

    emails: List[str] = field(default_factory=list)
    name: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self.emails)


def _clean(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def resolve_contact(conn: Any, supplier_id: Optional[str]) -> SupplierContact:
    """The addresses and contact name on file for ``supplier_id``.

    ``conn`` may expose ``lookup_supplier_emails`` / ``lookup_supplier_name``,
    the seam the existing dispatch-guard tests already use; otherwise the
    supplier master is queried.

    Never raises. A lookup that fails resolves to nothing, because a send path
    that crashes on a database blip is worse than one that declines to address
    a message.
    """

    if hasattr(conn, "lookup_supplier_emails"):
        emails = [e for e in (_clean(v) for v in (conn.lookup_supplier_emails(supplier_id) or [])) if e]
        name = None
        if hasattr(conn, "lookup_supplier_name"):
            name = _clean(conn.lookup_supplier_name(supplier_id))
        return SupplierContact(emails=emails, name=name)

    if not supplier_id:
        return SupplierContact()

    try:
        cur = conn.cursor()
        cur.execute(_SQL, (supplier_id,))
        rows = cur.fetchall() or []
    except Exception:  # noqa: BLE001 - resolved to "cannot address" by every caller
        logger.exception("supplier_contact: could not resolve %s", supplier_id)
        return SupplierContact()

    emails: List[str] = []
    name: Optional[str] = None
    for row in rows:
        values = list(row)
        for value in values[:2]:
            cleaned = _clean(value)
            if cleaned and cleaned.lower() not in {e.lower() for e in emails}:
                emails.append(cleaned)
        if name is None and len(values) > 2:
            name = _clean(values[2])

    return SupplierContact(emails=emails, name=name)


def resolve_emails(conn: Any, supplier_id: Optional[str]) -> List[str]:
    """Just the addresses. The shape ``email_dispatch_guard`` already consumed."""

    return resolve_contact(conn, supplier_id).emails
