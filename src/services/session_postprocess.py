"""Session-completion housekeeping.

Runs after a session's fast promotion + deal-linking pass:

1. **Stale-PO reconcile** — `po_not_found` / `po_pending_review` discrepancies
   whose cited purchase order has since reached `_stg`/`_trgt` are resolved.
   Within one upload batch the invoice is often dispatched before its own PO,
   and until 2026-07-30 the resulting "PO does not exist" flag stayed open
   forever even though the PO was two rows away (ses-20260730-UJF3).

2. **Deal proposal generation** — every un-established batch label in the
   session is clustered into proposed deals (spec §Upload path change). The
   batch label itself is never a deal; a human confirms proposals, and only
   that confirm mints deal identity. Established labels (amend-mode: the user
   explicitly targeted a real deal) are left alone.

Both steps are best-effort: failures log and leave the scheduled sweep to
catch up, exactly like the promotion steps before them.
"""
from __future__ import annotations

import logging

from src.services.db import get_conn
from src.services.deal_assignment_service import is_established_deal
from src.services.linking_engine import _PO_NORM_SQL

log = logging.getLogger(__name__)

_NORM_D = _PO_NORM_SQL.format(col="d.raw_value")
_NORM_STG = _PO_NORM_SQL.format(col="p.po_id")
_NORM_TRGT = _PO_NORM_SQL.format(col="t.po_id")

_RECONCILE_SQL = f"""
UPDATE proc.bp_extraction_discrepancy d
   SET status = 'resolved',
       resolved_at = now(),
       resolution_action = 'dismiss',
       resolved_by = 'session_postprocess',
       notes = coalesce(d.notes, '')
               || ' [auto-resolved: the cited purchase order has since reached the system]'
 WHERE d.issue_type IN ('po_not_found', 'po_pending_review')
   AND coalesce(d.status, 'open') <> 'resolved'
   AND d.raw_value IS NOT NULL
   AND (EXISTS (SELECT 1 FROM proc.bp_purchase_order_stg p
                 WHERE {_NORM_STG} = {_NORM_D})
     OR EXISTS (SELECT 1 FROM proc.bp_purchase_order_trgt t
                 WHERE {_NORM_TRGT} = {_NORM_D}))
"""


def reconcile_po_discrepancies(cur) -> int:
    """Resolve open PO-citation discrepancies whose PO now exists. Returns
    the number of rows resolved."""
    cur.execute(_RECONCILE_SQL)
    return cur.rowcount or 0


def _generate_proposals(batch_deal_id: str, session_id: str) -> dict:
    """Cluster one batch into proposals (module-level so tests can stub it)."""
    from src.api.routers.deal_proposals import _generate
    return _generate(batch_deal_id, session_id)


def postprocess_session(session_id: str) -> dict:
    out: dict = {"po_discrepancies_resolved": 0, "proposals": {}}
    labels: list[str] = []
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            out["po_discrepancies_resolved"] = reconcile_po_discrepancies(cur)
            conn.commit()
        except Exception:  # noqa: BLE001
            log.exception("stale-PO reconcile failed for session %s", session_id)
            conn.rollback()
        cur.execute(
            "select distinct deal_id from proc.process_monitor "
            "where session_id = %s and coalesce(deal_id, '') <> ''",
            (session_id,))
        candidates = [r[0] for r in cur.fetchall()]
        labels = [l for l in candidates if not is_established_deal(cur, l)]

    for label in labels:
        try:
            r = _generate_proposals(label, session_id)
            out["proposals"][label] = {
                "proposal_ids": r.get("proposal_ids", []),
                "ungrouped": r.get("ungrouped", []),
            }
        except Exception as exc:  # noqa: BLE001
            log.exception("proposal generation failed for batch %s", label)
            out["proposals"][label] = {"error": str(exc)}
    return out
