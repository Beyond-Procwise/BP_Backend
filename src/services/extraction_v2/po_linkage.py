"""Cross-document PO-linkage validator.

When an invoice carries a ``po_id``, fetch the referenced PO and assert:

  - The PO exists in ``proc.bp_purchase_order``.
  - For each invoice line, the cumulative billed quantity per item does
    not exceed the PO's authorized quantity by more than ``QTY_TOL``.

Over-billing is flagged ``critical`` — financial controls should not
let an invoice for more than the PO authorizes pass without review.

The validator implements the same :class:`Validator` interface as the
arithmetic invariants but is kept in a separate module because it is
the only one that does DB I/O. Production code injects the connection
factory; tests use a fake.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from src.services.extraction_v2.invariants import (
    Severity, Validator, ValidatorResult, _f, _line_qty,
)

logger = logging.getLogger(__name__)


__all__ = ["PoLinkage", "build_default_po_linkage"]


# Per-item cumulative-quantity tolerance: invoices may legitimately bill
# fractionally over (rounding, partial deliveries) — accept up to 1% or
# 1 unit, whichever is larger.
def _within_tol(billed: float, authorized: float) -> bool:
    if authorized <= 0:
        return billed <= 1.0
    abs_tol = max(1.0, 0.01 * authorized)
    return billed <= authorized + abs_tol


class PoLinkage(Validator):
    """Validates an invoice's ``po_id`` against ``proc.bp_purchase_order``."""
    name = "po_linkage"

    def __init__(self, get_conn: Optional[Callable] = None,
                 *, fetch_fn: Optional[Callable] = None):
        """Either ``get_conn`` (Postgres connection factory) OR
        ``fetch_fn(po_id) -> {item_id: authorized_qty}`` for tests.
        """
        self._get_conn = get_conn
        self._fetch_fn = fetch_fn

    def applicable(self, doc_type: str) -> bool:
        return doc_type == "Invoice"

    def _fetch_po_quantities(self, po_id: str) -> Optional[dict]:
        if self._fetch_fn is not None:
            return self._fetch_fn(po_id)
        if self._get_conn is None:
            return None
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    # Confirm PO exists at all
                    cur.execute(
                        "SELECT 1 FROM proc.bp_purchase_order WHERE po_id = %s",
                        (po_id,),
                    )
                    if cur.fetchone() is None:
                        return {}
                    cur.execute(
                        """SELECT item_id, COALESCE(quantity, 0)::float
                             FROM proc.bp_po_line_items
                            WHERE po_id = %s""",
                        (po_id,),
                    )
                    return {
                        (str(r[0]) if r[0] else ""): float(r[1] or 0)
                        for r in cur.fetchall() if r[0]
                    }
        except Exception as exc:
            logger.debug("PO linkage fetch failed for po_id=%s: %s", po_id, exc)
            return None

    def check(self, header, line_items, doc_type) -> ValidatorResult:
        po_id = (header.get("po_id") or "").strip()
        if not po_id:
            return ValidatorResult.na(self.name)
        if not line_items:
            return ValidatorResult.na(self.name)

        authorized = self._fetch_po_quantities(po_id)
        if authorized is None:
            # DB unavailable — don't fail the doc on infra; downgrade.
            return ValidatorResult(
                name=self.name, passed=True, severity=Severity.INFO,
                message=f"po_lookup_skipped (po_id={po_id})",
            )
        if authorized == {}:
            return ValidatorResult.fail(
                self.name,
                f"po_id={po_id!r} not found in proc.bp_purchase_order",
                severity=Severity.WARNING,
                fields=("po_id",),
            )

        # Aggregate billed quantity by item_id across the invoice's lines.
        billed: dict[str, float] = {}
        for item in line_items:
            iid = str(item.get("item_id") or item.get("item_description") or "").strip()
            qty = _line_qty(item) or 0.0
            if not iid:
                continue
            billed[iid] = billed.get(iid, 0.0) + qty

        violations: list[str] = []
        for iid, billed_qty in billed.items():
            auth_qty = authorized.get(iid)
            if auth_qty is None:
                # Item billed but not on the PO at all — over-billing.
                violations.append(f"{iid!r} billed={billed_qty} not_on_PO")
                continue
            if not _within_tol(billed_qty, auth_qty):
                violations.append(
                    f"{iid!r} billed={billed_qty} > authorized={auth_qty}"
                )

        if not violations:
            return ValidatorResult.ok(
                self.name, fields=("po_id", "line_items.item_id"),
            )
        return ValidatorResult.fail(
            self.name,
            f"over-billing vs PO {po_id}: " + "; ".join(violations[:5]),
            severity=Severity.CRITICAL,
            residual=float(len(violations)),
            fields=("po_id", "line_items.item_id", "line_items.quantity"),
        )


def build_default_po_linkage() -> PoLinkage:
    """Production constructor — wires the standard get_conn factory."""
    from services.db import get_conn
    return PoLinkage(get_conn=get_conn)
