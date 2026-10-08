"""Which suppliers could actually supply what the buyer asked for.

The ranking agent scores suppliers on delivery and risk. On its own that
shortlists whoever delivered anything well, whatever they sold: a laptop
requirement came back with logistics firms. A supplier is a candidate for an
item only when its own purchase, invoice or quote lines mention that item.

The supplier master carries no category, so line descriptions are the evidence.
Nothing here invents a category; a title that yields no usable term yields no
filter, and a term nobody has sold yields an empty set the caller must report.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Set

_BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_WORD = re.compile(r"[A-Za-z]{3,}")

# Header table per line table; the supplier lives on the header.
_SQL = """
SELECT p.supplier_id FROM proc.bp_po_line_items_trgt l
  JOIN proc.bp_purchase_order_trgt p ON p.po_id = l.po_id
 WHERE l.item_description ILIKE %s
UNION
SELECT i.supplier_id FROM proc.bp_invoice_line_items_trgt l
  JOIN proc.bp_invoice_trgt i ON i.invoice_id = l.invoice_id
 WHERE l.item_description ILIKE %s
UNION
SELECT q.supplier_id FROM proc.bp_quote_line_items_trgt l
  JOIN proc.bp_quote_trgt q ON q.quote_id = l.quote_id
 WHERE l.item_description ILIKE %s
"""


def product_term(requirement: Any) -> Optional[str]:
    """The item the buyer wants, as one searchable word: the head noun of the
    title ("Business Laptops (14-inch ...)" -> "laptop"). None when the title
    gives nothing usable."""
    if not isinstance(requirement, dict):
        return None
    title = _BRACKETS.sub(" ", str(requirement.get("title") or ""))
    words = _WORD.findall(title)
    if not words:
        return None
    head = words[-1].lower()
    if head.endswith("ies") and len(head) > 4:
        head = head[:-3] + "y"
    elif head.endswith("s") and not head.endswith("ss") and len(head) > 3:
        head = head[:-1]
    return head if len(head) >= 4 else None


def suppliers_selling(conn: Any, term: str) -> Set[str]:
    """Ids of suppliers with a PO, invoice or quote line mentioning ``term``."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"
    with conn.cursor() as cur:
        cur.execute(_SQL, (pattern, pattern, pattern))
        rows = cur.fetchall()
    return {str(r[0]).strip() for r in rows if r and r[0] is not None and str(r[0]).strip()}
