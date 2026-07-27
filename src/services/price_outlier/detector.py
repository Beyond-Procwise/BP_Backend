"""Find extreme prices across the target tables and describe them.

Detection is separated from the benchmark API deliberately: a GET request must
never write findings. This runs as a scheduled job.

The peer set uses exactly the match rule the benchmark engine uses — same item,
unit and currency, normalised the same way — so a flag and a benchmark can never
disagree about what counts as comparable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from services.benchmark_live import _norm_currency, _norm_item, _norm_uom
from services.price_outlier.rule import OutlierSettings, Verdict, assess

logger = logging.getLogger(__name__)

Key = tuple[str, str, str]

# Which tables are scanned, and how each names its columns. Invoice lines use
# line_no where the others use line_number.
_SOURCES: tuple[dict[str, str], ...] = (
    {"doc_type": "quote", "table": "proc.bp_quote_line_items_trgt",
     "doc_pk": "quote_id", "line_no": "line_number"},
    {"doc_type": "purchase_order", "table": "proc.bp_po_line_items_trgt",
     "doc_pk": "po_id", "line_no": "line_number"},
    {"doc_type": "invoice", "table": "proc.bp_invoice_line_items_trgt",
     "doc_pk": "invoice_id", "line_no": "line_no"},
)

# po_id and invoice_id are NOT disjoint namespaces (live bp_sqldb has a PO and
# an invoice both numbered "123456/22"), so a raw doc_pk cannot be used to
# exclude "the line's own document" from its peer set -- it would also
# exclude an unrelated document of the other type that happens to share the
# id, over-excluding real comparison evidence. The pool's point_id already
# carries the type prefix ("po:"/"inv:"); qualifying every document identity
# with that same prefix keeps the two namespaces apart everywhere they meet.
# Quote lines are never IN the pool (load_benchmark_pool reads only PO and
# invoice lines), so a quote line has no prefix to qualify with and excludes
# nothing from its peer set -- deliberately, not by omission.
_POOL_PREFIX_BY_DOC_TYPE: dict[str, str] = {
    "purchase_order": "po",
    "invoice": "inv",
}


def _pool_prefix(point_id: str) -> str:
    """Reuse the type prefix point_id already carries, rather than re-deriving
    which pool query a row came from some other way."""
    return point_id.split(":", 1)[0]


@dataclass(frozen=True)
class Finding:
    doc_type: str
    doc_pk: str
    line_number: int
    field_name: str
    item_description: str
    price: float
    verdict: Verdict
    note: str


def build_peer_index(
    pool_rows: Sequence[dict[str, Any]],
) -> dict[Key, list[tuple[Optional[str], float]]]:
    """Group the pool once, by the engine's match key.

    Indexed rather than rescanned: ~190,000 lines against a ~76,000-row pool is
    14 billion comparisons if every line filters the whole list, and about
    76,000 if the pool is grouped once up front.
    """
    index: dict[Key, list[tuple[Optional[str], float]]] = {}
    for row in pool_rows:
        key = (
            _norm_item(row["item_description"]),
            _norm_uom(row["unit_of_measure"]),
            _norm_currency(row["currency"]),
        )
        qualified_doc = f"{_pool_prefix(row['point_id'])}:{row.get('doc_id')}"
        index.setdefault(key, []).append(
            (qualified_doc, float(row["unit_price"])))
    return index


def peers_for(
    index: dict[Key, list[tuple[Optional[str], float]]], key: Key,
    own_doc: Optional[str],
) -> list[float]:
    """Comparable prices for one match key, excluding the line's own document."""
    return [price for doc, price in index.get(key, ()) if doc != own_doc]


def describe(line_number: int, item: str, price: float, verdict: Verdict) -> str:
    """The sentence a reviewer reads. Names the comparison, not the statistics.

    A bare "÷" is not a word a reviewer reads as English, and a "/" is rewritten
    by the output-safety gate downstream because it resembles a URL route -- so
    the direction is spelled out (ABOVE/BELOW) instead of encoded as a symbol.
    """
    above = verdict.ratio >= 1
    factor = verdict.ratio if above else 1.0 / verdict.ratio
    direction = "ABOVE" if above else "BELOW"
    return (
        f"line {line_number}: '{item}' at {price:,.2f} each is "
        f"{factor:,.1f}× {direction} the usual {verdict.median:,.2f} across "
        f"{verdict.peer_count} comparable purchases — "
        f"check the unit price and quantity"
    )


def _load_pool(cur) -> list[dict[str, Any]]:
    from services.benchmark_live import load_benchmark_pool

    return load_benchmark_pool(cur)


def _load_lines(cur, source: dict[str, str]) -> list[dict[str, Any]]:
    cur.execute(
        f"""
        SELECT {source['doc_pk']} AS doc_pk, {source['line_no']} AS line_number,
               item_description, unit_of_measure, unit_price
          FROM {source['table']}
         WHERE unit_price IS NOT NULL AND item_description IS NOT NULL
           AND {source['doc_pk']} IS NOT NULL
        """
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _line_currency(cur, source: dict[str, str]) -> dict[str, str]:
    """Header currency per document, since line currency is unreliable."""
    header = {
        "quote": ("proc.bp_quote_trgt", "quote_id"),
        "purchase_order": ("proc.bp_purchase_order_trgt", "po_id"),
        "invoice": ("proc.bp_invoice_trgt", "invoice_id"),
    }[source["doc_type"]]
    cur.execute(f"SELECT {header[1]}, currency FROM {header[0]}")
    return {row[0]: row[1] for row in cur.fetchall()}


def find_outliers(cur, settings: Optional[OutlierSettings] = None) -> list[Finding]:
    """Every line whose price is extreme against its comparable purchases."""
    settings = settings if settings is not None else OutlierSettings()
    index = build_peer_index(_load_pool(cur))
    findings: list[Finding] = []

    for source in _SOURCES:
        currency_by_doc = _line_currency(cur, source)
        pool_prefix = _POOL_PREFIX_BY_DOC_TYPE.get(source["doc_type"])
        for line in _load_lines(cur, source):
            key = (
                _norm_item(line["item_description"]),
                _norm_uom(line["unit_of_measure"]),
                _norm_currency(currency_by_doc.get(line["doc_pk"])),
            )
            # None for a quote line: quotes are never in the pool, so there is
            # no qualified id that could ever collide -- nothing is excluded.
            own_doc = (
                f"{pool_prefix}:{line['doc_pk']}" if pool_prefix is not None
                else None
            )
            peers = peers_for(index, key, own_doc=own_doc)
            verdict = assess(float(line["unit_price"]), peers, settings)
            if not verdict.flagged:
                continue
            findings.append(
                Finding(
                    doc_type=source["doc_type"],
                    doc_pk=str(line["doc_pk"]),
                    line_number=int(line["line_number"] or 0),
                    field_name=f"line_items[{line['line_number']}].unit_price",
                    item_description=line["item_description"],
                    price=float(line["unit_price"]),
                    verdict=verdict,
                    note=describe(
                        int(line["line_number"] or 0), line["item_description"],
                        float(line["unit_price"]), verdict),
                )
            )

    logger.info("price outlier scan produced %d findings", len(findings))
    return findings
