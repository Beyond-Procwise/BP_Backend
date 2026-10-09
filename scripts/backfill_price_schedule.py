"""Give documents already read their price schedule, pricing terms and escalation findings.

deploy/sql/2026-10-09_price_schedule.sql added `price_schedule` to the line tables; extraction now
reads it (dispatch -> price_schedule.py). This applies the SAME functions to each quote, PO and
invoice's latest stored read (its parser_snapshot.full_text), so nothing is re-extracted and no
printed figure is touched:
  - parser_snapshot.pricing_terms on that raw row (term, stated uplift, index, term total);
  - price_schedule on the lines it names: the raw lines of that read, and the document's _stg and
    _trgt lines (matched by description), where still NULL;
  - the escalation findings, through persistence.write_discrepancies (an upsert per document, so a
    re-run refreshes rather than stacks).

    ./venv/bin/python scripts/backfill_price_schedule.py            # dry run
    ./venv/bin/python scripts/backfill_price_schedule.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.db import get_conn  # noqa: E402
from src.services.extraction import persistence  # noqa: E402
from src.services.extraction import price_schedule as ps  # noqa: E402
from src.services.extraction.persistence import Discrepancy  # noqa: E402
from src.services.governed_limits import limit  # noqa: E402

# doc_type -> (raw header table, raw lines, stg lines, trgt lines, doc key on stg/trgt lines)
DOCS = {
    "quote": ("proc.bp_quote_raw", "proc.bp_quote_line_items_raw", "proc.bp_quote_line_items_stg",
              "proc.bp_quote_line_items_trgt", "quote_id"),
    "purchase_order": ("proc.bp_purchase_order_raw", "proc.bp_po_line_items_raw", "proc.bp_po_line_items_stg",
                       "proc.bp_po_line_items_trgt", "po_id"),
    "invoice": ("proc.bp_invoice_raw", "proc.bp_invoice_line_items_raw", "proc.bp_invoice_line_items_stg",
                "proc.bp_invoice_line_items_trgt", "invoice_id"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    tol = limit("reconciliation_tolerances", "uplift_tolerance_pp")
    n_docs = n_lines = n_find = 0
    with get_conn() as conn:
        cur = conn.cursor()
        for doc_type, (raw_t, raw_l, stg_l, trgt_l, key) in DOCS.items():
            cur.execute(f"""SELECT DISTINCT ON (doc_pk_candidate) raw_id, doc_pk_candidate, source_file,
                                   parser_snapshot->>'full_text'
                              FROM {raw_t}
                             WHERE parser_snapshot ? 'full_text' AND doc_pk_candidate IS NOT NULL
                             ORDER BY doc_pk_candidate, raw_id DESC""")
            for raw_id, pk, src, text in cur.fetchall():
                sched, terms = ps.schedules(text), ps.pricing_terms(text)
                lines = sched["lines"]
                if not (lines or terms.get("uplift_pct") is not None or terms.get("indexation")):
                    continue
                n_docs += 1
                findings = ps.escalation_findings(lines, terms, tol) if lines else []
                n_find += len(findings)
                print(f"{doc_type} {pk}: {len(lines)} scheduled lines, uplift {terms.get('uplift_pct')}"
                      f"{' / ' + terms['indexation'] if terms.get('indexation') else ''}"
                      + "".join(f"\n    {ps.finding_notes(f)}" for f in findings))
                if not args.apply:
                    n_lines += len(lines)
                    continue
                snap = {**terms, "stated_tcv": sched["stated_tcv"],
                        "term_total": round(sum(l["term_total"] for l in lines), 2) if lines else None,
                        "periods": max((len(l["periods"]) for l in lines), default=0)}
                cur.execute(f"UPDATE {raw_t} SET parser_snapshot = parser_snapshot || jsonb_build_object('pricing_terms', %s::jsonb) "
                            f"WHERE raw_id = %s", (json.dumps(snap), raw_id))
                for li in lines:
                    val = json.dumps({"periods": li["periods"], "term_total": li["term_total"]})
                    d = li["description"]
                    cur.execute(f"UPDATE {raw_l} SET price_schedule = %s WHERE raw_id = %s AND price_schedule IS NULL "
                                f"AND lower(regexp_replace(trim(item_description), '\\s+', ' ', 'g')) = lower(regexp_replace(trim(%s), '\\s+', ' ', 'g'))",
                                (val, raw_id, d))
                    n_lines += cur.rowcount
                    for t in (stg_l, trgt_l):
                        cur.execute(f"UPDATE {t} SET price_schedule = %s WHERE {key} = %s AND price_schedule IS NULL "
                                    f"AND lower(regexp_replace(trim(item_description), '\\s+', ' ', 'g')) = lower(regexp_replace(trim(%s), '\\s+', ' ', 'g'))",
                                    (val, pk, d))
                if findings:
                    persistence.write_discrepancies(
                        doc_type=doc_type, raw_id=raw_id, source_file=src, doc_pk_candidate=pk,
                        discrepancies=[Discrepancy(
                            field_name="price_schedule", issue_type=f["issue_type"], severity="warning",
                            blocks_promotion=False,
                            raw_value=f"{max(x['actual_pct'] for x in f['lines']):.2f}%",
                            expected_value=(f"{f['stated_pct']:g}%" if f["stated_pct"] is not None else None),
                            computed_value=f"{f['extra']:.2f}", notes=ps.finding_notes(f)) for f in findings])
    print(f"{'applied' if args.apply else 'dry run'}: {n_docs} documents, "
          f"{n_lines} {'raw lines given a schedule' if args.apply else 'scheduled lines'}, {n_find} findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
