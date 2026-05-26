"""LIVE derived-column computation audit for the _stg tables.

For every row currently in proc.bp_{invoice,purchase_order,quote}_stg (and their
line items), recompute each DERIVED column from its source columns and compare to
the stored value. Surfaces any column the pipeline failed to compute or computed
wrong, plus columns that SHOULD exist for a doc type but are missing from the
schema (e.g. quote USD conversion).

Read-only. Run: .venv/bin/python scripts/stg_computation_audit.py
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.db import get_conn  # noqa: E402
from src.services.extraction.context_layer import _FX_TO_USD  # noqa: E402

# doc_type -> (stg table, pk, subtotal col, total_incl col, line table, line fk, line amount col)
SPEC = {
    "invoice": ("bp_invoice_stg", "invoice_id", "invoice_amount", "invoice_total_incl_tax",
                "bp_invoice_line_items_stg", "invoice_id", "line_amount"),
    "purchase_order": ("bp_purchase_order_stg", "po_id", "total_amount", "total_amount_incl_tax",
                       "bp_po_line_items_stg", "po_id", "line_total"),
    "quote": ("bp_quote_stg", "quote_id", "total_amount", "total_amount_incl_tax",
              "bp_quote_line_items_stg", "quote_id", "line_total"),
}
TOL = 0.02  # currency rounding tolerance


def f(v):
    if v is None or v == "":
        return None
    try:
        return float(Decimal(str(v)))
    except Exception:  # noqa: BLE001
        return None


def near(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(0.01, tol * max(1.0, abs(b)))


def cols_of(cur, table):
    cur.execute(
        "select column_name from information_schema.columns "
        "where table_schema='proc' and table_name=%s", (table,))
    return {r[0] for r in cur.fetchall()}


def main() -> int:
    issues: list[str] = []
    missing_cols: list[str] = []
    with get_conn() as conn:
        for dt, (tbl, pk, sub_c, tot_c, ltbl, lfk, lamt) in SPEC.items():
            with conn.cursor() as cur:
                present = cols_of(cur, tbl)
                has_fx = "exchange_rate_to_usd" in present
                has_usd = "converted_amount_usd" in present
                if not has_fx:
                    missing_cols.append(f"{tbl}.exchange_rate_to_usd MISSING — USD conversion cannot persist")
                if not has_usd:
                    missing_cols.append(f"{tbl}.converted_amount_usd MISSING — USD conversion cannot persist")

                sel = [pk, "currency", sub_c, tot_c, "tax_percent", "tax_amount"]
                if has_fx:
                    sel.append("exchange_rate_to_usd")
                if has_usd:
                    sel.append("converted_amount_usd")
                cur.execute(f"select {', '.join(sel)} from proc.{tbl}")
                rows = cur.fetchall()
                colidx = {c: i for i, c in enumerate(sel)}

            print(f"\n== {dt} ({len(rows)} rows) ==  USD cols present: fx={has_fx} usd={has_usd}")
            for r in rows:
                pkv = r[colidx[pk]]
                ccy = r[colidx["currency"]]
                sub = f(r[colidx[sub_c]])
                tot = f(r[colidx[tot_c]])
                pct = f(r[colidx["tax_percent"]])
                tax = f(r[colidx["tax_amount"]])

                # 1. tax_amount = subtotal * pct/100
                if sub is not None and pct not in (None, 0) and tax is not None:
                    exp = round(sub * pct / 100.0, 2)
                    if not near(tax, exp):
                        issues.append(f"[{dt} {pkv}] tax_amount={tax} but subtotal {sub} × {pct}% = {exp}")
                # 2. total_incl_tax = subtotal + tax
                if sub is not None and tax is not None and tot is not None:
                    exp = round(sub + tax, 2)
                    if not near(tot, exp):
                        issues.append(f"[{dt} {pkv}] {tot_c}={tot} but subtotal {sub} + tax {tax} = {exp}")
                # 3. exchange_rate_to_usd = FX[currency]
                rate = _FX_TO_USD.get((ccy or "").upper()) if ccy else None
                if has_fx:
                    stored_rate = f(r[colidx["exchange_rate_to_usd"]])
                    if rate is not None and not near(stored_rate, rate, 0.001):
                        issues.append(f"[{dt} {pkv}] exchange_rate_to_usd={stored_rate} but FX[{ccy}]={rate}")
                # 4. converted_amount_usd = best_total * rate
                if has_usd and rate is not None:
                    best = tot if tot is not None else sub
                    stored_usd = f(r[colidx["converted_amount_usd"]])
                    if best is not None:
                        exp = round(best * rate, 2)
                        if not near(stored_usd, exp):
                            issues.append(f"[{dt} {pkv}] converted_amount_usd={stored_usd} but {best}×{rate}={exp}")
                    if stored_usd is None and best is not None:
                        issues.append(f"[{dt} {pkv}] converted_amount_usd is NULL (should be {round(best*rate,2)})")

            # line-item totals: line_total/line_amount == qty * unit_price
            with conn.cursor() as cur:
                lpresent = cols_of(cur, ltbl)
                lsel = [lfk]
                for c in ("quantity", "unit_price", lamt):
                    if c in lpresent:
                        lsel.append(c)
                cur.execute(f"select {', '.join(lsel)} from proc.{ltbl}")
                lrows = cur.fetchall()
                lci = {c: i for i, c in enumerate(lsel)}
            bad_lines = 0
            for r in lrows:
                if "quantity" in lci and "unit_price" in lci and lamt in lci:
                    q = f(r[lci["quantity"]]); up = f(r[lci["unit_price"]]); amt = f(r[lci[lamt]])
                    if q is not None and up is not None and amt is not None and q > 0 and up > 0:
                        exp = round(q * up, 2)
                        if not near(amt, exp, 0.02):
                            bad_lines += 1
                            if bad_lines <= 5:
                                issues.append(f"[{dt} {r[lci[lfk]]}] line {lamt}={amt} but qty {q} × unit {up} = {exp}")
            if bad_lines:
                print(f"   line-item qty×unit mismatches: {bad_lines}")

    print("\n" + "=" * 60)
    print("MISSING DERIVED COLUMNS (schema gaps):")
    for m in missing_cols:
        print("  -", m)
    print(f"\nCOMPUTATION MISMATCHES: {len(issues)}")
    for it in issues:
        print("  -", it)
    if not issues and not missing_cols:
        print("\n  ✓ every derived column reconciles from its source columns")
    return 0 if not issues and not missing_cols else 2


if __name__ == "__main__":
    sys.exit(main())
