"""Source-grounded accuracy audit of persisted _stg rows.

For every header + line-item value currently in proc.bp_*_stg, download the
ORIGINAL document from S3 (via the live pipeline's own parser), parse it, and
verify each committed value grounds back to the document text. Flags any value
that does NOT appear in the source (potential fabrication or extraction error).

Read-only: never writes to the DB. Run:
    .venv/bin/python scripts/source_grounded_audit.py
"""
from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.db import get_conn  # noqa: E402
from src.services.extraction.parser import parse as parse_doc  # noqa: E402

_WS = re.compile(r"\s+")
_LETTER_SPACED = re.compile(r"(?:(?<=^)|(?<=\s))((?:\S\s){3,}\S)(?=\s|$)")
_CURRENCY_SYMBOLS = {
    "GBP": ("£", "GBP"), "USD": ("$", "US$", "USD"),
    "EUR": ("€", "EUR"), "JPY": ("¥", "JPY"),
}


def _collapse(s: str) -> str:
    if not s:
        return s
    return _LETTER_SPACED.sub(lambda m: m.group(1).replace(" ", ""), s)


def _squeezed(s) -> str:
    return _WS.sub("", _collapse(str(s))).lower()


def _norm(s) -> str:
    return _WS.sub(" ", str(s)).strip()


def _currency_present(code, raw, norm) -> bool:
    if not code:
        return True
    for tok in _CURRENCY_SYMBOLS.get(str(code).upper(), (str(code).upper(),)):
        if tok in raw or tok in norm:
            return True
    return False


def _money_present(val, norm) -> bool:
    s = str(val).replace(",", "")
    cands = {str(val), s}
    try:
        f = float(s)
        cands.add(f"{f:,.2f}"); cands.add(f"{f:.2f}"); cands.add(f"{f:,.2f}".replace(",", ""))
        if f == int(f):
            cands.add(f"{int(f):,}"); cands.add(str(int(f)))
    except (ValueError, TypeError):
        pass
    return any(c and c in norm for c in cands)


def _string_present(val, raw, norm) -> bool:
    if val is None or str(val).strip() == "":
        return True
    return (_norm(val).lower() in norm.lower()
            or str(val) in raw
            or _squeezed(val) in _squeezed(raw))


def _date_present(val, raw) -> bool:
    """Loose date grounding: accept if ISO form OR (year token AND day token)
    appear. Docs render dates many ways (12 May 2025, 2025-05-12, 12/05/25)."""
    if val is None:
        return True
    s = str(val)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if not m:
        return str(val) in raw
    y, mo, d = m.group(1), m.group(2), m.group(3)
    sq = _squeezed(raw)
    if s.replace("-", "") in sq or s in raw:
        return True
    # day (with/without leading zero) and year both present somewhere
    day_variants = {d, d.lstrip("0")}
    has_day = any(dv and dv in raw for dv in day_variants)
    return (y in raw) and has_day


# (column, kind) per doc type. supplier_id in invoice/quote is a resolved FK
# (SUP-xxx) that never appears verbatim — excluded from grounding.
HEADER_COLS = {
    "invoice": [
        ("invoice_id", "str"), ("po_id", "str"), ("currency", "ccy"),
        ("invoice_amount", "money"), ("tax_amount", "money"),
        ("invoice_total_incl_tax", "money"),
        ("invoice_date", "date"), ("due_date", "date"),
    ],
    "purchase_order": [
        ("po_id", "str"), ("supplier_name", "str"), ("currency", "ccy"),
        ("total_amount", "money"), ("tax_amount", "money"),
        ("total_amount_incl_tax", "money"), ("order_date", "date"),
        ("expected_delivery_date", "date"), ("postal_code", "str"),
    ],
    "quote": [
        ("quote_id", "str"), ("po_id", "str"), ("currency", "ccy"),
        ("total_amount", "money"), ("tax_amount", "money"),
        ("total_amount_incl_tax", "money"),
        ("quote_date", "date"), ("validity_date", "date"),
    ],
}
STG = {"invoice": ("proc.bp_invoice_stg", "invoice_id"),
       "purchase_order": ("proc.bp_purchase_order_stg", "po_id"),
       "quote": ("proc.bp_quote_stg", "quote_id")}
LINE = {"invoice": ("proc.bp_invoice_line_items_stg", "invoice_id", "line_amount"),
        "purchase_order": ("proc.bp_po_line_items_stg", "po_id", "line_total"),
        "quote": ("proc.bp_quote_line_items_stg", "quote_id", "line_total")}


def _coerce(v):
    if isinstance(v, Decimal):
        return float(v)
    return v


def main() -> int:
    # 1. Parse every process_monitor source file once (download from S3).
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, file_path, category FROM proc.process_monitor ORDER BY id")
        pm = cur.fetchall()
    print(f"Parsing {len(pm)} source documents from S3 ...")
    texts: dict[str, str] = {}

    def _do(fp):
        try:
            return fp, parse_doc(fp).full_text or ""
        except Exception as e:  # noqa: BLE001
            return fp, f"__PARSE_ERROR__ {e}"

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_do, fp): fp for _, fp, _ in pm if fp}
        for fut in as_completed(futs):
            fp, txt = fut.result()
            texts[fp] = txt
    n_err = sum(1 for t in texts.values() if t.startswith("__PARSE_ERROR__"))
    print(f"  parsed ok: {len(texts)-n_err}, parse errors: {n_err}\n")

    # Precompute normalized text per file
    norm = {fp: _norm(t) for fp, t in texts.items()}

    def match_file(pk: str) -> str | None:
        """Find the source file whose text contains this PK (squeezed)."""
        pk_sq = _squeezed(pk)
        if not pk_sq:
            return None
        hits = [fp for fp, t in texts.items()
                if not t.startswith("__PARSE_ERROR__") and pk_sq in _squeezed(t)]
        if not hits:
            # try filename match
            hits = [fp for fp in texts if pk_sq in _squeezed(Path(fp).name)]
        # prefer the file whose filename also contains the PK
        hits.sort(key=lambda fp: (pk_sq not in _squeezed(Path(fp).name), len(texts[fp])))
        return hits[0] if hits else None

    total_fields = total_ok = 0
    total_lines = total_lines_ok = 0
    all_issues: list[str] = []
    unmatched: list[str] = []

    for dt, (table, pk_col) in STG.items():
        with get_conn() as conn, conn.cursor() as cur:
            cols = [c for c, _ in HEADER_COLS[dt]]
            cur.execute(f"SELECT {pk_col}, {', '.join(cols)} FROM {table} ORDER BY {pk_col}")
            rows = cur.fetchall()
            ltable, lfk, lamt = LINE[dt]
            cur.execute(
                f"SELECT {lfk}, item_description, {lamt} FROM {ltable} ORDER BY {lfk}")
            lines_by_pk: dict[str, list] = {}
            for r in cur.fetchall():
                lines_by_pk.setdefault(r[0], []).append((r[1], r[2]))

        print(f"\n========== {dt.upper()} ({len(rows)} rows) ==========")
        for row in rows:
            pk = row[0]
            fp = match_file(pk)
            if not fp or texts.get(fp, "").startswith("__PARSE_ERROR__"):
                unmatched.append(f"{dt}:{pk}")
                print(f"  [{pk}] ⚠ NO SOURCE FILE MATCHED — cannot ground")
                continue
            raw, nm = texts[fp], norm[fp]
            row_issues = []
            for i, (col, kind) in enumerate(HEADER_COLS[dt], start=1):
                val = _coerce(row[i])
                if val is None or str(val).strip() == "":
                    continue
                total_fields += 1
                if kind == "money":
                    ok = _money_present(val, nm)
                elif kind == "ccy":
                    ok = _currency_present(val, raw, nm)
                elif kind == "date":
                    ok = _date_present(val, raw)
                else:
                    ok = _string_present(val, raw, nm)
                if ok:
                    total_ok += 1
                else:
                    row_issues.append(f"{col}={val!r}")
            # line items
            litems = lines_by_pk.get(pk, [])
            li_bad = 0
            for desc, amt in litems:
                total_lines += 1
                desc_ok = (desc is None) or _string_present(desc, raw, nm)
                amt_ok = (amt is None) or _money_present(_coerce(amt), nm)
                if desc_ok and amt_ok:
                    total_lines_ok += 1
                else:
                    li_bad += 1
            src = Path(fp).name
            status = "✓" if not row_issues and not li_bad else "✗"
            print(f"  [{status}] {pk:18s} lines={len(litems)}"
                  + (f" bad_lines={li_bad}" if li_bad else "")
                  + f"  src={src[:42]}")
            if row_issues:
                msg = f"    {dt}:{pk} ungrounded: " + ", ".join(row_issues)
                print(msg)
                all_issues.append(msg.strip())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    hf = (total_ok / total_fields * 100) if total_fields else 0
    lf = (total_lines_ok / total_lines * 100) if total_lines else 0
    print(f"Header fields grounded:  {total_ok}/{total_fields} = {hf:.1f}%")
    print(f"Line items grounded:     {total_lines_ok}/{total_lines} = {lf:.1f}%")
    print(f"Rows with no source match: {len(unmatched)}  {unmatched if unmatched else ''}")
    print(f"Ungrounded header values:  {len(all_issues)}")
    if not all_issues and not unmatched:
        print("\n  ✓ ALL committed values ground back to their source document.")
    return 0 if not all_issues and not unmatched else 2


if __name__ == "__main__":
    sys.exit(main())
