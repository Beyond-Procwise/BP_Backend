"""Audit accuracy of newly promoted docs by comparing extracted values to the
parsed full_text.

Promoted rows are deleted from _raw after they land in _stg (by design — the
_raw table is a queue). So the audit:
  1. Reads /tmp/redispatch_log.jsonl for (source_file, doc_pk, prev_raw_id).
  2. Joins stg by doc_pk to get the final coerced values.
  3. Joins the prior _raw row (still discrepancy, but same parser_snapshot)
     to get the document text we should be able to ground each value against.
  4. Flags any value that does NOT appear (whitespace-normalised) in full_text.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.db import get_conn  # noqa: E402


RAW_TABLE = {
    "invoice": "proc.bp_invoice_raw",
    "purchase_order": "proc.bp_purchase_order_raw",
    "quote": "proc.bp_quote_raw",
}
STG_TABLE = {
    "invoice": "proc.bp_invoice_stg",
    "purchase_order": "proc.bp_purchase_order_stg",
    "quote": "proc.bp_quote_stg",
}
PK_COL = {
    "invoice": "invoice_id",
    "purchase_order": "po_id",
    "quote": "quote_id",
}

# Fields we audit per doc-type. We check the bare PK + amounts + currency.
# Supplier_name in invoice/quote stg is normalised to supplier_id (FK), so
# we skip it there; for PO stg we can audit supplier_name directly.
AUDIT_COLS = {
    "invoice": [
        ("invoice_id", "string"),
        ("invoice_amount", "money"),
        ("invoice_total_incl_tax", "money"),
        ("currency", "string"),
    ],
    "purchase_order": [
        ("po_id", "string"),
        ("total_amount", "money"),
        ("total_amount_incl_tax", "money"),
        ("currency", "string"),
        ("supplier_name", "string"),
    ],
    "quote": [
        ("quote_id", "string"),
        ("total_amount", "money"),
        ("total_amount_incl_tax", "money"),
        ("currency", "string"),
    ],
}


_WS = re.compile(r"\s+")
_LETTER_SPACED = re.compile(r"(?:(?<=^)|(?<=\s))((?:\S\s){3,}\S)(?=\s|$)")


def _collapse_letter_spacing(s: str) -> str:
    """Same OCR-letter-spacing collapse the judge's safety check uses."""
    if not s:
        return s
    return _LETTER_SPACED.sub(lambda m: m.group(1).replace(" ", ""), s)


def _squeezed(s: str) -> str:
    """Strip all whitespace AND collapse letter-spaced runs."""
    return _WS.sub("", _collapse_letter_spacing(str(s)))

# Symbol → ISO code mapping the renovation pipeline applies during type
# binding. A doc with "£1,200" persists as currency="GBP"; the audit must
# accept the SYMBOL or the CODE as ground for the column value.
_CURRENCY_SYMBOLS = {
    "GBP": ("£", "GBP"),
    "USD": ("$", "US$", "USD"),
    "EUR": ("€", "EUR"),
    "JPY": ("¥", "JPY"),
}


def _norm(s: str) -> str:
    return _WS.sub(" ", str(s)).strip()


def _currency_present(code: str, full_text: str, full_text_norm: str) -> bool:
    if not code:
        return False
    for token in _CURRENCY_SYMBOLS.get(code.upper(), (code.upper(),)):
        if token in full_text or token in full_text_norm:
            return True
    return False


def _money_present(val, full_text_norm: str) -> bool:
    """Money values are coerced (e.g. '£1,170.64' -> 1170.64). Verify that
    either the numeric token (with or without comma grouping) appears in the
    doc text, with or without the £/$/€ prefix."""
    s = str(val)
    s_clean = s.replace(",", "")
    candidates = {s, s_clean}
    try:
        f = float(s_clean)
        formatted = f"{f:,.2f}"
        candidates.add(formatted)
        candidates.add(formatted.replace(",", ""))
        if f == int(f):
            candidates.add(f"{int(f):,}")
            candidates.add(str(int(f)))
    except (ValueError, TypeError):
        pass
    for cand in candidates:
        if cand and cand in full_text_norm:
            return True
    return False


def audit() -> int:
    log_path = Path("/tmp/redispatch_log.jsonl")
    if not log_path.exists():
        print("no redispatch log; nothing to audit")
        return 1
    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    promoted = [e for e in entries if e.get("new_status") == "promoted"]
    print(f"promoted docs in last redispatch: {len(promoted)}")

    n_audited = 0
    issues: list[tuple[str, str, str, str]] = []
    missing_doc_text: list[tuple[str, str]] = []

    with get_conn() as conn:
        for e in promoted:
            dt = e["doc_type"]
            doc_pk = e["doc_pk"]
            prev_raw_id = e["prev_raw_id"]
            source_file = e["source_file"]

            stg_table = STG_TABLE[dt]
            raw_table = RAW_TABLE[dt]
            pk_col = PK_COL[dt]
            cols = AUDIT_COLS[dt]
            col_list = [c for c, _ in cols]

            # 1. Pull the parser snapshot from the older discrepancy raw row
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT parser_snapshot::jsonb ->> 'full_text' FROM {raw_table} WHERE raw_id = %s",
                    (prev_raw_id,),
                )
                row = cur.fetchone()
            if not row or not row[0]:
                missing_doc_text.append((dt, source_file))
                continue
            full_text_norm = _norm(row[0])

            # 2. Pull the persisted stg row by PK
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {', '.join(col_list)} FROM {stg_table} WHERE {pk_col} = %s "
                    f"ORDER BY last_modified_date DESC NULLS LAST LIMIT 1",
                    (doc_pk,),
                )
                stg_row = cur.fetchone()
            if not stg_row:
                issues.append((dt, source_file, "stg_row", f"no stg row for pk={doc_pk!r}"))
                continue
            n_audited += 1

            for i, (col, kind) in enumerate(cols):
                val = stg_row[i]
                if val is None or val == "":
                    continue
                if col == "currency":
                    if not _currency_present(str(val), row[0], full_text_norm):
                        issues.append((dt, source_file, col, str(val)))
                elif kind == "money":
                    if not _money_present(val, full_text_norm):
                        issues.append((dt, source_file, col, str(val)))
                elif kind == "string":
                    if (_norm(val) not in full_text_norm
                            and str(val) not in row[0]
                            and _squeezed(val) not in _squeezed(row[0])):
                        issues.append((dt, source_file, col, str(val)))

    print(f"\naudited {n_audited} promoted rows; missed parser_snapshot lookup for {len(missing_doc_text)}")
    if missing_doc_text:
        for dt, sf in missing_doc_text[:5]:
            print(f"  [no full_text] {dt} {sf}")
    print(f"accuracy issues found: {len(issues)}")
    for dt, sf, col, val in issues:
        print(f"  [{dt}] {sf}")
        print(f"    {col} = {val!r}  ← NOT in parsed full_text")
    if not issues:
        print("\n  ✓ all audited values ground back to the source document")
    return 0 if not issues else 2


if __name__ == "__main__":
    sys.exit(audit())
