# Extraction Gap-Control Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the extraction pipeline detect when a document's extraction is incomplete (missing required header field, missing line items, or line items that don't reconcile to the header total) and run a bounded recovery pass before promotion — so `_stg` tables are never *silently* filled with gaps.

**Architecture:** Add a pure `completeness` assessment module (the gate), a line-item recovery pass on the `context_layer` LLM (`AgentNick:extract`, already primed) that reuses the same grounding discipline as header synthesis, and wire a **single bounded recovery attempt** into `dispatch_document` between context-layer synthesis and discrepancy computation. The result carries a `completeness_status` so gaps are surfaced (tracked discrepancy + ops report), recovered, or escalated to HITL — never silently promoted. Header `missing_required` already blocks promotion → HITL, so no header-recovery is added (YAGNI); the loop focuses on the silent-loss case (line items).

**Tech Stack:** Python 3.12, pytest, psycopg2 (live `bp_sqldb`), Ollama (`BeyondProcwise/AgentNick:extract`). Existing modules: `src/services/extraction/{dispatch,context_layer,persistence}.py`, `src/services/extraction/engineered/table_extractor.py`.

---

## File Structure

- **Create** `src/services/extraction/completeness.py` — pure completeness assessment (`assess()`, `line_sum()`, `CompletenessReport`). No I/O.
- **Modify** `src/services/extraction/context_layer.py` — add `synthesize_line_items()` + private helpers (`_build_line_items_prompt`, `_parse_line_items_json`, `_coerce_number`, `_squeeze`). Reuses the existing `_call_llm` seam.
- **Modify** `src/services/extraction/dispatch.py` — insert the bounded recovery loop after context-layer synthesis / PK normalization and before discrepancy computation; add `completeness_status` to the result dict.
- **Create** `tests/extraction/test_completeness.py` — pure unit tests for the assessment module.
- **Create** `tests/extraction/test_line_item_recovery.py` — unit tests for `synthesize_line_items` (mock `_call_llm`).
- **Modify** `tests/extraction/test_dispatch.py` — add a live integration test asserting `completeness_status` is present and a multi-column fixture's lines reconcile or are flagged (not silently promoted).
- **Create** `scripts/extraction_completeness_report.py` — ops query listing `_stg` docs with line gaps.

---

### Task 1: Completeness assessment module (pure)

**Files:**
- Create: `src/services/extraction/completeness.py`
- Test: `tests/extraction/test_completeness.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/extraction/test_completeness.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.completeness import assess, line_sum  # noqa: E402


def test_complete_invoice_with_reconciling_lines():
    cols = {"invoice_id": "INV1", "invoice_amount": 100.0}
    lines = [{"line_amount": 60.0}, {"line_amount": 40.0}]
    r = assess("invoice", cols, lines, has_line_schema=True, missing_required=[])
    assert r.is_complete is True
    assert r.status == "complete"
    assert r.gaps == []


def test_missing_required_header_field():
    cols = {"invoice_amount": 100.0}
    lines = [{"line_amount": 100.0}]
    r = assess("invoice", cols, lines, has_line_schema=True, missing_required=["invoice_id"])
    assert r.is_complete is False
    assert r.status == "missing_required"


def test_no_line_items_when_expected():
    cols = {"invoice_id": "INV1", "invoice_amount": 100.0}
    r = assess("invoice", cols, [], has_line_schema=True, missing_required=[])
    assert r.status == "no_line_items"
    assert r.is_complete is False


def test_line_sum_mismatch_flagged():
    cols = {"invoice_id": "INV1", "invoice_amount": 100.0}
    lines = [{"line_amount": 30.0}]  # 30 vs 100 → mismatch
    r = assess("invoice", cols, lines, has_line_schema=True, missing_required=[])
    assert r.status == "line_sum_mismatch"
    assert r.is_complete is False


def test_within_tolerance_reconciles():
    cols = {"invoice_id": "INV1", "invoice_amount": 100.0}
    lines = [{"line_amount": 98.0}]  # 2% < 5% tolerance
    r = assess("invoice", cols, lines, has_line_schema=True, missing_required=[])
    assert r.is_complete is True


def test_no_header_total_cannot_flag_mismatch():
    cols = {"invoice_id": "INV1"}  # no invoice_amount
    lines = [{"line_amount": 30.0}]
    r = assess("invoice", cols, lines, has_line_schema=True, missing_required=[])
    assert r.status == "complete"  # can't reconcile without a total → don't flag


def test_doc_without_line_schema_is_complete_on_header():
    cols = {"contract_id": "C1"}
    r = assess("contract", cols, [], has_line_schema=False, missing_required=[])
    assert r.is_complete is True


def test_quote_uses_line_total_column():
    cols = {"quote_id": "Q1", "total_amount": 50.0}
    lines = [{"line_total": 50.0}]
    r = assess("quote", cols, lines, has_line_schema=True, missing_required=[])
    assert r.is_complete is True


def test_line_sum_helper():
    assert line_sum("invoice", [{"line_amount": 1.0}, {"line_amount": 2.5}]) == 3.5
    assert line_sum("quote", [{"line_total": 4.0}]) == 4.0
    assert line_sum("invoice", []) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/extraction/test_completeness.py -q -p no:cacheprovider`
Expected: collection ERROR / FAIL — `ModuleNotFoundError` or `ImportError: cannot import name 'assess'` (module does not exist yet).

- [ ] **Step 3: Write the module**

```python
# src/services/extraction/completeness.py
"""Completeness assessment for extracted documents.

Pure functions — no I/O. Given the built header columns + line items for a
document, decide whether the extraction is COMPLETE enough to promote, or has a
gap: missing required header field, missing line items, or line items that do
not reconcile to the header subtotal. The dispatch loop uses this to trigger a
bounded recovery pass before promotion, so gaps are recovered or surfaced —
never silently promoted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

# Header column holding the line-item subtotal, per doc type.
_SUBTOTAL_COL = {
    "invoice": "invoice_amount",
    "purchase_order": "total_amount",
    "quote": "total_amount",
}
# Line-item column holding the per-line amount, per doc type.
_LINE_AMOUNT_COL = {
    "invoice": "line_amount",
    "purchase_order": "line_total",
    "quote": "line_total",
}

_RECONCILE_TOLERANCE = 0.05  # 5%


@dataclass
class CompletenessReport:
    header_complete: bool
    lines_expected: bool
    lines_present: bool
    lines_reconcile: bool
    is_complete: bool
    status: str  # complete | missing_required | no_line_items | line_sum_mismatch
    gaps: list[str] = field(default_factory=list)


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(Decimal(str(v)))
    except (InvalidOperation, ValueError, TypeError):
        return None


def line_sum(doc_type: str, line_items: list[dict[str, Any]]) -> Optional[float]:
    """Sum the per-line amount column for this doc type. None if no usable amounts."""
    amt_col = _LINE_AMOUNT_COL.get(doc_type)
    if not amt_col or not line_items:
        return None
    vals = [_to_float(li.get(amt_col)) for li in line_items]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def _reconciles(lsum: Optional[float], header_total: Optional[float]) -> bool:
    # Can't check without a header total → don't flag a gap.
    if header_total in (None, 0) or header_total == 0.0:
        return True
    if lsum is None:
        return False
    return abs(lsum - header_total) <= max(0.01, _RECONCILE_TOLERANCE * abs(header_total))


def assess(
    doc_type: str,
    columns: dict[str, Any],
    line_items: list[dict[str, Any]],
    *,
    has_line_schema: bool,
    missing_required: list[str] | None = None,
) -> CompletenessReport:
    """Assess extraction completeness. Pure — no I/O."""
    missing_required = missing_required or []
    header_complete = not missing_required

    lines_expected = bool(has_line_schema)
    lines_present = bool(line_items)

    sub_col = _SUBTOTAL_COL.get(doc_type)
    header_total = _to_float(columns.get(sub_col)) if sub_col else None
    lsum = line_sum(doc_type, line_items)
    lines_reconcile = (
        _reconciles(lsum, header_total)
        if (lines_expected and lines_present)
        else True
    )

    gaps: list[str] = []
    if not header_complete:
        gaps.append("missing_required:" + ",".join(missing_required))
    if lines_expected and not lines_present:
        gaps.append("no_line_items")
    if lines_expected and lines_present and not lines_reconcile:
        gaps.append(f"line_sum_mismatch(lines={lsum},header={header_total})")

    if not header_complete:
        status = "missing_required"
    elif lines_expected and not lines_present:
        status = "no_line_items"
    elif lines_expected and lines_present and not lines_reconcile:
        status = "line_sum_mismatch"
    else:
        status = "complete"

    return CompletenessReport(
        header_complete=header_complete,
        lines_expected=lines_expected,
        lines_present=lines_present,
        lines_reconcile=lines_reconcile,
        is_complete=(status == "complete"),
        status=status,
        gaps=gaps,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/extraction/test_completeness.py -q -p no:cacheprovider`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/services/extraction/completeness.py tests/extraction/test_completeness.py
git commit -m "feat(extraction): completeness assessment module (gap gate)"
```

---

### Task 2: Line-item recovery pass on the context layer

**Files:**
- Modify: `src/services/extraction/context_layer.py` (add `synthesize_line_items` + helpers near the existing `synthesize`)
- Test: `tests/extraction/test_line_item_recovery.py`

- [ ] **Step 1: Write the failing tests** (mock the LLM seam `_call_llm`)

```python
# tests/extraction/test_line_item_recovery.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction import context_layer  # noqa: E402


FULL_TEXT = (
    "ACME LTD INVOICE INV9\n"
    "Design services - phase 1   600.00\n"
    "Hosting - annual            400.00\n"
    "Subtotal 1000.00\n"
)


def test_recovers_grounded_line_items(monkeypatch):
    monkeypatch.setattr(context_layer, "_call_llm", lambda prompt: (
        '[{"description":"Design services - phase 1","quantity":1,"unit_price":600.0,"amount":600.0},'
        '{"description":"Hosting - annual","quantity":1,"unit_price":400.0,"amount":400.0}]'
    ))
    items = context_layer.synthesize_line_items("invoice", FULL_TEXT, header_subtotal=1000.0)
    assert len(items) == 2
    assert items[0]["description"] == "Design services - phase 1"
    assert items[0]["amount"] == 600.0
    assert sum(i["amount"] for i in items) == 1000.0


def test_drops_ungrounded_rows(monkeypatch):
    # "Consulting" is NOT in FULL_TEXT → must be dropped (no fabrication).
    monkeypatch.setattr(context_layer, "_call_llm", lambda prompt: (
        '[{"description":"Consulting fee","amount":999.0},'
        '{"description":"Hosting - annual","amount":400.0}]'
    ))
    items = context_layer.synthesize_line_items("invoice", FULL_TEXT, header_subtotal=400.0)
    assert len(items) == 1
    assert items[0]["description"] == "Hosting - annual"


def test_llm_failure_returns_empty(monkeypatch):
    def _boom(prompt):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(context_layer, "_call_llm", _boom)
    assert context_layer.synthesize_line_items("invoice", FULL_TEXT) == []


def test_empty_text_returns_empty():
    assert context_layer.synthesize_line_items("invoice", "") == []


def test_non_json_response_returns_empty(monkeypatch):
    monkeypatch.setattr(context_layer, "_call_llm", lambda prompt: "sorry, no tables here")
    assert context_layer.synthesize_line_items("invoice", FULL_TEXT) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/extraction/test_line_item_recovery.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: module 'context_layer' has no attribute 'synthesize_line_items'`.

- [ ] **Step 3: Add the function + helpers to `context_layer.py`**

Add near the existing `synthesize` definition (after it is fine):

```python
import json as _json_le
import re as _re_le


def _squeeze(s: str) -> str:
    """All-whitespace-stripped, lowercased — for tolerant substring grounding."""
    return _re_le.sub(r"\s+", "", str(s)).lower()


def _coerce_number(v):
    """Best-effort numeric coercion; None on failure or empty."""
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", "").replace("£", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_line_items_json(raw: str) -> list[dict]:
    """Extract the first JSON array from the LLM response. [] on failure."""
    if not raw:
        return []
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end <= start:
        return []
    try:
        data = _json_le.loads(raw[start:end])
    except (ValueError, TypeError):
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def _build_line_items_prompt(doc_type: str, full_text: str, header_subtotal) -> str:
    sub = f"The line items should sum to approximately {header_subtotal}.\n" if header_subtotal else ""
    return (
        f"You are extracting the LINE ITEMS from a procurement {doc_type}.\n"
        "Read the DOCUMENT TEXT and output ONLY a JSON array. Each element:\n"
        '{"description": <string>, "quantity": <number|null>, '
        '"unit_price": <number|null>, "amount": <number>}\n\n'
        "RULES:\n"
        "1. Output ONLY the JSON array. No prose, no markdown fences.\n"
        "2. Every `description` MUST be a verbatim substring of the DOCUMENT TEXT. "
        "If you cannot find it verbatim, omit that row. DO NOT FABRICATE.\n"
        "3. `amount` is the per-line total as a NUMBER (no currency symbol/commas).\n"
        "4. Do NOT include subtotal / tax / total summary rows as line items.\n"
        f"5. {sub}"
        "\nDOCUMENT TEXT:\n"
        f"{full_text}\n"
    )


def synthesize_line_items(
    doc_type: str,
    full_text: str,
    header_subtotal: float | None = None,
) -> list[dict]:
    """Ask AgentNick to enumerate line items grounded in full_text.

    Recovery pass used when the structural/table extractor returned no lines or
    lines that don't reconcile to the header subtotal. Returns generic dicts
    {description, quantity, unit_price, amount}; the caller maps these to the
    schema's line-item db_columns. Never raises — returns [] on any failure.
    """
    if not (full_text and full_text.strip()):
        return []
    prompt = _build_line_items_prompt(doc_type, full_text, header_subtotal)
    try:
        raw = _call_llm(prompt)
    except Exception:  # noqa: BLE001
        return []
    items = _parse_line_items_json(raw or "")
    grounded: list[dict] = []
    ft_sq = _squeeze(full_text)
    for it in items:
        desc = (it.get("description") or "").strip()
        amt = _coerce_number(it.get("amount"))
        if not desc or amt is None:
            continue
        if desc not in full_text and _squeeze(desc) not in ft_sq:
            continue
        grounded.append({
            "description": desc,
            "quantity": _coerce_number(it.get("quantity")),
            "unit_price": _coerce_number(it.get("unit_price")),
            "amount": amt,
        })
    return grounded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/extraction/test_line_item_recovery.py -q -p no:cacheprovider`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/services/extraction/context_layer.py tests/extraction/test_line_item_recovery.py
git commit -m "feat(extraction): AgentNick line-item recovery pass (grounded, no fabrication)"
```

---

### Task 3: Wire the bounded recovery loop into dispatch

**Files:**
- Modify: `src/services/extraction/dispatch.py` (insert after PK normalization, before `discrepancies: list[Discrepancy] = []`; add `completeness_status` to result dict)
- Test: `tests/extraction/test_dispatch.py` (add live integration test)

- [ ] **Step 1: Write the failing live integration test**

Append to `tests/extraction/test_dispatch.py`:

```python
def test_dispatch_sets_completeness_status_and_recovers_lines():
    """Multi-column invoice fixture: lines must reconcile (recovered) or be
    flagged via completeness_status — never silently complete with a gap."""
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tests" / "extraction_v3" / "fixtures" / "invoices"
        / "INV-002-multi-column.pdf"
    )
    result = dispatch_document(
        process_monitor_id=None, file_path=str(fixture), doc_type="invoice",
    )
    assert "completeness_status" in result
    assert result["completeness_status"] in (
        "complete", "recovered", "line_sum_mismatch", "no_line_items", "missing_required",
    )
    # If it promoted, it must NOT be a silent line gap: either complete/recovered,
    # or explicitly flagged.
    if result["status"] == "promoted":
        assert result["completeness_status"] != "no_line_items" or result["line_items"] == 0
    # Cleanup raw row
    import psycopg2 as _pg
    with _conn() as c:
        cur = c.cursor()
        cur.execute("DELETE FROM proc.bp_extraction_provenance_v3 WHERE doc_pk=%s", (result["doc_pk"],))
        cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE raw_id=%s", (result["raw_id"],))
        cur.execute("DELETE FROM proc.bp_invoice_raw WHERE raw_id=%s", (result["raw_id"],))
        c.commit()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest "tests/extraction/test_dispatch.py::test_dispatch_sets_completeness_status_and_recovers_lines" -q -p no:cacheprovider`
Expected: FAIL — `KeyError: 'completeness_status'` (dispatch result has no such key yet).

- [ ] **Step 3: Add the recovery loop to `dispatch.py`**

3a. Add imports near the top (after `from src.services.extraction import persistence, promotion`):

```python
from src.services.extraction import completeness as _completeness
```

3b. Add this module-level mapping after `_QUOTE_ID_PREFIX` definition:

```python
# Map the generic recovery item shape → per-doc-type line-item db_columns.
_RECOVERED_LINE_MAP = {
    "invoice": {"amount": "line_amount", "no": "line_no"},
    "purchase_order": {"amount": "line_total", "no": "line_number"},
    "quote": {"amount": "line_total", "no": "line_number"},
}
```

3c. Insert the recovery block. Find this existing region (the PK-normalization block added previously, immediately before `discrepancies: list[Discrepancy] = []`):

```python
    # Canonicalize the primary key before it is used for _raw, provenance and
    # promotion (e.g. quote 'QUT136586' → '136586') so the same quote referenced
    # across documents resolves to a single _stg row.
    _pk_field = persistence._DOC_PK_FIELD[doc_type]
    if columns.get(_pk_field) not in (None, ""):
        columns[_pk_field] = normalize_doc_pk(doc_type, columns[_pk_field])

    discrepancies: list[Discrepancy] = []
```

Replace it with (PK block unchanged, recovery block added between it and `discrepancies = []`):

```python
    # Canonicalize the primary key before it is used for _raw, provenance and
    # promotion (e.g. quote 'QUT136586' → '136586') so the same quote referenced
    # across documents resolves to a single _stg row.
    _pk_field = persistence._DOC_PK_FIELD[doc_type]
    if columns.get(_pk_field) not in (None, ""):
        columns[_pk_field] = normalize_doc_pk(doc_type, columns[_pk_field])

    # --- Bounded gap-control recovery (one attempt) ---
    # If the structural/table extractor produced no lines or lines that don't
    # reconcile to the header subtotal, ask AgentNick to re-enumerate line
    # items from full_text. Accept the recovered set only when it reconciles
    # (or sums closer to the header total than what we had) — never fabricate.
    has_line_schema = bool(
        registry.schema.line_items and registry.schema.line_items.fields
    )
    pre = _completeness.assess(
        doc_type, columns, line_items, has_line_schema=has_line_schema,
    )
    if has_line_schema and full_text.strip() and pre.status in (
        "no_line_items", "line_sum_mismatch",
    ):
        try:
            sub_col = _completeness._SUBTOTAL_COL.get(doc_type)
            header_total = columns.get(sub_col) if sub_col else None
            from src.services.extraction.context_layer import (
                synthesize_line_items as _synth_lines,
            )
            recovered = _synth_lines(doc_type, full_text, header_total)
        except Exception as exc:  # noqa: BLE001
            log.warning("line-item recovery failed: %s", exc)
            recovered = []

        if recovered:
            lmap = _RECOVERED_LINE_MAP[doc_type]
            mapped: list[dict[str, Any]] = []
            for i, it in enumerate(recovered, start=1):
                row: dict[str, Any] = {
                    lmap["no"]: i,
                    "item_description": it["description"],
                    lmap["amount"]: it["amount"],
                }
                if it.get("quantity") is not None:
                    row["quantity"] = it["quantity"]
                if it.get("unit_price") is not None:
                    row["unit_price"] = it["unit_price"]
                mapped.append(row)

            # Accept only if the recovered set is at least as good.
            ht = _completeness._to_float(header_total)
            old_sum = _completeness.line_sum(doc_type, line_items)
            new_sum = _completeness.line_sum(doc_type, mapped)
            accept = False
            if ht:
                old_err = abs((old_sum or 0) - ht)
                new_err = abs((new_sum or 0) - ht)
                accept = new_err < old_err
            elif not line_items:
                accept = True  # had nothing; grounded recovery is strictly better
            if accept:
                log.info(
                    "line-item recovery: replaced %d lines with %d (sum %.2f→%.2f, header=%s)",
                    len(line_items), len(mapped), old_sum or 0, new_sum or 0, header_total,
                )
                line_items = mapped

    discrepancies: list[Discrepancy] = []
```

3d. Set `completeness_status` on the result. Find the `missing_required_fields = sorted({...})` block (just after the promotion block) and add, right after it:

```python
    completeness_status = _completeness.assess(
        doc_type, columns, line_items,
        has_line_schema=has_line_schema,
        missing_required=missing_required_fields,
    ).status
```

3e. Add `completeness_status` into the `result = {...}` dict (next to `"missing_required": missing_required_fields,`):

```python
        "completeness_status": completeness_status,
```

- [ ] **Step 4: Run the live integration test to verify it passes**

Pre-req: procwise can be running; the test calls `dispatch_document` directly and needs Ollama `AgentNick:extract` primed on GPU (it is, per session ops). Run:
`.venv/bin/python -m pytest "tests/extraction/test_dispatch.py::test_dispatch_sets_completeness_status_and_recovers_lines" -q -p no:cacheprovider`
Expected: PASS (`completeness_status` present; if `INV-002-multi-column` previously lost lines, they now reconcile via recovery or the doc is flagged, not silently complete).

- [ ] **Step 5: Run the full extraction test suite for regressions**

Run: `.venv/bin/python -m pytest tests/extraction/ -q -p no:cacheprovider`
Expected: the new tests pass and `test_dispatch_writes_raw_and_provenance` still passes. (Pre-existing failures `test_pattern_yaml_loader.py::{test_invoice_id_has_patterns,test_invoice_amount_has_patterns}` are unrelated to this change — see 2026-05-25 validation notes — and may still fail; do not "fix" them here.)

- [ ] **Step 6: Commit**

```bash
git add src/services/extraction/dispatch.py tests/extraction/test_dispatch.py
git commit -m "feat(extraction): bounded gap-control recovery loop + completeness_status"
```

---

### Task 4: Operational completeness report

**Files:**
- Create: `scripts/extraction_completeness_report.py`

- [ ] **Step 1: Write the report script**

```python
# scripts/extraction_completeness_report.py
"""Report _stg documents whose line items don't reconcile to the header total,
or that have no line items where the doc type expects them. Read-only.

Run: .venv/bin/python scripts/extraction_completeness_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.db import get_conn  # noqa: E402
from src.services.extraction.completeness import assess  # noqa: E402

SPEC = {
    "invoice": ("proc.bp_invoice_stg", "invoice_id", "invoice_amount",
                "proc.bp_invoice_line_items_stg", "line_amount"),
    "purchase_order": ("proc.bp_purchase_order_stg", "po_id", "total_amount",
                       "proc.bp_po_line_items_stg", "line_total"),
    "quote": ("proc.bp_quote_stg", "quote_id", "total_amount",
              "proc.bp_quote_line_items_stg", "line_total"),
}


def main() -> int:
    flagged = 0
    with get_conn() as conn:
        for dt, (htbl, pk, subcol, ltbl, amtcol) in SPEC.items():
            with conn.cursor() as cur:
                cur.execute(f"SELECT {pk}, {subcol} FROM {htbl}")
                headers = cur.fetchall()
            print(f"\n== {dt} ({len(headers)} rows) ==")
            for pk_val, subtotal in headers:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT {amtcol} FROM {ltbl} WHERE {pk} = %s", (pk_val,))
                    lines = [{amtcol: r[0]} for r in cur.fetchall()]
                r = assess(dt, {subcol: subtotal}, lines,
                           has_line_schema=True, missing_required=[])
                if not r.is_complete:
                    flagged += 1
                    print(f"  [{r.status:18s}] {pk_val}  lines={len(lines)} gaps={r.gaps}")
    print(f"\nTotal flagged: {flagged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it against live data**

Run: `.venv/bin/python scripts/extraction_completeness_report.py`
Expected: prints each doc type with any rows whose lines don't reconcile (status `line_sum_mismatch` / `no_line_items`) and a total. No traceback.

- [ ] **Step 3: Commit**

```bash
git add scripts/extraction_completeness_report.py
git commit -m "feat(extraction): completeness report for _stg line-item gaps"
```

---

### Task 5: Silence the broken daily fine-tune cron (cleanup)

The daily cron fires `agentnick_finetune_daily.sh` at 18:30 UTC and **fails every night** (`_train_model` stub + missing `unsloth`), spamming `STATUS: FAILED` logs. It targets the wrong model/data for orchestration anyway. Disable until a real trainer exists.

**Files:** crontab (user `muthu`).

- [ ] **Step 1: Capture current crontab**

Run: `crontab -l > /tmp/crontab.backup && cat /tmp/crontab.backup`
Expected: shows the `30 18 * * * .../agentnick_finetune_daily.sh` line.

- [ ] **Step 2: Comment out the finetune line**

Run:
```bash
crontab -l | sed 's#^\(30 18 \* \* \* .*agentnick_finetune_daily\.sh.*\)## DISABLED 2026-05-26 (stub trainer; re-enable when real): \1#' | crontab -
```

- [ ] **Step 3: Verify it's disabled**

Run: `crontab -l | grep -n finetune`
Expected: the line is present but prefixed with `# DISABLED 2026-05-26` (commented).

- [ ] **Step 4: Commit a note** (no repo file changes; record in the handoff doc)

```bash
printf '\n## 2026-05-26\nDaily finetune cron DISABLED (commented in crontab) — stub trainer + wrong target. Backup at /tmp/crontab.backup. Re-enable only after a real _train_model exists.\n' >> docs/model_tuning/agentnick_finetune_handoff_2026_05_25.md
git add docs/model_tuning/agentnick_finetune_handoff_2026_05_25.md
git commit -m "chore(training): disable broken daily finetune cron; note in handoff"
```

---

## Self-Review

- **Spec coverage:** completeness gate (Task 1), line-item recovery (Task 2), wired bounded loop + status + promotion-gate visibility (Task 3), ops visibility (Task 4), cron noise (Task 5). Header `missing_required` recovery intentionally omitted — already blocks → HITL (not a silent gap).
- **Placeholder scan:** none — every code/command step is concrete.
- **Type consistency:** `assess()`/`line_sum()`/`CompletenessReport`/`_SUBTOTAL_COL`/`_to_float` names match across Tasks 1, 3, 4. `synthesize_line_items` returns `{description,quantity,unit_price,amount}` consumed by Task 3's `_RECOVERED_LINE_MAP` mapping. `_call_llm` is the existing seam mocked in Task 2.
- **Ambiguity:** recovery accepts the new line set only when it reconciles closer to the header total (or when there were no lines at all) — explicit in 3c. Line gaps are surfaced via `completeness_status` + existing non-blocking line discrepancies, not made hard-blocking (respects the known multi-column-flatten limitation; no HITL flood).
