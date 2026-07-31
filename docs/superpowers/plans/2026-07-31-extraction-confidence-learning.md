# Extraction Confidence Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the extraction pipeline's confidence reflect how often each way of reading a field has actually been *right*, judged by what humans did to it — so a field that keeps getting corrected stops being trusted, and one that keeps being confirmed stops asking.

**Architecture:** Three things are missing and they chain. (1) Nothing records *what produced* each persisted value, so no outcome can be attributed to anything. (2) Nothing reads the human's verdict when a discrepancy is resolved — `resolved_value` and `resolution_action` are written and never looked at again. (3) The only "confidence" in the system measures completeness (how many fields are filled), not correctness. This plan adds a provenance row per persisted field, a verdict row per human resolution, an observed accuracy per (doc_type, field, producer) computed from those verdicts, and finally feeds that accuracy back in as the confidence the extractor uses — replacing the hand-set YAML prior once there is enough evidence to beat it.

**Tech Stack:** Python 3.12, psycopg2, PostgreSQL (`proc` schema), pytest. No new dependencies.

## Global Constraints

- **New tables use the `bp_` prefix; indexes use `ix_bp_<table>_<cols>`.** Repo convention.
- **Migrations are additive and idempotent** — `ADD COLUMN IF NOT EXISTS`, `CREATE TABLE IF NOT EXISTS`. Existing rows keep working unchanged. One file per migration in `deploy/sql/YYYY-MM-DD_<name>.sql`.
- **Never fabricate a value.** A confidence with too little evidence behind it is not a confidence; fall back to the static prior and say so. Absent data stays absent.
- **Accuracy learning must never silently promote a document that would otherwise have been reviewed.** Learned confidence may *raise* trust only up to the existing thresholds; it may always *lower* it. A regression in learned accuracy makes the pipeline more cautious, never less.
- **The live extraction path is `src/services/extraction/`** (`.env` sets `EXTRACTION_RENOVATION_ENABLED=1`). `src/services/extraction_v3/` is the previous generation and is not to be modified — only its `yaml_schema/loader.py` is imported, for `load_doc_schema`.
- **Run tests with the environment loaded:** `set -a; . ./.env; set +a; ./venv/bin/python -m pytest ...`. Anything importing `api.main` or `repositories` also needs `PYTHONPATH=.:src`.
- **Baseline test state:** `tests/services/` is 916 passed / 91 failed. All 91 are pre-existing (79 `test_style_*`, 7 `test_langextract_adapter`, 4 `test_agent_actions`, 1 `test_negotiate_dashboard`). Do not "fix" them; do not add to them.
- **Commit on `Development`. Never push.**

## What already exists (do not rebuild)

| Thing | Where | State |
|---|---|---|
| `Candidate.pattern_name`, `Candidate.source` | `src/services/extraction/types.py:41` | Populated by the extractor; **discarded before persistence** |
| `proc.bp_extraction_provenance` | live DB | Table exists — `id, parent_table, parent_pk, field_name, source, anchor_ref, derivation_trace, confidence, attempt, extracted_at`. **0 rows, no writer** |
| `proc.bp_extraction_patterns` | live DB | `id, doc_type, field_name, pattern_type, pattern_value, anchor_patterns, confidence, created_at`. **0 rows, no writer** |
| `proc.bp_extraction_telemetry` | live DB | 83 rows. Per-document outcome; read by the hint proposer |
| `proc.bp_extraction_discrepancy` | live DB | `resolved_value`, `resolution_action`, `resolved_by`, `resolved_at` all recorded. **46 resolutions, all `dismiss`** — no `apply_value` yet, so the loop starts cold |
| Hint proposer | `src/services/extraction_feedback/proposer.py` | Reads telemetry *failure counts* per vendor. Never reads resolutions |
| `_compute_confidence_score` | `src/services/extraction/promotion.py:126` | **Completeness**, not correctness. Keep it; do not repurpose it |
| `resolve_dollar_currency` | `src/services/extraction/context_layer.py` | Shipped 2026-07-31. Consults `row["supplier_default_currency"]` — **nothing populates that key today** (Task 5 fixes this) |
| `bp_supplier.default_currency` | live DB | Populated on 5,000 of 5,027 suppliers |

---

### Task 1: Record what produced each persisted field

**Files:**
- Create: `src/services/extraction/provenance.py`
- Modify: `src/services/extraction/dispatch.py` (in `_persist_and_promote`, after `columns` is final and `doc_pk` is known)
- Test: `tests/services/test_extraction_provenance.py`

**Interfaces:**
- Consumes: `Candidate` from `src/services/extraction/types.py` (fields: `field`, `value`, `source`, `pattern_name`, `confidence`).
- Produces:
  - `producer_of(field: str, value: Any, candidates: list) -> tuple[str, str | None, float | None]` — returns `(source, pattern_name, confidence)` for the candidate whose value matches what was persisted; `("context_layer", None, None)` when no candidate matches (the AI layer wrote it).
  - `record(cur, *, parent_table: str, parent_pk: str, columns: dict, candidates: list, attempt: int = 1) -> int` — writes one `proc.bp_extraction_provenance` row per non-null column, returns rows written.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_extraction_provenance.py
"""What produced each value we persisted.

Without this, no human correction can be attributed to anything: "the currency was wrong"
is useless unless we know whether a regex, the NER gap-filler or the AI layer said it.
"""
from src.services.extraction.provenance import producer_of, record
from src.services.extraction.types import Candidate


def _cand(field, value, source="regex", pattern_name="anchored_currency_iso", confidence=0.92):
    return Candidate(field=field, value=value, span=None, source=source,
                     pattern_name=pattern_name, confidence=confidence)


def test_a_value_a_pattern_produced_is_attributed_to_that_pattern():
    cands = [_cand("currency", "GBP"), _cand("currency", "USD", pattern_name="dollar_symbol",
                                             confidence=0.58)]
    assert producer_of("currency", "GBP", cands) == ("regex", "anchored_currency_iso", 0.92)


def test_a_value_no_candidate_offered_is_attributed_to_the_ai_layer():
    # context_layer is the authoritative gate and routinely writes a value no regex found.
    # Recording that honestly is the point: an unattributed value is not a regex win.
    assert producer_of("currency", "CAD", [_cand("currency", "GBP")]) == ("context_layer", None, None)


def test_matching_is_on_the_persisted_value_not_the_field_alone():
    cands = [_cand("currency", "GBP"), _cand("invoice_id", "INV-1", pattern_name="id_anchor")]
    assert producer_of("invoice_id", "INV-1", cands)[1] == "id_anchor"


def test_comparison_tolerates_the_shapes_a_column_arrives_in():
    # The candidate carries the captured STRING; the column may hold a Decimal or int.
    from decimal import Decimal
    cands = [_cand("invoice_amount", "1,234.50", pattern_name="total_labelled")]
    assert producer_of("invoice_amount", Decimal("1234.50"), cands)[1] == "total_labelled"
    assert producer_of("quantity", 5, [_cand("quantity", "5", pattern_name="qty")])[1] == "qty"


def test_record_writes_one_row_per_non_null_column():
    class _Cur:
        def __init__(self): self.rows = []
        def execute(self, sql, params): self.rows.append(params)
    cur = _Cur()
    n = record(cur, parent_table="proc.bp_invoice_stg", parent_pk="INV-1",
               columns={"currency": "GBP", "invoice_amount": 100, "buyer_id": None},
               candidates=[_cand("currency", "GBP")])
    assert n == 2                                  # the NULL column is not provenance
    fields = {p[2] for p in cur.rows}
    assert fields == {"currency", "invoice_amount"}


def test_record_is_a_no_op_without_a_primary_key():
    class _Cur:
        def __init__(self): self.rows = []
        def execute(self, sql, params): self.rows.append(params)
    cur = _Cur()
    assert record(cur, parent_table="proc.bp_invoice_stg", parent_pk="",
                  columns={"currency": "GBP"}, candidates=[]) == 0
    assert cur.rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_extraction_provenance.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.extraction.provenance'`

- [ ] **Step 3: Write the implementation**

```python
# src/services/extraction/provenance.py
"""Which reader produced each value we kept.

An extraction is several readers arguing: regex patterns (L1), an engineered/NER gap-filler
(L2), a grounded judge (L3), and the context layer (AgentNick), which is the authoritative
gate and frequently writes a value no pattern found. When a human later corrects a field,
"the currency was wrong" tells us nothing unless we know WHO said it — so this records the
producer of every value at the moment it is persisted.

Deliberately cheap: one INSERT per non-null column, best-effort, never able to fail a
promotion. A provenance row is evidence, not part of the document.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

log = logging.getLogger(__name__)

AI_SOURCE = "context_layer"

_INSERT = """
    INSERT INTO proc.bp_extraction_provenance
        (parent_table, parent_pk, field_name, source, anchor_ref, confidence, attempt)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


def _comparable(value: Any) -> str:
    """One shape for comparing a captured string against a bound column value.

    The candidate holds what was literally on the page ("1,234.50"); the column holds what
    the type binder made of it (Decimal('1234.50')). Same value, different shapes.
    """
    if value is None:
        return ""
    if isinstance(value, (int, float, Decimal)):
        return format(Decimal(str(value)).normalize(), "f")
    text = str(value).strip().replace(",", "")
    try:
        return format(Decimal(text).normalize(), "f")
    except Exception:            # noqa: BLE001 — not a number, compare as text
        return str(value).strip().lower()


def producer_of(field: str, value: Any, candidates: list) -> tuple[str, str | None, float | None]:
    """(source, pattern_name, confidence) for whoever produced this value.

    Attribution is on the VALUE, not the field: several patterns offer a currency and only
    one of them is what got kept. No match means no reader we track offered it, which on
    this pipeline means the context layer wrote it — recorded as such rather than guessed
    at, because crediting a pattern for the AI layer's work would poison its score.
    """
    target = _comparable(value)
    for c in candidates or []:
        if getattr(c, "field", None) == field and _comparable(getattr(c, "value", None)) == target:
            return (getattr(c, "source", None) or AI_SOURCE,
                    getattr(c, "pattern_name", None),
                    getattr(c, "confidence", None))
    return AI_SOURCE, None, None


def record(cur, *, parent_table: str, parent_pk: str, columns: dict,
           candidates: list, attempt: int = 1) -> int:
    """One provenance row per non-null column. Returns rows written."""
    if not parent_pk:
        return 0
    written = 0
    for field, value in (columns or {}).items():
        if value is None or value == "":
            continue          # a field we did not fill has no producer
        source, pattern_name, confidence = producer_of(field, value, candidates)
        cur.execute(_INSERT, (parent_table, str(parent_pk), field, source,
                              pattern_name, confidence, attempt))
        written += 1
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_extraction_provenance.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Wire it into dispatch**

In `src/services/extraction/dispatch.py`, find where the row is persisted and `doc_pk` is known (the block containing `persistence.update_promotion_status(...)` / the `final_status = "promoted"` branch). Immediately after the persist succeeds, add:

```python
    # Record who produced each value. Best-effort by construction: provenance is evidence
    # about the extraction, not part of the document, and must never fail a promotion.
    try:
        from src.services.extraction import provenance as _prov
        from src.services.db import get_conn as _get_conn
        with _get_conn() as _pconn:
            _pcur = _pconn.cursor()
            _n = _prov.record(_pcur, parent_table=f"proc.bp_{doc_type}_stg",
                              parent_pk=str(doc_pk or ""), columns=columns,
                              candidates=candidates)
            _pconn.commit()
        log.info("dispatch: recorded provenance for %d field(s) on %s", _n, doc_pk)
    except Exception:
        log.exception("dispatch: provenance write failed (non-fatal)")
```

- [ ] **Step 6: Verify it writes against the live corpus**

Run:
```bash
set -a; . ./.env; set +a; PYTHONPATH=.:src ./venv/bin/python - <<'PY'
from src.services.db import get_conn
with get_conn() as c:
    cur = c.cursor()
    cur.execute("select source, count(*) from proc.bp_extraction_provenance group by 1 order by 2 desc")
    print("provenance by source:", cur.fetchall())
PY
```
Expected: empty until a document is extracted. Re-upload one document through the running stack, then re-run — expect rows with `source` in `regex`/`ner`/`judge`/`context_layer`.

- [ ] **Step 7: Commit**

```bash
git add src/services/extraction/provenance.py src/services/extraction/dispatch.py tests/services/test_extraction_provenance.py
git commit -m "feat(extraction): record which reader produced each persisted value"
```

---

### Task 2: Record the human's verdict against that provenance

**Files:**
- Create: `deploy/sql/2026-08-01_bp_extraction_verdict.sql`
- Create: `src/services/extraction_feedback/verdict.py`
- Modify: `src/services/extraction/promotion.py` (inside `apply_hitl_fixes`, in the `for field_name, resolved_value, action in fixes:` loop)
- Test: `tests/services/test_extraction_verdict.py`

**Interfaces:**
- Consumes: Task 1's `proc.bp_extraction_provenance` rows.
- Produces: `record_verdict(cur, *, doc_type: str, doc_pk: str, field_name: str, action: str, resolved_value, extracted_value, resolved_by: str | None) -> str | None` — writes one `proc.bp_extraction_verdict` row and returns the verdict (`"confirmed" | "corrected" | "rejected"`), or `None` when the action carries no verdict.

- [ ] **Step 1: Write the migration**

```sql
-- deploy/sql/2026-08-01_bp_extraction_verdict.sql
-- What a human decided about a value we extracted.
--
-- bp_extraction_discrepancy already records resolved_value / resolution_action / resolved_by,
-- but those describe the FINDING. This records the judgement on the VALUE, joined to the
-- reader that produced it (proc.bp_extraction_provenance), which is what makes an accuracy
-- rate computable per reader rather than per document.
--
-- Additive + idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_extraction_verdict (
    verdict_id       BIGSERIAL PRIMARY KEY,
    doc_type         TEXT        NOT NULL,
    doc_pk           TEXT        NOT NULL,
    field_name       TEXT        NOT NULL,
    -- Copied from provenance at verdict time so the rate survives a provenance purge and
    -- stays correct if the same field is later re-extracted by a different reader.
    source           TEXT,
    pattern_name     TEXT,
    prior_confidence NUMERIC,
    verdict          TEXT        NOT NULL CHECK (verdict IN ('confirmed', 'corrected', 'rejected')),
    extracted_value  TEXT,
    corrected_value  TEXT,
    decided_by       TEXT,
    decided_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE proc.bp_extraction_verdict IS
    'One row per human judgement on an extracted value. confirmed = a human saw it and let '
    'it stand; corrected = a human replaced it (the strongest negative signal there is); '
    'rejected = the finding was dismissed as not a real problem, which is a vote FOR the '
    'extracted value, not against it.';

CREATE INDEX IF NOT EXISTS ix_bp_extraction_verdict_doc_type_field
    ON proc.bp_extraction_verdict (doc_type, field_name);
CREATE INDEX IF NOT EXISTS ix_bp_extraction_verdict_source_pattern
    ON proc.bp_extraction_verdict (source, pattern_name);

COMMIT;
```

Apply it:
```bash
set -a; . ./.env; set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -f deploy/sql/2026-08-01_bp_extraction_verdict.sql
```
Expected: `BEGIN / CREATE TABLE / COMMENT / CREATE INDEX / CREATE INDEX / COMMIT`

- [ ] **Step 2: Write the failing test**

```python
# tests/services/test_extraction_verdict.py
"""What a human decided about a value, joined to the reader that produced it.

'dismiss' is the subtle one: dismissing a finding says the finding was wrong, which means
the extracted VALUE was right. Recording it as a negative would teach the system to distrust
exactly the readers people keep agreeing with.
"""
import pytest

from src.services.extraction_feedback.verdict import record_verdict, verdict_for


class _Cur:
    def __init__(self, provenance=None):
        self.rows, self._prov = [], provenance
        self.description = None
        self._result = []

    def execute(self, sql, params=()):
        if "FROM proc.bp_extraction_provenance" in sql:
            self._result = [self._prov] if self._prov else []
            return
        self.rows.append((sql, params))
        self._result = []

    def fetchone(self):
        return self._result[0] if self._result else None


def test_dismiss_is_a_vote_FOR_the_extracted_value():
    assert verdict_for("dismiss", resolved_value=None, extracted_value="GBP") == "rejected"


def test_replacing_the_value_is_the_strongest_negative():
    assert verdict_for("apply_value", resolved_value="CAD", extracted_value="USD") == "corrected"


def test_applying_the_same_value_back_is_a_confirmation_not_a_correction():
    # A human who retypes what was already there has agreed with it.
    assert verdict_for("apply_value", resolved_value="USD", extracted_value="USD") == "confirmed"


def test_clearing_a_value_is_a_correction():
    assert verdict_for("keep_null", resolved_value=None, extracted_value="USD") == "corrected"


def test_an_action_that_decides_nothing_has_no_verdict():
    assert verdict_for("flag", resolved_value=None, extracted_value="USD") is None
    assert verdict_for(None, resolved_value=None, extracted_value="USD") is None


def test_the_verdict_carries_the_producer_it_is_about():
    cur = _Cur(provenance=("regex", "dollar_symbol", 0.58))
    got = record_verdict(cur, doc_type="invoice", doc_pk="INV-1", field_name="currency",
                         action="apply_value", resolved_value="CAD", extracted_value="USD",
                         resolved_by="ap@example.com")
    assert got == "corrected"
    sql, params = cur.rows[0]
    assert "INSERT INTO proc.bp_extraction_verdict" in sql
    assert "regex" in params and "dollar_symbol" in params and "corrected" in params


def test_a_value_with_no_provenance_is_still_recorded():
    # Documents extracted before Task 1 shipped have no provenance. The verdict is still
    # worth keeping — it just cannot be attributed to a reader.
    cur = _Cur(provenance=None)
    assert record_verdict(cur, doc_type="invoice", doc_pk="INV-1", field_name="currency",
                          action="apply_value", resolved_value="CAD", extracted_value="USD",
                          resolved_by=None) == "corrected"
    _, params = cur.rows[0]
    assert None in params


def test_no_verdict_writes_no_row():
    cur = _Cur()
    assert record_verdict(cur, doc_type="invoice", doc_pk="INV-1", field_name="currency",
                          action="flag", resolved_value=None, extracted_value="USD",
                          resolved_by=None) is None
    assert cur.rows == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_extraction_verdict.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.extraction_feedback.verdict'`

- [ ] **Step 4: Write the implementation**

```python
# src/services/extraction_feedback/verdict.py
"""The human's judgement on an extracted value, attached to the reader that produced it.

This is the signal the feedback loop has been missing. The existing proposer counts how
often a field FAILED per vendor; it never reads what the human said the right answer was.
A correction is the most valuable datum the system can get — somebody looked at the
document and the value and told us we were wrong — and it was being thrown away.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

_PROVENANCE_SQL = """
    SELECT source, anchor_ref, confidence
      FROM proc.bp_extraction_provenance
     WHERE parent_table = %s AND parent_pk = %s AND field_name = %s
     ORDER BY id DESC
     LIMIT 1
"""

_INSERT = """
    INSERT INTO proc.bp_extraction_verdict
        (doc_type, doc_pk, field_name, source, pattern_name, prior_confidence,
         verdict, extracted_value, corrected_value, decided_by)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _same(a: Any, b: Any) -> bool:
    return str(a if a is not None else "").strip() == str(b if b is not None else "").strip()


def verdict_for(action: Optional[str], *, resolved_value: Any,
                extracted_value: Any) -> Optional[str]:
    """What a resolution says about the VALUE (not about the finding).

    'dismiss' reads as 'rejected' — the finding was rejected, which is agreement with what
    was extracted. Scoring it as a negative would teach the system to distrust precisely the
    readers people keep agreeing with.
    """
    verb = (action or "").strip().lower()
    if verb == "dismiss":
        return "rejected"
    if verb == "keep_null":
        return "corrected" if extracted_value not in (None, "") else "confirmed"
    if verb == "apply_value":
        return "confirmed" if _same(resolved_value, extracted_value) else "corrected"
    return None


def record_verdict(cur, *, doc_type: str, doc_pk: str, field_name: str,
                   action: Optional[str], resolved_value: Any, extracted_value: Any,
                   resolved_by: Optional[str]) -> Optional[str]:
    """Write one verdict row. Returns the verdict, or None when there is nothing to say."""
    verdict = verdict_for(action, resolved_value=resolved_value,
                          extracted_value=extracted_value)
    if verdict is None:
        return None
    source = pattern_name = prior = None
    try:
        cur.execute(_PROVENANCE_SQL, (f"proc.bp_{doc_type}_stg", str(doc_pk), field_name))
        row = cur.fetchone()
        if row:
            source, pattern_name, prior = row[0], row[1], row[2]
    except Exception:
        # Provenance is an optimisation for attribution; a verdict without it is still worth
        # keeping. Documents extracted before Task 1 shipped have none at all.
        log.debug("verdict: no provenance for %s.%s", doc_pk, field_name, exc_info=True)
    cur.execute(_INSERT, (doc_type, str(doc_pk), field_name, source, pattern_name, prior,
                          verdict,
                          None if extracted_value is None else str(extracted_value),
                          None if resolved_value is None else str(resolved_value),
                          resolved_by))
    return verdict
```

- [ ] **Step 5: Run test to verify it passes**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_extraction_verdict.py -q`
Expected: PASS (8 tests)

- [ ] **Step 6: Wire it into the HITL apply path**

In `src/services/extraction/promotion.py`, `apply_hitl_fixes` currently selects `field_name, resolved_value, resolution_action`. Widen the SELECT and record a verdict per fix. Replace the SELECT with:

```python
            cur.execute("""
                SELECT field_name, resolved_value, resolution_action, raw_value, resolved_by
                  FROM proc.bp_extraction_discrepancy
                 WHERE raw_id=%s AND status='resolved'
                   AND blocks_promotion=TRUE
            """, (raw_id,))
            fixes = cur.fetchall()
```

and the loop header with:

```python
            for field_name, resolved_value, action, extracted_value, resolved_by in fixes:
```

then, immediately before `conn.commit()` at the end of that `try`, add:

```python
            # Learn from it. The human has just told us what the value should have been —
            # the single most valuable signal the pipeline can receive, and until now it was
            # written to the discrepancy row and never read again.
            try:
                from src.services.extraction_feedback.verdict import record_verdict
                cur.execute(f"SELECT {_RAW_TO_STG[doc_type][1]} FROM {raw_t} WHERE raw_id=%s",
                            (raw_id,))
                _pk_row = cur.fetchone()
                _doc_pk = _pk_row[0] if _pk_row else None
                if _doc_pk:
                    for field_name, resolved_value, action, extracted_value, resolved_by in fixes:
                        record_verdict(cur, doc_type=doc_type, doc_pk=_doc_pk,
                                       field_name=field_name, action=action,
                                       resolved_value=resolved_value,
                                       extracted_value=extracted_value,
                                       resolved_by=resolved_by)
            except Exception:
                log.exception("apply_hitl_fixes: verdict capture failed (non-fatal)")
```

**Note for the implementer:** confirm `_RAW_TO_STG[doc_type]` is a `(table, pk_column)` tuple before using index `[1]` — read its definition at the top of `promotion.py`. If the shape differs, derive the pk column from `registry.schema` instead. Do not guess.

- [ ] **Step 7: Verify no regression in the promotion suite**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/ -q -k "promotion or hitl or verdict or provenance"`
Expected: PASS, no new failures.

- [ ] **Step 8: Commit**

```bash
git add deploy/sql/2026-08-01_bp_extraction_verdict.sql src/services/extraction_feedback/verdict.py src/services/extraction/promotion.py tests/services/test_extraction_verdict.py
git commit -m "feat(extraction): capture the human's verdict on every corrected value"
```

---

### Task 3: Turn verdicts into an observed accuracy

**Files:**
- Create: `src/services/extraction_feedback/accuracy.py`
- Test: `tests/services/test_extraction_accuracy.py`

**Interfaces:**
- Consumes: Task 2's `proc.bp_extraction_verdict` rows.
- Produces:
  - `observed_accuracy(rows: list[dict], *, min_sample: int = 8) -> dict[tuple[str, str, str | None], float]` — pure; maps `(doc_type, field_name, pattern_name_or_source)` to a rate in `[0, 1]`. Keys below `min_sample` are absent, not zero.
  - `load_accuracy(conn=None, *, window_days: int = 180, min_sample: int = 8) -> dict` — same map, read from the DB.
  - `MIN_SAMPLE = 8`

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_extraction_accuracy.py
"""How often each reader has actually been right.

The number that matters is not "how confident was the pattern" — that was hand-written in a
YAML file — but "how often did a human let this reader's answer stand". Two rules protect it
from being noise: nothing is scored until there is enough evidence to mean anything, and a
key with too little evidence is ABSENT rather than zero, so callers fall back to the static
prior instead of treating silence as failure.
"""
from src.services.extraction_feedback.accuracy import (
    MIN_SAMPLE, observed_accuracy,
)


def _v(verdict, field="currency", pattern="dollar_symbol", doc_type="invoice"):
    return {"doc_type": doc_type, "field_name": field, "pattern_name": pattern,
            "source": "regex", "verdict": verdict}


def test_a_reader_humans_keep_agreeing_with_scores_high():
    rows = [_v("confirmed")] * 9 + [_v("rejected")]
    acc = observed_accuracy(rows)
    assert acc[("invoice", "currency", "dollar_symbol")] == 1.0


def test_a_reader_humans_keep_correcting_scores_low():
    rows = [_v("corrected")] * 8 + [_v("confirmed")] * 2
    assert observed_accuracy(rows)[("invoice", "currency", "dollar_symbol")] == 0.2


def test_rejected_counts_AS_agreement():
    # Dismissing a finding says the value was fine. Counting it against the reader would
    # invert the whole signal.
    rows = [_v("rejected")] * MIN_SAMPLE
    assert observed_accuracy(rows)[("invoice", "currency", "dollar_symbol")] == 1.0


def test_too_little_evidence_is_absent_not_zero():
    rows = [_v("corrected")] * (MIN_SAMPLE - 1)
    assert ("invoice", "currency", "dollar_symbol") not in observed_accuracy(rows)


def test_the_sample_floor_is_adjustable_for_a_caller_that_wants_to_be_stricter():
    rows = [_v("confirmed")] * 10
    assert ("invoice", "currency", "dollar_symbol") not in observed_accuracy(rows, min_sample=20)


def test_readers_are_scored_separately_not_pooled():
    rows = ([_v("confirmed", pattern="anchored_currency_iso")] * MIN_SAMPLE
            + [_v("corrected", pattern="dollar_symbol")] * MIN_SAMPLE)
    acc = observed_accuracy(rows)
    assert acc[("invoice", "currency", "anchored_currency_iso")] == 1.0
    assert acc[("invoice", "currency", "dollar_symbol")] == 0.0


def test_the_ai_layer_is_scored_under_its_source_since_it_has_no_pattern():
    rows = [{"doc_type": "invoice", "field_name": "currency", "pattern_name": None,
             "source": "context_layer", "verdict": "confirmed"}] * MIN_SAMPLE
    assert observed_accuracy(rows)[("invoice", "currency", "context_layer")] == 1.0


def test_doc_types_are_scored_separately():
    rows = ([_v("confirmed")] * MIN_SAMPLE
            + [_v("corrected", doc_type="quote")] * MIN_SAMPLE)
    acc = observed_accuracy(rows)
    assert acc[("invoice", "currency", "dollar_symbol")] == 1.0
    assert acc[("quote", "currency", "dollar_symbol")] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_extraction_accuracy.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/services/extraction_feedback/accuracy.py
"""Observed accuracy per reader, from what humans actually did.

A rate here is a measurement, not a setting. It answers one question: when this reader
produced this field, how often did a person let the answer stand?
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

log = logging.getLogger(__name__)

# Below this many judgements a rate is noise, and acting on noise is worse than acting on
# the hand-set prior — one bad afternoon of corrections would otherwise switch a reader off.
MIN_SAMPLE = 8

# A verdict that counts as the reader having been RIGHT. 'rejected' means the human dismissed
# the finding, i.e. agreed with the extracted value.
_AGREES = {"confirmed", "rejected"}

_LOAD_SQL = """
    SELECT doc_type, field_name, pattern_name, source, verdict
      FROM proc.bp_extraction_verdict
     WHERE decided_at >= now() - (%s || ' days')::interval
"""


def _key(row: dict) -> tuple[str, str, Optional[str]]:
    # A pattern is the finest attribution we have; the AI layer has no pattern, so it is
    # scored under its source. Never pool the two — they are different readers.
    return (row.get("doc_type"), row.get("field_name"),
            row.get("pattern_name") or row.get("source"))


def observed_accuracy(rows: list[dict], *, min_sample: int = MIN_SAMPLE) -> dict:
    """Map (doc_type, field, reader) -> agreement rate, for readers with enough evidence.

    A key with fewer than ``min_sample`` judgements is ABSENT, not zero: callers must fall
    back to the static prior rather than read silence as failure.
    """
    agree: dict = defaultdict(int)
    total: dict = defaultdict(int)
    for row in rows or []:
        k = _key(row)
        total[k] += 1
        if row.get("verdict") in _AGREES:
            agree[k] += 1
    return {k: round(agree[k] / n, 4) for k, n in total.items() if n >= min_sample}


def load_accuracy(conn=None, *, window_days: int = 180,
                  min_sample: int = MIN_SAMPLE) -> dict:
    """observed_accuracy over the last ``window_days`` of verdicts.

    Windowed because a reader that was fixed six months ago should not be judged forever on
    what it did before the fix.
    """
    def _run(c):
        cur = c.cursor()
        cur.execute(_LOAD_SQL, (str(window_days),))
        cols = [d[0] for d in (cur.description or [])]
        return observed_accuracy([dict(zip(cols, r)) for r in cur.fetchall()],
                                 min_sample=min_sample)
    if conn is not None:
        return _run(conn)
    from src.services.db import get_conn
    with get_conn() as own:
        return _run(own)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_extraction_accuracy.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/services/extraction_feedback/accuracy.py tests/services/test_extraction_accuracy.py
git commit -m "feat(extraction): observed accuracy per reader, from human verdicts"
```

---

### Task 4: Let the measured rate replace the hand-set prior

**Files:**
- Modify: `src/services/extraction/pattern_registry.py` (`PatternRegistry._compile`, and a new `apply_observed` method)
- Test: `tests/services/test_pattern_confidence_learning.py`

**Interfaces:**
- Consumes: Task 3's `observed_accuracy` map.
- Produces: `PatternRegistry.apply_observed(accuracy: dict) -> int` — overrides `prior_confidence` on compiled patterns from the measured rate, returns the number changed. Also re-sorts each field's patterns, since the ordering *is* the preference between readers.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_pattern_confidence_learning.py
"""The hand-set prior is a starting guess. A measurement beats it.

Two safety rules, both non-negotiable: a measured rate may never raise a reader ABOVE its
hand-set prior (learning must not quietly promote documents that used to be reviewed), and a
reader with no measurement keeps exactly the prior it always had.
"""
from src.services.extraction.pattern_registry import PatternRegistry


def _priors(reg, field):
    return {p.name: p.prior_confidence for p in reg._by_field.get(field, [])}


def test_a_reader_humans_keep_correcting_is_trusted_less():
    reg = PatternRegistry("invoice")
    before = _priors(reg, "currency")["dollar_symbol"]
    changed = reg.apply_observed({("invoice", "currency", "dollar_symbol"): 0.20})
    assert changed == 1
    assert _priors(reg, "currency")["dollar_symbol"] == 0.20
    assert _priors(reg, "currency")["dollar_symbol"] < before


def test_a_measurement_never_raises_a_reader_above_its_hand_set_prior():
    # Learning may make the pipeline more cautious. It must never make it bolder than a
    # human intended — that would promote documents that used to stop for review.
    reg = PatternRegistry("invoice")
    before = _priors(reg, "currency")["dollar_symbol"]
    reg.apply_observed({("invoice", "currency", "dollar_symbol"): 1.0})
    assert _priors(reg, "currency")["dollar_symbol"] == before


def test_a_reader_with_no_measurement_is_untouched():
    reg = PatternRegistry("invoice")
    before = _priors(reg, "currency")
    reg.apply_observed({("invoice", "invoice_id", "some_other_pattern"): 0.1})
    assert _priors(reg, "currency") == before


def test_an_empty_map_changes_nothing():
    reg = PatternRegistry("invoice")
    before = _priors(reg, "currency")
    assert reg.apply_observed({}) == 0
    assert _priors(reg, "currency") == before


def test_the_preference_order_follows_the_measurement():
    # Order IS the preference between readers: the extractor takes the highest prior first.
    # A demoted reader must actually fall behind the ones that outperform it.
    reg = PatternRegistry("invoice")
    reg.apply_observed({("invoice", "currency", "anchored_currency_iso"): 0.10})
    order = [p.name for p in reg._by_field["currency"]]
    assert order[-1] == "anchored_currency_iso"


def test_another_doc_type_s_measurement_does_not_leak():
    reg = PatternRegistry("invoice")
    before = _priors(reg, "currency")
    reg.apply_observed({("quote", "currency", "dollar_symbol"): 0.05})
    assert _priors(reg, "currency") == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_pattern_confidence_learning.py -q`
Expected: FAIL — `AttributeError: 'PatternRegistry' object has no attribute 'apply_observed'`

- [ ] **Step 3: Write the implementation**

`CompiledPattern` is a dataclass; if it is frozen, rebuild the entry rather than mutating. Add to `PatternRegistry`:

```python
    def apply_observed(self, accuracy: dict) -> int:
        """Replace hand-set priors with measured agreement rates. Returns patterns changed.

        A prior is somebody's opening guess at how much a rule deserves to be believed. Once
        there is a measurement — how often a human let this reader's answer stand — the
        measurement is simply better information.

        It may only ever lower trust. Raising a reader above the prior a human set would let
        learning promote documents that used to stop for review, which is the one failure
        mode this must not have: the pipeline may become more cautious on its own, never
        bolder.
        """
        if not accuracy:
            return 0
        changed = 0
        for field, patterns in self._by_field.items():
            for i, pat in enumerate(patterns):
                rate = accuracy.get((self.doc_type, field, pat.name))
                if rate is None or rate >= pat.prior_confidence:
                    continue
                patterns[i] = replace(pat, prior_confidence=float(rate))
                changed += 1
            # Order is the preference between readers — the extractor tries the highest
            # prior first — so a demoted reader has to actually move.
            patterns.sort(key=lambda cp: cp.prior_confidence, reverse=True)
        return changed
```

Add `from dataclasses import replace` to the imports at the top of `pattern_registry.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_pattern_confidence_learning.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Load the measurement at extraction time**

In `src/services/extraction/dispatch.py`, immediately after the registry is built (`registry = PatternRegistry(doc_type)` or equivalent — read the file to find it), add:

```python
    # Apply what we have learned about each reader before extracting anything. Cached for
    # the process; a rate computed over 180 days does not move minute to minute, and a DB
    # round trip per document would be paid on every upload.
    try:
        from src.services.extraction_feedback.accuracy import load_accuracy
        _n = registry.apply_observed(_cached_accuracy())
        if _n:
            log.info("dispatch: %d pattern prior(s) replaced by measured accuracy", _n)
    except Exception:
        log.exception("dispatch: could not apply learned accuracy (using static priors)")
```

and near the top of the module:

```python
_ACCURACY_CACHE: dict = {}
_ACCURACY_CACHED_AT: float = 0.0
_ACCURACY_TTL_SECONDS = 900


def _cached_accuracy() -> dict:
    """Measured accuracy, refreshed at most every 15 minutes."""
    global _ACCURACY_CACHE, _ACCURACY_CACHED_AT
    import time
    from src.services.extraction_feedback.accuracy import load_accuracy
    now = time.monotonic()
    if now - _ACCURACY_CACHED_AT > _ACCURACY_TTL_SECONDS:
        _ACCURACY_CACHE = load_accuracy()
        _ACCURACY_CACHED_AT = now
    return _ACCURACY_CACHE
```

- [ ] **Step 6: Verify nothing regressed**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/ -q -k "pattern or dispatch or extraction or currency"`
Expected: PASS, no new failures. With an empty verdict table `apply_observed` changes nothing, so behaviour is identical to today.

- [ ] **Step 7: Commit**

```bash
git add src/services/extraction/pattern_registry.py src/services/extraction/dispatch.py tests/services/test_pattern_confidence_learning.py
git commit -m "feat(extraction): measured accuracy replaces the hand-set prior"
```

---

### Task 5: Learn this supplier's currency

**Files:**
- Create: `src/services/extraction_feedback/supplier_currency.py`
- Modify: `src/services/extraction/dispatch.py` (populate `supplier_default_currency` on `columns` before the `$` resolver runs — the currency block added 2026-07-31)
- Test: `tests/services/test_supplier_currency_learning.py`

**Interfaces:**
- Consumes: Task 2's verdicts; `proc.bp_supplier.default_currency`.
- Produces:
  - `learned_currency(rows: list[dict], *, min_agreements: int = 3) -> dict[str, str]` — pure; supplier_id → the currency humans have settled on.
  - `default_for(cur, supplier_id: str) -> str | None` — the learned answer if there is one, else `bp_supplier.default_currency`.

**Why this task exists:** `resolve_dollar_currency` (shipped 2026-07-31) already consults `row["supplier_default_currency"]` as its last resort, but **nothing populates that key** — the branch is dead. This task makes it live *and* makes it learn.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_supplier_currency_learning.py
"""If a person keeps telling us this supplier bills in CAD, stop asking.

This is the concrete payoff of the verdict loop: the fourth "$" invoice from a Canadian
supplier resolves itself instead of stopping for review, because three people already
answered the question.
"""
from src.services.extraction_feedback.supplier_currency import (
    learned_currency, MIN_AGREEMENTS,
)


def _row(supplier="SUP-A", value="CAD", verdict="corrected"):
    return {"supplier_id": supplier, "corrected_value": value, "verdict": verdict}


def test_three_people_saying_CAD_settles_it():
    assert learned_currency([_row()] * MIN_AGREEMENTS) == {"SUP-A": "CAD"}


def test_two_is_not_enough():
    assert learned_currency([_row()] * (MIN_AGREEMENTS - 1)) == {}


def test_a_supplier_people_disagree_about_is_left_alone():
    # Genuinely ambiguous, or the supplier really does bill in both. Either way, guessing
    # is exactly what this whole feature exists to stop.
    rows = [_row(value="CAD")] * MIN_AGREEMENTS + [_row(value="USD")] * MIN_AGREEMENTS
    assert learned_currency(rows) == {}


def test_a_clear_majority_settles_it_even_with_one_dissenter():
    rows = [_row(value="CAD")] * 5 + [_row(value="USD")]
    assert learned_currency(rows) == {"SUP-A": "CAD"}


def test_only_corrections_teach_it():
    # A confirmation says the value we already had was right; it does not tell us what this
    # supplier's currency IS when we had nothing.
    assert learned_currency([_row(verdict="confirmed")] * MIN_AGREEMENTS) == {}


def test_suppliers_are_learned_independently():
    rows = [_row(supplier="SUP-A", value="CAD")] * MIN_AGREEMENTS + \
           [_row(supplier="SUP-B", value="SGD")] * MIN_AGREEMENTS
    assert learned_currency(rows) == {"SUP-A": "CAD", "SUP-B": "SGD"}


def test_a_blank_correction_teaches_nothing():
    assert learned_currency([_row(value=None)] * MIN_AGREEMENTS) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_supplier_currency_learning.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/services/extraction_feedback/supplier_currency.py
"""What currency this supplier actually bills in, learned from corrections.

The supplier master has a default_currency, but it describes the supplier in general. A
correction describes THIS supplier's invoices, which is better evidence — and it is the
difference between stopping a buyer four times and stopping them three.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

log = logging.getLogger(__name__)

# Three independent people agreeing is a policy, not a coincidence. One is an opinion.
MIN_AGREEMENTS = 3
# Below this share of the votes the supplier is genuinely ambiguous — or really does bill in
# more than one currency — and guessing is what this feature exists to prevent.
MAJORITY = 0.75

_LOAD_SQL = """
    SELECT i.supplier_id, v.corrected_value, v.verdict
      FROM proc.bp_extraction_verdict v
      JOIN proc.bp_invoice_trgt i ON i.invoice_id = v.doc_pk
     WHERE v.field_name = 'currency' AND v.doc_type = 'invoice'
       AND i.supplier_id IS NOT NULL
"""


def learned_currency(rows: list[dict], *,
                     min_agreements: int = MIN_AGREEMENTS) -> dict[str, str]:
    """supplier_id -> the currency corrections have settled on, for suppliers where they have.

    Only CORRECTIONS teach: a confirmation says the value we had was right, which tells us
    nothing about what the currency is when we had nothing at all.
    """
    votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows or []:
        if row.get("verdict") != "corrected":
            continue
        value = str(row.get("corrected_value") or "").strip().upper()
        supplier = row.get("supplier_id")
        if value and supplier:
            votes[supplier][value] += 1

    out: dict[str, str] = {}
    for supplier, tally in votes.items():
        total = sum(tally.values())
        code, n = max(tally.items(), key=lambda kv: kv[1])
        if n >= min_agreements and (n / total) >= MAJORITY:
            out[supplier] = code
    return out


def default_for(cur, supplier_id: str) -> Optional[str]:
    """The currency to assume for this supplier: what people have taught us, else the
    supplier master's own default, else nothing."""
    if not supplier_id:
        return None
    cur.execute(_LOAD_SQL)
    cols = [d[0] for d in (cur.description or [])]
    learned = learned_currency([dict(zip(cols, r)) for r in cur.fetchall()])
    if supplier_id in learned:
        return learned[supplier_id]
    cur.execute("SELECT default_currency FROM proc.bp_supplier WHERE supplier_id = %s",
                (supplier_id,))
    row = cur.fetchone()
    return (row[0] or None) if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_supplier_currency_learning.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Feed it to the resolver**

In `src/services/extraction/dispatch.py`, in the currency block added 2026-07-31, immediately before `_resolved = _ccy_rules.resolve_dollar_currency(columns, full_text)`, add:

```python
            # resolve_dollar_currency consults this as its LAST resort, after anything the
            # document itself says. Nothing populated it until now, so that branch was dead.
            try:
                from src.services.extraction_feedback.supplier_currency import default_for
                from src.services.db import get_conn as _sc_conn
                _sup = columns.get("supplier_id")
                if _sup:
                    with _sc_conn() as _c:
                        columns["supplier_default_currency"] = default_for(_c.cursor(), _sup)
            except Exception:
                log.exception("dispatch: supplier currency lookup failed (non-fatal)")
```

**Important:** `supplier_default_currency` is a hint for the resolver, not a column. Remove it from `columns` before persistence:

```python
        columns.pop("supplier_default_currency", None)
```
Place this immediately after the currency block ends, before the `# Invariants` comment.

- [ ] **Step 6: Verify the resolver now reaches its last resort**

Run:
```bash
set -a; . ./.env; set +a; PYTHONPATH=.:src ./venv/bin/python - <<'PY'
from src.services.extraction.context_layer import resolve_dollar_currency
print(resolve_dollar_currency({"supplier_default_currency": "SGD"}, "Total $1,200.00"))
PY
```
Expected: `('SGD', "the supplier master records SGD as this supplier's currency")`

- [ ] **Step 7: Commit**

```bash
git add src/services/extraction_feedback/supplier_currency.py src/services/extraction/dispatch.py tests/services/test_supplier_currency_learning.py
git commit -m "feat(extraction): learn a supplier's currency from repeated corrections"
```

---

### Task 6: Say how ACCURATE a document is, not just how complete

**Files:**
- Create: `deploy/sql/2026-08-01_bp_extraction_accuracy_score.sql`
- Modify: `src/services/extraction/promotion.py` (alongside `_compute_confidence_score`)
- Test: `tests/services/test_accuracy_score.py`

**Interfaces:**
- Consumes: Task 3's `load_accuracy`; Task 1's provenance.
- Produces: `_compute_accuracy_score(doc_type: str, columns: dict, candidates: list, accuracy: dict) -> Decimal | None` — the mean measured accuracy of the readers that produced this document's fields, or `None` when none of them has been measured.

**Why separate:** `confidence_score` is completeness and must stay that way — plenty of things read it. This is the second number: *given who read this document, how often has that been right?* A document can be 100% complete and entirely wrong.

- [ ] **Step 1: Write the migration**

```sql
-- deploy/sql/2026-08-01_bp_extraction_accuracy_score.sql
-- How much the READERS of this document have historically been trusted.
--
-- confidence_score is completeness: how many fields are filled. It rises when a human types
-- a value in, whether or not the value is right. This is the other half — the mean measured
-- agreement rate of whoever produced this document's fields — so "complete" and "correct"
-- stop being the same word.
--
-- NULL means no reader on this document has enough verdicts to have a rate yet, which is
-- the honest answer and is what every row will say until the loop has run for a while.
--
-- Additive + idempotent.
BEGIN;

ALTER TABLE proc.bp_invoice_stg         ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;
ALTER TABLE proc.bp_invoice_trgt        ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;
ALTER TABLE proc.bp_quote_stg           ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;
ALTER TABLE proc.bp_quote_trgt          ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;
ALTER TABLE proc.bp_purchase_order_stg  ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;
ALTER TABLE proc.bp_purchase_order_trgt ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;

COMMENT ON COLUMN proc.bp_invoice_trgt.accuracy_score IS
    'Mean measured agreement rate (0-100) of the readers that produced this document''s '
    'fields, from proc.bp_extraction_verdict. NULL = not enough verdicts yet. Distinct from '
    'confidence_score, which measures completeness.';

COMMIT;
```

**Implementer note:** confirm each table exists before running — `\dt proc.bp_*_stg` — and drop any line whose table is absent rather than inventing it.

- [ ] **Step 2: Write the failing test**

```python
# tests/services/test_accuracy_score.py
"""Complete and correct are different words.

A document can have every field filled and every one of them wrong. confidence_score cannot
tell you that — it counts filled fields. This can, once there are verdicts behind it.
"""
from decimal import Decimal

from src.services.extraction.promotion import _compute_accuracy_score
from src.services.extraction.types import Candidate


def _cand(field, value, pattern):
    return Candidate(field=field, value=value, span=None, source="regex",
                     pattern_name=pattern, confidence=0.9)


def test_no_measurements_means_no_score_not_a_zero():
    # Every row will say this until the loop has run for a while. Zero would read as
    # "we know this document is wrong", which is a different and false claim.
    assert _compute_accuracy_score("invoice", {"currency": "GBP"},
                                   [_cand("currency", "GBP", "iso_code_in_text")], {}) is None


def test_it_averages_the_readers_that_actually_produced_this_document():
    acc = {("invoice", "currency", "iso_code_in_text"): 1.0,
           ("invoice", "invoice_amount", "total_labelled"): 0.5}
    got = _compute_accuracy_score(
        "invoice", {"currency": "GBP", "invoice_amount": 100},
        [_cand("currency", "GBP", "iso_code_in_text"),
         _cand("invoice_amount", "100", "total_labelled")], acc)
    assert got == Decimal("75.00")


def test_an_unmeasured_reader_is_skipped_not_counted_as_zero():
    acc = {("invoice", "currency", "iso_code_in_text"): 1.0}
    got = _compute_accuracy_score(
        "invoice", {"currency": "GBP", "invoice_amount": 100},
        [_cand("currency", "GBP", "iso_code_in_text"),
         _cand("invoice_amount", "100", "unmeasured_pattern")], acc)
    assert got == Decimal("100.00")


def test_a_document_whose_readers_keep_being_corrected_scores_low():
    acc = {("invoice", "currency", "dollar_symbol"): 0.2}
    got = _compute_accuracy_score("invoice", {"currency": "USD"},
                                  [_cand("currency", "USD", "dollar_symbol")], acc)
    assert got == Decimal("20.00")


def test_null_columns_contribute_nothing():
    acc = {("invoice", "currency", "iso_code_in_text"): 1.0}
    got = _compute_accuracy_score("invoice", {"currency": "GBP", "buyer_id": None},
                                  [_cand("currency", "GBP", "iso_code_in_text")], acc)
    assert got == Decimal("100.00")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_accuracy_score.py -q`
Expected: FAIL — `ImportError: cannot import name '_compute_accuracy_score'`

- [ ] **Step 4: Write the implementation**

Add to `src/services/extraction/promotion.py`, directly below `_compute_confidence_score`:

```python
def _compute_accuracy_score(
    doc_type: str, columns: dict[str, Any], candidates: list, accuracy: dict,
) -> Decimal | None:
    """0-100: how often the readers that produced THIS document have been right.

    Distinct from _compute_confidence_score, which measures completeness — how many fields
    are filled. A document can be entirely complete and entirely wrong, and until now
    nothing could tell the difference.

    None when no reader here has enough verdicts to have a rate. That is the honest answer,
    and it is what every document will say until the feedback loop has been running a while.
    A zero would read as "we know this is wrong", which is a different claim.
    """
    from src.services.extraction.provenance import producer_of

    rates = []
    for field, value in (columns or {}).items():
        if value is None or value == "":
            continue
        _source, pattern_name, _conf = producer_of(field, value, candidates)
        rate = accuracy.get((doc_type, field, pattern_name or _source))
        if rate is not None:
            rates.append(rate)
    if not rates:
        return None
    return Decimal(f"{(sum(rates) / len(rates)) * 100:.2f}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_accuracy_score.py -q`
Expected: PASS (5 tests)

- [ ] **Step 6: Persist it alongside the confidence score**

In `promotion.py`, find `raw_data["confidence_score"] = _compute_confidence_score(...)` (around line 696) and add immediately after:

```python
            # The second number: complete is not the same as correct.
            try:
                from src.services.extraction_feedback.accuracy import load_accuracy
                raw_data["accuracy_score"] = _compute_accuracy_score(
                    doc_type, raw_data, candidates, load_accuracy(),
                )
            except Exception:
                log.exception("promotion: accuracy score unavailable (non-fatal)")
```

**Implementer note:** `candidates` may not be in scope in `promote()`. If it is not, thread it through from `dispatch` or read provenance back from `proc.bp_extraction_provenance` for this `parent_pk` instead of re-deriving it. Read the function before editing; do not invent a variable.

- [ ] **Step 7: Verify no regression**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/ -q -k "promotion or accuracy or confidence"`
Expected: PASS, no new failures.

- [ ] **Step 8: Commit**

```bash
git add deploy/sql/2026-08-01_bp_extraction_accuracy_score.sql src/services/extraction/promotion.py tests/services/test_accuracy_score.py
git commit -m "feat(extraction): an accuracy score, separate from completeness"
```

---

### Task 7: Prove the loop closes, and write down what it does

**Files:**
- Create: `scripts/show_extraction_learning.py`
- Modify: `docs/superpowers/specs/` — add `2026-08-01-extraction-confidence-learning.md` recording the measured before/after
- Test: `tests/services/test_learning_loop_e2e.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `scripts/show_extraction_learning.py` — prints, for each reader, its static prior, its measured rate, its verdict count, and whether the measurement is currently in force.

- [ ] **Step 1: Write the end-to-end test**

```python
# tests/services/test_learning_loop_e2e.py
"""The whole loop, in one test: extract -> human corrects -> the reader is trusted less.

This is the claim the feature makes. Everything else is machinery.
"""
from src.services.extraction.pattern_registry import PatternRegistry
from src.services.extraction_feedback.accuracy import MIN_SAMPLE, observed_accuracy
from src.services.extraction_feedback.verdict import verdict_for


def test_repeated_corrections_demote_the_reader_that_keeps_being_wrong():
    # 1. A reader produces a value; a human replaces it. Ten times.
    verdicts = []
    for _ in range(MIN_SAMPLE + 2):
        v = verdict_for("apply_value", resolved_value="CAD", extracted_value="USD")
        assert v == "corrected"
        verdicts.append({"doc_type": "invoice", "field_name": "currency",
                         "pattern_name": "dollar_symbol", "source": "regex", "verdict": v})

    # 2. That becomes a measured rate.
    acc = observed_accuracy(verdicts)
    assert acc[("invoice", "currency", "dollar_symbol")] == 0.0

    # 3. Which the extractor then believes over its hand-set prior.
    reg = PatternRegistry("invoice")
    before = {p.name: p.prior_confidence for p in reg._by_field["currency"]}
    assert before["dollar_symbol"] > 0
    reg.apply_observed(acc)
    after = {p.name: p.prior_confidence for p in reg._by_field["currency"]}
    assert after["dollar_symbol"] == 0.0
    assert after["dollar_symbol"] < before["dollar_symbol"]

    # 4. And it is now the reader of last resort rather than a trusted one.
    assert [p.name for p in reg._by_field["currency"]][-1] == "dollar_symbol"


def test_agreement_leaves_a_good_reader_exactly_where_it_was():
    verdicts = [{"doc_type": "invoice", "field_name": "currency",
                 "pattern_name": "anchored_currency_iso", "source": "regex",
                 "verdict": "confirmed"}] * (MIN_SAMPLE + 2)
    reg = PatternRegistry("invoice")
    before = {p.name: p.prior_confidence for p in reg._by_field["currency"]}
    reg.apply_observed(observed_accuracy(verdicts))
    after = {p.name: p.prior_confidence for p in reg._by_field["currency"]}
    assert after == before
```

- [ ] **Step 2: Run it**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/test_learning_loop_e2e.py -q`
Expected: PASS (2 tests)

- [ ] **Step 3: Write the inspection script**

```python
# scripts/show_extraction_learning.py
"""What the pipeline has learned, and what it is still guessing at.

    set -a; . ./.env; set +a
    PYTHONPATH=.:src ./venv/bin/python scripts/show_extraction_learning.py
"""
from __future__ import annotations

from src.services.extraction.pattern_registry import PatternRegistry
from src.services.extraction_feedback.accuracy import MIN_SAMPLE, load_accuracy
from src.services.db import get_conn


def main() -> int:
    accuracy = load_accuracy()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT doc_type, field_name, COALESCE(pattern_name, source),
                              COUNT(*),
                              COUNT(*) FILTER (WHERE verdict IN ('confirmed','rejected'))
                         FROM proc.bp_extraction_verdict
                        GROUP BY 1,2,3 ORDER BY 4 DESC""")
        counts = cur.fetchall()

    if not counts:
        print("No verdicts recorded yet — every reader is still on its hand-set prior.")
        print(f"A reader needs {MIN_SAMPLE} judgements before its measured rate is used.")
        return 0

    print(f"{'doc_type':<10} {'field':<20} {'reader':<26} {'n':>4} {'measured':>9} "
          f"{'prior':>6}  in force")
    for doc_type, field, reader, n, agreed in counts:
        rate = accuracy.get((doc_type, field, reader))
        prior = None
        try:
            reg = PatternRegistry(doc_type)
            prior = next((p.prior_confidence for p in reg._by_field.get(field, [])
                          if p.name == reader), None)
        except Exception:
            pass
        in_force = "yes" if (rate is not None and prior is not None and rate < prior) else "no"
        print(f"{doc_type:<10} {field:<20} {reader:<26} {n:>4} "
              f"{(f'{rate:.2f}' if rate is not None else '—'):>9} "
              f"{(f'{prior:.2f}' if prior is not None else '—'):>6}  {in_force}"
              f"{'' if n >= MIN_SAMPLE else f'  (needs {MIN_SAMPLE - n} more)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it against the live database**

Run: `set -a; . ./.env; set +a; PYTHONPATH=.:src ./venv/bin/python scripts/show_extraction_learning.py`
Expected on first run: `No verdicts recorded yet — every reader is still on its hand-set prior.` — which is correct, since the live corpus has 46 resolutions and all of them are `dismiss` on findings that predate Task 2.

- [ ] **Step 5: Write the spec**

Create `docs/superpowers/specs/2026-08-01-extraction-confidence-learning.md` containing, at minimum: what each of the four tables/columns holds; the two safety rules (measurement may only lower trust; too little evidence means absent, not zero); the `MIN_SAMPLE = 8` and `MIN_AGREEMENTS = 3` thresholds and why they are those numbers; and the measured output of `scripts/show_extraction_learning.py` at the time of writing.

- [ ] **Step 6: Full regression sweep**

Run: `set -a; . ./.env; set +a; ./venv/bin/python -m pytest tests/services/ -q | tail -2`
Expected: 91 pre-existing failures, no more. Passed count up by the ~40 tests this plan adds.

- [ ] **Step 7: Commit**

```bash
git add scripts/show_extraction_learning.py tests/services/test_learning_loop_e2e.py docs/superpowers/specs/2026-08-01-extraction-confidence-learning.md
git commit -m "feat(extraction): prove the learning loop closes, and document it"
```

---

## Self-review record

- **Spec coverage:** the three things I told the user were missing all have tasks — record what produced a value (T1), read the human's verdict (T2), score readers on outcomes (T3) and use the score (T4); learn the per-supplier answer (T5); separate complete from correct (T6); prove and document (T7).
- **Known cold start:** the live corpus has 46 resolutions, every one a `dismiss`, all on findings that predate the verdict table. The loop therefore starts with **zero** learned rates and every reader on its static prior — behaviour identical to today. This is expected, not a defect, and Task 7 Step 4 asserts exactly that. The loop only becomes visible as people work the 300 duplicate findings and the currency-ambiguity findings that Task 3 of the currency work now raises.
- **Deliberate deviation:** I am NOT touching `_compute_confidence_score`. Repurposing it would silently change the meaning of a number several other services already read (`bp_extraction_telemetry.confidence`, the training-example collector's ≥0.90 gate). The new number sits beside it.
- **Type consistency:** `producer_of` returns `(source, pattern_name, confidence)` in T1 and is consumed with that shape in T2 and T6. The accuracy map key is `(doc_type, field_name, pattern_name or source)` in T3 and is read with that key in T4 and T6. `MIN_SAMPLE` is defined once in T3 and imported in T4/T7.
- **Risk to watch:** T4 Step 5 caches the accuracy map per process for 15 minutes. If a reviewer wants a correction to take effect immediately, that TTL is the knob — not a redesign.
