# Contract Structures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recognise every contract structure on upload, store that answer as a fact, and propose each contract document's parent so the relationship maths has something to connect.

**Architecture:** The vocabulary gains two structures (`order form`, `sales order`) and a data-driven stand-down rule so a child structure only claims a page that names its parent. The resolved structure is written to `proc.bp_contract_raw` and carried into `proc.bp_contracts`. A new `contract_hierarchy` scoring profile registers through `linking_engine.register_profile()` and writes parent proposals into `proc.bp_extraction_discrepancy` — the queue buyers already work — never auto-linking.

**Tech Stack:** Python 3, psycopg2, PostgreSQL (`proc` schema), pytest. No new dependencies.

**Spec:** `specs/2026-10-02-contract-structures-design.md` (APPROVED by Nick 2026-10-02, both stated assumptions confirmed)

**Predecessor rulings:** `specs/2026-10-01-document-relationship-layer-rulings.md` — read before changing any classification behaviour.

---

## Global Constraints

- **Every guard must be proven to fail.** Break the behaviour on purpose, watch the test go red, restore, watch it go green. A test whose red state was never observed is not evidence. Fourteen guards in the predecessor plan were found green while checking nothing.
- **Migrations are additive, idempotent and reversible.** Every `deploy/sql/<name>.sql` gets a `deploy/sql/<name>_rollback.sql` sibling. `ON CONFLICT DO NOTHING` on seeds so a re-run never overwrites a row a human has edited.
- **Both databases.** Every migration applies to `bp_testdb` (the configured `.env` database) **and** `bp_sqldb`. Task 11 owns that; no task is done until Task 11 covers it.
- **Never modify source data.** `proc.bp_contract_master.contract_type` is read, never written.
- **NULL when absent.** A page that states no structure stores `NULL`, never a guessed or declared-by-default value.
- **Alias append order is part of the data.** `tests/services/concepts/test_concept_table.py::test_document_type_rows_equal_the_seed_column_for_column` compares the `aliases` array as an **ordered list** against `seed.py`. New aliases go last, in the same order in both the migration and the seed.
- **New tables take the `bp_` prefix; indexes are `ix_bp_<table>_<cols>`.** This plan creates no new table.
- **`get_conn()` is autocommit.** `rollback()` is a no-op and `FOR UPDATE` locks end with the statement. Set `conn.autocommit = False` explicitly when a multi-statement transaction is required, as `persistence.write_raw` does.
- **Findings carry a normalised `source_file`.** The open-row key is `(doc_type, doc_pk_candidate, coalesce(source_file,''), issue_type, field_name)` and `persistence.normalise_source_file()` must be applied on write. Never reduce it to a basename.
- **Test invocation:**
  ```bash
  set -a && . ./.env && set +a
  CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest <path> -v
  ```
  Add `PROCWISE_TEST_LIVE_DB=1` for tests that read the database. `CUDA_VISIBLE_DEVICES=""` keeps the suite off the GPU. Use `./venv/bin/python` (the test venv), not `.venv`.
- **Committing: this checkout and its index are shared with another session.** Never `git add -A`, never a blind `git commit -o`. Use a private index:
  ```bash
  export GIT_INDEX_FILE=/tmp/claude-1001/<session>/scratchpad/idx
  git read-tree HEAD && git add <my paths> && TREE=$(git write-tree)
  COMMIT=$(printf '%s\n' "<subject>" "" "<body>" "" "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" \
           | git commit-tree "$TREE" -p HEAD)
  git update-ref refs/heads/Development "$COMMIT"
  unset GIT_INDEX_FILE && git reset -q HEAD -- <my paths>
  ```
  Confirm afterwards that `git diff --cached --name-status | wc -l` still shows the other session's staged entries.
- **Work stays on `Development`.** Never push to `main`.
- **Implementers must not be Haiku** — it has twice swept another session's staged work into commits on this shared index.

---

## Review Focus

Five input classes the spec implies but no task's own tests would otherwise exercise. Each line's test is added to the task that owns the code, named in brackets.

1. **A title naming two structures at once** — "Order Form and Framework Agreement" in one title segment must stay `unresolved` with both candidates, not silently pick one. The stand-down rule must not turn a genuine tie into a confident single answer. [Task 3]
2. **`requires_parent_evidence = true` with an empty `parent_evidence_phrases`** — the structure could then never match anything, muting it silently. A validation check must report it. [Task 2]
3. **A re-read of the same contract document** — the stored structure must refresh, not go stale. `_stg` updates while `_trgt` stays stale is an existing, measured failure in this product. [Task 6]
4. **Two different contract documents carrying the same `contract_id` value** — the proposal's idempotency key must separate them by document, or one document's proposal suppresses the other's. This is exactly the collision that cost 65 days of findings. [Task 10]
5. **A child structure whose candidate parent set is empty** — nothing proposed must be distinguishable from everything already parented, or an empty screen reads as success. [Task 10]

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `tests/fixtures/contract_structures/classification_baseline.json` | Create: today's resolver outcome for all 53 corpus documents with stored text | 1 |
| `tests/services/extraction/test_classification_baseline.py` | Create: fails if any of the 53 documents resolves differently | 1 |
| `deploy/sql/2026-10-02_document_type_parent_evidence.sql` + `_rollback.sql` | Create: the two new columns | 2 |
| `src/services/concepts/seed.py` | Modify: `DocumentType` gains two fields; two new rows | 2, 4 |
| `src/services/concepts/vocabulary.py` | Modify: `_DOC_TYPE_SQL`, `build_vocabulary`, `SEED_VOCABULARY` thread the two columns | 2 |
| `src/services/concepts/validate.py` | Modify: a flagged structure with no phrases is a violation | 2 |
| `src/services/extraction/type_resolver.py` | Modify: stand-down at the one status check (line 417-418); `refined` agreement | 3, 5 |
| `deploy/sql/2026-10-02_document_type_order_form_sales_order.sql` + `_rollback.sql` | Create: the two new structure rows | 4 |
| `deploy/sql/2026-10-02_contract_raw_resolved_type.sql` + `_rollback.sql` | Create: three columns on `bp_contract_raw` and `bp_contracts` | 6 |
| `src/services/extraction/persistence.py` | Modify: `write_raw` accepts and writes the three columns | 6 |
| `src/services/extraction/dispatch.py` | Modify: pass the resolution into `write_raw` | 6 |
| `src/services/concepts/contract_type_map.py` | Create: free-text `contract_type` → concept code, read-only | 7 |
| `extraction_schemas/contract.yaml` | Modify: `framework_ref`, `parent_agreement_ref` | 8 |
| `src/services/graph_resolution/profiles/contract_hierarchy.py` | Create: the five signal comparators and the profile | 9 |
| `src/services/contract_links.py` | Create: the runner, the proposal writer, `confirm()` | 10 |

---

## Task 1: Capture the classification baseline

Nothing may change how a document that classifies correctly today classifies tomorrow (spec §1 criterion 5). That claim needs a committed baseline, captured **before** any change, or later tasks are asserting it rather than measuring it.

**Files:**
- Create: `tests/fixtures/contract_structures/classification_baseline.json`
- Create: `tests/services/extraction/test_classification_baseline.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the fixture file, and `_resolve_stored_documents()` — a helper every later regression check reuses, returning `dict[str, dict]` keyed by `source_file` with `{"agreement": str, "evidence_concept": str | None, "status": str, "candidates": list[str]}`.

- [ ] **Step 1: Write the baseline capture script**

Create `tests/services/extraction/test_classification_baseline.py`:

```python
"""Today's classification outcome for every document whose parsed text is stored.

The contract-structures work (specs/2026-10-02-contract-structures-design.md)
adds two structures and a stand-down rule. Its first promise is that no document
which classifies correctly today classifies differently afterwards. That is a
measurement, not an assertion, so the outcome is captured here as a fixture and
compared on every run.

115 raw rows carry parsed text, covering 57 distinct source_file values of which 53 are
corpus documents under `documents/` (the same document
has several raw rows from repeated extractions); the newest row per source_file
is the one read.

Live-only. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/extraction/test_classification_baseline.py -v

To re-capture after an INTENDED change, and only then:
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python \
        tests/services/extraction/test_classification_baseline.py --recapture
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

BASELINE = Path(__file__).resolve().parents[2] / "fixtures" / "contract_structures" / "classification_baseline.json"

#: The uploader's declared concept, derived from the S3 key's zone folder. The
#: live path declares it from proc.process_monitor.category; this reproduces the
#: same answer for a stored document without needing the monitor row.
_ZONE_TO_CONCEPT = (
    ("/quote/", "doctype.quote"),
    ("/invoice/", "doctype.invoice"),
    ("/po/", "doctype.order"),
    ("/purchase", "doctype.order"),
    ("/contract", "doctype.contract_unspecified"),
)

_RAW_TABLES = ("bp_quote_raw", "bp_invoice_raw", "bp_purchase_order_raw", "bp_contract_raw")


def declared_concept_for(source_file: str) -> str | None:
    low = (source_file or "").lower()
    for seg, code in _ZONE_TO_CONCEPT:
        if seg in low:
            return code
    return None


def _stored_documents() -> dict[str, str]:
    """source_file -> full_text, taking the NEWEST raw row per document."""
    from src.services.db import get_conn

    out: dict[str, str] = {}
    with get_conn() as conn:
        cur = conn.cursor()
        for table in _RAW_TABLES:
            cur.execute(
                f"""SELECT source_file, parser_snapshot
                      FROM proc.{table}
                     WHERE parser_snapshot IS NOT NULL
                     ORDER BY raw_id ASC"""
            )
            for source_file, snapshot in cur.fetchall():
                snap = snapshot if isinstance(snapshot, dict) else json.loads(snapshot)
                text = (snap or {}).get("full_text") or ""
                if text.strip():
                    out[source_file] = text   # ORDER BY ASC: the last write wins
    return out


def _resolve_stored_documents() -> dict[str, dict]:
    from src.services.extraction.type_resolver import resolve_document_type

    result = {}
    for source_file, text in _stored_documents().items():
        r = resolve_document_type(
            declared_concept=declared_concept_for(source_file), full_text=text,
        )
        result[source_file] = {
            "agreement": r.agreement,
            "evidence_concept": r.evidence_concept,
            "status": r.status,
            "candidates": list(r.candidates),
        }
    return result


def test_the_baseline_covers_every_stored_document():
    """A baseline that silently lost documents would pass while checking nothing."""
    assert BASELINE.exists(), f"baseline fixture missing: {BASELINE}"
    recorded = json.loads(BASELINE.read_text())
    live = _stored_documents()
    assert len(recorded) == len(live), (
        f"baseline holds {len(recorded)} documents, the database has {len(live)}. "
        "A document was added or removed; re-capture deliberately."
    )


def test_no_stored_document_classifies_differently_than_the_baseline():
    recorded = json.loads(BASELINE.read_text())
    live = _resolve_stored_documents()
    drift = []
    for source_file, want in sorted(recorded.items()):
        got = live.get(source_file)
        if got is None:
            drift.append(f"{source_file}: GONE from the database")
        elif got != want:
            drift.append(f"{source_file}:\n      baseline={want}\n      now     ={got}")
    assert not drift, (
        f"{len(drift)} of {len(recorded)} documents classify differently:\n  "
        + "\n  ".join(drift)
    )


if __name__ == "__main__":
    if "--recapture" in sys.argv:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        data = _resolve_stored_documents()
        BASELINE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"captured {len(data)} documents to {BASELINE}")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_classification_baseline.py -v
```

Expected: both tests FAIL with "baseline fixture missing".

- [ ] **Step 3: Capture the baseline**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python \
    tests/services/extraction/test_classification_baseline.py --recapture
```

Expected: `captured 53 documents to .../classification_baseline.json`.

If the count is not 53, stop and report it. The design's measurements rest on that number; a different one means the corpus changed and §4's figures need re-measuring before anything is built.

- [ ] **Step 4: Run the test to verify it passes**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_classification_baseline.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Prove the guard fails (required)**

Edit the fixture by hand: change one document's `"agreement"` from `"agreed"` to `"disagreed"`.

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_classification_baseline.py::test_no_stored_document_classifies_differently_than_the_baseline -v
```

Expected: FAIL, naming that one document with `baseline=` and `now=` lines. Then restore the fixture (`git checkout -- <fixture>` is unsafe on this shared index — re-run `--recapture` instead) and confirm it passes again.

Also prove the coverage guard: delete one entry from the fixture and confirm `test_the_baseline_covers_every_stored_document` fails with "baseline holds 56 documents, the database has 53". Re-capture to restore.

- [ ] **Step 6: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add tests/fixtures/contract_structures/classification_baseline.json \
        tests/services/extraction/test_classification_baseline.py
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "test(contracts): baseline every stored document's classification" "" \
  "Task 1 of specs/2026-10-02-contract-structures-plan.md. 53 documents," \
  "captured before any change, so 'nothing regresses' is measured rather" \
  "than asserted. Both guards proven red." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- tests/fixtures/contract_structures/classification_baseline.json \
                     tests/services/extraction/test_classification_baseline.py
git diff --cached --name-status | wc -l   # the other session's staged entries must survive
```

---

## Task 2: The parent-evidence columns, threaded end to end

Which structures the stand-down rule governs must be **data**, so turning it on for another structure later is an `UPDATE` and not a code edit. This task adds the columns and carries them from table to `Vocabulary`, with no behaviour change: no row sets the flag yet.

**Files:**
- Create: `deploy/sql/2026-10-02_document_type_parent_evidence.sql`
- Create: `deploy/sql/2026-10-02_document_type_parent_evidence_rollback.sql`
- Modify: `src/services/concepts/seed.py` (the `DocumentType` dataclass, ~line 30-41, and `SEED_VOCABULARY`'s row dicts in `vocabulary.py`)
- Modify: `src/services/concepts/vocabulary.py:46-51` (`_DOC_TYPE_SQL`), `:140-152` (`build_vocabulary`), `:185-200` (`SEED_VOCABULARY`)
- Modify: `src/services/concepts/validate.py`
- Modify: `tests/services/concepts/test_concept_table.py` (fixture `SELECT` and the column-for-column comparison)
- Test: `tests/services/concepts/test_parent_evidence_columns.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DocumentType.requires_parent_evidence: bool = False`
  - `DocumentType.parent_evidence_phrases: Tuple[str, ...] = ()`
  - `validate.check_flagged_types_have_parent_evidence_phrases(doc_type_rows) -> list[Violation]`

- [ ] **Step 1: Write the failing tests**

Create `tests/services/concepts/test_parent_evidence_columns.py`:

```python
"""The parent-evidence flag and its phrases reach Vocabulary from the table.

A column the loader does not read is a column the resolver cannot act on, and
the failure is silent: every flagged structure behaves as though unflagged.

Live-only for the table test. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/concepts/test_parent_evidence_columns.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import validate as V          # noqa: E402
from src.services.concepts.vocabulary import build_vocabulary  # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")


def _concept_row(code, domain="DOCUMENT_TYPE"):
    return {"concept_code": code, "domain": domain, "definition": "d",
            "not_to_be_confused_with": [], "status": "active", "rejection_reason": None}


def _doc_row(code, **over):
    row = {"concept_code": code, "role": "role.master", "default_parent_type": None,
           "execution_mode": None, "aliases": ["x"], "identifiers": [],
           "structural_signals": [], "pipeline_doc_type": "contract",
           "status": "active", "requires_parent_evidence": False,
           "parent_evidence_phrases": []}
    row.update(over)
    return row


def test_the_flag_and_phrases_reach_the_vocabulary():
    vocab = build_vocabulary(
        [_concept_row("doctype.a")],
        [_doc_row("doctype.a", requires_parent_evidence=True,
                  parent_evidence_phrases=["framework", "order of precedence"])],
        source="test",
    )
    dt = vocab.document_types["doctype.a"]
    assert dt.requires_parent_evidence is True
    assert dt.parent_evidence_phrases == ("framework", "order of precedence")


def test_an_unflagged_type_defaults_to_false_and_no_phrases():
    vocab = build_vocabulary([_concept_row("doctype.a")], [_doc_row("doctype.a")], source="test")
    dt = vocab.document_types["doctype.a"]
    assert dt.requires_parent_evidence is False
    assert dt.parent_evidence_phrases == ()


def test_a_missing_column_is_read_as_unflagged_not_as_an_error():
    """A row dict from an older query shape must not blow up the loader.

    vocabulary.build_vocabulary is public and callers construct rows by hand.
    """
    row = _doc_row("doctype.a")
    del row["requires_parent_evidence"]
    del row["parent_evidence_phrases"]
    vocab = build_vocabulary([_concept_row("doctype.a")], [row], source="test")
    assert vocab.document_types["doctype.a"].requires_parent_evidence is False


def test_a_flagged_type_with_no_phrases_is_a_violation():
    """Review Focus 2: it could never match, muting the structure silently."""
    rows = [_doc_row("doctype.a", requires_parent_evidence=True, parent_evidence_phrases=[])]
    violations = V.check_flagged_types_have_parent_evidence_phrases(rows)
    assert len(violations) == 1
    assert "doctype.a" in violations[0].subject


def test_a_flagged_type_with_phrases_is_not_a_violation():
    rows = [_doc_row("doctype.a", requires_parent_evidence=True,
                     parent_evidence_phrases=["framework"])]
    assert V.check_flagged_types_have_parent_evidence_phrases(rows) == []


def test_an_unflagged_type_with_phrases_is_not_a_violation():
    """Phrases without the flag are inert, not wrong: they pre-stage a later UPDATE."""
    rows = [_doc_row("doctype.a", parent_evidence_phrases=["framework"])]
    assert V.check_flagged_types_have_parent_evidence_phrases(rows) == []


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_the_live_table_has_both_columns_with_the_right_defaults():
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT column_name, data_type, is_nullable, column_default
                 FROM information_schema.columns
                WHERE table_schema='proc' AND table_name='bp_document_type'
                  AND column_name IN ('requires_parent_evidence','parent_evidence_phrases')
                ORDER BY column_name"""
        )
        got = {r[0]: (r[1], r[2], r[3]) for r in cur.fetchall()}
    assert set(got) == {"requires_parent_evidence", "parent_evidence_phrases"}, got
    assert got["requires_parent_evidence"][0] == "boolean"
    assert got["requires_parent_evidence"][1] == "NO"          # NOT NULL
    assert "false" in (got["requires_parent_evidence"][2] or "")
    assert got["parent_evidence_phrases"][0] == "ARRAY"
    assert got["parent_evidence_phrases"][1] == "NO"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/concepts/test_parent_evidence_columns.py -v
```

Expected: `test_the_flag_and_phrases_reach_the_vocabulary` FAILS with `AttributeError: 'DocumentType' object has no attribute 'requires_parent_evidence'`; the two `validate` tests FAIL with `AttributeError: module ... has no attribute 'check_flagged_types_have_parent_evidence_phrases'`; the live test FAILS on the empty column set.

- [ ] **Step 3: Write the migration**

Create `deploy/sql/2026-10-02_document_type_parent_evidence.sql`:

```sql
-- Two columns on proc.bp_document_type, so the stand-down rule is DATA.
--
-- A structure whose purpose is to sit beneath a parent should only claim a page
-- that actually names that parent. 'order form' is the measured case: it titles
-- every quote-template workbook on this corpus, and added as a plain structure
-- it flips 13 quote documents to 'disagreed' with no true positive
-- (specs/2026-10-02-contract-structures-design.md §4).
--
-- requires_parent_evidence  -- does this structure have to show its parent?
-- parent_evidence_phrases   -- the matchable phrases that count as showing it.
--
-- Phrases, not prose. The existing structural_signals are sentences
-- ("lists incorporated documents") and cannot match a page -- open item 5 of
-- specs/2026-10-01-document-relationship-layer-rulings.md. These are compared
-- with the same fold() normalisation every alias uses.
--
-- NO ROW SETS THE FLAG HERE. Behaviour is unchanged by this migration; the
-- order-form row arrives in 2026-10-02_document_type_order_form_sales_order.sql.
--
-- Additive, idempotent, reversible.
BEGIN;

ALTER TABLE proc.bp_document_type
    ADD COLUMN IF NOT EXISTS requires_parent_evidence boolean NOT NULL DEFAULT false;

ALTER TABLE proc.bp_document_type
    ADD COLUMN IF NOT EXISTS parent_evidence_phrases text[] NOT NULL DEFAULT '{}';

COMMENT ON COLUMN proc.bp_document_type.requires_parent_evidence IS
    'When true, this structure only claims a document that names its parent or '
    'states an order of precedence. Set for doctype.order_form only; the other '
    'child structures (call-off, SOW, schedule) are conceptually the same but '
    'their current behaviour is measured and no evidence calls for changing it.';
COMMENT ON COLUMN proc.bp_document_type.parent_evidence_phrases IS
    'Matchable phrases that count as naming a parent. Compared with the same '
    'fold() normalisation as aliases. Prose belongs in structural_signals.';

COMMIT;
```

Create `deploy/sql/2026-10-02_document_type_parent_evidence_rollback.sql`:

```sql
-- Removes exactly the two columns 2026-10-02_document_type_parent_evidence.sql adds.
-- Run the order_form/sales_order rollback FIRST if that migration has been applied:
-- dropping these columns while doctype.order_form relies on the flag would make
-- 'order form' claim every quote-template workbook again.
BEGIN;

ALTER TABLE proc.bp_document_type DROP COLUMN IF EXISTS parent_evidence_phrases;
ALTER TABLE proc.bp_document_type DROP COLUMN IF EXISTS requires_parent_evidence;

COMMIT;
```

- [ ] **Step 4: Apply the migration to bp_testdb**

```bash
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    -v ON_ERROR_STOP=1 -f deploy/sql/2026-10-02_document_type_parent_evidence.sql
```

Then run it a second time and confirm it succeeds unchanged — that is the idempotency proof, not an assumption.

- [ ] **Step 5: Extend the dataclass**

In `src/services/concepts/seed.py`, add two fields to `DocumentType`, after `pipeline_doc_type` and before `status` (field order matters: `status` has a default, so every field after it must too):

```python
@dataclass(frozen=True)
class DocumentType:
    concept_code: str
    role: str
    default_parent_type: Optional[str]
    execution_mode: Optional[str]
    aliases: Tuple[str, ...]
    identifiers: Tuple[Mapping[str, Optional[str]], ...]
    structural_signals: Tuple[str, ...]
    #: Which of the four physical pipelines ingests this type. None means the
    #: type is recognised but nothing can ingest it yet — an honest state.
    pipeline_doc_type: Optional[str]
    status: str = "active"
    #: A structure whose purpose is to sit beneath a parent only claims a page
    #: that names that parent. Data, so flagging another structure later is an
    #: UPDATE rather than a code edit.
    requires_parent_evidence: bool = False
    #: Matchable phrases that count as naming a parent, folded like aliases.
    parent_evidence_phrases: Tuple[str, ...] = ()
```

- [ ] **Step 6: Thread the columns through the loader**

In `src/services/concepts/vocabulary.py`, extend `_DOC_TYPE_SQL`:

```python
_DOC_TYPE_SQL = """
    SELECT concept_code, role, default_parent_type, execution_mode,
           aliases, identifiers, structural_signals, pipeline_doc_type,
           status, requires_parent_evidence, parent_evidence_phrases,
           recorded_at
      FROM proc.bp_document_type
     WHERE status = 'active'
"""
```

In `build_vocabulary`, inside the `doc_type_rows` loop, extend the `DocumentType(...)` construction:

```python
        document_types[code] = DocumentType(
            concept_code=code,
            role=str(row.get("role") or ""),
            default_parent_type=row.get("default_parent_type"),
            execution_mode=row.get("execution_mode"),
            aliases=aliases,
            identifiers=identifiers,
            structural_signals=tuple(row.get("structural_signals") or ()),
            pipeline_doc_type=row.get("pipeline_doc_type"),
            status="active",
            # .get with a default, not row["..."]: build_vocabulary is public and
            # callers build rows by hand, so an older row shape must read as
            # "unflagged" rather than raise.
            requires_parent_evidence=bool(row.get("requires_parent_evidence") or False),
            parent_evidence_phrases=tuple(row.get("parent_evidence_phrases") or ()),
        )
```

And in `SEED_VOCABULARY`'s document-type dicts:

```python
        {
            "concept_code": d.concept_code,
            "role": d.role,
            "default_parent_type": d.default_parent_type,
            "execution_mode": d.execution_mode,
            "aliases": list(d.aliases),
            "identifiers": list(d.identifiers),
            "structural_signals": list(d.structural_signals),
            "pipeline_doc_type": d.pipeline_doc_type,
            "status": d.status,
            "requires_parent_evidence": d.requires_parent_evidence,
            "parent_evidence_phrases": list(d.parent_evidence_phrases),
        }
        for d in DOCUMENT_TYPES.values()
```

- [ ] **Step 7: Add the validation check**

In `src/services/concepts/validate.py`, following the shape of the existing checks in that module (match the local `Violation` type and the registration list exactly as the file defines them):

```python
def check_flagged_types_have_parent_evidence_phrases(doc_type_rows) -> list[Violation]:
    """A structure that must show its parent needs phrases to recognise one by.

    requires_parent_evidence with an empty parent_evidence_phrases can never be
    satisfied, so the structure matches nothing and does so silently — the worst
    shape a reference-data error can take. Reported rather than defaulted,
    because guessing a phrase set is how a classifier starts inventing parents.
    """
    out: list[Violation] = []
    for row in doc_type_rows:
        if not row.get("requires_parent_evidence"):
            continue
        if not (row.get("parent_evidence_phrases") or ()):
            out.append(Violation(
                check="flagged_types_have_parent_evidence_phrases",
                subject=str(row.get("concept_code")),
                detail=(
                    "requires_parent_evidence is set but parent_evidence_phrases "
                    "is empty, so this structure can never claim a document"
                ),
            ))
    return out
```

Register it alongside the existing four §8.1 checks so `.github/workflows/reference-data-checks.yml` runs it.

- [ ] **Step 8: Extend the drift test**

In `tests/services/concepts/test_concept_table.py`, add both columns to the `doc_type_rows` fixture `SELECT`:

```python
            SELECT concept_code, role, default_parent_type, execution_mode,
                   aliases, identifiers, structural_signals, pipeline_doc_type,
                   status, source, requires_parent_evidence, parent_evidence_phrases
              FROM proc.bp_document_type
```

and to the `pairs` list in `test_document_type_rows_equal_the_seed_column_for_column`:

```python
            ("requires_parent_evidence", bool(r["requires_parent_evidence"]),
             seed.requires_parent_evidence),
            ("parent_evidence_phrases", list(r["parent_evidence_phrases"] or []),
             list(seed.parent_evidence_phrases)),
```

- [ ] **Step 9: Run the tests to verify they pass**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/concepts/test_parent_evidence_columns.py \
    tests/services/concepts/test_concept_table.py \
    tests/services/concepts/test_validation_checks.py \
    tests/services/concepts/test_vocabulary_runtime_load.py -v
```

Expected: all pass.

- [ ] **Step 10: Confirm nothing changed for any document**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_classification_baseline.py -v
```

Expected: 2 passed. This task adds columns only; a single changed document here means the loader change altered behaviour and must be investigated before continuing.

- [ ] **Step 11: Prove the guards fail (required)**

Three separate proofs, each restored afterwards:

1. **The loader reads the columns.** In `build_vocabulary`, hard-code `requires_parent_evidence=False`. Expected: `test_the_flag_and_phrases_reach_the_vocabulary` FAILS.
2. **The validation check fires.** Make `check_flagged_types_have_parent_evidence_phrases` return `[]` unconditionally. Expected: `test_a_flagged_type_with_no_phrases_is_a_violation` FAILS.
3. **The drift test covers the new columns.** On `bp_testdb`, run
   `UPDATE proc.bp_document_type SET requires_parent_evidence = true WHERE concept_code = 'doctype.nda';`
   Expected: `test_document_type_rows_equal_the_seed_column_for_column` FAILS naming `doctype.nda.requires_parent_evidence`. Then
   `UPDATE proc.bp_document_type SET requires_parent_evidence = false WHERE concept_code = 'doctype.nda';`
   and confirm green.

- [ ] **Step 12: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add deploy/sql/2026-10-02_document_type_parent_evidence.sql \
        deploy/sql/2026-10-02_document_type_parent_evidence_rollback.sql \
        src/services/concepts/seed.py src/services/concepts/vocabulary.py \
        src/services/concepts/validate.py \
        tests/services/concepts/test_parent_evidence_columns.py \
        tests/services/concepts/test_concept_table.py
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "feat(concepts): the parent-evidence rule becomes data" "" \
  "Task 2 of specs/2026-10-02-contract-structures-plan.md." "" \
  "requires_parent_evidence and parent_evidence_phrases on" \
  "proc.bp_document_type, threaded to Vocabulary, plus a validation check that" \
  "a flagged structure with no phrases is a violation rather than a structure" \
  "that silently matches nothing." "" \
  "No row sets the flag, so no document classifies differently: the 53-document" \
  "baseline is unchanged. Three guards proven red." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- deploy/sql/2026-10-02_document_type_parent_evidence.sql \
    deploy/sql/2026-10-02_document_type_parent_evidence_rollback.sql \
    src/services/concepts/seed.py src/services/concepts/vocabulary.py \
    src/services/concepts/validate.py \
    tests/services/concepts/test_parent_evidence_columns.py \
    tests/services/concepts/test_concept_table.py
git diff --cached --name-status | wc -l
```

---
## Task 3: The resolver stands a child structure down

**Files:**
- Modify: `src/services/extraction/type_resolver.py:417-418` (the one status check) and a new module-level helper
- Test: `tests/services/extraction/test_parent_evidence_stand_down.py`

**Interfaces:**
- Consumes: `DocumentType.requires_parent_evidence`, `DocumentType.parent_evidence_phrases` (Task 2).
- Produces: `type_resolver._names_a_parent(dt: DocumentType, lowered: str) -> bool`. No signature change to `resolve_document_type`.

**Why the hook goes at line 417-418 and nowhere else.** That loop is the single place the module enforces "only active types resolve", and its comment says so at length: a second copy of a status rule downstream could not fire and so could not fail. Standing a structure down there means it is absent from `owners`, `hits` and `signals` for that page — so **tier 2 cannot name it either**. Filtering `title_concepts` later would leave a stood-down structure winning on body mentions alone, which is the same defect one line further down. This is also exactly what the design's measurement simulated (two vocabularies, one with the structure and one without), so the measured result transfers directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/extraction/test_parent_evidence_stand_down.py`:

```python
"""A child structure only claims a page that names its parent.

Measured reason this exists: 'order form' titles every quote-template workbook
on this corpus. Added as a plain structure it flips 13 quote documents to
'disagreed' with no true positive. With this rule all 53 documents resolve
exactly as they do today, and a real order form that names its framework still
classifies. See specs/2026-10-02-contract-structures-design.md §4.

Offline — these build a Vocabulary by hand, so they run with no database.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_parent_evidence_stand_down.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.seed import Concept, DocumentType          # noqa: E402
from src.services.concepts.vocabulary import build_vocabulary          # noqa: E402
from src.services.extraction.type_resolver import resolve_document_type  # noqa: E402

PHRASES = ("framework", "order of precedence", "incorporated", "call off")


def _vocab(*types: DocumentType):
    concepts = [
        {"concept_code": t.concept_code, "domain": "DOCUMENT_TYPE", "definition": "d",
         "not_to_be_confused_with": [], "status": "active", "rejection_reason": None}
        for t in types
    ]
    rows = [
        {"concept_code": t.concept_code, "role": t.role,
         "default_parent_type": t.default_parent_type, "execution_mode": t.execution_mode,
         "aliases": list(t.aliases), "identifiers": list(t.identifiers),
         "structural_signals": list(t.structural_signals),
         "pipeline_doc_type": t.pipeline_doc_type, "status": t.status,
         "requires_parent_evidence": t.requires_parent_evidence,
         "parent_evidence_phrases": list(t.parent_evidence_phrases)}
        for t in types
    ]
    return build_vocabulary(concepts, rows, source="test")


ORDER_FORM = DocumentType(
    "doctype.order_form", "role.master", "doctype.framework_agreement", "exec.bilateral",
    ("order form",), (), (), "contract",
    requires_parent_evidence=True, parent_evidence_phrases=PHRASES,
)
QUOTE = DocumentType(
    "doctype.quote", "role.supporting", None, "exec.unilateral",
    ("quote", "quotation"), (), (), "quote",
)
FRAMEWORK = DocumentType(
    "doctype.framework_agreement", "role.framework", None, "exec.bilateral",
    ("framework agreement", "framework"), (), (), "contract",
)


def test_an_order_form_that_names_no_parent_does_not_claim_the_page():
    page = "ORDER FORM\n\nQuote Ref Q-1234   Valid Until 2026-12-01\n"
    r = resolve_document_type(declared_concept="doctype.quote", full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept != "doctype.order_form"
    assert r.agreement != "disagreed", (
        "the quote workbook shape must not become a disagreement again"
    )


def test_an_order_form_that_names_its_framework_does_claim_the_page():
    page = ("ORDER FORM\n\nThis Order Form is made under Framework Agreement "
            "FW-2024-0012.\n")
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept == "doctype.order_form", r
    assert r.status == "matched"


def test_an_order_of_precedence_clause_is_also_parent_evidence():
    page = ("ORDER FORM\n\n2.1 The documents take effect in the following order of "
            "precedence.\n")
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept == "doctype.order_form", r


def test_standing_down_also_removes_the_structure_from_tier_two():
    """Not just the title. A body-only mention must not win either.

    Filtering title_concepts alone would leave a stood-down structure winning on
    repeated body mentions, which is the same defect one line further down.
    """
    page = ("Quotation\n\nPlease sign the order form and return it. The order form "
            "must be returned with the order form cover sheet.\n")
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept != "doctype.order_form", r
    assert "doctype.order_form" not in r.candidates
    assert all(ev.concept_code != "doctype.order_form" for ev in r.evidence), (
        "a structure that stood down must not appear as evidence either"
    )


def test_a_title_naming_both_stays_unresolved_with_both_candidates():
    """Review Focus 1: the rule must not turn a genuine tie into a confident answer.

    The page names a framework, so the order form does NOT stand down — and then
    two structures claim the same title segment, which is a tie a person settles.
    """
    page = "ORDER FORM / FRAMEWORK AGREEMENT\n\nmade between the parties.\n"
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE, FRAMEWORK))
    assert r.status == "unresolved", r
    assert set(r.candidates) == {"doctype.order_form", "doctype.framework_agreement"}, r
    assert r.evidence_concept is None


def test_an_unflagged_structure_is_never_stood_down():
    page = "QUOTATION\n\nValid Until 2026-12-01\n"
    r = resolve_document_type(declared_concept="doctype.quote", full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept == "doctype.quote"
    assert r.agreement == "agreed"


def test_a_flagged_structure_with_no_phrases_claims_nothing():
    """The validation check in Task 2 reports this; the resolver must not guess.

    A structure that must show its parent, with no phrase to recognise one by,
    cannot be satisfied — and inferring a default phrase set is how a classifier
    starts inventing parents.
    """
    muted = DocumentType(
        "doctype.order_form", "role.master", "doctype.framework_agreement",
        "exec.bilateral", ("order form",), (), (), "contract",
        requires_parent_evidence=True, parent_evidence_phrases=(),
    )
    page = ("ORDER FORM\n\nmade under Framework Agreement FW-1 in the following order "
            "of precedence.\n")
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(muted, QUOTE))
    assert r.evidence_concept != "doctype.order_form", r


def test_parent_evidence_matches_with_alias_normalisation():
    """'Call-Off' on the page satisfies the phrase 'call off'.

    The match copy treats '_' and '-' as spaces exactly as fold() does, so the
    two sides agree without the phrase list having to spell both.
    """
    page = "ORDER FORM\n\nissued under the Call-Off procedure.\n"
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept == "doctype.order_form", r
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_parent_evidence_stand_down.py -v
```

Expected: `test_an_order_form_that_names_no_parent_does_not_claim_the_page`, `test_standing_down_also_removes_the_structure_from_tier_two` and `test_a_flagged_structure_with_no_phrases_claims_nothing` FAIL — the flag is not read yet, so the order form claims every page carrying the words. The rest pass already.

- [ ] **Step 3: Add the helper**

In `src/services/extraction/type_resolver.py`, beside the other module-level helpers (near `_title_owners`, ~line 250):

```python
def _names_a_parent(dt: "DocumentType", lowered: str) -> bool:
    """Does this page name the parent the structure must sit under?

    Compared against ``lowered`` -- the same match copy every alias uses, whose
    length equals the page's -- so the phrase side is folded and the page side is
    not. A phrase found here sits at a real offset in the document.

    An empty phrase list returns False, so a structure flagged with nothing to
    recognise a parent by claims nothing. That is deliberate: inferring a default
    phrase set is how a classifier starts inventing parents.
    ``validate.check_flagged_types_have_parent_evidence_phrases`` reports the
    row so the silence is visible.
    """
    for phrase in dt.parent_evidence_phrases:
        folded = fold(phrase)
        if folded and folded in lowered:
            return True
    return False
```

- [ ] **Step 4: Hook it into the one status check**

In `resolve_document_type`, at the top of the `for code, dt in vocab.document_types.items():` loop (line 417-418), directly after the existing `status` check:

```python
    for code, dt in vocab.document_types.items():
        if dt.status != "active":
            continue
        # A child structure stands down on a page that does not name its parent.
        #
        # HERE, beside the status rule, and not by filtering title_concepts
        # later: this is the one place the module decides whether a type exists
        # for this document at all, so standing down here removes it from
        # `owners`, `hits`, `signals`, tier 1 AND tier 2 together. A filter
        # further down would leave the structure winning on body mentions alone
        # -- the same defect, one line later, and invisible in any single score.
        #
        # This is also precisely what the design's measurement simulated: two
        # vocabularies, one holding the structure and one without. 13 quote
        # workbooks turn on this line.
        if dt.requires_parent_evidence and not _names_a_parent(dt, lowered):
            continue
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_parent_evidence_stand_down.py -v
```

Expected: 9 passed.

- [ ] **Step 6: Run the resolver's existing suite and the baseline**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_type_resolver.py \
    tests/services/extraction/test_type_resolver_matched_evidence.py \
    tests/services/extraction/test_type_resolution_review_items.py -v
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_classification_baseline.py -v
```

Expected: all pass. No seeded row sets the flag yet, so every document must still resolve identically.

- [ ] **Step 7: Prove the guard fails (required)**

1. **The stand-down fires.** Change the hook to `if False and dt.requires_parent_evidence ...`. Expected: three tests FAIL. Restore.
2. **It covers tier 2, not only the title.** Move the check out of the loop and apply it to `title_concepts` instead (after `named = _title_owners(...)`, drop flagged concepts with no evidence). Expected: `test_standing_down_also_removes_the_structure_from_tier_two` FAILS while the title-only tests still pass — which is the whole reason the hook is where it is. Restore.
3. **The tie is preserved.** Make `_names_a_parent` return `False` unconditionally. Expected: `test_a_title_naming_both_stays_unresolved_with_both_candidates` FAILS, because the order form stands down and the framework wins alone. Restore.

- [ ] **Step 8: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add src/services/extraction/type_resolver.py \
        tests/services/extraction/test_parent_evidence_stand_down.py
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "feat(type resolver): a child structure must show its parent" "" \
  "Task 3 of specs/2026-10-02-contract-structures-plan.md." "" \
  "The check sits beside the one status rule, so a stood-down structure is" \
  "absent from tier 1 AND tier 2 together. Filtering title_concepts instead" \
  "leaves it winning on body mentions alone -- proven by moving the check and" \
  "watching the tier-2 test go red." "" \
  "No seeded row sets the flag, so the 53-document baseline is unchanged." \
  "Three guards proven red." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- src/services/extraction/type_resolver.py \
    tests/services/extraction/test_parent_evidence_stand_down.py
git diff --cached --name-status | wc -l
```

---
## Task 4: The two new structures

**Files:**
- Create: `deploy/sql/2026-10-02_document_type_order_form_sales_order.sql`
- Create: `deploy/sql/2026-10-02_document_type_order_form_sales_order_rollback.sql`
- Modify: `src/services/concepts/seed.py` (`_DOCUMENT_TYPE_CONCEPTS` and `DOCUMENT_TYPES`)
- Test: `tests/services/extraction/test_order_form_and_sales_order.py`

**Interfaces:**
- Consumes: Task 2's columns, Task 3's stand-down rule.
- Produces: the concept codes `doctype.order_form` and `doctype.sales_order`, live in the vocabulary and therefore acceptable upload categories (`routing.pipeline_for_category`).

**Two consequences to know before applying this.** An alias is also an acceptable upload category, so after this migration `order form` routes to the contract pipeline and `sales order` to the purchase-order pipeline. And `doctype.order_form` is the **first** row to carry `requires_parent_evidence`, so Task 3's rule becomes live behaviour here, not in Task 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/extraction/test_order_form_and_sales_order.py`:

```python
"""The two structures Nick named, and the measured reason order form is safe now.

'order form' was dropped as an alias of doctype.call_off_contract on 2026-10-01:
it titled every quote-template workbook, giving 12 false disagreements out of 12
uses and no true positive. It returns here as a structure in its own right,
governed by requires_parent_evidence, which is the build spec's own distinction —
"'order form' means one thing under a framework and another on its own".

Live-only. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/extraction/test_order_form_and_sales_order.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import routing as R                       # noqa: E402
from src.services.concepts import vocabulary as V                    # noqa: E402
from src.services.extraction.type_resolver import resolve_document_type  # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


@pytest.fixture()
def vocab():
    V.invalidate()
    return V.ensure_vocabulary()


def test_both_structures_are_live_in_the_vocabulary(vocab):
    assert "doctype.order_form" in vocab.document_types
    assert "doctype.sales_order" in vocab.document_types


def test_order_form_is_the_only_structure_requiring_parent_evidence(vocab):
    """Nick's ruling, 2026-10-02: the rule governs doctype.order_form alone to start.

    call-off, SOW and schedule are conceptually the same but their behaviour is
    measured at 47 agreed / 0 disagreed and no evidence calls for changing it.
    """
    flagged = sorted(
        code for code, dt in vocab.document_types.items() if dt.requires_parent_evidence
    )
    assert flagged == ["doctype.order_form"], flagged


def test_order_form_carries_phrases_to_recognise_a_parent_by(vocab):
    phrases = vocab.document_types["doctype.order_form"].parent_evidence_phrases
    assert phrases, "a flagged structure with no phrases claims nothing"
    assert "framework" in phrases


def test_order_form_is_not_an_alias_of_the_call_off_contract(vocab):
    """The 2026-10-01 ruling stands: 'order form' must not resolve to a call-off."""
    owners = V.resolve_alias("order form", vocab)
    assert owners == ("doctype.order_form",), owners
    assert "order form" not in vocab.document_types["doctype.call_off_contract"].aliases


def test_a_real_call_off_still_matches_its_own_names(vocab):
    for spelling in ("call-off contract", "call off contract", "call-off"):
        assert V.resolve_alias(spelling, vocab) == ("doctype.call_off_contract",), spelling


def test_sales_order_wins_over_the_bare_order_alias(vocab):
    """Longest match wins: 'sales order' is not doctype.order with a word in front."""
    assert V.resolve_alias("sales order", vocab) == ("doctype.sales_order",)
    assert V.resolve_alias("order", vocab) == ("doctype.order",)
    page = "SALES ORDER\n\nAcknowledgement of your order.\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=vocab)
    assert r.evidence_concept == "doctype.sales_order", r


def test_a_sales_order_routes_at_the_purchase_order_pipeline(vocab):
    """Nick's ruling, 2026-10-02: a sales order extracts with the PO schema.

    It is the supplier's mirror of a purchase order and carries lines, quantities
    and a total, so the purchase-order schema is the one that fits it.
    """
    pipeline, code = R.pipeline_for_category("sales order", vocabulary=vocab)
    assert (pipeline, code) == ("purchase_order", "doctype.sales_order")


def test_an_order_form_routes_at_the_contract_pipeline(vocab):
    pipeline, code = R.pipeline_for_category("order form", vocabulary=vocab)
    assert (pipeline, code) == ("contract", "doctype.order_form")


def test_no_quote_workbook_became_a_disagreement(vocab):
    """THE measurement this task exists to protect.

    Thirteen stored quote workbooks carry an 'Order Form' title cell. Adding the
    structure without the parent-evidence rule flips every one of them to
    'disagreed'. None of them names a framework, an order of precedence, an
    incorporation clause or a call-off, so all thirteen stand the structure down.
    """
    from tests.services.extraction.test_classification_baseline import (
        _resolve_stored_documents,
    )

    live = _resolve_stored_documents()
    became = {
        src: out for src, out in live.items()
        if out["evidence_concept"] == "doctype.order_form"
    }
    assert not became, (
        f"{len(became)} stored document(s) now read as an order form: {sorted(became)}"
    )
    disagreed = {s: o for s, o in live.items() if o["agreement"] == "disagreed"}
    assert not disagreed, f"disagreements appeared: {sorted(disagreed)}"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_order_form_and_sales_order.py -v
```

Expected: the first eight FAIL (`'doctype.order_form' not in vocab.document_types`, then `KeyError`/empty alias resolutions). `test_no_quote_workbook_became_a_disagreement` PASSES already — it is the guard that must stay green through this task, not one that turns green.

- [ ] **Step 3: Write the migration**

Create `deploy/sql/2026-10-02_document_type_order_form_sales_order.sql`:

```sql
-- The two structures Nick named on 2026-10-02: order form and sales order.
--
-- ORDER FORM comes back as a STRUCTURE IN ITS OWN RIGHT, not as an alias of
-- doctype.call_off_contract. It was dropped as that alias on 2026-10-01 because
-- it titles every quote-template workbook on this corpus: 12 false
-- disagreements out of 12 uses, no true positive
-- (2026-10-02_concept_vocabulary_drop_order_form_alias.sql).
--
-- What makes it safe now is requires_parent_evidence, added by
-- 2026-10-02_document_type_parent_evidence.sql. An order form only claims a
-- document that names the agreement it sits under or states an order of
-- precedence. Measured over the 53 corpus documents with stored parsed text: without
-- the rule, 13 quote workbooks flip to 'disagreed'; with it, every document
-- resolves exactly as it does today. This is the build spec's own distinction --
-- "'order form' means one thing under a framework and another on its own".
--
-- THIS MIGRATION IS WHERE THAT RULE BECOMES LIVE BEHAVIOUR. doctype.order_form
-- is the first and only row to carry the flag. Do not apply it before
-- 2026-10-02_document_type_parent_evidence.sql: without the column the structure
-- claims every page carrying the words.
--
-- SALES ORDER routes at the PURCHASE_ORDER pipeline, not the contract pipeline
-- (Nick's ruling, 2026-10-02): it is the supplier's mirror of a purchase order
-- and carries lines, quantities and a total, so the PO schema is the one that
-- fits it. pipeline_doc_type only takes effect when an uploader types that
-- category; a document dropped in the Contracts zone still routes on 'contract'.
--
-- Consequence to know before running: an alias is also an acceptable UPLOAD
-- CATEGORY (src/services/concepts/routing.py). After this,
--   'order form'  -> ('contract', 'doctype.order_form')
--   'sales order' -> ('purchase_order', 'doctype.sales_order')
-- Neither raised before; both now route.
--
-- APPEND ORDER IS PART OF THE DATA: the full-column drift test compares
-- not_to_be_confused_with and aliases as ORDERED lists against seed.py. The two
-- new pointers appended to existing concepts go LAST in both places.
--
-- Additive (ON CONFLICT DO NOTHING), idempotent, reversible.
-- Order matters: bp_document_type's foreign keys need the concepts first.
BEGIN;

INSERT INTO proc.bp_concept
    (concept_code, domain, definition, not_to_be_confused_with, status, source, rejection_reason)
VALUES
    ('doctype.order_form', 'DOCUMENT_TYPE',
     'Orders specific goods or services on the terms of an agreement it names.',
     ARRAY['doctype.call_off_contract','doctype.quote','doctype.order']::text[],
     'active', 'seed', NULL),
    ('doctype.sales_order', 'DOCUMENT_TYPE',
     'The supplier''s own confirmation of an order it has received.',
     ARRAY['doctype.order','doctype.invoice']::text[],
     'active', 'seed', NULL)
ON CONFLICT (concept_code) DO NOTHING;

-- The table exists to record what a term is mistaken for, so the two
-- already-seeded concepts most at risk gain a pointer at the new ones.
UPDATE proc.bp_concept
   SET not_to_be_confused_with = array_append(not_to_be_confused_with, 'doctype.order_form')
 WHERE concept_code = 'doctype.call_off_contract'
   AND NOT (not_to_be_confused_with @> ARRAY['doctype.order_form']::text[]);

UPDATE proc.bp_concept
   SET not_to_be_confused_with = array_append(not_to_be_confused_with, 'doctype.sales_order')
 WHERE concept_code = 'doctype.order'
   AND NOT (not_to_be_confused_with @> ARRAY['doctype.sales_order']::text[]);

INSERT INTO proc.bp_document_type
    (concept_code, role, default_parent_type, execution_mode, aliases, identifiers,
     structural_signals, pipeline_doc_type, status, source,
     requires_parent_evidence, parent_evidence_phrases)
VALUES
    ('doctype.order_form', 'role.master', 'doctype.framework_agreement', 'exec.bilateral',
     ARRAY['order form']::text[],
     '[{"field": "framework_ref", "pattern": null, "parent_type": "doctype.framework_agreement"}]'::jsonb,
     ARRAY['lists incorporated documents','states an order of precedence']::text[],
     'contract', 'active', 'seed',
     true,
     -- Phrases, not prose: these are compared against the page with the same
     -- fold() normalisation as an alias, so 'Call-Off' satisfies 'call off'.
     ARRAY['framework','order of precedence','incorporated','call off',
           'framework agreement no','framework agreement number','framework agreement ref',
           'master agreement no','master agreement number','master agreement ref',
           'parent agreement no','parent contract no','principal agreement no']::text[]),
    ('doctype.sales_order', 'role.transaction', 'doctype.order', 'exec.unilateral',
     ARRAY['sales order','sales order acknowledgement','order acknowledgement']::text[],
     '[{"field": "po_id", "pattern": null, "parent_type": "doctype.order"}]'::jsonb,
     ARRAY['line items with quantities and a total']::text[],
     'purchase_order', 'active', 'seed',
     false, ARRAY[]::text[])
ON CONFLICT (concept_code) DO NOTHING;

COMMIT;
```

Create `deploy/sql/2026-10-02_document_type_order_form_sales_order_rollback.sql`:

```sql
-- Removes exactly what 2026-10-02_document_type_order_form_sales_order.sql adds.
--
-- After this, 'order form' and 'sales order' are unknown upload categories again
-- and routing REFUSES them, which is the pre-2026-10-02 behaviour.
--
-- bp_document_type first: its concept_code is a foreign key into bp_concept.
BEGIN;

DELETE FROM proc.bp_document_type
 WHERE concept_code IN ('doctype.order_form', 'doctype.sales_order');

UPDATE proc.bp_concept
   SET not_to_be_confused_with = array_remove(not_to_be_confused_with, 'doctype.order_form')
 WHERE concept_code = 'doctype.call_off_contract';

UPDATE proc.bp_concept
   SET not_to_be_confused_with = array_remove(not_to_be_confused_with, 'doctype.sales_order')
 WHERE concept_code = 'doctype.order';

DELETE FROM proc.bp_concept
 WHERE concept_code IN ('doctype.order_form', 'doctype.sales_order');

COMMIT;
```

- [ ] **Step 4: Mirror both rows in the seed**

In `src/services/concepts/seed.py`, append to `_DOCUMENT_TYPE_CONCEPTS` — **after** `doctype.policy_document`, so the seed's order matches the migration's insert order:

```python
    ("doctype.order_form",
     "Orders specific goods or services on the terms of an agreement it names.",
     ("doctype.call_off_contract", "doctype.quote", "doctype.order")),
    ("doctype.sales_order",
     "The supplier's own confirmation of an order it has received.",
     ("doctype.order", "doctype.invoice")),
```

Append `"doctype.order_form"` **last** in `doctype.call_off_contract`'s tuple and `"doctype.sales_order"` **last** in `doctype.order`'s tuple, matching the migration's `array_append`:

```python
    ("doctype.call_off_contract",
     "Orders work under a framework agreement, on that framework's terms.",
     ("doctype.order", "doctype.sow", "doctype.framework_agreement",
      "doctype.order_form")),
    ...
    ("doctype.order",
     "Commits a buyer to specific goods or services at an agreed price.",
     ("doctype.call_off_contract", "doctype.sales_order")),
```

(Keep each existing definition string exactly as it is — only the third tuple element changes.)

Append two `DocumentType` entries to `DOCUMENT_TYPES`, after `doctype.policy_document`:

```python
        DocumentType(
            # NOT an alias of doctype.call_off_contract -- that alias titled every
            # quote-template workbook and gave 12 false disagreements out of 12
            # uses. As its own structure with requires_parent_evidence it claims
            # only a document that names the agreement it sits under, which is the
            # build spec's own distinction: "'order form' means one thing under a
            # framework and another on its own".
            "doctype.order_form", "role.master", "doctype.framework_agreement",
            "exec.bilateral",
            ("order form",),
            ({"field": "framework_ref", "pattern": None,
              "parent_type": "doctype.framework_agreement"},),
            ("lists incorporated documents", "states an order of precedence"),
            "contract",
            requires_parent_evidence=True,
            parent_evidence_phrases=(
                "framework", "order of precedence", "incorporated", "call off",
                "framework agreement no", "framework agreement number",
                "framework agreement ref", "master agreement no",
                "master agreement number", "master agreement ref",
                "parent agreement no", "parent contract no", "principal agreement no",
            ),
        ),
        DocumentType(
            # The supplier's mirror of a purchase order, so it extracts with the
            # PO schema (Nick's ruling, 2026-10-02): lines, quantities, a total.
            "doctype.sales_order", "role.transaction", "doctype.order",
            "exec.unilateral",
            ("sales order", "sales order acknowledgement", "order acknowledgement"),
            ({"field": "po_id", "pattern": None, "parent_type": "doctype.order"},),
            ("line items with quantities and a total",),
            "purchase_order",
        ),
```

- [ ] **Step 5: Apply the migration to bp_testdb**

```bash
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    -v ON_ERROR_STOP=1 -f deploy/sql/2026-10-02_document_type_order_form_sales_order.sql
```

Run it a second time and confirm it succeeds unchanged, and that `not_to_be_confused_with` has not gained a duplicate:

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "SELECT concept_code, not_to_be_confused_with FROM proc.bp_concept
    WHERE concept_code IN ('doctype.call_off_contract','doctype.order');"
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_order_form_and_sales_order.py \
    tests/services/concepts/test_concept_table.py \
    tests/services/concepts/test_category_routing.py \
    tests/services/concepts/test_validation_checks.py \
    tests/services/extraction/test_classification_baseline.py -v
```

Expected: all pass — including the 53-document baseline, unchanged. The predecessor's own guard
`test_order_form_is_not_a_call_off_alias_and_quote_workbooks_stay_agreed` must also still pass; run the file it lives in and do not edit it.

- [ ] **Step 7: Prove the guard fails (required)**

The measurement is the deliverable, so this proof is the task:

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "UPDATE proc.bp_document_type SET requires_parent_evidence = false
    WHERE concept_code = 'doctype.order_form';"
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_order_form_and_sales_order.py::test_no_quote_workbook_became_a_disagreement \
    tests/services/extraction/test_classification_baseline.py -v
```

Expected: FAIL, naming **13** documents that now read as `doctype.order_form`, and 13 baseline drifts. That number is the design's measurement; if it is not 13, stop and report it. Then restore:

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "UPDATE proc.bp_document_type SET requires_parent_evidence = true
    WHERE concept_code = 'doctype.order_form';"
```

and confirm both go green. Also prove the alias guard: `UPDATE proc.bp_document_type SET aliases = array_append(aliases, 'order form') WHERE concept_code = 'doctype.call_off_contract';` must make `test_order_form_is_not_an_alias_of_the_call_off_contract` and the no-alias-claimed-by-two-concepts guard FAIL. Restore with `array_remove`.

- [ ] **Step 8: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add deploy/sql/2026-10-02_document_type_order_form_sales_order.sql \
        deploy/sql/2026-10-02_document_type_order_form_sales_order_rollback.sql \
        src/services/concepts/seed.py \
        tests/services/extraction/test_order_form_and_sales_order.py
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "feat(concepts): order form and sales order join the vocabulary" "" \
  "Task 4 of specs/2026-10-02-contract-structures-plan.md." "" \
  "order form returns as a structure in its own right, not as the call-off" \
  "alias that gave 12 false disagreements out of 12 uses. It is the only row" \
  "carrying requires_parent_evidence, so it claims only a document naming the" \
  "agreement it sits under." "" \
  "Measured: clearing the flag flips 13 stored quote workbooks to disagreed;" \
  "with it all 53 documents resolve exactly as before. sales order routes at" \
  "the purchase-order pipeline per Nick's ruling." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- deploy/sql/2026-10-02_document_type_order_form_sales_order.sql \
    deploy/sql/2026-10-02_document_type_order_form_sales_order_rollback.sql \
    src/services/concepts/seed.py \
    tests/services/extraction/test_order_form_and_sales_order.py
git diff --cached --name-status | wc -l
```

---
## Task 5: A specific structure under a generic declaration is a refinement

Uploading into the Contracts zone declares the category `contract`, which resolves to `doctype.contract_unspecified`. Today a page naming its real structure — "Master Agreement", "Statement of Work" — gives `evidence_concept != declared_concept`, which is `agreement = "disagreed"` (`type_resolver.py:615`) and therefore a `document_type_disagreement` review item (`type_resolver.py:694`). A document uploaded as "a contract" that turns out to be a master agreement has contradicted nobody, and left alone this puts a review item on **every contract Nick uploads**.

**Files:**
- Modify: `src/services/extraction/type_resolver.py:194` (the docstring enumeration), `:607-620` (the agreement block), plus a new helper
- Test: `tests/services/extraction/test_generic_contract_refinement.py`

**Interfaces:**
- Consumes: `Vocabulary.document_types[...].pipeline_doc_type`.
- Produces: `type_resolver._is_refinement(declared: str, evidence: str, vocab: Vocabulary) -> bool`; a sixth `agreement` value, `refined`.

**Why no change is needed in `type_resolution_discrepancies`.** That function raises on `agreement == "disagreed"`, then on `status == "unresolved"`. A refinement is `agreement="refined"`, `status="matched"`, so it falls through and raises nothing. The suppression is structural rather than a second branch that could drift from the first.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/extraction/test_generic_contract_refinement.py`:

```python
"""'Contract' is the zone's name, not a claim about which contract it is.

The Contracts upload zone declares the category 'contract', which resolves to
doctype.contract_unspecified. A document that then names its real structure has
refined the declaration, not contradicted it. Without this, every contract Nick
uploads lands a review item for having been more specific than the dropdown.

Offline — hand-built vocabularies, no database.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_generic_contract_refinement.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.seed import DocumentType                  # noqa: E402
from src.services.concepts.vocabulary import build_vocabulary         # noqa: E402
from src.services.extraction.type_resolver import (                   # noqa: E402
    resolve_document_type, type_resolution_discrepancies,
)

GENERIC = DocumentType(
    "doctype.contract_unspecified", "role.master", None, None,
    ("contract", "agreement", "contracts"), (), (), "contract",
)
MSA = DocumentType(
    "doctype.master_agreement", "role.master", None, "exec.bilateral",
    ("master agreement", "msa"), (), (), "contract",
)
SOW = DocumentType(
    "doctype.sow", "role.master", "doctype.master_agreement", "exec.bilateral",
    ("sow", "statement of work"), (), (), "contract",
)
INVOICE = DocumentType(
    "doctype.invoice", "role.transaction", None, "exec.unilateral",
    ("invoice", "tax invoice"), (), (), "invoice",
)
NOTICE = DocumentType(
    # pipeline_doc_type is None: recognised, but nothing ingests it.
    "doctype.notice_general", "role.notice", None, "exec.unilateral",
    ("general notice", "notice"), (), (), None,
)


def _vocab(*types):
    concepts = [
        {"concept_code": t.concept_code, "domain": "DOCUMENT_TYPE", "definition": "d",
         "not_to_be_confused_with": [], "status": "active", "rejection_reason": None}
        for t in types
    ]
    rows = [
        {"concept_code": t.concept_code, "role": t.role,
         "default_parent_type": t.default_parent_type, "execution_mode": t.execution_mode,
         "aliases": list(t.aliases), "identifiers": [], "structural_signals": [],
         "pipeline_doc_type": t.pipeline_doc_type, "status": "active",
         "requires_parent_evidence": False, "parent_evidence_phrases": []}
        for t in types
    ]
    return build_vocabulary(concepts, rows, source="test")


V = _vocab(GENERIC, MSA, SOW, INVOICE, NOTICE)


def test_a_master_agreement_uploaded_as_a_contract_is_a_refinement():
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="MASTER AGREEMENT\n\nbetween the parties.\n", vocabulary=V,
    )
    assert (r.agreement, r.evidence_concept) == ("refined", "doctype.master_agreement")
    assert r.status == "matched"


def test_a_refinement_raises_no_review_item():
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="STATEMENT OF WORK\n\nDeliverables and milestones.\n", vocabulary=V,
    )
    assert r.agreement == "refined"
    assert type_resolution_discrepancies(r) == []


def test_an_invoice_uploaded_as_a_contract_still_disagrees():
    """Refinement is bounded by the pipeline, so a transaction document is not one."""
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="TAX INVOICE\n\nAmount due on receipt.\n", vocabulary=V,
    )
    assert (r.agreement, r.evidence_concept) == ("disagreed", "doctype.invoice")
    items = type_resolution_discrepancies(r)
    assert len(items) == 1
    assert items[0].issue_type == "document_type_disagreement"


def test_a_structure_with_no_pipeline_is_not_a_refinement():
    """doctype.notice_general has pipeline_doc_type NULL.

    'Recognised but nothing ingests it' is not 'a kind of contract', and reading
    it as a refinement would silence a finding on a document the contract
    pipeline cannot actually handle.
    """
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="GENERAL NOTICE\n\nFor information only.\n", vocabulary=V,
    )
    assert r.agreement == "disagreed", r
    assert len(type_resolution_discrepancies(r)) == 1


def test_one_specific_structure_declared_against_another_still_disagrees():
    """Only the GENERIC declaration can be refined.

    Uploading as a SOW a document that reads as a master agreement is a real
    contradiction: the uploader made a specific claim and the page disputes it.
    """
    r = resolve_document_type(
        declared_concept="doctype.sow",
        full_text="MASTER AGREEMENT\n\nbetween the parties.\n", vocabulary=V,
    )
    assert (r.agreement, r.evidence_concept) == ("disagreed", "doctype.master_agreement")
    assert len(type_resolution_discrepancies(r)) == 1


def test_a_generic_contract_reading_as_generic_is_agreed_not_refined():
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="CONTRACT\n\nnumbered clauses and a signature block.\n", vocabulary=V,
    )
    assert r.agreement == "agreed"


def test_an_unresolved_tie_under_a_generic_declaration_still_reaches_a_human():
    """A refinement must not swallow a tie: two candidates still need a person.

    status, never agreement alone, is what the review-item builder reads.
    """
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="MASTER AGREEMENT / STATEMENT OF WORK\n\nbetween the parties.\n",
        vocabulary=V,
    )
    assert r.status == "unresolved", r
    assert len(type_resolution_discrepancies(r)) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_generic_contract_refinement.py -v
```

Expected: `test_a_master_agreement_uploaded_as_a_contract_is_a_refinement` and `test_a_refinement_raises_no_review_item` FAIL (`'disagreed' != 'refined'`, and one review item where none belongs). The four tests asserting `disagreed` PASS already — they are what must stay true.

- [ ] **Step 3: Add the helper**

In `src/services/extraction/type_resolver.py`, beside `_names_a_parent`:

```python
def _is_refinement(declared: str, evidence: str, vocab: "Vocabulary") -> bool:
    """Has the page named which KIND of contract this is, rather than contradicted it?

    'contract' is the name of the upload zone, not a claim about which contract
    the document is: routing.pipeline_for_category maps it to
    doctype.contract_unspecified, whose whole purpose is to say "a contract, kind
    not stated". A page that then says "Master Agreement" has refined that, and
    calling it a disagreement would put a review item on every contract uploaded.

    Bounded two ways, because a refinement SILENCES a finding and a silence that
    is too wide is the expensive direction:
      * only the generic declaration can be refined -- declaring a SOW and
        reading a master agreement is a real contradiction, since the uploader
        made a specific claim;
      * only by a structure the contract pipeline actually ingests. A structure
        with pipeline_doc_type NULL is "recognised, but nothing ingests it",
        which is not a kind of contract.
    """
    if declared != "doctype.contract_unspecified" or evidence == declared:
        return False
    dt = vocab.document_types.get(evidence)
    return bool(dt is not None and dt.pipeline_doc_type == "contract")
```

- [ ] **Step 4: Extend the agreement block**

Replace the `if declared_concept:` branch (lines 607-620) so the new case sits between `agreed` and `disagreed`:

```python
    if declared_concept:
        # A declared type is a human's statement and always stands. The page
        # only ever adds a second reading.
        if evidence_concept is None:
            agreement = "declared_only"
        elif evidence_concept == declared_concept:
            agreement = "agreed"
        elif _is_refinement(declared_concept, evidence_concept, vocab):
            # Not a contradiction: the zone said "a contract", the page said
            # which one. type_resolution_discrepancies raises on 'disagreed' and
            # on an unresolved status, so this falls through and raises nothing
            # -- the suppression is structural, not a second branch that could
            # drift from the first.
            agreement = "refined"
        else:
            agreement = "disagreed"
        if status != "unresolved":
            status = "matched"
    else:
        agreement = "evidence_only" if evidence_concept else "neither"
```

Update the `TypeResolution.agreement` comment at line 194:

```python
    agreement: str      # agreed | refined | declared_only | evidence_only | disagreed | neither
```

and add a line to `type_resolution_discrepancies`' docstring, after the paragraph about reading `status` rather than `agreement`:

```
    'refined' raises nothing by falling through: the Contracts zone declares the
    generic 'contract' and a page naming its actual structure has not
    contradicted anybody. See _is_refinement for the two bounds on that silence.
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_generic_contract_refinement.py \
    tests/services/extraction/test_type_resolver.py \
    tests/services/extraction/test_type_resolver_matched_evidence.py \
    tests/services/extraction/test_type_resolution_review_items.py -v
```

Expected: all pass.

- [ ] **Step 6: Confirm the baseline is unchanged**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_classification_baseline.py \
    tests/services/extraction/test_type_findings_lifecycle.py -v
```

Expected: pass. Zero contract-category documents have ever been uploaded, so no stored document can take the new path — if the baseline moves, the bounds in `_is_refinement` are too wide and a non-contract document is being silenced.

- [ ] **Step 7: Prove the guards fail (required)**

1. **The refinement fires.** Make `_is_refinement` return `False` unconditionally. Expected: the first two tests FAIL. Restore.
2. **The pipeline bound holds.** Change the last line to `return dt is not None`. Expected: `test_a_structure_with_no_pipeline_is_not_a_refinement` FAILS. Restore.
3. **The generic bound holds.** Drop the `declared != "doctype.contract_unspecified"` condition. Expected: `test_one_specific_structure_declared_against_another_still_disagrees` FAILS. Restore.
4. **A tie still reaches a human.** Make the `refined` branch also set `status = "matched"` unconditionally (above the existing `if status != "unresolved"`). Expected: `test_an_unresolved_tie_under_a_generic_declaration_still_reaches_a_human` FAILS. Restore.

- [ ] **Step 8: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add src/services/extraction/type_resolver.py \
        tests/services/extraction/test_generic_contract_refinement.py
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "fix(type resolver): naming which contract it is refines, not contradicts" "" \
  "Task 5 of specs/2026-10-02-contract-structures-plan.md." "" \
  "The Contracts zone declares the generic 'contract'. A page saying 'Master" \
  "Agreement' was therefore a disagreement, which would have put a review item" \
  "on every contract uploaded -- invisible today only because no contract ever" \
  "has been." "" \
  "The silence is bounded twice: only the generic declaration can be refined," \
  "and only by a structure the contract pipeline ingests. Both bounds proven" \
  "red, as is the tie that must still reach a person." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- src/services/extraction/type_resolver.py \
    tests/services/extraction/test_generic_contract_refinement.py
git diff --cached --name-status | wc -l
```

---
## Task 6: The answer gets stored

This is the blocker the whole design turns on. `dispatch.py:542` records the classification *"never acted on"* — a log line and, on disagreement, a review item. Nothing can ask "is this a SOW?", so no maths can run.

**Files:**
- Create: `deploy/sql/2026-10-02_contract_raw_resolved_type.sql`
- Create: `deploy/sql/2026-10-02_contract_raw_resolved_type_rollback.sql`
- Modify: `src/services/extraction/persistence.py:159-210` (`write_raw`)
- Modify: `src/services/extraction/dispatch.py:798-807` (the `write_raw` call)
- Test: `tests/services/extraction/test_resolved_type_is_stored.py`

**Interfaces:**
- Consumes: the `TypeResolution` already computed in `dispatch.py:551`.
- Produces: `write_raw(..., resolved_doc_type: str | None = None, resolved_role: str | None = None, type_agreement: str | None = None) -> int`. Three columns on `proc.bp_contract_raw` and `proc.bp_contracts`.

**Why `bp_contract_raw` and not `bp_contract_master`.** `bp_contract_raw` is the only contract table carrying `process_monitor_id` and `source_file` — the per-document identity. `bp_contracts` and `bp_contract_master` carry neither, so a classification written there could not be traced back to the document that produced it.

**Why promotion needs no code change.** `promotion.py:858` computes `target_cols = [c for c in stg_cols if c in raw_data and c not in _CONTROL_COLS]`, reading `stg_cols` from `information_schema`. Adding the same three columns to both tables means promotion carries them. That is a claim about existing behaviour, so Step 7 proves it rather than assuming it.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/extraction/test_resolved_type_is_stored.py`:

```python
"""A classification nothing records is a classification no maths can use.

dispatch.py records the resolved structure "never acted on": a log line, and on
disagreement a review item. These tests pin it to a column, and pin that the
column survives promotion and refreshes on a re-read.

Live-only. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/extraction/test_resolved_type_is_stored.py -v
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction import persistence                      # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_COLS = ("resolved_doc_type", "resolved_role", "type_agreement")


@pytest.fixture()
def cleanup():
    """Remove only the rows this test made, by its own unique contract_id."""
    made: list[str] = []
    yield made
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        for contract_id in made:
            cur.execute("DELETE FROM proc.bp_contracts WHERE contract_id = %s", (contract_id,))
            cur.execute("DELETE FROM proc.bp_contract_raw WHERE contract_id = %s", (contract_id,))


def _write(contract_id, *, resolved="doctype.sow", role="role.master",
           agreement="refined", source_file=None):
    return persistence.write_raw(
        doc_type="contract",
        file_path=source_file or f"documents/contract/{contract_id}.pdf",
        process_monitor_id=None,
        trace_id=uuid.uuid4(),
        pipeline_version="test",
        columns={"contract_id": contract_id, "supplier_id": "S-TEST"},
        parser_snapshot={"full_text": "STATEMENT OF WORK\n"},
        promotion_status="pending",
        resolved_doc_type=resolved,
        resolved_role=role,
        type_agreement=agreement,
    )


def _read(table, contract_id):
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(_COLS)} FROM proc.{table} WHERE contract_id = %s "
            f"ORDER BY {'raw_id DESC' if table == 'bp_contract_raw' else 'contract_id'} LIMIT 1",
            (contract_id,),
        )
        row = cur.fetchone()
    return dict(zip(_COLS, row)) if row else None


def test_both_tables_have_the_three_columns():
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        for table in ("bp_contract_raw", "bp_contracts"):
            cur.execute(
                """SELECT column_name FROM information_schema.columns
                    WHERE table_schema='proc' AND table_name=%s AND column_name = ANY(%s)""",
                (table, list(_COLS)),
            )
            got = sorted(r[0] for r in cur.fetchall())
            assert got == sorted(_COLS), f"{table} is missing columns: {got}"


def test_the_resolved_structure_reaches_the_raw_row(cleanup):
    cid = f"CT-{uuid.uuid4().hex[:10].upper()}"
    cleanup.append(cid)
    _write(cid)
    assert _read("bp_contract_raw", cid) == {
        "resolved_doc_type": "doctype.sow",
        "resolved_role": "role.master",
        "type_agreement": "refined",
    }


def test_a_page_that_states_nothing_stores_null_never_a_guess(cleanup):
    """No fabrication: an unrecognised page records no structure.

    The tempting fallback -- store the declared type when the page is silent --
    would make every document look as though it had confirmed its own category.
    """
    cid = f"CT-{uuid.uuid4().hex[:10].upper()}"
    cleanup.append(cid)
    _write(cid, resolved=None, role=None, agreement="declared_only")
    assert _read("bp_contract_raw", cid) == {
        "resolved_doc_type": None, "resolved_role": None,
        "type_agreement": "declared_only",
    }


def test_the_structure_survives_promotion(cleanup):
    cid = f"CT-{uuid.uuid4().hex[:10].upper()}"
    cleanup.append(cid)
    raw_id = _write(cid, resolved="doctype.master_agreement", role="role.master",
                    agreement="refined")
    from src.services.extraction import promotion
    promotion.promote(raw_id, "contract")
    assert _read("bp_contracts", cid) == {
        "resolved_doc_type": "doctype.master_agreement",
        "resolved_role": "role.master",
        "type_agreement": "refined",
    }


def test_a_reread_refreshes_the_stored_structure(cleanup):
    """Review Focus 3: _stg updating while the promoted row stays stale is an
    existing, measured failure in this product. A re-read that corrects the
    structure must correct it everywhere, or the maths links on a stale answer.
    """
    cid = f"CT-{uuid.uuid4().hex[:10].upper()}"
    cleanup.append(cid)
    from src.services.extraction import promotion

    first = _write(cid, resolved="doctype.contract_unspecified", agreement="agreed")
    promotion.promote(first, "contract")
    assert _read("bp_contracts", cid)["resolved_doc_type"] == "doctype.contract_unspecified"

    second = _write(cid, resolved="doctype.sow", agreement="refined")
    promotion.promote(second, "contract")
    assert _read("bp_contracts", cid) == {
        "resolved_doc_type": "doctype.sow",
        "resolved_role": "role.master",
        "type_agreement": "refined",
    }, "the promoted row kept the first read's structure"


def test_the_other_three_doc_types_are_unaffected():
    """Only bp_contract_raw has these columns, so write_raw must not send them
    to an invoice, quote or purchase order -- that would be an UndefinedColumn
    error on the live ingestion path for three of the four pipelines.
    """
    import inspect
    src = inspect.getsource(persistence.write_raw)
    assert 'doc_type == "contract"' in src, (
        "write_raw must gate the three columns on the contract doc_type"
    )
    cid = f"IV-{uuid.uuid4().hex[:10].upper()}"
    raw_id = persistence.write_raw(
        doc_type="invoice",
        file_path=f"documents/invoice/{cid}.pdf",
        process_monitor_id=None,
        trace_id=uuid.uuid4(),
        pipeline_version="test",
        columns={"invoice_id": cid},
        parser_snapshot={"full_text": "TAX INVOICE\n"},
        promotion_status="pending",
        resolved_doc_type="doctype.invoice",
        resolved_role="role.transaction",
        type_agreement="agreed",
    )
    assert raw_id
    from src.services.db import get_conn
    with get_conn() as conn:
        conn.cursor().execute("DELETE FROM proc.bp_invoice_raw WHERE raw_id = %s", (raw_id,))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_resolved_type_is_stored.py -v
```

Expected: all FAIL — the column test on an empty set, the rest with `TypeError: write_raw() got an unexpected keyword argument 'resolved_doc_type'`.

**`promotion.promote(raw_id, doc_type)` DELETES the `_raw` row once it has copied the columns** (`promotion.py:583` — "Copy _raw flat columns into _stg, delete _raw"). So never assert on `bp_contract_raw` after promoting, and expect the `cleanup` fixture's delete from that table to match nothing for a promoted document. The re-read test works with this rather than against it: each read writes a fresh raw row and promotes it, which is exactly what a live re-extraction does.

- [ ] **Step 3: Write the migration**

Create `deploy/sql/2026-10-02_contract_raw_resolved_type.sql`:

```sql
-- Where a contract document's recognised structure is recorded.
--
-- src/services/extraction/dispatch.py:542 says the classification is
-- "Recorded, never acted on" -- it reaches a log line and, on disagreement, a
-- review item, and is never written to the document's row. So nothing could ask
-- "is this a SOW?", and no relationship maths could run. These three columns are
-- the answer to that.
--
-- bp_contract_raw, because it is the ONLY contract table carrying
-- process_monitor_id and source_file -- the per-document identity. bp_contracts
-- and bp_contract_master carry neither, so a classification written there could
-- not be traced to the document that produced it.
--
-- Both tables get the columns because promotion copies the intersection of raw
-- and target columns (promotion.py:858, via information_schema) minus
-- _CONTROL_COLS. None of these three is a control column, so promotion carries
-- them with no code change.
--
-- NULLABLE ON PURPOSE. A page that states no structure stores NULL, never the
-- declared category as a stand-in: that would make every document look as though
-- it had confirmed its own upload category.
--
-- Not a foreign key to proc.bp_concept. A concept can be retired or renamed, and
-- an FK would then either block the retirement or rewrite history on a document
-- that genuinely did read as the old structure. The value is a record of what the
-- classifier concluded at extraction time.
--
-- Additive, idempotent, reversible.
BEGIN;

ALTER TABLE proc.bp_contract_raw
    ADD COLUMN IF NOT EXISTS resolved_doc_type text,
    ADD COLUMN IF NOT EXISTS resolved_role     text,
    ADD COLUMN IF NOT EXISTS type_agreement    text;

ALTER TABLE proc.bp_contracts
    ADD COLUMN IF NOT EXISTS resolved_doc_type text,
    ADD COLUMN IF NOT EXISTS resolved_role     text,
    ADD COLUMN IF NOT EXISTS type_agreement    text;

-- Candidate parents are looked up by structure, so the lookup gets an index.
CREATE INDEX IF NOT EXISTS ix_bp_contracts_resolved_doc_type
    ON proc.bp_contracts (resolved_doc_type);

COMMENT ON COLUMN proc.bp_contracts.resolved_doc_type IS
    'The concept_code the document''s own text resolved to, e.g. doctype.sow. '
    'NULL when the page stated nothing -- never the declared category as a '
    'stand-in.';
COMMENT ON COLUMN proc.bp_contracts.resolved_role IS
    'That structure''s relationship role, e.g. role.master, copied at extraction '
    'time.';
COMMENT ON COLUMN proc.bp_contracts.type_agreement IS
    'agreed | refined | declared_only | evidence_only | disagreed | neither -- '
    'how the page''s reading compared with the uploader''s declaration.';

COMMIT;
```

Create `deploy/sql/2026-10-02_contract_raw_resolved_type_rollback.sql`:

```sql
-- Removes exactly what 2026-10-02_contract_raw_resolved_type.sql adds.
-- This DISCARDS every recorded structure: the values are re-derivable only by
-- re-extracting each document, so take a copy first if the classifications
-- matter.
BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_contracts_resolved_doc_type;

ALTER TABLE proc.bp_contracts
    DROP COLUMN IF EXISTS type_agreement,
    DROP COLUMN IF EXISTS resolved_role,
    DROP COLUMN IF EXISTS resolved_doc_type;

ALTER TABLE proc.bp_contract_raw
    DROP COLUMN IF EXISTS type_agreement,
    DROP COLUMN IF EXISTS resolved_role,
    DROP COLUMN IF EXISTS resolved_doc_type;

COMMIT;
```

- [ ] **Step 4: Apply the migration to bp_testdb**

```bash
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    -v ON_ERROR_STOP=1 -f deploy/sql/2026-10-02_contract_raw_resolved_type.sql
```

Run it twice; the second run must succeed unchanged.

- [ ] **Step 5: Teach `write_raw` the three columns**

In `src/services/extraction/persistence.py`, add the three keyword arguments and append them to `base_cols` **only for the contract doc type**:

```python
def write_raw(
    *,
    doc_type: str,
    file_path: str,
    process_monitor_id: int | None,
    trace_id: UUID,
    pipeline_version: str,
    columns: Mapping[str, Any],
    parser_snapshot: Mapping[str, Any],
    promotion_status: str,
    resolved_doc_type: str | None = None,
    resolved_role: str | None = None,
    type_agreement: str | None = None,
) -> int:
    """INSERT one row into proc.bp_<doctype>_raw. Returns raw_id.

    `columns` already has only db_columns present in the _raw table.

    The three ``resolved_*`` / ``type_agreement`` arguments carry what the
    document's own text said it was. They are accepted for every doc_type and
    WRITTEN only for 'contract', because only proc.bp_contract_raw has the
    columns -- sending them elsewhere would be an UndefinedColumn error on the
    live ingestion path for the other three pipelines. Callers therefore need no
    doc-type branch of their own.
    """
```

and after the existing `base_vals` assignment:

```python
    # Only proc.bp_contract_raw has these columns
    # (deploy/sql/2026-10-02_contract_raw_resolved_type.sql). Gated here rather
    # than at the call site so dispatch stays free of a doc-type branch, and so
    # the gate lives next to the SQL it protects.
    if doc_type == "contract":
        base_cols += ["resolved_doc_type", "resolved_role", "type_agreement"]
        base_vals += [resolved_doc_type, resolved_role, type_agreement]
```

- [ ] **Step 6: Pass the resolution from dispatch**

In `src/services/extraction/dispatch.py`, extend the `write_raw` call at line 798. `type_resolution` is already in scope from line 551 and is `None` when the resolver raised, so each lookup must tolerate that:

```python
    # What the document said it was, carried to the row so something other than a
    # log line can read it. type_resolution is None when the resolver raised --
    # extraction continues in that case, and the columns stay NULL rather than
    # recording a structure nobody derived.
    _resolved = type_resolution.evidence_concept if type_resolution else None
    _role = None
    if _resolved:
        from src.services.concepts.vocabulary import ensure_vocabulary
        _dt = ensure_vocabulary().document_types.get(_resolved)
        _role = _dt.role if _dt else None

    raw_id = persistence.write_raw(
        doc_type=doc_type,
        file_path=file_path,
        process_monitor_id=process_monitor_id,
        trace_id=trace_id,
        pipeline_version=pipeline_version,
        columns=columns,
        parser_snapshot=_parser_snapshot,
        promotion_status=promotion_status,
        resolved_doc_type=_resolved,
        resolved_role=_role,
        type_agreement=type_resolution.agreement if type_resolution else None,
    )
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/test_resolved_type_is_stored.py -v
```

Expected: 6 passed.

Then the surrounding suite, because `write_raw` is on the live path for all four pipelines:

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/ tests/services/concepts/ -v
```

- [ ] **Step 8: Prove the guards fail (required)**

1. **The write happens.** Change `base_vals += [resolved_doc_type, ...]` to `+= [None, None, None]`. Expected: `test_the_resolved_structure_reaches_the_raw_row` FAILS. Restore.
2. **Promotion carries the columns** — the claim about existing behaviour. Add `"resolved_doc_type"` to `_CONTROL_COLS` in `promotion.py`. Expected: `test_the_structure_survives_promotion` FAILS with `None` in the promoted row. Restore. This is the proof that promotion needed no change, rather than the assumption.
3. **A re-read refreshes.** In `promotion.py`, change the upsert's `DO UPDATE SET` to `DO NOTHING`. Expected: `test_a_reread_refreshes_the_stored_structure` FAILS, the promoted row holding `doctype.contract_unspecified`. Restore.
4. **No fabrication.** In `dispatch.py`, change `_resolved` to `type_resolution.evidence_concept or declared_concept`. Expected: `test_a_page_that_states_nothing_stores_null_never_a_guess` FAILS. Restore.
5. **The contract gate holds.** Remove the `if doc_type == "contract":` condition so the columns are always sent. Expected: `test_the_other_three_doc_types_are_unaffected` FAILS with an `UndefinedColumn` error on `bp_invoice_raw`. Restore.

- [ ] **Step 9: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add deploy/sql/2026-10-02_contract_raw_resolved_type.sql \
        deploy/sql/2026-10-02_contract_raw_resolved_type_rollback.sql \
        src/services/extraction/persistence.py \
        src/services/extraction/dispatch.py \
        tests/services/extraction/test_resolved_type_is_stored.py
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "feat(contracts): the recognised structure is stored, not just logged" "" \
  "Task 6 of specs/2026-10-02-contract-structures-plan.md." "" \
  "dispatch.py recorded the classification 'never acted on' -- a log line and," \
  "on disagreement, a review item. Nothing could ask whether a document is a" \
  "SOW, so no maths could run. resolved_doc_type, resolved_role and" \
  "type_agreement now reach bp_contract_raw and survive promotion." "" \
  "A silent page stores NULL, never the declared category. Promotion needed no" \
  "code change, and that is proven by adding the column to _CONTROL_COLS and" \
  "watching the test go red rather than by assuming it. Five guards proven." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- deploy/sql/2026-10-02_contract_raw_resolved_type.sql \
    deploy/sql/2026-10-02_contract_raw_resolved_type_rollback.sql \
    src/services/extraction/persistence.py src/services/extraction/dispatch.py \
    tests/services/extraction/test_resolved_type_is_stored.py
git diff --cached --name-status | wc -l
```

---
## Task 7: The existing corpus's free-text type becomes a structure

3,051 contracts already sit in `proc.bp_contract_master` with a free-text `contract_type`. The maths needs candidate parents, and a candidate set drawn only from newly uploaded documents would be empty for a long time. Reading the existing corpus's structure makes those 3,051 rows available as parents on day one.

**Files:**
- Create: `src/services/concepts/contract_type_map.py`
- Test: `tests/services/concepts/test_contract_type_map.py`

**Interfaces:**
- Consumes: `vocabulary.ensure_vocabulary`, `vocabulary.resolve_alias`.
- Produces:
  - `structure_for_contract_type(value: str | None, *, vocabulary: Vocabulary | None = None) -> str | None`
  - `coverage(rows: Iterable[Mapping]) -> dict` → `{"mapped": int, "unmapped": int, "unmapped_values": dict[str, int]}`

**This task writes nothing.** `proc.bp_contract_master.contract_type` is source data and is never modified (`project_extraction_accuracy_priority`). The mapping is a read-time function, so a vocabulary change takes effect without a backfill and without a second copy of the answer to keep in step.

**Measured, 2026-10-02:** 9 of the 13 distinct values resolve, covering **3,016 of 3,051 rows**. The 35 that do not are `NULL` (29), `Service` (2), `Indirect Procurement` (1) and `Policy` (3). `Policy` is not a vocabulary gap — `doctype.policy_document` claims the alias `policy` but its concept is `status='proposed'`, and a proposed concept deliberately never resolves.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/concepts/test_contract_type_map.py`:

```python
"""The 3,051-row corpus's free-text contract_type, read as a structure.

Read-only: proc.bp_contract_master.contract_type is source data and is never
written. The mapping is a function of the vocabulary, so confirming a concept
takes effect with no backfill and leaves no second copy to drift.

Live test needs the database. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/concepts/test_contract_type_map.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import contract_type_map as M   # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")

#: Every distinct value in proc.bp_contract_master, measured 2026-10-02, with the
#: structure it must resolve to. None means "must not resolve".
CORPUS_VALUES = {
    "Consulting": "doctype.consulting_agreement",
    "NDA": "doctype.nda",
    "SLA": "doctype.sla",
    "Master Agreement": "doctype.master_agreement",
    "Service Agreement": "doctype.service_agreement",
    "Service Contract": "doctype.service_agreement",
    "Invoice": "doctype.invoice",
    "Amendment": "doctype.variation",
    "Purchase Order": "doctype.order",
    "Policy": None,                 # doctype.policy_document is status='proposed'
    "Service": None,
    "Indirect Procurement": None,
    None: None,
}


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
@pytest.mark.parametrize("value,expected", list(CORPUS_VALUES.items()))
def test_every_corpus_value_maps_as_measured(value, expected):
    assert M.structure_for_contract_type(value) == expected


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_a_proposed_concept_never_resolves():
    """'Policy' is the mechanism working, not a gap.

    doctype.policy_document claims the alias 'policy'. Its concept is
    status='proposed', so it must not resolve -- if it did, confirming it would
    be moot and an unreviewed structure would be feeding the maths.
    """
    assert M.structure_for_contract_type("Policy") is None
    assert M.structure_for_contract_type("policy") is None


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_the_whole_corpus_maps_at_the_measured_rate():
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT contract_type, count(*) FROM proc.bp_contract_master GROUP BY 1"
        )
        rows = [{"contract_type": r[0], "n": r[1]} for r in cur.fetchall()]
    total = sum(r["n"] for r in rows)
    assert total == 3051, f"the corpus changed size: {total}"
    cov = M.coverage(rows)
    assert cov["mapped"] == 3016, cov
    assert cov["unmapped"] == 35, cov
    assert cov["unmapped_values"] == {
        "": 29, "Policy": 3, "Service": 2, "Indirect Procurement": 1,
    }, cov["unmapped_values"]


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_the_mapping_is_case_and_spacing_insensitive():
    """It goes through the same fold() every alias comparison uses."""
    for spelling in ("master agreement", "MASTER AGREEMENT", "  Master   Agreement ",
                     "master-agreement", "master_agreement"):
        assert M.structure_for_contract_type(spelling) == "doctype.master_agreement", spelling


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_an_ambiguous_value_resolves_to_nothing_rather_than_picking():
    """Two structures claiming one spelling is a collision, not a tie to break.

    resolve_alias returns every owner; choosing the first would make the answer
    depend on row order and then record that choice as a fact.
    """
    import dataclasses
    from src.services.concepts import vocabulary as V
    from src.services.concepts.seed import DocumentType

    vocab = V.ensure_vocabulary()
    clash = DocumentType("doctype.sla", "role.attachment", None, "exec.incorporated",
                         ("sla", "consulting"), (), (), "contract")
    types = dict(vocab.document_types)
    types["doctype.sla"] = clash
    contested = dataclasses.replace(vocab, document_types=types)
    contested = V.build_vocabulary(
        [{"concept_code": c.concept_code, "domain": c.domain, "definition": c.definition,
          "not_to_be_confused_with": list(c.not_to_be_confused_with), "status": "active",
          "rejection_reason": None} for c in contested.concepts.values()],
        [{"concept_code": d.concept_code, "role": d.role,
          "default_parent_type": d.default_parent_type, "execution_mode": d.execution_mode,
          "aliases": list(d.aliases), "identifiers": list(d.identifiers),
          "structural_signals": list(d.structural_signals),
          "pipeline_doc_type": d.pipeline_doc_type, "status": d.status,
          "requires_parent_evidence": d.requires_parent_evidence,
          "parent_evidence_phrases": list(d.parent_evidence_phrases)}
         for d in types.values()],
        source="test-contested",
    )
    assert M.structure_for_contract_type("consulting", vocabulary=contested) is None


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_blank_and_whitespace_are_not_a_structure():
    """An absent type is not a structure, and must not become one by accident."""
    for value in (None, "", "   ", "\n", "\t  \n"):
        assert M.structure_for_contract_type(value) is None, repr(value)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/concepts/test_contract_type_map.py -v
```

Expected: collection FAILS with `ModuleNotFoundError: No module named 'src.services.concepts.contract_type_map'`.

- [ ] **Step 3: Write the module**

Create `src/services/concepts/contract_type_map.py`:

```python
"""Read the existing corpus's free-text contract_type as a structure.

proc.bp_contract_master holds 3,051 contracts whose contract_type is free text:
'Consulting', 'NDA', 'SLA', 'Master Agreement' and ten more. The relationship
maths needs candidate parents, and a candidate set drawn only from newly
uploaded documents would be empty for a long time -- so these 3,051 rows have to
be readable as structures.

Two decisions worth knowing:

  * NOTHING IS WRITTEN. contract_type is source data and is never modified
    (see the extraction-accuracy standing rule). This is a read-time function,
    so confirming a concept takes effect with no backfill and leaves no second
    copy of the answer to drift from the first.
  * IT GOES THROUGH THE ALIAS INDEX, not a hand-written dictionary. A dictionary
    here would be a fourth place the vocabulary lives, and the three that already
    exist (table, seed, alias index) are kept in step only by a drift test. The
    consequence is that a 'proposed' concept does not resolve -- which is the
    mechanism working, not a gap: 'Policy' matches doctype.policy_document's
    alias but that concept awaits confirmation.

Measured 2026-10-02: 9 of the 13 distinct values resolve, covering 3,016 of the
3,051 rows. The other 35 are NULL (29), 'Service' (2), 'Indirect Procurement'
(1) and 'Policy' (3).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Optional

from .vocabulary import Vocabulary, ensure_vocabulary, resolve_alias


def structure_for_contract_type(
    value: Optional[str],
    *,
    vocabulary: Optional[Vocabulary] = None,
) -> Optional[str]:
    """The concept_code this free-text contract_type names, or None.

    None for a blank value, for a value nothing claims, and for a value TWO
    structures claim. The last of those is the one worth stating: picking the
    first owner would make the answer depend on row order, and the structure it
    chose would then be recorded as a fact and linked on.
    """
    raw = (value or "").strip()
    if not raw:
        return None
    vocab = vocabulary if vocabulary is not None else ensure_vocabulary()
    owners = resolve_alias(raw, vocab)
    return owners[0] if len(owners) == 1 else None


def coverage(
    rows: Iterable[Mapping],
    *,
    vocabulary: Optional[Vocabulary] = None,
) -> dict:
    """How much of the corpus maps, and what the remainder actually is.

    ``rows`` are mappings with ``contract_type`` and ``n`` (a row count).

    The unmapped values are returned, not just counted. 'unmapped: 35' invites
    the assumption that the vocabulary is short of 35 documents' worth of
    structures; naming them shows that 29 are NULL in the source and 3 are a
    concept awaiting confirmation, which are three different problems.
    """
    vocab = vocabulary if vocabulary is not None else ensure_vocabulary()
    mapped = 0
    unmapped = 0
    unmapped_values: Counter = Counter()
    for row in rows:
        value = row.get("contract_type")
        count = int(row.get("n") or 0)
        if structure_for_contract_type(value, vocabulary=vocab):
            mapped += count
        else:
            unmapped += count
            unmapped_values[(value or "").strip()] += count
    return {
        "mapped": mapped,
        "unmapped": unmapped,
        "unmapped_values": dict(unmapped_values),
    }


__all__ = ["structure_for_contract_type", "coverage"]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/concepts/test_contract_type_map.py -v
```

Expected: all pass, including the exact 3,016 / 35 split. A different split means the corpus or the vocabulary changed; report the numbers rather than editing the expectation to match.

- [ ] **Step 5: Confirm the source column is untouched**

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "SELECT contract_type, count(*) FROM proc.bp_contract_master GROUP BY 1 ORDER BY 2 DESC;"
```

Expected: the same 13 values with the same counts as §2 of the design — `Consulting` 645, `NDA` 612, `SLA` 585, `Master Agreement` 581, `Service Agreement` 579, `NULL` 29, `Service Contract` 7, `Invoice` 4, `Policy` 3, `Service` 2, `Amendment` 2, `Indirect Procurement` 1, `Purchase Order` 1.

- [ ] **Step 6: Prove the guards fail (required)**

1. **Ambiguity is refused.** Change the return to `return owners[0] if owners else None`. Expected: `test_an_ambiguous_value_resolves_to_nothing_rather_than_picking` FAILS. Restore.
2. **A proposed concept stays out.** On `bp_testdb`, run
   `UPDATE proc.bp_concept SET status='active' WHERE concept_code='doctype.policy_document';`
   and `UPDATE proc.bp_document_type SET status='active' WHERE concept_code='doctype.policy_document';`
   Expected: `test_a_proposed_concept_never_resolves` and the coverage test FAIL (3,019 / 32). This proves the behaviour is the status rule and not a coincidence. Restore both to `'proposed'` and confirm green.
3. **Coverage names the remainder.** Make `coverage` return `{"mapped": m, "unmapped": u, "unmapped_values": {}}`. Expected: the coverage test FAILS on the `unmapped_values` assertion. Restore.

- [ ] **Step 7: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add src/services/concepts/contract_type_map.py \
        tests/services/concepts/test_contract_type_map.py
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "feat(concepts): read the corpus's free-text contract_type as a structure" "" \
  "Task 7 of specs/2026-10-02-contract-structures-plan.md." "" \
  "3,051 existing contracts become candidate parents on day one instead of" \
  "waiting for uploads. Measured: 9 of 13 distinct values resolve, covering" \
  "3,016 of 3,051 rows." "" \
  "Read-time through the alias index, not a hand-written dictionary -- a fourth" \
  "copy of the vocabulary would need a fourth drift test. Nothing is written:" \
  "contract_type is source data. 'Policy' not resolving is the proposed-status" \
  "rule working, proven by activating the concept and watching the count move." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- src/services/concepts/contract_type_map.py \
    tests/services/concepts/test_contract_type_map.py
git diff --cached --name-status | wc -l
```

---
## Task 8: The parent reference becomes extractable

The vocabulary declares that a call-off points at its framework through a field named `framework_ref`, and that a SOW sits under a master agreement. **`framework_ref` does not exist** in `extraction_schemas/contract.yaml`, so the pointer is declared and never filled.

**Files:**
- Modify: `extraction_schemas/contract.yaml` (insert both fields after the `parent_contract_id` block, ~line 385)
- Test: `tests/services/extraction/test_contract_parent_reference_fields.py`

**Interfaces:**
- Consumes: the YAML loader `src/services/extraction_v3/yaml_schema/loader.load_doc_schema`.
- Produces: two extracted fields, `framework_ref` and `parent_agreement_ref`, reaching `columns` in `dispatch.py` and therefore `proc.bp_contract_raw`.

**Why two fields and not one.** `parent_contract_id` already anchors on "Parent/Master/Principal Contract|Agreement" and on "amends|amendment to|supplements|varies", and keeps that job. A SOW names its MSA **without amending anything**, so collapsing the two would make "the document this one sits under" and "the document this one changes" the same fact — and the hierarchy maths needs to tell a child from an amendment.

**Both are `required: false`.** A contract naming no parent extracts and promotes exactly as it does now. Making either required would block promotion on every standalone agreement, which is most of them.

- [ ] **Step 1: Add the migration for the two columns**

The raw and promoted tables need somewhere to put the values. Create `deploy/sql/2026-10-02_contract_parent_reference_columns.sql`:

```sql
-- Two reference columns the vocabulary has always declared and the schema never
-- had a field for.
--
-- proc.bp_document_type says doctype.call_off_contract points at its framework
-- through a field called framework_ref, and doctype.order_form does the same.
-- No such field existed in extraction_schemas/contract.yaml, so the pointer was
-- declared and never filled.
--
-- SEPARATE FROM parent_contract_id, which stays as it is. A SOW names the master
-- agreement it sits under without amending anything; parent_contract_id anchors
-- on 'amends', 'supplements', 'varies'. "The document this one sits under" and
-- "the document this one changes" are different facts, and the hierarchy maths
-- needs to tell a child from an amendment.
--
-- Additive, idempotent, reversible.
BEGIN;

ALTER TABLE proc.bp_contract_raw
    ADD COLUMN IF NOT EXISTS framework_ref        text,
    ADD COLUMN IF NOT EXISTS parent_agreement_ref text;

ALTER TABLE proc.bp_contracts
    ADD COLUMN IF NOT EXISTS framework_ref        text,
    ADD COLUMN IF NOT EXISTS parent_agreement_ref text;

COMMENT ON COLUMN proc.bp_contracts.framework_ref IS
    'The framework agreement this document is called off under, as the document '
    'states it. A reference as printed, never a resolved key -- 1,561 existing '
    'parent_contract_id values resolve to 0 real contracts, which is why nothing '
    'auto-links on a reference alone.';
COMMENT ON COLUMN proc.bp_contracts.parent_agreement_ref IS
    'The agreement this document sits under without amending, as the document '
    'states it. parent_contract_id carries the amends/supplements/varies case.';

COMMIT;
```

Create `deploy/sql/2026-10-02_contract_parent_reference_columns_rollback.sql`:

```sql
-- Removes exactly what 2026-10-02_contract_parent_reference_columns.sql adds.
-- Discards every extracted reference; they are recoverable only by re-extraction.
BEGIN;

ALTER TABLE proc.bp_contracts
    DROP COLUMN IF EXISTS parent_agreement_ref,
    DROP COLUMN IF EXISTS framework_ref;

ALTER TABLE proc.bp_contract_raw
    DROP COLUMN IF EXISTS parent_agreement_ref,
    DROP COLUMN IF EXISTS framework_ref;

COMMIT;
```

Apply it to `bp_testdb`, twice, confirming the second run succeeds unchanged.

- [ ] **Step 2: Write the failing tests**

Create `tests/services/extraction/test_contract_parent_reference_fields.py`:

```python
"""The references the vocabulary declares must be extractable.

proc.bp_document_type says a call-off points at its framework through
framework_ref. That field did not exist in extraction_schemas/contract.yaml, so
the pointer was declared and never filled.

Offline — the registry compiles from YAML with no database.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_contract_parent_reference_fields.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.pattern_extractor import run_pattern_extractor  # noqa: E402
from src.services.extraction.pattern_registry import get_registry           # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return get_registry("contract")


def _value(registry, text, field):
    """The value the L1 pattern layer reads for one field, or None."""
    results = run_pattern_extractor(text, registry)
    for r in results:
        name = getattr(r, "field", None) or (r.get("field") if isinstance(r, dict) else None)
        if name == field:
            return getattr(r, "value", None) or (r.get("value") if isinstance(r, dict) else None)
    return None


def test_both_fields_exist_and_are_optional(registry):
    names = {f.name for f in registry.schema.fields}
    assert {"framework_ref", "parent_agreement_ref"} <= names, sorted(names)
    for field in registry.schema.fields:
        if field.name in ("framework_ref", "parent_agreement_ref"):
            assert field.required is False, (
                f"{field.name} must be optional: most contracts name no parent, and "
                "a required field would block promotion on every standalone agreement"
            )
            assert field.db_column == field.name


def test_a_framework_reference_is_read(registry):
    text = "ORDER FORM\n\nThis Order Form is made under Framework Agreement No: FW-2024-0012.\n"
    assert _value(registry, text, "framework_ref") == "FW-2024-0012"


def test_the_made_under_phrasing_is_read(registry):
    text = "CALL-OFF CONTRACT\n\nmade under Framework Agreement RM6187.\n"
    assert _value(registry, text, "framework_ref") == "RM6187"


def test_a_parent_agreement_reference_is_read(registry):
    text = "STATEMENT OF WORK\n\nThis SOW is issued under Master Agreement MSA-4417.\n"
    assert _value(registry, text, "parent_agreement_ref") == "MSA-4417"


def test_a_placeholder_is_refused_not_stored(registry):
    """'N/A' is not a reference. Storing it would make the maths link on a word.

    Matches the rejection branch parent_contract_id already carries.
    """
    for placeholder in ("N/A", "NA", "TBC", "TBD", "NONE"):
        text = f"ORDER FORM\n\nFramework Agreement No: {placeholder}\n"
        assert _value(registry, text, "framework_ref") is None, placeholder


def test_a_sow_naming_its_msa_does_not_populate_parent_contract_id(registry):
    """Sitting under an agreement is not amending it.

    parent_contract_id anchors on amends/supplements/varies. If a plain "issued
    under" filled it too, the maths could not tell a child from an amendment.
    """
    text = "STATEMENT OF WORK\n\nThis SOW is issued under Master Agreement MSA-4417.\n"
    assert _value(registry, text, "parent_agreement_ref") == "MSA-4417"
    assert _value(registry, text, "parent_contract_id") is None


def test_an_amendment_still_populates_parent_contract_id(registry):
    """The existing behaviour must not move."""
    text = "VARIATION\n\nThis deed amends Contract MSA-4417.\n"
    assert _value(registry, text, "parent_contract_id") == "MSA-4417"


def test_a_contract_naming_no_parent_reads_neither_field(registry):
    text = "MASTER AGREEMENT\n\nbetween Acme Ltd and Beta Ltd, numbered clauses.\n"
    assert _value(registry, text, "framework_ref") is None
    assert _value(registry, text, "parent_agreement_ref") is None
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_contract_parent_reference_fields.py -v
```

Expected: `test_both_fields_exist_and_are_optional` FAILS on the missing names; the three value tests FAIL with `None`. `test_an_amendment_still_populates_parent_contract_id` and `test_a_contract_naming_no_parent_reads_neither_field` may already pass.

If `run_pattern_extractor`'s result shape does not match `_value`'s two accessors, read `src/services/extraction/pattern_extractor.py` and adjust `_value` only — never the assertions.

- [ ] **Step 4: Add both fields to the schema**

In `extraction_schemas/contract.yaml`, insert after the `parent_contract_id` block (which ends at `invariants: []`, ~line 385) and before `cost_centre_id`. The value regex is copied from `parent_contract_id` deliberately: one reference grammar for the file, and it already rejects the placeholders.

```yaml
  # The framework this document is called off under. proc.bp_document_type has
  # declared this field as doctype.call_off_contract's pointer at its framework
  # since the vocabulary was seeded; until now no field filled it.
  - name: framework_ref
    type: string
    required: false
    db_column: framework_ref
    canonical_labels:
      - "Framework Agreement"
      - "Framework Contract"
      - "Framework Reference"
      - "Framework No"
    patterns:
      - name: anchored_framework_ref
        anchor: '(?i)\bframework\s+(?:agreement|contract)\s*(?:number|no\.?|reference|ref\.?|#)?\s*[:\-]\s*'
        value: '\A(?!(?:N\/A|NA|TBC|TBD|NONE|SEE)\b)([A-Z][A-Z0-9\-\/\.]{1,29}[A-Z0-9]|\d{4,12}(?:[A-Z0-9\-\/\.]{0,19}[A-Z0-9])?)'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.90
      # "made under Framework Agreement RM6187" is prose and carries no colon,
      # so the connector stays optional and the value regex does the rejecting.
      - name: made_under_framework
        anchor: '(?i)\b(?:made|issued|called\s+off|placed)\s+under\s+(?:the\s+)?framework\s+(?:agreement|contract)\s*(?:number|no\.?|#)?\s*[:\-]?\s*'
        value: '\A(?!(?:N\/A|NA|TBC|TBD|NONE|SEE)\b)([A-Z][A-Z0-9\-\/\.]{1,29}[A-Z0-9]|\d{4,12}(?:[A-Z0-9\-\/\.]{0,19}[A-Z0-9])?)'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.84
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []

  # The agreement this document sits under WITHOUT amending it -- a SOW under its
  # MSA. Deliberately separate from parent_contract_id, which anchors on
  # amends/supplements/varies: "sits under" and "changes" are different facts,
  # and the hierarchy maths needs to tell a child from an amendment.
  - name: parent_agreement_ref
    type: string
    required: false
    db_column: parent_agreement_ref
    canonical_labels:
      - "Master Agreement"
      - "Master Services Agreement"
      - "Under Agreement"
      - "Pursuant To"
    patterns:
      - name: anchored_parent_agreement
        anchor: '(?i)\bmaster\s+(?:services?\s+)?agreement\s*(?:number|no\.?|reference|ref\.?|#)?\s*[:\-]\s*'
        value: '\A(?!(?:N\/A|NA|TBC|TBD|NONE|SEE)\b)([A-Z][A-Z0-9\-\/\.]{1,29}[A-Z0-9]|\d{4,12}(?:[A-Z0-9\-\/\.]{0,19}[A-Z0-9])?)'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.90
      - name: issued_under_agreement
        anchor: '(?i)\b(?:issued|made|entered\s+into|executed)\s+(?:under|pursuant\s+to)\s+(?:the\s+)?(?:master\s+(?:services?\s+)?)?agreement\s*(?:number|no\.?|#)?\s*[:\-]?\s*'
        value: '\A(?!(?:N\/A|NA|TBC|TBD|NONE|SEE)\b)([A-Z][A-Z0-9\-\/\.]{1,29}[A-Z0-9]|\d{4,12}(?:[A-Z0-9\-\/\.]{0,19}[A-Z0-9])?)'
        max_span_after_anchor_chars: 60
        prior_confidence: 0.84
    confidence_threshold: 0.75
    judge:
      tiebreaker: true
      grounded_last_resort: false
      ner_type_check: "none"
    invariants: []
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_contract_parent_reference_fields.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Run the schema and extraction suites**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/extraction/ -v
```

Two existing guards matter here and must still pass: anything asserting the contract schema's field count, and `completeness`/`missing_required` behaviour. Both new fields are optional, so neither may change.

- [ ] **Step 7: Prove the guards fail (required)**

1. **The placeholder rejection is load-bearing.** Delete `(?!(?:N\/A|NA|TBC|TBD|NONE|SEE)\b)` from `anchored_framework_ref`'s value regex. Expected: `test_a_placeholder_is_refused_not_stored` FAILS, storing `N/A` as a framework reference. Restore.
2. **The two facts stay separate.** Add an `issued under` anchor to `parent_contract_id`'s patterns. Expected: `test_a_sow_naming_its_msa_does_not_populate_parent_contract_id` FAILS. Restore.
3. **Optionality is load-bearing.** Set `required: true` on `framework_ref`. Expected: a `missing_required` discrepancy appears for a standalone master agreement in the extraction suite. Restore.

- [ ] **Step 8: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add deploy/sql/2026-10-02_contract_parent_reference_columns.sql \
        deploy/sql/2026-10-02_contract_parent_reference_columns_rollback.sql \
        extraction_schemas/contract.yaml \
        tests/services/extraction/test_contract_parent_reference_fields.py
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "feat(extraction): the framework reference the vocabulary always declared" "" \
  "Task 8 of specs/2026-10-02-contract-structures-plan.md." "" \
  "proc.bp_document_type has named framework_ref as the call-off's pointer at" \
  "its framework since the vocabulary was seeded, and no field filled it." "" \
  "parent_agreement_ref is separate from parent_contract_id on purpose: a SOW" \
  "names its MSA without amending it, and the hierarchy maths has to tell a" \
  "child from an amendment. Both optional, so a standalone agreement extracts" \
  "unchanged. Three guards proven red, including the N/A rejection." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- deploy/sql/2026-10-02_contract_parent_reference_columns.sql \
    deploy/sql/2026-10-02_contract_parent_reference_columns_rollback.sql \
    extraction_schemas/contract.yaml \
    tests/services/extraction/test_contract_parent_reference_fields.py
git diff --cached --name-status | wc -l
```

---
## Task 9: The contract-hierarchy scoring profile

**Files:**
- Create: `src/services/graph_resolution/profiles/contract_hierarchy.py`
- Modify: `src/services/graph_resolution/edge_writer.py:24-26` (`UNCALIBRATED_PROFILES`)
- Test: `tests/services/graph_resolution/test_contract_hierarchy.py`

**Interfaces:**
- Consumes: `resolved_doc_type` / `resolved_role` (Task 6), `framework_ref` / `parent_agreement_ref` (Task 8), `vocabulary.ensure_vocabulary` for each structure's `default_parent_type`.
- Produces:
  - `contract_hierarchy.PROFILE = "contract_hierarchy"`, `VERSION`, `SIGNALS`
  - `expected_parent_type(child_structure: str, *, vocabulary=None) -> str | None`
  - `observations_for(src: dict, tgt: dict) -> dict[str, frozenset]`
  - the profile registered under `linking_engine.PROFILES["contract_hierarchy"]`, scored through `linking_engine.score_link(child_row, parent_row, "contract_hierarchy")`

**Model this file on `src/services/graph_resolution/profiles/contract_succession.py`.** Same shape: private `_cmp_*` functions returning `(score, status)`, `register_signal` with a `csh_` prefix, a `SIGNALS` list carrying `reads`, `register_profile`, then `observations_for`. Do not invent a new structure.

**The direction is child → parent.** `score_link(source_row=child, target_row=parent, ...)`. Every comparator below assumes it.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/graph_resolution/test_contract_hierarchy.py`:

```python
"""Scoring a contract document against a candidate parent.

The one rule this file exists to hold: an exact reference match does NOT link.
The build spec's principle 3 says exact identifiers link automatically, and the
Discovery Report overturned it on measurement -- parent_contract_id is populated
on 1,561 contracts and resolves on 0, because the references were minted in
another namespace. An exact match after normalisation is the strongest signal
available; it is not a decision.

Offline — pure scoring, no database.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_contract_hierarchy.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services import linking_engine as le                              # noqa: E402
from src.services.graph_resolution.profiles import contract_hierarchy as ch  # noqa: E402


def _sow(**over):
    row = {
        "contract_id": "SOW-001",
        "resolved_doc_type": "doctype.sow",
        "resolved_role": "role.master",
        "parent_agreement_ref": "MSA-4417",
        "framework_ref": None,
        "parent_contract_id": None,
        "supplier_id": "S-100",
        "contract_title": "Statement of Work Data Migration",
        "contract_start_date": "2026-03-01",
        "contract_end_date": "2026-09-30",
    }
    row.update(over)
    return row


def _msa(**over):
    row = {
        "contract_id": "MSA-4417",
        "resolved_doc_type": "doctype.master_agreement",
        "resolved_role": "role.master",
        "supplier_id": "S-100",
        "contract_title": "Master Services Agreement Data Migration",
        "contract_start_date": "2026-01-01",
        "contract_end_date": "2027-12-31",
    }
    row.update(over)
    return row


def test_the_profile_is_registered():
    assert ch.PROFILE in le.PROFILES


def test_the_profile_can_never_auto_link():
    """The product's standing rule for an uncalibrated profile, and the reason
    is measured: no labelled sample of true parent links exists.
    """
    from src.services.graph_resolution.edge_writer import UNCALIBRATED_PROFILES
    assert ch.PROFILE in UNCALIBRATED_PROFILES


def test_the_expected_parent_of_a_sow_is_a_master_agreement():
    assert ch.expected_parent_type("doctype.sow") == "doctype.master_agreement"


def test_the_expected_parent_of_a_call_off_is_a_framework():
    assert ch.expected_parent_type("doctype.call_off_contract") == "doctype.framework_agreement"


def test_a_structure_with_no_declared_parent_expects_none():
    assert ch.expected_parent_type("doctype.master_agreement") is None


def test_a_full_match_scores_in_a_band_a_person_sees():
    r = le.score_link(_sow(), _msa(), ch.PROFILE)
    assert r["band"] in ("review", "auto_link_with_warning", "auto_link"), r
    assert r["F"] > 0


def test_an_exact_reference_alone_does_not_reach_the_auto_band():
    """THE rule. Everything else missing, the reference matching exactly.

    1,561 existing parent pointers resolve to 0 real contracts. A reference that
    matches is evidence; on its own it must still put a person in the loop.
    """
    bare_child = _sow(supplier_id=None, contract_title=None,
                      contract_start_date=None, contract_end_date=None)
    bare_parent = _msa(supplier_id=None, contract_title=None,
                       contract_start_date=None, contract_end_date=None)
    r = le.score_link(bare_child, bare_parent, ch.PROFILE)
    assert r["band"] != "auto_link", r
    assert r["F"] < 92, r


def test_the_wrong_structure_of_parent_conflicts():
    """A SOW's parent is a master agreement, not an invoice."""
    r = le.score_link(_sow(), _msa(resolved_doc_type="doctype.invoice"), ch.PROFILE)
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["expected_structure"] == "CONFLICT", detail


def test_a_different_supplier_conflicts():
    r = le.score_link(_sow(), _msa(supplier_id="S-999"), ch.PROFILE)
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["supplier"] == "CONFLICT", detail


def test_a_child_outside_the_parents_term_conflicts():
    r = le.score_link(
        _sow(contract_start_date="2029-01-01", contract_end_date="2029-06-30"),
        _msa(), ch.PROFILE,
    )
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["term_containment"] == "CONFLICT", detail


def test_a_child_inside_the_parents_term_is_ok():
    r = le.score_link(_sow(), _msa(), ch.PROFILE)
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["term_containment"] == "OK", detail


def test_a_missing_field_is_missing_not_a_conflict():
    """MISSING and CONFLICT are different answers.

    Treating absence as contradiction would make every sparsely-filled contract
    look like a wrong parent, which is how a review queue fills with noise.
    """
    r = le.score_link(_sow(contract_start_date=None, contract_end_date=None),
                      _msa(), ch.PROFILE)
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["term_containment"] == "MISSING", detail


def test_the_reference_is_compared_after_normalisation():
    """'msa 4417' and 'MSA-4417' are the same reference printed differently."""
    r = le.score_link(_sow(parent_agreement_ref=" msa 4417 "), _msa(), ch.PROFILE)
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["declared_reference"] == "OK", detail


def test_any_of_the_three_reference_fields_can_carry_the_pointer():
    for field in ("parent_agreement_ref", "framework_ref", "parent_contract_id"):
        child = _sow(parent_agreement_ref=None, framework_ref=None,
                     parent_contract_id=None, **{field: "MSA-4417"})
        r = le.score_link(child, _msa(), ch.PROFILE)
        detail = {d["id"]: d["status"] for d in r["signals"]}
        assert detail["declared_reference"] == "OK", (field, detail)


def test_a_reference_naming_a_different_contract_conflicts():
    r = le.score_link(_sow(parent_agreement_ref="MSA-9999"), _msa(), ch.PROFILE)
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["declared_reference"] == "CONFLICT", detail


def test_a_shared_generic_word_is_not_title_evidence():
    """Every contract shares 'Agreement' and 'Services' with every other one.

    Without the stopword set this signal would score the vocabulary rather than
    the documents, and two unrelated contracts from one supplier would look like
    a parent and child.
    """
    r = le.score_link(
        _sow(contract_title="Services Agreement"),
        _msa(contract_title="Services Agreement"),
        ch.PROFILE,
    )
    detail = {d["id"]: d["status"] for d in r["signals"]}
    assert detail["title_overlap"] != "OK", detail


def test_observations_are_reported_for_every_signal():
    """composition.remap_clusters needs one observation set per signal id."""
    obs = ch.observations_for(_sow(), _msa())
    assert set(obs) == {s["id"] for s in ch.SIGNALS}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/graph_resolution/test_contract_hierarchy.py -v
```

Expected: collection FAILS with `ImportError: cannot import name 'contract_hierarchy'`.

- [ ] **Step 3: Write the profile**

Create `src/services/graph_resolution/profiles/contract_hierarchy.py`:

```python
"""Which contract does this document sit under?

Not succession (which contract REPLACED which -- contract_succession.py) and not
coverage (what spend sits under a contract -- contract_coverage.py). This is the
parent-child hierarchy the GPSS build spec is about: a SOW under its master
agreement, a call-off or order form under its framework, a variation against the
contract it changes.

THE ONE RULE WORTH READING BEFORE CHANGING ANYTHING HERE. The build spec's
principle 3 says exact identifiers link automatically. The Discovery Report
overturned it on measurement and this profile is built on the overturned version:
proc.bp_contract_master.parent_contract_id is populated on 1,561 contracts and
resolves to a real contract on ZERO of them, because the references were minted
in a different namespace (C00002 pointing at C1543, which does not exist). So a
reference that matches exactly is the strongest signal available and is still
only a signal. Scoring it heavily enough to auto-link would link 1,561 contracts
to nothing.

Direction is CHILD -> PARENT: score_link(source_row=child, target_row=parent).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from src.services import linking_engine as _le
from src.services.concepts.vocabulary import Vocabulary, ensure_vocabulary
from ..observations import Observation

PROFILE = "contract_hierarchy"
VERSION = "1.0.0"

#: The reference fields a child may carry its parent's identifier in, strongest
#: first. framework_ref and parent_agreement_ref are declared pointers (Task 8);
#: parent_contract_id is the amends/supplements case and is read last because a
#: document that amends its parent also sits under it.
_REFERENCE_FIELDS = ("framework_ref", "parent_agreement_ref", "parent_contract_id")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PLACEHOLDERS = {"", "na", "n a", "tbc", "tbd", "none", "see"}


def _norm_ref(value) -> str:
    """One normalisation for every reference comparison.

    'MSA-4417', ' msa 4417 ' and 'MSA/4417' are the same reference printed
    differently, and a comparison that called them different would discard the
    heaviest signal on a formatting difference.
    """
    return _NON_ALNUM.sub("", str(value or "").strip().lower())


def _to_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def expected_parent_type(
    child_structure: Optional[str],
    *,
    vocabulary: Optional[Vocabulary] = None,
) -> Optional[str]:
    """The structure the vocabulary says this one sits under, or None.

    Read from proc.bp_document_type.default_parent_type rather than hard-coded
    here, so the hierarchy is the same single fact the classifier and the upload
    gate already read. A fifth copy would need a fifth drift test.
    """
    if not child_structure:
        return None
    vocab = vocabulary if vocabulary is not None else ensure_vocabulary()
    dt = vocab.document_types.get(child_structure)
    return dt.default_parent_type if dt else None


def _cmp_reference(src, tgt) -> tuple[float, str]:
    """Does the child name this parent's identifier?

    MISSING when the child names no parent at all -- absence is not a
    contradiction, and treating it as one would make every standalone document
    look like a wrong parent.
    """
    parent_id = _norm_ref(tgt.get("contract_id"))
    if not parent_id:
        return 0.5, "MISSING"
    claimed = [
        _norm_ref(src.get(f)) for f in _REFERENCE_FIELDS
        if _norm_ref(src.get(f)) not in _PLACEHOLDERS
    ]
    if not claimed:
        return 0.5, "MISSING"
    if parent_id in claimed:
        return 1.0, "OK"
    return 0.0, "CONFLICT"


def _cmp_expected_structure(src, tgt) -> tuple[float, str]:
    """Is this candidate the KIND of document the child's structure sits under?"""
    want = expected_parent_type(src.get("resolved_doc_type"))
    have = tgt.get("resolved_doc_type")
    if not want or not have:
        return 0.5, "MISSING"
    return (1.0, "OK") if want == have else (0.0, "CONFLICT")


def _cmp_supplier(src, tgt) -> tuple[float, str]:
    a, b = src.get("supplier_id"), tgt.get("supplier_id")
    if not a or not b:
        return 0.5, "MISSING"
    return (1.0, "OK") if str(a) == str(b) else (0.0, "CONFLICT")


def _cmp_term_containment(src, tgt) -> tuple[float, str]:
    """Does the child's term sit inside the parent's?

    A child that starts before its parent or ends after it is not impossible --
    signature lag and extensions both happen -- so this is a graded signal, not a
    gate. Only a child wholly outside the parent's term CONFLICTs.
    """
    cs, ce = _to_date(src.get("contract_start_date")), _to_date(src.get("contract_end_date"))
    ps, pe = _to_date(tgt.get("contract_start_date")), _to_date(tgt.get("contract_end_date"))
    if cs is None or ps is None:
        return 0.5, "MISSING"
    if pe is not None and cs > pe:
        return 0.0, "CONFLICT"       # starts after the parent ended
    if ce is not None and ce < ps:
        return 0.0, "CONFLICT"       # ended before the parent began
    inside_start = cs >= ps
    inside_end = pe is None or ce is None or ce <= pe
    if inside_start and inside_end:
        return 1.0, "OK"
    return 0.6, "WEAK"


_STOPWORDS = {"the", "and", "of", "for", "agreement", "contract", "services",
              "service", "statement", "work", "master", "framework", "order",
              "form", "schedule", "ltd", "limited", "plc"}


def _cmp_title_overlap(src, tgt) -> tuple[float, str]:
    """Shared DISTINCTIVE words, so 'Agreement' is not evidence.

    Without the stopword set every contract shares 'Agreement' and 'Services'
    with every other, and the signal would score the vocabulary rather than the
    documents.
    """
    def words(row):
        raw = (row.get("contract_title") or "").lower()
        return {w for w in _NON_ALNUM.sub(" ", raw).split() if w and w not in _STOPWORDS}

    a, b = words(src), words(tgt)
    if not a or not b:
        return 0.5, "MISSING"
    j = len(a & b) / len(a | b)
    if j >= 0.5:
        return 1.0, "OK"
    if j >= 0.2:
        return 0.6, "WEAK"
    return 0.0, "CONFLICT"


_le.register_signal("csh_reference", lambda s, t, sl, tl: _cmp_reference(s, t))
_le.register_signal("csh_structure", lambda s, t, sl, tl: _cmp_expected_structure(s, t))
_le.register_signal("csh_supplier", lambda s, t, sl, tl: _cmp_supplier(s, t))
_le.register_signal("csh_term", lambda s, t, sl, tl: _cmp_term_containment(s, t))
_le.register_signal("csh_title", lambda s, t, sl, tl: _cmp_title_overlap(s, t))

SIGNALS = [
    # The reference is tier 1 and weight 5 -- the heaviest available -- and its
    # conflict_cap of 0.45 is what stops it deciding alone. See the module
    # docstring: 1,561 references resolve to nothing.
    {"id": "declared_reference", "cluster": "reference", "tier": 1, "weight": 5,
     "appl": 1.0, "cap": 0.45, "kind": "csh_reference",
     "reads": ["framework_ref", "parent_agreement_ref", "parent_contract_id"]},
    {"id": "expected_structure", "cluster": "structure", "tier": 1, "weight": 5,
     "appl": 1.0, "cap": 0.45, "kind": "csh_structure",
     "reads": ["resolved_doc_type"]},
    {"id": "supplier", "cluster": "identity", "tier": 1, "weight": 5,
     "appl": 1.0, "cap": 0.45, "kind": "csh_supplier",
     "reads": ["supplier_id"]},
    {"id": "term_containment", "cluster": "temporal", "tier": 2, "weight": 3,
     "appl": 1.0, "cap": 0.70, "kind": "csh_term",
     "reads": ["contract_start_date", "contract_end_date"]},
    {"id": "title_overlap", "cluster": "description", "tier": 2, "weight": 3,
     "appl": 1.0, "cap": 0.70, "kind": "csh_title",
     "reads": ["contract_title"]},
]

# DECLARED UNMEASURED, like contract_succession and contract_coverage: no
# labelled sample of true parent links exists, because the only parent pointers
# in the corpus all dangle.
_le.register_profile(PROFILE, {
    "p0": 0.02, "alpha": 0.35, "floor": 0.55,
    "signals": SIGNALS, "date_field": "contract_start_date",
})


def observations_for(src: dict, tgt: dict) -> dict[str, frozenset]:
    sid, tid = str(src.get("contract_id")), str(tgt.get("contract_id"))
    out: dict[str, frozenset] = {}
    for spec in SIGNALS:
        obs: set[Observation] = set()
        for field in spec["reads"]:
            if src.get(field) is not None:
                obs.add((sid, field))
            if tgt.get(field) is not None:
                obs.add((tid, field))
        out[spec["id"]] = frozenset(obs)
    return out
```

- [ ] **Step 4: Add it to the never-auto-link set**

In `src/services/graph_resolution/edge_writer.py`:

```python
UNCALIBRATED_PROFILES = frozenset({
    "contract_coverage", "contract_succession", "contract_hierarchy",
    "supplier_identity",
    # ... keep every existing member exactly as it is
})
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/graph_resolution/test_contract_hierarchy.py -v
```

Expected: 17 passed.

**If `test_an_exact_reference_alone_does_not_reach_the_auto_band` fails**, the weights are wrong and the fix is the weights, never the assertion. Record the measured `F` for a reference-only match in the module docstring so the next reader knows the headroom.

- [ ] **Step 6: Run the graph-resolution suite**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/graph_resolution/ -v
```

`test_pass_order.py` and the `assert_dag_safe` guard both read `PASS_ORDER`. This profile is **not** added to `PASS_ORDER` — it does not write Neo4j edges and is not part of that pipeline's dependency chain. If a test requires every registered profile to appear in `PASS_ORDER`, stop and report it rather than adding the profile to a pipeline it does not belong to.

- [ ] **Step 7: Prove the guards fail (required)**

1. **The reference cannot decide alone.** Raise `declared_reference`'s `cap` to `0.99` and its `weight` to `20`. Expected: `test_an_exact_reference_alone_does_not_reach_the_auto_band` FAILS. Restore. This is the single most important proof in the task.
2. **It can never auto-link.** Remove `"contract_hierarchy"` from `UNCALIBRATED_PROFILES`. Expected: `test_the_profile_can_never_auto_link` FAILS. Restore.
3. **MISSING is not CONFLICT.** In `_cmp_term_containment`, change the `cs is None or ps is None` branch to `return 0.0, "CONFLICT"`. Expected: `test_a_missing_field_is_missing_not_a_conflict` FAILS. Restore.
4. **Normalisation is load-bearing.** Make `_norm_ref` return `str(value or "")`. Expected: `test_the_reference_is_compared_after_normalisation` FAILS. Restore.
5. **The hierarchy is read from the vocabulary.** Make `expected_parent_type` return `None` unconditionally. Expected: the two `expected_parent_type` tests and `test_the_wrong_structure_of_parent_conflicts` FAIL. Restore.
6. **Stopwords are load-bearing.** Empty `_STOPWORDS`. Expected: `test_a_shared_generic_word_is_not_title_evidence` FAILS — two contracts both titled "Services Agreement" now score `title_overlap` as `OK`, so the signal is scoring the vocabulary rather than the documents. Restore.

- [ ] **Step 8: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add src/services/graph_resolution/profiles/contract_hierarchy.py \
        src/services/graph_resolution/edge_writer.py \
        tests/services/graph_resolution/test_contract_hierarchy.py
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "feat(linking): score a contract document against its candidate parent" "" \
  "Task 9 of specs/2026-10-02-contract-structures-plan.md." "" \
  "Five signals: the declared reference, whether the candidate is the KIND of" \
  "document the child's structure sits under, supplier, term containment and" \
  "distinctive title overlap. The hierarchy is read from" \
  "bp_document_type.default_parent_type, not hard-coded here." "" \
  "An exact reference match does NOT auto-link, against the build spec's" \
  "principle 3 and with the Discovery Report's measurement behind it: 1,561" \
  "parent pointers resolve to 0 real contracts. Proven by raising the weight" \
  "and watching the band reach auto_link. Six guards proven red." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- src/services/graph_resolution/profiles/contract_hierarchy.py \
    src/services/graph_resolution/edge_writer.py \
    tests/services/graph_resolution/test_contract_hierarchy.py
git diff --cached --name-status | wc -l
```

---
## Task 10: The maths runs, and proposes

A scored profile that nothing calls is the exact failure this layer must not repeat: `contract_succession.py` has had a full scoring function and unit tests since it was written, and `pass_runner.run_all()` has never called it.

**Files:**
- Create: `src/services/contract_links.py`
- Test: `tests/services/test_contract_links.py`

**Interfaces:**
- Consumes: `contract_hierarchy.PROFILE`, `linking_engine.score_link`, `contract_type_map.structure_for_contract_type`, `persistence.normalise_source_file`.
- Produces:
  - `candidate_parents(cur, child: dict) -> list[dict]`
  - `propose_parent_links(limit: int | None = None) -> dict` → `{"proposed": int, "contested": int, "no_candidate": int, "considered": dict, "details": list}`
  - `confirm(contract_id: str, parent_contract_id: str, source_file: str, reviewer: str | None = None) -> bool`

**Where a proposal lands, and why not a new table.** `proc.bp_extraction_discrepancy` — the Action Centre findings surface buyers already work. `deal_link_proposals.py:137-155` already writes parent proposals there with `issue_type='deal_link_proposed'`, `severity='info'`, `blocks_promotion=false`. A new table would be a second queue nobody opens.

**The idempotency key is the open-row key, and it includes `source_file`.** `(doc_type, doc_pk_candidate, coalesce(source_file,''), issue_type, field_name)` is enforced by the partial unique index `ix_bp_extraction_discrepancy_open_key`. Three consequences the implementer must hold:
- `source_file` must go through `persistence.normalise_source_file()` on write, and the idempotency `SELECT` must normalise the same way. A key written one spelling and read another re-proposes on every tick.
- `doc_pk_candidate` is a `contract_id` read **out of** a document, so two documents can carry the same one. `source_file` is what separates them — that collision is what cost 65 days of findings on `bp_sqldb`.
- **One open proposal per document.** Two candidate parents for the same child would produce the same five key columns and the second write would be rejected by the index. So the best candidate goes in `expected_value` and the runners-up in `computed_value`, with `routing` saying whether the evidence actually separated them.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_contract_links.py`:

```python
"""Proposing a contract document's parent, and never linking it.

The failure this file exists to prevent is not a wrong score. It is
contract_succession.py: a scored, unit-tested profile that nothing ever calls.
test_the_runner_is_actually_called is therefore the load-bearing test here.

Live-only. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/test_contract_links.py -v
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services import contract_links as CL                       # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

ISSUE = "contract_parent_proposed"


@pytest.fixture()
def fixture_contracts():
    """A master agreement and a SOW naming it, removed again afterwards."""
    tag = uuid.uuid4().hex[:8].upper()
    msa = f"MSA-{tag}"
    sow = f"SOW-{tag}"
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proc.bp_contracts
                   (contract_id, contract_title, supplier_id, contract_start_date,
                    contract_end_date, resolved_doc_type, resolved_role, type_agreement)
               VALUES (%s, 'Master Services Agreement Helix Migration', %s,
                       '2026-01-01', '2027-12-31',
                       'doctype.master_agreement', 'role.master', 'refined')""",
            (msa, f"S-{tag}"),
        )
        cur.execute(
            """INSERT INTO proc.bp_contracts
                   (contract_id, contract_title, supplier_id, contract_start_date,
                    contract_end_date, resolved_doc_type, resolved_role, type_agreement,
                    parent_agreement_ref)
               VALUES (%s, 'Statement of Work Helix Migration', %s,
                       '2026-03-01', '2026-09-30',
                       'doctype.sow', 'role.master', 'refined', %s)""",
            (sow, f"S-{tag}", msa),
        )
    yield {"msa": msa, "sow": sow, "supplier": f"S-{tag}", "tag": tag}
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate = ANY(%s)",
                    ([msa, sow],))
        cur.execute("DELETE FROM proc.bp_contracts WHERE contract_id = ANY(%s)", ([msa, sow],))


def _open_proposals(doc_pk):
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT expected_value, computed_value, severity, blocks_promotion,
                      source_file, field_name, notes
                 FROM proc.bp_extraction_discrepancy
                WHERE doc_pk_candidate = %s AND issue_type = %s AND status = 'open'""",
            (doc_pk, ISSUE),
        )
        cols = ("expected_value", "computed_value", "severity", "blocks_promotion",
                "source_file", "field_name", "notes")
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def test_the_runner_is_actually_called(fixture_contracts):
    """THE test. contract_succession has a scorer and no runner; this must not.

    A profile nothing calls is indistinguishable from a profile that found
    nothing, and both report zero.
    """
    result = CL.propose_parent_links()
    assert result["considered"]["children"] > 0, (
        "the runner looked at no children at all, so it cannot have scored anything"
    )
    assert result["proposed"] + result["contested"] >= 1, result


def test_the_sow_is_proposed_under_its_master_agreement(fixture_contracts):
    CL.propose_parent_links()
    rows = _open_proposals(fixture_contracts["sow"])
    assert len(rows) == 1, rows
    assert rows[0]["expected_value"] == fixture_contracts["msa"]


def test_a_proposal_never_blocks_promotion_and_is_informational(fixture_contracts):
    CL.propose_parent_links()
    row = _open_proposals(fixture_contracts["sow"])[0]
    assert row["blocks_promotion"] is False
    assert row["severity"] == "info"
    assert row["field_name"] == "parent_contract_id"


def test_nothing_is_linked_without_a_person(fixture_contracts):
    """A proposal is a proposal. parent_contract_id stays untouched."""
    CL.propose_parent_links()
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT parent_contract_id FROM proc.bp_contracts WHERE contract_id = %s",
                    (fixture_contracts["sow"],))
        assert cur.fetchone()[0] is None


def test_running_twice_does_not_stack_proposals(fixture_contracts):
    CL.propose_parent_links()
    CL.propose_parent_links()
    CL.propose_parent_links()
    assert len(_open_proposals(fixture_contracts["sow"])) == 1


def test_two_documents_sharing_a_contract_id_each_keep_their_proposal(fixture_contracts):
    """Review Focus 4. doc_pk_candidate is a value read OUT of a document, so two
    documents can carry the same one; source_file is what separates them. Exactly
    the collision that cost 65 days of findings on bp_sqldb.
    """
    from src.services.db import get_conn
    sow = fixture_contracts["sow"]
    CL.propose_parent_links()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proc.bp_extraction_discrepancy
                   (doc_type, source_file, doc_pk_candidate, field_name, issue_type,
                    severity, expected_value, blocks_promotion, notes, status)
               VALUES ('contract', %s, %s, 'parent_contract_id', %s, 'info', %s,
                       false, 'a second document carrying the same contract_id', 'open')""",
            (f"documents/contract/other-{fixture_contracts['tag']}.pdf", sow, ISSUE,
             fixture_contracts["msa"]),
        )
    rows = _open_proposals(sow)
    assert len(rows) == 2, "the second document's proposal was rejected or merged"
    assert len({r["source_file"] for r in rows}) == 2


def test_the_source_file_is_stored_normalised(fixture_contracts):
    from src.services.extraction.persistence import normalise_source_file
    CL.propose_parent_links()
    row = _open_proposals(fixture_contracts["sow"])[0]
    assert row["source_file"] == normalise_source_file(row["source_file"])
    assert "/" in row["source_file"] or row["source_file"].startswith("contract:"), (
        "never reduce source_file to a basename"
    )


def test_nothing_proposed_is_distinguishable_from_everything_parented():
    """Review Focus 5. An empty screen must not read as success.

    'proposals: 0' invites the reading that every contract has a parent. The
    considered counts are what let a screen tell that apart from 'no contract
    resembled a parent its supplier holds'.
    """
    result = CL.propose_parent_links()
    assert set(result["considered"]) >= {"children", "with_structure", "with_candidates"}
    assert all(isinstance(v, int) for v in result["considered"].values())


def test_a_child_with_no_candidate_parent_proposes_nothing(fixture_contracts):
    from src.services.db import get_conn
    orphan = f"SOW-ORPH-{fixture_contracts['tag']}"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proc.bp_contracts
                   (contract_id, contract_title, supplier_id, resolved_doc_type,
                    resolved_role, type_agreement)
               VALUES (%s, 'Statement of Work Nothing Above It', 'S-NOBODY',
                       'doctype.sow', 'role.master', 'refined')""",
            (orphan,),
        )
    try:
        result = CL.propose_parent_links()
        assert _open_proposals(orphan) == []
        assert result["no_candidate"] >= 1
    finally:
        with get_conn() as conn:
            conn.cursor().execute("DELETE FROM proc.bp_contracts WHERE contract_id = %s",
                                  (orphan,))


def test_a_structure_with_no_declared_parent_is_not_a_child(fixture_contracts):
    """A master agreement sits under nothing, so it is never proposed a parent."""
    CL.propose_parent_links()
    assert _open_proposals(fixture_contracts["msa"]) == []


def test_the_existing_corpus_supplies_candidate_parents():
    """Task 7 pays off here: 3,016 bp_contract_master rows read as structures.

    Without them the candidate set would be empty until enough contracts had
    been uploaded to form a hierarchy.
    """
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        child = {"contract_id": "X", "resolved_doc_type": "doctype.sow",
                 "supplier_id": None, "contract_start_date": None}
        cur.execute("SELECT supplier_id FROM proc.bp_contract_master "
                    "WHERE contract_type = 'Master Agreement' LIMIT 1")
        row = cur.fetchone()
        assert row, "the corpus has no Master Agreement to be a candidate parent"
        child["supplier_id"] = row[0]
        candidates = CL.candidate_parents(cur, child)
    assert candidates, "no candidate parent came from the 3,051-row corpus"
    assert all(c["resolved_doc_type"] == "doctype.master_agreement" for c in candidates)


def test_confirming_sets_the_parent_and_closes_the_proposal(fixture_contracts):
    CL.propose_parent_links()
    assert CL.confirm(
        fixture_contracts["sow"], fixture_contracts["msa"],
        _open_proposals(fixture_contracts["sow"])[0]["source_file"],
        reviewer="test",
    ) is True
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT parent_contract_id FROM proc.bp_contracts WHERE contract_id = %s",
                    (fixture_contracts["sow"],))
        assert cur.fetchone()[0] == fixture_contracts["msa"]
    assert _open_proposals(fixture_contracts["sow"]) == []


def test_the_notes_say_why_in_words_a_person_can_check(fixture_contracts):
    CL.propose_parent_links()
    notes = _open_proposals(fixture_contracts["sow"])[0]["notes"]
    assert fixture_contracts["msa"] in notes
    assert "confirm" in notes.lower()
    for word in ("reference", "supplier", "term"):
        assert word in notes.lower(), f"the reason omits {word}: {notes}"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/test_contract_links.py -v
```

Expected: collection FAILS with `ModuleNotFoundError: No module named 'src.services.contract_links'`.

- [ ] **Step 3: Write the module**

Create `src/services/contract_links.py`. Model the proposal write on `deal_link_proposals.py:117-160` — the same `issue_type` / `severity='info'` / `blocks_promotion=false` shape, the same idempotency `SELECT` before the `INSERT`.

```python
"""Propose which contract a contract document sits under. Never link it.

The failure this module exists not to repeat: contract_succession.py has a full
scoring function and unit tests, and pass_runner.run_all() has never called it.
A profile nothing calls is indistinguishable from a profile that found nothing.

WHY A PROPOSAL AND NOT A LINK. proc.bp_contract_master.parent_contract_id is
populated on 1,561 contracts and resolves to a real contract on ZERO of them.
Writing a parent on a score would be writing the same kind of value that is
already wrong 1,561 times over. A person confirms, through the queue they
already work.

WHERE IT LANDS. proc.bp_extraction_discrepancy, the Action Centre findings
surface -- the same place deal_link_proposals.py puts its parent proposals. A new
table would be a second queue nobody opens.

THE KEY. The open-row key is
(doc_type, doc_pk_candidate, coalesce(source_file,''), issue_type, field_name),
enforced by the partial unique index ix_bp_extraction_discrepancy_open_key.
Three things follow, and all three have bitten this product:
  * source_file goes through normalise_source_file on write AND in the
    idempotency SELECT. One spelling written and another read re-proposes on
    every scheduler tick.
  * doc_pk_candidate is a contract_id read OUT of a document, so two documents
    can carry the same one. source_file separates them. That collision cost 65
    days of findings on bp_sqldb.
  * ONE open proposal per document. Two candidates for one child produce the
    same five key columns and the index rejects the second, so the best
    candidate goes in expected_value and the rest in computed_value.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.services import linking_engine as _le
from src.services.concepts.contract_type_map import structure_for_contract_type
from src.services.db import get_conn
from src.services.extraction.persistence import normalise_source_file
from src.services.graph_resolution.profiles import contract_hierarchy as _ch

log = logging.getLogger(__name__)

ISSUE_TYPE = "contract_parent_proposed"
FIELD_NAME = "parent_contract_id"

#: Below this the evidence is too thin to be worth a person's attention. The
#: linking engine's own review band -- not a number invented here.
MIN_SCORE = 65.0

#: How far apart the best and second-best must be for the proposal to read as
#: "confirm this" rather than "choose between these".
SEPARATION = 8.0

_CHILD_SQL = """
    SELECT contract_id, contract_title, supplier_id, resolved_doc_type,
           resolved_role, framework_ref, parent_agreement_ref, parent_contract_id,
           contract_start_date, contract_end_date, total_contract_value, currency
      FROM proc.bp_contracts
     WHERE resolved_doc_type IS NOT NULL
       AND parent_contract_id IS NULL
"""


def candidate_parents(cur, child: dict) -> list[dict]:
    """Contracts that could be this child's parent.

    Two sources, because one alone would be empty for a long time:
      * proc.bp_contracts -- uploaded documents with a recognised structure;
      * proc.bp_contract_master -- the 3,051-row corpus, whose free-text
        contract_type reads as a structure through contract_type_map (3,016 of
        them do).

    Narrowed by supplier in SQL rather than in Python: without it this is 3,051
    score_link calls per child, and the deal-assignment service has already
    taught this product what an unnarrowed per-document query costs.
    """
    want = _ch.expected_parent_type(child.get("resolved_doc_type"))
    if not want:
        return []
    supplier = child.get("supplier_id")
    if not supplier:
        return []

    out: list[dict] = []
    cur.execute(
        """SELECT contract_id, contract_title, supplier_id, resolved_doc_type,
                  contract_start_date, contract_end_date
             FROM proc.bp_contracts
            WHERE supplier_id = %s AND resolved_doc_type = %s
              AND contract_id <> %s""",
        (supplier, want, child.get("contract_id")),
    )
    for r in cur.fetchall():
        out.append(dict(zip(
            ("contract_id", "contract_title", "supplier_id", "resolved_doc_type",
             "contract_start_date", "contract_end_date"), r)))

    cur.execute(
        """SELECT contract_id, contract_title, supplier_id, contract_type,
                  contract_start_date, contract_end_date
             FROM proc.bp_contract_master
            WHERE supplier_id = %s AND contract_id <> %s""",
        (supplier, child.get("contract_id")),
    )
    for r in cur.fetchall():
        row = dict(zip(
            ("contract_id", "contract_title", "supplier_id", "contract_type",
             "contract_start_date", "contract_end_date"), r))
        # The corpus's free-text type, read as a structure. Not written back:
        # contract_type is source data.
        row["resolved_doc_type"] = structure_for_contract_type(row.pop("contract_type"))
        if row["resolved_doc_type"] == want:
            out.append(row)
    return out


def _source_file_for(cur, contract_id: str) -> str:
    """The document this contract came from, normalised.

    Falls back to 'contract:<id>' for a corpus row that never arrived as an
    upload -- a stable key, and never a basename.
    """
    cur.execute(
        "SELECT source_file FROM proc.bp_contract_raw WHERE contract_id = %s "
        "ORDER BY raw_id DESC LIMIT 1",
        (contract_id,),
    )
    row = cur.fetchone()
    return normalise_source_file(row[0]) if row and row[0] else f"contract:{contract_id}"


def propose_parent_links(limit: Optional[int] = None) -> dict:
    """Score every parentless contract document and propose its best parent.

    The ``considered`` counts are not decoration. 'proposed: 0' reads as "every
    contract has a parent" when the truth may be "no contract resembled a parent
    its supplier holds", and those are different problems with different fixes.
    """
    proposed = contested = no_candidate = 0
    considered = {"children": 0, "with_structure": 0, "with_candidates": 0}
    details: list[dict] = []

    # One connection for the pass, exactly as deal_link_proposals.propose does.
    # get_conn() is AUTOCOMMIT, so each INSERT lands on execute and there is no
    # transaction to commit or roll back.
    with get_conn() as conn:
        cur = conn.cursor()
        sql = _CHILD_SQL + (" LIMIT %s" if limit else "")
        cur.execute(sql, (limit,) if limit else ())
        cols = [d[0] for d in cur.description]
        children = [dict(zip(cols, r)) for r in cur.fetchall()]

        for child in children:
            considered["children"] += 1
            if not _ch.expected_parent_type(child.get("resolved_doc_type")):
                continue              # sits under nothing: not a child at all
            considered["with_structure"] += 1

            candidates = candidate_parents(cur, child)
            if not candidates:
                no_candidate += 1
                continue
            considered["with_candidates"] += 1

            scored = sorted(
                ((_le.score_link(child, parent, _ch.PROFILE), parent)
                 for parent in candidates),
                key=lambda pair: -pair[0]["F"],
            )
            best, best_parent = scored[0]
            if best["F"] < MIN_SCORE:
                no_candidate += 1
                continue

            runner_up = scored[1][0]["F"] if len(scored) > 1 else None
            separated = runner_up is None or (best["F"] - runner_up) >= SEPARATION
            routing = "suggested" if separated else "contested"
            alternatives = [p["contract_id"] for _s, p in scored[1:4]]

            source_file = _source_file_for(cur, child["contract_id"])

            # Idempotent: the same five key columns the unique index enforces, so
            # the SELECT and the index agree. A scheduler tick must not stack.
            cur.execute(
                """SELECT 1 FROM proc.bp_extraction_discrepancy
                    WHERE doc_type = 'contract' AND doc_pk_candidate = %s
                      AND coalesce(source_file,'') = %s AND issue_type = %s
                      AND field_name = %s AND status = 'open' LIMIT 1""",
                (child["contract_id"], source_file, ISSUE_TYPE, FIELD_NAME),
            )
            if cur.fetchone():
                continue

            why = {d["id"]: d["status"] for d in best["signals"]}
            cur.execute(
                """INSERT INTO proc.bp_extraction_discrepancy
                       (doc_type, source_file, doc_pk_candidate, field_name,
                        issue_type, severity, raw_value, expected_value,
                        computed_value, blocks_promotion, notes)
                   VALUES ('contract', %s, %s, %s, %s, 'info', NULL, %s, %s,
                           false, %s)""",
                (
                    source_file, child["contract_id"], FIELD_NAME, ISSUE_TYPE,
                    best_parent["contract_id"],
                    ", ".join(alternatives) or None,
                    (
                        f"this {child['resolved_doc_type'].split('.')[-1]} appears to "
                        f"sit under contract {best_parent['contract_id']} "
                        f"(score {best['F']:.1f}, {routing}). "
                        f"reference: {why.get('declared_reference')}; "
                        f"structure: {why.get('expected_structure')}; "
                        f"supplier: {why.get('supplier')}; "
                        f"term: {why.get('term_containment')}; "
                        f"title: {why.get('title_overlap')}. "
                        + (f"Other candidates: {', '.join(alternatives)}. "
                           if alternatives else "")
                        + f"Nothing has been linked. Confirm to set "
                          f"parent_contract_id = {best_parent['contract_id']}."
                    ),
                ),
            )
            proposed += 1 if routing == "suggested" else 0
            contested += 1 if routing == "contested" else 0
            details.append({"contract_id": child["contract_id"],
                            "parent": best_parent["contract_id"],
                            "F": best["F"], "routing": routing})

    result = {"proposed": proposed, "contested": contested,
              "no_candidate": no_candidate, "considered": considered,
              "details": details}
    log.info("contract parent proposals: %s",
             {k: v for k, v in result.items() if k != "details"})
    return result


def confirm(contract_id: str, parent_contract_id: str, source_file: str,
            reviewer: Optional[str] = None) -> bool:
    """A person accepted the proposal: set the parent and close the finding."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE proc.bp_contracts SET parent_contract_id = %s WHERE contract_id = %s",
            (parent_contract_id, contract_id),
        )
        if cur.rowcount == 0:
            return False
        cur.execute(
            """UPDATE proc.bp_extraction_discrepancy
                  SET status = 'resolved', resolved_by = %s, resolved_at = now()
                WHERE doc_type = 'contract' AND doc_pk_candidate = %s
                  AND coalesce(source_file,'') = %s AND issue_type = %s
                  AND field_name = %s AND status = 'open'""",
            (reviewer or "contract-parent-confirm", contract_id,
             normalise_source_file(source_file) or "", ISSUE_TYPE, FIELD_NAME),
        )
    return True


__all__ = ["candidate_parents", "propose_parent_links", "confirm",
           "ISSUE_TYPE", "FIELD_NAME", "MIN_SCORE", "SEPARATION"]
```

**One thing to check against the live table before running, not to guess at:** confirm `proc.bp_extraction_discrepancy` really has `resolved_by` and `resolved_at`, and that `bp_lifecycle_guard` permits `open → resolved` (it is known to block `ignored → resolved`). If a `status` check constraint rejects anything used here, report it rather than widening the constraint.

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c \
  "SELECT column_name FROM information_schema.columns
    WHERE table_schema='proc' AND table_name='bp_extraction_discrepancy'
      AND column_name IN ('resolved_by','resolved_at','computed_value','status');"
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/test_contract_links.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Prove the guards fail (required)**

1. **The runner runs.** Make `propose_parent_links` return its zeroed result before the query. Expected: `test_the_runner_is_actually_called` FAILS with "the runner looked at no children at all". Restore. This is the proof against the `contract_succession` failure mode.
2. **Nothing links itself.** Add `UPDATE proc.bp_contracts SET parent_contract_id = ...` beside the `INSERT`. Expected: `test_nothing_is_linked_without_a_person` FAILS. Restore.
3. **Idempotency holds.** Delete the `SELECT 1 ... LIMIT 1` guard. Expected: `test_running_twice_does_not_stack_proposals` FAILS — note whether it fails on a count of 3 or on a unique-violation error; either proves the guard. Restore.
4. **`source_file` separates two documents.** Drop `coalesce(source_file,'') = %s` from the idempotency `SELECT`. Expected: `test_two_documents_sharing_a_contract_id_each_keep_their_proposal` FAILS. Restore.
5. **`source_file` is normalised.** Return `row[0]` raw from `_source_file_for`. Expected: `test_the_source_file_is_stored_normalised` FAILS for a path needing normalisation. Restore.
6. **The considered counts are real.** Hard-code `considered = {"children": 0, "with_structure": 0, "with_candidates": 0}` at the return. Expected: `test_the_runner_is_actually_called` FAILS. Restore.
7. **The corpus supplies parents.** Delete the `bp_contract_master` block from `candidate_parents`. Expected: `test_the_existing_corpus_supplies_candidate_parents` FAILS. Restore.

- [ ] **Step 6: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add src/services/contract_links.py tests/services/test_contract_links.py
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "feat(contracts): propose each document's parent, link none of them" "" \
  "Task 10 of specs/2026-10-02-contract-structures-plan.md." "" \
  "Scores every parentless contract document against the candidate parents its" \
  "supplier holds -- from uploads and from the 3,051-row corpus -- and writes" \
  "one proposal per document into the Action Centre queue buyers already work." "" \
  "Nothing is linked: 1,561 existing parent pointers resolve to 0 real" \
  "contracts, so a person confirms. The idempotency key matches the partial" \
  "unique index column for column, source_file included and normalised, so two" \
  "documents sharing a contract_id keep their own proposals." "" \
  "The load-bearing test is that the runner is called at all: contract_" \
  "succession has had a scorer and no runner since it was written. Seven" \
  "guards proven red." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- src/services/contract_links.py tests/services/test_contract_links.py
git diff --cached --name-status | wc -l
```

---
## Task 11: Deploy to bp_sqldb, and prove it on the running server

Every task above applied its migration to `bp_testdb` only — the database `.env` points at. `bp_sqldb` is a separate deployment and has a history of falling behind: it was 8 governance migrations behind as recently as 2026-09-15, and a single missing index there silently rejected **65 days** of findings while looking exactly like a clean corpus.

**Files:**
- Create: `specs/2026-10-02-contract-structures-verification.md` (the evidence record)
- No source changes. If this task needs a code change, something above is incomplete — go back and fix it there.

**Interfaces:**
- Consumes: all five migrations and every module from Tasks 1-10.
- Produces: a written verification record, and both databases at parity.

**The five migrations, in dependency order.** Applying 4 before 2 would let `order form` claim every quote-template workbook on `bp_sqldb`:

1. `2026-10-02_document_type_parent_evidence.sql`
2. `2026-10-02_document_type_order_form_sales_order.sql`
3. `2026-10-02_contract_raw_resolved_type.sql`
4. `2026-10-02_contract_parent_reference_columns.sql`

(Task 1 added no migration; Tasks 7, 9 and 10 added none.)

- [ ] **Step 1: Record what bp_sqldb looks like before anything is applied**

```bash
PGPASSWORD="<bp_sqldb password>" psql -h <bp_sqldb host> -U <user> -d bp_sqldb -c \
  "SELECT count(*) FILTER (WHERE status='active') AS active_types,
          max(recorded_at) AS newest
     FROM proc.bp_document_type;"
PGPASSWORD=... psql ... -d bp_sqldb -c \
  "SELECT column_name FROM information_schema.columns
    WHERE table_schema='proc' AND table_name='bp_document_type'
      AND column_name IN ('requires_parent_evidence','parent_evidence_phrases');"
```

Paste both results into the verification record. "It was already there" and "I added it" are different facts and only the before-state tells them apart.

Use the same connection details the predecessor used for `deploy/sql/2026-10-01_sqldb_discrepancy_open_key.sql`. `bp_sqldb` credentials are **not** in `.env` — `.env` points at `bp_testdb`. If they cannot be found, stop and ask rather than guessing a host.

- [ ] **Step 2: Apply all four migrations to bp_sqldb, in order**

```bash
for m in 2026-10-02_document_type_parent_evidence \
         2026-10-02_document_type_order_form_sales_order \
         2026-10-02_contract_raw_resolved_type \
         2026-10-02_contract_parent_reference_columns; do
  echo "=== $m ==="
  PGPASSWORD=... psql -h <host> -U <user> -d bp_sqldb -v ON_ERROR_STOP=1 \
      -f "deploy/sql/$m.sql" || break
done
```

Then run the whole loop a second time. Every migration must succeed unchanged — that is the idempotency proof on the database that matters.

- [ ] **Step 3: Prove parity between the two databases**

```bash
PGPASSWORD=... psql -h <host> -U <user> -d bp_sqldb -tA -c \
  "SELECT md5(string_agg(concept_code || '|' || role || '|' ||
              coalesce(default_parent_type,'') || '|' ||
              array_to_string(aliases,',') || '|' ||
              coalesce(pipeline_doc_type,'') || '|' || status || '|' ||
              requires_parent_evidence::text || '|' ||
              array_to_string(parent_evidence_phrases,','), E'\n' ORDER BY concept_code))
     FROM proc.bp_document_type;"
```

Run the identical query against `bp_testdb`. **The two md5 values must match.** The predecessor held both databases at md5 parity for this table and that is the standard to keep.

If they differ, find out which rows differ before changing anything — `bp_sqldb` may carry a row a human confirmed there, which is legitimate and must not be overwritten by a seed.

- [ ] **Step 4: Run the drift test against bp_sqldb**

```bash
set -a && . ./.env && set +a
DB_NAME=bp_sqldb DB_HOST=<bp_sqldb host> DB_USER=<user> DB_PASSWORD=<pw> \
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/concepts/test_concept_table.py \
    tests/services/concepts/test_contract_type_map.py -v
```

Expected: the concept-table tests pass. `test_contract_type_map` may legitimately **fail its 3,051-row count** — that figure is `bp_testdb`'s corpus. Record `bp_sqldb`'s own distribution in the verification record and do not change the test; `bp_testdb` is the database its numbers describe.

- [ ] **Step 5: Run the whole suite against bp_testdb**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/ -q
```

Record the pass/fail counts. Compare against the predecessor's baseline at this tree (10,553 passed / 1 failed, the failure being another session's uncommitted `deal_assignment_service.py`). **Any new failure is this work's to fix**, including the three files that run in no CI job: `test_gate_wiring` (spacy), `test_agent_manifest_slices` (botocore+qdrant), `test_type_findings_lifecycle` (numpy+scipy). Run those three explicitly — they are not optional just because CI skips them.

- [ ] **Step 6: Restart the local server and confirm it comes up**

```bash
# Never `pkill -f uvicorn` -- it kills other sessions' servers.
# Use the project's own restart path, and check the route count.
```

Expected: the API starts and reports its routes. A vocabulary that fails to load falls back to the seed rather than raising, so a startup that *looks* clean is not proof the table loaded — check the log line `loaded N active concepts and M active document types` and confirm `M` has risen by 2.

- [ ] **Step 7: Live verification — upload a real contract**

This is the step the work is for, and tests do not substitute for it (`feedback_demonstrate_on_local_server_live_data`).

Upload a real contract document into the **Contracts** zone on the running local server, and record in the verification file:

1. The `proc.process_monitor` row — its `category` and `document_type`.
2. `SELECT resolved_doc_type, resolved_role, type_agreement FROM proc.bp_contract_raw WHERE source_file = '<key>';` — the structure the page's own text produced.
3. Whether a `document_type_disagreement` finding was raised. For a contract uploaded as "contract" that names its structure, there must be **none** — `type_agreement` should read `refined` (Task 5).
4. `SELECT framework_ref, parent_agreement_ref, parent_contract_id FROM proc.bp_contracts WHERE contract_id = '<id>';` — whatever reference the document actually printed, or `NULL` where it printed none.
5. `propose_parent_links()`'s return value, and the resulting `proc.bp_extraction_discrepancy` row with its `notes` in full.
6. That `parent_contract_id` is **still NULL** until `confirm()` is called.

**The only contract PDF known to exist locally is a Marketing Agreement that happens to list no schedules** — the very shape that broke the predecessor's round 1. If that is the only document available, say so plainly in the record and label the verification partial. Do not describe one document as a golden set.

A second upload worth doing if any order-form document can be produced: one that names a framework and one that does not, to show the stand-down rule on a real file rather than a constructed page.

- [ ] **Step 8: Write the verification record**

Create `specs/2026-10-02-contract-structures-verification.md` with: the before/after state of both databases, the md5 parity result, the suite counts against the predecessor's baseline, the full live-upload evidence from Step 7, and a plainly-labelled list of **what remains unproven**. At minimum that list carries:

- zero real order forms, frameworks, call-offs or SOWs exist in the corpus, so the stand-down rule's positive half is still demonstrated on a constructed page;
- `sales_order` is seeded on a reasoned mapping and no document in either database contains the words;
- `contract_hierarchy` has no labelled sample, so it stays in `UNCALIBRATED_PROFILES` and its score thresholds are unvalidated;
- `bp_sqldb`'s finding history is meaningful from 2026-10-01 forward only.

- [ ] **Step 9: Commit**

```bash
export GIT_INDEX_FILE="$SCRATCH/idx"
git read-tree HEAD
git add specs/2026-10-02-contract-structures-verification.md
TREE=$(git write-tree)
COMMIT=$(printf '%s\n' \
  "docs(contracts): deployment and live verification record" "" \
  "Task 11 of specs/2026-10-02-contract-structures-plan.md." "" \
  "Four migrations applied to bp_sqldb in dependency order, md5 parity with" \
  "bp_testdb confirmed on proc.bp_document_type, suite counts recorded against" \
  "the predecessor's baseline, and a real contract taken through the live" \
  "pipeline from upload to proposal." "" \
  "What remains unproven is listed explicitly, including that no real order" \
  "form, framework, call-off or SOW exists in either corpus." "" \
  "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" | git commit-tree "$TREE" -p HEAD)
git update-ref refs/heads/Development "$COMMIT"
unset GIT_INDEX_FILE
git reset -q HEAD -- specs/2026-10-02-contract-structures-verification.md
git diff --cached --name-status | wc -l
```

- [ ] **Step 10: Push**

Only after Step 5's suite is clean and Step 7's evidence is recorded.

```bash
git log --oneline origin/Development..Development    # expect 10 commits
git push origin Development
```

Work stays on `Development`. Never push to `main`.

---

## Plan Self-Review

Run after the plan is written, before handing it over.

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §3 vocabulary carries every structure (15 existing + order form + sales order) | 4 |
| §4 a child structure must show its parent | 2 (data), 3 (rule), 4 (the row) |
| §5 the answer gets stored | 6 |
| §5 the existing corpus's free-text type | 7 |
| §6 a specific structure under a generic declaration is a refinement | 5 |
| §7 the parent reference becomes extractable | 8 |
| §8 the maths runs, and proposes | 9 (scoring), 10 (runner + proposals) |
| §9 testing — all 13 guards | 1, 2, 3, 4, 5, 6, 8, 9, 10 |
| §9 live verification | 11 |
| §11 unproven items stated plainly | 11 Step 8 |
| §1 criterion 5 — nothing regresses | 1 (baseline), re-run in 2, 3, 4, 5 |

**2. Spec §9 guard-by-guard**

| Spec guard | Task | Test |
|---|---|---|
| 1 order form with no parent does not claim | 3, 4 | `test_an_order_form_that_names_no_parent_does_not_claim_the_page`, `test_no_quote_workbook_became_a_disagreement` |
| 2 order form naming its framework does classify | 3 | `test_an_order_form_that_names_its_framework_does_claim_the_page` |
| 3 the 53 documents resolve identically | 1 | `test_no_stored_document_classifies_differently_than_the_baseline` |
| 4 sales order wins over the `order` alias | 4 | `test_sales_order_wins_over_the_bare_order_alias` |
| 5 the structure reaches raw and survives promotion | 6 | `test_the_structure_survives_promotion` |
| 6 a silent page stores NULL | 6 | `test_a_page_that_states_nothing_stores_null_never_a_guess` |
| 7 generic + specific = refined, no review item | 5 | `test_a_refinement_raises_no_review_item` |
| 8 a genuine mismatch still raises one | 5 | `test_an_invoice_uploaded_as_a_contract_still_disagrees` |
| 9 `framework_ref` extracts and rejects `N/A` | 8 | `test_a_framework_reference_is_read`, `test_a_placeholder_is_refused_not_stored` |
| 10 `contract_hierarchy` proposes, never links | 9, 10 | `test_the_profile_can_never_auto_link`, `test_nothing_is_linked_without_a_person` |
| 11 an exact reference alone does not auto-link | 9 | `test_an_exact_reference_alone_does_not_reach_the_auto_band` |
| 12 the runner is actually called | 10 | `test_the_runner_is_actually_called` |
| 13 seed and table do not drift | 2, 4 | `test_document_type_rows_equal_the_seed_column_for_column` |

All 13 covered.

**3. Review Focus coverage**

| # | Task | Test |
|---|---|---|
| 1 a title naming two structures | 3 | `test_a_title_naming_both_stays_unresolved_with_both_candidates` |
| 2 flagged with no phrases | 2, 3 | `test_a_flagged_type_with_no_phrases_is_a_violation`, `test_a_flagged_structure_with_no_phrases_claims_nothing` |
| 3 a re-read refreshes | 6 | `test_a_reread_refreshes_the_stored_structure` |
| 4 two documents sharing a `contract_id` | 10 | `test_two_documents_sharing_a_contract_id_each_keep_their_proposal` |
| 5 nothing proposed ≠ everything parented | 10 | `test_nothing_proposed_is_distinguishable_from_everything_parented` |

**4. Type and name consistency**

- `requires_parent_evidence: bool`, `parent_evidence_phrases: Tuple[str, ...]` — declared in Task 2, used by the same names in 3, 4, 9.
- `resolved_doc_type` / `resolved_role` / `type_agreement` — declared in Task 6, read by the same names in 9 and 10.
- `framework_ref` / `parent_agreement_ref` — declared in Task 8, read in 9's `_REFERENCE_FIELDS` and 10's `_CHILD_SQL`.
- `expected_parent_type(child_structure, *, vocabulary=None)` — defined in Task 9, called in 10 by that signature.
- `structure_for_contract_type(value, *, vocabulary=None)` — defined in Task 7, called in 10.
- `promotion.promote(raw_id, doc_type)` — the real signature, verified against `promotion.py:583`.
- `normalise_source_file` — existing, in `persistence`, used in 10.

**5. Known deviation from the spec, carried deliberately**

Spec §8 says proposals go "through `link_proposals.py`". They do not: `LinkProposal` is purchase-order shaped (its parent field is literally `po_id`), so Task 10 creates a sibling module `src/services/contract_links.py` instead — matching how `deal_link_proposals.py` already sits beside `link_proposals.py`. The destination the spec actually promised, the queue a buyer already works, is unchanged: `proc.bp_extraction_discrepancy`. Flagged rather than silently done.
