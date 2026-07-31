# Extraction Confidence Learning — closing the loop between human corrections and reader trust

**Date:** 2026-08-01
**Status:** Implemented (Tasks 1–7), verified live 2026-07-31 against bp_sqldb
**Origin:** Plan `docs/superpowers/plans/2026-07-31-extraction-confidence-learning.md`

## What this is, in one sentence

Every extracted field is produced by a specific reader (a regex pattern, an engineered
gap-filler, or the context-layer AI). Every time a human corrects a field, this feature now
records *who* produced the wrong value, turns repeated correction into a measured accuracy
rate per reader, and lets that measurement — never a hand-set guess — decide which reader the
pipeline tries first. A reader nobody corrects keeps its static prior forever. A reader
humans keep overriding is demoted, automatically, to last resort in its field.

## The four tables/columns, and what each one holds

### 1. `proc.bp_extraction_provenance` — who produced this value

Pre-existing table (2026-04-21), given a documented read contract by Task 1 rather than
schema changes:

```
id, parent_table, parent_pk, field_name, source, anchor_ref (jsonb), confidence,
attempt, extracted_at
```

One row per non-null column at promotion time, normally. A field a human corrects gets
**two** rows instead of one: `source != 'hitl'` for the reader that produced the value
*before* the override (the one being judged), and `source = 'hitl'` for what is actually
stored now. `attempt` distinguishes a re-promoted document; consumers must filter to the
canonical (latest) attempt, which `verdict.py`'s query does via `ORDER BY attempt DESC, id
DESC`. `anchor_ref` carries the pattern name as a JSON string when `source` is a regex
reader; it is `NULL` for `context_layer` and `hitl` rows, which have no pattern to name.

### 2. `proc.bp_extraction_verdict` — what the human decided about that value

New table (Task 2, `deploy/sql/2026-08-01_bp_extraction_verdict.sql`):

```sql
CREATE TABLE proc.bp_extraction_verdict (
    verdict_id       BIGSERIAL PRIMARY KEY,
    doc_type         TEXT        NOT NULL,
    doc_pk           TEXT        NOT NULL,
    field_name       TEXT        NOT NULL,
    source           TEXT,               -- copied from provenance, survives a provenance purge
    pattern_name     TEXT,
    prior_confidence NUMERIC,
    verdict          TEXT        NOT NULL CHECK (verdict IN ('confirmed','corrected','rejected')),
    extracted_value  TEXT,
    corrected_value  TEXT,
    decided_by       TEXT,
    decided_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

One row per human judgement, written from `apply_hitl_fixes_and_promote` (the HITL
resolution path) with savepoint isolation so a verdict-capture failure cannot abort the
promotion it rides alongside. `verdict_for()` (`src/services/extraction_feedback/verdict.py`)
maps a resolution action to one of three values:

- **`confirmed`** — the human's `apply_value` matched the extracted value: the reader was
  right.
- **`corrected`** — the human's `apply_value` (or a `keep_null` over a non-empty extracted
  value) replaced it: the reader was wrong. The strongest negative signal there is.
- **`rejected`** — the finding was *dismissed*. This reads as agreement WITH the extracted
  value, not against it — a dismissed finding means a person looked and decided nothing was
  wrong. Scoring it as a negative would teach the system to distrust exactly the readers
  people keep agreeing with.

### 3. Per-pattern `prior_confidence` — the hand-set starting guess

Lives in `extraction_schemas/<doc_type>.yaml`, not a database table — a deliberate choice:
these are the values every pattern shipped with, and `PatternRegistry.apply_observed()`
mutates the in-memory compiled copy for the life of the process, never the file on disk. A
fresh process (or `clear_cache()`) always starts from the YAML numbers again; a measurement
is a runtime overlay, not a rewrite of the source of truth.

### 4. `accuracy_score` — the reader-trust half of a document's confidence

New column on all six invoice/quote/PO `_stg`/`_trgt` tables (Task 6,
`deploy/sql/2026-08-01_bp_extraction_accuracy_score.sql`), `NUMERIC`, nullable. Distinct from
the pre-existing `confidence_score`, which measures *completeness* (how many fields are
filled, regardless of whether they are right). `accuracy_score` is the mean measured
agreement rate of the readers that produced this document's fields, from
`proc.bp_extraction_verdict` via `load_accuracy()`. `NULL` means no reader on the document has
enough verdicts yet to have a rate — the honest, and currently universal, answer.
`_compute_confidence_score` (which telemetry and the training-example ≥0.90 gate already
read) was deliberately left untouched; `accuracy_score` sits beside it rather than
repurposing it.

## The two safety rules

1. **A measurement may only ever LOWER trust, never raise it.**
   `PatternRegistry.apply_observed()` skips any `(doc_type, field, reader)` whose measured
   rate is `>=` the current prior. Raising a reader above what a human set as its prior would
   let learning silently promote documents that used to stop for review — a system that gets
   *bolder* on its own is the one failure mode this cannot have. It may only become more
   cautious.

2. **Too little evidence means ABSENT, not zero.**
   `observed_accuracy()` (Task 3) only emits a key once it has seen at least `MIN_SAMPLE`
   judgements for that `(doc_type, field, reader)`. Below the threshold, the key is missing
   from the returned map entirely — callers (`apply_observed`, `load_accuracy`'s consumers)
   must read a missing key as "keep the static prior," never as a rate of 0. Acting on a
   handful of corrections as if they were a settled rate would let one bad afternoon of
   corrections switch a reader off.

## The thresholds, and why they are those numbers

- **`MIN_SAMPLE = 8`** (`src/services/extraction_feedback/accuracy.py`) — the number of
  judgements a `(doc_type, field, reader)` needs before its measured rate replaces the
  hand-set prior. Below 8, a rate is noise: a run of bad luck (or a genuinely ambiguous batch
  of documents) could produce an ugly rate on a handful of samples that says nothing durable
  about the reader. Acting on noise is worse than acting on the prior a human already
  considered.

- **`MIN_AGREEMENTS = 3`** with a **`MAJORITY = 0.75`** share
  (`src/services/extraction_feedback/supplier_currency.py`) — the threshold for accepting a
  *learned per-supplier currency* from corrections. Three independent people correcting the
  same supplier to the same currency is a policy; one correction is an opinion, and could be a
  one-off error on a single invoice. The 0.75 majority guards the case where corrections for a
  supplier are split across two currencies (a supplier that genuinely bills in more than one,
  or where the corpus itself is ambiguous) — guessing in that situation is exactly what this
  feature exists to prevent, so it deliberately produces no answer rather than a shaky one.
  Only `corrected` verdicts vote; a `confirmed` verdict says the value we had was already
  right and carries no information about what the currency *is* when we had nothing.

## The measured output, at time of writing (2026-07-31, live bp_sqldb)

```
$ set -a; . ./.env; set +a; PYTHONPATH=.:src ./venv/bin/python scripts/show_extraction_learning.py
No verdicts recorded yet — every reader is still on its hand-set prior.
A reader needs 8 judgements before its measured rate is used.
```

This is the correct cold-start state, not a defect. The live corpus has 46 human
resolutions, and every one of them is a `dismiss` on a finding that predates this plan's
verdict-capture wiring (Task 2) — none of them went through `record_verdict`, so
`proc.bp_extraction_verdict` is genuinely empty. Consequently:

- `load_accuracy()` returns `{}`.
- Every document's `accuracy_score` is `NULL`.
- Every pattern in every `PatternRegistry` sits on its YAML-authored `prior_confidence`,
  unchanged.
- **Pipeline behaviour today is bit-for-bit identical to before this plan started.**

The loop only becomes visible once people work through live findings under the new wiring —
the 300 duplicate-invoice findings and the currency-ambiguity findings Task 5 now raises are
the population that will populate `bp_extraction_verdict` going forward.

## What proves the loop actually closes

Tasks 1–6 each proved their own link in the chain in isolation (provenance attribution,
verdict capture, rate computation, prior demotion, per-supplier currency, accuracy scoring).
None of them chained a raw human action through to a demoted reader in one test.
`tests/services/test_learning_loop_e2e.py` does:

1. `verdict_for("apply_value", resolved_value="CAD", extracted_value="USD")` — a human
   correcting a currency reader, ten times — produces `"corrected"` verdicts.
2. `observed_accuracy()` turns those ten rows into a measured rate of `0.0` for
   `("invoice", "currency", "dollar_symbol")`.
3. A **freshly constructed** `PatternRegistry("invoice")` — confirmed, not assumed, to be an
   independent object per call; only `get_registry()` uses the process-wide cache — starts
   `dollar_symbol` above zero, then `apply_observed(acc)` drives its prior to `0.0`, strictly
   below where it started.
4. `dollar_symbol` is now sorted *last* among `currency` readers: the reader of last resort,
   not a trusted one.

A companion test (`test_agreement_leaves_a_good_reader_exactly_where_it_was`) runs the same
chain with `confirmed` verdicts and asserts every prior is unchanged — the loop only ever
moves a reader down, never up, and never touches a reader nobody corrected.

## Verification

- `tests/services/test_learning_loop_e2e.py` — 2/2 passed.
- Full `tests/services/` sweep: see Task 7 report for the exact counts; delta over the
  91-pre-existing-failure baseline is the ~40 tests this plan added across Tasks 1–7, with
  zero new failures.
- `scripts/show_extraction_learning.py` run against live bp_sqldb: output reproduced above.

## What would make the loop visible

Nothing further needs building. The mechanism is complete and tested; it is idle only
because `proc.bp_extraction_verdict` has no rows yet. As HITL corrections are made against
documents processed under the new provenance/verdict wiring, `MIN_SAMPLE` will start being
crossed field-by-field, `scripts/show_extraction_learning.py` will start printing measured
rates instead of the cold-start message, and readers that are genuinely being corrected will
start losing their place in line — automatically, with no further code change.
