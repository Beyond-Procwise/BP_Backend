# Extraction Confidence Learning — closing the loop between human corrections and reader trust

**Date:** 2026-08-01
**Status:** Implemented (Tasks 1–7), verified live 2026-07-31 against bp_sqldb
**Origin:** Plan `docs/superpowers/plans/2026-07-31-extraction-confidence-learning.md`

## What this is, in one sentence

Every extracted field is produced by a specific reader (a regex pattern, an engineered
gap-filler, or the context-layer AI). Every time a human corrects a field, this feature now
records *who* produced the wrong value and turns repeated correction into a measured
accuracy rate per reader. A reader nobody corrects keeps its static prior forever. A regex
pattern humans keep overriding is demoted, automatically, to last resort in its field.

### Which readers can actually be demoted — and which cannot

Be precise about the reach, because it is narrower than "the measurement decides which
reader the pipeline tries first" would suggest:

| Reader | Demotable? | Why |
|---|---|---|
| **regex** (L1 patterns) | **Yes** | `PatternRegistry.apply_observed` matches a measured rate against `pat.name` from the YAML registry and rewrites that pattern's `prior_confidence`. This is the only reader with a per-reader prior to lower and an ordering to fall down. |
| **context_layer** (AgentNick) | **No** | Its candidates carry `pattern_name=None`, so they are scored under the source `"context_layer"` — a real, visible rate — but there is no prior anywhere to lower and no ordering it participates in. On a live sample it produced **10 of 13** fields, so most of the corpus's attribution lands in a bucket that is measured and reported but cannot move the pipeline. |
| **ner** (L2 gap-filler) | **No** | Same shape: `pattern_name=None`, scored under its source, no prior. |
| **judge** (L3) | **No** | Carries `pattern_name="grounded_last_resort"`, which is in no registry, so `apply_observed` never matches it. |
| **hitl** | **N/A** | A human's own value is not a reader and is deliberately never scored. |

The measurement is therefore fully honest as a *measurement* for every reader, and acts
automatically only on regex patterns. Making the other three respond to it would mean giving
them a comparable per-reader prior first — a separate piece of work, not a wiring gap here.

## The four tables/columns, and what each one holds

### 1. `proc.bp_extraction_provenance` — who produced this value

Pre-existing table (2026-04-21), given a documented read contract by Task 1 rather than
schema changes:

```
id, parent_table, parent_pk, field_name, source, anchor_ref (jsonb), confidence,
attempt, extracted_at
```

One row per non-null column at promotion time, normally — minus an exclusion set
(`promotion._PROVENANCE_EXCLUDED_COLS`) for columns no reader ever produced: the four audit
stamps, `confidence_score` / `accuracy_score`, the derived FX pair, and the `deal_id` /
`deal_name` carried over from `process_monitor`. Those were writing ~5 rows per document
into a permanent table, all credited to `context_layer`, which both bloated the table and
diluted the very rate this feature measures.

A field a human corrects gets **two** rows instead of one: `source != 'hitl'` for the reader
that produced the value *before* the override (the one being judged), and `source = 'hitl'`
for what is actually stored now. `anchor_ref` carries the pattern name as a JSON string when
`source` is a regex reader; it is `NULL` for `context_layer` and `hitl` rows, which have no
pattern to name.

**Finding the latest claim: order by `id`, not by `attempt`.** `attempt` is numbered per
`raw_id`, not per document, so a re-extraction (a *new* `raw_id` for the same `parent_pk`)
starts again at `attempt=1` and `ORDER BY attempt DESC` would hand back the older `raw_id`'s
`attempt=2` row in preference to the newer claim. `id` is a monotonic `BIGSERIAL` over the
whole table and is the only correct "most recent". `attempt` is for telling the passes apart
after the fact.

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

**Where the reader on a verdict row actually comes from.** `proc.bp_extraction_provenance`
is written inside `promotion.promote()`, and the population that produces verdicts is by
construction documents that were *never promoted*: a blocking discrepancy sets
`promotion_status='discrepancy'`, `dispatch` skips the inline `promote()`, and
`promote_pending` only scans `'pending'`. So at the moment a human resolves a blocking
finding there is not one provenance row for that document.

`record_verdict` therefore takes a `snapshot` argument — the `_field_provenance` map
`dispatch` froze into `_raw.parser_snapshot` while the `Candidate` objects were still in
memory, which is exactly what `promote()` itself falls back to. `apply_hitl_fixes_and_promote`
reads it off the `_raw` row on the same query that fetches the doc pk and passes it in. The
provenance table still wins when it *does* have a row (a re-promoted document whose reader
has since changed); the snapshot is what fires in practice. Without it, `source`,
`pattern_name` and `prior_confidence` would be `NULL` on every real verdict — and a `NULL`
reader matches nothing in `apply_observed` (no pattern would ever be demoted) and nothing in
`_compute_accuracy_score` (`accuracy_score` would stay `NULL` forever), which would make the
whole loop decorative.

Verdict capture stays *before* `promote()` deliberately: a promotion that fails must still
leave the human's judgement recorded.

**Machine principals are not recorded at all.** `verdict.MACHINE_PRINCIPALS` —
`dedup-migration` (`deploy/sql/2026-07-30_discrepancy_dedup.sql`) and `session_postprocess`
(`src/services/session_postprocess.py`) — already write `status='resolved',
resolution_action='dismiss'` rows in bulk: 41 live, 20 of them with `blocks_promotion=TRUE`,
so they reach `apply_hitl_fixes_and_promote`. Each would otherwise mint a `rejected` verdict,
which `_AGREES` counts as "the reader was right"; eight of them crosses `MIN_SAMPLE` on its
own and would manufacture a perfect score for a reader no person ever endorsed, or bury a
demotion real corrections had earned. The skip is at **write** time rather than in
`accuracy._LOAD_SQL` because more than one consumer reads this table —
`supplier_currency`'s own `_LOAD_SQL` never goes through it — and because it keeps the
table's stated meaning ("one row per human judgement") literally true for every present and
future reader of it.

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
   `PatternRegistry.apply_observed()` leaves any `(doc_type, field, reader)` whose measured
   rate is `>=` its **YAML baseline** sitting exactly on that baseline. Raising a reader
   above what a human set as its prior would let learning silently promote documents that
   used to stop for review — a system that gets *bolder* on its own is the one failure mode
   this cannot have. It may only become more cautious.

   Every application is computed against the YAML baseline captured at compile time, never
   against whatever the previous application left on the object. The registry is a
   process-wide singleton (`get_registry`) and the accuracy map behind it refreshes on a
   15-minute timer, so comparing a fresh rate against an already-demoted prior would
   *ratchet*: `0.55` is not `< 0.20`, so a reader knocked down by one bad window could never
   recover before a process restart, and two workers started at different times would read
   the same document differently. Against the baseline, the compiled priors are a pure
   function of (YAML, current measurement): a reader whose rate recovers returns to exactly
   the prior a human wrote for it — and never one point above it.

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

### `proc.bp_supplier.default_currency` deliberately does not resolve a document

`resolve_dollar_currency`'s last resort reads `row["supplier_default_currency"]`, and
`dispatch._resolve_bare_dollar_currency_hint` populates that key **only** from
`learned_currency()` — three agreeing human corrections. It is *not* populated from
`proc.bp_supplier.default_currency`, even though that column is set on 5,000 of 5,027
suppliers (481 of them to a dollar currency: 456 USD, 19 SGD, 6 AUD).

Feeding the vendor master in would have changed extraction behaviour on day one, in the
bolder direction: a bare-`$` invoice with no currency code and no country used to raise a
blocking `currency_ambiguous` finding and stop for a person, and would instead have
auto-resolved from a static attribute of the supplier. The governing rule for this work is
that anything short of certainty goes to the Action page for a human, and confidence changes
over time *based on the human's response*. A row in the supplier master is not somebody
agreeing with us. Three corrections are.

`supplier_currency.default_for` still contains the master lookup behind an explicit
`allow_supplier_master=True`, so a caller can show a reviewer what the vendor record says
while they decide. The extraction pipeline never passes it.

## The measured output, at time of writing (2026-07-31, live bp_sqldb)

```
$ set -a; . ./.env; set +a; PYTHONPATH=.:src ./venv/bin/python scripts/show_extraction_learning.py
No verdicts recorded yet — every reader is still on its hand-set prior.
A reader needs 8 judgements before its measured rate is used.
```

This is the correct cold-start state, not a defect. The live corpus has 47 resolutions, and
46 of them are `dismiss` rows written by a machine principal on a finding that predates this
plan's verdict-capture wiring — none went through `record_verdict`, so
`proc.bp_extraction_verdict` is genuinely empty. Consequently:

- `load_accuracy()` returns `{}` (verified live).
- `cached_accuracy()` returns `{}`, so `_compute_accuracy_score` returns `None` and every
  document's `accuracy_score` is `NULL`.
- `PatternRegistry.apply_observed({})` on a fresh registry returns `0` and does not touch
  `_by_field` at all (verified live: the compiled pattern list is byte-identical before and
  after). Every pattern sits on its YAML-authored `prior_confidence`.
- `supplier_currency.default_for()` returns `None` for a live supplier whose master
  `default_currency` is `USD`, so `dispatch._resolve_bare_dollar_currency_hint` leaves
  `supplier_default_currency` absent from `columns` and a bare-`$` document with no code and
  no country raises its blocking `currency_ambiguous` finding exactly as before (verified
  live against three real USD-defaulted suppliers).
- **Extraction and promotion behaviour on the current cold corpus is identical to before
  this plan started.** The only observable differences are additive records that no document
  reads: rows in `bp_extraction_provenance` and `bp_extraction_verdict`, and an
  `accuracy_score` column that is `NULL` everywhere.

The loop only becomes visible once people work through live findings under the new wiring —
the 300 duplicate-invoice findings and the currency-ambiguity findings are the population
that will populate `bp_extraction_verdict` going forward.

### Performance

`load_accuracy()` is not cheap: `src.services.db.get_conn` has no pool (every call is a
fresh `psycopg2.connect`) and `_LOAD_SQL` filters on `decided_at`, which has no index — so
an uncached call is a TCP connection plus a sequential scan of the whole verdict table. It
was being paid per document by `dispatch` *and* per promoted document by `promote()`. Both
now share one process-wide 15-minute cache (`accuracy.cached_accuracy`), and `promote()`
passes its **existing** connection in, so a refresh needs no second connection. The refresh
runs inside a `SAVEPOINT` when that connection is transactional, so a failed read cannot
abort the promotion it is riding inside; a failed refresh keeps the previous map and does
*not* mark the cache fresh, so the next document retries rather than pinning an empty map
for 15 minutes. `provenance.record` batches its dozen-odd inserts into one `executemany`.

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

That chain proves the *arithmetic*. What proves the **wiring** — that a real human action
produces a verdict naming a real reader — is
`tests/test_promotion_hitl_security.py::TestVerdictCarriesTheRealReader`, which drives the
actual `apply_hitl_fixes_and_promote` over a blocked-then-corrected document with no
provenance rows (the universal production case) and asserts the verdict `INSERT` carries
`regex` / `dollar_symbol` off the snapshot. It was confirmed to fail against the pre-fix code
by reverting the snapshot argument and watching it go red, then restoring it — the earlier
tests all passed against the broken code because they hand-fed the provenance tuple into a
fake cursor.

## Verification

- `tests/services/test_learning_loop_e2e.py` — 2/2 passed.
- Full `tests/services/` sweep: **985 passed / 91 failed / 27 skipped**. The 91 are the
  documented pre-existing failures (79 `test_style_*`, 7 `test_langextract_adapter`, 4
  `test_agent_actions`, 1 `test_negotiate_dashboard`) — unchanged.
- `tests/test_promotion_hitl_security.py` (21) and
  `tests/extraction/test_dispatch_currency_resolution.py` (6) — all pass. Note the latter
  lives outside the `tests/services` sweep.
- `tests/extraction/test_dispatch.py` has 2 failures confirmed pre-existing via `git stash`.
- Live against bp_sqldb: `load_accuracy() == {}`; `apply_observed({})` leaves the compiled
  registry byte-identical; `default_for()` returns `None` for a supplier whose master says
  `USD`; three real USD-defaulted supplier names produce no currency hint.

## What would make the loop visible

Nothing further needs building. The mechanism is complete and tested; it is idle only
because `proc.bp_extraction_verdict` has no rows yet. As HITL corrections are made against
documents processed under the new provenance/verdict wiring, `MIN_SAMPLE` will start being
crossed field-by-field, `scripts/show_extraction_learning.py` will start printing measured
rates instead of the cold-start message, and readers that are genuinely being corrected will
start losing their place in line — automatically, with no further code change.
