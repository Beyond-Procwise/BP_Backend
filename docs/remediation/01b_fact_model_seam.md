# Phase 1b — Carrying the Fact Model Across the Seam

**Status:** implemented 2026-08-07 on `Development`.
**Plan:** `docs/superpowers/plans/2026-08-07-phase-1b-fact-model-across-the-seam.md`

A typed, provenanced `CommercialFact` is now the atomic unit that survives from
extraction to the opportunity record. For any opportunity, a single SQL
statement reconstructs the unit price, quantity, unit of measure, currency,
term and source document of every number contributing to it, without parsing
free text.

No LLM is involved anywhere in this phase. The assembler, the UoM normaliser,
the FX resolver and the arithmetic checker are pure functions or plain SQL.

---

## 1. What was built

| Artefact | Responsibility |
|---|---|
| `src/services/facts/uom.py` | Deterministic UoM normalisation; refuses what it cannot map |
| `src/services/facts/fx.py` | Rate resolution against `bp_fx_rates`; `FX_UNAVAILABLE` |
| `src/services/facts/models.py` | `CommercialFact`, `FactProvenance`, `Constraint`, enums, validators |
| `src/services/facts/arithmetic.py` | `quantity × unit_rate = extended_line` role check |
| `src/services/facts/assembler.py` | Builds facts from `_trgt` rows + provenance |
| `src/services/facts/store.py` | Persistence for facts, provenance, constraints, finding links |
| `src/services/facts/concept_codes.py` | Concept vocabulary derived from the extraction schemas |
| `src/services/facts/deprecation.py` | Logged one-release shim over `calculation_details` |
| `deploy/sql/2026-08-07_commercial_fact.sql` | Four new tables (+ rollback) |
| `deploy/sql/2026-08-07_opportunity_structured_columns.sql` | Structured columns on `bp_opportunity` (+ rollback) |
| `scripts/backfill_opportunity_structured.py` | The honest harvest out of `calculation_details` |

Four new tables: `bp_commercial_fact`, `bp_fact_provenance`, `bp_constraint`,
`bp_finding_fact`. All carry `tenant_id` and the bitemporal trio
(`valid_from`, `valid_to`, `recorded_at`). Both migrations are additive,
idempotent and reversible, and both were applied to `bp_sqldb` and `bp_testdb`;
both rollbacks were executed and re-applied on `bp_testdb`.

**108 tests pass** (`tests/services/facts/`, plus the backfill suite). The
86 pre-existing opportunity-miner and store tests still pass.

---

## 2. Why `bp_finding_fact` is a join table, not flat columns

Measured on `bp_opportunity`: **301 of 308 rows draw on 3 source documents**,
one on 4, six on 1. Flattening `unit_price` onto the opportunity row would
force choosing one of three documents arbitrarily, and nothing downstream could
tell which had been chosen.

Facts are also independent of findings — one invoice line can support both an
overbilling finding and a duplicate finding — so the relationship is
many-to-many with a `role`, keyed `(opportunity_ref_id, fact_id, role)`.

There is deliberately **no foreign key to `bp_opportunity`**: its primary key
is `opportunity_id`, while findings are addressed here by `opportunity_ref_id`,
which carries no unique index.

---

## 3. Provenance is mandatory — enforced twice

A `CommercialFact` with empty provenance cannot be constructed. This is
enforced in **two independent places**, because either alone is insufficient:

1. A Pydantic `field_validator` on `CommercialFact.provenance`. Catches every
   path through the model.
2. A **deferred constraint trigger** on `proc.bp_commercial_fact`. Catches the
   backfill, any direct `INSERT`, and anything that bypasses the model.

A `CHECK` constraint cannot express this rule: a `CHECK` sees only its own row,
and "at least one row exists in another table" is a cross-table predicate that
may not contain a subquery. A plain `AFTER INSERT` trigger cannot work either —
the fact must exist before its provenance can reference it, so an immediate
check would reject every correct insertion. `DEFERRABLE INITIALLY DEFERRED`
runs at `COMMIT`, the only point at which the question is meaningful.

Both were proven to reject, not merely to exist:

```
fact with no provenance     -> ERROR at COMMIT, 0 rows survive
fact + provenance together  -> COMMIT, 1 row
deleting last provenance    -> ERROR, 1 provenance row remains
unit_rate, basis_uom NULL   -> ck_bp_commercial_fact_unit_rate_needs_basis
priced, arithmetic NULL     -> ck_bp_commercial_fact_priced_needs_arithmetic
measure_role = 'price'      -> ck_bp_commercial_fact_measure_role
```

### 3.1 A trigger bug the acceptance test caught

The first version of the trigger rejected a correct **insert-then-delete within
one transaction**. Postgres does not discard an already-queued deferred
constraint-trigger event when the row is later deleted, so the check fired at
`COMMIT` for a fact that no longer existed. Any test fixture cleaning up after
itself hit it. The trigger now returns early when the fact row is gone — there
is no unprovenanced fact to protect against once the fact itself has been
deleted. The rejection guarantee was re-proven after the fix.

### 3.2 `persist_facts` refuses an autocommit connection

`src.services.db.get_conn()` returns a connection with **autocommit ON**. On
such a connection every statement is its own transaction, so the deferred
trigger is evaluated the instant the fact row is inserted — before its
provenance can possibly exist. `store.persist_facts` now raises a clear error
rather than failing obscurely. The subtler danger is the case that does *not*
fail: a fact and its provenance committed as two separate transactions are
briefly committed apart, and a crash between them leaves behind exactly the
unprovenanced fact this phase exists to prevent.

---

## 4. F6 — a number's role is carried, never inferred

**This is the load-bearing design decision of the phase.**

A number's role cannot be inferred from its field name. "Price" may mean a unit
rate or an extended total; "volume" may mean a count or a physical measure. The
schemas here *already* name `unit_price` apart from `line_total` — and a real
bug in this codebase still booked a line total as a unit price. Correct names,
wrong values. A dictionary would not have caught it.

What discriminates is **type plus arithmetic**: `quantity × unit_price =
line_total` *tests* the assignment instead of trusting the label. So the model
gained three columns — `measure_role`, `basis_uom` and `arithmetic_state` — each
enforced by a database `CHECK` as well as by Pydantic.

`value_basis` (as-supplied / baseline-corrected / normalised) says *whose
baseline*, not *what kind of measure*; it does not answer this question and was
never intended to.

### 4.1 The measured invariant

Run by `check_line_arithmetic` over **all** lines on `bp_sqldb`:

| Line table | Rows | Testable | Consistent | `quantity = 1` |
|---|---|---|---|---|
| quote | 316 | 136 | 102 (**75.0%**) | 14.9% |
| invoice | 152 | 81 | 78 (**96.3%**) | 11.8% |
| purchase order | 171 | 85 | 79 (**92.9%**) | 22.8% |

The plan's F6 recorded 78.1% / 92.9% / 91.9%. The rates agree within a few
points and the `quantity = 1` share matches almost exactly (14.9% vs 15%).

The *testable* counts differ, and the reason matters: F6 counted quote lines as
183 testable / 143 consistent, against 136 / 102 here. The difference is 47,
and there are 48 quote lines with `quantity = 1`. **F6's own query counted
qty-1 lines as consistent.** They are not — at `quantity = 1` a unit rate and a
total are numerically identical and the arithmetic carries no information about
the role either way. The shipped checker abstains there. It is therefore
*stricter* than the measurement that produced F6, not divergent from it.

### 4.2 The `quantity = 1` blind spot

Between 11.8% and 22.8% of real lines have `quantity = 1`. On those, no
arithmetic can separate a unit rate from a total. This is precisely the blind
spot that let the historical unit-price-as-total bug ship unnoticed. Such facts
are recorded `UNTESTABLE_QUANTITY_ONE` — **never** `CONSISTENT` — so a fact
whose role was never verified cannot look identical to one that was. The
ordering of that branch (before the equality comparison) is load-bearing and is
covered by a test that fails if the branches are swapped.

### 4.3 An `INCONSISTENT` line is kept, not dropped

The record carries its own reliability. Suppressing inconsistent lines would
hide 7–25% of the corpus and quietly improve the apparent numbers.

On `bp_testdb` the invariant holds at 100%, but only because the seeder
computed it that way. That figure is evidence of nothing.

---

## 5. F2 — mandatory provenance is expensive, and that is correct

This is the headline operational finding, and anyone testing on `bp_testdb`
will otherwise conclude the assembler is broken.

| | Lines | Provenance rows | Facts produced |
|---|---|---|---|
| **`bp_sqldb`** | 639 | 64,118 | **108 (16.9%)** |
| **`bp_testdb`** | 193,857 | 722 | **0** (1,200-doc sample) |

`bp_testdb` was **seeded, not extracted**. Nearly every row there fails the
mandatory-provenance rule and produces no fact. That is correct fail-closed
behaviour, not a defect.

Even on `bp_sqldb`, mandatory provenance costs **83% of lines**. Per document
type: invoice 46.1% of lines became facts, quote 8.9%, purchase order 5.8%.

Distribution across the 108 facts assembled and persisted on `bp_sqldb`:

```
consistent                  70   64.8%
untestable_quantity_one     22   20.4%
untestable_missing_input    13   12.0%
inconsistent                 2    1.9%
(no role)                    1    0.9%
-> of 72 TESTABLE facts, 70 consistent (97.2%)
```

97.2% sits above the corpus-wide rates in §4.1 because these 108 facts are a
provenance-filtered subset: 70 of them are invoice facts, and invoices are the
best-covered document type at 96.3% corpus-wide. The `quantity = 1` share
(20.4%) matches the corpus.

**Consequence for anyone testing this:** acceptance must run against
`bp_sqldb`. `bp_testdb` will show zero facts and that is the expected result.

---

## 6. F1 — the join direction is what makes this possible

Joining *provenance → `_trgt`* looks catastrophic: of 125 quote `doc_pk`s in
`bp_extraction_provenance_v3`, only 45 match a `bp_quote_trgt.quote_id`;
invoice 51 of 208; purchase order 39 of 81.

That is not data loss. The unmatched `doc_pk`s are values like `'10'`,
`'048597'`, `'005-022'` — failed extraction attempts whose primary key was
garbage and which never promoted. Provenance records **every attempt**.

Driven the other way, coverage is near-perfect. The assembler therefore always
drives from the `_trgt` row and looks provenance up by
`(doc_type, doc_pk, field_path)`. It never enumerates provenance and joins
forward. Mandatory provenance is achievable precisely because of this
direction, and a test asserts the query shape rather than trusting the comment.

### 6.1 The line index is 0-based against 1-based — verified, not assumed

Provenance writes `line_items[0].unit_price` for the **first** line; the `_trgt`
line tables number lines from **1**. Verified against `bp_sqldb` by value:
matching on `line_no - 1` reproduces the `_trgt` row's own `unit_price` (995.00
= 995), while matching on `line_no` returns a *different* line's price (300,
205, 400). An off-by-one here would silently attach the wrong line's evidence
to a price — worse than carrying no evidence at all.

### 6.2 Duplicate provenance attempts

The same `(doc_pk, field_path)` appears once per extraction attempt. The best
row is selected by confidence, then recency, then row id. The ranking is done
**in Python**, not left to the SQL `ORDER BY` alone: the in-memory store this
codebase substitutes under pytest does not implement ordering, and determinism
that only holds against real PostgreSQL is not determinism.

---

## 7. F5 — the UoM normaliser refuses what it cannot map

Fourteen of the 28 distinct `unit_of_measure` values across the three `_trgt`
line tables are units. Fourteen are payment terms, scope descriptions or prices
that landed in the UoM column: `'30 days from quote date'`, `'annual in
advance'`, `'included'`, `'transition 7 weeks'`, `'onboarding 10 weeks'`,
`'implementation (one-off, fixed) — £72,000.00'`. All fourteen yield
`UOM_UNMAPPED` and carry forward un-normalised.

**Matching is exact on the normalised key. Never substring.** This is the
single most important rule in the module: `'transition 7 weeks'` contains
`week`. Proven by mutation — introducing a substring fallback coerces 10 of the
14 junk values into units and turns the suite red.

An absent UoM is `UOM_UNMAPPED`, never defaulted to `each`. Time units carry a
factor in days, with `CALENDAR_CONVENTION_30D_365D` stamped on month and year
so the ambiguity travels with the number.

### 7.1 Correction to F5: UoM is absent far more often than it is junk

F5 enumerated the distinct *non-null* values, so it never surfaced the null
rate. Measured on `bp_sqldb`:

| Line table | Rows | `unit_of_measure` NULL |
|---|---|---|
| invoice | 152 | **152 (100%)** |
| purchase order | 171 | **171 (100%)** |
| quote | 316 | 205 (65%) |

A literal reading of the model rule (`unit_rate` ⟹ `basis_uom` non-empty) would
therefore have rejected **every** invoice and purchase-order priced fact in the
corpus. Three options existed and only one is honest:

- default to `each` — fabricates a unit for the entire corpus;
- call it an `extended_line` — restates a unit price as a total, the exact bug
  F6 exists to prevent;
- **record explicitly that the document did not say.**

The third was taken: `basis_uom = 'unstated'` with reason code `UOM_ABSENT`.
The sentinel is deliberately not a unit name, does not normalise to any
canonical unit, and a test asserts that collision cannot happen. This follows
the plan's own principle for the adjacent case — *"never NULL, because a NULL
would make the model reject a fact that genuinely exists"*.

**106 of the 108 facts on `bp_sqldb` carry `UOM_ABSENT`.** A comparison layer
must refuse to compare two rates whose basis is unstated; that is Phase 5's job
and the reason code is what makes it possible.

The other two are recovered from the item description (§14.4): where the unit
column fails and the description ends in a bare period adverb — `HR Advisory &
Employment Law Retainer (Quarterly)` — the basis is read from there and
stamped `BASIS_FROM_DESCRIPTION`, so a derived basis never looks like one the
document stated outright.

---

## 8. F4 — FX is reproducible, not historical

`bp_fx_rates` has columns `(base_currency, currency, rate, fetched_at)`.
`fetched_at` records when the rate was *fetched*, not when it was effective, so
the table cannot answer "what was GBP/USD on 2025-04-01".

Each fact therefore stamps its own `fx_rate`, `fx_rate_date` and
`fx_rate_source`. This satisfies the brief's **reproducibility** requirement —
re-rendering tomorrow cannot move yesterday's number — and it does **not**
deliver historical accuracy. That limitation is recorded, not papered over. A
dated rate corpus is separate work.

### 8.1 Correction to F4: there are 7 snapshots, not one

F4 recorded a single snapshot dated 2026-07-16. Live `bp_sqldb` now holds
**1,162 rows across 7 distinct `fetched_at` values** (to 2026-07-28), and
`base_currency` is always `USD`. Two consequences:

- A query with no ordering returns an arbitrary duplicate, so the same report
  could stamp different rates on two runs. Every lookup orders by
  `fetched_at DESC LIMIT 1`, and a test asserts the ordering is in the SQL.
- Because the only base is USD, a pair such as EUR→GBP has no direct row and is
  crossed through the table's base currency.

Live check: USD→GBP 0.743043, GBP→USD 1.345817 (exact reciprocal), EUR→GBP
0.856715 via cross-rate, `XYZ`→GBP `FX_UNAVAILABLE` with a NULL rate — never a
guessed parity.

---

## 9. B3 resolved — no data dictionary is needed

The brief names GPSS as "the vocabulary" and says to extend it rather than
shadow it. **There is no GPSS dictionary in this project, and on inspection
there does not need to be one.**

| What the brief wants from a vocabulary | What already provides it |
|---|---|
| Stable field names | `extraction_schemas/*.yaml` — a versioned, declarative registry of **66 distinct field definitions** across four document types. This *is* a data dictionary; it simply is not called one. |
| A category key | `proc.bp_category` holds an `L1~L2~L3` hierarchy (49 rows, 22 distinct on `bp_sqldb`); `bp_requirement.category` is populated on 6,009 of 6,010 rows. |
| Cross-document concept identity | Implicit today (same name); closed by one `concept:` key per schema field when Phase 5 needs it. |
| Detecting an unknown field | The schema registry is the checklist. |

A formal external dictionary is only required for **interoperability** —
exchanging facts with another system, or mapping onto a customer's taxonomy.
Nothing in Phases 1–5 requires that.

**Decision:** the column is `concept_code`, not `gpss_code`, and its vocabulary
is **derived at import time** from the extraction-schema field names rather than
hand-listed. A hand-typed list becomes a second vocabulary the moment either
side changes — exactly the shadowing failure the constraint exists to prevent.
An unknown code is rejected by a validator rather than stored.

Naming the column after a standard not actually in use here would invite the
next reader to assume external authority behind the values. **If GPSS is later
adopted, `concept_code` is the column it maps into.**

> ⚠️ **Correction to the Phase 0 seam map.** That document recorded
> `bp_category` as empty. It is empty on `bp_testdb` but **populated on
> `bp_sqldb`** with the hierarchy above. The seam map has been amended.

The UoM normaliser stays **category-independent** — it maps a string to a unit
and a dimension, not to a per-category basis. That is not a missing-dictionary
workaround: the per-category basis (software → per user per month; services →
per day per grade) is a *rendering* concern belonging to the Category Profile
Resolver in Phase 5, keyed on the hierarchy above.

---

## 10. The backfill harvest was narrow, and says so

⚠️ **`bp_opportunity` holds 0 rows on `bp_sqldb` and 308 on `bp_testdb`** — the
inverse of the fact tables. The plan's Task 7 shapes were measured on
`bp_testdb`. Both databases were migrated and back-filled regardless.

Result on `bp_testdb` (308 rows):

| Detector | Rows | Harvested | `facts_state` |
|---|---|---|---|
| Duplicate Invoice Recovery | 300 | `currency`, `amount_native` | RESOLVED |
| Price Benchmark Variance | 2 | `quantity`, `unit_price` (from `actual_price`) | RESOLVED |
| Invoice Overbilling | 6 | nothing structured | **INDETERMINATE** |

**302 RESOLVED, 6 INDETERMINATE.** Verified in the database: every
INDETERMINATE row has all eleven new columns NULL.

Invoice Overbilling carries `po_total`, `quote_total` and `invoice_total` —
three *different documents'* totals. None is "the" amount of the finding, and
picking one would be a guess presented as a harvest.

Two values were deliberately **not** harvested: `amount_gbp` (already
converted; writing it into `amount_native` would restate a converted figure as
the document's own) and `benchmark_price` (the comparator, not what was paid).

`INDETERMINATE` is not a synonym for zero. A finding whose amount could not be
resolved must not read as a finding worth nothing.

The harvest is driven by **key, not by detector name** — a detector renamed
tomorrow keeps working, whereas keying on the display string would fail
silently the day someone edits it.

---

## 11. The deprecation shim

`calculation_details` is no longer the system of record but is still written
for one release. Phase 0 measured the surface: **2 producers, 4 reader groups,
1 store, and no API or render path reads it at all.**

Every read now goes through `read_calculation_detail(record, key)`, which
prefers the structured column and **logs every JSONB fallback** with the key
and the opportunity id. That log is the retirement criterion: when it stops
firing in production, the column can go. "We think nothing reads it any more"
is not evidence, and a shim nobody can measure cannot be retired.

Two details that would otherwise defeat it:

- **Aliases.** The benchmark detector writes `actual_price`; the column is
  `unit_price`. Without the alias the shim would fall back forever, so the
  column would look unused and the JSONB indispensable — exactly backwards.
- **A NULL column is not authoritative.** An INDETERMINATE row has NULL columns
  by design; letting NULL win would silently drop values the JSONB still holds
  during the shim release.

---

## 12. Acceptance

| Criterion | Result |
|---|---|
| All new tests pass | ✅ 108 passed, 4 skipped (live-only) |
| A `CommercialFact` cannot exist without provenance | ✅ by test **and** by trigger rejecting a commit |
| Reconstruction with no free-text parsing | ✅ asserted textually and behaviourally |
| Both migrations on both databases; both rollbacks executed and re-applied | ✅ |
| The 14 `UOM_UNMAPPED` values refused, not coerced | ✅ proven by mutation |
| `unit_rate` needs `basis_uom`; priced fact needs `arithmetic_state` | ✅ by test **and** by database `CHECK` |
| `quantity = 1` yields `UNTESTABLE_QUANTITY_ONE`, never `CONSISTENT` | ✅ branch order covered by test |
| Observed `arithmetic_state` within reach of F6 | ✅ 75.0 / 96.3 / 92.9 vs 78.1 / 92.9 / 91.9; divergence explained in §4.1 |
| No LLM call anywhere in `src/services/facts/` | ✅ |

The acceptance query joins `bp_opportunity → bp_finding_fact →
bp_commercial_fact → bp_fact_provenance` in one statement and returns unit
price, quantity, UoM, currency, contract reference, term, `measure_role`,
`basis_uom`, `arithmetic_state` and `(document_id, page, locator,
verbatim_snippet)` for every contributing fact — with no `->>`, no
`jsonb_extract`, no regex, and no reference to `calculation_details`, asserted
textually against the query string.

The comparability test is the one that matters most: two facts with the
**same** `unit_price` value but roles `unit_rate` and `extended_line` are
distinguished without inspecting the number, and two `unit_rate` facts with
different `basis_uom` are not treated as comparable.

---

## 13. Open items carried forward

1. **`UOM_ABSENT` on 100% of assembled facts.** Every fact on `bp_sqldb` has an
   unstated basis, because the corpus supplies no unit of measure for invoices
   or purchase orders. Cross-supplier rate comparison is therefore **not yet
   safe** on this corpus, and Phase 5 must refuse rather than assume. Fixing it
   properly means extracting UoM, not defaulting it.
2. **FX has no historical dimension** (§8). A dated rate corpus is separate work.
3. **`category_l1..l4` are nullable and unpopulated.** Deliberately out of
   scope so the assembler keeps one responsibility. `bp_category` supplies the
   hierarchy on `bp_sqldb` (49 rows) but is empty on `bp_testdb`, so any future
   population must fail closed rather than default.
4. **Provenance coverage is the binding constraint on fact volume** — 16.9% of
   lines on `bp_sqldb`, 0% on `bp_testdb`. See §14: on the live database the
   cause was not a backlog but a pipeline gap, now fixed.
5. **`bp_constraint` is written by nothing yet.** The corpus contains zero
   contracts, so there is nothing to populate it with. The model and table
   exist; the producer is Phase 4's.
6. **RLS is not enabled.** `tenant_id` is on every new table per the B2
   decision, defaulted to one constant. There is no second tenant and no tenant
   dimension anywhere in `proc`, so a policy would be theatre. Recorded rather
   than faked.
7. **No FK from `bp_finding_fact` to `bp_opportunity`** (§2), because
   `opportunity_ref_id` has no unique index. Adding one would make the
   reference enforceable.
8. **`bp_po_trgt` does not exist** — the surviving header table is
   `bp_po_trgt_june12`. Purchase-order facts therefore carry no supplier, buyer
   or contract id from a header.

**Explicitly not in scope:** the reference corpus and benchmark resolution
(Phase 2); variance decomposition, baseline integrity and the
correlation-adjusted rollup (Phase 3); the Interpretation Plane (Phase 4);
report blocks (Phase 5).

---

## 14. Addendum 2026-08-07 — the live database, and why the phase produced nothing on it

Everything above validates against `bp_sqldb`, per the plan's F2. That was
correct as instructed and **incomplete as an operational picture**, because
`.env` sets `DB_NAME="bp_testdb"`: the backend runs against `bp_testdb`, not
`bp_sqldb`. Three findings follow, all measured.

### 14.1 The live extraction path never wrote line-item provenance

`bp_testdb` held 722 provenance rows across 84 documents, and **not one was a
`line_items` path** — only header fields (`currency`, `quote_date`, totals).
`bp_sqldb` has 32,116 line-provenance rows, written by the older
`v4.0.0-hybrid` pipeline; the live `e9da2f7-renov` path stopped producing them.

The cause is mechanical, not a deliberate decision to discard evidence:

```python
# build_header_record()
for c in candidates:
    if c.field.startswith("line_items["):
        continue          # excluded -- and write_provenance() is fed this output
```

The registry keys line fields as `line_items.<field>` while candidates carry
`line_items[<idx>].<field>`, so `registry.meta()` raises `KeyError` on any line
candidate. Excluding them made the writer work; it also silently dropped the
evidence for every unit price and quantity in the system.

**Consequence:** the fact assembler requires line provenance by construction and
fails closed without it, so Phase 1b produced **zero facts on the live database
— for every document, past and future.** Not a backlog. A structural gap.

**Fixed** (`919945d`): `pick_line_candidates()` selects the highest-confidence
candidate per `line_items[i].field` that the schema binds to a column, and
`write_provenance` now receives header and line picks together. Row-building was
split into `build_provenance_rows()` so the shape is assertable without a
database — the `field_path` it emits is a contract with the assembler's lookup,
and a mismatch there would fail silently.

Verified end to end against a real `bp_testdb` invoice inside a rolled-back
transaction: **0 facts before, 4 after**, each with 4 evidence spans, roles
assigned and arithmetic `consistent`. `tests/extraction/` holds at its 6
pre-existing failures.

### 14.2 The canonical masters are in `uicanvas`, and the live database has none

`uicanvas` is authoritative (confirmed by the owner). It holds the real
reference data; the live database holds almost none of it.

| | `uicanvas` (authoritative) | `bp_testdb` (live) | `bp_sqldb` |
|---|---|---|---|
| Category taxonomy | **246 rows, 5 levels, 100% UNSPSC-coded** | **0** | 49 rows, 2 columns |
| Product master | **186 rows, incl. `unit_of_measure`** | *table absent* | *table absent* |
| Category→product map | 21 | *table absent* | *table absent* |
| Contracts | 3,051 | 0 | 0 |

`bp_sqldb.bp_category` is **not the same table** as `uicanvas.proc.bp_category`
— it is `(item_description, category)`, flat, with no hierarchy and no UNSPSC.

This explains the previously recorded "no category dimension" corpus gap: the
dimension exists, in another database, unwired.

**Resolved 2026-08-07 (`d345b1f`): a read-only `postgres_fdw` bridge, not a copy.**
The owner confirmed `uicanvas` is authoritative and chose an FDW view, so there
is exactly one place the canonical data lives — a copy or a scheduled sync
would immediately raise the question of which side is right when they diverge,
and that question has a cost every time anyone asks it.

Exposed on both databases as `proc.bp_category_master`, `bp_product_master`,
`bp_category_product_map`, `bp_supplier_master`, `bp_contract_master`, over
foreign tables in a `canonical` schema. Three properties, each verified rather
than asserted:

- **Read-only is enforced by the wrapper**, not by convention: the server
  carries `updatable 'false'`, so `UPDATE`/`INSERT`/`DELETE` through the views
  are refused (`foreign table … does not allow updates`). A write reaching
  `uicanvas` from here would corrupt the authoritative copy for every system
  that reads it.
- **`proc.bp_category` is deliberately untouched.** It is a different, flat
  `(item_description, category)` table, and silently replacing it would break
  whatever still reads it.
- **No credentials in the repository** — host, port, user and password are
  passed as psql variables at apply time.

Rollback verified: views and server removed, re-applied, idempotent on a second
run. Guarded by `tests/services/test_canonical_masters.py` (11 live-only tests),
because the failure mode is silent — if the link disappears, queries do not fail
loudly, they quietly find a local empty table with a similar name instead.

### 14.3 B3 must be revised: a coded standard does exist

B3 (§9) concluded *"there is no GPSS dictionary in this project, and there does
not need to be one."* That reasoning was drawn entirely from `bp_sqldb` and
`bp_testdb`, where the canonical data is absent. **`uicanvas.proc.bp_category`
carries a UNSPSC code on all 246 rows** — a genuine external standard.

The revised position:

- **For field names, B3 holds.** `concept_code` derives from the extraction
  schemas, 66 names, drift structurally impossible. Nothing changes.
- **For values, B3 does not hold.** `uicanvas.proc.bp_products` carries a
  `unit_of_measure` on 81 of 186 products across 18 distinct values. Measured
  through the new bridge, `uom.py` covers **63 of 81 products (78%) but only 7
  of 18 distinct values (39%)**:

  | | Values |
  |---|---|
  | mapped | `month` `hour` `unit` `each` `licence` `Unit` `Month` |
  | **rejected** | `set` `pen` `service` `programme` `quarter` `Monthly` `retainer` `module` `sheet` `roll` `audit` |

  The rejections fall into three kinds, and only the first is a plain bug:
  a casing/spelling gap (`Monthly`); genuine missing units (`set`, `sheet`,
  `roll`, `pen`, `module`, `quarter`); and **service-engagement bases**
  (`service`, `programme`, `retainer`, `audit`) which are arguably not units at
  all but lump-sum engagement types — the `extended_line` case in §4, and a
  modelling decision rather than a missing map entry.

So the hand-typed unit map in `uom.py` was already drifting from canonical data
that predates this phase. The `UOM_ABSENT` gap in §13.1 is therefore partly a
*wiring* problem, not only an extraction one: a real unit vocabulary and a real
product master exist and are simply not connected.

Proposed, not built — two tables that **reference** the canonical data rather
than restate it:

- `bp_uom_canonical` — units as data rather than Python: `uom_code`,
  `dimension`, `aliases`, `factor_days`, `is_billing_basis`, and a
  `status` of `proposed`/`active` so a newly observed unit is queued for
  confirmation instead of silently becoming `UOM_UNMAPPED` forever.
- `bp_category_basis` — what a category is normally priced per, keyed on the
  **existing UNSPSC/level id** rather than on category names, with
  `expected_uom`, `alternate_uoms`, `evidence_n` and a human-gated
  `proposed`/`confirmed` status mirroring `bp_supplier_review`.

The prerequisite for both is deciding how canonical data reaches the live
database (FDW view, sync, or repointing). That decision is worth more than
either table.

---

## 15. Addendum — recovering a billing period from the description

Some documents state the deliverable in the unit column and the *period* in the
description: `HR Advisory & Employment Law Retainer (Quarterly)` is billed per
quarter, and `quarter` is a unit the vocabulary already maps. Recovering that
turns an incomparable lump sum into a real rate.

This is inference over free text, which is where fabrication starts, so the
rule is deliberately narrow: **a parenthetical whose entire content is a period
adverb**, bounded to 20 characters. Every other shape in this corpus is a trap,
and each of these is a real string from the data:

| Description | Why parsing it would be wrong |
|---|---|
| `Enterprise Licence - 12 months` | a **term length**; billed per licence |
| `Advanced Package (3 months)` | a **duration**, not a rate |
| `Tier 3 Marketing Services (Months 1-10)` | a **range** |
| `(4 visits per month)` | frequency of **visits**, not of billing |
| `Monthly Design & Marketing Package` | an adjective in the **product's name** |
| `Quarterly Business Review Service` | the deliverable is a QBR; the period is unstated |

Bare period *nouns* are deliberately excluded — only the adverbial forms
(`Monthly`, `Quarterly`, `per quarter`) qualify, because "12 months" and
"4 visits per month" both contain the noun and neither says the price is per
month. Two different periods in one description is a refusal, not a coin toss.

Three rules make it safe:

- **A stated unit is never overridden.** The description is consulted only when
  the unit column fails to resolve. One product row carries
  `unit_of_measure = 'month'` against a description reading `(Quarterly)` — the
  two columns contradict each other, and the column is the more direct
  statement.
- **A rescue does not absolve the column.** If the unit column held something
  that is not a unit, `UOM_UNMAPPED` is still recorded; the value being
  recoverable elsewhere does not make the column right.
- **`BASIS_FROM_DESCRIPTION` is stamped on every derived basis**, so what was
  worked out never becomes indistinguishable from what the page said.

**Measured yield, honestly: 6 of 679 unresolved lines on `bp_sqldb` (0.9%), 2
of 328 on `bp_testdb`.** Every match is correct, and they are all the same two
service lines. This is a precision-first, low-recall rule — nothing else in
this corpus states its period in a parseable parenthetical. Its value is that
it is right, it cannot fabricate, and it generalises to future documents using
the `(Monthly)` / `(Quarterly)` convention.
