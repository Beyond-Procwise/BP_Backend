# Test Dataset Build Log

**Date:** 2026-07-27
**Seed:** 42
**Targets:** bp_testdb, uicanvas_test (same cluster as live, separate databases)
**Command:** `.venv/bin/python -m scripts.testdata.build --target bp_testdb --uicanvas-target uicanvas_test --seed 42 --drop-first`

## Result

| | |
|---|---|
| Exit code | 0 |
| Suppliers | 5,000 (both ID conventions, 5,000 crosswalk rows) |
| Business units | 400 under a 6/40/120/240 tree |
| Cost centres | 500 |
| Catalogue items | 5,000 across 246 L5 leaves |
| Documents generated | 44,455 (6,000 requirements, 21,020 quotes, 5,037 POs, 12,398 invoices) |
| Line items generated | 193,568 |
| Defect instances planted | 7,070 (5,610 true positive, 1,460 negative control) |
| Defect types with zero instances | none — all 30 planted |
| Defect types off their declared target | none — all 30 exact |
| Blocking checks passed | 3/3 implemented, 9 not yet implemented |
| Scored checks | V10 not yet implemented, V12 not yet implemented |

### Verification output

```
V01 PASS  proc.bp_supplier has 5000 rows (expected 5000)
V02 PASS  0 orphan invoice line items
V03 PASS  crosswalk 5000, bp_supplier 5000, uicanvas supplier 5000
V04 SKIP  not yet implemented
V05 SKIP  not yet implemented
V06 SKIP  not yet implemented
V07 SKIP  not yet implemented
V08 SKIP  not yet implemented
V09 SKIP  not yet implemented
V10 SKIP  not yet implemented
V11 SKIP  not yet implemented
V12 SKIP  not yet implemented
V13 SKIP  not yet implemented
V14 PASS  live row counts unchanged
```

Checks with no implementation print `SKIP`, not `PASS`. A partial run must not
read as a complete one.

### Structure cloned

| | bp_testdb | uicanvas_test |
|---|---|---|
| Tables | 118 | cloned from live |
| Views | 5 | cloned from live |
| Functions | 10 | cloned from live |
| Triggers | 8 | cloned from live |

### Reference data copied verbatim

`bp_fx_rates` 830, `bp_policy` 10, `bp_prompt` 12, `bp_admin_config` 2,
`bp_vendor_extraction_profiles` 29, `bp_complaince_metric_prty_lkup` 9,
`procurement_patterns` 33, `bp_category` 246 (the real 5-level taxonomy),
`category` 932.

## Isolation

```
Live row counts before: 308 tables, 405,032 rows
Live row counts after:  308 tables, 405,032 rows
Result: UNCHANGED
```

Proven twice: by check V14 inside the run, and by an independent snapshot
comparison afterwards against `live_before.json`.

## Determinism

Two builds with seed 42, in **separate processes**, produced checksum
`bb000e8d13778a552cb06b866c10c7d1305d56dc94b13cc17b2b6b6303995228`.

## Known gaps

These are real and deliberate, not oversights:

1. **Only suppliers and the crosswalk are persisted.** The 44,455 documents,
   193,568 line items, 400 business units and 500 cost centres are generated,
   defect-planted and checksummed in memory, then discarded. Plan 1 specifies no
   persistence for them — `deals.py` and `downstream.py` appear in the file
   structure with no task attached, and the plan folds them into Plan 3. Until
   they land, the test databases hold structure, reference data and suppliers
   only.
2. **V04–V13 have no implementation.** They inspect data that Plans 2 and 3
   produce. They report `SKIP` and do not block.
3. **Population defects are recorded, not manufactured.** D10–D21, D24–D26 and
   D28–D30 name the subjects that constitute the expected set, but the
   underlying patterns (price spread, single-source concentration, expired
   certificates) are not yet forced into the data. Plan 3 has to make them
   genuinely hold before a detector can find them. The plan calls this out as a
   known rough edge and it remains one.

## Deviations from the plan

Four, all fixing defects found while executing it:

1. **`defects.plant()` looked positions up with `chains.index()`.** Every
   planting block replaces the chain it touches with a new frozen copy, so from
   the second block onward the object being searched for was no longer in the
   list and `list.index()` raised `ValueError`. Now keyed by `requirement_id`,
   which also removes an O(n²) scan.
2. **D04 would have planted nothing.** D22 nulls the quantity on line 1 of the
   leading chains; D04 only looked at line 1 and skipped on `None`. D04 now runs
   before D22 and targets the first line that still carries a quantity.
3. **The COPY loader rejected `jsonb` and array columns.** psycopg2 decodes
   `jsonb` to a Python dict, whose `str()` is Python repr rather than JSON, and
   `int[]` to a list, whose `str()` is not an array literal. json now reads back
   as raw server text — which is what "verbatim" should mean — and arrays render
   as properly quoted Postgres literals.
4. **`build.py` never wrote `uicanvas.proc.supplier`.** The crosswalk would have
   named 5,000 suppliers on a side holding none, failing blocking check V03. The
   table is column-identical to `bp_supplier`, so the same rows are written with
   the identifier swapped.
5. **D05 overshot its declared target by 2×.** It capped on chains but recorded
   one instance per invoice, and a chain carries one to three — 1,290 planted
   against a target of 620 on the first build. It now caps on the instances it
   records. A regression test asserts no defect type exceeds its declared count,
   sized large enough to actually reach the cap; the 1,200-chain fixture
   exhausts the no-PO pool first and hides the fault.

---

# Stage S1 — Organisation and Catalogue

**Date:** 2026-07-27
**Plan:** `docs/superpowers/plans/2026-07-27-testdata-s1-organisation-catalogue.md`
**Command:** `.venv/bin/python -m scripts.testdata.build --target bp_testdb --uicanvas-target uicanvas_test --seed 42 --drop-first`

## Result

| | |
|---|---|
| Exit code | 0 |
| `uicanvas_test.proc.business_unit` | 400 rows (16 columns) — live holds 0 |
| `uicanvas_test.proc.cost_centre` | 500 rows (26 columns) |
| `uicanvas_test.proc.item` | 5,000 rows (15 columns) |
| Blocking checks | V01, V02, V03, V07, V14 PASS |
| V07 detail | `500 cost centres over 400 business units, 0 unresolved; entity and group levels not verifiable (no entity column in the schema)` |
| Tests | 221 pass |

## Isolation

```
Live row counts before: 313 tables, 405,805 rows
Live row counts after:  313 tables, 405,805 rows
Result: UNCHANGED
```

## Two schema facts this stage had to respect

1. **There is no entity or organisation table.** Neither `business_unit` nor
   `cost_centre` carries an entity column, so the entity and group levels of the
   roll-up cannot be verified in the database. V07 checks what the schema can
   express — every cost centre resolves to a business unit that exists — and its
   detail string names the gap rather than reporting a pass for a narrower
   question than the one asked.

2. **`category_level_5_id` is not unique** — 121 distinct values across 246
   leaves, because it identifies a node within its branch rather than a leaf
   globally. Joining `item` to `bp_category` on it returns 11,752 rows for 5,000
   items. That fan-out is the data's shape, not duplicate items.

## Bug fixed on the way

`org.py` wrote the **UNSPSC code** into `cost_centre.linked_category_level_5_id`.
The column wants `bp_category.category_level_5_id` (`C-5101`-style). `TaxonomyLeaf`
now carries all five level identifiers, which the catalogue also needs so a line
can resolve to a category path.

## Deferred by design

- `cost_centre.po_id` and `invoice_id` stay NULL until stage S3 has documents to
  point at.
- `item.manufacturer`, `brand`, `spec_sheet_url` and `uom_conversion` stay NULL:
  the generator does not model them, and inventing values would put unverifiable
  strings in columns nothing reads.
- V04, V05, V06 and V08–V13 still report `SKIP`, never `PASS`.

## Note on parallel work

A second session committed `scripts/testdata/persist.py` — the document mapping
for stage S3's six `_trgt` tables — while this plan was being written. S1 adapted
to it rather than replacing it: `persist_org.py` is a sibling in the same idiom,
and `loader.py` is shared by both.

---

# Stage S2 — Supplier Master and Reference Data

**Date:** 2026-07-27

## Result

| Table | Rows | Was |
|---|---|---|
| `uicanvas_test.proc.bp_supplier` | 5,000 (51 columns) | 0 |
| `uicanvas_test.proc.esg_data` | 5,000 (15 columns) | 0 — empty in live too |
| `uicanvas_test.proc.contact` | 5,000 (14 columns) | 0 — empty in live too |
| `uicanvas_test.proc.bp_contact` | 5,000 (14 columns) | 0 — empty in live too |
| `bp_testdb.proc.bp_tprm_supplier` | 120 (10 columns) | 0 |

Exit code 0. Blocking checks V01, V02, V03, V07, V14 PASS. 238 tests pass.
Live unchanged at 313 tables / 405,805 rows.

## Decisions

**Third-party risk covers the Strategic tier only** — 120 of 5,000 suppliers.
A TPRM register is maintained for suppliers that warrant the effort; filling all
5,000 would misrepresent how it is used.

**ESG figures derive from the supplier's own master record.** A supplier holding
ISO 14001 scores 62–94 and reports 45–95% renewable energy; one without scores
28–70 and 5–55%. Drawing them independently would have produced suppliers that
are simultaneously certified and worst-in-class. `carbon_emission_tco2` is the
sum of its three scopes, and a test asserts it.

**Contacts come from the master record**, not invented separately, so
`contact.contact_email` equals the supplier's `contact_email_1`.

**`supplier_risk_scores` was reclassified from SEED to OUTPUT.** It carries
`score`, `model_version`, `feature_summary` and `computed_at` — that is a model's
conclusion, not reference data about the supplier. Seeding it would have left the
scoring model untestable. This is the correction the classification tests exist
to force.

## Still excluded, and why

`bp_supplier_ranking`, `bp_supplier_review`, `bp_supplier_enrichment`,
`bp_supplier_alias`, `bp_supplier_name_reject`, `supplier_risk_signals` and
`supplier_risk_scores` are all conclusions the product reaches. Seeding any of
them would make the agent that produces it untestable.

`_write_suppliers` still writes `bp_supplier` and `supplier` through its own path
rather than the staged loader, because it carries bespoke identifier-swap logic
and is already covered by V01 and V03. It does not get the required-set check.

---

# Stage S3 (part 1) — Documents and Requirements in bp_sqldb

**Date:** 2026-07-27

## Result

| Table | Rows |
|---|---|
| `bp_requirement` | 6,000 |
| `bp_quote_trgt` | 21,020 |
| `bp_quote_line_items_trgt` | 115,610 |
| `bp_purchase_order_trgt` | 5,037 |
| `bp_po_line_items_trgt` | 22,537 |
| `bp_invoice_trgt` | 12,398 |
| `bp_invoice_line_items_trgt` | 55,421 |
| **Total** | **238,023** |

Exit code 0. 247 tests pass. 28 tables now hold data; 277,383 rows across both
test databases.

**V04 now passes** — it was `SKIP`: `0 documents whose lines do not sum to their
total; 417 exempted as planted arithmetic defects`. The exemption set is built
from the answer key (D03, D04, D06, D20), so the check distinguishes a correct
loader from one that simply found no defects.

## A CHECK constraint caught what the required-set check could not

`bp_requirement.status` carries `CHECK (status IN ('draft','gathering',
'complete','handed_off','abandoned'))`. The first mapping invented `awarded` and
`open`, which are reasonable English and wrong. The load failed outright.

The required-set check cannot catch this — the value was present, just not in the
schema's vocabulary. A test now reads the constraint definition from live and
asserts the declared statuses match it, so the mapping fails loudly if the
constraint changes rather than at the next full build.

Requirements map to `handed_off` when a purchase order was raised and `complete`
otherwise.

## Isolation

The build's own V14 passed: live row counts were identical immediately before and
after the run.

A wider comparison against the baseline captured at the start of S1 showed one
difference — `bp_sqldb.proc.bp_prompt` 12 → 13. That row is `prompt_id 100,
'style_draft_system'`, inserted by a concurrent session's response-style work, not
by this build. It is recorded here rather than explained away: the isolation
guarantee covers what the build does, and a cross-session comparison is a
different question from the one V14 answers.

## Still outstanding in S3

The `raw` and `stg` tiers (12 tables) and `raw_invoice` / `raw_purchase_order` /
`raw_quotes` are mapped by no module yet. Until they land, the outcome triggers on
the three `_trgt` tables find no matching `raw` row and return early, so
`session_document_outcome` stays empty.
