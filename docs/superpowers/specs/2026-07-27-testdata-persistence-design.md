# Test Dataset Persistence — Design

**Date:** 2026-07-27
**Status:** approved, not yet implemented
**Parent spec:** `docs/superpowers/specs/2026-07-24-full-scope-test-dataset-design.md`
**Predecessor:** `docs/superpowers/plans/2026-07-24-test-dataset-seeder.md` (Plan 1, shipped)

## 1. Purpose

Plan 1 generates 44,455 documents, 193,568 line items, 400 business units and 500
cost centres, plants 7,070 defects across them, and then discards all of it. Only
suppliers and the ID crosswalk reach a database.

That makes the parent spec's first acceptance criterion false — the build does not
"populate both test databases" — and leaves six blocking verification checks with
nothing to inspect. This design closes that gap.

## 2. The governing principle

**The seeder writes inputs, never outputs.**

A document is a fact: it says what it says, and loading it asserts nothing about
the product. `deal_id`, supplier rankings, quote evaluations, decisions, summaries,
opportunities and detection findings are different in kind — they are conclusions
the product derives. Anything the seeder writes there, no test can afterwards
prove.

This is not a stylistic preference. Test case C1 proves the grouping engine
assembles a deal correctly and C2 proves it does not wrongly merge unrelated
documents. If the seeder stamps `deal_id`, both cases reduce to checking that the
labels we wrote match the labels we read back.

## 3. Scope

### 3.1 In scope

| Data | Destination | Rows |
|---|---|---|
| Quotes | `bp_testdb.proc.bp_quote_trgt` | ~21,020 |
| Purchase orders | `bp_testdb.proc.bp_purchase_order_trgt` | ~5,037 |
| Invoices | `bp_testdb.proc.bp_invoice_trgt` | ~12,398 |
| Quote lines | `bp_testdb.proc.bp_quote_line_items_trgt` | ~115,000 |
| PO lines | `bp_testdb.proc.bp_po_line_items_trgt` | ~28,000 |
| Invoice lines | `bp_testdb.proc.bp_invoice_line_items_trgt` | ~50,000 |
| Requirements | `bp_testdb.proc.bp_requirement` | 6,000 |
| Business units | `uicanvas_test.proc.business_unit` | 400 |
| Cost centres | `uicanvas_test.proc.cost_centre` | 500 |

Line-item counts are approximate because defect planting adds duplicate invoices,
near-duplicates and credit notes after generation.

**Requirements need content the generator does not yet produce.** A `Chain`
carries only a `requirement_id`; `bp_requirement` also wants `title`, `category`,
`description`, `quantity`, `unit`, `target_budget`, `currency`,
`needed_by_date`, `priority` and `status`. These are derived from the chain's own
facts — its awarded quote's category leaf, its cost centre's currency, its
earliest quote date — so the requirement stays consistent with the documents
beneath it rather than being invented independently.

Verification checks implemented: **V04** (line totals sum to document totals) and
**V07** (organisation roll-up balances, per currency — see §5).

### 3.2 Explicitly out of scope

- **The catalogue** (`uicanvas_test.proc.item`). Deferred with V06.
- **FX conversion.** `exchange_rate_to_usd` and `converted_amount_usd` are left
  NULL on all three `_trgt` tables. Deferred with V05.
- **Derived data.** `deal_id`, `bp_supplier_ranking`, `bp_quote_evaluation`,
  `bp_decision`, `bp_action`, `bp_summary`, `bp_analysis_summary`,
  `bp_opportunity`, `bp_detection_finding`. Per §2 these belong to the product.
- **The raw and staging tiers.** Documents are written directly to `_trgt`. The
  32 test cases in areas B–E read `_trgt`; extraction accuracy (area A) is proven
  by the golden document set against real files, not these generated ones.

### 3.3 Deferred checks

V05 (FX re-derivation) and V06 (category path resolution) continue to report
`SKIP`, alongside V08–V13. They must not report `PASS`: a partial run must never
read as a complete one.

## 4. Load design

### 4.1 Declarative column mappings

A new `scripts/testdata/persist.py` holds one table specification per target:
the destination table, its column list, and how each column is derived from the
domain object.

The mapping is the whole difficulty. Nine tables carry 20 to 41 columns each, and
the same concept is named differently across them. A line's net value is
`line_total` on `bp_quote_line_items_trgt` and `bp_po_line_items_trgt` but
`line_amount` on `bp_invoice_line_items_trgt`; the gross value is `total_amount`
on the first two and `total_amount_incl_tax` on the third. Document headers
diverge further — a quote's value is `total_amount`, an invoice's is
`invoice_amount`, and `bp_purchase_order_trgt` carries `supplier_name` alongside
`supplier_id` where the other two carry only the identifier.

Expressing this as data rather than as branching code keeps it reviewable, and
lets a test assert that every mapping names only columns that actually exist on
the target table.

The database will not catch a mapping mistake for us. Of the nine target tables,
seven have **no NOT NULL constraint at all**; `bp_requirement` requires only
`requirement_id` and `status`, `business_unit` only `business_unit_id`, and
`cost_centre` only `cost_centre_level_id`. A load that silently wrote NULL into
`invoice_amount` for all 12,398 invoices would be accepted without complaint.

Each table specification therefore declares its own **required set** — the
columns that must be non-NULL for the data to mean anything — and a test asserts
every loaded row satisfies it. That set is a deliberate choice recorded in the
spec for each table, not an inference from the schema.

### 4.2 Load order

Reference data → business units → cost centres → requirements → quotes →
purchase orders → invoices → line items.

Each table is truncated and then bulk-loaded through the existing `COPY FROM
STDIN` path in `db.copy_rows`, so re-running the build is idempotent.

### 4.3 Triggers

`bp_invoice_trgt`, `bp_purchase_order_trgt` and `bp_quote_trgt` each carry an
outcome trigger that looks up a matching row in the corresponding `bp_*_raw`
table and records a `session_document_outcome`. Because the raw tier is out of
scope, the lookup finds nothing and the trigger returns early. This is correct,
not accidental: there was no ingestion session, so there is no outcome to record.

## 5. Roll-up and currency

Each entity trades in one currency (UK GBP, US USD, DE and IE EUR, IN INR,
AE AED), and every cost centre inherits its entity's currency. So the roll-up
cost centre → business unit → entity is single-currency and must balance to the
penny.

The **group** total spans five currencies. Adding them requires FX, which is out
of scope, so V07 verifies the roll-up per currency up to entity level and stops
there. The cross-currency group total is deferred with V05 and named as such in
the check's own detail string, not silently omitted.

## 6. deal_id

Documents are loaded with `deal_id`, `deal_name` and `deal_date` NULL. The build
then invokes the product's own `src/services/deal_assignment_service.py` over the
seeded documents.

The correct grouping is known to the seeder — one chain is one deal — and is
written to the answer key as expected truth, so C1 and C2 can compare the
product's output against it.

**Open risk.** It is not yet established that `deal_assignment_service` can run in
batch against an arbitrary target database rather than assuming `bp_sqldb` and a
running scheduler. The implementation plan must verify this before depending on
it. If it cannot, the fallback is to leave `deal_id` NULL, record V08 (screen
queries return non-empty) as blocked with the reason named, and resolve it in
Plan 3. The fallback must not be to stamp the identifiers.

## 7. Testing

- **Unit, no database:** every mapping produces its declared column count; row
  builders emit correct types; the roll-up arithmetic balances on a fixture.
- **Integration, scratch database only:** every mapped column exists on its
  target table; every loaded row satisfies that table's declared required set
  (the schema will not enforce this — see §4.1); row counts match; V04 and V07
  pass; re-running the load is idempotent.

Integration tests target `bp_testdb_it` / `uicanvas_test_it`. They must never
open a connection to `bp_testdb` or `uicanvas_test`; `test_scratch_isolation.py`
enforces this, after an earlier `drop_first=True` clone in the test suite silently
destroyed a completed build.

## 8. Acceptance criteria

1. `python -m scripts.testdata.build --target bp_testdb --seed 42` loads every
   table in §3.1 and reports its row counts.
2. V04 passes: on every document not carrying a planted arithmetic defect, the
   line totals sum to the document total.
3. V07 passes per currency: cost centre → business unit → entity balances to the
   penny.
4. V05 and V06 still report `SKIP`, never `PASS`.
5. Re-running the build produces identical row counts and content.
6. `bp_sqldb` and `uicanvas` remain provably unchanged (V14).
7. `deal_id` is either populated by the product's own service, or NULL with the
   reason recorded. It is never stamped by the seeder.
