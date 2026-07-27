# Test Dataset Persistence — Design

**Date:** 2026-07-27
**Status:** approved, not yet implemented
**Parent spec:** `docs/superpowers/specs/2026-07-24-full-scope-test-dataset-design.md`
**Predecessor:** `docs/superpowers/plans/2026-07-24-test-dataset-seeder.md` (Plan 1, shipped)
**Coverage evidence:** `docs/testdata/BP_Schema_Coverage.xlsx` (generated from live)

## 1. Purpose

Plan 1 generates 44,455 documents, 193,568 line items, 400 business units and 500
cost centres, plants 7,070 defects across them, and then discards all of it. Only
suppliers and the ID crosswalk reach a database.

That makes the parent spec's first acceptance criterion false — the build does not
"populate both test databases" — and leaves six blocking verification checks with
nothing to inspect. This design closes that gap across the whole schema, not a
chosen subset.

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

## 3. Scope: every table, classified

`scripts/testdata/coverage.py` assigns all 293 live tables to exactly one bucket
by explicit rule. The classification is under test, and an integration test fails
if a table appears in live with no classification — so a new production table
becomes a decision, never a silent omission.

| Bucket | Tables | Treatment |
|---|---|---|
| **SEED** | 68 | Business data the seeder synthesises |
| **REFERENCE** | 30 | Configuration copied verbatim from live |
| **OUTPUT** | 113 | Left empty — the product derives it (§2) |
| **BACKUP** | 82 | Left empty — dated or duplicated copies |

98 of 293 tables are filled. The 195 left empty are empty *by argument*, and the
workbook records the argument per table.

### 3.1 SEED — what gets written

**Documents, all three tiers.** `bp_sqldb` carries `raw → _stg → _trgt` for
quotes, purchase orders and invoices, plus the matching line-item tables — 18
tables. Writing only `_trgt` would leave the promotion path untested and the
`_raw` lookup in each outcome trigger permanently empty. `uicanvas` holds its own
document tables (`bp_invoice`, `bp_invoice_trgt`, `invoice`, `purchase_order`,
`po_line_items` and others) which the SpendIQ screens read.

> This reverses an earlier decision to write `_trgt` only. Full-schema coverage
> was chosen after the coverage workbook showed what a target-only load omits.

**Organisation.** `business_unit` (16 columns, empty in live) and `cost_centre`
(26 columns, 500 placeholder rows in live).

**Supplier master and its reference data.** `bp_supplier` and `supplier` are
already loaded. This design adds the supplier tables that are *inputs the product
reads* rather than conclusions it writes: `bp_tprm_supplier`,
`supplier_risk_scores`, `esg_data`, `contact` and `bp_contact`.

`bp_supplier_ranking`, `bp_supplier_review`, `bp_supplier_enrichment`,
`bp_supplier_alias` and `bp_supplier_name_reject` stay empty: each is a conclusion
the product reaches. `supplier_risk_signals` likewise — signals are detected, and
scores are supplied.

**Catalogue.** `item` (15 columns), carrying the 5,000 generated items with their
`category_id` resolved against the real taxonomy. This is what lets V06 resolve a
line to an L1–L5 path.

**Contracts.** `bp_contracts`, `contract`, `contracts`, `bp_contract_raw` and the
`raw_*` landing tables. The corpus has no contracts today, so obligation
extraction has never been exercised against data.

**Cross-reference maps.** `sup_mapping`, `po_mapping`, `po_supplier_mapping`,
`old_new_supplier_mapping`, `contract_id`, `inv_mapping_inv_itms_new` and
`item_mapping_inv_itms_new` join the two databases' differing identifier
conventions.

### 3.2 REFERENCE — copied verbatim

The 7 tables Plan 1 already copies, plus `bp_style_profile`, `bp_style_intent`,
`bp_style_exemplar`, `bp_style_ingest_staging` and `bp_mailbox_binding` (response
style and mailbox configuration steer product behaviour, so the test databases
must carry production's values), `bp_products`, `bp_category_product_mapping`,
`cat_product_mapping`, `policy`, `static_policy`, `prompt`, `pricing`,
`pricing_matrix_ranking_policy`, `quote_weighting`, `quote_values` and
`vendor_profile`.

### 3.3 Still out of scope

- **FX conversion of document totals.** Line prices are now converted into the
  cost-centre currency, but `exchange_rate_to_usd` and `converted_amount_usd` on
  the `_trgt` headers remain NULL. V05 continues to report `SKIP`.
- **Derived data**, per §2.

## 4. Load design

### 4.1 Declarative column mappings

A new `scripts/testdata/persist.py` holds one table specification per target: the
destination table, its column list, and how each column is derived from the domain
object.

The mapping is the whole difficulty. The 68 seed tables carry roughly 1,270
columns between them, and the same concept is named differently across them. A
line's net value is `line_total` on `bp_quote_line_items_trgt` and
`bp_po_line_items_trgt` but `line_amount` on `bp_invoice_line_items_trgt`; the
gross value is `total_amount` on the first two and `total_amount_incl_tax` on the
third. Document headers diverge further — a quote's value is `total_amount`, an
invoice's is `invoice_amount`, and `bp_purchase_order_trgt` carries
`supplier_name` alongside `supplier_id` where the other two carry only the
identifier.

Expressing this as data rather than as branching code keeps it reviewable, and
lets a test assert that every mapping names only columns that actually exist on
the target table.

**The database will not catch a mapping mistake.** Of the document tables, none
has a NOT NULL constraint; `bp_requirement` requires only `requirement_id` and
`status`, `business_unit` only `business_unit_id`, `cost_centre` only
`cost_centre_level_id`. A load that silently wrote NULL into `invoice_amount` for
all 12,398 invoices would be accepted without complaint.

Each table specification therefore declares its own **required set** — the columns
that must be non-NULL for the data to mean anything — and a test asserts every
loaded row satisfies it. That set is a deliberate choice recorded per table, not
an inference from the schema.

### 4.2 Load order

Reference data → organisation → suppliers and supplier reference data →
catalogue → contracts → requirements → quotes → purchase orders → invoices →
line items → cross-reference maps.

Each table is truncated then bulk-loaded through `db.copy_rows`, so re-running the
build is idempotent.

### 4.3 Three tiers

`raw` holds the document as extracted, `_stg` the cleaned form, `_trgt` the
promoted record. The seeder writes all three with consistent content: there is no
extraction step here, so `raw` is not a degraded version — it is the same
document, carrying `source_file` and `doc_pk_candidate` so the outcome triggers
on the `_trgt` tables resolve rather than silently no-op.

### 4.4 Requirements need content the generator does not produce

A `Chain` carries only a `requirement_id`; `bp_requirement` also wants `title`,
`category`, `description`, `quantity`, `unit`, `target_budget`, `currency`,
`needed_by_date`, `priority` and `status`. These are derived from the chain's own
facts — its awarded quote's category leaf, its cost centre's currency, its
earliest quote date — so the requirement stays consistent with the documents
beneath it rather than being invented independently.

## 5. Roll-up and currency

Each entity trades in one currency (UK GBP, US USD, DE and IE EUR, IN INR,
AE AED), and every cost centre inherits its entity's currency, so the roll-up
cost centre → business unit → entity is single-currency and must balance to the
penny.

The **group** total spans five currencies. Adding them requires converting the
document headers, which §3.3 defers, so V07 verifies the roll-up per currency up
to entity level and stops there. The cross-currency group total is deferred with
V05 and named in the check's own detail string, not silently omitted.

## 6. deal_id

Documents are loaded with `deal_id`, `deal_name` and `deal_date` NULL. The build
then invokes the product's own `src/services/deal_assignment_service.py` over the
seeded documents. The correct grouping is known to the seeder — one chain is one
deal — and is written to the answer key as expected truth, so C1 and C2 can
compare the product's output against it.

**Open risk.** It is not established that `deal_assignment_service` can run in
batch against an arbitrary target database rather than assuming `bp_sqldb` and a
running scheduler. The implementation plan must verify this before depending on
it. If it cannot, the fallback is to leave `deal_id` NULL, record V08 as blocked
with the reason named, and resolve it in Plan 3. The fallback must not be to stamp
the identifiers.

## 7. Testing

- **Unit, no database:** every mapping produces its declared column count; row
  builders emit correct types; the roll-up arithmetic balances on a fixture; the
  table classification is exhaustive.
- **Integration, scratch database only:** every mapped column exists on its target
  table; every loaded row satisfies that table's declared required set (the schema
  will not enforce this — §4.1); row counts match; V04 and V07 pass; re-running
  the load is idempotent; no live table is left unclassified.

Integration tests target `bp_testdb_it` / `uicanvas_test_it` and must never open a
connection to `bp_testdb` or `uicanvas_test`. `test_scratch_isolation.py` enforces
this, after a `drop_first=True` clone in the suite silently destroyed a completed
build.

## 8. Acceptance criteria

1. `python -m scripts.testdata.build --target bp_testdb --seed 42` loads every
   SEED and REFERENCE table and reports its row count.
2. No table classified SEED or REFERENCE is left at zero rows unless its live
   source is also empty.
3. V04 passes: on every document not carrying a planted arithmetic defect, the
   line totals sum to the document total.
4. V07 passes per currency: cost centre → business unit → entity balances to the
   penny.
5. V05 still reports `SKIP`, never `PASS`.
6. Re-running the build produces identical row counts and content.
7. `bp_sqldb` and `uicanvas` remain provably unchanged (V14).
8. `deal_id` is either populated by the product's own service, or NULL with the
   reason recorded. It is never stamped by the seeder.
9. The coverage workbook regenerates and shows no table in the UNCLEAR bucket.
