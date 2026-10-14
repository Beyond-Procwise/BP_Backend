# Reseller Catalog, Cost and Sell-Side Model — Design Spec

**Date:** 2026-09-09
**Branch target:** `Development`
**Status:** Draft for review
**Scope:** Gaps 1–3 of the reseller capability audit (2026-09-09). Gaps 4–10 are out of scope and named as such in §8.
**DDL:** `deploy/sql/2026-09-11_bp_catalog.sql`, `deploy/sql/2026-09-11_bp_sell_side.sql`

---

## 1. Problem

A reseller account manager asked whether this platform can review a distributor catalog against a customer's purchase history and produce quotable upsell opportunities. The audit found the file-reading layer, the purchase-history model and the benchmark pricing engine all fit the job, and three things block it completely:

1. **There is no catalog.** `proc.bp_product_master` holds 186 rows and 12 columns — `product_id`, `item_description`, `current_unit_price`, `currency`, `unit_of_measure`, timestamps, `source_doc_type/id`, `occurrence_count`. It is a by-product of extraction, not a catalog. Of the six fields a distributor feed carries, it holds one, and that one is the last price observed on a document rather than a list price.
2. **There is no cost.** A search of every column in the `proc` schema for `cost`, `margin`, `list_price`, `stock`, `availab`, `lifecycle`, `eol`, `replace`, `sku`, `mpn`, `manufactur` returns nothing. Margin is therefore not computable in any form.
3. **There is no sell side.** Every one of the 21,049 `bp_quote_trgt` rows is a quote a supplier sent *us*. `bp_opportunity`'s value axis is `financial_impact_gbp` / `realised_savings_gbp` — money not spent. The UI already ships a complete Sales mode (`src/config/processModes/sales.js`) and a seeded three-phase ladder (`src/lib/processTaxonomy/salesLifecycle.js`), and the backend has no table, column or endpoint behind any of it.

## 2. The governing idea

> **A catalog is asserted. A document is read. They must not share a pipeline.**

The extraction stack exists to answer one question: *did the model read this document correctly?* That question has a confidence, a grounding guard, a judge and a promotion gate, and all of it is warranted, because a PDF is evidence and a model's reading of it is a claim.

A column in a distributor's price file is not a claim. The distributor is telling us their cost. There is nothing to ground it against and no second opinion to weigh. Putting a catalog through `raw → _stg → _trgt` would attach a confidence score to a fact that has none, and would spend GPU minutes per row on a 5,000-row file to do it.

So the catalog gets a **deterministic column-mapping importer** and its own tables. The parser it reuses already returns structured cells — `spreadsheet_backend.py` builds `Table(rows=table_cells)` alongside the markdown rendering — so the reader is a mapping, not a model.

Three consequences follow, and they shape the DDL:

- **The map is a row a human owns**, not a heuristic. `bp_catalog_mapping` records which of a distributor's headings feeds which of our columns. An unmapped required column rejects the import; it never guesses.
- **Absence stays absent.** Every commercial column is nullable. A feed with no stock figures produces NULL, not 0; no lifecycle column produces NULL, not `active`.
- **A price is a version, not an update.** See §4.

## 3. Goals

- A distributor catalog with cost, list price, category, pack size, availability, lifecycle and distributor-asserted succession, versioned over time.
- Our cost, held at flat and volume-break granularity, and structurally incapable of reaching a customer-facing surface.
- Accounts, sell-side opportunities and outbound quotes, on the phase vocabulary the UI already seeds.
- Every number on a quote reconstructable months later exactly as it was sent.

## 4. Design decisions

### 4.1 The catalog is versioned, not updated

A distributor reprices monthly. If `cost_price` were updated in place, a quote sent in March would report April's margin, and the number we defended to a customer would stop being the number on their desk. A price change therefore closes the current row (`valid_to = now()`) and opens a new one — the same bitemporal shape `bp_uom_canonical` and `bp_fact_provenance` already use.

A partial unique index, not a plain `UNIQUE`, enforces one *current* version per SKU:

```sql
CREATE UNIQUE INDEX ux_bp_catalog_item_current
    ON proc.bp_catalog_item (distributor_id, distributor_sku)
    WHERE valid_to IS NULL;
```

### 4.2 Cost and price are snapshotted onto the quote line

Versioning makes a live join *possible*. Taking it would still be wrong. `bp_sales_quote_line.unit_cost` and `list_price_at_quote` are copied at draft time and never re-read. The catalog version is the audit trail; the line is the record of what was quoted.

`cost_tier_applied` stores which volume break produced the cost, so a margin can be re-derived by hand — the same standard the benchmark engine already holds itself to, where every intermediate is an output field.

### 4.3 Cost is an internal field, and the schema says so

Every cost and margin column in `sql/bp_sell_side.sql` is marked `INTERNAL` in the DDL comment: `bp_sales_quote.total_cost`, `total_margin`, `margin_pct`, and the four `unit_cost` / `line_margin` columns on the line. `bp_sales_justification.customer_safe` is the enforcement point — a justification row with `customer_safe = FALSE` never renders in a customer-facing artifact.

This is a schema convention, not a control. The control is a serialiser allowlist in the quote-rendering service, which this spec requires and does not design (§7, step 5).

### 4.4 An account is not a supplier

The same legal entity is frequently both — we buy from a distributor that also buys from us. They still do not belong in one table. A `bp_supplier` row carries bank details we pay *into* and a risk score about *their* delivery; a `bp_account` row carries credit we extend and a probability *they* buy. Merging them makes both ambiguous. `bp_account.also_supplier_id` records the overlap without erasing the distinction.

### 4.5 Win probability is NULL until it is measured

`bp_sales_opportunity.win_probability` is nullable with a companion `win_probability_basis` (`calibrated` | `manual` | NULL). An uncalibrated 0.5 is indistinguishable from a measured one once it is in a column, and the audit's ranking requirement is worthless if the third factor is decoration.

This is why `bp_sales_quote_outcome` is **in** this gap rather than deferred to audit gap 8: it is the only thing that can ever populate that column. Ship the model without it and `win_probability` stays NULL by construction, permanently.

### 4.6 Matching is a claim, so it gets its own table

`bp_catalog_item_match` is separate from `bp_catalog_item` because a SKU-to-history match has a method and a confidence and can be wrong in a way a catalog row cannot. It carries `status` (`proposed` | `confirmed` | `rejected`) and follows the pattern already established for supplier entity resolution: a fuzzy match is proposed for human confirmation, never silently applied.

`match_method` is ordered by trustworthiness — `mpn_exact`, `sku_exact`, `description_fuzzy`, `human` — and `confidence` is NULL for the exact and human methods, because a number there would imply a judgement that was not made.

### 4.7 History scope is recorded, because absence is ambiguous

A reseller sees their own invoices to a customer completely, and that customer's spend with everyone else only if the customer shared it. `bp_account_history_scope` records which is which per account.

Without it, "this account buys nothing from us in networking" and "this account buys nothing in networking" are the same query result — and the first is an opportunity while the second is a dead end.

## 5. Prerequisites, and one thing that cannot be fixed

**`proc.bp_supplier`** is a local base table with 5,028 rows and 5,028 distinct `supplier_id`, and no declared key. `sql/bp_catalog.sql` adds `PRIMARY KEY (supplier_id)`, guarded in a `DO` block because `ADD CONSTRAINT` has no `IF NOT EXISTS` and both files must re-apply cleanly. Two references depend on it: `bp_catalog_source.distributor_id` and `bp_account.also_supplier_id`, both real foreign keys.

**The category reference can never be a foreign key.** `proc.bp_category_master` is a *view* over `canonical.bp_category`, which is a *foreign table* reaching another database through the `uicanvas_srv` wrapper. Postgres cannot declare a foreign key to a foreign table, and a constraint added on the far side would not be enforced here.

So `bp_catalog_item.unspsc_code` is a soft reference, and the importer must validate it at load time against the 246 live `unspsc_code` values and reject the row when it does not match. That validation is a build requirement, not an optimisation — without it the column silently accepts anything.

Two supporting facts, both verified: `unspsc_code` carries the reference because `category_level_5_id` is **not unique** in that view (121 distinct values across 246 rows) and could not carry it even if the plumbing allowed; and `proc.bp_product_master` is a view over the same wrapper, which is a second reason the new catalog is its own local table rather than an extension of the existing one.

## 6. What the DDL delivers

**`deploy/sql/2026-09-11_bp_catalog.sql`** — gaps 1 and 2.

| Table | Purpose |
|---|---|
| `bp_catalog_source` | One row per ingested feed. `content_sha256` makes a re-send idempotent; `status` / `rows_rejected` stop a failed import from looking like an empty catalog. |
| `bp_catalog_mapping` | Human-owned column map per distributor profile. |
| `bp_catalog_item` | The catalog. Identity (`distributor_sku`, `mpn`, `manufacturer`, `brand`), classification (`unspsc_code`), unit economics (`unit_of_measure`, `pack_size`, `pack_uom`, `currency`), **`list_price` and `cost_price` + `cost_basis`**, availability (`availability_status`, `stock_qty`, `lead_time_days`), lifecycle (`lifecycle_status`, `end_of_sale_date`, `end_of_life_date`), bitemporal validity. |
| `bp_catalog_cost_tier` | Volume-break cost. The commonest distributor structure there is, and the one that changes margin at exactly the quantities a large-value quote turns on. |
| `bp_catalog_item_relation` | `replaced_by`, `upgrade_of`, `refill_of`, `accessory_of`, `requires` — **distributor-asserted only**. |
| `bp_catalog_item_match` | Catalog SKU ↔ purchase-history `item_id`, proposed / confirmed / rejected. |

**`deploy/sql/2026-09-11_bp_sell_side.sql`** — gap 3.

| Table | Purpose |
|---|---|
| `bp_account` | The customer. `also_supplier_id` for the overlap. |
| `bp_account_contact` | Named contacts, `is_primary`. Replaces `bp_supplier`'s two-contact-column pattern. |
| `bp_account_history_scope` | What we can see of this account's spend, and how completely. |
| `bp_sales_opportunity` | Account × catalog item. `opportunity_type`, native `currency`, revenue / cost / **margin**, `win_probability` + basis, seeded `phase_id` / `subprocess_id`, `outcome`. |
| `bp_sales_justification` | Evidence rows, `customer_safe` gated. |
| `bp_sales_quote` | The outbound quote. Stored totals, `valid_until NOT NULL`, `supersedes_id`, approval fields written only from the authenticated token. |
| `bp_sales_quote_line` | Snapshot cost and list price, `cost_tier_applied`, per-line `currency`. |
| `bp_sales_quote_outcome` | Won / lost + reason + competitor. Calibrates `win_probability`. |

### Two corpus faults these tables fix by construction

- **Line-level currency.** `currency` is populated on **0 of 55,483** existing invoice line items; it lives only on the header. `bp_sales_quote_line.currency` and `bp_sales_opportunity.currency` are `NOT NULL`.
- **Pack reconciliation.** `pack_size` / `pack_uom` are what let a catalog "box of 10" reconcile with a history line of "10 each". `unit_of_measure` soft-references `bp_uom_canonical.uom_code` rather than storing a fresh literal vocabulary.

## 7. Build order

1. **Prerequisites** (§5) — the `bp_supplier` primary key, plus the importer-side `unspsc_code` validation that stands in for the foreign key the FDW makes impossible.
2. **`sql/bp_catalog.sql`**, then a `CatalogImportService`: parse via the existing `spreadsheet_backend` `Table` cells → apply `bp_catalog_mapping` → version rows into `bp_catalog_item` → write the `bp_catalog_source` receipt. Deterministic; no model call. **Delivered** as `src/services/catalog_import.py` with `tests/services/test_catalog_import.py` (19 tests). Cost-tier and relation feeds are *not* in it and are not stubbed there — they arrive as separate files with their own shapes.

   Two things writing it changed in this spec: `bp_catalog_source` gained `attempt_count` / `first_attempt_at`, because `UNIQUE (distributor_id, content_sha256)` would otherwise make a failed import permanent and forbid the retry (the receipt is now upserted, holding the latest attempt's outcome); and the importer **manages its own transaction** and cannot be called inside a caller's, because it commits the receipt before any item write. Live verification consequently needs a scratch schema, not a rolled-back transaction.
3. **Matching**: `mpn_exact` → `sku_exact` → `description_fuzzy` into `bp_catalog_item_match` as `proposed`, with a confirmation surface. Reuse the supplier entity-resolution confirmation pattern rather than a second one.
4. **`sql/bp_sell_side.sql`** + account and opportunity write paths.
5. **Quote rendering with a serialiser allowlist** (§4.3). No cost or margin field is serialisable to a customer-facing artifact. This is the control; the DDL comments are only the convention.
6. **Outcome capture**, then a calibration job that sets `win_probability` / `win_probability_basis = 'calibrated'` once there is enough closed history — and leaves both NULL until then.

## 8. Explicit non-goals

Named so nobody reads their absence as an oversight.

- **Back-end rebates.** Retrospective, accrued, usually paid quarterly against volume commitments across a whole vendor line. That is period accounting, not line pricing, and modelling it as a per-line cost adjustment would be wrong. `cost_basis` and the bitemporal shape are the seam it attaches to later. **Until it lands, `line_margin` is front-end margin only, and any UI rendering it must say so.**
- **Affinity / co-purchase inference** (audit gap 6). `bp_catalog_item_relation` deliberately holds only what a distributor asserts — an inferred edge stored beside a manufacturer's succession notice becomes indistinguishable from it on the next query.
- **Product→category classification at scale** (audit gap 4). `unspsc_code` is the target column; the classifier that fills it is separate work. Today **0 of 4,845** distinct `item_id`s resolve to a category.
- **Migrating `bp_product_master`.** Its 186 rows live behind the FDW and hold one field the new catalog wants (`current_unit_price`, which is an observed document price, not a list price). Nothing is backfilled from it; the two coexist until the extraction pipeline is repointed, which is separate work.
- **External market benchmark data** (audit gap 9). The benchmark engine already accepts `source: 'external'` points and no store exists.
- **Tenancy.** `tenant_id` is present and nullable on the new tables, matching `bp_finding_fact` and `bp_uom_canonical`. It is not enforced anywhere, and these tables inherit that gap rather than solving it.

## 9. Acceptance criteria

1. Both DDL files apply cleanly to `proc` and re-apply idempotently. *(Verified 2026-09-09 against `bp_testdb` in a rolled-back transaction: 12 tables, applied twice, clean. The first run of this check is what found the FDW problem in §5 and a non-idempotent `ADD CONSTRAINT`.)*
2. Importing the same catalog file twice produces one `bp_catalog_source` row and no duplicate `bp_catalog_item` rows.
3. Importing a repriced feed leaves exactly one row with `valid_to IS NULL` per `(distributor_id, distributor_sku)`, and the prior row retains its old `cost_price`.
4. A feed with no cost column loads, and `cost_price` is NULL on every row it produced — not 0.
5. A quote line drafted before a repricing still reports its original `unit_cost` and `line_margin` after it.
6. No cost or margin field appears in any customer-facing quote artifact. **Prove this by attempting to render one and watching it fail**, not by inspecting the allowlist.
8. Criteria 2–4 are covered by `tests/services/test_catalog_import.py` and were each verified by mutation: the guard was broken on purpose and the suite went red. A guard whose mutation leaves the suite green is checking nothing.
7. `win_probability` is NULL on every opportunity until the calibration job has closed outcomes to work from.
