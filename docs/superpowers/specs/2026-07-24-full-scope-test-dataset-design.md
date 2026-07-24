# Full-Scope Test Dataset — Design

**Date:** 2026-07-24
**Status:** Approved for planning
**Reviewable summary:** `docs/testdata/BP_TestData_Design.xlsx` (regenerate with
`.venv/bin/python -m scripts.testdata.design_workbook`)

## 1. Purpose

Build a complete synthetic dataset that exercises every part of the BP_Backend /
SpendIQ product, and a test harness of 32 numbered cases that proves the product
works against it.

The dataset is not an end in itself. Its purpose is to make 32 assertions
falsifiable. Every design choice below exists to serve one or more of those
assertions, and anything that serves none of them is out of scope.

## 2. Problem statement

The live corpus cannot support meaningful product testing:

- 123 suppliers, ~135 documents, 16 deals.
- Whole product areas hold zero rows: opportunities, contracts, contract
  obligations, detection findings, quote evaluations, supplier aliases.
- `uicanvas.proc.business_unit` has a 5-level hierarchy and **zero rows**.
- `uicanvas.proc.cost_centre` holds 500 placeholder rows referencing business
  units that do not exist.
- The corpus is entirely `qty = 1`, which concealed a 2x unit-price defect until
  it was found by inspection rather than by test.
- There is no ground truth, so a detector that finds nothing and a detector that
  is broken are indistinguishable.

## 3. Scope

### 3.1 In scope

**314 relations** across three databases, of which **233** are seeded:

| Database | Schema | Relations |
|---|---|---|
| `bp_sqldb` | `proc` | 111 tables + 5 views |
| `bp_sqldb` | `proc_stage`, `public` | 7 |
| `uicanvas` | `proc` | 177 |
| `uicanvas` | `admin`, `proc_stage`, `public` | 13 |
| `ses` | `public` | 1 |

The remaining 81 are backups, dated snapshots and scratch copies (`_bkp`,
`_june12`, `_260925`, `_may_4th`, `_new1`, `_excel`, `_test`, `_stage`). Their
structure is cloned so nothing breaks on startup; they are left empty.

### 3.2 Explicitly out of scope

- Modifying `bp_sqldb` or `uicanvas`. The generator refuses to target either.
- Extracting all 40,000 documents through the LLM pipeline. Only the 60-document
  golden set is genuinely extracted; bulk data is inserted as rows.
- Seeding `bp_opportunity`, `bp_detection_finding`, `bp_extraction_discrepancy`
  or `bp_discrepancy_data`. These MUST start empty — they are outputs the tests
  measure, and seeding them would make the tests vacuous.
- Fixing the product defects the tests expose. Test E6 is expected to fail; that
  is a finding, not a task in this work.

## 4. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Seed into new `bp_testdb` and `uicanvas_test` | `DB_NAME` is already env-driven and the schema name `proc` is hardcoded throughout, so a sibling database is the only isolation that needs no code change |
| D2 | Cover both databases, coherently linked | BP_Backend reads `bp_sqldb`; the SpendIQ gateway reads `uicanvas`. Testing one proves nothing about the product a user sees |
| D3 | Hybrid ingestion: bulk rows + 60 golden documents | Extraction runs at 30–90s/document on a GPU-bound local model. 40,000 documents is weeks; 60 is an hour |
| D4 | Plant defects, publish an answer key | Without ground truth, a detector's output cannot be scored |
| D5 | 3.5 years, ~40k documents, ~£480M | Enough for year-on-year trend and seasonality without a 4GB database |
| D6 | All five test areas (A–E) | User decision |
| D7 | Use the real 5-level taxonomy from `uicanvas.proc.bp_category` | It exists and is well-formed. Inventing one would test the generator, not the product |
| D8 | Reproduce both supplier ID conventions plus a crosswalk | The mismatch is real; hiding it behind one convention would make it untestable |
| D9 | Multi-entity: 6 legal entities under a group | User decision. The schema already models it |
| D10 | Include test E6 (entity data isolation), expected to fail | Converts a known gap from a note into a measurable target |

## 5. Reference data: copied, not invented

Copied verbatim from live so governance, FX and categorisation behave identically:

`bp_fx_rates` (830 rows), `bp_policy`, `bp_prompt`, `bp_admin_config`,
`bp_vendor_extraction_profiles`, `bp_complaince_metric_prty_lkup`,
`procurement_patterns`, `roles_and_access`, and the full taxonomy
(`uicanvas.proc.bp_category`, `category`, `category_mapping`).

Only *business* data is synthetic.

## 6. Data model

### 6.1 Organisation

```
ORG-GRP  Beyond Procurement Group plc          (non-trading, owns no cost centres)
├── ORG-UK  Beyond Procurement UK Ltd            GBP  42%  185 cost centres
├── ORG-US  Beyond Procurement North America Inc USD  18%   98
├── ORG-DE  Beyond Procurement Deutschland GmbH  EUR  16%   92
├── ORG-IE  Beyond Procurement Ireland Ltd       EUR  12%   61
├── ORG-IN  Beyond Procurement India Pvt Ltd     INR   8%   42
└── ORG-AE  Beyond Procurement Middle East FZE   AED   4%   22
```

Business unit tree, populating the currently empty `business_unit` table:

| Level | Count | Content |
|---|---|---|
| L1 | 6 | Function: Operations, Sales, Finance, Corporate, Technology, Supply Chain |
| L2 | 40 | Region within function: Europe, North America, Middle East, LATAM, APAC |
| L3 | 120 | Department |
| L4 | 240 | Sub-department |
| L5 | 400 | Team, each with a named head and email |

L1/L2 follow the convention already present in the existing `cost_centre` rows.

500 cost centres replace the 500 placeholder rows, each carrying
`business_unit_id`, `finance_account_code`, manager name and email,
`spend_threshold_limit` (£5k–£250k), `budget_allocated_annual`,
`actual_spend_ytd`, `forecast_spend_annual`, `cost_centre_type` and
`linked_category_level_5_id`.

Every document, deal and contract carries `buyer_org_id`, `business_unit_id` and
`cost_centre_id`.

### 6.2 Category taxonomy

Copied verbatim: 246 leaf rows, 242 distinct L5 names, across
6 / 19 / 54 / 114 / 242 levels, with `unspsc_code`, `esg_impact`,
`category_status`, `spend_classification`, `category_risk_rating`,
`category_owner_name`, `category_owner_email`, `audit_frequency` and
`policy_coverage` preserved.

Families: IT & Technology (75 leaves), Marketing & Media (43),
Facilities & Real Estate (39), Professional Services (37),
Logistics & Supply Chain (36), Office & Administrative Supplies (16).

### 6.3 Suppliers

5,000 suppliers, all 51 columns of `bp_supplier` populated.

| Tier | Count | % spend | Documents each |
|---|---|---|---|
| Strategic | 120 | 38% | 60–140 |
| Core | 700 | 41% | 12–45 |
| Tail | 3,020 | 18% | 2–9 |
| One-off | 1,160 | 3% | 1 |

Distributions for `supplier_type`, `legal_structure`, `incoterms` and country
match the vocabularies already present in `uicanvas.proc.supplier`.
`risk_score` is a numeric 0–100 held as text, matching the existing convention.

Each supplier has one primary L5 category (apportioned so the 246 leaves sum to
exactly 5,000) plus one or two secondary categories.

### 6.4 Cross-database coherence

Every supplier exists in both databases under both conventions:

- `bp_testdb.proc.bp_supplier.supplier_id` = `SUP-<PascalCaseName>`
- `uicanvas_test.proc.supplier.supplier_id` = `SI######`

A new crosswalk table `bp_supplier_id_crosswalk` (columns: `bp_supplier_id`,
`uicanvas_supplier_id`, `legal_entity_key`, `created_date`) maps them. The same
approach applies to categories and cost centres.

**Verification V03 asserts that every supplier and category resolves identically
on both sides.** If it does not, the seeder is wrong.

### 6.5 Documents and volumes

Coherent chains: Requirement → 2–5 competing Quotes → award → PO → 1–3 Invoices,
plus Contracts for strategic and core suppliers. Written to all three tiers
(`_raw`, `_stg`, `_trgt`) consistently.

| | 2023 | 2024 | 2025 | 2026 (Jan–Jul) | Total |
|---|---|---|---|---|---|
| Requirements | 1,720 | 1,850 | 1,780 | 650 | 6,000 |
| Quotes | 3,980 | 4,290 | 4,180 | 1,550 | 14,000 |
| Purchase Orders | 2,280 | 2,450 | 2,380 | 890 | 8,000 |
| Invoices | 3,280 | 3,520 | 3,420 | 1,280 | 11,500 |
| Contracts | 170 | 185 | 175 | 70 | 600 |
| **Documents** | 11,430 | 12,295 | 11,935 | 4,440 | **40,100** |
| Line items | 69,590 | 74,945 | 72,875 | 27,790 | **245,200** |
| Deals | 2,290 | 2,470 | 2,400 | 840 | **8,000** |
| Spend (£m, net of VAT) | 120.0 | 135.0 | 152.0 | 73.0 | **480.0** |

Also: 5,000 catalogue items mapped to L5 leaves, 4,200 contract obligations,
240 users scoped per entity.

Currency mix: GBP 70%, EUR 14%, USD 11%, then AED, SEK, PLN, INR, CHF, SGD, JPY,
AUD. Spend reports in GBP net of VAT, converted at the document-date rate from
`bp_fx_rates`. JPY is included specifically as a zero-decimal edge case and AUD
as a `$`-ambiguity control against USD.

### 6.6 Determinism

Fixed seed. `--seed 42` always rebuilds byte-identical data, so the answer key
stays valid across regenerations. Verification V13 asserts this by comparing
checksums across two builds.

## 7. Planted defects

30 types: **24 true positives** the detectors should find and **6 negative
controls** they must not flag. Full table on the Planted Defects tab of the
workbook. Summary:

**True positives (selected):** duplicate invoices (180 exact, 120 near),
PO↔invoice unit-price mismatch (340), quantity mismatch (210), invoice with no PO
(620), tolerance breach (260), approval bypass (145), split PO (75), award not to
lowest compliant quote (400), wide unit-price spread (520), tail-spend
fragmentation (340), single-source concentration (160), expired insurance (310),
lapsed ESG certification (275), near-duplicate supplier names (240 clusters),
obligations breached (90) and expiring (130), missed renewal notice (45), payment
terms breach (230), arithmetic/FX mismatch (95), missing required fields (380),
cost-centre budget overrun (85), cross-entity price inconsistency (220),
supplier onboarded separately per entity (140).

**Negative controls — the important half:**

| Ref | Control | Count | Must not be flagged because |
|---|---|---|---|
| D22 | Services line with no quantity | 800 | Lump-sum services legitimately have no qty or unit price |
| D23 | Legitimate credit note | 190 | Negative value is correct, not an anomaly |
| D24 | Contracted CPI uplift | 150 | The contract permits the rise |
| D25 | Justified sole source | 60 | Justification and approval are on file |
| D26 | Genuinely distinct near-name suppliers | 80 | Different VAT and registration numbers |
| D30 | Justified cross-entity price difference | 180 | Explained by currency, region, volume tier or Incoterms |

The answer key is written to `docs/testdata/answer-key.json` (machine-readable,
keyed by defect ref and document/supplier id) and `answer-key.md` (readable).

## 8. Golden document set

60 real files, uploaded to S3 and genuinely extracted via
`POST /documents/extract-from-s3`.

- **G01–G24:** baseline clean documents, 6 families × 4 document types (Quote,
  Purchase Order, Invoice, Contract).
- **G25–G60:** 36 edge cases — qty > 1 (the regression guard for the past 2x
  unit-price bug), lump-sum services, rate card with version-history table,
  page-break split table, 100+ line items, XLSX, CSV, VAT-inclusive, reverse
  charge, credit note, EUR comma decimals, USD/AUD `$` ambiguity, JPY
  zero-decimal, ambiguous date format, supplier/buyer block adjacency, two-column
  layout, quote revision v2, PO amendment, partial delivery, consolidated
  invoice, freight surcharges, 4-decimal unit price, thousands separators,
  zero-value line, discount line, obligation-rich prose contract, auto-renewal
  clause, CPI uplift clause, PO referencing a quote number, scanned appearance,
  rotated landscape, exact re-upload, amended re-upload, missing supplier VAT,
  wrapping descriptions, multi-currency lines.

Each has a hand-checked expected-values file at
`docs/testdata/golden/<ref>.expected.json`. This doubles as a permanent
extraction regression suite, re-runnable after any model or pipeline change.

## 9. Test harness

32 numbered cases across five areas. Full detail on the Test Scenarios tab.

| Area | Cases | Proves |
|---|---|---|
| A. Extraction accuracy | A1–A8 | Header fields, line completeness and arithmetic, qty>1 regression, 5-level category assignment, absent-data-stays-NULL, FX, dedup/amend, missing-field routing |
| B. Three-way match | B1–B6 | Over-billing, quantity variance, maverick spend, tolerance/bypass/split-PO, duplicate not double-counted, **B6 negative** |
| C. Deals & ranking | C1–C6 | Correct grouping, **C2 negative: no incorrect merges**, version supersession, ranking order, wrong-award detection, negotiation round |
| D. Opportunities & compliance | D1–D6 | Price variance, tail consolidation, single-source, compliance failures, obligations, **D6 negative** |
| E. Multi-entity | E1–E6 | Roll-up integrity, budget overrun, per-cost-centre thresholds, cross-entity price inconsistency, **E5 negative**, **E6 data isolation (expected to fail)** |

Four cases (B6, C2, D6, E5) pass only when the product produces **zero** output.
These discriminate a working detector from one that flags everything.

Two targets are deliberately below 100%:

- **A4** (L5 category assignment): ≥90% correct L5, 100% correct L1. Mapping free
  text to one of 242 leaves is genuinely hard; a 100% target would guarantee a red
  result that teaches nothing.
- **E6** (entity data isolation): expected to FAIL. `/workflows/ask` has no
  authorization and `user_id` does not scope retrieval.

Results are written to `docs/testdata/results/<timestamp>.json` with per-case
pass/fail, and precision/recall per defect type.

## 10. Verification

14 checks; 12 block sign-off.

| Ref | Check | Severity |
|---|---|---|
| V01 | Every in-scope table reaches its planned row count | Blocks |
| V02 | No orphan line items, deal maps or document references | Blocks |
| V03 | Every supplier and category resolves identically on both sides | Blocks |
| V04 | Line totals sum to document totals on every non-defect document | Blocks |
| V05 | Every converted total re-derives from `bp_fx_rates` at the document date | Blocks |
| V06 | Every line resolves to a valid L1–L5 path | Blocks |
| V07 | Group spend = sum of entities = sum of BUs = sum of cost centres, to the penny | Blocks |
| V08 | Every UI screen's backing query returns non-empty | Blocks |
| V09 | Every FastAPI and gateway route returns 200 with a non-empty payload | Blocks |
| V10 | Each planted defect is found by its detector | Scored |
| V11 | No negative control produces a finding | Blocks |
| V12 | Golden set extraction matches expected values | Scored |
| V13 | Two builds with the same seed produce identical checksums | Blocks |
| V14 | `bp_sqldb` and `uicanvas` row counts unchanged before and after | Blocks |

V14 is the isolation guarantee and runs first and last.

## 11. Safety

1. The generator refuses to run when the target database name is `bp_sqldb` or
   `uicanvas`. There is no override flag.
2. Reads from live databases are read-only: schema introspection and reference
   data copies only.
3. V14 captures row counts of both live databases before and after every build
   and fails the run if any differ.
4. Target databases are created by the generator, not assumed to exist. A
   `--drop-first` flag recreates them; it is refused against live names by the
   same guard.

## 12. Module layout

```
scripts/testdata/
    __init__.py
    design_workbook.py     # exists: builds the reviewable workbook
    build.py               # entry point: --target, --seed, --drop-first
    guards.py              # target-name refusal, live row-count snapshots
    schema.py              # clone structure from live into the test databases
    reference.py           # copy FX, policies, prompts, taxonomy verbatim
    org.py                 # entities, business units, cost centres, users
    suppliers.py           # 5,000 suppliers, both ID conventions, crosswalk
    catalogue.py           # 5,000 items mapped to L5 leaves, price curves
    documents.py           # requirement -> quote -> PO -> invoice chains
    deals.py               # deal assembly and document mapping
    downstream.py          # rankings, evaluations, decisions, actions, summaries
    defects.py             # plant the 30 defect types, emit the answer key
    golden.py              # generate the 60 real files + expected-values
    verify.py              # V01-V14
    scenarios/             # one module per test case A1-E6
```

## 13. Acceptance criteria

This work is complete when:

1. `python -m scripts.testdata.build --target bp_testdb --seed 42` completes and
   populates both test databases.
2. All 12 blocking verification checks pass.
3. All 32 test cases execute and produce a scored result file.
4. The four negative cases (B6, C2, D6, E5) produce zero findings.
5. E6 is recorded as failing, with the specific gap named.
6. `bp_sqldb` and `uicanvas` are provably unchanged (V14).
7. Rebuilding with the same seed reproduces identical checksums (V13).

Note that criterion 3 requires the cases to *execute and be scored* — not that
they all pass. A red result from a correctly built test is a product finding and
is reported as such, not worked around.
