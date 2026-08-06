# Phase 0 — Seam Map

**Date:** 2026-08-06
**Branch:** `Development`
**Repo:** `BP_Backend` (the brief calls it "ProcureIQ"; that name appears nowhere in this codebase)
**Evidence:** live reads against `bp_sqldb` and `bp_testdb` on `procwisemvpdb01.cluster-cpae0sg4mrk8.eu-west-1.rds.amazonaws.com`, plus static reads of `main`-lineage code on `Development`.

---

## 0. Premise verification — read this first

The brief says three findings are "already documented in `Opportunity_Aggregation_Valuation_Audit_FINDINGS.md` and `Negotiation_Agent_Audit_FINDINGS.md`; do not re-derive them, but do verify each still holds."

**Neither document exists in this repository.** There is a root `FINDINGS.md` (2026-07-14, extraction pipeline) which is a different artefact. So the three findings have been derived from code and live data rather than re-verified against a prior write-up. Everything below is first-hand evidence.

**"GPSS" / "Global Procurement Semantic Standard" appears nowhere** in the codebase, in `docs/`, or in any schema. The brief's constraint "GPSS is the vocabulary… extend the dictionary rather than shadowing it" has no dictionary to extend. This is a blocker for Phase 1 field naming and for the Category Profile Resolver in Phase 5 — see §7.

### Verdict on the three findings

| # | Finding as stated | Verdict | Correction |
|---|---|---|---|
| 1 | The fact model is destroyed at the opportunity seam | **HOLDS — and is worse than stated** | Several named fields are not destroyed at the seam. They are **never extracted at all**. See §2. |
| 2 | The valuation math sums where it should decompose | **HOLDS exactly as stated** | `SUM(financial_impact_gbp)` confirmed at 8 call sites. No netting, no ranges, no probability weighting anywhere. |
| 3 | There is no reference corpus | **HOLDS, with one nuance in our favour** | Market intelligence is indeed free-text. But a deterministic benchmark engine *is* wired and running against internal price history — a partial corpus already exists. See §4. |

**The material correction to Finding 1.** The brief asserts "Extraction captures `unit_price`, `currency`, `uom`, `quantity`, `contract_id`, `term_start/end`, `cost_centre`, `parent_contract_id`. These collapse into unstructured `calculation_details`." That is only half right, and the half that is wrong changes Phase 1's scope from a *plumbing* job into a *plumbing plus extraction-schema* job:

- `unit_price`, `quantity`, `currency` — genuinely captured, genuinely lost at the seam. ✅ as stated.
- `uom` — declared in **one** of four extraction schemas (`quote.yaml` line items only). Across 64,118 real provenance records in `bp_sqldb`, `unit_of_measure` was extracted **12 times**, against 344 quote-line `unit_price` extractions. It is absent from `invoice.yaml` and `purchase_order.yaml` entirely.
- `contract_id` — declared only in `contract.yaml`, which has **no line items at all** (`db_lines_table: null`). `proc.bp_purchase_order_trgt.contract_id` is populated on **0 of 5,041 rows**. There is no path from any transaction to its governing contract.
- `term_start/end` — `contract_start_date` / `contract_end_date` exist in `contract.yaml`. Extracted **2 times** ever.
- `cost_centre` — **not in any extraction schema.** The column `proc.bp_contracts.cost_centre_id` exists and is 100% NULL.
- `parent_contract_id` — **not in any extraction schema.** The column exists and is 100% NULL. There is no amendment-chain model of any kind.
- `escalator_pct`, `escalator_cap`, `term_months`, `billing_frequency`, `amendment_ref`, `document_version` — **do not exist anywhere in the codebase.** Zero grep hits outside this document.

**And the load-bearing fact underneath all of it: `proc.bp_contracts` has 0 rows in both `bp_sqldb` and `bp_testdb`. `proc.bp_contract_obligation` has 0 rows in both.** The corpus contains no contracts. See §7 — this blocks Phase 3.2 and most of Phase 4 outright.

---

## 1. The record at every hop

Actual field lists, not intended ones.

### Hop 1 — Extraction output (`Candidate`)
`src/services/extraction/types.py`, produced by `src/services/extraction/dispatch.py`

```
Candidate: field, value, confidence, span(page, bbox, text), extractor
```

**This is the richest the record ever gets.** Every candidate is span-bound: it carries page, bounding box and the verbatim snippet it came from. This is the provenance foundation Phase 1 needs and it already exists.

### Hop 2 — Header record / line items (`dict[db_column, coerced_value]`)
`src/services/extraction/persistence.py:82` `build_header_record`, `:120` `build_line_items`

```
{ db_column: coerced_scalar, ... }        # spans discarded
```

The span is **detached here**. It survives separately into `proc.bp_extraction_provenance_v3` (via `write_provenance`, `persistence.py:243`) keyed on `(doc_type, doc_pk, field_path)`, but the value record itself is a bare dict of scalars. Reuniting a figure with its locator later requires a join nobody performs.

### Hop 3 — `_stg` tables
`src/services/extraction/promotion.py:565` `promote`

`_stg_columns()` (`:108`) and `_table_columns()` (`:118`) intersect the produced dict against the physical table. Anything not in both is dropped without a record.

### Hop 4 — `_trgt` tables (the final destination)
Live column lists, `bp_testdb`:

| Table | Columns | Commercially relevant content |
|---|---|---|
| `bp_quote_trgt` | 31 | quote_id, supplier_id, quote_date, validity_date, currency, total_amount, tax, fx-to-USD, deal_id, confidence_score |
| `bp_quote_line_items_trgt` | 20 | item_description, quantity, **unit_of_measure**, unit_price, line_total, tax, currency |
| `bp_purchase_order_trgt` | 42 | + incoterm, payment_terms, **contract_id (0% filled)** |
| `bp_po_line_items_trgt` | 21 | item_description, quantity, unit_price, unit_of_measure, currency |
| `bp_invoice_trgt` | 34 | invoice_date, due_date, paid_date, payment_terms, currency, amounts, fx |
| `bp_invoice_line_items_trgt` | 24 | + delivery_date |
| `bp_contracts` | 27 | contract dates, cost_centre_id, business_unit_id, is_amendment, parent_contract_id — **0 rows** |

No table anywhere carries: term_months, billing_frequency, escalator, amendment_ref, document_version, bundle/comparison grouping, value_basis, or `tenant_id`.

### Hop 5 — `Finding`
`src/agents/opportunity_miner_agent.py:309`, constructed at `:4583` in `_build_finding` (`:4510`)

```python
Finding:
  opportunity_id, opportunity_ref_id, detector_type, policy_id,
  supplier_id, supplier_name, category_id, item_id, item_reference,
  financial_impact_gbp: float,          # single point estimate, GBP only
  calculation_details: Dict,            # <-- everything else, untyped
  source_records: List[str],            # <-- bare document ids, no locator
  detected_on, weightage, candidate_suppliers, context_documents,
  ml_priority_score, feedback_*, is_rejected
```

**This is the seam.** One float and an untyped dict. There is no unit price, no currency, no quantity, no UoM, no term, no contract reference, no document version, no confidence, no validation state, no provenance. `source_records` is a list of plain identifier strings — it cannot resolve to `(document_id, extraction_id, locator)` because no extraction_id or locator is carried.

### Hop 6 — `proc.bp_opportunity`
`deploy/sql/2026-06-15_bp_opportunity.sql`, upserted by `src/services/opportunity_store.py:23`

25 columns, of which the only commercial content is `financial_impact_gbp NUMERIC`, `realised_savings_gbp NUMERIC`, and `calculation_details JSONB`.

### Hop 7 — Analyst / API input
`src/api/routers/opportunities.py:60`

```python
cols = ["opportunity_id", "detector_type", "category_id", "supplier_name",
        "item_description", "financial_impact_gbp", "stage", ...]
```

**Seven columns. `calculation_details` is not selected.** The JSONB that everything was flattened into is never read back by any API route or render path — see §3. The fact model dies twice: once into the JSONB, and again because nothing reads the JSONB.

### Hop 8 — Rendered output
`src/services/deal_analysis_service.py:414` `to_ui_row`

```python
return {"id":…, "supplier":…, "category":…, "value":…, "volume":…,
        "unitPrice":…, "priceChange":…, "volumeChange":…, "efficiency":…, "items":…}
```

Ten hardcoded keys. **Every value is pre-formatted to a string** in Python ("the UI search filter lowercases each field"). Absent values become the literal en-dash `"–"`. By the time a figure reaches the browser it is a string with no type, no units, no currency code, no confidence and no identity — binding a `fact_id` to it is impossible without changing this function's contract.

**Net.** Of the eight hops, the record loses structure at four of them (2, 3, 5, 8) and is never re-enriched. A quote line that entered extraction as `{value: "£86.94", page: 3, bbox: […], text: "Unit rate £86.94/user/month"}` leaves as the string `"£86.94"` in a dict key called `unitPrice`.

---

## 2. Field mortality table

Death site for every field declared in `extraction_schemas/*.yaml`, plus the fields the brief expects that are not declared at all.

### 2a. Declared fields that die in transit

| Field | Schema | Dies at | Mechanism |
|---|---|---|---|
| *any header field with no `db_column`* | all | `persistence.py:100–101` | `if meta.db_column is None: continue` — silent skip, no discrepancy raised |
| *any line field with no `db_column` or unknown name* | all | `persistence.py:139–141` | `if fs is None or fs.db_column is None: continue` — silent skip |
| `supplier_name` (invoice) | invoice.yaml | `persistence.py:100` | declared with `db_column: None` → dropped every time |
| `span` (page/bbox/verbatim) on **every** field | all | `persistence.py:82–115` | value coerced into a scalar dict; span written only to `bp_extraction_provenance_v3`, never carried on the record |
| *any field whose `db_column` is absent from `_stg`* | all | `promotion.py:108` `_stg_columns` | set-intersection against physical table; no log of the difference |
| `unit_of_measure` | quote.yaml lines only | not dropped — **never populated** | 12 provenance rows in 64,118 (`bp_sqldb`) |
| `confidence_score` | computed at `promotion.py:163` | `opportunity_miner_agent.py:4583` | not read by the miner; `Finding` has no confidence field |
| **all** commercial detail | any | `opportunity_miner_agent.py:4583` | collapsed into `calculation_details: Dict` |
| `calculation_details` itself | — | `api/routers/opportunities.py:60` | not in the selected column list; never reaches a consumer |
| all numeric typing, units, currency | — | `deal_analysis_service.py:414` `to_ui_row` | `_money()` / `_signed_pct()` stringify; `None` → `"–"` |

### 2b. Fields the brief expects that are never extracted

| Field | Extraction schema | DB column | Live fill | Note |
|---|---|---|---|---|
| `uom` (invoice, PO lines) | ❌ absent | ✅ exists | seeded only | schema gap, not a seam loss |
| `contract_id` on transactions | ❌ absent | ✅ `bp_purchase_order_trgt.contract_id` | **0 / 5,041** | no transaction→contract link exists |
| `cost_centre` | ❌ absent | ✅ `bp_contracts.cost_centre_id` | 0 (table empty) | present in legacy `data_extraction_agent.py:335` only |
| `parent_contract_id` | ❌ absent | ✅ `bp_contracts.parent_contract_id` | 0 (table empty) | legacy `:337` only |
| `is_amendment` / `amendment_ref` | ❌ absent | ✅ `bp_contracts.is_amendment` | 0 (table empty) | no amendment chain model |
| `term_months`, `billing_frequency` | ❌ absent | ❌ absent | — | zero grep hits repo-wide |
| `escalator_pct` / `_basis` / `_cap` | ❌ absent | ❌ absent | — | zero grep hits repo-wide |
| `document_version`, `superseded_by`/`supersedes` | ❌ absent | ❌ absent | — | no supersession model |
| `tenant_id` / `org_id` | ❌ absent | ❌ absent | — | **0 such columns in the whole `proc` schema, both databases** |

### 2c. Extraction reality check — provenance coverage, `bp_sqldb` (64,118 records)

| doc_type | provenance rows | distinct fields | key field counts |
|---|---|---|---|
| invoice | 41,238 | 110 | `line_items[].unit_price` 4,180 · `quantity` 4,178 · `currency` 2,046 · `payment_terms` 936 |
| purchase_order | 16,267 | 63 | `quantity` 2,623 · `unit_price` 1,607 · `currency` 1,135 · `payment_terms` 203 |
| quote | 6,594 | 125 | `currency` 742 · `quantity` 356 · `unit_price` 344 · **`unit_of_measure` 12** |
| contract | **19** | **6** | `contract_id` 4 · `contract_signatory_name` 4 · `contract_start_date` 2 |

Zero provenance records exist for cost_centre, escalator, amendment, parent contract, term length, or billing frequency — in either database.

> ⚠️ **Measurement trap.** In `bp_testdb`, `unit_of_measure` is populated on 115,722 / 115,814 quote lines, 22,537 / 22,560 PO lines, and 55,421 / 55,483 invoice lines — 99.9%, and identical in `_stg` and `_trgt`. That data was **seeded**, not extracted: the uom values are near-uniformly distributed across ten values (`case` 12,313, `tonne` 12,266, `each` 11,727, `month` 11,648, `hour` 11,477 …), and `purchase_order.yaml`/`invoice.yaml` have no `unit_of_measure` field for an extractor to fill. Any mortality measurement taken against `bp_testdb` column fill rates will conclude UoM survives fine. It does not. Measure against `bp_extraction_provenance_v3`, not against column NULL counts.

---

## 3. Consumers of `calculation_details`

Complete list of production call sites (tests excluded).

| Site | Role | Migration burden |
|---|---|---|
| `opportunity_miner_agent.py:4583` | **producer** — `_build_finding` | primary rewrite target |
| `opportunity_miner_agent.py:830–881` | dedup merge; unions two dicts | low |
| `opportunity_miner_agent.py:2007–2010` | reads `item_reference` / `item_id` back out | replace with structured column |
| `opportunity_miner_agent.py:2183–2186` | mutates in place | low |
| `opportunity_miner_agent.py:5191–5192, 5361–5402, 5649` | `setdefault` of item_reference, item_description, catalog_product, deal_id | replace with structured columns |
| `duplicate_invoice_detector.py:442` | second producer — builds its own dict | secondary rewrite target |
| `opportunity_store.py:34, 44, 56` | reads `deal_id`/`quote_id`/`po_id`/`invoice_id`/`item_description` out of the JSONB to fill real columns; writes the rest as JSONB | **this is the pattern to generalise** — it already does structured extraction from the blob for five fields |
| `tests/test_opportunity_retirement.py:101` | deal_id fallback | test |

**Total: 2 producers, 4 reader groups, 1 store. No API route and no render path reads it.**

This is smaller than expected and it is good news: the migration surface is one agent file, one detector, and one store module. `opportunity_store.upsert_opportunity` already promotes five keys out of the JSONB into real columns — the Phase 1.2 migration extends an existing pattern rather than inventing one.

---

## 4. Subsystem status

| Subsystem | Status | Evidence |
|---|---|---|
| **Benchmark Pricing Engine** | **WIRED** | `src/services/benchmark/engine.py` (pure, no I/O, no LLM, Excel-parity rounding via `excel_round`). Live wiring `benchmark_live.py:229 benchmark_deal`. Router `api/routers/benchmark.py` registered at `api/main.py:427`. Consumed by `price_outlier/detector.py`. Golden fixtures in `tests/fixtures/benchmark`, parity test `tests/test_benchmark_parity.py`. |
| — evidence-threshold gate | **WIRED, fails closed** | `BenchmarkSettings.min_data_points = 3` |
| — five adjustment factors | **WIRED but four are neutralised on live data** | `benchmark_live.py:29` feeds `_NEUTRAL_SCORE = 5.0` to both sides so spec/SLA factors resolve to 1.0; no location or index tables exist, so those default. **Volume is the only live adjustment.** This is disclosed, not hidden (`DISCLOSURES`, `benchmark_live.py:33`). |
| — corpus | **internal history only** | `load_benchmark_pool` (`benchmark_live.py:74`) = all PO + invoice lines with a price, excluding the subject deal. No list prices, no discount bands, no external source. |
| **Correlation-adjusted rollup** (PERT / Pearson / Bayesian shrinkage / PSD) | **SPEC_ONLY — in fact, ABSENT** | Zero matches repo-wide for `PERT`, `pearson`, `bayesian shrink`, `semi.definite`, `correlation matrix`. Not merely unwired — never written. |
| **Contract Conformance Engine** | **SPEC_ONLY** | Single reference: a docstring in `src/engines/decision_engine.py:3` ("Phase 3 of the conformance design (2026-06-28)"). No conformance module, no clause-gap type, no `NOT_DETECTED` state anywhere. |
| **CISE** (criticality / concentration / switching cost) | **SPEC_ONLY** | Zero matches for `CISE`, `supplier_criticality`, `switching_cost`, `concentration_risk`. |
| **Supplier Ranking** | **EXISTS** | `proc.bp_supplier_ranking` table present; deal-scoped per prior work. Not yet a block producer. |
| **Decision Engine** | **WIRED** | `src/engines/decision_engine.py` → `api/routers/decisions.py` (6 call sites). Already enforces two rules Phase 4 wants: facts gathered from DB first, escalate rather than guess, decisions written to `proc.bp_decision` with `facts` + `evidence`. **This is a working precedent for the Adjudication Plane.** |
| **Obligations** | **EXISTS_UNWIRED (no data)** | `src/services/obligations/` with `ContractObligation` schema, grounding check (`is_quote_grounded`), `bp_contract_obligation` table (`obligation_id, document_id, contract_id, name, obligation_type, clause_ref, source_quote, grounded`). **0 rows in both databases** because there are no contracts. Closest existing analogue to Phase 1.4 `Constraint`. |
| **Negotiation Agent** | **WIRED, point-estimate-driven** | `NegotiationContext` (`negotiation_agent.py:268`) = `current_offer: float`, `target_price: float`, plus scalar 0–1 dials (`aggressiveness`, `leverage`, `urgency`). `SupplierSignals` = `offer_prev`, `offer_new`, `message_text: str`. Confirms Finding 2. |
| **Market intelligence** | **free text, as stated** | `negotiation_advice/ranking.py:199–209` reads `market.get("demand_trend")` and compares to the strings `"rising"` / `"high"`; `market.get("inflation")`. Confirms Finding 3. |
| **`bp_negotiation_advice_fact`** | **EXISTS, empty, untyped** | columns `advice_id, fact_key, fact_value, stated_by, stated_at, withdrawn_at`. 0 rows. Untyped key/value strings, but `withdrawn_at` is a partial bitemporal precedent. |
| **`bp_extraction_provenance_v3`** | **WIRED, populated** | `provenance_id, doc_type, doc_pk, field_path, value, page, bbox_*, evidence_text, model, model_confidence, judge_actions, final_confidence, extracted_at, pipeline_version`. 64,118 rows in `bp_sqldb`. **This is the single strongest existing asset for Phase 1** — span-bound provenance already exists and is already written; it is simply never joined back to a value at any downstream hop. |
| **`bp_detection_finding`** | **EXISTS, empty** | Has `observed_value`, `expected_value`, `delta`, `blocks_promotion`, `regulation`, lifecycle columns. 0 rows. Shape is close to what Phase 3.2 Baseline Integrity findings need. |
| **`bp_fx_rates`** | **WIRED, populated** | 1,162 rows (`bp_sqldb`), 3,154 (`bp_testdb`). Dated FX exists — Phase 1.3's "dated FX rate stored on the fact, never a live lookup" is achievable today. |

---

## 5. Current absence handling

### What `completeness` measures
`src/services/extraction/completeness.py` — pure functions, `assess()` at `:135`, `CompletenessReport` at `:53`.

It answers exactly two questions:
1. **Schema fill** — is every `required: true` header field present, and are there any line items?
2. **Arithmetic reconciliation** — does `sum(line_amount)` reconcile to the header subtotal, within `_RECONCILE_ABS_TOLERANCE = 1.00` absolute (deliberately not a percentage; the comment records that a former 5% tolerance hid a £1,000 discrepancy on two quotes).

**It is schema field fill rate, not commercial completeness.** It cannot detect an absent escalator cap, an unpriced exit-assistance line, a referenced-but-not-included rate schedule, or a support tier quoted without response times — none of those are schema fields, and a document missing all four reconciles perfectly. This is precisely the distinction the brief draws, and only the second kind is useful to a buyer. Phase 4.6's `expected_element_set` has no existing implementation to extend.

One thing worth keeping: the module is already *pure and testable*, and the dispatch loop already uses a completeness gap to trigger a bounded recovery pass before promotion rather than promoting silently. The control-flow shape Phase 4.6 needs already exists; only the predicate is too narrow.

### Is `NOT_DETECTED` reachable for quotes?
**No — the state does not exist.** There is no Contract Conformance Engine (§4), so there is no `NOT_DETECTED` for quotes or for contracts. Absence is currently representable in exactly three ways, all lossy:
- a `Discrepancy` with `issue_type="type_bind_error"` (`persistence.py:57`) — only for a value that was found but could not be coerced;
- a silent `continue` for a field with no `db_column` — no record at all;
- the string `"–"` at render (`to_ui_row`).

The four states the brief requires — **not provided / not applicable / not extractable / not yet processed** — currently all render as the same en-dash.

---

## 6. Current UI rigidity

**Phase 5 is a rebuild, not an extension.** Evidence:

| Question | Answer |
|---|---|
| Columns hardcoded or resolved? | **Hardcoded, on both sides.** Backend `to_ui_row` (`deal_analysis_service.py:414`) returns a fixed 10-key literal. The UI (`beyond_procwise_ui/src/utils/draftObjects.jsx:124`) carries mock rows with the same 10 keys. No field-role indirection, no category profile, no renderer registry. |
| What happens to a field with no renderer? | **Dropped.** It is never in the dict, so it cannot render. This is the seam failure of §1 repeated at the last hop. |
| Any density control (`summary` / `standard` / `forensic`)? | **None.** One shape, one density. |
| Can a number bind to a `fact_id`? | **No.** `to_ui_row` returns strings — `_money()` and `_signed_pct()` format at the backend precisely so the UI's search filter can lowercase every field. There is no object to hang an identity on. |
| Distinct absence states? | **No.** All absence collapses to `"–"`. |
| Extension slots / sub-blocks? | **None.** |
| Overflow behaviour specified? | **Nowhere.** No virtualisation, no row cap, no column-count handling, no long-string policy in either repo. |

The `AnalysisBlock` contract, field-role resolution, density levels, extension slots, the unknown-field `additional_attributes` area, and epistemic-state marking all have to be built from nothing. Note also that the render layer lives in `beyond_procwise_ui` (a separate repository), so Phase 5 crosses a repo boundary that Phases 1–4 do not.

---

## 7. Blockers — decisions needed before Phase 1 starts

Per the working instruction to say so and stop where a specification conflicts with what the code needs.

### B1 — There are no contracts. (blocks Phase 3.2 and most of Phase 4)
`proc.bp_contracts`: **0 rows**, both databases. `proc.bp_contract_obligation`: **0 rows**, both. Real contract extraction has produced 19 provenance records across 6 fields, ever.

Everything below depends on a contract corpus that does not exist:
- Phase 3.2 Baseline Integrity — superseded rate applied, amendment chain integrity, escalator conformance, co-termination. All four need contracts and amendments.
- Phase 4.4 Constraint Testing — the worked "500 named users" example needs a contracting schedule and a price schedule.
- Phase 4.6 `ABSENT_VS_PRECEDENT` — needs a prior contract to compare against.

Additionally `bp_purchase_order_trgt.contract_id` is 0 / 5,041, so even with contracts loaded there is currently no join from a transaction to its governing agreement.

**Needs a decision:** source a contract corpus (the ServiceNow/YBS golden case documents would do), or descope the contract-dependent checks and say so explicitly.

### B2 — There is no tenant dimension. (blocks the multi-tenant constraint and Phase 7 test 6)
**Zero `tenant_id` or `org_id` columns exist in the `proc` schema, in either database.** There is no RLS, no pooling, no cohort boundary. The brief's "all new tables carry `tenant_id` under RLS", the pooling neutrality test, and "a benchmark computed for tenant A must never be influenced by tenant B's rows" describe a property of a system that is currently single-tenant.

**Needs a decision:** (a) introduce a tenant dimension as Phase 1 work — a large change touching every table and every query; (b) carry `tenant_id` on *new* tables only, defaulted to a single tenant, so the constraint is honoured going forward without a retrofit; or (c) descope. **My recommendation is (b)** — it satisfies "all new tables carry `tenant_id`" literally, costs almost nothing now, and leaves the retrofit as a separable project. It does mean the Phase 7 pooling-neutrality test is vacuous until a second tenant exists, which should be stated rather than papered over.

### B3 — GPSS does not exist. (blocks Phase 1 field naming and Phase 5 field-role resolution)
No dictionary, no codes, no `category_l1..l4` taxonomy anywhere. `CommercialFact.gpss_code`, `Constraint.gpss_code`, the Category Profile Resolver's GPSS→role bindings, and the unmapped-GPSS-code test (Phase 7 test 14) all reference a vocabulary with no source.

**Needs a decision:** supply the GPSS dictionary, or authorise defining a minimal internal code set for the categories actually in the corpus and treat it as the seed of the dictionary rather than a shadow of it.

### B4 — Four of five benchmark adjustment factors are inert on live data.
The engine implements all five correctly. But `benchmark_live.py` neutralises spec, SLA, location and index because the corpus holds no spec scores, SLA scores, location cost indices or price indices. Volume is the only live factor. This is honest and disclosed, but it means Phase 2.2's "five multiplicative adjustment factors declared explicitly on the output" will, on this corpus, declare four as neutral. Phase 2.1's reference corpus is the fix; flagging it so the Phase 2 acceptance bar is set against reality.

### B5 — The opportunity layer barely fires.
`proc.bp_opportunity`, `bp_testdb`: **308 rows, 3 detector types**.

| Detector | Findings | Value |
|---|---|---|
| Duplicate Invoice Recovery | 300 | £3,218,074.41 |
| Invoice Overbilling | 6 | £261,581.47 |
| Price Benchmark Variance | 2 | £40,414.70 |

`bp_sqldb`: **0 rows.** 97% of findings and 91% of value come from one detector. The brief's framing — "the analyst layer is shallow because it reads an impoverished record" — is right, but incomplete: the layer feeding it is also barely producing. Phase 1 will make the records rich; it will not by itself make them numerous. Worth agreeing what "measurably richer findings" means as the Phase 4→5 gate before Phase 1 starts, so the bar is not set retrospectively.

---

## 8. What is genuinely in our favour

Not everything is a gap, and the plan should exploit these rather than rebuild them:

1. **Span-bound provenance already exists and is already written.** `bp_extraction_provenance_v3` has page, bbox, verbatim text, model, confidence and pipeline version on 64,118 real extractions. `CommercialFact.provenance` does not need new capture — it needs the join that hop 2 currently drops.
2. **Dated FX already exists.** `bp_fx_rates`, populated. Phase 1.3's "dated FX rate on the fact, never a live lookup" is reachable now.
3. **A deterministic, LLM-free, Excel-parity benchmark engine is wired and tested**, with a fail-closed evidence gate and explicit disclosure of every fallback. Phase 2.2 is a wrapper, not a build.
4. **The Decision Engine is a working precedent for the Adjudication Plane** — facts from the database first, escalate rather than guess, verdict written with `facts` + `evidence` so it can be re-derived. Phase 4's plane separation can follow a pattern that already ships here.
5. **The `calculation_details` migration surface is small** — 2 producers, 4 reader groups, 1 store — and `opportunity_store.upsert_opportunity` already promotes five keys out of the JSONB into real columns.
6. **`completeness.py` is pure and already gates promotion.** Phase 4.6 needs a wider predicate, not new control flow.

---

## 9. Recommended amendment to the phase order

Phase 1 as written assumes the named fields exist and merely need carrying. §2b shows six of them do not exist at any layer. I propose splitting Phase 1:

- **Phase 1a — extraction schema extension.** Add the missing fields to `extraction_schemas/*.yaml` (uom on invoice/PO lines; contract_id on transactions; cost_centre, parent_contract_id, is_amendment, term, escalator, billing_frequency on contract), with `db_column` bindings and the migrations to back them. Without this, `CommercialFact` is a richer container for the same absent data.
- **Phase 1b — the seam carry** exactly as the brief specifies: `CommercialFact`, `Constraint`, mandatory provenance enforced in the validator, UoM/FX normalisation, structured columns on `Finding`/`bp_opportunity`, deprecation shim on `calculation_details`.

Doing 1b first would produce a beautifully typed, fully provenanced record whose `escalator_pct` is NULL on every row — the flexible-stage-with-thin-material failure the brief warns about, one layer down.

---

## Stop point

Phase 0 is complete. **Blockers B1, B2 and B3 need your decision before Phase 1 can be planned**, and I would like agreement on the §9 split before I write the Phase 1 plan.
