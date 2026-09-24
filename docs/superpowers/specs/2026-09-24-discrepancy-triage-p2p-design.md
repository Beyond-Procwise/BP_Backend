# Discrepancy Triage — Procure-to-Pay Slice (Design)

**Date:** 2026-09-24
**Status:** Draft, awaiting user review
**Parent document:** *Discrepancy Triage Specification* Draft v0.2 (owner Nick Geelen, 24 Sep 2026) — referred to below as "the triage spec", with its section numbers written §N.

## 1. Goal

Compare the quote, purchase order and invoices that belong to one deal; decide which
differences matter; and put only those in front of people, in the existing SpendIQ
Action Centre. Everything else is kept in an audit table and can be inspected, but is
not shown.

The first release covers **procure-to-pay only** and must be proven **at scale**: a
full backfill of every deal in `bp_testdb` (5,041 deals), with a report of the triage
spec's §15 measures.

### Success criteria

1. Full backfill of all deals in `bp_testdb` finishes in **under 15 minutes**, with no
   LLM or GPU call anywhere in the engine.
2. Re-running the backfill with unchanged data produces **no new findings and no
   duplicates** (idempotent).
3. Any single run can be removed by `scripts/triage_rollback.py --run-id <id>` without
   touching findings a person has acted on.
4. The run report states: runtime and deals/second; **noise ratio** (findings shown ÷
   raw differences, target < 5%); deals per verdict; exposure by severity; findings per
   rule; failed deals; known unchecked areas.
5. Planted-error recall (§9.3) is reported, and every "always S1" rule has been shown to
   go red when broken.
6. Findings are visible through the gateway's real `GET /discrepancies` endpoint on the
   running local stack, and 10 of them have been checked by hand against the raw rows.

### Decisions already made (user, 2026-09-24)

- **Presentation:** Action Centre only. Findings are written to
  `proc.bp_detection_finding`, which the gateway already serves. No UI or gateway
  change. A per-deal verdict is served by a new BP_Backend endpoint.
- **Existing deals:** backfill all of them once, to test the engine at scale.

## 2. Scope

### In scope

- Engine package `src/services/triage/` implementing the triage spec pipeline (§6):
  normalise → link → compare → group → score → write, for quote/PO/invoice/credit note.
- The thirteen checks in §5 below, the grouping patterns in §6, scoring and overrides in §7.
- Three new tables, one new `bp_policy` limit row.
- Scheduler job, two API endpoints, backfill script, rollback script.

### Out of scope (each has a seam, named where relevant)

| Triage spec feature | Why not now | Seam |
|---|---|---|
| Customer / category / counterparty leniency hierarchy (§7.3–7.6) | No tenant dimension; no category dimension (0 rows in `bp_category`) | `resolve_tolerance(check, ctx)` is the only place a tolerance is read |
| Rule learning from user decisions (§11) | Needs decision capture first | Overrides are applied after any future rule layer, so rules can never soften them |
| Goods receipts / Fulfilment role | No goods-receipt table exists | Quantity authority is one entry in the authority map |
| Payee bank-detail check (§8.3) | Invoices do not capture bank details | Reported as a known gap in every run report |
| Quote-vs-PO price check | The PO is the agreed price by definition; not asked for by the triage spec | — |
| Other domains (finance, supply chain, trade finance) | No documents | Checks are registered per profile; this is profile `procure_to_pay` |
| New UI screens | User decision | Verdict endpoint is what a future screen would call |
| A status-transition trigger on `bp_detection_finding` | Separate change to a table the gateway also writes | Engine obeys the existing finding edges itself (§8.2) |

`src/services/reconciliation.py` and `DiscrepancyDetectionAgent` are **left untouched**;
the deal summary depends on the former. The duplicate-invoice detector is left
untouched and **read** as an input (check 10).

## 3. Data facts this design rests on

Measured on `bp_testdb`, 2026-09-24. The corpus is almost all seeded test data
(`created_by='testdata'` on 12,398 of 12,408 invoices), so these are shapes, not truths.

| Fact | Value | Consequence |
|---|---|---|
| Invoice lines matching a PO line on `po_id` + `item_id` | 45,966 of 55,483 | Primary line link is exact |
| …of which same unit price / same quantity | 97.9% / 97.8% | Most lines are MATCH |
| Invoice lines naming a PO but matching no PO line | 829 | Check 12 has real work |
| Invoice vs PO currency differs / supplier differs | 0 / 0 | Checks 7–8 should stay silent |
| Unit of measure differs between linked lines | 0 | No UoM conversion or "systematic mapping" pattern needed now |
| Invoices with `tax_percent` set | 10 of 12,408 | Tax is checked from the implied rate (tax ÷ net), not the stated one |
| net + tax = gross | 12,408 of 12,408 | Check 4 should stay silent |
| Credit notes (negative `invoice_amount`) | 191 | Count negative toward a PO's running total |
| POs whose invoices exceed the PO total by > 1% | 3,412 of 5,040 | Mostly repeated identical invoices — expect ~3,400 Blocked deals |
| `confidence_score` present on invoices / POs | 10 of 12,408 / 4 of 5,041, scale 0–100 | Missing confidence means "not reported", factor 1.0 |
| FX | `proc.bp_fx_rates` is a USD-based **snapshot** (`fetched_at`), not dated history; `facts/fx.py:resolve_fx` resolves it | Exposure uses the snapshot rate; the rate and its `fetched_at` are recorded |
| Every PO has `quote_reference` | 5,041 of 5,041 | PO → quote link is exact |

## 4. Architecture

```
src/services/triage/
  __init__.py
  model.py        dataclasses: DocumentSet, Doc, Line, Link, Result, Finding, Verdict
  loader.py       load_deal_sets(cur, deal_ids) -> dict[deal_id, DocumentSet]   (batched SQL)
  normalise.py    Decimal money, FX to GBP (via facts/fx.resolve_fx, cached per batch),
                  supplier-name cleanup, confidence 0-100 -> 0-1
  link.py         invoice -> PO; invoice line -> PO line; PO -> quote; link confidence
  checks.py       the thirteen checks; each returns list[Result] with an outcome
  tolerance.py    resolve_tolerance(check, ctx) -> Tolerance (reads governed limits)
  group.py        results -> findings (cascade, duplicate→overbilling, uplift, offset, bad ref)
  score.py        materiality, bands, overrides
  verdict.py      findings + notes -> Verdict
  writer.py       audit rows, finding upsert/supersede, run row (one transaction per batch)
  engine.py       run_triage(deal_ids, mode) -> RunReport ; triage_deal(deal_id)
  report.py       RunReport -> printed/JSON scale report
```

Every module except `loader.py` and `writer.py` is **pure**: it takes dataclasses and
returns dataclasses, with no database, clock or network. This is what makes each step
testable on hand-built deals and keeps the engine fast.

### Data flow

```
load_deal_sets(batch of 200 deal_ids)      fixed number of queries per batch
  → normalise → link → checks → group → score → verdict
  → writer: bp_triage_result (all), bp_detection_finding (S1, S2), bp_triage_finding (map)
```

### Authority map (profile `procure_to_pay`)

| Field | Authoritative | Checked against | Rule |
|---|---|---|---|
| Unit price | PO line (quote line if the PO line has no price) | Invoice line | Invoice ≤ PO + tolerance |
| Quantity | PO line (until goods receipts exist) | Cumulative invoiced quantity | Cumulative ≤ PO + over-allowance |
| Total value | PO total (ceiling) | Cumulative invoices − credit notes | ≤ PO + tolerance |
| Tax | Invoice | Allowed rates (policy) | Implied rate is an allowed rate |
| Supplier | PO | Invoice | Equal |
| Currency | PO | Invoice | Equal |
| Dates | PO order date | Invoice date | Invoice not before order |
| Description | PO line | Invoice line | Must not conflict; max S3 |

## 5. Linking and the thirteen checks

### 5.1 Linking

1. **Invoice → PO:** `bp_invoice_trgt.po_id` (line-level `po_id` if the header's is null).
2. **Invoice line → PO line:**
   - same `po_id` and `item_id` → link confidence **1.0**;
   - otherwise, within that PO, best candidate by description similarity with the line
     amount as a tiebreak → link confidence = similarity score;
   - below `min_link_confidence` → unlinked.
3. **PO → quote:** `bp_purchase_order_trgt.quote_reference` → `bp_quote_trgt.quote_id`.
4. **Credit notes** link like invoices and count negative in cumulative totals.

### 5.2 Checks

Each check emits exactly one outcome per compared pair (triage spec §5): `MATCH`,
`WITHIN_TOL`, `EXPLAINED`, `ABSENT_SUBORDINATE`, `ABSENT_AUTHORITATIVE`, `CONFLICT`,
`UNVERIFIABLE`.

| # | Check (`rule_id`) | Compared | Failure outcome and exposure |
|---|---|---|---|
| 1 | `unit_price` | Invoice line vs PO line (quote fallback) | CONFLICT above tolerance (over and under have separate tolerances). Exposure = price diff × invoiced qty |
| 2 | `quantity` | Cumulative invoiced qty on a PO line vs PO line qty | CONFLICT above over-allowance; exposure = excess × PO price. Under = EXPLAINED (partial invoicing) |
| 3 | `line_arithmetic` | qty × price vs line amount, on the invoice | CONFLICT outside rounding tolerance; exposure = the difference |
| 4 | `invoice_totals` | Σ lines = net; net + tax = gross | CONFLICT outside rounding tolerance × line count; exposure = the difference |
| 5 | `cumulative_total` | Σ invoices − credit notes vs PO total | CONFLICT above tolerance; exposure = overage. **Always S1** |
| 6 | `tax_rate` | tax ÷ net vs `allowed_tax_rates` | CONFLICT if no allowed rate is within rounding; exposure = tax − nearest allowed-rate tax |
| 7 | `currency` | Invoice vs PO | CONFLICT; exposure = invoice net. **Always S1** |
| 8 | `supplier` | Invoice `supplier_id` vs PO `supplier_id` | CONFLICT; criticality 1.0; exposure = invoice net |
| 9 | `invoice_date` | Invoice date vs PO order date | CONFLICT if before; exposure = 0, so its severity comes from the **min S2** override |
| 10 | `duplicate` | Open `duplicate_invoice` rows in `bp_extraction_discrepancy` for the deal's invoices | CONFLICT; exposure = duplicate invoice net. **Always S1** |
| 11 | `description` | Invoice vs PO line description where `item_id` matches | CONFLICT on low similarity; **max S3** |
| 12 | `unlinked_line` | Invoice line with no PO line | ABSENT_AUTHORITATIVE; exposure = line amount |
| 13 | `payment_terms` | Invoice vs PO `payment_terms` (normalised text) | CONFLICT; exposure = 0, so its severity comes from the **min S2** override |

**EXPLAINED:** several invoice lines for the same PO line whose sum equals it (split);
partial invoicing (check 2 under). **ABSENT_SUBORDINATE:** a field present on the invoice
and absent on the PO (e.g. line `delivery_date`). Both become S3 notes.

**UNVERIFIABLE:** a would-be CONFLICT where either document's extraction confidence is
below `min_extraction_confidence`, or the line link is below `min_link_confidence` but
above the unlinked cut-off, becomes UNVERIFIABLE, **capped at S2** (min S2 on money and
supplier fields — §7.3). A missing confidence score means "not reported" and counts as
1.0; the audit row records that it was not reported.

### 5.3 Tolerances

Read only through `tolerance.resolve_tolerance(check, ctx)`, which reads a new governed
limit row `triage_tolerances` via `governed_limits.limit()`. A missing value raises
`LimitUnavailable` and **the run stops before writing anything**. Starting values
(from the triage spec §14 example; tune after the backfill):

| Key | Start value |
|---|---|
| `unit_price_over_pct` / `unit_price_over_abs` / combine | 1.0 / 5.00 GBP / min |
| `unit_price_under_pct` | 5.0 |
| `quantity_over_pct` | 5.0 |
| `rounding_per_line` | 0.01 |
| `cumulative_total_pct` / `cumulative_total_abs` / combine | 0.5 / 50.00 GBP / min |
| `allowed_tax_rates` | [0, 5, 20] |
| `min_link_confidence` / `unlinked_below` | 0.80 / 0.50 |
| `min_extraction_confidence` | 0.70 |
| `description_min_similarity` | 0.40 |
| `materiality_pct_of_total` / `materiality_floor` / `materiality_ceiling` | 0.5 / 25 / 5000 GBP |
| `band_s1` / `band_s2` | 70 / 40 |
| `uplift_min_lines` / `uplift_same_pct_within` | 3 / 0.1 |
| `batch_size` | 200 |

`combine: min` means the stricter of the percentage and absolute amounts, as in the
triage spec §7.4.

## 6. Grouping (one cause, one finding)

Applied to scored-eligible results (CONFLICT, ABSENT_AUTHORITATIVE, UNVERIFIABLE). Each
finding has one **cause** and a list of **effects**; effects are not scored separately.

1. **Cascade.** A line-level unit-price or quantity CONFLICT is the cause; the
   resulting differences in line amount, invoice net, tax, gross and the PO's running
   total are its effects. A total-level CONFLICT becomes its own finding **only for the
   part not explained** by line-level causes (difference − Σ line exposures, beyond
   rounding tolerance).
2. **Duplicate absorbs over-billing.** If a PO's cumulative overage is ≤ the sum of
   duplicate invoice amounts on that PO, check 5 becomes an effect of the duplicate
   finding(s). Any remainder is its own over-billing finding.
3. **Uniform uplift.** ≥ `uplift_min_lines` unit-price CONFLICTs on one invoice whose
   percentage difference agrees within `uplift_same_pct_within` points become one
   finding ("Prices 3.5% above PO on 42 lines, +£1,840"); exposure = sum.
4. **Offsetting.** Quantity CONFLICTs on one invoice that net to within rounding of
   zero become one finding, severity **S2**, exposure = **gross** (sum of absolute
   exposures), so errors cannot hide each other.
5. **Bad PO reference.** An invoice whose `po_id` matches no PO yields one `linking`
   finding, not one `unlinked_line` per line.

Finding severity = highest severity among its causes. Finding exposure = gross.

## 7. Scoring

### 7.1 Materiality (triage spec §8.2)

```
threshold  = clamp(materiality_pct_of_total% × invoice gross in GBP, floor, ceiling)
ratio      = exposure_gbp / threshold
impact     = clamp(50 × (1 + log10(ratio)), 0, 100)        (0 when exposure is 0)
score      = clamp(impact × criticality × confidence, 0, 100)
S1 ≥ band_s1 · S2 ≥ band_s2 · S3 otherwise
```

Criticality: party/currency 1.0 · money/quantity 0.8 · dates/references 0.6 ·
description 0.2. Confidence = extraction confidence × link confidence. `domain_modifier`
is 1.0 for this profile and is omitted.

Worked example (invoice gross £9,000 → threshold £45): +£450 price → impact 100 →
score 80 → **S1**; +£45 → 40 → **S2**; +£10 → impact ≈17, score ≈14 → **S3**.

If no FX rate resolves for a document's currency, the finding is still raised, capped
at **S2**, and its text says "no FX rate".

### 7.2 Outcome to severity

| Outcome | Severity |
|---|---|
| MATCH, WITHIN_TOL | S0 |
| EXPLAINED, ABSENT_SUBORDINATE | S3 (note) |
| CONFLICT, ABSENT_AUTHORITATIVE | scored |
| UNVERIFIABLE | scored, capped at S2 |

### 7.3 Overrides (applied last; nothing earlier can soften them)

- **Always S1:** `cumulative_total` beyond tolerance; `currency`; `duplicate`.
- **Min S2:** `payment_terms`; `invoice_date`; any UNVERIFIABLE on a money or supplier
  field.
- **Max S3:** `description`.

## 8. Storage

### 8.1 Where each severity goes

| Severity | `bp_triage_result` (audit) | `bp_detection_finding` (Action Centre) | Verdict |
|---|---|---|---|
| S1 | yes | `severity='critical'`, `blocks_promotion=true` | counted |
| S2 | yes | `severity='warning'` | counted |
| S3 | yes | **no** — the Action Centre cannot collapse notes, and split lines alone would add tens of thousands of rows | counted as notes |
| S0 | yes | no | — |

`bp_detection_finding` columns used: `engine_run_id` (run id), `rule_id` (check or
grouping pattern), `category` (price, quantity, overbilling, duplicate, tax, currency,
supplier, date, linking, arithmetic), `severity`, `doc_type`, `doc_pk` (invoice id),
`deal_id`, `pipeline_record_id` (= deal_id, per the gateway entity), `field_name`,
`observed_value` (claim value with its document id), `expected_value` (authoritative
value with its document id), `delta` (exposure, GBP and document currency), `confidence`,
`notes` (the plain-language finding text and its effects line), `status='open'`,
`lifecycle_status='open'`. `stage_id` stays null: stage gating is not in scope.

Finding text follows triage spec §10 "Writing findings": field and direction first,
both values with document ids, exposure in GBP (and document currency where it
differs), effects on one line, known reason where there is one.

### 8.2 New tables (migration + rollback in `deploy/sql/`, applied to both DBs)

- `proc.bp_triage_run` — `run_id uuid PK`, `mode` (backfill | scheduled | single),
  `started_at`, `finished_at`, `config_fingerprint` (hash of the resolved
  `triage_tolerances` values), `config_values jsonb`, `deal_count`, `failed_deals jsonb`,
  `report jsonb`.
- `proc.bp_triage_result` — `result_id bigserial PK`, `run_id`, `deal_id`, `rule_id`,
  `claim_doc`, `claim_line`, `auth_doc`, `auth_line`, `field_name`, `claim_value`,
  `auth_value`, `outcome`, `severity`, `exposure_gbp`, `score`, `score_inputs jsonb`,
  `tolerance jsonb` (value + policy key it came from), `fingerprint`, `finding_id`
  (nullable). Indexes `ix_bp_triage_result_run`, `ix_bp_triage_result_deal`.
- `proc.bp_triage_finding` — `fingerprint text PK`, `finding_id` (→
  `bp_detection_finding`), `deal_id`, `first_run_id`, `last_run_id`, `last_severity`.

Fingerprint = hash of (deal_id, rule_id, cause key), where the cause key is the
invoice id plus line (or PO id for PO-level findings). It is stable across runs.

### 8.3 Re-run rules

The engine moves `bp_detection_finding` only along the finding edges already declared
in `proc.bp_lifecycle_transition` (open → resolved | ignored | superseded; superseded is
final), and only ever from `open`:

| On re-run | Action |
|---|---|
| Fingerprint seen, finding open, problem still present | Update values/exposure/severity in place; same `finding_id` |
| Fingerprint seen, finding open, problem gone | `status='superseded'`, `lifecycle_status='resolved'` |
| Fingerprint seen, finding resolved or ignored by a person | Leave alone. If the new severity is **higher** than `last_severity`, open a new finding whose notes reference the old id |
| New fingerprint | Insert |

### 8.4 Transactions

`get_conn()` is autocommit, so the writer explicitly opens a transaction per batch
(`conn.autocommit = False` … commit), making each batch all-or-nothing. The run row is
written first and finalised last.

## 9. Running it

### 9.1 Entry points

- **Scheduler job** (`backend_scheduler`): selects deals whose `_trgt` rows have
  `last_modified_date` later than the deal's last triage run, and runs them in batches.
  A job, not a promotion hook, so a triage fault can never break promotion.
- **`POST /triage/deals/{deal_id}/run`** (requires `require_user`) — triage one deal now.
- **`GET /triage/deals/{deal_id}`** — verdict, counts and findings for one deal.
- **`scripts/triage_backfill.py`** — `--all`, `--deals A,B,…`, `--dry-run` (computes
  and prints the report; writes nothing).
- **`scripts/triage_rollback.py --run-id X`** — deletes that run's audit rows and the
  findings it created (`bp_triage_finding.first_run_id = X`) whose `status` and
  `lifecycle_status` are both still `open` and whose `owner`, `due_date` and
  `resolved_by` are all null — i.e. no person has touched them. Everything else stays.

### 9.2 Verdict

| Verdict | Condition |
|---|---|
| Blocked | any S1 |
| Needs review | S2, no S1 |
| Matched with notes | S3 only |
| Matched | none |
| Incomplete | a PO with no invoice yet, or an invoice with no PO |

Summary line: "Blocked · 2 findings need action · 14 notes · exposure £570.00".

### 9.3 Failure handling

- Missing policy value → run aborts before any write.
- One deal raises → recorded in `failed_deals` with the error; the batch continues
  without it.
- Batch write fails → that batch rolls back; the run records it as failed and continues.

## 10. Testing

1. **Unit (pure, no DB, no GPU):** for every check, grouping pattern and override, one
   hand-built deal that trips it and one that does not; the §7.1 worked examples;
   tolerance combine/min; fingerprint stability.
2. **Guards proven by breaking them:** for each always-S1 rule and the max-S3 rule,
   temporarily disable it and confirm its test fails, then restore.
3. **Planted-error recall:** load 200 real deals, apply known mutations **in memory
   only** (price +10% on a line, extra quantity, duplicate invoice, currency swap, line
   with no PO line), run the engine, report recall per mutation type. Source data is
   never modified.
4. **Database (`PROCWISE_TEST_LIVE_DB=1`):** loader returns the expected set; writer is
   idempotent (two runs → identical finding set); resolved finding not reopened;
   superseded on disappearance; rollback removes only untouched rows.
5. **Live proof:** on the running local stack against `bp_testdb` — full backfill
   within the 15-minute target; the report; 10 findings hand-checked against raw rows;
   findings returned by the gateway's `GET /discrepancies`.

## 11. Known limitations

- FX uses the rate snapshot, not the rate on the invoice date; the rate and its
  `fetched_at` are stored with each finding.
- Quantity is checked against the PO, not a goods receipt.
- Bank-detail changes are not checked (no invoice-side bank data).
- On this corpus ~3,400 deals are expected to be Blocked, driven by the seeded
  repeated invoices. That is correct behaviour for the data; the rollback script
  removes the backfill if it is not wanted.
