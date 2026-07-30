# Value Found — the money number (W1)

**Date:** 2026-07-30
**Status:** Approved design, pending implementation plan
**Origin:** Product gap review (impact roadmap item W1)

## What this is, in one sentence

One evidence-backed headline — *"£47,200 value found · £3,100 recovered"* — rolled up
from the three finding sources that already exist, shown on the Home screen and the
SpendIQ Dashboard, with a drill-down that lands on the real evidence.

## Decisions taken (with the user, 2026-07-30)

1. **Two-tier number.** The headline counts only *verified* findings (document-level
   evidence or human-confirmed opportunities). A visually quieter companion figure shows
   *potential* (model-estimated). We do not publish one big soft number.
2. **Placement: Home + SpendIQ Dashboard.** Same figure in both places, from one
   endpoint. Never computed twice.
3. **Lifecycle: found-to-date + recovered.** The headline is cumulative: successfully
   resolved findings stay in *found* and additionally count in *recovered*, so working
   the queue never shrinks the headline. The only removal is a finding dismissed as a
   false positive — that was never value, and keeping it would be fabrication.

## Architecture

A read-model, not a new store. The three source tables already persist every finding,
its lifecycle state, and even realised savings — a new "value ledger" table would
duplicate state and drift. One new BP_Backend service aggregates at query time.

```
proc.bp_extraction_discrepancy  ─┐
proc.bp_opportunity             ─┼─► value_summary_service ─► GET /spendiq/value-summary
benchmark price-history deltas  ─┘         (BP_Backend)              │
                                                     Home hero tile ─┴─ SpendIQ Dashboard tile
                                                            └── findings drawer (deep links)
```

* **New service:** `src/services/value_summary_service.py` — pure aggregation over the
  three sources; no writes.
* **New router:** `GET /spendiq/value-summary` in BP_Backend (the UI already calls
  BP_Backend `/spendiq/*` endpoints via its authenticated `ai` axios — same pattern as
  `/spendiq/deals/:id`).
* **UI:** a hero tile on `Home.jsx` and a Dashboard tile via the SpendIQ engine's
  `SD(...)` slot pattern; both open the same findings drawer. No new screen.

## Aggregation rules

### Tiers

| Tier | Source | Rule |
|---|---|---|
| **Verified** (headline) | Discrepancies | Findings with `issue_type IN ('amount_over_po', 'line_amount_over_po')` — the over-billing delta — in **any** status except resolved-`false_positive`. (Historical rows resolved before the outcome column existed have outcome null: they stay in *verified found* but never count as *recovered*.) |
| **Verified** (headline) | Opportunities | `financial_impact_gbp` where `stage = 'agreed'` — a human has confirmed it. |
| **Recovered** (companion) | Opportunities | `realised_savings_gbp` where `stage = 'realised'`. |
| **Recovered** (companion) | Discrepancies | Findings resolved with outcome `recovered`; amount = the recorded recovered amount, defaulting to the finding's delta. |
| **Potential** (quiet) | Opportunities | `financial_impact_gbp` where `stage = 'identified'` — mined but not yet human-confirmed. Excludes `rejected` and `closed`. |
| **Potential** (quiet) | Benchmark | Positive price-vs-own-history deltas on open quote lines that clear the evidence threshold (≥3 comparable observations). Near zero today because of the entity-resolution gap; grows automatically when that is fixed. |

Implementation note: verify the full `stage` vocabulary against live `bp_opportunity`
before coding the mapping — the values above (`identified/agreed/realised/rejected/closed`)
are the ones observed in `opportunity_store.py` and `opportunity_dashboard.py`; any
additional stage found must be explicitly assigned to a tier, never silently dropped.

### Delta arithmetic — known trap

`bp_extraction_discrepancy.computed_value` has **two conventions**: for some issue types
the signed value is *already the delta*, for others the delta is `observed − expected`
(see `docs` note from the 2026-07-24 £950-vs-£27,660 incident, and the gateway's strict
`num()` parser in `spendiq.service.ts getDiscrepancies`). The service must branch on
`issue_type`, reuse the same strict numeric parse (whole-value only — a SHA-256
`raw_value` must parse to null, not a number), and unit-test both conventions.

### Double-counting guard

The same £ can surface as a discrepancy *and* an opportunity *and* a benchmark delta.
Precedence by evidence strength: **discrepancy > opportunity > benchmark**, deduplicated
on `(deal_id, doc_pk, normalised item)`. A finding counts once, in its strongest source.
When a lower-precedence duplicate is suppressed, it is still returned in the findings
list flagged `superseded_by`, so the drawer can explain rather than mysteriously omit.

### Currency

* Opportunity figures are already native GBP (`financial_impact_gbp`,
  `realised_savings_gbp`).
* Discrepancy and benchmark amounts are in document currency. Convert via the existing
  FX service (`fx` router's rate source); every converted amount carries
  `converted_from: {currency, amount, rate_date}` in the response. **Never** render a
  `£` on an unconverted foreign amount and never invent a rate (no-fabrication rule).
* The UI display-currency selector then converts the GBP total for display exactly as it
  does for the spend KPI.

## API contract

`GET /spendiq/value-summary`

```json
{
  "verified_found_gbp": 47200.0,
  "recovered_gbp": 3100.0,
  "potential_gbp": 12400.0,
  "finding_count": 9,
  "findings": [
    {
      "id": "…",
      "tier": "verified",
      "source": "discrepancy | opportunity | benchmark",
      "amount_gbp": 950.0,
      "converted_from": null,
      "title": "Invoice INV-1042 bills £950 over PO-2210",
      "deal_id": "…", "doc_pk": "…",
      "link": {"screen": "actions | opportunities | analysis-report", "id": "…"},
      "status": "open | recovered | …",
      "superseded_by": null
    }
  ],
  "sources": {"discrepancies": "ok", "opportunities": "ok", "benchmark": "ok"},
  "generated_at": "…"
}
```

* `sources` reports per-source availability. If a source query fails, its key is
  `"unavailable"`, its contribution is omitted, and the UI shows the partial figure with
  a "excludes N unavailable source(s)" note — a partial honest number, never a silently
  wrong one.
* Deep links reuse existing screens: Action Centre for discrepancies, Opportunities view
  for opportunities, the deal analysis report where a `deal_id` exists.

## Schema change (the one small touch)

`bp_extraction_discrepancy` gains two nullable columns:

* `resolution_outcome text` — `recovered | accepted | false_positive`
* `recovered_amount numeric` — optional; defaults to the finding's delta when outcome is
  `recovered`

The gateway's `POST /spendiq/discrepancies/resolve` accepts the two new optional fields
and writes them. Existing resolved rows (outcome null) count toward *verified found* but
not *recovered* — we do not guess what an undifferentiated historical "resolved" meant.
Migration follows the house pattern: SQL file applied to both `bp_sqldb` and `bp_testdb`
with md5 record.

## UI

* **Home hero tile** (next to the spend KPI): headline `£X value found`, companion
  `£Y recovered`, quiet sub-line `£Z further potential · N findings →`.
* **SpendIQ Dashboard tile**: same figures via an `SD('home.valueSummary', …)`-style
  slot; the engine re-boots when the query resolves (established pattern).
* **Findings drawer**: opened from either tile. Each row: tier chip, amount, title,
  source, deal/doc reference, deep link. Zero findings renders an honest empty state
  ("No verified findings yet — upload documents to begin"), never sample rows.
* Respects the existing display-currency selector.

## Error handling

* DB unavailable → endpoint returns 200 with all sources `unavailable`, zeros, and the
  UI tile shows an unavailable state (matches the product's REDUCED-mode philosophy).
* Benchmark computation too slow → the benchmark source has a per-request time budget;
  on timeout it reports `unavailable` rather than delaying the headline. The other two
  sources are simple SQL and fast.
* A finding whose document can no longer be resolved keeps its amount (found-to-date is
  cumulative) but its link degrades gracefully to the Action Centre list.

## Known caveats (accepted, documented on-screen where relevant)

* **Deal mis-grouping (open bug C2)** — a finding's `deal_id` comes from the deal-document
  map and can be wrong until that bug is fixed. Amounts are per-document and unaffected;
  only the deal *label* on a finding row can mislead. Not a blocker.
* **Benchmark tier is ~zero today** (entity-resolution gap A2). The tier ships anyway so
  the number grows without further UI work when matching is fixed.

## Testing & verification

1. **Unit (BP_Backend):** tier mapping per stage and issue type; both `computed_value`
   conventions; dedup precedence incl. `superseded_by`; FX conversion provenance; strict
   numeric parse rejects hash-like values; per-source failure isolation.
2. **Contract (UI):** tile renders all three figures; drawer rows deep-link; empty,
   partial-source, and unavailable states; display-currency conversion.
3. **Live verification (house rule):** on the running local server against live
   `bp_sqldb` — the £950 over-billing finding must appear in *verified found*, resolving
   it with outcome `recovered` must move £950 to *recovered*, and Home and Dashboard
   must show identical figures.

## Out of scope (deliberately)

* No new persistence/ledger table.
* No new screen — drill-down reuses existing screens.
* No trend-over-time chart of value found (a later, cheap addition once the endpoint exists).
* No change to how detectors find things — this feature only *presents* what they find.
