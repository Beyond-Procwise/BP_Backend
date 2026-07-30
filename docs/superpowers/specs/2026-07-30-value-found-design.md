# Value Found — the money number (W1)

**Date:** 2026-07-30 (extended same day with quick-payback additions)
**Status:** Approved design, pending implementation plan
**Origin:** Product gap review (impact roadmap item W1)

## What this is, in one sentence

One evidence-backed headline — *"£47,200 value found · £3,100 recovered"* — rolled up
from hard-evidence finding sources, shown on the Home screen and the SpendIQ Dashboard,
with a drill-down that lands on the real evidence and a one-click path from *found* to
*recovered*.

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
4. **Quick-payback additions approved:** the Query-it recovery action, the
   duplicate-invoice detector, the weekly value digest, supplier breakdown + finding
   age in the drawer, and the clarity fixes (time window, methodology note, naming).

## Phases

| Phase | Ships | Depends on |
|---|---|---|
| **1 — The number** | Aggregation service, `GET /spendiq/value-summary`, Home + Dashboard tiles, findings drawer (with supplier breakdown + age), discrepancy outcome columns | nothing |
| **2 — Duplicate-invoice detector** | New hard-evidence finding source feeding the existing discrepancy table | nothing (lands in Phase 1's headline automatically) |
| **3 — Query-it recovery action** | One-click supplier query email on over-billing and duplicate findings, via the existing drafting engine + approval gate; reply resolves the finding as recovered | Phase 1 |
| **4 — Weekly value digest** | Proactive weekly email: found / recovered / top findings | Phase 1 |

Each phase is independently shippable; the tile never waits on the later phases.

## Architecture

A read-model, not a new store. The source tables already persist every finding, its
lifecycle state, and even realised savings — a new "value ledger" table would duplicate
state and drift. One new BP_Backend service aggregates at query time.

```
proc.bp_extraction_discrepancy  ─┐   (incl. new duplicate-invoice findings, Phase 2)
proc.bp_opportunity             ─┼─► value_summary_service ─► GET /spendiq/value-summary
benchmark price-history deltas  ─┘         (BP_Backend)              │
                                                     Home hero tile ─┴─ SpendIQ Dashboard tile
                                                            └── findings drawer
                                                                 ├─ deep links (existing screens)
                                                                 └─ "Query it" action (Phase 3)
```

* **New service:** `src/services/value_summary_service.py` — pure aggregation over the
  sources; no writes.
* **New router:** `GET /spendiq/value-summary` in BP_Backend (the UI already calls
  BP_Backend `/spendiq/*` endpoints via its authenticated `ai` axios — same pattern as
  `/spendiq/deals/:id`).
* **UI:** a hero tile on `Home.jsx` and a Dashboard tile via the SpendIQ engine's
  `SD(...)` slot pattern; both open the same findings drawer. No new screen.

## Aggregation rules

### Tiers

| Tier | Source | Rule |
|---|---|---|
| **Verified** (headline) | Discrepancies | Findings with `issue_type IN ('amount_over_po', 'line_amount_over_po', 'duplicate_invoice')` — in **any** status except resolved-`false_positive`. (Historical rows resolved before the outcome column existed have outcome null: they stay in *verified found* but never count as *recovered*.) |
| **Verified** (headline) | Opportunities | `financial_impact_gbp` where `stage = 'agreed'` — a human has confirmed it. |
| **Recovered** (companion) | Opportunities | `realised_savings_gbp` where `stage = 'realised'`. |
| **Recovered** (companion) | Discrepancies | Findings resolved with outcome `recovered`; amount = the recorded recovered amount, defaulting to the finding's delta. |
| **Potential** (quiet) | Opportunities | `financial_impact_gbp` where `stage = 'identified'` — mined but not yet human-confirmed. Excludes `rejected` and `closed`. |
| **Potential** (quiet) | Benchmark | Positive price-vs-own-history deltas on open quote lines that clear the evidence threshold (≥3 comparable observations). Near zero today because of the entity-resolution gap; grows automatically when that is fixed. |

Implementation note: verify the full `stage` vocabulary against live `bp_opportunity`
before coding the mapping — the values above (`identified/agreed/realised/rejected/closed`)
are the ones observed in `opportunity_store.py` and `opportunity_dashboard.py`; any
additional stage found must be explicitly assigned to a tier, never silently dropped.

### Time window (clarity fix)

*Found-to-date* means **all time** in v1. The response carries `"since": null` (all
time); a financial-year or rolling-window filter is a later query parameter, not a
redesign. Every finding carries its `found_at` date so any future window is a filter,
not a recomputation.

### Delta arithmetic — known trap

`bp_extraction_discrepancy.computed_value` has **two conventions**: for some issue types
the signed value is *already the delta*, for others the delta is `observed − expected`
(see the 2026-07-24 £950-vs-£27,660 incident, and the gateway's strict `num()` parser in
`spendiq.service.ts getDiscrepancies`). The service must branch on `issue_type`, reuse
the same strict numeric parse (whole-value only — a SHA-256 `raw_value` must parse to
null, not a number), and unit-test both conventions.

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
  "since": null,
  "by_supplier": [
    {"supplier_name": "Techworld", "verified_found_gbp": 8100.0, "finding_count": 3}
  ],
  "findings": [
    {
      "id": "…",
      "tier": "verified",
      "source": "discrepancy | opportunity | benchmark",
      "amount_gbp": 950.0,
      "converted_from": null,
      "title": "Invoice INV-1042 bills £950 over PO-2210",
      "supplier_name": "…",
      "deal_id": "…", "doc_pk": "…",
      "found_at": "…", "age_days": 45,
      "link": {"screen": "actions | opportunities | analysis-report", "id": "…"},
      "status": "open | recovered | …",
      "queryable": true,
      "superseded_by": null
    }
  ],
  "sources": {"discrepancies": "ok", "opportunities": "ok", "benchmark": "ok"},
  "generated_at": "…"
}
```

* `by_supplier` is the drawer's grouping — one `GROUP BY` over verified findings; it also
  hands the negotiation agents their opening argument.
* `age_days` drives the urgency cue ("found 45 days ago, untouched").
* `queryable` marks findings the Phase-3 action supports (over-billing and duplicate
  types with a resolvable supplier).
* `sources` reports per-source availability. If a source query fails, its key is
  `"unavailable"`, its contribution is omitted, and the UI shows the partial figure with
  an "excludes N unavailable source(s)" note — a partial honest number, never a silently
  wrong one.
* Deep links reuse existing screens: Action Centre for discrepancies, Opportunities view
  for opportunities, the deal analysis report where a `deal_id` exists.

## Schema change

`bp_extraction_discrepancy` gains two nullable columns:

* `resolution_outcome text` — `recovered | accepted | false_positive`
* `recovered_amount numeric` — optional; defaults to the finding's delta when outcome is
  `recovered`

The gateway's `POST /spendiq/discrepancies/resolve` accepts the two new optional fields
and writes them. Existing resolved rows (outcome null) count toward *verified found* but
not *recovered* — we do not guess what an undifferentiated historical "resolved" meant.
Migration follows the house pattern: SQL file applied to both `bp_sqldb` and `bp_testdb`
with md5 record.

## Phase 2 — Duplicate-invoice detector

Classic quick-payback AP recovery: the same obligation billed twice. The upload-time
content-hash dedup only catches byte-identical *files*; this detector catches distinct
documents billing the same thing.

* **Rule (deliberately conservative):** two invoices from the same resolved supplier
  with equal totals (post-normalisation) and the same PO reference, or invoice
  references within a trivial edit distance, within a 90-day window. Flag the *later*
  invoice; delta = its full amount.
* **Output:** a row in `bp_extraction_discrepancy` with `issue_type = 'duplicate_invoice'`,
  severity high, evidence in `notes` (both doc refs). It then flows into the headline,
  the Action Centre, and the Query-it action with zero further wiring.
* **Runs** after `_trgt` promotion (same hook as the three-way match) and as a one-off
  backfill over existing invoices.
* **Honesty:** genuinely recurring equal charges exist (monthly service fees). The PO/
  reference condition — not amount alone — is what keeps false positives rare, and
  `false_positive` resolution removes any that slip through.

## Phase 3 — Query-it recovery action

The distance from *found* to *recovered*, collapsed to one click. Scope is deliberately
narrow: over-billing (`amount_over_po`, `line_amount_over_po`) and `duplicate_invoice`
findings only — the cases where the evidence is contractual and suppliers rarely argue.

* **Flow:** drawer button → the existing email-drafting engine composes a supplier query
  citing invoice ref, PO ref, and the delta (grounded in the finding's stored values —
  the draft must not restate figures the model generated itself) → existing approval
  gate (HITL) → send via the existing email path → the finding is stamped `query_sent`.
* **Close:** when a reply arrives (EmailWatcher) or manually, the user resolves the
  finding with outcome `recovered` (amount editable) or `accepted`. No auto-resolution
  from reply parsing in this phase — that is W2's job later.
* **State:** `query_sent` is a note/status on the discrepancy row, not a new workflow
  table. One new email template in the style engine, governed via `bp_prompt`.
* **Guard rails:** action is per-finding (no bulk send), only for findings with a
  resolvable supplier email, and every send lands in `bp_agent_actions`.

## Phase 4 — Weekly value digest

A scheduled weekly email to the product's users: value found this week, recovered this
week, top 3 open findings by amount (with age), one link into the drawer. Composed from
the same `/spendiq/value-summary` data (`found_at` gives the weekly slice), sent through
the existing email infrastructure, template governed via `bp_prompt`. No new data model.
Opt-out is a user setting; send nothing when the week has no findings and nothing was
recovered — an empty digest trains people to ignore it.

## UI

* **Home hero tile** (next to the spend KPI): headline `£X value found`, companion
  `£Y recovered`, quiet sub-line `£Z further potential · N findings →`.
* **SpendIQ Dashboard tile**: same figures via an `SD('home.valueSummary', …)`-style
  slot; the engine re-boots when the query resolves (established pattern).
* **Findings drawer**: opened from either tile. Grouped by supplier (from
  `by_supplier`), each row: tier chip, amount, title, source, age ("found 45 days ago"),
  deal/doc reference, deep link, and — Phase 3 — the Query-it button on `queryable`
  rows. Zero findings renders an honest empty state ("No verified findings yet — upload
  documents to begin"), never sample rows.
* **Methodology note (clarity fix):** an info tooltip on the tile stating in one
  sentence what counts as verified vs potential and that converted amounts use the shown
  FX rate. Trust in the number is the product's whole brand.
* **Naming (clarity fix):** the vocabulary is **value found / recovered / potential**
  everywhere this feature touches. "Savings" is reserved for genuine price reductions;
  an over-billing correction is recovery, not saving. The Opportunities screen's KPI
  labels adopt the same vocabulary as a copy-only change in this work.
* Respects the existing display-currency selector.

## Error handling

* DB unavailable → endpoint returns 200 with all sources `unavailable`, zeros, and the
  UI tile shows an unavailable state (matches the product's REDUCED-mode philosophy).
* Benchmark computation too slow → the benchmark source has a per-request time budget;
  on timeout it reports `unavailable` rather than delaying the headline. The other
  sources are simple SQL and fast.
* A finding whose document can no longer be resolved keeps its amount (found-to-date is
  cumulative) but its link degrades gracefully to the Action Centre list.
* Query-it send failure leaves the finding untouched (no `query_sent` stamp) and
  surfaces the error in the drawer row.

## Known caveats (accepted, documented on-screen where relevant)

* **Deal mis-grouping (open bug C2)** — a finding's `deal_id` comes from the deal-document
  map and can be wrong until that bug is fixed. Amounts are per-document and unaffected;
  only the deal *label* on a finding row can mislead. Not a blocker.
* **Benchmark tier is ~zero today** (entity-resolution gap A2). The tier ships anyway so
  the number grows without further UI work when matching is fixed.
* **Supplier grouping inherits supplier-name variants** (the entity-resolution gap
  again): the same supplier under two spellings appears as two groups until aliasing
  improves. Group by the resolver's canonical name where one exists.

## Testing & verification

1. **Unit (BP_Backend):** tier mapping per stage and issue type; both `computed_value`
   conventions; dedup precedence incl. `superseded_by`; FX conversion provenance; strict
   numeric parse rejects hash-like values; per-source failure isolation; duplicate
   detector positive/negative cases (incl. recurring monthly fee non-flag); Query-it
   draft grounding (figures byte-equal to stored finding values).
2. **Contract (UI):** tile renders all three figures; drawer grouping, age, deep links;
   empty, partial-source, and unavailable states; display-currency conversion;
   methodology tooltip; Query-it button only on `queryable` rows.
3. **Live verification (house rule):** on the running local server against live
   `bp_sqldb` — the £950 over-billing finding must appear in *verified found*; the
   Query-it flow must produce an approvable draft citing the correct refs and delta;
   resolving with outcome `recovered` must move £950 to *recovered*; Home and Dashboard
   must show identical figures; the duplicate-detector backfill must report its finding
   count and each finding must trace to two real documents.

## Out of scope (deliberately)

* No new persistence/ledger table.
* No new screen — drill-down reuses existing screens.
* No trend-over-time chart of value found (cheap later addition once the endpoint exists).
* No ROI-vs-subscription-cost counter (needs commercial inputs the product doesn't hold).
* No evidence-pack PDF export (report builder integration, later).
* No auto-resolution from supplier reply parsing (that is W2's loop; Phase 3 stops at
  human resolve).
* No change to how existing detectors find things — apart from the new duplicate
  detector, this feature only *presents* what they find.
