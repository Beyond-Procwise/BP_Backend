# Deal Clustering & Confirmation — Design

**Date:** 2026-07-20
**Status:** Approved, ready for implementation plan
**Supersedes:** nothing. Extends `2026-06-06-linking-engine-and-promotion-design.md`
(the relationship scorer) and the deal-linking layer of 2026-06-11.

## Problem

An upload batch becomes exactly one deal, regardless of what is in it.

Upload `Analysis Set_190726` (2026-07-19) put **40 documents from 12 suppliers across 4
unrelated categories into a single `deal_id`** — `ANALYSISSET_19072620260719339`. The
batch contains freight lane pricing (`London (Heathrow) → Edinburgh - FTL`) and managed
IT services (`Service Desk (24x7, 3,000 users) — annual`) in the same "deal".

Observed state of that batch:

| | Count |
|---|---|
| Documents | 40 (28 quote, 5 PO, 7 invoice) |
| Distinct suppliers in quotes | 12 |
| Purchase orders | 5, each a **different** supplier |
| Quotes carrying a `po_id` | **0 / 28** |
| PO lines carrying a `quote_number` | **0 / 32** |
| Invoices carrying a `po_id` | 7 / 7 |

Consequences that reach the screen: `bp_deal_overview` aggregates all 40 documents into
one row, so `max(supplier_name)` picks an arbitrary supplier, `price_variance_pct`
compares unrelated totals (the UI renders `+1578.5%` bid spread), and `three_way_match`
reports `true` because it only tests *"this deal contains ≥1 quote AND ≥1 PO AND ≥1
invoice"* — trivially satisfied by a 40-document bucket and therefore meaningless.

## Root cause

The clustering intelligence **already exists and already runs**. It is gated behind a
condition the upload path guarantees will never be true.

`deal_assignment_service._run()` orders its passes:

```python
fwd = _look_forward(cur)    # stamps the user-typed batch name onto every document
back = _look_back(cur)      # then: "if (cur_deal or '').strip(): continue"
```

`_look_forward` reads `proc.process_monitor.deal_id` — which the Node gateway populated
for every file in the batch from the deal *name* the user typed
(`data-integration.service.ts:70-91` `generateDealId()`: `prefix + dateStr + sequence`).
`_look_back`, the pass that forms one deal per canonical PO using relationship scoring,
then finds every document already linked and skips all of them
(`deal_assignment_service.py:423-424`).

Two documentation defects found while tracing this, both worth correcting:

- `linking_engine.py:11-13` states `deal_id` "is assigned by a separate SQL trigger".
  **No such trigger or stored procedure exists.** `deploy/sql/2026-06-11_deal_linking.sql`
  is purely additive DDL. `deal_id` is owned by TypeScript in the gateway.
- `generateDealId()` builds its sequence with `String(count + 1).padStart(2, '0')` over
  the whole `analyse` table, which silently overflows past 99 (this batch got `339`).

## What a deal is

**A deal is a sourcing event**: one requirement, competitively bid by several suppliers,
optionally awarded via a PO and settled by invoices.

The uploaded batch decomposes cleanly under this definition:

| Sourcing event | Competing suppliers | Awarded PO |
|---|---|---|
| Managed IT / MSA | PrimeOps, Fortis, Synapse | PO-2024-0128 Synapse |
| Platform licence (3-yr) | ClearPath, NexusFlow, Orbis | PO-2024-0145 Orbis |
| Professional services | Meridian Consulting, Apex, Vantage | PO-2024-0114 Meridian |
| Freight | Swift, Condor, Meridian Freight | PO-2024-0091 Swift |
| Building works | *(none)* | PO-2024-0163 Caldwell |

Rejected alternatives: **one deal per PO** (leaves quote-only bidding groups orphaned and
cannot represent a sourcing event before award); **one deal per supplier** (destroys the
competitive comparison, which is the point of the Analyse screen).

### Quote versions are not separate bids

28 quotes collapse to **12 distinct base references**; 20 carry a version suffix:

```
CPS-Q-3380 · CPS-Q-3380 (V2) · CPS-Q-3380 (V3 (BAFO))   -> ONE bid, three rounds
```

Counting these as three competing bids inflates supplier counts and corrupts bid-spread
maths. Version collapse is a prerequisite for correct clustering, not a nicety.

## Approach

Add a clustering pass that **proposes** groupings, and a UI to **confirm** them. Reuse
the existing scorer; add no new matching maths.

`linking_engine._line_pair_score` (`linking_engine.py:162-178`) already scores exactly
the product/calculation signals required:

```python
parts.append((0.5,  jaccard(tokens(desc_a), tokens(desc_b))))   # product description
parts.append((0.25, 1.0 if abs(qa - qb) < 1e-9 else 0.0))       # quantity
parts.append((0.25, 1.0 if abs(pa - pb) < 1e-6 else 0.0))       # unit price
```

### Grouping algorithm

Applied per upload batch, most reliable signal first:

1. **Collapse quote versions.** Normalise `quote_id` by stripping a trailing
   `(V<n>...)` suffix. Members of one base reference are rounds of one bid; the highest
   version is the current offer.
2. **Attach invoices to POs** via the existing `po_id` (7/7 populated — deterministic,
   no scoring needed).
3. **Cluster the remainder by line-item similarity.** Score every collapsed-quote ↔ PO
   pair with `score_link(..., "quote_po")`; group above the existing `_BAND_REVIEW`
   threshold (65). Quotes that match each other but no PO form a **quote-only sourcing
   event** — a valid deal with no award yet.
4. **Leave genuine orphans orphaned.** PO-2024-0163 (Caldwell) has no quotes in the
   batch. It must surface as awaiting-quote, never be forced into a neighbouring group.

Supplier identity narrows candidates but must not gate grouping: normalised supplier-key
matching resolves only **3 of 5** POs on this batch. Swift (PO-2024-0091) fails because
`SDP-Q-44120` has a **null `supplier_id`** (3 of 28 quotes and 3 of 7 invoices are
missing it). Product-level matching is what recovers that group — which is precisely why
line-item similarity, not supplier, is the primary clustering signal.

### Proposals are never auto-applied

Clustering writes to new tables and changes nothing that exists:

```sql
CREATE TABLE proc.bp_deal_proposal (
    proposal_id     BIGSERIAL PRIMARY KEY,
    batch_deal_id   VARCHAR NOT NULL,   -- the upload batch it came from
    session_id      TEXT,
    proposed_name   VARCHAR,
    confidence      NUMERIC(5,2),       -- min pairwise F across members
    status          VARCHAR NOT NULL DEFAULT 'proposed',
                    -- proposed | confirmed | rejected | superseded
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    confirmed_at    TIMESTAMPTZ,
    confirmed_by    VARCHAR
);

CREATE TABLE proc.bp_deal_proposal_member (
    proposal_id     BIGINT NOT NULL REFERENCES proc.bp_deal_proposal(proposal_id),
    doc_type        VARCHAR NOT NULL,   -- quote | po | invoice
    doc_pk          VARCHAR NOT NULL,
    base_reference  VARCHAR,            -- version-collapsed quote ref
    role            VARCHAR,            -- anchor_quote | competing_quote | po | invoice
    match_score     NUMERIC(5,2),
    match_evidence  JSONB,              -- score_link per-signal breakdown
    PRIMARY KEY (proposal_id, doc_type, doc_pk)
);
```

Table names follow the `bp_` prefix convention; indexes as `ix_bp_deal_proposal_*`.

`match_evidence` stores `score_link`'s per-signal output so the UI can show *why* two
documents were grouped, rather than an unexplained number. There are **no product codes
anywhere in the schema** — no SKU, part number or item code in any extraction schema — so
matching is fuzzy text over `item_description`. That is exactly why a human confirmation
step is required and why evidence must be visible.

### Confirmation assigns identity

`deal_id` is minted **at confirm time**, never at proposal time. This is deliberate:
`mint_document_id` embeds the deal (`f"{deal_id}::{doc_type}::{doc_pk}"`), so a document's
identity depends on its grouping. Deciding the group before minting avoids a primary-key
migration entirely for new uploads.

On confirm: mint `deal_id`, write `deal_id`/`deal_name`/`document_id` to `_stg`/`_trgt`
and line-item tables via the existing `_persist_deal`, upsert `bp_deal_document_map`, and
insert the `proc.bp_deal` row with `is_tracked = false` (drafts stay drafts; promotion
remains the separate existing gate).

### Upload path change

`_look_forward` must stop treating a batch name as an authoritative deal. The batch
identifier is retained on `process_monitor` as a **batch label** for traceability, but is
no longer stamped onto documents as their deal. Documents arriving from an upload batch
enter clustering unlinked, which is the condition `_look_back` and the new pass require.

Existing non-batch behaviour of `_look_forward` (amending an existing, already-confirmed
deal) is preserved: when the user explicitly targets a known deal, that assignment stays
authoritative and clustering does not run.

## Components

| Unit | Responsibility | Depends on |
|---|---|---|
| `deal_clustering.py` | Batch → proposed clusters. Pure; no writes. | `linking_engine.score_link` |
| `version_collapse.py` | `quote_id` → base reference + round | none |
| `proposal_store.py` | Persist/read/confirm proposals | the two tables above |
| `POST /deals/proposals/generate` | Cluster a batch, store proposals | clustering + store |
| `GET /deals/proposals?batch=` | Proposals + members + evidence | store |
| `POST /deals/proposals/{id}/confirm` | Mint deal, assign documents | store + `_persist_deal` |
| `PATCH /deals/proposals/{id}/members` | Move/remove a document pre-confirm | store |
| Review screen (SpendIQ) | Show, adjust, confirm groupings | the endpoints |

`deal_clustering.py` is kept pure and write-free so it can be tested on fixtures without
a database, and so a bad clustering run can never corrupt assigned deals.

## Data flow

```
upload batch (unlinked docs)
   -> version collapse        (28 quotes -> 12 base references)
   -> invoice→PO via po_id    (deterministic)
   -> line-item clustering    (score_link, band >= 65)
   -> bp_deal_proposal[_member]        <-- nothing else written
   -> [ USER REVIEWS AND CONFIRMS ]
   -> mint deal_id -> _stg/_trgt/lines + bp_deal_document_map + bp_deal(is_tracked=false)
```

## Error handling

- **Fetch/scoring failure** → proposal generation fails loudly; no partial proposals are
  written. A batch is clustered atomically or not at all.
- **Missing line items on either side** — `cmp_line_composite` returns `(0.5, "MISSING")`,
  contributing nothing to the score. Such pairs must **not** be grouped on a
  near-zero signal; they fall to review. There is currently no telemetry on line-item
  extraction rate; the clustering run will record members-with-lines so this stops being
  invisible.
- **Null supplier_id** (3/28 quotes here) — must not block grouping; product similarity
  carries it, and the proposal is flagged so the user sees the supplier is unidentified.
- **No confident cluster** → documents remain unassigned and are listed as ungrouped.
  Never invent a deal to absorb them.
- **Confirm on a stale proposal** (documents re-extracted since) → reject with a conflict
  and regenerate, rather than assigning against outdated members.

## Testing

- **Fixtures from this real batch** — the 4 sourcing events above are the golden case;
  clustering must reproduce them from the 40 documents.
- **Version collapse** — 28 quotes → 12 base references, `(V3 (BAFO))` recognised as the
  current round.
- **Swift/null-supplier** — PO-2024-0091 groups via product similarity despite
  `SDP-Q-44120` having no supplier_id.
- **Caldwell orphan** — PO-2024-0163 stays orphaned; asserting it is *not* absorbed.
- **Cross-category negative** — no freight quote ever joins the IT services group.
- **Idempotence** — regenerating proposals for an unchanged batch is stable.
- **Non-destructive** — proposal generation leaves every existing `deal_id` byte-identical.

## Migration

The existing `ANALYSISSET_19072620260719339` batch is backfilled through the same path:
generate proposals from its 40 documents, present them for confirmation. The current
single deal is marked superseded once its members are confirmed into real deals. No
existing row is rewritten before the user confirms.

## Out of scope

Product code / SKU extraction (no source field exists today); merging deals across
different upload batches; automatic award detection; changing the draft→tracked promotion
gate; fixing `bp_deal_overview.three_way_match` semantics (recorded as a follow-up —
correct grouping makes it meaningful again, but the presence-count definition remains
wrong on its own terms).
