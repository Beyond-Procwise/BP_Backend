# Procurement Document Linking Engine + Confidence-Gated `_stg→_trgt` Promotion

**Date:** 2026-06-06
**Status:** Approved; single-engine build, live-tested
**Source model:** `Procurement Relationship Math Models.pdf` (deterministic linking model)

## Domain concept

A **deal** is a sourcing event. It can contain **multiple competing quotes** (suppliers
bidding), **one or more purchase orders** (awarded), and **invoices** (billing). The same
`deal_id` ties a deal's documents together. **`deal_id` assignment is done by a separate
SQL trigger — NOT by this engine.** The engine's responsibility is *linkage accuracy*:
deterministically verify the real document relationships so only correctly-linked,
high-confidence rows are promoted to the final `_trgt` tables. Accurate links are what
make the downstream `deal_id` grouping sound.

Relationships in the current data (PO is the anchor):
- **invoice → PO** via `po_id` (N:1, many invoices per PO)
- **quote → PO** via `po_id` (the awarded quote's PO)

## Decisions (user, 2026-06-06)

- Promote at **F ≥ 80** (auto_link, or auto_link_with_warning for 80–92, flagged).
- **Hold all unlinked docs** — POs (anchors) and docs with absent/unfound parent stay in
  `_stg`. Consequence: POs reach `_trgt` only via SQL ingestion, not this engine.
- Extraction **`confidence_score` ≥ 90** required, in addition to the link gate.
- **Live testing** (no unit tests).
- Build as a **single engine**.

## 1. Single engine — `src/services/linking_engine.py`

One module exposing both the deterministic scorer and the gated promoter.

### 1a. Deterministic linker (faithful to the PDF stages)
`score_link(source_row, target_row, source_lines, target_lines, profile) -> LinkResult`
returning `{F, decision, P_raw, C, S, Q, F_cap, signals:[{id,cluster,tier,weight,s,q,r,c,status}], evidence}`.

Stages: 1A comparator→`s_i`; 1B `q_field`; 1C `r_i=min(q_src,q_tgt)·applicability`;
1D `c_i=w_i·r_i·(2s_i−1)`; 1E gates; 2 cluster dampening (n_active: 1→1.0, 2→0.85, 3+→0.70);
3 `L=logit(p0)+α·Σcluster_score`; 4 `P_raw=1/(1+e^−L)`; 5 coverage `rho=Σ(w·r)/Σw`,
`C=floor+(1−floor)·rho`; 6 separation `S` (=1; we score the single explicitly-referenced PO,
candidate_count=1); 7A caps `F_cap=MIN(profile_cap, tier_conflict_cap, gate_cap)`;
7B `F=MIN(F_cap, P_raw·C·S·Q)·100`; 7C decision band
(≥92 auto_link, ≥80 warning, ≥65 review, ≥45 weak, else block).

Profile constants: `p0=0.02, α=0.30, floor=0.55`.

**Signals (computable fields only; weights from PDF D.11/D.12):**

| id | cluster | tier | weight | comparator | source→target field |
|---|---|---|--:|---|---|
| po_ref | reference | T1 | 5 | EXACT_REF | po_id ↔ po_id |
| supplier_id | identity | T1 | 5 | EXACT_ID | supplier_id ↔ supplier_id |
| amount | commercial | T2 | 3 | NUMERIC_TOL (1%) | converted_amount_usd |
| currency | commercial | T3 | 2 | EXACT_REF | currency |
| line_set | line | T2 | 4 | LINE_COMPOSITE | *_line_items_stg |
| temporal | temporal | T2 | 3 | TEMPORAL | invoice/quote date vs PO order/expected_delivery |
| location | context | T3 | 2 | LOCATION | country+region vs ship_to_country+delivery_region |

(SIG `supplier_vat` is omitted — no VAT column in `_stg`.) Total weight = 24.

**Comparators:** EXACT_REF/EXACT_ID normalize (lowercase, strip non-alphanumerics) then
1.0 equal / 0.0 both-present-differ / 0.5 either-missing; NUMERIC_TOL: 1.0 within tol,
linear decay to 0 by 10% drift, 0.5 if either missing; LINE_COMPOSITE: greedy match lines by
item_id/description, per-line score (desc token overlap, qty exact, unit_price exact),
scaled by count coverage; TEMPORAL: MIN(after-parent, within-window, not-expired), 0.5 if
dates missing; LOCATION: 0.5·country_match + 0.5·region_match.

**`q_field` adaptation (documented deviation):** `_stg` has no per-field OCR/extraction
sub-scores, so `q_field = confidence_score/100` per document, `1.0` for normalized/system
fields (ISO currency). `Q = min(source_conf, target_conf)/100`. With `_stg` confidence
94–100 this stays close to the worked example's 0.94–1.0; the example's exact 88.57 is not
bit-reproducible.

**Conflict caps:** a Tier-1 signal that conflicts (both present, normalized-unequal) caps
`F_cap ≤ 0.45` — so a wrong `po_id`/supplier cannot auto-promote.

### 1b. Gated promoter
`promote_ready(conn=None, doc_types=('invoice','quote'), limit=None) -> summary`
- Sweep `_stg` invoice + quote rows whose PK is **not yet in `_trgt`** (PK not null).
- For each: resolve parent PO by `po_id` (prefer `_trgt`, else `_stg`); if absent/not found →
  **hold** (`no_parent_reference` / `parent_not_found`). If `confidence_score < 90` →
  **hold** (`low_extraction_confidence`). Else `score_link`; if `F ≥ 80` → **promote**, else
  **hold** (`low_link_score`).
- **Copy** = dynamic `_stg ∩ _trgt` column intersection **minus `deal_id`,`deal_name`,
  `document_id`**. Idempotent upsert by existence check: INSERT (non-deal cols) if PK absent
  — so the SQL trigger fires and sets `deal_id`; UPDATE (non-deal cols) if present —
  **preserving** any trigger-assigned `deal_id`/`deal_name`/`document_id`. Line items
  delete-then-insert by parent PK (intersection minus deal cols).
- **No `ON CONFLICT`** (no unique constraint guaranteed on `_trgt` PKs) — existence check
  instead.
- **Log** one `agent_actions` row per doc: `phase='consolidation'`,
  `action_type='promote_to_trgt'` (status `ok`, or `warn` for 80≤F<92) or `'promote_held'`
  (status `skipped`/`warn`), `details` = F, decision, gate reason, signal breakdown.
- Thresholds env-overridable: `PROMOTE_MIN_CONFIDENCE=90`, `PROMOTE_MIN_LINK_SCORE=80`.

## 2. Endpoint — `src/api/routers/promotion.py`

`POST /promotion/run` (optional `?doc_type=&limit=`) → runs the sweep, returns
`{promoted, held, by_reason, details:[{doc_type, doc_pk, F, decision, action, reason}]}`.
Registered in `src/api/main.py`.

## 3. Behavior / safety

- Never writes `deal_id`/`deal_name`/`document_id` (SQL trigger owns them); preserves them on
  update.
- Read-only on `_stg` and on parent PO rows; writes only `_trgt` + `agent_actions`. Idempotent.
- A Tier-1 conflict (wrong PO/supplier) hard-caps F ≤ 45 → cannot promote.
- Live-tested against real `_stg` data: confirm linked invoices/quotes promote, POs and
  unlinked/low-confidence docs are held, `_trgt` receives the rows (deal cols left for the
  trigger), and `agent_actions` records each decision with its F-score.
