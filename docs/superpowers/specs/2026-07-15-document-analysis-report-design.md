# Document Analysis Report — Integration Design (Phase 1)

**Date:** 2026-07-15
**Status:** Draft for review
**Scope:** Cross-repo — `beyond_procwise_ui` (UI), `beyond-procwaise-Api` (gateway), `BP_Backend` (FastAPI + `proc.*` schema)
**Source mockup:** `BP_Backend/UI Improvements/doc_analysis_report_mockup_v27.html` (rendered live at artifact `1a0b6030-…` during brainstorming)

---

## 1. Goal

After a user uploads documents from **Home → "Find an opportunity"**, land them on a **Document Analysis Report** that populates with **live data** from the analysis of *their* documents: an Overview, the Documents/cases, a Validation & Actions queue, Opportunities found, and a Compliance check — plus a six-tab Deal Detail overlay. The report is the front door to committing that analysis into a tracked pipeline deal.

This is confirmed as an **`engine.js`-native** build (see §3), not a new React component tree.

## 2. What ships in Phase 1 vs Phase 2

**Phase 1 (this spec):**
- Routing so a **new analysis** lands on the report; an **existing-deal amend** goes straight to that deal.
- The **draft ↔ tracked** deal model (net-new) + **Promote** / **Save to references**.
- All **5 report tabs on live data** + the **six-tab Deal Detail overlay** (reusing the existing `dealView()`).
- Real, audit-logged **Assign / Reject / Decision** actions.
- The 3 compliance measures with no detector shown **fail-closed** ("Not measured yet").
- A **"working…"** state driven by the existing session WebSocket.

**Phase 2 (tracked, not in this spec):**
- The two floating draggable agent panels (Ask-Agent + email-drafting agent with real send), including a keyboard-accessible interaction path.
- Real detectors for **Split-PO**, **Approval-bypass**, **Duplicate-invoice**.
- Export CSV/PDF (link to the existing partial Report-builder, do not rebuild).
- "Add documents" from within the report (reuse the existing `/analyse` upload).

## 3. Architecture decision (confirmed)

The live SpendIQ surface is an **imperative `engine.js`** that renders views via HTML strings against `modules/SpendIQ/styles.css`. It is **not** a React component tree. The mockup's design tokens (`--brand:#2f6df6`, Inter) **already are** SpendIQ's tokens — there is no Ocean/Teal/Violet collision.

Therefore the report is a **new render function `analysisReportView()` in `engine.js`**, reusing existing primitives rather than introducing a parallel framework:

| Mockup primitive | Reused SpendIQ asset |
|---|---|
| Horizontally-scrolling KPI row | `.hscroll` / `kpiScroll` |
| RAG status chip | `rag(label, cls)` + `.rag.g/.a/.r` |
| Collapsible grouped lists (`case-grp`) | `case-grp` / `toggleCase` |
| Queue + detail (`ac-grid`) | `ac-grid` (Action-Centre) |
| PO·Invoice·Quote·Contract comparison matrix | `.cmx` / `cmxGen` / `compareDocs` |
| Six-tab Deal Detail | `dealView()` — **reused as-is, not forked** |

**Consequence for the pasted prompt:** its "typed React components / no innerHTML / define TS interfaces" mandate does not apply here; it was written without knowledge that SpendIQ is imperative. Data contracts are still defined (§7) but as documented JSON shapes, not TS types.

## 4. Routing (two paths after upload)

Upload mode already exists in `AnalyseUpload/index.jsx` as `mode ∈ {'new','amend','bulk'}`.

- **Path A — new analysis** (`mode = 'new'` or `'bulk'`): the upload produces a **draft** deal (§5). Navigate to the **new report view**:
  `/spendiq?view=analysis-report&deal=<dealId>&session=<sessionId>`
  Opens immediately in the **"working…"** state (§8) and fills in as analysis completes.
- **Path B — existing deal amend** (`mode = 'amend'`, i.e. the `/analyse?deal=&name=` deep-link reached from an opportunity/deal): navigate **straight to the existing deal's Deal Detail** (the six-tab `dealView`), not the report. Effectively today's behaviour, minus the detour through `analytics`.

Only the post-upload navigation in `handleAnalyse()` changes; the presigned-S3 upload path is untouched.

## 5. Draft ↔ tracked deal model (net-new)

**Finding that motivates this:** today there is **no draft/tracked distinction** — every uploaded deal appears in Pipeline the instant it has documents (`getDeals()` filters only `deal_id IS NOT NULL`). The mockup's "Live analysis / Promote / Save to references" is a client-side toast with no backing. To make Path A real, we build the distinction.

### 5.1 Schema (BP_Backend, `deploy/sql`)
New deal-header table (because `proc.bp_deal_overview` is a VIEW and cannot hold state):

```sql
CREATE TABLE IF NOT EXISTS proc.bp_deal (
    deal_id              text PRIMARY KEY,
    is_tracked           boolean NOT NULL DEFAULT false,   -- false = draft/live analysis
    is_saved_reference   boolean NOT NULL DEFAULT false,   -- intentionally kept draft
    tracked_at           timestamptz,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_deal_is_tracked ON proc.bp_deal (is_tracked);
```
(Follows the `bp_` / `ix_bp_*` naming convention.)

**Backfill (one-time, in the same migration):** insert every existing `deal_id` from `bp_deal_overview` with `is_tracked = true`, so no currently-visible Pipeline deal disappears.

### 5.2 Draft creation on upload (gateway)
`deal_id` is minted in the gateway (`data-integration.service.ts generateDealId`, **not** a DB proc — the docstrings elsewhere are stale; see §12). At **`confirm-upload`**, upsert a draft header:
`INSERT INTO proc.bp_deal (deal_id, is_tracked) VALUES ($1, false) ON CONFLICT (deal_id) DO NOTHING;`
Amend uploads reuse an existing `deal_id`, which is already tracked → no change.

### 5.3 Promote (BP_Backend)
`POST /deals/{deal_id}/promote` →
`UPDATE proc.bp_deal SET is_tracked = true, is_saved_reference = false, tracked_at = now(), updated_at = now() WHERE deal_id = %s;`
and, **where a linked opportunity exists** (§6.4), advance its `bp_opportunity.stage` (`identified → negotiation`) via the existing `set_stage`. Flipping `is_tracked` is the primary, always-reliable effect; stage-advance is best-effort pending opportunity↔deal linkage.

### 5.4 Save to references (BP_Backend)
`POST /deals/{deal_id}/save-reference` → `UPDATE proc.bp_deal SET is_saved_reference = true, updated_at = now() WHERE deal_id = %s;` (stays a draft, reopenable). Closing the overlay with unsaved changes prompts **Promote / Discard / Cancel** (never silently discards).

### 5.5 Pipeline / Live-analyses split (gateway)
- `getDeals()` (Pipeline): add join/predicate so it lists **`is_tracked = true`** only.
- New **"Live analyses"** listing: `is_tracked = false` (optionally `is_saved_reference = true` first), surfaced in the Analyse/SpendIQ area so drafts are reachable and reopenable.

## 6. Live-data wiring per tab

All numbers/statuses are **deterministic backend reads traceable to source documents**. Only the executive summary and (Phase 2) email/negotiation prose are LLM-authored, and those are labelled **"✨ Agent draft · AI-generated · review before acting."** Missing/ambiguous data **fails closed** (surfaces as an exception), never a clean default.

### 6.1 Overview
- KPI row from `GET /spendiq/metrics` scoped to the deal + `/spendiq/nav-counts`.
- RAG chip computed from the deal's open discrepancies / value-at-risk (deterministic), not asserted.
- Highlights / Watch: two capped lists (max 10) ranked by significance = f(value-at-risk, severity, confidence).
- Executive summary from `GET /deals/{deal_id}/summary` (existing LLM narrative), labelled as agent output.

### 6.2 Documents
- `GET /spendiq/deals/:id` → `bp_deal_documents` grouped into **cases** by document mix; each case shows a doc-type/count mix, a one-line "what's uncovered/mismatched", and a link into Validation. Uses `case-grp`; expandable case detail reuses `ac-grid`.
- 3-way match + cycle metrics (`three_way_match`, `price_variance_pct`, `cycle_days_*`) come from the same endpoint.

### 6.3 Data Validation & Actions
- Queue from `GET /spendiq/discrepancies` (grouped by case, segmented All / Verify / Resolve, prioritised by confidence × policy-risk × commercial impact).
- Detail pane: evidence, confidence, "why flagged", key/values, and a **Decision** block whose options are **filtered to the finding's issue type** from a centrally-maintained list (three-way-match ≠ compliance ≠ sourcing subsets).
- Line-by-line comparison reuses `.cmx` over the case's documents.

### 6.4 Opportunities
- `GET /opportunities?deal=<id>` — but **`bp_opportunity.deal_id` is NULL on every row today**, so the deal scope must first be made real:
  - **Backfill + forward-fill** `bp_opportunity.deal_id` by resolving each finding's `quote_id` → `bp_deal_documents.deal_id`. (Additive; no source data mutated.)
- Derived-from-existing-data (no new detectors):
  - **levers** ← distinct `detector_type`(s) per deal, stacked chips **capped at 3 + "+N more"**.
  - **document coverage chips** ← which of Quote/PO/Invoice/Contract exist in `bp_deal_documents` (missing = dashed/muted).
  - **Ready / Needs-validation** ← Needs-validation iff the deal has open discrepancies/missing-required; else Ready. Needs-validation rows do **not** expose Pursue/Assign/Reject.
- Filter (deal/supplier search) + sort (savings, lever count, status-ready-first).

### 6.5 Compliance Check
- `GET /compliance/getComplianceData` — real measures: Off-PO/Non-PO spend, Invoice-but-no-PO, Non-contract (unlinked) spend, Invoice>PO variance, Backdated PO. Tiles clickable → scroll/highlight the measure row.
- **Fail-closed**: Split-PO detection, Approval-bypass rate, Duplicate invoices show **"Not measured yet"** with a Phase-2 note — never a green/clean value.
- Issues grouped by severity; an issue with a matching open action in Validation is **"Reviewing"** and links to it; otherwise **"Open"**.
- Measures list collapsed by default; each row carries name, value/flag, what it checks, how it's calculated (auditable).

## 7. Data contracts (documented JSON shapes)

Define concrete shapes (documented, not TS types) for the report composition, so UI and gateway agree: `AnalysisReport` (header, RAG, KPI set, sessionId, isTracked), `Case` (doc mix, match rate, discrepancy summary, finding links), `ValidationException` (type, severity, confidence, evidence, reasons, applicable decision types, state), `Opportunity` (deal ref, category, coverage, levers[], saving, status, assignee, rejectionReason), `ComplianceMeasure` (name, value, description, calculation, standard|custom, category, `measured: bool`), `ComplianceIssue` (ref, type, supplier, exposure, severity, status open|reviewing, linkedExceptionId). Deal Detail reuses the existing `dealView` payload extended with `{ isDraft, sessionId }`.

## 8. "Working…" completion state

On mounting the report, subscribe to the existing `WS /ws/session/{session_id}`; fall back to polling `proc.process_monitor` via `/session/*`. Each tab shows a skeleton/"reading your documents…" until the session reaches a terminal state (`target`/`discrepancy`/`failed`), then renders live data. Failures surface per-document, fail-closed.

## 9. Real actions (audit-logged, not UI-only)

- **Assign / Reject** — one shared confirm/cancel popover component (the mockup's duplicated `toggleAssign`/`confirmAssign` collapse into one). Reject requires justification, writes via `POST /spendiq/discrepancies/resolve`; both append to the `bp_agent_actions` audit log.
- **Decision** — `POST /decisions/finding/{id}/action` (`apply_value` / `flag` / `dismiss`).
- **Promote / Save to references** — §5.3 / §5.4.
- **Pursue** (Phase 2) — opens the email agent.

Colour is never the only status signal — severity, Ready/Needs-validation, Open/Reviewing each carry text/icon in addition to colour (keep the mockup's `-ink` AA text colours).

## 10. Component boundaries & build order

**BP_Backend**
1. Migration: `proc.bp_deal` + indexes + backfill (§5.1).
2. Endpoints: `POST /deals/{id}/promote`, `POST /deals/{id}/save-reference` (§5.3–5.4).
3. Opportunity↔deal linkage backfill + forward-fill (§6.4).

**Gateway (`beyond-procwaise-Api`)**
4. `confirm-upload`: create draft `bp_deal` row (§5.2).
5. `getDeals()` → filter `is_tracked = true`; add "Live analyses" listing (`is_tracked = false`) (§5.5).
6. Any deal-scoping needed on metrics/opportunities/discrepancies for the report.

**UI (`beyond_procwise_ui`, branch `spendiq-ui`)**
7. `AnalyseUpload` post-upload routing: Path A → `?view=analysis-report`; Path B → deal detail (§4).
8. `analysisReportView()` in `engine.js` + the 5 tabs, reusing primitives (§3, §6).
9. "working…" WS state (§8).
10. Deal Detail overlay wiring (reuse `dealView`) + draft banner + Promote/Save/close-prompt.
11. Assign/Reject shared popover + Decision block wiring (§9).
12. "Live analyses" surface for drafts.

Suggested order mirrors the mockup: routing+shell → Overview → Documents → Validation → Opportunities → Compliance → Deal Detail → live-data verification.

## 11. Verification (live, not just mocks)

Per the "demonstrate on the running local server against live `bp_sqldb`" rule:
- Pick a real `deal_id` in `bp_sqldb`; confirm each tab renders real, source-traceable values.
- Fresh upload → lands as **draft** on the report, **absent** from Pipeline; **Promote** → appears in Pipeline; **Save to references** → stays in Live analyses.
- Existing-deal amend → routes straight to that deal.
- Assign/Reject/Decision produce real `bp_agent_actions` entries.

## 12. Open items / risks / notes

- **Memory correction to verify:** the note "deal_id is owned by a DB stored procedure — never assign it in app code (confirmed 2026-07-13)" is contradicted by code — `deal_id` is minted in the gateway (`generateDealId`) and no such trigger exists in `deploy/sql`. This build never writes `deal_id`, so it's unaffected, but the note should be re-verified/corrected.
- **Pipeline behaviour change:** filtering to `is_tracked = true` changes when deals appear. Backfill mitigates disappearance; still worth a conscious confirm at implementation.
- **Opportunity↔deal linkage** via `quote_id` assumes findings carry a resolvable `quote_id`; rows without one remain deal-unscoped (shown globally or omitted, fail-closed — decide during implementation).
- **Bulk mode** currently mints a single `deal_id` (no real fan-out); this spec does not change that.
