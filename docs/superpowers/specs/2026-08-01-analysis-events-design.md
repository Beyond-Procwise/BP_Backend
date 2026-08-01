# Analysis Events

**Date:** 2026-08-01
**Status:** Design approved, ready for planning
**Repos touched:** BP_Backend (primary), spendiq-ui, Node gateway (no change — see §7)

**Relationship to `2026-07-15-document-analysis-report-design.md`:** that spec built the report
screen and is otherwise intact. This one supersedes two of its decisions — its §2 routing rule that
"an existing-deal amend goes straight to that deal" (§4 below), and its treatment of the report as a
deep-link destination keyed on `deal` (§8 below). It also closes its Phase-2 item "Add documents
from within the report", via the "Run a new analysis" control on the deal.

---

## 1. The problem

Today, "an analysis" is not a thing that exists in the system. What exists is a **draft deal**.

- `/analyse` — "Find an opportunity in your documents" — uploads files, then redirects to
  `?view=analysis-report&deal=<id>&session=<sid>` (`spendiq-ui/src/modules/AnalyseUpload/nextRoute.js`).
- That report view is **deep-link-only**. It is registered in the view table purely so the generic
  header lookup has something to read, and is never reachable from the sidebar
  (`spendiq-ui/src/modules/SpendIQ/engine.js:977-980`).
- The Analyse area's "Live analyses (drafts)" list is really a query for deals with
  `is_tracked = false` (`beyond-procwaise-Api/.../spendiq.service.ts:1145-1170`).

Every symptom follows from that one choice:

| Symptom | Cause |
|---|---|
| Document Analysis feels like a separate place, not part of Analyse | The report is a deep-link destination outside the nav |
| An analysis vanishes once the deal becomes real | Promotion sets `is_tracked = true`, so it drops out of the drafts query |
| A one-off analysis over several documents has nowhere to live | No deal id → `nextRoute.js` lands on an empty report |
| An upload producing two deals can only show one | The URL carries a single `deal=` |
| A deal has no analysis history | Nothing records "this analysis, on this date, produced this deal" |
| **Amending a deal produces no record at all** | `nextRoute.js` sends amend uploads straight to the deal detail, skipping the report entirely |

That last row is the sharpest. The amend flow already exists and works — pick an existing deal,
confirm its reference, upload more documents (`AnalyseUpload/index.jsx:433`, `:513`). It is exactly
the "Find an opportunity → links to a deal" path. And it leaves no trace.

There **is** already a per-upload record in the database — `process_monitor.session_id` plus
`proc.session_document_outcome`, resolved by triggers that fire `pg_notify('session_status')`
(`deploy/sql/2026-06-30_session_action_status.sql`). It is a processing log, invisible to users.
It is the right spine to build on.

---

## 2. What an analysis event is

An analysis event is **a name, a date, the documents it read, what it found, and which deals it
produced.** It has its own identity and does not depend on a deal existing.

One row per *analysis run*. Nothing is ever overwritten.

---

## 3. Lifecycle and the freeze point

| State | Meaning | Behaviour on screen |
|---|---|---|
| `running` | Documents still extracting / linking | **Live.** The working surface immediately after upload — unchanged from today |
| `complete` | Upload session resolved; findings captured | **Frozen.** Shows what was found that day. Action controls link through to the live deal |
| `failed` | No document reached a usable state | Frozen, with the reason |

**The freeze happens exactly once**, when the upload session resolves. The hook already exists:
`SessionNotifyListener._link_then_broadcast()`
(`src/services/session_notify_listener.py:186`) currently does two things in order — link the
documents to their deals, then broadcast the terminal WebSocket frame that tells the report page to
stop waiting. The snapshot is captured **between** those two steps, because by then the deals exist
but the client has not yet been told to render.

Reopening a `complete` analysis shows the frozen record. Anything that would change state
(resolve a discrepancy, approve an unconnected document, generate a deal proposal, refetch a
benchmark) is not offered on a frozen analysis — those controls become links to the deal, where the
live state lives. This is deliberate: a frozen record that can be edited is neither a record nor
frozen.

### Failure handling

Two ways an analysis can get stuck in `running`:

1. **The listener dies mid-session.** The existing 15-minute scheduled sweep gains a pass: any
   `bp_analysis` in `running` whose session has resolved (`process_monitor.action_status IS NOT NULL`)
   gets frozen by the same code path the listener uses.
2. **The session never resolves.** Any `bp_analysis` in `running` for more than 60 minutes is marked
   `failed` with `failure_reason = 'session did not resolve'`. Sixty minutes is deliberately far
   beyond the UI's 6-minute patience cap (`engine.js` `REPORT_WAIT_CAP_MS`) — a slow large upload
   must never be declared failed while it is still working.

Freezing is idempotent: it is a no-op against any analysis not in `running`.

---

## 4. Versioning

Every run is a new event with its own timestamp. On a deal, those events form a dated history:

```
Deal DEALV3-77 · Northwind Cloud
  Analysis history
    v3 · 1 Aug 2026  · 4 documents · £12,400 found      ← latest
    v2 · 14 Jul 2026 · 2 documents ·  £9,100 found
    v1 · 2 Jul 2026  · 6 documents ·  £9,100 found      ← created this deal
```

**The version number lives on the link between analysis and deal, not on the analysis itself.**
One run can touch three deals at once, and each of those deals is at a different point in its own
history — the same run may be v3 for one deal and v1 for another. `version` is therefore a column on
`bp_analysis_deal`, allocated per `deal_id`, unique on `(deal_id, version)`.

Between consecutive versions on a deal the UI shows a **headline delta** — documents added, change
in value found, opportunities opened or closed. Both sides are frozen snapshots of the same shape,
so this is a comparison of stored numbers, never a re-computation.

A field-by-field diff of the findings is **out of scope**.

### Entry points for a new analysis on an existing deal

- **From the deal** — a "Run a new analysis" control on the deal detail, which opens `/analyse`
  pre-set to amend mode with that deal already selected and confirmed.
- **From Find an opportunity** — the amend flow that already exists.

Both land on the **new analysis event**, not on the bare deal. This changes
`nextRouteAfterUpload()`: every mode now routes to the analysis. The analysis header names the deal
it belongs to and links to it.

---

## 5. Data model

Three new tables in `proc`, `bp_` prefixed, indexes `ix_bp_*`.

### `proc.bp_analysis` — the event

```sql
CREATE TABLE IF NOT EXISTS proc.bp_analysis (
    analysis_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT,
    mode            TEXT NOT NULL CHECK (mode IN ('new','amend','bulk')),
    session_id      TEXT UNIQUE,          -- links to process_monitor.session_id
    status          TEXT NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running','complete','failed')),
    failure_reason  TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    created_by      TEXT,
    document_count  INTEGER,
    value_found     NUMERIC(18,2),
    currency        VARCHAR(8),
    findings        JSONB                 -- the frozen snapshot; NULL while running
);

CREATE INDEX IF NOT EXISTS ix_bp_analysis_started
    ON proc.bp_analysis (started_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_analysis_status
    ON proc.bp_analysis (status) WHERE status = 'running';
```

`session_id` is `UNIQUE` so that creating an event is idempotent — the same upload can never produce
two events.

### `proc.bp_analysis_document` — what it read

```sql
CREATE TABLE IF NOT EXISTS proc.bp_analysis_document (
    analysis_id   UUID NOT NULL REFERENCES proc.bp_analysis(analysis_id) ON DELETE CASCADE,
    doc_type      TEXT,
    doc_pk        TEXT,
    file_path     TEXT NOT NULL,
    file_name     TEXT,
    outcome       TEXT CHECK (outcome IN ('target','discrepancy','failed')),
    PRIMARY KEY (analysis_id, file_path)
);
```

`outcome` mirrors `session_document_outcome` and is copied from it at freeze time. This table is
what lets a **one-off analysis stand on its own**: it has documents even when no deal was formed.

### `proc.bp_analysis_deal` — the link

```sql
CREATE TABLE IF NOT EXISTS proc.bp_analysis_deal (
    analysis_id  UUID NOT NULL REFERENCES proc.bp_analysis(analysis_id) ON DELETE CASCADE,
    deal_id      VARCHAR(25) NOT NULL,
    version      INTEGER NOT NULL,
    is_latest    BOOLEAN NOT NULL DEFAULT true,
    linked_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (analysis_id, deal_id),
    UNIQUE (deal_id, version)
);

CREATE INDEX IF NOT EXISTS ix_bp_analysis_deal_deal
    ON proc.bp_analysis_deal (deal_id, version DESC);
```

Many-to-many in both directions: one analysis → several deals (a multi-group upload), one deal →
several analyses over time (the version history).

### Why `findings` is one JSONB column

The report assembles itself from six separate live sources today
(`spendiq-ui/src/modules/SpendIQ/data/endpoints.js:35` plus the Benchmark tab): deal detail,
discrepancies, compliance, opportunities, the deal summary, and `/benchmark/by-deal/:id`. Freezing
means capturing what those returned. Normalising all six into columns would be a large schema
surface that is only ever read back as a whole. One snapshot column — written once, read once — is
honest about what it is.

The seventh thing the report fetches, the unconnected-documents review queue
(`/promotion/review-queue`, `engine.js:5665`), is deliberately **not** snapshotted. It is a queue of
work to do, not a finding — live state that belongs to the deal. A frozen analysis does not show it.

The headline figures (`document_count`, `value_found`, `currency`) are lifted out as real columns
because the list view sorts on them and the version delta compares them.

`value_found` uses the **existing** Value Found definition — `bp_opportunity.financial_impact_gbp`
plus `bp_extraction_discrepancy.recovered_amount` (`scripts/migrations/2026-07-30-value-found-columns.sql`,
`spendiq-ui/src/lib/valueFound.js`) — computed at freeze time. This spec introduces no new
definition of value.

### Not touched

`proc.bp_analysis_summary` (`deploy/sql/2026-06-19_create_bp_analysis_summary.sql`) is, despite the
name, per-deal *current* analytics with an `is_current` pattern. Different thing. It keeps working
unchanged.

---

## 6. Write path

### Creating the event (`running`)

The UI calls `POST /analysis` on BP_Backend directly, immediately after the upload loop returns a
`sessionId` and before it navigates (`AnalyseUpload/index.jsx:355-362`). It already holds everything
needed: `mode`, `name`, `batchSessionId`, `dealId`.

**This call is an optimisation, not a correctness dependency.** If the browser closes between upload
and the POST, the scheduled sweep creates the missing event from `process_monitor` rows for any
session with no `bp_analysis` row — it just won't have the user's chosen name, and falls back to the
deal name or the session id. Nothing is lost.

### Freezing (`complete`)

Inside `SessionNotifyListener._link_then_broadcast()`, after `_link_session()` and before
`_broadcast()`:

1. Copy `session_document_outcome` rows for this session into `bp_analysis_document`.
2. Resolve the deals via `SELECT DISTINCT deal_id FROM proc.process_monitor WHERE session_id = %s
   AND deal_id IS NOT NULL` — `process_monitor` already carries `deal_id` and `deal_name`
   (verified against live `bp_sqldb`; not in the base DDL at
   `scripts/migrations/2026-05-12-bp_sqldb-two-stage-init.sql:167`, added later by the gateway's
   schema). Insert `bp_analysis_deal` rows, allocating
   `version = COALESCE(MAX(version), 0) + 1` per `deal_id` and clearing `is_latest` on that deal's
   previous rows.
3. Fetch the six finding sources listed in §5 for those deals and store the result in `findings`.
4. Compute `document_count`, `value_found`, `currency`.
5. Set `status = 'complete'`, `completed_at = now()`.

All five steps in one transaction. Failure is logged and the analysis stays `running` for the sweep
to retry — it must never block the broadcast, matching the existing fail-open contract documented at
`session_notify_listener.py:187-192`.

New module: `src/services/analysis_store.py`, holding `start()`, `freeze()`, `sweep()`. The listener
and the scheduler both call into it; neither owns the logic.

---

## 7. Read API

All reads go **directly to BP_Backend**. The UI already talks to it via `__SPENDIQ_API_AI__` /
`VITE_AI_API_URL` for the Analyse, Negotiate and Opportunities surfaces, so no proxying is needed.

New router `src/api/routers/analysis.py`, prefix `/analysis`:

| Endpoint | Purpose |
|---|---|
| `POST /analysis` | Start an event. Idempotent on `session_id`. Returns `{analysis_id}` |
| `GET /analysis` | List events, newest first. `?status=&deal_id=&limit=&offset=` |
| `GET /analysis/{analysis_id}` | Full record — header, documents, linked deals, frozen findings |
| `GET /analysis/by-deal/{deal_id}` | Version history for a deal, newest first, with headline deltas |
| `GET /analysis/by-session/{session_id}` | What the report polls while `running` |

### Gateway

**No gateway changes.** `GET /spendiq/live-analyses` (`spendiq.controller.ts:81`,
`spendiq.service.ts:1145`) stays exactly as it is and keeps serving its current shape — the UI simply
stops calling it. Removing it is a separate cleanup, not part of this work.

---

## 8. UI changes

All in `spendiq-ui`.

**Analyse area (`view=analytics`)**

- `loadLiveAnalyses()` (`engine.js:1561`) repoints from `/spendiq/live-analyses` to
  `GET /analysis`. Its three-state contract — `null` = not loaded, `Error` = failed (must never
  collapse to "no drafts"), `[]` = genuinely empty — is preserved exactly.
- `liveAnalysesPanel()` becomes **"Analyses"**, listing every event regardless of whether its deal
  was promoted. Each row shows name, date, document count, value found, and the deals it produced.
- `analysisSubNav()` (`engine.js:7361`) lists the same events.

**Opening an analysis**

- Route becomes `?view=analytics&analysis=<id>`. The existing report screen and its six tabs are
  reused as-is; only its data source changes — frozen `findings` when `complete`, live fetches when
  `running`.
- `openSnapshot(id)` (`engine.js:3856`) takes an `analysis_id` instead of a deal id.
- The `analysis-report` view entry (`engine.js:977-980`) and its `?view=analysis-report&deal=`
  deep link are kept as a redirect to the new route for one release, so existing links and the
  Actions-list handler at `engine.js:1601` do not break.

**Deal detail**

- New **"Analysis history"** section, fed by `GET /analysis/by-deal/{deal_id}`: versions newest
  first, headline delta between consecutive versions, each opening that analysis scoped to this deal.
- When an analysis spans several deals, the scoped view carries a "this analysis also covered N other
  deals — show all" control.
- New **"Run a new analysis"** control routing to `/analyse` in amend mode with the deal preselected.

**Pipeline** inherits this with no change — Pipeline rows are deals, and the history travels with the
deal.

**Upload**

- `AnalyseUpload/index.jsx` gains the `POST /analysis` call described in §6.
- `nextRouteAfterUpload()` returns `/spendiq?view=analytics&analysis=<id>` for **all** modes,
  including amend.

---

## 9. Existing data

### What is actually there (verified against live `bp_sqldb`, 2026-08-01)

| Fact | Count |
|---|---|
| Deals in `proc.bp_deal` | 5,043 (5,037 tracked, 6 draft) |
| `process_monitor` rows with a `session_id` | 59 |
| Distinct upload sessions | **3** |
| `session_document_outcome` rows | 55 |
| Deals that ever came through an upload session | **3** |
| Sessions producing more than one deal | 0 |
| Deals with more than one session | 0 |

**This is the single most important thing to know before building.** Only three of 5,043 deals
were created by an upload; the rest were bulk-ingested and have no session at all. So:

- The backfill produces exactly **three** analysis events.
- **5,040 deals will show an empty Analysis history.** That is the normal, correct state for them —
  not an error, and not a bug to chase. The deal detail must render it as *"No analysis has been run
  on this deal"* with the "Run a new analysis" control, and must never render it as a zero, a
  spinner, or a failure.
- The many-to-many between analyses and deals is **not exercised by any existing data**. It is a
  forward-looking capability, justified by the amend flow (a deal will accumulate versions from its
  second analysis onward) and by multi-group uploads. Its tests must therefore be written against
  constructed fixtures, not against the corpus.

### The backfill script

`scripts/backfill_analysis_events.py`:

- For every distinct `process_monitor.session_id` with at least one `session_document_outcome` row,
  create a `bp_analysis` in `complete` with `started_at = MIN(created_date)` for that session.
  **`created_date`, not `start_ts`** — the watcher's stale-row cleanup sets `start_ts = NULL` on
  rows it reaps (`src/services/process_monitor_watcher.py:1046`), so it is not a durable timestamp.
- `name` comes from `process_monitor.deal_name`, falling back to the session id.
- `mode = 'new'` for all backfilled rows. `process_monitor.is_new_deal` exists but is NULL on all 59
  rows, so it cannot be used to distinguish new from amend.
- Populate `bp_analysis_document` from `session_document_outcome`.
- Populate `bp_analysis_deal` from `process_monitor.deal_id`, allocating `version` in `created_date`
  order per deal.
- Leave `findings` **NULL**. The historical findings were never captured and cannot be honestly
  reconstructed — reading live data now and presenting it as "what we found then" would be
  fabrication. The UI renders a NULL `findings` as *"Findings were not captured for this analysis —
  see the deal for current detail"*, with a link.

Idempotent, re-runnable, and driven by the `session_id UNIQUE` constraint.

---

## 10. Out of scope

- Re-running an analysis over documents already in the system with no new upload. No entry point
  asks for it.
- Field-by-field diff between versions. Headline delta only.
- Deleting or archiving analysis events.
- Any permission model beyond the existing RBAC nav gating (`__SPENDIQ_NAV_DENY__`).
- Removing `GET /spendiq/live-analyses` from the gateway.

---

## 11. Testing

**BP_Backend (pytest, `./venv/bin/python -m pytest` with `.env` loaded)**

- `analysis_store.start()` is idempotent on `session_id` — two calls, one row.
- `freeze()` is idempotent and a no-op on a non-`running` analysis.
- `freeze()` allocates `version` per deal, not globally: one analysis touching two deals that are at
  different history depths gets different version numbers on each link.
- `freeze()` clears `is_latest` on the deal's prior rows and sets it on the new one.
- A failure inside `freeze()` leaves `status = 'running'` and does not raise into the listener.
- `sweep()` freezes a resolved-but-stuck analysis; marks a >60-minute unresolved one `failed`;
  creates a missing event for a session that has no `bp_analysis` row.
- Endpoint contract tests for all five routes, including `by-deal` ordering and delta arithmetic.

**spendiq-ui (contract tests, matching the existing `*.contract.test.js` convention)**

- `nextRouteAfterUpload()` returns an analysis route for all three modes — this is the regression
  guard for the amend path currently skipping the report.
- `loadLiveAnalyses()` keeps its three-state contract against the new endpoint: an error must not
  render as "no analyses".
- The version-delta helper is pure and tested directly against two snapshot fixtures.
- A `complete` analysis renders no state-changing controls; a `running` one does.
- A NULL `findings` renders the backfill message, never a zero.
- A deal with **no** analysis history renders "No analysis has been run on this deal" plus the
  "Run a new analysis" control — never a zero, spinner or error. This is the state 5,040 of 5,043
  deals are in (§9), so it is the common path, not the edge case.

**Live verification (per the standing requirement to prove changes on the running local server
against live `bp_sqldb`, not only in tests)**

1. Upload a new multi-document deal via `/analyse` → an analysis event appears in Analyse, goes
   `running` → `complete`, and the deal shows it as v1.
2. Amend that deal with one more document → v2 appears on the deal with a headline delta, and the
   upload lands on the new analysis.
3. Promote the deal to Pipeline → **both** analyses remain visible in the Analyse list. This is the
   direct regression test for the "it disappears" symptom.
4. Upload documents that form no single deal → a one-off analysis with documents and no deal link.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Freeze adds latency between linking and the terminal WebSocket frame, delaying the report | Freeze reads data that was just written and is already warm; if it exceeds a budget, log and broadcast anyway — the fail-open contract already in `_link_then_broadcast` |
| Deal mis-grouping (known open issue) puts the wrong deals on an analysis | The analysis links whatever the assignment service decided. It reflects grouping, it does not fix it. No new failure mode |
| Multi-group uploads sharing one session are known to break one-shot completion checks | The freeze is driven by session *resolution*, which the triggers compute across all documents in the session, not by a per-group check |
| `findings` snapshots grow the table | One row per upload, bounded by upload volume. Revisit if `pg_total_relation_size` warrants it |
| Backfilled analyses with NULL findings look broken | Explicit UI message and a link to the deal. Honest absence beats a fabricated number |
| **Almost every deal shows an empty Analysis history**, making the feature look unfinished | Real and unavoidable — 5,040 of 5,043 deals never came through an upload (§9). Handled by an explicit empty state plus the "Run a new analysis" control. Worth setting expectations before the first demo: the value shows up on deals created or amended *after* this ships |
| The many-to-many is untested by real data | No corpus row exercises it (§9). Covered by constructed fixtures, and by the live-verification steps in §11 which deliberately create a second version on a deal |
