# Value Ledger: recording the money recovered

**Date:** 2026-09-25
**Status:** approved in conversation 2026-09-25; awaiting written-spec review
**Origin:** Step 1 of the build-status plan (`docs/architecture/BUILD_STATUS_1_TO_5.md`, item 1; published report https://claude.ai/artifact/AtzSv6NLwXqBBnVpeP58pU)
**Repos:** BP_Backend (table, service, routes, readers, tests) and beyond_procwise_ui (Action Centre panel, Value Found drawer, opportunity action). The gateway does not change.

## 1. Why

SpendIQ promises a buyer that it finds money leaking out of procurement and helps get it back. Today the product cannot show the second half:

- **Recovered value is always £0.** Two overwrite-in-place columns exist: `bp_extraction_discrepancy.resolution_outcome`/`recovered_amount` (`scripts/migrations/2026-07-30-value-found-columns.sql`) and `bp_opportunity.realised_savings_gbp`. The gateway resolve can write them. The UI only ever sends `{id, action}` (UI `engine.js:3310,3575`; `useHomeData.js:155`), so 0 of 5,372 findings and 0 of 308 opportunities in bp_testdb carry a value.
- **The "money found" total undercounts.** `value_summary_service.DISCREPANCY_VALUE_TYPES` counts only `amount_over_po`, `line_amount_over_po` and `duplicate_invoice`. The triage engine (shipped 2026-09-24) files its money findings as `quantity_invoiced_above_po` (3,254 open), `invoices_exceed_po_total` (447) and `unit_price_differs_from_po` (117). None of these are counted.

## 2. Understanding (agreed 2026-09-25)

**User rulings:**
- Recovery is **two steps**. Resolving a finding records the money as *claimed*. A later "credit received" makes it *recovered*. Only confirmed money counts in the headline.
- Approved in conversation: the design in §3–§6.

**Assumptions stated and accepted:**
- An overcharge or duplicate stopped **before payment** is *avoided*, confirmed in one step.
- Amounts are kept in the document's currency. GBP is converted **when the outcome is recorded**, and the rate and date are stored with it. If no rate exists, the GBP amount stays NULL and that row counts in no GBP total. No figure is invented.
- **Customer separation is out of scope.** `tenant_id` is the constant `'default'`, as on `bp_commercial_fact` (the B2 decision). No RLS in this work.

**Success, proven on the local server against bp_testdb:**
1. Resolve a real overcharge as "Claiming it back, £X".
2. Record "Credit received £X, credit note CN-…".
3. The home Value Found tile, the opportunities dashboard's monthly KPI and the executive-summary report all show the same £X recovered.
4. The table holds two rows (claimed, then recovered) and refuses an attempt to edit either.

## 3. Data: `proc.bp_value_outcome`

A new migration pair: `deploy/sql/2026-09-26_bp_value_outcome.sql` plus `_rollback.sql`.

| column | type | rule |
|---|---|---|
| `outcome_id` | `bigserial` PK | |
| `tenant_id` | `text NOT NULL DEFAULT 'default'` | constant until the tenancy project |
| `source_type` | `text NOT NULL` | CHECK IN (`'finding'`, `'opportunity'`) |
| `source_id` | `text NOT NULL` | `bp_extraction_discrepancy.discrepancy_id` (bigint, stored as text) or `bp_opportunity.opportunity_id` (varchar); both types verified live 2026-09-25 |
| `outcome_type` | `text NOT NULL` | CHECK IN (`'avoided'`, `'claimed'`, `'recovered'`, `'claim_dropped'`, `'realised_saving'`, `'terms_improved'`, `'cycle_time'`) |
| `amount` | `numeric(18,2)` | NOT NULL for every type except `claim_dropped`; `> 0` |
| `currency` | `char(3)` | NOT NULL whenever `amount` is money (every type except `cycle_time`, where `amount` is days) |
| `amount_gbp` | `numeric(18,2)` | NULL when unconvertible |
| `fx_rate`, `fx_as_of` | `numeric`, `timestamptz` | the rate used; NULL when no conversion happened |
| `evidence_ref` | `text` | **required** for `recovered` (credit-note number or document reference) |
| `note` | `text` | optional free text |
| `supersedes_id` | `bigint REFERENCES proc.bp_value_outcome` | set on a correcting row |
| `recorded_by` | `text NOT NULL` | from the authenticated principal |
| `valid_from` | `date NOT NULL` | the day the money moved; user-entered, defaults to today |
| `recorded_at` | `timestamptz NOT NULL DEFAULT now()` | transaction time |

**Rules, enforced in the database:**
- **Append-only.** A BEFORE UPDATE OR DELETE row trigger plus a statement-level TRUNCATE trigger raise, copying the pattern and reasoning of `deploy/sql/2026-09-16_bp_agent_actions_immutable.sql`. A REVOKE would not bind the owning role.
- **Corrections.** A correction is a new row with `supersedes_id`. Readers ignore any row that another row supersedes.
- **Indexes:** `ix_bp_value_outcome_source (source_type, source_id)`, `ix_bp_value_outcome_type_valid (outcome_type, valid_from)`.
- **Applied to both databases:** bp_testdb and bp_sqldb, following the hand-applied pattern of `deploy/sql` (there is no migration-tracking table).

**Claim state is derived, not stored.** For a finding, the current state is its latest non-superseded row: `claimed` means being claimed; `recovered` or `claim_dropped` means settled.
- A `recovered` amount may differ from the claim. The claimed figure is kept; the recovered figure is what counts.
- A partial credit is recorded as `recovered` with the smaller amount. No separate partial state (YAGNI).

## 4. Writing outcomes (backend)

A new service, `src/services/value_ledger.py`, holds pure validation and GBP conversion plus a thin SQL layer. The routes go in the existing `src/api/routers/value_summary.py`, and every one requires `principal=Depends(require_user)` (P8).

| route | effect |
|---|---|
| `POST /value/findings/{discrepancy_id}/outcome` body `{outcome: avoided\|claimed\|accepted, amount, currency, valid_from?, note?}` | Closes the finding (`status='resolved'`, `resolved_by`, `resolved_at`) **and** writes the outcome row, in one transaction. `accepted` closes the finding and writes no outcome row (no money moved). Refuses with 409 if the finding is not open: the lifecycle trigger (`2026-09-17_bp_lifecycle_transitions.sql`) raises `PG_LIFECYCLE_REFUSED`, and the route reports it the way the gateway does. |
| `POST /value/findings/{discrepancy_id}/settle` body `{outcome: recovered\|claim_dropped, amount?, currency?, evidence_ref?, valid_from?, note?}` | Refused unless the finding's current ledger state is `claimed`. Writes one row. |
| `POST /value/opportunities/{opportunity_id}/realise` body `{amount, currency, valid_from?, evidence_ref?, note?}` | Moves the stage to `realised` through the existing `opportunity_store.set_stage` (the lifecycle trigger applies) and writes a `realised_saving` row, in one transaction. |
| `POST /value/outcomes/{outcome_id}/correct` body `{amount, currency, evidence_ref?, note}` | Writes a correcting row with `supersedes_id`. The note is required. |
| `GET /value/findings/{discrepancy_id}/outcomes` | The row history for one finding, oldest first. |

**Transactions.** `get_conn()` is AUTOCOMMIT: `rollback` is a no-op and a `FOR UPDATE` lock ends with its statement. So each write route opens an explicit transaction (`BEGIN … COMMIT`) on its own connection and takes `SELECT … FOR UPDATE` on the finding or opportunity row. If the outcome insert fails, the status change rolls back with it.

**Audit.** Each write also records an action on `proc.bp_agent_actions` (`action_type='value.outcome_recorded'`) through `record_action_or_fail`. If the audit row cannot be written, the write is refused.

**Currency.** The finding's currency comes from its invoice (`bp_invoice_trgt.currency`, the join `value_summary_service` already uses). It is sent to the UI as the default and can be changed. GBP conversion uses `repositories.fx_rate_repo.get_or_refresh_rates()` with the formula documented in `value_summary_service.py` (`amount / rates[ccy] * rates["GBP"]`).

**The legacy columns** (`resolution_outcome`, `recovered_amount`, `realised_savings_gbp`) are no longer written by the new routes and no longer read (§5). The gateway resolve endpoint is left unchanged, so nothing breaks for its other callers. A follow-up can drop the columns once this has shipped.

## 5. Reading outcomes

**`value_summary_service.build_value_summary`:**
- Recovered, avoided and being-claimed figures come from `bp_value_outcome`, using only non-superseded rows and each finding's latest state.
- New response fields: `avoided_gbp`, `recovered_gbp` (redefined to mean ledger-recovered), `saved_gbp` (= avoided + recovered), `claimed_open_gbp` (claims not yet settled) and `by_month` (sums by `date_trunc('month', valid_from)`).
- Existing fields stay, so current UI readers keep working.
- **Triage findings counted.** `DISCREPANCY_VALUE_TYPES` gains `quantity_invoiced_above_po`, `invoices_exceed_po_total` and `unit_price_differs_from_po`.
  - Their £ amount is taken from `bp_triage_result.exposure_gbp` through `bp_triage_finding.mirror_id`, never parsed from `notes`. The mirror leaves `computed_value` NULL on purpose (`triage/writer.py`), and `raw_value`/`expected_value` are not a reliable currency basis.
  - The existing `dedupe()` keeps one figure per (deal, document).
  - **Before implementing:** confirm on bp_testdb that `invoices_exceed_po_total` and line-level findings on the same PO do not double-count within one document. If they do, count the document-level figure only.

**Opportunities dashboard** (`opportunity_dashboard.py:68,80-81,112`): the realised totals and month-on-month KPI read `realised_saving` rows by `valid_from`.

**Executive summary report** (`rga/builders/exec_procurement_summary.py:75-76,223-233`): the "Realised savings (GBP)" fact reads the ledger and becomes "Saved (GBP)", with the recovered/avoided split and provenance naming `proc.bp_value_outcome`. When no rows exist it keeps its existing honest "not being captured" reason.

**Weekly digest** (`value_digest.py`): reads the same summary, with no code change beyond the new fields. Fix the stale test `tests/services/test_value_digest.py::test_sends_to_every_configured_recipient` by setting `VALUE_DIGEST_SENT_AS` in the test. It has been red since `ee01b12`.

## 6. Screens (beyond_procwise_ui)

**Action Centre, resolving a money finding** (`engine.js` `resolveFinding` and the decision flow near `:3575`; `useHomeData.js` `decideFinding`):
- For findings in the money types, the closing action opens a small panel titled **"What happened to this money?"** with three options:
  - "Stopped before payment" (avoided)
  - "Claiming it back from the supplier" (claimed)
  - "Accept the charge"
- Amount is pre-filled with the finding's figure and can be edited. Currency is pre-filled. The date defaults to today.
- It posts to `POST /value/findings/{id}/outcome` via the AI-API bridge (`window.__SPENDIQ_API_AI_POST__`) **instead of** the gateway resolve.
- Other findings keep today's flow unchanged.
- The decision-engine check (`POST /decisions/finding/{id}`) still runs first, and a conflicting verdict still stops the close, as today.

**Value Found drawer** (`ValueFoundDrawer.jsx`, `lib/valueFound.js`, `heroFigure.js`):
- The headline reads **Saved £X**, with the line "£A stopped before payment · £R recovered".
- A separate **Being claimed** section lists open claims (supplier, amount, days since claimed). Each has two actions:
  - **Credit received**: amount, credit-note reference (required), date.
  - **Claim dropped**: optional note.

**Opportunity actions:** a **Mark realised** action (amount, currency, date, optional reference) on opportunity rows that expose stage actions, posting to `/value/opportunities/{id}/realise`.

**Copy:** all new strings go through the i18n catalogue (`t(...)`), matching the rest of SpendIQ.

**Shared checkout:** the UI working tree holds other sessions' uncommitted edits (i18n files, `App.jsx`, `index.css`). Stage only this work's hunks and verify the staged tree in a clean worktree before committing.

## 7. Errors

| situation | behaviour |
|---|---|
| Finding already closed by someone else | 409 `finding_already_moved`; the UI says it was already settled and refreshes |
| `settle` on a finding with no open claim | 409 `no_open_claim` |
| `recovered` without an evidence reference | 422; the UI marks the field required before sending |
| Amount ≤ 0, or non-numeric | 422 |
| Unknown currency, or FX unavailable | the row is written with `amount_gbp` NULL; the UI shows "£ figure unavailable (no rate for XXX)" and the item stays out of GBP totals |
| Audit write fails | the whole write is refused (500 with reason); nothing is committed |
| Update or delete attempted on the table | the database raises; covered by a test |

## 8. Testing

- **Unit** (fake DB): validation, GBP conversion, latest-state derivation, supersede handling, and summary sums including `by_month` and the triage exposure mapping.
- **Live** (`PROCWISE_TEST_LIVE_DB=1`, bp_testdb), each test cleaning up after itself:
  - the append-only trigger refuses UPDATE, DELETE and TRUNCATE
  - the outcome route closes the finding and writes the row in one transaction, and a forced insert failure leaves the finding open
  - the lifecycle 409 path
  - Cleanup: every test row carries `recorded_by = 'pytest-value-ledger'`. A teardown fixture removes them in one transaction: `ALTER TABLE … DISABLE TRIGGER` on the guard, `DELETE … WHERE recorded_by = 'pytest-value-ledger'`, `ENABLE TRIGGER`, `COMMIT`. The app role owns the table, so it may do this. The findings and opportunities the tests closed are re-opened through the lifecycle's allowed `resolved → open` edge.
- **Router:** 401 without a user, 409 and 422 paths.
- **UI:** vitest contract tests for the panel's payload, the drawer's new fields and the Being-claimed actions. Existing `valueSummary.contract.test.js` stays green.
- **Stale test:** the digest test is fixed.
- **Demonstration** on the local server with live bp_testdb data, walking the §2 success path, with screenshots.
- **Suite hygiene:** run with GPU and Ollama isolated. Compare against the 2026-09-25 baseline (13,260 passed / 318 failed / 293 errors) and add no new failures.

## 9. Out of scope

- Tenant separation and RLS (a separate project).
- Automatic recovery from ingested credit notes.
- Screens for `terms_improved` and `cycle_time` (the table allows the types; nothing writes them yet).
- Dropping the legacy columns.
- The gateway.
- Dispositions and net payable (build-status step 2).
