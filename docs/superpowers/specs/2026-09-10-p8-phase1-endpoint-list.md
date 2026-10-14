# P8 phase 1 — the write/send endpoints that do not resolve a caller

Produced 2026-09-10, before any code change, as phase 1 asks. Derived by parsing
`src/api/routers/*.py` rather than by reading, so the counts are the code's and
not an impression of it (`scratchpad/p8_scan.py`).

## The shape of it

| | count |
|---|---|
| write/send endpoints (POST/PUT/PATCH/DELETE) in `src/api/routers` | 87 |
| …taking `Depends(require_user)` | 30 |
| …**not** taking it | **57** |

Every one of the 57 is still *authenticated* — `api/main.py` mounts every router
behind `require_user`, so an unidentified caller is refused at the door. What
they do not do is **resolve** the caller into the handler, so the handler cannot
attribute what it writes, and cannot apply "an agent may never exceed the human
it acts for" because it never learns who the human is.

## Phase 1 is not all 57

The prompt says the highest-risk group only: anything mutating the supplier
master, a deal, an approval, a policy, an agent or a workflow. Rather than
judging "high risk" by feel, the cut is evidential — **the endpoints that
currently accept an identity from the caller and write it**. Those are the ones
where attribution is not merely absent but *forgeable*: today a caller types a
name and the record keeps it.

There are eight, and they are all inside the named categories.

| Router | Endpoint | What it mutates | Caller-supplied identity |
|---|---|---|---|
| `promotion.py` | `POST /promotion/link-proposals/{doc_type}/{doc_pk}/confirm` | a document link, which feeds `_stg → _trgt` — the financial record | `body.reviewer` |
| `promotion.py` | `POST /promotion/link-proposals/{doc_type}/{doc_pk}/reject` | same | `body.reviewer` |
| `supplier_review.py` | `POST /suppliers/reviews/{review_id}/confirm` | the supplier master (alias / merge) | `body.reviewer` |
| `supplier_review.py` | `POST /suppliers/reviews/{review_id}/reject` | supplier review state | `body.reviewer` |
| `supplier_research.py` | `POST /suppliers/enrichment/{id}/reject` | enrichment review state | `body.reviewer` |
| `analysis.py` | `POST /analysis` | `proc.bp_analysis` — a deal's analysis event | `body.created_by` |
| `workflows.py` | `POST /workflows/opportunities/{id}/reject` | an opportunity (deal pipeline) | `req.user_id` |
| `workflows.py` | `POST /workflows/email/{unique_id}/attachments` | a draft that will be SENT to a supplier | `user_id` → `added_by` |

`promotion.py`'s third write endpoint, `POST /promotion/review/{doc_type}/{doc_pk}/approve`,
already does this correctly (`getattr(principal, "subject", None) or body.get("reviewer")`)
and is the pattern the eight are being brought to.

## The other 49, for phase 2

Grouped by router, with the phase-1 category they touch. None of them accepts a
caller-supplied identity today, so nothing can be *forged* through them — what
they lack is attribution and any basis for a per-role limit.

- **agent / workflow** — `agent_groups.py` (create, update, delete),
  `agents.py` (execute, reason, process-document), `agent_workflows.py`
  (runs/{id}/input), `run.py` (run), `stream.py` (plan),
  `workflows.py` (extract, rank, negotiate, quotes/evaluate, opportunities,
  supplier-interaction, discrepancy, approvals, attachment delete)
- **policy / governance** — `governance.py` (govern), `extraction_feedback.py`
  (proposals run / approve / reject)
- **deal** — `deal_proposals.py` (generate, confirm, reject, members),
  `deal_summary.py` (promote, reconcile, save-reference, analysis-summary/sync),
  `negotiate.py` (advice message, advice fact delete),
  `opportunities.py` (sync, stage, link-deals)
- **supplier** — `supplier_review.py` (reviews/sweep), `vendors.py`
  (onboard upload, correct, save), `promotion.py` (canonicalize-po)
- **other** — `analysis.py`, `benchmark.py` (preview), `email.py`
  (emailwatcher), `requirements.py` (message, run-workflow), `summary.py`
  (post, precompute), `support.py` (contact, contact/stream, confirm)

## The limitation that survives phase 1

Resolving the principal only helps where there IS one. This sandbox runs
`ASK_AUTH_MODE=off`, so `require_user` returns `None` and every endpoint changed
here falls back to recording nothing rather than a verified subject. P3's
self-approval bar, P4's upload ownership and P5's enrichment attribution all
carry the same caveat. Phase 1 makes the code correct; only enforcement makes it
true.
