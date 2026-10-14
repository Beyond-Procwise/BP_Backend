# Playbook Layer — Governed Strategy, Proposed Not Executed (Conformance P4)

**Date:** 2026-09-27
**Status:** Design, awaiting review
**Author:** Nick + Claude
**Supersedes:** the P4 section of `2026-06-28-policy-decision-playbook-layer-design.md` (commit 5974fcc)

## Intent

A playbook is human-authored expert strategy that the system executes and never
invents. This layer gives the system a way to say: *when a finding like this
appears, the strategy a procurement expert wrote for it is that one* — and then
to put that recommendation in front of a person rather than acting on it.

**Stated by Nick, 2026-09-27:**

- A selected playbook is **proposed, not run**. The selector queues it; a person
  approves; only then does it execute. Nothing runs unattended.
- The three `supplier_ranking` policy rows **stay where they are**. Moving them
  is a separate change with its own parity proof.

**Assumptions I am making** (correct these if wrong):

- The expert authors strategy by drawing a workflow on the existing canvas, not
  by writing a new kind of artefact. A playbook points at that drawing.
- A proposal is reviewed wherever findings are already reviewed; this spec adds
  the data and the endpoints, not a new screen design.

**Success looks like:** an expert can author a strategy, get it approved, and
from then on every matching finding raises a proposal naming that strategy and
the evidence for the match — with no proposal ever appearing twice for the same
finding, and no strategy ever running because the system chose to.

## What already exists (verified 2026-09-27, bp_testdb)

The June design predates most of its own P4. Three of its four tables would now
duplicate working machinery.

| June design wanted | Already exists |
|---|---|
| `bp_playbook_step` | `proc.bp_agent_workflow.graph` (JSONB) — 10 saved graphs, `workflow_id BIGSERIAL`, `entry_node`, `is_active` |
| step execution | `orchestration/workflow_compiler.compile_graph()` → `WorkflowGraph` → `WorkflowEngine`, with per-node statuses |
| `bp_playbook_run` | `proc.bp_workflow_run` — `run_id TEXT`, `agent_workflow_id BIGINT`, `payload`, `status`, `initiated_by` (15 rows) |
| `bp_playbook_run_step` `mode=human_gated` | `proc.bp_workflow_input_request` — `node_name`, `prompt`, `answer`, `answered_by` (22 rows) |
| gate before each step | `gate("workflow.run" / "workflow.save")` in `api/routers/agent_workflows.py` |
| `bp_governance_change` audit | `proc.bp_agent_actions` — append-only, immutable by trigger (2026-09-16) |

**What does not exist, and is therefore what this builds:**

1. **Selection.** Nothing connects a finding to a strategy. All 10 workflows run
   only because a person pressed run.
2. **Authorship lifecycle.** `bp_agent_workflow` has an `is_active` boolean. No
   draft → pending_approval → active → retired, no `approved_by`, no version.
3. **A proposal queue.** No record of "this strategy was recommended for this
   finding, and here is who decided what."

### The finding surfaces a playbook can trigger on

Two stores, two vocabularies. Both are in scope; they are matched separately,
never merged.

`proc.bp_detection_finding` — 8,147 rows, 4,838 `status='open'`:

| `rule_id` | `category` | severities present |
|---|---|---|
| `quantity` | quantity | critical (6,415), warning (144) |
| `cumulative_total` | overbilling | critical (447) |
| `line_arithmetic` | arithmetic | warning (407), critical (309) |
| `duplicate` | duplicate | critical (300) |
| `unit_price` | price | warning (59), critical (58) |
| `uniform_uplift` | price | warning (4) |
| `bad_po_ref` | linking | warning (3) |
| `unlinked_line` | linking | critical (1) |

`proc.bp_opportunity` — 308 rows, matched on `detector_type`: Duplicate Invoice
Recovery (300), Invoice Overbilling (6), Price Benchmark Variance (2).

## Scope

**In.** Two tables; a playbook store; a deterministic selector; a proposer; a
scheduler sweep; list/approve/reject endpoints; two new action names and their
policies; the authorship lifecycle including the self-approval bar.

**Out, deliberately.**

- **Auto-execution.** Nothing runs without a person. Revisit only as its own change.
- **Moving the `supplier_ranking` rows.** They sit in a live scoring path.
- **Reconciling the two rule vocabularies.** `bp_detection_finding.rule_id`
  holds triage check codes (`cumulative_total`, `line_arithmetic`, …) which are
  a different namespace from `proc.bp_rule.detector_slug`
  (`price_variance_check`, …). This spec matches each store on its own field
  and does not unify them. Recorded as a known seam below.
- **A new review screen.** Endpoints and data only.

## Data model

Both tables are additive and idempotent, applied to bp_testdb and bp_sqldb, with
a rollback script. `bp_` prefix, `ix_bp_*` / `ux_bp_*` indexes, per convention.

```sql
CREATE TABLE IF NOT EXISTS proc.bp_playbook (
    playbook_id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    playbook_name     TEXT        NOT NULL,
    description       TEXT,
    -- Which finding store this playbook watches.
    trigger_source    TEXT        NOT NULL
        CHECK (trigger_source IN ('detection_finding', 'opportunity')),
    -- Equality match against that store's own columns. Keys are validated at
    -- write time against the per-source allow-list below; an unknown key is
    -- refused rather than stored, because a key that matches nothing is a
    -- playbook that silently never fires.
    --   detection_finding: rule_id, category, severity, doc_type, blocks_promotion
    --   opportunity:       detector_type, supplier_id, category_id
    trigger_match     JSONB       NOT NULL DEFAULT '{}',
    -- The expert's strategy: a graph already drawn and saved.
    agent_workflow_id BIGINT      NOT NULL REFERENCES proc.bp_agent_workflow (workflow_id),
    -- Extra static inputs merged into the run payload on execution.
    params            JSONB       NOT NULL DEFAULT '{}',
    playbook_status   TEXT        NOT NULL DEFAULT 'draft'
        CHECK (playbook_status IN ('draft','pending_approval','active','retired')),
    version           INTEGER     NOT NULL DEFAULT 1,
    authored_by       TEXT        NOT NULL,
    approved_by       TEXT,
    approved_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_by  TEXT        NOT NULL DEFAULT 'system',
    -- Active means approved. The two cannot drift apart.
    CONSTRAINT ck_bp_playbook_active_is_approved
        CHECK (playbook_status <> 'active' OR approved_by IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS ix_bp_playbook_status
    ON proc.bp_playbook (playbook_status);
CREATE INDEX IF NOT EXISTS ix_bp_playbook_source
    ON proc.bp_playbook (trigger_source, playbook_status);

CREATE TABLE IF NOT EXISTS proc.bp_playbook_proposal (
    proposal_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    playbook_id    BIGINT      NOT NULL REFERENCES proc.bp_playbook (playbook_id),
    finding_source TEXT        NOT NULL
        CHECK (finding_source IN ('detection_finding', 'opportunity')),
    -- TEXT because the two stores disagree on type:
    -- bp_detection_finding.finding_id is BIGINT (stored here as ::text) and
    -- bp_opportunity.opportunity_id is VARCHAR. Deliberately no foreign key --
    -- one column cannot reference two tables, and finding_source says which.
    finding_id     TEXT        NOT NULL,
    deal_id        TEXT,
    proposal_status TEXT       NOT NULL DEFAULT 'proposed'
        CHECK (proposal_status IN ('proposed','approved','rejected','executed','superseded')),
    -- Which match keys fired, and what the finding's values were. A proposal
    -- must be re-derivable from source, like a decision.
    evidence       JSONB       NOT NULL DEFAULT '{}',
    run_id         TEXT        REFERENCES proc.bp_workflow_run (run_id),
    proposed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_by     TEXT,
    decided_at     TIMESTAMPTZ,
    decision_reason TEXT
);

-- Idempotency. A sweep that runs twice proposes once.
CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_playbook_proposal_finding
    ON proc.bp_playbook_proposal (playbook_id, finding_source, finding_id);
CREATE INDEX IF NOT EXISTS ix_bp_playbook_proposal_status
    ON proc.bp_playbook_proposal (proposal_status, proposed_at DESC);
```

Nothing is written to `proc.bp_agent_workflow`; a playbook references it.

## Components

Each has one job and is testable with injected rows, following `RuleBook`.

**`src/services/playbooks/store.py`** — `PlaybookStore`. Loads active playbooks,
caches, `reload()`. Constructor takes `agent_nick` / `connection_factory` /
`playbook_rows=` exactly as `RuleBook` does.

**`src/services/playbooks/selector.py`** — `select(finding) -> Selection | None`.
Pure. No I/O. Given a normalised finding and the loaded playbooks, returns the
one that governs it, or nothing. Algorithm below.

**`src/services/playbooks/finding_source.py`** — normalises a row from either
store into `Finding(source, finding_id, deal_id, attrs: dict)`, where `attrs` is
that store's own match fields. Keeps the two vocabularies apart while giving the
selector one shape.

**`src/services/playbooks/proposer.py`** — `propose(finding, selection)`. Writes
one row, `ON CONFLICT DO NOTHING` on the unique index, and records a
`playbook.propose` event to `bp_agent_actions`.

**`src/services/playbooks/sweep.py`** — iterates open findings from both sources
in batches, selects, proposes. Returns counts. Registered on `BackendScheduler`
following the existing `_register_*_sweep_job` / `_run_*_sweep` pattern.

**`src/api/routers/playbooks.py`** — CRUD for playbooks under the lifecycle,
plus `GET /playbooks/proposals`, `POST /playbooks/proposals/{id}/approve`,
`POST /playbooks/proposals/{id}/reject`.

## Selection

Deterministic, and it refuses rather than guesses.

1. Take the active playbooks whose `trigger_source` equals the finding's source.
2. Keep those whose every `trigger_match` key equals the finding's value for that
   key. Equality only — no fuzzy matching, no aliases, no substring matching.
3. Of those, the winner is the one with the **most match keys** (most specific).
4. **If two or more tie on key count, propose nothing.** Log both playbooks by
   name and id at ERROR, and record a `playbook.ambiguous` event.

Step 4 is the load-bearing one. The detection registry this codebase just
removed bound policies to detectors by accumulating aliases and letting whichever
matched last win; four of five bindings were silently wrong for months. An
ambiguous playbook match is a configuration error, and a configuration error that
resolves itself quietly is the same failure in a new table. A tie is visible or
it is nothing.

An empty `trigger_match` (`{}`) matches every finding of that source with zero
keys — legitimate as a catch-all, and it loses to any more specific playbook.

## Data flow

```
scheduler sweep
   │  open findings: bp_detection_finding (status='open'), bp_opportunity
   ▼
finding_source.normalise  →  Finding(source, finding_id, deal_id, attrs)
   ▼
selector.select           →  the one governing playbook, or nothing
   ▼
proposer.propose          →  bp_playbook_proposal (idempotent) + bp_agent_actions
   ▼
   ── a person reviews ──
   ▼
POST /playbooks/proposals/{id}/approve
   │  gate("workflow.run", principal)
   ▼
existing agent_workflows run path: compile_graph → WorkflowEngine
   ▼
proposal.run_id set, proposal_status='executed'
```

The run itself is the existing path, unchanged. Human-gated steps inside the
graph continue to use `bp_workflow_input_request`.

## Governance and authorization

Two names added to the closed vocabulary in `services/actions.py`, each with a
`bp_policy` authority row. The class vocabulary is fixed by `RoleDefinitionPolicy`
(`read, compute, write, communicate, share, transact, configure, delegate,
approve_email`) — there is no generic `approve` class, so both follow the
`policy.write` / `prompt.write` precedent:

- `playbook.write` — class **`configure`**. Create or edit a playbook, submit for
  approval.
- `playbook.approve` — class **`configure`**. Move a playbook to `active`.

Approving a *proposal* is a different act — it causes a workflow to run — and
gates on the **existing `workflow.run`** (class `delegate`), which is already
policied. No new action is introduced for it.

**Self-approval is barred.** `approved_by` must differ from `authored_by`; the
endpoint refuses otherwise. This follows the existing bar recorded for drafts and
`ReportSignoffPolicy.self_approval`.

**Lifecycle.** `draft → pending_approval → active → retired`. Only `active`
playbooks are loaded by the store and can therefore propose anything. Editing an
`active` playbook returns it to `pending_approval` and increments `version`, so
an approved strategy cannot be changed underneath its approval.

**Audit.** `playbook.authored`, `playbook.approved`, `playbook.retired`,
`playbook.propose`, `playbook.ambiguous`, `proposal.approved`,
`proposal.rejected`, `proposal.executed` are written to `proc.bp_agent_actions`.

## Error handling

**An unreadable store raises.** A failed query against `bp_playbook` raises
`PlaybookStoreUnavailable`; it does not return `[]`.

**An empty store does not raise, and this differs from the rule book on
purpose.** `RuleBook` treats zero rules as an outage because a sweep that runs no
detectors reports no findings and looks exactly like a clean scan. Zero
*playbooks* is the honest state on the day this ships and for as long as nobody
has authored one; failing closed there would make the service unbootable until an
expert writes a strategy. The distinction: an empty rule book hides work that
should have happened, an empty playbook table simply means no strategy is on
file. The sweep logs the count it proposed each run, so "zero" stays visible.

**A playbook pointing at a deleted or inactive workflow** is skipped at selection
with an ERROR naming both, and cannot be moved to `active` in the first place —
the approve endpoint checks the target workflow exists and is active.

**A proposal whose finding has since been resolved** is marked `superseded` at
approval time rather than executed, so a stale queue cannot act on closed work.

## Testing

Unit, with injected rows, no live DB:

- store: loads active only; unreadable raises; empty does not raise
- selector: exact match; most-specific wins; **tie proposes nothing and names both**;
  empty `trigger_match` is a catch-all and loses to a specific one; wrong source never matches
- finding_source: both stores normalise; the two vocabularies do not cross
- proposer: same finding twice → one row
- lifecycle: edit of an `active` playbook returns it to `pending_approval`, bumps version
- self-approval refused; approving a playbook whose workflow is inactive refused

Live, gated on `PROCWISE_TEST_LIVE_DB=1`, both databases:

- tables and both indexes exist, including the unique proposal index
- the `active ⇒ approved_by IS NOT NULL` constraint holds
- migration is idempotent across repeated runs

**Guard proofs are part of the deliverable, not a nicety.** Every guard above is
broken on purpose and watched fail before it is believed: drop the unique index
and re-sweep to see duplicates appear; make two playbooks tie and confirm nothing
is proposed; set `approved_by = authored_by` and confirm refusal; insert an
`active` row with a null `approved_by` and confirm the constraint rejects it.
Three guards have shipped green while checking nothing in this codebase; these
will not be the next three.

## Known seams, recorded not fixed

- **Two rule vocabularies.** `bp_detection_finding.rule_id` (triage check codes)
  and `proc.bp_rule.detector_slug` (opportunity detectors) are separate
  namespaces. A playbook matches within one source and never across. Unifying
  them is its own piece of work.
- **`_enrich_provided_policy`** still reads active `bp_policy` rows to enrich
  caller-supplied policies. Closed by data, not structure, and guarded by a live
  test. Unchanged here.
- **Shadow mode.** Thirteen actions, including `workflow.run`, are gate-evaluated
  but permitted until 2026-10-09. Until that lapses, the approve endpoint's gate
  records its verdict without enforcing it. The proposal queue is unaffected —
  the human step is structural, not policy-enforced.

## Rollout

1. Migration applied to both databases, with rollback.
2. Store, selector, finding source, proposer — unit-tested, no wiring.
3. Endpoints under the lifecycle and the two new gates.
4. Sweep registered on the scheduler, initially proposing into an empty playbook
   table, so it is a no-op until an expert authors something.
5. One real playbook authored end to end against live findings as the acceptance
   proof, on the local server, with the proposal it raises inspected by hand.
