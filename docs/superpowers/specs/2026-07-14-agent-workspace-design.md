# Agent Workspace — Base Version

**Date:** 2026-07-14
**Status:** Approved, ready for implementation plan
**Repos:** `BP_Backend` (engine, endpoints, tables) + `beyond_procwise_ui` (SpendIQ workspace view)

---

## 1. Premise

The Agent Workspace canvas *is* the procurement process. The flow the user draws is the flow
that runs. Connectors mean "then" — plain sequence, no conditional meaning.

## 2. What exists today (and what is wrong with it)

Established by direct inspection of the running system on 2026-07-14.

| Claim the UI makes | Reality |
|---|---|
| A palette of 8 agents | **14** agents are live (`agent_definitions.json`) |
| "Save" saves your workflow | `wfSave()` (engine.js:4730) deep-copies into a JS array. No API, no localStorage. It dies on page reload. |
| Three previously saved workflows | Fabricated fixtures (`WF_SAVED`, engine.js:4678) |
| Running the canvas runs what you drew | `wfRun()` (engine.js:4754) sends **only `wfNodes[0]`**. The graph and every edge are discarded; the backend re-expands its own hardcoded workflow from that entry point. **Whatever you draw, the same fixed workflow runs.** |

Missing from the UI palette entirely: `quote_comparison`, `email_dispatch`, `email_watcher`,
`discrepancy_detection`, `rag`, `requirements`.

### Governance coverage is 5 of 14

| Table | Rows | Agents covered |
|---|---|---|
| `proc.bp_prompt` | 7 | `supplier_ranking`, `negotiation`, `requirements` (+ `summary_agent`, not one of the 14) |
| `proc.bp_policy` | 10 | `supplier_ranking`, `approvals`, `opportunity_miner`, `requirements` |

**Ungoverned (no prompt AND no policy):** `data_extraction`, `quote_comparison`,
`email_drafting`, `supplier_interaction`, `quote_evaluation`, `email_dispatch`,
`email_watcher`, `discrepancy_detection`, `rag` — 9 agents.

Decision: these run on their built-in defaults and are **labelled ungoverned in the UI**. We do
not show a pretend tick. Authoring the missing prompts/policies is content work, tracked
separately and explicitly out of scope here.

Naming wrinkle: `bp_prompt.prompt_linked_agents` uses `supplier_ranking_agent` while the
registry slug is `supplier_ranking`. The resolver must normalise (`<slug>` ↔ `<slug>_agent`).

## 3. What we can reuse (no change needed)

- **`WorkflowEngine.execute(graph, ...)`** (`src/orchestration/workflow_engine.py:359`) takes a
  `WorkflowGraph` **object**, not a registry name. A graph built at runtime from saved JSON
  executes through the existing engine unchanged.
- **`WorkflowGraph.validate()`** already checks structural validity.
- **Shared blackboard wiring** (`agent_wiring` on the engine) already passes each node's output
  to downstream nodes.
- **`BaseAgent`** already resolves prompt/policy via `PromptEngine` / `PolicyEngine`
  (`src/agents/base_agent.py:249-320`).
- **Pause/resume**: `WorkflowState.checkpoint()` (`:114`) and `execute(resume_state=...)` (`:366`).
- **Run persistence**: `proc.workflow_execution`, `proc.node_execution`, `proc.workflow_events`
  (`src/orchestration/migrations/001_workflow_execution.sql`).
- **Observability endpoints**: `GET /workflows/workflows/{id}/status`, `.../events`.

## 4. Design

### 4.1 Palette — the real 14

Driven from `GET /workflows/types`, which already returns the full `agent_definitions.json`
catalogue with `description`, `capabilities` and `required_inputs` per agent. The 8-agent
fixture (`AGENT_LIB`, `AGENTS`) and the fabricated `WF_SAVED` entries are deleted.

### 4.2 Canvas — a DAG

Nodes are agent instances; edges are sequence. Saved shape:

```json
{
  "name": "Quote to PO",
  "entry_node": "n1",
  "nodes": [
    {"id": "n1", "agent_slug": "data_extraction", "x": 120, "y": 80},
    {"id": "n2", "agent_slug": "supplier_ranking", "x": 320, "y": 80}
  ],
  "edges": [{"source": "n1", "target": "n2"}]
}
```

### 4.3 Save — persists

New table:

```sql
CREATE TABLE proc.bp_agent_workflow (
    workflow_id    BIGSERIAL PRIMARY KEY,
    name           TEXT NOT NULL,
    description    TEXT,
    graph          JSONB NOT NULL,           -- {nodes:[...], edges:[...]}
    entry_node     TEXT NOT NULL,
    is_active      BOOLEAN NOT NULL DEFAULT TRUE,
    created_by     TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_bp_agent_workflow_active ON proc.bp_agent_workflow (is_active, updated_at DESC);
```

Validation at save (reject, with the reason, rather than failing mysteriously at run):
- graph is a DAG (no cycles)
- exactly one entry node (no inbound edges)
- no unreachable nodes
- every `agent_slug` exists in the registry

### 4.4 Run — HITL input elicitation

This is the core of the base version.

#### The manifests under-declare what agents need — this must be fixed first

`agent_definitions.json` already carries `inputs {required, optional}` and `outputs` per agent.
But the `required` lists are mostly empty and the genuinely-essential inputs are marked
*optional*:

```
data_extraction   required=[]         optional=[s3_prefix, s3_object_key]   outputs=[details, summary, mismatches]
supplier_ranking  required=[query]    optional=[criteria, supplier_data, ...] outputs=[ranking, justification]
negotiation       required=[]         optional=[supplier, current_offer, ...]
approvals         required=[]         optional=[policy_context]
```

So a naive "ask for unsatisfied `required_inputs`" rule would ask for **nothing** on a
`data_extraction` node — the agent would run with no documents and silently produce nothing.
That is exactly the failure this feature exists to prevent.

Elicitation therefore runs off an explicit, declarative contract added to each agent in
`agent_definitions.json`:

```json
"elicit": [
  {
    "any_of": ["s3_prefix", "s3_object_key", "document_ids"],
    "type": "document_ids",
    "prompt": "Which documents should I extract from?"
  }
]
```

Each entry is an input *group*: at least one member must be satisfied. `type` drives the UI
control (a document picker, a text field, a number). `prompt` is what the human is asked.
This is declarative, per-agent, and general — extraction is not special-cased in code.

Authoring `elicit` for the 14 agents is part of this work (it is small, and it is the contract
the whole HITL behaviour rests on).

#### The rule

On Run, walk the DAG in topological order. For each node, for each `elicit` group:

```
satisfied(group) = any member key is
                     produced by an upstream node (its declared `outputs`)
                  OR present in the run payload
                  OR already supplied by a human answer on this run
```

Any unsatisfied group becomes an input request against that node.

Note this also catches the non-obvious cases for free: nothing in the system produces `query`,
so a `supplier_ranking` node always asks the human what to rank for.

If `missing` is non-empty for any node, the run does **not** guess and does **not** fabricate.
It enters `awaiting_input`, checkpoints, and raises an input request bound to that node. The UI
renders the request on the node itself.

The motivating case: a `data_extraction` node with no upstream producer needs documents, so the
workflow asks the user which documents to extract from. The same mechanism covers every other
unsatisfied input on every other agent — nothing is special-cased to extraction.

The human answers; the answer is merged into `shared_data`; the run resumes from the checkpoint.

Requests and answers are persisted so a run is auditable — who supplied what, and when:

```sql
CREATE TABLE proc.bp_workflow_input_request (
    request_id     BIGSERIAL PRIMARY KEY,
    workflow_id    TEXT NOT NULL,          -- the RUN id (workflow_execution.workflow_id)
    node_name      TEXT NOT NULL,
    agent_slug     TEXT NOT NULL,
    required_field TEXT NOT NULL,
    field_type     TEXT,                   -- e.g. 'document_ids', 'text', 'number'
    status         TEXT NOT NULL DEFAULT 'pending',   -- pending | answered | cancelled
    answer         JSONB,
    answered_by    TEXT,
    requested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    answered_at    TIMESTAMPTZ
);
CREATE INDEX ix_bp_workflow_input_request_open
    ON proc.bp_workflow_input_request (workflow_id, status);
```

### 4.5 Execution

The saved JSON is compiled to a `WorkflowGraph` at runtime:

- `WorkflowNode(name=node.id, agent_type=node.agent_slug, output_to_shared=<agent's declared outputs>)`
- `WorkflowEdge(source, target, condition=None)` — unconditional; a drawn edge always fires
- `entry_node` from the saved graph

Then `WorkflowEngine.execute(graph, input_data=..., user_id=..., workflow_id=...)`. No engine
rewrite. Data flows node→node over the existing shared blackboard.

### 4.6 Governance per node

Each node's agent resolves its prompt and policy through the engines at run time. The run trace
records **which** prompt and policy were actually applied (name + version), and the UI shows it
on the node. Ungoverned agents show "built-in default" — explicitly, not silently.

### 4.7 Progress

The canvas reflects real node state (pending → running → **awaiting input** → completed /
failed / skipped), polled from `GET /workflows/workflows/{id}/status` and `.../events`.

### 4.8 Robustness

- Cycle / multi-entry / unreachable-node / unknown-agent → rejected at save with the reason.
- A **required** node that fails halts its downstream; a non-required node is skipped and the
  run continues.
- Retries and timeouts use the per-node fields the engine already supports.
- Every run, node execution, event and human input is persisted.

## 4.9 Manifest changes

Add an `elicit` block (§4.4) to each of the 14 agents in `agent_definitions.json`. Existing
fields (`inputs`, `outputs`, `required_inputs`) are left untouched — other code reads them, and
`elicit` is additive so nothing else changes behaviour.

## 5. Engine changes (small, additive)

- `NodeStatus.AWAITING_INPUT`
- `WorkflowState.status = "awaiting_input"`
- Pause on unsatisfied input → checkpoint + emit input request
- Resume path that merges supplied answers into `shared_data`

## 6. New endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/agent-workflows` | list saved workflows |
| POST | `/agent-workflows` | create (validates the DAG) |
| GET | `/agent-workflows/{id}` | fetch one |
| PUT | `/agent-workflows/{id}` | update |
| DELETE | `/agent-workflows/{id}` | soft-delete (`is_active=false`) |
| POST | `/agent-workflows/{id}/run` | compile + execute; may return `awaiting_input` |
| GET | `/agent-workflows/runs/{run_id}` | run status, node states, pending input request |
| POST | `/agent-workflows/runs/{run_id}/input` | submit human answer, resume the run |

## 7. UI changes (`src/modules/SpendIQ/engine.js`)

- `agentWorkspaceView()` (4810-4837) rewritten
- Delete `AGENT_LIB` (4650), `AGENTS` (4666), `WF_SAVED` fixtures (4677-4681)
- `wfSave()` (4730) → real `POST /agent-workflows`
- `wfRun()` (4754) → real `POST /agent-workflows/{id}/run`; stop discarding the graph
- New: HITL input panel on a node, live node status, governance badge per node
- Network goes through the `window` bridge in `index.jsx` (engine.js is a classic script and
  cannot `import` or use `axios`)

## 8. Testing

- **Unit (backend):** DAG compiler (JSON → `WorkflowGraph`); validation rejects cycles, multiple
  entries, unreachable nodes, unknown agent slugs; the `elicit` satisfaction rule (a group is
  satisfied by an upstream `outputs` key, a run-payload key, or a prior human answer — and
  unsatisfied otherwise); the prompt/policy name normaliser
  (`supplier_ranking` ↔ `supplier_ranking_agent`).
- **Regression guard:** a `data_extraction` node with no upstream producer MUST raise a document
  request. This is the case today's manifests would silently skip, so it gets an explicit test.
- **Unit (UI):** graph serialisation round-trip.
- **Integration:** save a 2-node flow (`data_extraction` → `supplier_ranking`); run it; assert it
  halts at `awaiting_input` asking for documents; submit documents; assert it resumes, both nodes
  complete, and the extraction actually ran on the supplied documents.
- **Live:** demonstrated on the running local stack against `bp_sqldb`, not only in tests.

## 9. Explicitly out of scope

- Conditional / branching edges (connectors carry no condition by decision)
- Authoring the 9 missing prompts and policies
- Retry/timeout configuration in the UI
- Parallel fan-out authoring (the engine supports it; the base UI draws sequential DAGs)
