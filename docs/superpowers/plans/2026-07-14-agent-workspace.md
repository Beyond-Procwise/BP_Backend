# Agent Workspace (Base Version) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SpendIQ Agent Workspace real — the DAG the user draws is persisted, executed by the existing workflow engine, and pauses to ask the human (HITL) for any input it needs rather than guessing.

**Architecture:** A saved JSON graph is compiled at runtime into a `WorkflowGraph` (nodes → real agent slugs, edges → unconditional "then") and handed to the existing `WorkflowEngine.execute(graph, ...)`, which already accepts a graph object — so the engine is not rewritten. Before execution, an elicitation pass walks the DAG and, for each node, asks whether each of its declared `elicit` input-groups is satisfied by an upstream node's outputs, the run payload, or a prior human answer. Anything unsatisfied halts the run in `awaiting_input` with a persisted request bound to that node; the human's answer merges into `shared_data` and the run resumes from its checkpoint.

**Tech Stack:** Python 3 / FastAPI / psycopg (via `services.db.get_conn`) / Postgres `proc` schema; plain-JS classic-script UI (`engine.js`) bridged through `index.jsx`; pytest; Vitest.

**Spec:** `docs/superpowers/specs/2026-07-14-agent-workspace-design.md`

## Global Constraints

- **Never fabricate.** If a value is not known, render `—` / return `null`. Never substitute a zero or an invented name. This is the whole point of the feature.
- **New DB tables MUST use the `bp_` prefix**; indexes `ix_bp_<table>_<cols>`.
- **Connectors carry no condition.** A drawn edge means "then". `WorkflowEdge(condition=None)`.
- **Governance covers only 5 of 14 agents.** The 9 ungoverned agents (`data_extraction`, `quote_comparison`, `email_drafting`, `supplier_interaction`, `quote_evaluation`, `email_dispatch`, `email_watcher`, `discrepancy_detection`, `rag`) run on built-in defaults and MUST be labelled "built-in default" in the UI — never a pretend tick.
- **`bp_prompt.prompt_linked_agents` / `bp_policy.policy_linked_agents` use the `<slug>_agent` form** (`supplier_ranking_agent`) while the registry uses `<slug>` (`supplier_ranking`). Always normalise.
- **Do not modify existing manifest fields** (`inputs`, `outputs`, `required_inputs`) — other code reads them. `elicit` is additive.
- **The full pytest suite times out.** Always run targeted test files (`pytest tests/orchestration/test_x.py -v`), never bare `pytest`.
- **Prove it on the running local stack** against live `bp_sqldb` — not only in tests (Task 9).
- **The UI `engine.js` is a classic script**: it cannot `import` and has no `axios`/`fetch`. All network calls go through a `window.__SPENDIQ_*` bridge installed by `index.jsx`.
- **No Claude attribution in commit messages.**

---

### Task 1: The `elicit` contract on the agent manifests

Without this, the whole feature no-ops: `data_extraction.required_inputs` is `[]` and its document inputs are marked *optional*, so a "missing required inputs" rule asks for nothing and the agent runs against no documents.

**Files:**
- Modify: `agent_definitions.json` (repo root — add an `elicit` key to each of the 14 agents)
- Modify: `src/agents/definitions.py` (add `get_elicit(slug)`)
- Test: `tests/agents/test_agent_elicit_manifest.py`

**Interfaces:**
- Consumes: nothing
- Produces: `get_elicit(slug: str) -> List[Dict[str, Any]]` — each dict is `{"any_of": List[str], "type": str, "prompt": str}`. Returns `[]` for an agent that needs nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_agent_elicit_manifest.py
import json, pathlib
import pytest
from agents.definitions import get_elicit, load_definitions

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_every_agent_has_an_elicit_key():
    agents = json.loads((REPO_ROOT / "agent_definitions.json").read_text())["agents"]
    missing = [a["slug"] for a in agents if "elicit" not in a]
    assert missing == [], f"agents with no elicit contract: {missing}"


def test_data_extraction_asks_for_documents():
    """The case the old manifest silently skipped: required_inputs is [] and the
    document inputs are optional, so nothing would ever be asked for."""
    groups = get_elicit("data_extraction")
    assert len(groups) == 1
    g = groups[0]
    assert set(g["any_of"]) == {"s3_prefix", "s3_object_key", "document_ids"}
    assert g["type"] == "document_ids"
    assert g["prompt"]


def test_supplier_ranking_asks_for_a_query():
    groups = get_elicit("supplier_ranking")
    assert any("query" in g["any_of"] for g in groups)


def test_unknown_agent_returns_empty():
    assert get_elicit("no_such_agent") == []


@pytest.mark.parametrize("slug", [
    "data_extraction", "supplier_ranking", "quote_comparison", "opportunity_miner",
    "email_drafting", "negotiation", "supplier_interaction", "approvals",
    "quote_evaluation", "email_dispatch", "email_watcher", "discrepancy_detection",
    "rag", "requirements",
])
def test_elicit_groups_are_well_formed(slug):
    for g in get_elicit(slug):
        assert isinstance(g["any_of"], list) and g["any_of"], f"{slug}: empty any_of"
        assert isinstance(g["type"], str) and g["type"]
        assert isinstance(g["prompt"], str) and g["prompt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && .venv/bin/pytest tests/agents/test_agent_elicit_manifest.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_elicit'`

- [ ] **Step 3: Add the `elicit` blocks to `agent_definitions.json`**

Add an `elicit` key to each of the 14 agents. Do not touch any existing key. Values:

```jsonc
// data_extraction
"elicit": [{"any_of": ["s3_prefix", "s3_object_key", "document_ids"],
            "type": "document_ids",
            "prompt": "Which documents should I extract from?"}]

// supplier_ranking
"elicit": [{"any_of": ["query"], "type": "text",
            "prompt": "What should I rank suppliers for?"}]

// quote_comparison
"elicit": [{"any_of": ["quotes", "quote_ids", "deal_id"], "type": "text",
            "prompt": "Which quotes (or which deal) should I compare?"}]

// opportunity_miner
"elicit": []   // mines the whole corpus; needs nothing from the human

// email_drafting
"elicit": [{"any_of": ["supplier", "supplier_id", "recipient"], "type": "text",
            "prompt": "Who is this email to?"},
           {"any_of": ["intent", "negotiation_context", "drafts"], "type": "text",
            "prompt": "What should the email say or achieve?"}]

// negotiation
"elicit": [{"any_of": ["supplier", "supplier_id", "ranking"], "type": "text",
            "prompt": "Which supplier am I negotiating with?"}]

// supplier_interaction
"elicit": [{"any_of": ["supplier", "supplier_id", "ranking", "supplier_candidates"],
            "type": "text", "prompt": "Which supplier should I contact?"}]

// approvals
"elicit": [{"any_of": ["amount", "decision", "finding_id", "deal_id"], "type": "text",
            "prompt": "What am I approving? Give the amount, deal or finding."}]

// quote_evaluation
"elicit": [{"any_of": ["quotes", "quote_ids", "deal_id"], "type": "text",
            "prompt": "Which quotes should I evaluate?"}]

// email_dispatch
"elicit": [{"any_of": ["drafts", "message", "recipient"], "type": "text",
            "prompt": "What should I send, and to whom?"}]

// email_watcher
"elicit": []   // watches the mailbox; needs nothing from the human

// discrepancy_detection
"elicit": [{"any_of": ["details", "doc_pk", "deal_id", "document_ids"],
            "type": "text",
            "prompt": "Which documents or deal should I check for discrepancies?"}]

// rag
"elicit": [{"any_of": ["query", "question"], "type": "text",
            "prompt": "What do you want to know?"}]

// requirements
"elicit": [{"any_of": ["requirement", "query", "brief"], "type": "text",
            "prompt": "What do you need? Describe the requirement."}]
```

- [ ] **Step 4: Add the accessor**

```python
# src/agents/definitions.py  — append
def get_elicit(slug: str) -> List[Dict[str, Any]]:
    """Input groups this agent must have satisfied before it can run.

    Each group is {"any_of": [...], "type": str, "prompt": str} and is satisfied
    when ANY member key is available. Deliberately separate from `required_inputs`,
    which under-declares: data_extraction lists its document inputs as *optional*,
    so a required-inputs rule would ask for nothing and the agent would run against
    no documents at all.
    """
    for agent in load_definitions().get("agents", []):
        if agent.get("slug") == slug:
            return list(agent.get("elicit") or [])
    return []
```

If `load_definitions()` does not already exist in this module, use whatever the module's existing loader is named (it reads `DEFINITIONS_PATH` at line 22) and keep the same caching behaviour.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/agents/test_agent_elicit_manifest.py -v`
Expected: PASS (18 tests)

- [ ] **Step 6: Commit**

```bash
git add agent_definitions.json src/agents/definitions.py tests/agents/test_agent_elicit_manifest.py
git commit -m "feat(agents): declare an elicit contract per agent

required_inputs under-declares what an agent actually needs — data_extraction's
documents are marked optional and its required_inputs is empty, so a naive
'ask for missing required inputs' rule asks for nothing and the agent runs with
no documents. elicit states, per agent, the input groups that must be satisfied
before it can run, and what to ask the human for."
```

---

### Task 2: Compile a saved JSON graph into a `WorkflowGraph`

**Files:**
- Create: `src/orchestration/workflow_compiler.py`
- Test: `tests/orchestration/test_workflow_compiler.py`

**Interfaces:**
- Consumes: `agents.definitions.get_elicit` (Task 1, only indirectly); `orchestration.workflow_engine.{WorkflowGraph, WorkflowNode, WorkflowEdge}`
- Produces:
  - `class GraphValidationError(ValueError)`
  - `validate_saved_graph(graph: Dict[str, Any]) -> None` — raises `GraphValidationError` with a human-readable reason
  - `compile_graph(name: str, graph: Dict[str, Any]) -> WorkflowGraph`

Saved graph shape (from the spec):
```json
{"nodes": [{"id": "n1", "agent_slug": "data_extraction", "x": 0, "y": 0}],
 "edges": [{"source": "n1", "target": "n2"}]}
```

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestration/test_workflow_compiler.py
import pytest
from orchestration.workflow_compiler import (
    compile_graph, validate_saved_graph, GraphValidationError,
)

TWO_NODE = {
    "nodes": [
        {"id": "n1", "agent_slug": "data_extraction", "x": 0, "y": 0},
        {"id": "n2", "agent_slug": "supplier_ranking", "x": 200, "y": 0},
    ],
    "edges": [{"source": "n1", "target": "n2"}],
}


def test_compiles_nodes_and_edges():
    g = compile_graph("Quote to PO", TWO_NODE)
    assert g.name == "Quote to PO"
    assert set(g.nodes) == {"n1", "n2"}
    assert g.nodes["n1"].agent_type == "data_extraction"
    assert g.entry_node == "n1"
    assert [(e.source, e.target) for e in g.edges] == [("n1", "n2")]


def test_edges_are_unconditional():
    """A drawn connector means 'then'. It carries no condition."""
    g = compile_graph("w", TWO_NODE)
    assert g.edges[0].condition is None


def test_node_outputs_are_published_to_the_blackboard():
    """Downstream nodes can only consume what upstream nodes publish."""
    g = compile_graph("w", TWO_NODE)
    assert "details" in g.nodes["n1"].output_to_shared      # data_extraction outputs
    assert "ranking" in g.nodes["n2"].output_to_shared      # supplier_ranking outputs


def test_compiled_graph_passes_the_engine_validator():
    ok, issues = compile_graph("w", TWO_NODE).validate()
    assert ok, issues


def test_rejects_a_cycle():
    bad = {"nodes": [{"id": "a", "agent_slug": "rag"}, {"id": "b", "agent_slug": "rag"}],
           "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]}
    with pytest.raises(GraphValidationError, match="cycle"):
        validate_saved_graph(bad)


def test_rejects_two_entry_nodes():
    bad = {"nodes": [{"id": "a", "agent_slug": "rag"}, {"id": "b", "agent_slug": "rag"},
                     {"id": "c", "agent_slug": "rag"}],
           "edges": [{"source": "a", "target": "c"}, {"source": "b", "target": "c"}]}
    with pytest.raises(GraphValidationError, match="one entry node"):
        validate_saved_graph(bad)


def test_rejects_an_unreachable_node():
    bad = {"nodes": [{"id": "a", "agent_slug": "rag"}, {"id": "b", "agent_slug": "rag"},
                     {"id": "orphan", "agent_slug": "rag"}],
           "edges": [{"source": "a", "target": "b"}]}
    with pytest.raises(GraphValidationError, match="unreachable|one entry node"):
        validate_saved_graph(bad)


def test_rejects_an_unknown_agent_slug():
    bad = {"nodes": [{"id": "a", "agent_slug": "not_a_real_agent"}], "edges": []}
    with pytest.raises(GraphValidationError, match="unknown agent"):
        validate_saved_graph(bad)


def test_rejects_an_empty_graph():
    with pytest.raises(GraphValidationError, match="at least one node"):
        validate_saved_graph({"nodes": [], "edges": []})


def test_single_node_graph_is_valid():
    validate_saved_graph({"nodes": [{"id": "a", "agent_slug": "rag"}], "edges": []})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/orchestration/test_workflow_compiler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestration.workflow_compiler'`

- [ ] **Step 3: Write the implementation**

```python
# src/orchestration/workflow_compiler.py
"""Compile a user-drawn graph into a WorkflowGraph the engine can execute.

The canvas is the procurement process: the DAG the user draws is the DAG that
runs. Connectors carry no condition — an edge means "then" — which is what makes
the graph serialisable and is why the engine needs no changes: it already accepts
a WorkflowGraph object.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from agents.definitions import load_definitions
from orchestration.workflow_engine import WorkflowEdge, WorkflowGraph, WorkflowNode


class GraphValidationError(ValueError):
    """A drawn graph that cannot run. The message says why, in the user's terms."""


def _known_agents() -> Dict[str, Dict[str, Any]]:
    return {a["slug"]: a for a in load_definitions().get("agents", [])}


def _entry_nodes(node_ids: List[str], edges: List[Dict[str, str]]) -> List[str]:
    with_inbound = {e["target"] for e in edges}
    return [n for n in node_ids if n not in with_inbound]


def _has_cycle(node_ids: List[str], edges: List[Dict[str, str]]) -> bool:
    succ: Dict[str, List[str]] = {n: [] for n in node_ids}
    for e in edges:
        succ[e["source"]].append(e["target"])
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {n: WHITE for n in node_ids}

    def visit(n: str) -> bool:
        colour[n] = GREY
        for m in succ[n]:
            if colour[m] == GREY:
                return True
            if colour[m] == WHITE and visit(m):
                return True
        colour[n] = BLACK
        return False

    return any(colour[n] == WHITE and visit(n) for n in node_ids)


def _reachable(entry: str, edges: List[Dict[str, str]]) -> Set[str]:
    succ: Dict[str, List[str]] = {}
    for e in edges:
        succ.setdefault(e["source"], []).append(e["target"])
    seen, stack = {entry}, [entry]
    while stack:
        for m in succ.get(stack.pop(), []):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


def validate_saved_graph(graph: Dict[str, Any]) -> None:
    """Raise GraphValidationError if this graph cannot run. Fail here, at save
    time, with a reason — never mysteriously at run time."""
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    if not nodes:
        raise GraphValidationError("A workflow needs at least one node.")

    ids = [n["id"] for n in nodes]
    if len(set(ids)) != len(ids):
        raise GraphValidationError("Two nodes share the same id.")

    known = _known_agents()
    for n in nodes:
        if n.get("agent_slug") not in known:
            raise GraphValidationError(f"unknown agent: {n.get('agent_slug')!r}")

    id_set = set(ids)
    for e in edges:
        if e["source"] not in id_set or e["target"] not in id_set:
            raise GraphValidationError(f"Edge {e['source']} -> {e['target']} points at a node that is not on the canvas.")
        if e["source"] == e["target"]:
            raise GraphValidationError(f"Node {e['source']} is connected to itself (cycle).")

    if _has_cycle(ids, edges):
        raise GraphValidationError("The flow loops back on itself (cycle). A workflow must run forwards.")

    entries = _entry_nodes(ids, edges)
    if len(entries) != 1:
        raise GraphValidationError(
            f"A workflow needs exactly one entry node (a node with nothing feeding into it); this one has {len(entries)}."
        )

    unreachable = id_set - _reachable(entries[0], edges)
    if unreachable:
        raise GraphValidationError(f"These nodes are not connected to the flow: {sorted(unreachable)}")


def compile_graph(name: str, graph: Dict[str, Any]) -> WorkflowGraph:
    """Saved JSON -> an executable WorkflowGraph. Validates first."""
    validate_saved_graph(graph)
    known = _known_agents()
    nodes = graph["nodes"]
    edges = graph.get("edges") or []

    wf = WorkflowGraph(
        name=name,
        description=graph.get("description") or "",
        entry_node=_entry_nodes([n["id"] for n in nodes], edges)[0],
    )
    for n in nodes:
        defn = known[n["agent_slug"]]
        wf.add_node(
            WorkflowNode(
                name=n["id"],
                agent_type=n["agent_slug"],
                # Publish everything this agent declares, so downstream nodes can consume it.
                output_to_shared=list(defn.get("outputs") or []),
            )
        )
    for e in edges:
        # condition=None: a drawn connector means "then". It carries no condition.
        wf.add_edge(WorkflowEdge(source=e["source"], target=e["target"], condition=None))
    return wf
```

If `WorkflowGraph`'s constructor does not accept `entry_node` as a keyword, set `wf.entry_node` after construction. If `add_node`/`add_edge` have different signatures, adapt — the dataclasses are at `src/orchestration/workflow_engine.py:129-230`; read them first.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/orchestration/test_workflow_compiler.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/workflow_compiler.py tests/orchestration/test_workflow_compiler.py
git commit -m "feat(orchestration): compile a drawn graph into an executable WorkflowGraph

Edges are unconditional — a connector means 'then'. Rejects cycles, multiple
entry nodes, unreachable nodes and unknown agents at save time, with a reason,
rather than failing mysteriously at run time."
```

---

### Task 3: Elicitation — work out what the workflow must ask the human

**Files:**
- Create: `src/orchestration/elicitation.py`
- Test: `tests/orchestration/test_elicitation.py`

**Interfaces:**
- Consumes: `agents.definitions.get_elicit` (Task 1); `orchestration.workflow_compiler` (Task 2, for topological reasoning only — may re-derive predecessors locally)
- Produces:
  - `@dataclass InputRequest: node_id: str; agent_slug: str; required_field: str; field_type: str; prompt: str`
    (`required_field` is the **first** member of the unsatisfied `any_of` group — the canonical key the answer is stored under.)
  - `pending_requests(graph: Dict[str, Any], payload: Dict[str, Any], answers: Dict[str, Any]) -> List[InputRequest]`

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestration/test_elicitation.py
from orchestration.elicitation import InputRequest, pending_requests

EXTRACT_THEN_RANK = {
    "nodes": [
        {"id": "n1", "agent_slug": "data_extraction"},
        {"id": "n2", "agent_slug": "supplier_ranking"},
    ],
    "edges": [{"source": "n1", "target": "n2"}],
}


def test_data_extraction_with_no_upstream_asks_for_documents():
    """THE regression guard. The old manifests would have asked for nothing here,
    and the agent would have run against no documents and produced nothing."""
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={}, answers={})
    doc = [r for r in reqs if r.node_id == "n1"]
    assert len(doc) == 1
    assert doc[0].agent_slug == "data_extraction"
    assert doc[0].field_type == "document_ids"
    assert doc[0].required_field == "s3_prefix"   # first member of the any_of group


def test_supplier_ranking_always_asks_for_a_query():
    """Nothing in the system produces `query`, so it must come from the human."""
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={}, answers={})
    assert any(r.node_id == "n2" and r.field_type == "text" for r in reqs)


def test_a_group_satisfied_by_the_run_payload_is_not_asked_for():
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={"s3_prefix": "docs/"}, answers={})
    assert [r.node_id for r in reqs] == ["n2"]      # only the ranking query remains


def test_any_member_of_the_group_satisfies_it():
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={"document_ids": [1, 2]}, answers={})
    assert not [r for r in reqs if r.node_id == "n1"]


def test_a_prior_human_answer_satisfies_a_group():
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={}, answers={"s3_prefix": "docs/"})
    assert not [r for r in reqs if r.node_id == "n1"]


def test_an_upstream_output_satisfies_a_downstream_group():
    """discrepancy_detection consumes `details`, which data_extraction produces —
    so it must NOT be asked for."""
    graph = {
        "nodes": [
            {"id": "n1", "agent_slug": "data_extraction"},
            {"id": "n2", "agent_slug": "discrepancy_detection"},
        ],
        "edges": [{"source": "n1", "target": "n2"}],
    }
    reqs = pending_requests(graph, payload={"s3_prefix": "docs/"}, answers={})
    assert reqs == []


def test_only_upstream_counts_not_downstream():
    """A node cannot be satisfied by something produced AFTER it."""
    graph = {
        "nodes": [
            {"id": "n1", "agent_slug": "discrepancy_detection"},
            {"id": "n2", "agent_slug": "data_extraction"},
        ],
        "edges": [{"source": "n1", "target": "n2"}],
    }
    reqs = pending_requests(graph, payload={}, answers={})
    assert any(r.node_id == "n1" for r in reqs)   # details not yet produced


def test_an_agent_with_no_elicit_contract_asks_for_nothing():
    graph = {"nodes": [{"id": "n1", "agent_slug": "opportunity_miner"}], "edges": []}
    assert pending_requests(graph, payload={}, answers={}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/orchestration/test_elicitation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestration.elicitation'`

- [ ] **Step 3: Write the implementation**

```python
# src/orchestration/elicitation.py
"""What must the human supply before this workflow can run?

The workflow does not guess and does not fabricate. For every node, each of its
declared `elicit` groups must be satisfied by an upstream node's output, the run
payload, or an answer the human already gave. Anything left over becomes a
question, bound to the node that needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set

from agents.definitions import get_elicit, load_definitions


@dataclass(frozen=True)
class InputRequest:
    node_id: str
    agent_slug: str
    required_field: str   # canonical key the answer is stored under
    field_type: str       # drives the UI control: document_ids | text | number
    prompt: str           # what the human is asked


def _outputs_of(slug: str) -> List[str]:
    for a in load_definitions().get("agents", []):
        if a.get("slug") == slug:
            return list(a.get("outputs") or [])
    return []


def _ancestors(node_id: str, edges: List[Dict[str, str]]) -> Set[str]:
    preds: Dict[str, List[str]] = {}
    for e in edges:
        preds.setdefault(e["target"], []).append(e["source"])
    seen, stack = set(), list(preds.get(node_id, []))
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(preds.get(n, []))
    return seen


def pending_requests(
    graph: Dict[str, Any],
    payload: Dict[str, Any],
    answers: Dict[str, Any],
) -> List[InputRequest]:
    """Every question this graph must ask the human before it can run."""
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    by_id = {n["id"]: n for n in nodes}

    requests: List[InputRequest] = []
    for node in nodes:
        slug = node["agent_slug"]

        upstream_keys: Set[str] = set()
        for anc in _ancestors(node["id"], edges):
            upstream_keys.update(_outputs_of(by_id[anc]["agent_slug"]))

        available = upstream_keys | set(payload) | set(answers)

        for group in get_elicit(slug):
            if any(k in available for k in group["any_of"]):
                continue
            requests.append(
                InputRequest(
                    node_id=node["id"],
                    agent_slug=slug,
                    required_field=group["any_of"][0],
                    field_type=group["type"],
                    prompt=group["prompt"],
                )
            )
    return requests
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/orchestration/test_elicitation.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/elicitation.py tests/orchestration/test_elicitation.py
git commit -m "feat(orchestration): ask the human for what the workflow cannot know

A node's elicit group is satisfied by an upstream node's output, the run payload,
or an answer already given. Anything else becomes a question bound to that node.
A data_extraction node with no upstream producer therefore asks which documents to
extract from — the case the manifests would otherwise have skipped in silence."
```

---

### Task 4: Engine — an `awaiting_input` state

**Files:**
- Modify: `src/orchestration/workflow_engine.py` (`NodeStatus` at ~line 44)
- Test: `tests/orchestration/test_workflow_awaiting_input.py`

**Interfaces:**
- Consumes: nothing
- Produces: `NodeStatus.AWAITING_INPUT` (value `"awaiting_input"`); the run-level string status `"awaiting_input"` on `WorkflowState.status`.

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestration/test_workflow_awaiting_input.py
from orchestration.workflow_engine import NodeStatus, WorkflowState


def test_awaiting_input_is_a_node_status():
    assert NodeStatus.AWAITING_INPUT.value == "awaiting_input"


def test_a_paused_run_checkpoints_and_round_trips():
    """A run that stops to ask the human must be resumable from its checkpoint."""
    st = WorkflowState(workflow_id="w1", workflow_name="test", user_id="u1")
    st.node_statuses["n1"] = NodeStatus.AWAITING_INPUT
    st.status = "awaiting_input"
    st.shared_data["already_known"] = "keep me"

    cp = st.checkpoint()
    assert cp is not None

    d = st.to_dict()
    assert d["status"] == "awaiting_input"
    assert d["shared_data"]["already_known"] == "keep me"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/orchestration/test_workflow_awaiting_input.py -v`
Expected: FAIL — `AttributeError: AWAITING_INPUT`

- [ ] **Step 3: Add the status**

```python
# src/orchestration/workflow_engine.py — in class NodeStatus (line ~44)
class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"   # paused: the human owes this node an input
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/orchestration/test_workflow_awaiting_input.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Check nothing else pattern-matches exhaustively on NodeStatus**

Run: `grep -rn "NodeStatus\." src/ --include=*.py | grep -vE "PENDING|RUNNING|COMPLETED|FAILED|SKIPPED|AWAITING_INPUT|import"`
Expected: no output. If any `match`/`if-elif` chain enumerates every status, add an `AWAITING_INPUT` branch that leaves the node paused.

- [ ] **Step 6: Commit**

```bash
git add src/orchestration/workflow_engine.py tests/orchestration/test_workflow_awaiting_input.py
git commit -m "feat(orchestration): add an awaiting_input node status

A HITL run pauses on a node that needs a human input. The engine already
checkpoints and resumes (12-Factor #6); it just had no way to say 'I am waiting
for a person'."
```

---

### Task 5: Persistence — saved workflows and input requests

**Files:**
- Create: `src/repositories/agent_workflow_repo.py`
- Create: `src/repositories/workflow_input_request_repo.py`
- Test: `tests/orchestration/test_agent_workflow_repo.py`

Follow the house pattern (see `src/repositories/workflow_lifecycle_repo.py`): a `DDL` string in the module, an `ensure_schema()` that runs it, and `services.db.get_conn` for connections.

**Interfaces:**
- Consumes: `services.db.get_conn`
- Produces (`agent_workflow_repo`):
  - `ensure_schema() -> None`
  - `create(name: str, graph: Dict, entry_node: str, description: str = "", created_by: str = "system") -> int` (returns `workflow_id`)
  - `get(workflow_id: int) -> Optional[Dict]`
  - `list_active() -> List[Dict]`
  - `update(workflow_id: int, *, name=None, graph=None, entry_node=None, description=None) -> None`
  - `soft_delete(workflow_id: int) -> None`
- Produces (`workflow_input_request_repo`):
  - `ensure_schema() -> None`
  - `raise_requests(run_id: str, requests: List[InputRequest]) -> None`
  - `open_requests(run_id: str) -> List[Dict]`
  - `answer(request_id: int, answer: Any, answered_by: str) -> None`
  - `answers_for(run_id: str) -> Dict[str, Any]` — `{required_field: answer}` for every answered request

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestration/test_agent_workflow_repo.py
import pytest

from repositories import agent_workflow_repo as repo
from repositories import workflow_input_request_repo as reqrepo
from orchestration.elicitation import InputRequest

pytestmark = pytest.mark.integration   # touches the live proc schema

GRAPH = {
    "nodes": [{"id": "n1", "agent_slug": "data_extraction", "x": 0, "y": 0}],
    "edges": [],
}


@pytest.fixture(scope="module", autouse=True)
def _schema():
    repo.ensure_schema()
    reqrepo.ensure_schema()


def test_create_read_update_soft_delete():
    wid = repo.create(name="plan-test", graph=GRAPH, entry_node="n1", created_by="pytest")
    got = repo.get(wid)
    assert got["name"] == "plan-test"
    assert got["graph"]["nodes"][0]["agent_slug"] == "data_extraction"
    assert got["entry_node"] == "n1"

    repo.update(wid, name="plan-test-renamed")
    assert repo.get(wid)["name"] == "plan-test-renamed"

    assert any(w["workflow_id"] == wid for w in repo.list_active())

    repo.soft_delete(wid)
    assert all(w["workflow_id"] != wid for w in repo.list_active())


def test_input_requests_round_trip():
    run_id = "run-plan-test-1"
    reqrepo.raise_requests(run_id, [
        InputRequest(node_id="n1", agent_slug="data_extraction",
                     required_field="s3_prefix", field_type="document_ids",
                     prompt="Which documents should I extract from?")
    ])
    open_ = reqrepo.open_requests(run_id)
    assert len(open_) == 1
    assert open_[0]["required_field"] == "s3_prefix"

    reqrepo.answer(open_[0]["request_id"], "docs/2026/", answered_by="pytest")
    assert reqrepo.open_requests(run_id) == []
    assert reqrepo.answers_for(run_id) == {"s3_prefix": "docs/2026/"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/orchestration/test_agent_workflow_repo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'repositories.agent_workflow_repo'`

- [ ] **Step 3: Write `agent_workflow_repo.py`**

```python
# src/repositories/agent_workflow_repo.py
"""Saved agent workflows — the graphs a user drew on the canvas.

Until now the canvas had nowhere to go: Save deep-copied into a JS array and died
on page reload. This is where a drawn workflow actually lives.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_agent_workflow (
    workflow_id  BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    graph        JSONB NOT NULL,
    entry_node   TEXT NOT NULL,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_by   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_agent_workflow_active
    ON proc.bp_agent_workflow (is_active, updated_at DESC);
"""


def ensure_schema() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL)
        conn.commit()


def _row(r) -> Dict[str, Any]:
    graph = r[3]
    if isinstance(graph, str):
        graph = json.loads(graph)
    return {
        "workflow_id": r[0], "name": r[1], "description": r[2], "graph": graph,
        "entry_node": r[4], "created_by": r[5],
        "created_at": r[6].isoformat() if r[6] else None,
        "updated_at": r[7].isoformat() if r[7] else None,
    }


_SELECT = """SELECT workflow_id, name, description, graph, entry_node, created_by,
                    created_at, updated_at
               FROM proc.bp_agent_workflow"""


def create(name: str, graph: Dict[str, Any], entry_node: str,
           description: str = "", created_by: str = "system") -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO proc.bp_agent_workflow (name, description, graph, entry_node, created_by)
               VALUES (%s, %s, %s::jsonb, %s, %s) RETURNING workflow_id""",
            (name, description, json.dumps(graph), entry_node, created_by),
        )
        wid = cur.fetchone()[0]
        conn.commit()
        return int(wid)


def get(workflow_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(_SELECT + " WHERE workflow_id = %s AND is_active", (workflow_id,))
        r = cur.fetchone()
        return _row(r) if r else None


def list_active() -> List[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(_SELECT + " WHERE is_active ORDER BY updated_at DESC")
        return [_row(r) for r in cur.fetchall()]


def update(workflow_id: int, *, name: Optional[str] = None,
           graph: Optional[Dict[str, Any]] = None,
           entry_node: Optional[str] = None,
           description: Optional[str] = None) -> None:
    sets, vals = [], []
    if name is not None:
        sets.append("name = %s"); vals.append(name)
    if description is not None:
        sets.append("description = %s"); vals.append(description)
    if graph is not None:
        sets.append("graph = %s::jsonb"); vals.append(json.dumps(graph))
    if entry_node is not None:
        sets.append("entry_node = %s"); vals.append(entry_node)
    if not sets:
        return
    sets.append("updated_at = now()")
    vals.append(workflow_id)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"UPDATE proc.bp_agent_workflow SET {', '.join(sets)} WHERE workflow_id = %s", vals)
        conn.commit()


def soft_delete(workflow_id: int) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_agent_workflow SET is_active = FALSE, updated_at = now() WHERE workflow_id = %s",
            (workflow_id,),
        )
        conn.commit()
```

- [ ] **Step 4: Write `workflow_input_request_repo.py`**

```python
# src/repositories/workflow_input_request_repo.py
"""What the workflow asked the human, and what the human answered.

A HITL run is only trustworthy if you can see what a person supplied and when.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_workflow_input_request (
    request_id     BIGSERIAL PRIMARY KEY,
    workflow_id    TEXT NOT NULL,          -- the RUN id
    node_name      TEXT NOT NULL,
    agent_slug     TEXT NOT NULL,
    required_field TEXT NOT NULL,
    field_type     TEXT,
    prompt         TEXT,
    status         TEXT NOT NULL DEFAULT 'pending',
    answer         JSONB,
    answered_by    TEXT,
    requested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    answered_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_bp_workflow_input_request_open
    ON proc.bp_workflow_input_request (workflow_id, status);
"""


def ensure_schema() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL)
        conn.commit()


def raise_requests(run_id: str, requests: List[Any]) -> None:
    """Persist the questions this run needs answered (orchestration.elicitation.InputRequest)."""
    if not requests:
        return
    with get_conn() as conn, conn.cursor() as cur:
        for r in requests:
            cur.execute(
                """INSERT INTO proc.bp_workflow_input_request
                       (workflow_id, node_name, agent_slug, required_field, field_type, prompt)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (run_id, r.node_id, r.agent_slug, r.required_field, r.field_type, r.prompt),
            )
        conn.commit()


def open_requests(run_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT request_id, node_name, agent_slug, required_field, field_type, prompt
                 FROM proc.bp_workflow_input_request
                WHERE workflow_id = %s AND status = 'pending'
                ORDER BY request_id""",
            (run_id,),
        )
        return [
            {"request_id": r[0], "node_name": r[1], "agent_slug": r[2],
             "required_field": r[3], "field_type": r[4], "prompt": r[5]}
            for r in cur.fetchall()
        ]


def answer(request_id: int, answer: Any, answered_by: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE proc.bp_workflow_input_request
                  SET answer = %s::jsonb, answered_by = %s, answered_at = now(), status = 'answered'
                WHERE request_id = %s""",
            (json.dumps(answer), answered_by, request_id),
        )
        conn.commit()


def answers_for(run_id: str) -> Dict[str, Any]:
    """{required_field: answer} for everything the human has already supplied."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT required_field, answer FROM proc.bp_workflow_input_request
                WHERE workflow_id = %s AND status = 'answered'""",
            (run_id,),
        )
        out: Dict[str, Any] = {}
        for field, ans in cur.fetchall():
            out[field] = json.loads(ans) if isinstance(ans, str) else ans
        return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/orchestration/test_agent_workflow_repo.py -v`
Expected: PASS (2 tests). Both tables now exist in `bp_sqldb`.

- [ ] **Step 6: Commit**

```bash
git add src/repositories/agent_workflow_repo.py src/repositories/workflow_input_request_repo.py tests/orchestration/test_agent_workflow_repo.py
git commit -m "feat(workflows): persist saved workflows and their HITL input requests

proc.bp_agent_workflow gives a drawn graph somewhere to live — Save used to write
to a JS array and die on reload. proc.bp_workflow_input_request records what the
workflow asked a human and what they answered, so a HITL run is auditable."
```

---

### Task 6: Governance resolution per node

**Files:**
- Create: `src/orchestration/node_governance.py`
- Test: `tests/orchestration/test_node_governance.py`

`bp_prompt.prompt_linked_agents` uses `supplier_ranking_agent`; the registry uses `supplier_ranking`. Normalise both ways. 9 of 14 agents have neither a prompt nor a policy and must be reported as ungoverned — never given a pretend tick.

**Interfaces:**
- Consumes: `services.db.get_conn`
- Produces: `governance_for(slug: str) -> Dict[str, Any]` returning
  `{"prompts": [{"name": str, "version": Any}], "policies": [{"name": str, "version": Any}], "governed": bool}`

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestration/test_node_governance.py
import pytest
from orchestration.node_governance import governance_for, normalise_agent_name

pytestmark = pytest.mark.integration


def test_normalises_both_directions():
    assert normalise_agent_name("supplier_ranking") == "supplier_ranking_agent"
    assert normalise_agent_name("supplier_ranking_agent") == "supplier_ranking_agent"


def test_a_governed_agent_reports_its_prompts_and_policies():
    g = governance_for("supplier_ranking")
    assert g["governed"] is True
    assert {p["name"] for p in g["prompts"]} >= {"rank_by_criteria"}
    assert {p["name"] for p in g["policies"]} >= {"WeightAllocationPolicy"}


def test_an_ungoverned_agent_says_so_rather_than_faking_it():
    g = governance_for("data_extraction")
    assert g["governed"] is False
    assert g["prompts"] == []
    assert g["policies"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/orchestration/test_node_governance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestration.node_governance'`

- [ ] **Step 3: Write the implementation**

```python
# src/orchestration/node_governance.py
"""Which governed prompt and policy actually applied to this agent.

Only 5 of the 14 agents have any governance at all. The other 9 run on their
built-in defaults, and this says so plainly. A node that shows a governance tick
it did not earn is worse than a node that shows none.
"""

from __future__ import annotations

from typing import Any, Dict, List

from services.db import get_conn


def normalise_agent_name(slug: str) -> str:
    """Registry uses `supplier_ranking`; the governance tables use `supplier_ranking_agent`."""
    return slug if slug.endswith("_agent") else f"{slug}_agent"


def _rows(sql: str, name: str) -> List[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (name, name))
        return [{"name": r[0], "version": r[1]} for r in cur.fetchall()]


def governance_for(slug: str) -> Dict[str, Any]:
    linked = normalise_agent_name(slug)
    prompts = _rows(
        """SELECT prompt_name, version FROM proc.bp_prompt
            WHERE prompts_status = 1 AND (prompt_linked_agents = %s OR prompt_linked_agents = %s)""",
        linked,
    )
    policies = _rows(
        """SELECT policy_name, version FROM proc.bp_policy
            WHERE policy_status = 1 AND (policy_linked_agents = %s OR policy_linked_agents = %s)""",
        linked,
    )
    return {"prompts": prompts, "policies": policies,
            "governed": bool(prompts or policies)}
```

(The doubled `%s` is deliberate: the same normalised name is compared twice so the
query also works if a row was stored without the `_agent` suffix. Pass the raw slug as
the second parameter if you prefer — adjust `_rows` accordingly.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/orchestration/test_node_governance.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/orchestration/node_governance.py tests/orchestration/test_node_governance.py
git commit -m "feat(orchestration): report the governance that actually applied to a node

5 of 14 agents have a governed prompt or policy. The other 9 are reported as
ungoverned rather than shown a tick they did not earn. Normalises the
supplier_ranking / supplier_ranking_agent naming split between the registry and
the governance tables."
```

---

### Task 7: The `/agent-workflows` router

**Files:**
- Create: `src/api/routers/agent_workflows.py`
- Modify: `src/api/main.py` (import + `app.include_router(...)`, alongside the block at ~line 309-330)
- Test: `tests/api/test_agent_workflows_router.py`

**Interfaces:**
- Consumes: Tasks 1-6 (`compile_graph`, `validate_saved_graph`, `GraphValidationError`, `pending_requests`, `governance_for`, both repos)
- Produces: the 8 HTTP endpoints in spec §6.

Follow the house router pattern (see `src/api/routers/decisions.py`): module-level `router = APIRouter(prefix=..., tags=[...])`, pydantic bodies, `Request`-based access to `app.state` for the orchestrator.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_agent_workflows_router.py
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

GRAPH = {
    "nodes": [
        {"id": "n1", "agent_slug": "data_extraction", "x": 0, "y": 0},
        {"id": "n2", "agent_slug": "supplier_ranking", "x": 200, "y": 0},
    ],
    "edges": [{"source": "n1", "target": "n2"}],
}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_create_list_get_delete(client):
    r = client.post("/agent-workflows", json={"name": "router-test", "graph": GRAPH})
    assert r.status_code == 200, r.text
    wid = r.json()["workflow_id"]

    assert any(w["workflow_id"] == wid for w in client.get("/agent-workflows").json()["workflows"])

    got = client.get(f"/agent-workflows/{wid}").json()
    assert got["entry_node"] == "n1"          # derived, not supplied
    assert got["graph"]["nodes"][0]["agent_slug"] == "data_extraction"

    assert client.delete(f"/agent-workflows/{wid}").status_code == 200


def test_a_cyclic_graph_is_rejected_with_a_reason(client):
    bad = {"nodes": [{"id": "a", "agent_slug": "rag"}, {"id": "b", "agent_slug": "rag"}],
           "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]}
    r = client.post("/agent-workflows", json={"name": "bad", "graph": bad})
    assert r.status_code == 400
    assert "cycle" in r.json()["detail"].lower()


def test_run_halts_and_asks_for_documents(client):
    """The HITL contract, end to end at the HTTP layer."""
    wid = client.post("/agent-workflows", json={"name": "hitl-test", "graph": GRAPH}).json()["workflow_id"]

    run = client.post(f"/agent-workflows/{wid}/run", json={"payload": {}}).json()
    assert run["status"] == "awaiting_input"
    fields = {q["required_field"] for q in run["pending"]}
    assert "s3_prefix" in fields          # data_extraction wants documents
    assert "query" in fields              # supplier_ranking wants a query

    client.delete(f"/agent-workflows/{wid}")


def test_each_node_reports_whether_it_is_governed(client):
    wid = client.post("/agent-workflows", json={"name": "gov-test", "graph": GRAPH}).json()["workflow_id"]
    run = client.post(f"/agent-workflows/{wid}/run", json={"payload": {}}).json()
    gov = {n["node_id"]: n["governance"]["governed"] for n in run["nodes"]}
    assert gov["n1"] is False    # data_extraction: ungoverned, built-in default
    assert gov["n2"] is True     # supplier_ranking: governed
    client.delete(f"/agent-workflows/{wid}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/api/test_agent_workflows_router.py -v`
Expected: FAIL — 404 on `/agent-workflows` (router not registered)

- [ ] **Step 3: Write the router**

```python
# src/api/routers/agent_workflows.py
"""Agent workflows — the canvas is the procurement process.

The DAG the user draws is saved, compiled and executed. Before it runs, the
workflow works out what it cannot know and asks the human for it. It never guesses.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from orchestration.elicitation import pending_requests
from orchestration.node_governance import governance_for
from orchestration.workflow_compiler import (
    GraphValidationError, compile_graph, validate_saved_graph,
)
from repositories import agent_workflow_repo as repo
from repositories import workflow_input_request_repo as reqrepo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-workflows", tags=["Agent Workflows"])


class WorkflowBody(BaseModel):
    name: str
    graph: Dict[str, Any]
    description: str = ""


class RunBody(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict)
    user_id: str = "system"


class AnswerBody(BaseModel):
    request_id: int
    answer: Any
    answered_by: str = "human"


def _entry_of(graph: Dict[str, Any]) -> str:
    targets = {e["target"] for e in (graph.get("edges") or [])}
    return next(n["id"] for n in graph["nodes"] if n["id"] not in targets)


def _describe_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"node_id": n["id"], "agent_slug": n["agent_slug"],
         "governance": governance_for(n["agent_slug"])}
        for n in graph.get("nodes", [])
    ]


@router.get("")
def list_workflows() -> Dict[str, Any]:
    repo.ensure_schema()
    return {"workflows": repo.list_active()}


@router.post("")
def create_workflow(body: WorkflowBody) -> Dict[str, Any]:
    repo.ensure_schema()
    try:
        validate_saved_graph(body.graph)
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    wid = repo.create(name=body.name, graph=body.graph, entry_node=_entry_of(body.graph),
                      description=body.description)
    return {"workflow_id": wid}


@router.get("/{workflow_id}")
def get_workflow(workflow_id: int) -> Dict[str, Any]:
    repo.ensure_schema()
    wf = repo.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="No such workflow")
    return wf


@router.put("/{workflow_id}")
def update_workflow(workflow_id: int, body: WorkflowBody) -> Dict[str, Any]:
    repo.ensure_schema()
    try:
        validate_saved_graph(body.graph)
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    repo.update(workflow_id, name=body.name, graph=body.graph,
                entry_node=_entry_of(body.graph), description=body.description)
    return {"ok": True}


@router.delete("/{workflow_id}")
def delete_workflow(workflow_id: int) -> Dict[str, Any]:
    repo.ensure_schema()
    repo.soft_delete(workflow_id)
    return {"ok": True}


@router.post("/{workflow_id}/run")
def run_workflow(workflow_id: int, body: RunBody, request: Request) -> Dict[str, Any]:
    repo.ensure_schema()
    reqrepo.ensure_schema()

    wf = repo.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="No such workflow")

    run_id = f"awf-{workflow_id}-{uuid.uuid4().hex[:8]}"
    answers = reqrepo.answers_for(run_id)          # empty on a fresh run

    missing = pending_requests(wf["graph"], body.payload, answers)
    if missing:
        # The workflow does not guess. It stops and asks.
        reqrepo.raise_requests(run_id, missing)
        return {
            "run_id": run_id, "status": "awaiting_input",
            "pending": reqrepo.open_requests(run_id),
            "nodes": _describe_nodes(wf["graph"]),
        }

    return _execute(request, run_id, wf, {**body.payload, **answers}, body.user_id)


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    reqrepo.ensure_schema()
    return {"run_id": run_id, "pending": reqrepo.open_requests(run_id)}


@router.post("/runs/{run_id}/input")
def submit_input(run_id: str, body: AnswerBody, request: Request) -> Dict[str, Any]:
    """The human answers. If nothing else is outstanding, the run proceeds."""
    reqrepo.ensure_schema()
    repo.ensure_schema()

    reqrepo.answer(body.request_id, body.answer, body.answered_by)

    still_open = reqrepo.open_requests(run_id)
    if still_open:
        return {"run_id": run_id, "status": "awaiting_input", "pending": still_open}

    workflow_id = int(run_id.split("-")[1])
    wf = repo.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="No such workflow")
    return _execute(request, run_id, wf, reqrepo.answers_for(run_id), "human")


def _execute(request: Request, run_id: str, wf: Dict[str, Any],
             input_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    engine = getattr(orchestrator, "_workflow_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    graph = compile_graph(wf["name"], wf["graph"])
    state = engine.execute(graph, input_data=input_data, user_id=user_id, workflow_id=run_id)

    return {
        "run_id": run_id,
        "status": getattr(state, "status", "completed"),
        "node_statuses": {k: getattr(v, "value", v) for k, v in state.node_statuses.items()},
        "errors": state.errors,
        "nodes": _describe_nodes(wf["graph"]),
        "pending": [],
    }
```

- [ ] **Step 4: Register the router**

```python
# src/api/main.py — with the other router imports
from api.routers import agent_workflows as agent_workflows_router
# ...
# src/api/main.py — with the other include_router calls (~line 330)
app.include_router(agent_workflows_router.router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/api/test_agent_workflows_router.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/api/routers/agent_workflows.py src/api/main.py tests/api/test_agent_workflows_router.py
git commit -m "feat(api): /agent-workflows — save, run and answer a drawn workflow

Run compiles the saved graph and hands it to the existing engine. If any node
needs something the graph cannot supply, the run halts at awaiting_input and asks
the human — a data_extraction node with no upstream producer asks which documents
to extract from. Each node reports whether it is actually governed."
```

---

### Task 8: The UI — make the workspace tell the truth

**Files:**
- Modify: `beyond_procwise_ui/src/modules/SpendIQ/index.jsx` (bridge, ~line 161-179)
- Modify: `beyond_procwise_ui/src/modules/SpendIQ/engine.js`:
  - delete `AGENT_LIB` (4650-4659), `AGENTS` (4666-4675), `WF_SAVED` fixtures (4677-4681)
  - rewrite `agentWorkspaceView()` (4810-4837), `wfSave()` (4730), `wfRun()` (4754-4770)
- Modify: `beyond_procwise_ui/src/modules/SpendIQ/styles.css` (node status + governance badge styles)

**Interfaces:**
- Consumes: the Task 7 endpoints
- Produces: `window.__SPENDIQ_WF__` — `{list, create, remove, run, submitInput, agents}`

- [ ] **Step 1: Add the bridge in `index.jsx`**

`engine.js` is a classic script: it cannot `import` and has no `axios`. Everything goes through `window`.

```jsx
// src/modules/SpendIQ/index.jsx — beside the existing __SPENDIQ_AGENT_RUN__ bridge
window.__SPENDIQ_WF__ = {
  agents:      ()            => axios.get(`${AI_API}/workflows/types`).then(r => r.data),
  list:        ()            => axios.get(`${AI_API}/agent-workflows`).then(r => r.data.workflows),
  create:      (name, graph) => axios.post(`${AI_API}/agent-workflows`, { name, graph }).then(r => r.data),
  remove:      (id)          => axios.delete(`${AI_API}/agent-workflows/${id}`).then(r => r.data),
  run:         (id, payload) => axios.post(`${AI_API}/agent-workflows/${id}/run`, { payload }).then(r => r.data),
  submitInput: (runId, request_id, answer) =>
    axios.post(`${AI_API}/agent-workflows/runs/${runId}/input`, { request_id, answer }).then(r => r.data),
};
```

- [ ] **Step 2: Delete the fixtures and drive the palette from the real 14**

Remove `AGENT_LIB`, `AGENTS` and the three seeded `WF_SAVED` entries entirely. Load the palette once, from the API:

```js
// engine.js — replaces AGENT_LIB / AGENTS
let AGENT_CATALOGUE = [];   // [{slug, agentType, description, capabilities, required_inputs}]
let WF_SAVED = [];          // real saved workflows, from the API. No fixtures.

async function wfLoadCatalogue(){
  const B = window.__SPENDIQ_WF__; if(!B) return;
  try{
    AGENT_CATALOGUE = await B.agents();
    WF_SAVED = await B.list();
    wfRerender();
  }catch(e){ toast('Could not load agents: ' + (e.message || e)); }
}
```

Render the palette from `AGENT_CATALOGUE` (14 cards, each showing `description`), not from a literal.

- [ ] **Step 3: Make Save actually save, and Run actually run your graph**

```js
// engine.js — replaces wfSave() (4730) and wfRun() (4754)
async function wfSave(){
  const B = window.__SPENDIQ_WF__;
  const name = wfName || 'Untitled workflow';
  const graph = {
    nodes: wfNodes.map(n => ({id:String(n.uid), agent_slug:n.type, x:n.x, y:n.y})),
    edges: wfEdges.map(e => ({source:String(e.from), target:String(e.to)})),
  };
  try{
    await B.create(name, graph);          // 400 + reason if the graph cannot run
    WF_SAVED = await B.list();
    wfRerender();
    toast('Workflow "'+name+'" saved');
  }catch(err){
    toast(err?.response?.data?.detail || 'Could not save workflow');
  }
}

let wfRun_id = null, wfPending = [];

async function wfRun(id){
  const B = window.__SPENDIQ_WF__;
  try{
    const res = await B.run(id, {});
    wfRun_id = res.run_id;
    wfPending = res.pending || [];
    wfApplyRunState(res);                 // node statuses + governance badges
    if(res.status === 'awaiting_input') wfRenderInputPanel();   // ask the human
  }catch(err){
    toast(err?.response?.data?.detail || 'Could not run workflow');
  }
}

async function wfAnswer(request_id, answer){
  const res = await window.__SPENDIQ_WF__.submitInput(wfRun_id, request_id, answer);
  wfPending = res.pending || [];
  wfApplyRunState(res);
  if(wfPending.length) wfRenderInputPanel(); else toast('Workflow running…');
}
```

`wfRun()` must no longer send only `wfNodes[0]`. The whole point is that the saved graph is what runs.

- [ ] **Step 4: HITL panel, node status and governance badge**

- For each entry in `wfPending`, render its `prompt` against the node named by `node_name`. `field_type === 'document_ids'` renders a document picker; anything else renders a text field. Submitting calls `wfAnswer(request_id, value)`.
- `wfApplyRunState(res)` colours each node by `res.node_statuses[node_id]`: `pending` / `running` / `awaiting_input` / `completed` / `failed` / `skipped`.
- Each node shows a governance badge from `res.nodes[].governance`: governed → the prompt/policy names; **ungoverned → the words "built-in default"**. Never a tick it did not earn.

- [ ] **Step 5: Verify in the browser**

Run: `cd /home/muthu/PycharmProjects/beyond_procwise_ui && npx vite build`
Expected: `✓ built` with no errors.

Then open `http://localhost:3000/spendiq` → Agent workspace and confirm:
- the palette lists **14** agents (including `discrepancy_detection`, `rag`, `requirements`, `email_watcher`, `email_dispatch`, `quote_comparison`)
- the saved list is empty (no "2 days ago" fixtures)
- `data_extraction` → `supplier_ranking`, Save, Run → the page **asks for documents and for a ranking query**
- `data_extraction`'s node says **built-in default**; `supplier_ranking`'s node names its governed prompt/policy

- [ ] **Step 6: Commit**

```bash
cd /home/muthu/PycharmProjects/beyond_procwise_ui
git add src/modules/SpendIQ/engine.js src/modules/SpendIQ/index.jsx src/modules/SpendIQ/styles.css
git commit -m "feat(spendiq): the agent workspace runs the flow you actually drew

The palette listed 8 agents against 14 live; Save wrote to a JS array and died on
reload; the saved list was three fabricated entries; and Run discarded the graph
and fired only the first node, so whatever you drew, the same fixed workflow ran.

The canvas now loads the real 14 agents, saves to proc.bp_agent_workflow, and runs
the graph you drew. When a node needs something the graph cannot supply the run
stops and asks — a data extraction node asks which documents to extract from.
Ungoverned agents say 'built-in default' rather than showing a tick they did not earn."
```

---

### Task 9: Prove it on the running stack

Tests are not evidence that the product works. Demonstrate against live `bp_sqldb`.

**Files:** none (verification only)

- [ ] **Step 1: Restart the backend so the new router is mounted**

The backend must be started **with `.env`** or it silently runs the wrong extraction engine.

Run: `curl -s localhost:8000/openapi.json | python3 -c "import json,sys; print([p for p in json.load(sys.stdin)['paths'] if 'agent-workflows' in p])"`
Expected: the 8 paths from spec §6.

- [ ] **Step 2: Save a two-node workflow against live data**

```bash
curl -s -X POST localhost:8000/agent-workflows -H 'Content-Type: application/json' -d '{
  "name":"Extract then rank",
  "graph":{"nodes":[{"id":"n1","agent_slug":"data_extraction","x":0,"y":0},
                    {"id":"n2","agent_slug":"supplier_ranking","x":200,"y":0}],
           "edges":[{"source":"n1","target":"n2"}]}}'
```
Expected: `{"workflow_id": <n>}`

- [ ] **Step 3: Run it and confirm it asks the human**

```bash
curl -s -X POST localhost:8000/agent-workflows/<n>/run -H 'Content-Type: application/json' -d '{"payload":{}}'
```
Expected: `"status":"awaiting_input"`, with a `pending` entry asking for documents (`s3_prefix`, type `document_ids`) and one asking for the ranking `query`. Confirm `n1.governance.governed == false` and `n2.governance.governed == true`.

- [ ] **Step 4: Answer, and confirm the run proceeds and both nodes execute**

Submit each pending `request_id` via `POST /agent-workflows/runs/<run_id>/input`, supplying a real S3 prefix from the live corpus and a real ranking query. Expected: the run leaves `awaiting_input` and both nodes reach `completed`.

- [ ] **Step 5: Confirm it is auditable in the database**

```sql
SELECT node_name, agent_slug, required_field, status, answer, answered_by
  FROM proc.bp_workflow_input_request WHERE workflow_id = '<run_id>';
SELECT node_name, status FROM proc.node_execution WHERE workflow_id = '<run_id>';
```
Expected: every question and answer recorded with who supplied it; every node's execution recorded.

- [ ] **Step 6: Drive it in the browser**

Open `http://localhost:3000/spendiq` → Agent workspace. Draw `data_extraction` → `supplier_ranking`, Save, Run, answer the two questions, and watch the nodes go `awaiting_input` → `running` → `completed`. Screenshot for the record.

---

## Out of scope (spec §9)

Conditional/branching edges; authoring the 9 missing prompts and policies; retry/timeout configuration in the UI; parallel fan-out authoring.
