"""Compile a user-drawn graph into a WorkflowGraph the engine can execute.

The canvas is the procurement process: the DAG the user draws is the DAG that
runs. Connectors carry no condition — an edge means "then" — which is what makes
the graph serialisable and is why the engine needs no changes: it already accepts
a WorkflowGraph object.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from agents.definitions import load_agent_definitions
from orchestration.workflow_engine import WorkflowGraph, WorkflowNode


class GraphValidationError(ValueError):
    """A drawn graph that cannot run. The message says why, in the user's terms."""


def _known_agents() -> Dict[str, Dict[str, Any]]:
    return {a["slug"]: a for a in load_agent_definitions()}


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
            raise GraphValidationError(
                f"Edge {e['source']} -> {e['target']} points at a node that is not on the canvas."
            )
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
        wf.add_edge(e["source"], e["target"], condition=None)
    return wf
