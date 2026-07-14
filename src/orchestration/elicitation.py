"""What must the human supply before this workflow can run?

The workflow does not guess and does not fabricate. For every node, each of its
declared `elicit` groups must be satisfied by an upstream node's output, the run
payload, or an answer the human already gave. Anything left over becomes a
question, bound to the node that needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set

from agents.definitions import get_elicit, load_agent_definitions


@dataclass(frozen=True)
class InputRequest:
    node_id: str
    agent_slug: str
    required_field: str   # canonical key the answer is stored under
    field_type: str       # drives the UI control: document_ids | text | number
    prompt: str           # what the human is asked


def _outputs_of(slug: str) -> List[str]:
    for agent in load_agent_definitions():
        if agent.get("slug") == slug:
            return list(agent.get("outputs") or [])
    return []


def _ancestors(node_id: str, edges: List[Dict[str, str]]) -> Set[str]:
    preds: Dict[str, List[str]] = {}
    for e in edges:
        preds.setdefault(e["target"], []).append(e["source"])
    seen: Set[str] = set()
    stack = list(preds.get(node_id, []))
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(preds.get(n, []))
    return seen


def _has_real_value(container: Dict[str, Any], key: str) -> bool:
    """Is ``key`` present in ``container`` AND carrying an actual, non-empty value?

    A group used to be satisfied by mere key PRESENCE — ``available = ... | set(payload)
    | set(answers)`` never looked at the values. That meant a blank input
    ({"s3_prefix": ""}) or an unanswered None ({"query": None}) satisfied the group
    exactly as well as a real answer, and the workflow executed on nothing. A key
    whose value is None/""/[]/{} does NOT satisfy — only a genuine value does.
    """
    if key not in container:
        return False
    value = container[key]
    if value is None:
        return False
    if isinstance(value, (str, list, dict, tuple, set)) and len(value) == 0:
        return False
    return True


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

        for group in get_elicit(slug):
            # An upstream node's output is presence-only here (its actual value
            # doesn't exist yet at elicitation time — it is produced when that
            # node runs), so key presence is the right test for it. payload and
            # answers, by contrast, carry real values right now, so a blank one
            # must not satisfy the group.
            satisfied = any(k in upstream_keys for k in group["any_of"]) or any(
                _has_real_value(payload, k) or _has_real_value(answers, k)
                for k in group["any_of"]
            )
            if satisfied:
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
