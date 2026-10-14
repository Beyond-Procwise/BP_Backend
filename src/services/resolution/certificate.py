"""Why a request cannot be satisfied, said in procurement terms.

"model infeasible" tells a buyer nothing. "PO-100 line 3 has capacity 1400.00;
INV-44 claims 2100.00, exceeding tolerance 20.00" tells them exactly which
document and which line to look at.

Infeasible here means *no* assignment satisfies the constraints, including the
assignment that places nothing. Losing a competition for scarce capacity is not
infeasible — the loser is reported unassigned and the request still resolves.
"""
from __future__ import annotations

from .model import ProblemModel


def _claimants(pm: ProblemModel, resource_id: str) -> list[tuple[str, float]]:
    claims: dict[str, float] = {}
    for e in pm.edges:
        if resource_id in e.consumes:
            claims[e.source_id] = claims.get(e.source_id, 0.0) + float(e.consumes[resource_id])
    return sorted(claims.items())


def _resource_line(pm: ProblemModel, resource_id: str) -> str:
    cap = pm.capacities[resource_id]
    claims = _claimants(pm, resource_id)
    total = sum(v for _, v in claims)
    who = ", ".join(name for name, _ in claims)
    return (
        f"{resource_id} has capacity {cap.capacity:.2f}; "
        f"{who} claim {total:.2f} combined, exceeding tolerance {cap.tolerance:.2f}"
    )


def _edge_is_placeable(pm: ProblemModel, i: int) -> bool:
    """Could this edge be taken at all, against an otherwise empty assignment?"""
    edge = pm.edges[i]
    for resource_id, qty in edge.consumes.items():
        cap = pm.capacities.get(resource_id)
        if cap is not None and float(qty) > cap.bound:
            return False
    if (pm.target_bounds[edge.target_id] or 1) < 1:
        return False
    if (pm.source_bounds[edge.source_id] or 1) < 1:
        return False
    return True


def certify(pm: ProblemModel) -> tuple[str, ...]:
    """The constraints that cannot be satisfied together, or () if none.

    Two structural causes, both checked before the objective is ever considered:
    a bound no assignment can meet, and a document with nowhere it could go.
    """
    reasons: list[str] = []

    for resource_id in sorted(pm.capacities):
        cap = pm.capacities[resource_id]
        if cap.bound < 0:
            reasons.append(
                f"{resource_id} has capacity {cap.capacity:.2f} and tolerance "
                f"{cap.tolerance:.2f}, leaving {cap.bound:.2f} available — no "
                f"assignment, not even an empty one, can satisfy it"
            )

    for source in pm.sources:
        edge_ids = pm.edges_by_source[source]
        if any(_edge_is_placeable(pm, i) for i in edge_ids):
            continue
        for i in edge_ids:
            edge = pm.edges[i]
            for resource_id, qty in sorted(edge.consumes.items()):
                cap = pm.capacities.get(resource_id)
                if cap is None or float(qty) <= cap.bound:
                    continue
                reasons.append(
                    f"{source} cannot be placed on {edge.target_id}: "
                    + _resource_line(pm, resource_id)
                    + f" — {source} alone claims {float(qty):.2f}"
                )
        if not any(source in r for r in reasons):
            reasons.append(
                f"{source} cannot be placed on any of its candidates "
                f"({', '.join(sorted({pm.edges[i].target_id for i in edge_ids}))}): "
                f"every one is ruled out by a cardinality limit"
            )

    return tuple(reasons)
