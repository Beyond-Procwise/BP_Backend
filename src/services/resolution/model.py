"""Turn a ResolutionRequest into the integer program the solver runs.

Everything here is deterministic and inspectable: edges are put into canonical
order once, and every constraint carries the domain object it came from so an
infeasible solve can be explained in procurement terms rather than solver terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import CandidateEdge, CardinalityRule, ResolutionRequest

# --- Cost scaling -----------------------------------------------------------
# Edge costs must be integers so the lexicographic tie-break below is exact.
# log-odds arrive as floats; 1e4 keeps four decimal places, which is the same
# precision the scorer itself reports (linking_engine rounds L to 6dp and F to
# 4dp), and keeps the largest product well inside float64's exact-integer range.
COST_SCALE = 10_000

# --- Unassignment -----------------------------------------------------------
# Leaving a source unassigned costs this much (in scaled cost units). It has to
# beat every edge cost so that assignment is always preferred where one is
# feasible: real log-odds from the scorer sit in roughly [-12, +12], so a
# worst-case edge costs 12 * COST_SCALE = 120_000. This is an order of magnitude
# above that, and it is a fixed constant rather than a per-run tuning knob.
# It can never make a solve infeasible: the unassignment variable is always free
# to take the value 1.
UNASSIGNMENT_PENALTY = 1_000_000

# A link whose normalised margin falls below this is not determined by the
# evidence, and the whole result is reported DEGENERATE.
DEGENERACY_FLOOR = 0.01


@dataclass(frozen=True)
class Bounds:
    """Resolved cardinality limits for one node."""

    max_links: Optional[int]


def _shape_defaults(rule: CardinalityRule) -> tuple[Optional[int], Optional[int]]:
    """(max_sources_per_target, max_targets_per_source) implied by the shape,
    unless the rule states them explicitly."""
    per_target, per_source = rule.max_sources_per_target, rule.max_targets_per_source
    if rule.shape == "1:1":
        per_target = 1 if per_target is None else per_target
        per_source = 1 if per_source is None else per_source
    elif rule.shape == "N:1":
        # many sources may share one target; each source takes one target
        per_source = 1 if per_source is None else per_source
    elif rule.shape == "1:N":
        per_target = 1 if per_target is None else per_target
    return per_target, per_source


def _tighten(current: Optional[int], candidate: Optional[int]) -> Optional[int]:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return min(current, candidate)


class ProblemModel:
    """Canonical, solver-agnostic view of one resolution request."""

    def __init__(self, request: ResolutionRequest):
        self.request = request
        # Canonical order. Every downstream index — variable position, cost
        # vector, tie-break rank — derives from this and nothing else, which is
        # what makes a shuffled input produce a byte-identical model.
        self.edges: tuple[CandidateEdge, ...] = tuple(
            sorted(request.edges, key=lambda e: (e.source_id, e.target_id, e.profile_id))
        )
        self.index: dict[tuple[str, str, str], int] = {
            (e.source_id, e.target_id, e.profile_id): i for i, e in enumerate(self.edges)
        }
        self.sources: tuple[str, ...] = tuple(sorted({e.source_id for e in self.edges}))
        self.targets: tuple[str, ...] = tuple(sorted({e.target_id for e in self.edges}))
        self.rules: dict[str, CardinalityRule] = {r.profile_id: r for r in request.rules}
        self.capacities = {c.resource_id: c for c in request.capacities}

        self.edges_by_source: dict[str, list[int]] = {s: [] for s in self.sources}
        self.edges_by_target: dict[str, list[int]] = {t: [] for t in self.targets}
        for i, e in enumerate(self.edges):
            self.edges_by_source[e.source_id].append(i)
            self.edges_by_target[e.target_id].append(i)

        self.source_bounds = {s: self._bound(self.edges_by_source[s], "source")
                              for s in self.sources}
        self.target_bounds = {t: self._bound(self.edges_by_target[t], "target")
                              for t in self.targets}

        # Tie-break scale: strictly larger than the largest achievable sum of
        # ranks, so the rank term can only ever separate solutions that are
        # already equal on true cost. See DECISIONS.md.
        n = len(self.edges)
        self.tiebreak_scale = n * n + 1

    def _bound(self, edge_ids: list[int], end: str) -> Optional[int]:
        limit: Optional[int] = None
        for i in edge_ids:
            rule = self.rules.get(self.edges[i].profile_id)
            if rule is None:
                return None  # an unruled profile leaves this end unbounded
            per_target, per_source = _shape_defaults(rule)
            limit = _tighten(limit, per_target if end == "target" else per_source)
        return limit

    def primary_cost(self, i: int) -> int:
        """Integer cost of taking edge i. Minimising -log_odds maximises the
        total evidence for the assignment."""
        return int(round(-self.edges[i].log_odds * COST_SCALE))

    def resources(self) -> tuple[str, ...]:
        """Resources any edge actually draws on, in canonical order."""
        seen = {r for e in self.edges for r in e.consumes}
        return tuple(sorted(seen))


def partition(pm: "ProblemModel") -> tuple[tuple[int, ...], ...]:
    """Split the edges into groups that no constraint couples, in canonical order.

    Two edges share a group when a constraint row can hold them both: they share
    a source (every source has a coverage row), they share a target that carries
    a cardinality limit, or they draw on the same declared-capacity resource.
    A target nobody bounded and a resource nobody gave a capacity write no row,
    so they couple nothing.

    Because the objective is a plain sum over edges and no row spans two groups,
    minimising each group separately minimises the whole — and so does the
    rank tie-break, which is also a per-edge sum. The split is exact, not a
    heuristic decomposition: it is the same optimum, found in smaller pieces.
    """
    parent: dict[object, object] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, e in enumerate(pm.edges):
        union(("e", i), ("s", e.source_id))
        if pm.target_bounds[e.target_id] is not None:
            union(("e", i), ("t", e.target_id))
        for r in e.consumes:
            if r in pm.capacities:
                union(("e", i), ("r", r))

    groups: dict[object, list[int]] = {}
    for i in range(len(pm.edges)):
        groups.setdefault(find(("e", i)), []).append(i)
    # Ordered by their lowest edge index, so the groups themselves are canonical.
    return tuple(tuple(g) for g in sorted(groups.values(), key=lambda g: g[0]))
