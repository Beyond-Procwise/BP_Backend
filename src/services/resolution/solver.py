"""The integer program and its solves.

One model is built per request and then re-solved with single edges forbidden to
measure how forced each chosen link was. Nothing is randomised and nothing is
seeded: the same request produces the same model, and the same model produces
the same answer.

Every speed-up here is exact. Nothing is approximated, no gap is tolerated, and
no constraint is relaxed — see DECISIONS.md.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csr_matrix

from .model import COST_SCALE, UNASSIGNMENT_PENALTY, ProblemModel, partition

SOLVER_VERSION = "resolution-1.1/scipy-highs"

# Cost units are scaled integers; results are reported in log-odds units.
PENALTY_LOG_ODDS = UNASSIGNMENT_PENALTY / COST_SCALE

# Independent re-solves run in parallel. HiGHS releases the GIL, and each solve
# reads a shared immutable model and writes only its own result, so the answer
# does not depend on how the work was scheduled.
_MAX_WORKERS = min(8, (os.cpu_count() or 1))

# Below this many re-solves the thread pool costs more than it saves.
_PARALLEL_THRESHOLD = 8


@dataclass(frozen=True)
class Solution:
    feasible: bool
    cost: float                     # objective in log-odds units
    chosen: tuple[int, ...]         # edge indices, canonical order
    unassigned: tuple[str, ...]


class Program:
    """The linear program for one independent group of edges.

    ``edge_ids`` are indices into ``pm.edges`` — the group this program owns.
    Variables are local to the group, but every index that reaches the outside
    (``Solution.chosen``) and every tie-break rank is the *global* edge index, so
    splitting a request into groups cannot change which assignment is chosen.
    """

    def __init__(self, pm: ProblemModel, edge_ids: Optional[Sequence[int]] = None):
        self.pm = pm
        self.gid: tuple[int, ...] = (
            tuple(range(len(pm.edges))) if edge_ids is None else tuple(edge_ids)
        )
        edges = [pm.edges[g] for g in self.gid]
        n = len(self.gid)

        self.sources: tuple[str, ...] = tuple(sorted({e.source_id for e in edges}))
        targets: tuple[str, ...] = tuple(sorted({e.target_id for e in edges}))
        m = len(self.sources)
        self.n, self.m = n, m

        local_by_source: dict[str, list[int]] = {s: [] for s in self.sources}
        local_by_target: dict[str, list[int]] = {t: [] for t in targets}
        for k, e in enumerate(edges):
            local_by_source[e.source_id].append(k)
            local_by_target[e.target_id].append(k)
        self._local = {g: k for k, g in enumerate(self.gid)}

        # True cost, scaled to integers and then lifted by the tie-break scale so
        # that the rank term added below can never outweigh a unit of real
        # evidence. The rank is the global edge index, so a group decides its own
        # ties exactly as the whole request would have decided them.
        scale = pm.tiebreak_scale
        cost = np.empty(n + m, dtype=float)
        for k, g in enumerate(self.gid):
            cost[k] = pm.primary_cost(g) * scale
        cost[n:] = UNASSIGNMENT_PENALTY * scale
        # Every solve, primary or margin, carries the rank term, so every
        # assignment this program returns is decided by the rule rather than by
        # HiGHS wherever the rule reaches. See DECISIONS.md for where it does not.
        for k, g in enumerate(self.gid):
            cost[k] += g
        self.ranked = cost

        rows: list[np.ndarray] = []
        lb: list[float] = []
        ub: list[float] = []

        for si, s in enumerate(self.sources):
            limit = pm.source_bounds[s]
            row = np.zeros(n + m)
            for k in local_by_source[s]:
                row[k] = 1.0
            row[n + si] = 1.0
            if limit == 1:
                # Exactly one of: a link, or the source is declared unassigned.
                # One equality row says what a coverage row plus a cardinality
                # row said separately, and halves the model for the N:1 shape
                # that dominates real invoice-to-PO work.
                rows.append(row)
                lb.append(1.0)
                ub.append(1.0)
                continue
            rows.append(row)
            lb.append(1.0)
            ub.append(np.inf)
            if limit is not None:
                capped = np.zeros(n + m)
                for k in local_by_source[s]:
                    capped[k] = 1.0
                rows.append(capped)
                lb.append(-np.inf)
                ub.append(float(limit))

        for t in targets:
            limit = pm.target_bounds[t]
            if limit is None:
                continue
            row = np.zeros(n + m)
            for k in local_by_target[t]:
                row[k] = 1.0
            rows.append(row)
            lb.append(-np.inf)
            ub.append(float(limit))

        for r in sorted({r for e in edges for r in e.consumes}):
            cap = pm.capacities.get(r)
            if cap is None:
                continue  # a resource nobody declared a capacity for is unbounded
            row = np.zeros(n + m)
            for k, e in enumerate(edges):
                if r in e.consumes:
                    row[k] = float(e.consumes[r])
            rows.append(row)
            lb.append(-np.inf)
            ub.append(float(cap.bound))

        self.constraint = (
            LinearConstraint(csr_matrix(np.vstack(rows)), np.array(lb), np.array(ub))
            if rows else None
        )

        # Only the edge variables need to be declared integral. Each source's
        # coverage row forces its unassignment variable to 1 minus an integral
        # sum, so it lands on an integer without being searched for.
        self.integrality = np.zeros(n + m)
        self.integrality[:n] = 1.0

        self._lower = np.zeros(n + m)
        self._ones = np.ones(n + m)

    def solve(self, forbidden: Optional[int] = None) -> Solution:
        """Optimum for this group, optionally with one global edge id forbidden."""
        upper = self._ones
        if forbidden is not None:
            upper = self._ones.copy()
            upper[self._local[forbidden]] = 0.0

        res = milp(
            c=self.ranked,
            constraints=self.constraint,
            integrality=self.integrality,
            bounds=(self._lower, upper),
        )
        if res.x is None or res.status != 0:
            return Solution(False, float("inf"), (), self.sources)

        x = np.round(res.x).astype(int)
        chosen = tuple(self.gid[k] for k in range(self.n) if x[k] == 1)
        unassigned = tuple(
            self.sources[j] for j in range(self.m) if x[self.n + j] == 1
        )
        cost = (
            sum(-self.pm.edges[g].log_odds for g in chosen)
            + PENALTY_LOG_ODDS * len(unassigned)
        )
        return Solution(True, cost, chosen, unassigned)


def _combine(parts: Sequence[Solution]) -> Solution:
    """One solution for the whole request, from one solution per group.

    The groups share no variable and no constraint row, so the objective is the
    plain sum and the union of the parts is optimal for the whole.
    """
    if any(not p.feasible for p in parts):
        return Solution(False, float("inf"), (), ())
    chosen = tuple(sorted(i for p in parts for i in p.chosen))
    unassigned = tuple(sorted(u for p in parts for u in p.unassigned))
    return Solution(True, sum(p.cost for p in parts), chosen, unassigned)


class Portfolio:
    """Every independent group of one request, solved together.

    A margin re-solve forbids one edge, and an edge can only perturb its own
    group — no constraint reaches across one — so only that group is solved
    again and the other groups' costs carry over untouched. On a request that
    falls into k groups this turns N+1 solves of the whole model into N+1 solves
    of one group each, and the answer is identical.
    """

    def __init__(self, pm: ProblemModel):
        self.pm = pm
        self.groups = partition(pm)
        self.programs = tuple(Program(pm, g) for g in self.groups)
        self.group_of: dict[int, int] = {}
        for gi, group in enumerate(self.groups):
            for i in group:
                self.group_of[i] = gi
        self._base: tuple[Solution, ...] = ()

    def solve(self) -> Solution:
        self._base = tuple(p.solve() for p in self.programs)
        return _combine(self._base)

    def solve_many(self, forbidden: list[int]) -> list[Solution]:
        """Independent re-solves, batched. Order of results matches the input."""
        if not forbidden:
            return []

        def one(edge_id: int) -> Solution:
            gi = self.group_of[edge_id]
            return self.programs[gi].solve(edge_id)

        if len(forbidden) < _PARALLEL_THRESHOLD or _MAX_WORKERS == 1:
            replaced = [one(i) for i in forbidden]
        else:
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
                replaced = list(pool.map(one, forbidden))

        out = []
        for edge_id, part in zip(forbidden, replaced):
            gi = self.group_of[edge_id]
            parts = list(self._base)
            parts[gi] = part
            out.append(_combine(parts))
        return out
