"""The integer program and its solves.

One model is built per request and then re-solved with single edges forbidden to
measure how forced each chosen link was. Nothing is randomised and nothing is
seeded: the same request produces the same model, and the same model produces
the same answer.

Every speed-up here is exact. Nothing is approximated, no gap is tolerated, and
no constraint is relaxed — see DECISIONS.md for the one place where a tie-break
is applied locally rather than globally, and why that is still an explicit rule.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csr_matrix

from .model import COST_SCALE, UNASSIGNMENT_PENALTY, ProblemModel

SOLVER_VERSION = "resolution-1.0/scipy-highs"

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
    """The linear program for one request, built once and re-solved cheaply."""

    def __init__(self, pm: ProblemModel):
        self.pm = pm
        n = len(pm.edges)
        m = len(pm.sources)
        self.n, self.m = n, m
        self.scale = pm.tiebreak_scale

        # True cost, scaled to integers and then lifted by the tie-break scale so
        # that a rank term added below can never outweigh a unit of real evidence.
        cost = np.empty(n + m, dtype=float)
        for i in range(n):
            cost[i] = pm.primary_cost(i) * self.scale
        cost[n:] = UNASSIGNMENT_PENALTY * self.scale
        self.cost = cost

        rows: list[np.ndarray] = []
        lb: list[float] = []
        ub: list[float] = []

        for si, s in enumerate(pm.sources):
            limit = pm.source_bounds[s]
            row = np.zeros(n + m)
            for i in pm.edges_by_source[s]:
                row[i] = 1.0
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
                for i in pm.edges_by_source[s]:
                    capped[i] = 1.0
                rows.append(capped)
                lb.append(-np.inf)
                ub.append(float(limit))

        for t in pm.targets:
            limit = pm.target_bounds[t]
            if limit is None:
                continue
            row = np.zeros(n + m)
            for i in pm.edges_by_target[t]:
                row[i] = 1.0
            rows.append(row)
            lb.append(-np.inf)
            ub.append(float(limit))

        for r in pm.resources():
            cap = pm.capacities.get(r)
            if cap is None:
                continue  # a resource nobody declared a capacity for is unbounded
            row = np.zeros(n + m)
            for i, e in enumerate(pm.edges):
                if r in e.consumes:
                    row[i] = float(e.consumes[r])
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

    # -- tie-breaking ------------------------------------------------------
    def _objective(self, rank_over: Optional[str]) -> np.ndarray:
        """True cost plus a rank term worth strictly less than one unit of cost.

        ``rank_over`` restricts the rank term to one source's edges. See
        DECISIONS.md: the returned assignment is ranked globally, while a margin
        re-solve only needs the affected source's own choice pinned down.
        """
        c = self.cost.copy()
        if rank_over is None:
            for i in range(self.n):
                c[i] += i
        else:
            for i in self.pm.edges_by_source.get(rank_over, ()):
                c[i] += i
        return c

    def solve(self, forbidden: Optional[int] = None) -> Solution:
        rank_over = None if forbidden is None else self.pm.edges[forbidden].source_id
        return self._run(self._objective(rank_over), forbidden)

    def solve_many(self, forbidden: list[int]) -> list[Solution]:
        """Independent re-solves, batched. Order of results matches the input."""
        if not forbidden:
            return []
        if len(forbidden) < _PARALLEL_THRESHOLD or _MAX_WORKERS == 1:
            return [self.solve(i) for i in forbidden]
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            return list(pool.map(self.solve, forbidden))

    def _run(self, c: np.ndarray, forbidden: Optional[int]) -> Solution:
        upper = self._ones if forbidden is None else self._ones.copy()
        if forbidden is not None:
            upper[forbidden] = 0.0
        res = milp(
            c=c,
            constraints=self.constraint,
            integrality=self.integrality,
            bounds=(self._lower, upper),
        )
        if res.x is None or res.status != 0:
            return Solution(False, float("inf"), (), tuple(self.pm.sources))

        x = np.round(res.x).astype(int)
        chosen = tuple(i for i in range(self.n) if x[i] == 1)
        unassigned = tuple(
            self.pm.sources[j] for j in range(self.m) if x[self.n + j] == 1
        )
        cost = (
            sum(-self.pm.edges[i].log_odds for i in chosen)
            + PENALTY_LOG_ODDS * len(unassigned)
        )
        return Solution(True, cost, chosen, unassigned)
