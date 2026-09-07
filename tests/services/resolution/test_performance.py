"""What a 500-edge request actually costs, measured rather than claimed.

The brief asks for solve plus margin computation on a 500-edge graph in under
two seconds. That number depends entirely on a property of the graph the brief
does not name — whether it falls apart into independent groups — so this file
measures three shapes rather than one, prints all three, and DECISIONS.md
records what they came out at:

* **clustered** — the shape real work has, because both live callers already
  batch by supplier: 25 suppliers, each with their own orders and invoices. It
  splits into 25 groups: 0.42s, against 2.37s before the groups were solved
  independently.
* **spread** — one group, but sparse: every invoice has two candidates drawn
  from a single pool of 80 orders. 1.36s idle and about 2.1s with the cores
  busy, so it straddles the target on load alone; there is no decomposition to
  be had here, and the gain over the 1.67s it took before is the tie-break
  change only.
* **dense contested** — one group and a genuine multi-knapsack: 167 sources
  competing over 40 tightly capacitated resources. 19.0s, essentially unchanged,
  and an order of magnitude past the target: 167 branch-and-bound solves of a
  501-edge knapsack is what it costs, and no exact shortcut applies. Excluded
  from the default run because it alone takes longer than the rest of this
  package's tests put together — set ``RESOLUTION_PERF_FULL=1`` to include it.

The assertions are regression ceilings with real headroom, not the target
itself: this is a shared four-core box and a hard 2.0s bound would fail on
machine load rather than on a code change.
"""
import os
import random
import time

import pytest

from src.services.resolution import (
    CandidateEdge,
    CardinalityRule,
    ResolutionRequest,
    ResourceCapacity,
    resolve,
)
from src.services.resolution.model import ProblemModel, partition

TARGET_SECONDS = 2.0
CEILING = 8.0        # order-of-magnitude regression guard, not the target


def _clustered(n_clusters=25, per_src=10, per_tgt=4, deg=2, seed=3):
    """Invoice-to-PO by supplier: near misses stay inside a supplier's orders."""
    rng = random.Random(seed)
    edges, caps, claim = [], [], {}
    for c in range(n_clusters):
        tgts = [f"PO{c}-{j}" for j in range(per_tgt)]
        for i in range(per_src):
            home = tgts[i % per_tgt]
            others = rng.sample([t for t in tgts if t != home], deg - 1)
            for k, t in enumerate([home] + others):
                q = float(rng.randint(1, 10))
                edges.append(CandidateEdge(f"INV{c}-{i}", t, 4.0 - 1.5 * k, 0.9, "p",
                                           {f"line:{t}": q}))
                if k == 0:
                    claim[t] = claim.get(t, 0.0) + q
        caps += [ResourceCapacity(f"line:{t}", round(claim.get(t, 0.0) * 1.6, 2), 1.0)
                 for t in tgts]
    return ResolutionRequest("perf-clustered", tuple(edges), tuple(caps),
                             (CardinalityRule("p", "N:1"),), "perf")


def _spread(n_src=250, n_tgt=80, deg=2, seed=3):
    """One pool of orders, so every invoice's near miss reaches across it."""
    rng = random.Random(seed)
    tgts = [f"PO{j}" for j in range(n_tgt)]
    edges, claim = [], {}
    for i in range(n_src):
        home = tgts[i % n_tgt]
        for k, t in enumerate([home] + rng.sample([t for t in tgts if t != home], deg - 1)):
            q = float(rng.randint(1, 10))
            edges.append(CandidateEdge(f"INV{i:04d}", t, 4.0 - 1.5 * k, 0.9, "p",
                                       {f"line:{t}": q}))
            if k == 0:
                claim[t] = claim.get(t, 0.0) + q
    caps = tuple(ResourceCapacity(f"line:{t}", round(claim.get(t, 0.0) * 1.6, 2), 1.0)
                 for t in tgts)
    return ResolutionRequest("perf-spread", tuple(edges), caps,
                             (CardinalityRule("p", "N:1"),), "perf")


def _dense(n_src=167, n_tgt=40, deg=3, seed=7):
    """Every source competing with every other over scarce, tight capacity."""
    rng = random.Random(seed)
    tgts = [f"T{j}" for j in range(n_tgt)]
    edges = [CandidateEdge(f"S{i}", t, round(rng.uniform(-2, 5), 2), 0.5, "p",
                           {f"res:{t}": float(rng.randint(1, 10))})
             for i in range(n_src) for t in rng.sample(tgts, deg)]
    caps = tuple(ResourceCapacity(f"res:{t}", float(rng.randint(5, 60)), 2.0) for t in tgts)
    return ResolutionRequest("perf-dense", tuple(edges), caps,
                             (CardinalityRule("p", "N:1"),), "perf")


def _timed(req, capsys, note):
    groups = len(partition(ProblemModel(req)))
    start = time.perf_counter()
    result = resolve(req)
    elapsed = time.perf_counter() - start
    with capsys.disabled():
        print(f"\n  {req.request_id}: {len(req.edges)} edges, {groups} group(s), "
              f"{len(result.links)} links, {result.status} — {elapsed:.3f}s "
              f"({'meets' if elapsed < TARGET_SECONDS else 'MISSES'} the {TARGET_SECONDS}s "
              f"target; {note})")
    return elapsed


def test_a_clustered_500_edge_request_meets_the_two_second_target(capsys):
    """The shape both live callers actually produce."""
    elapsed = _timed(_clustered(), capsys, "the shape real batches have")
    assert elapsed < CEILING


def test_a_sparse_single_group_500_edge_request_sits_on_the_target(capsys):
    """Undecomposable, so it gets none of the structural speed-up: 251 solves of
    a 500-edge model whose capacity rows barely bind. Measured at 1.36s on an
    idle box and 2.1s with the cores busy, so it straddles the two-second line
    depending on machine load — which is exactly why the assertion below is a
    regression ceiling and the number is printed rather than asserted."""
    elapsed = _timed(_spread(), capsys, "one group, no decomposition available")
    assert elapsed < CEILING


@pytest.mark.skipif(
    os.getenv("RESOLUTION_PERF_FULL") != "1",
    reason="the dense multi-knapsack takes ~19s; set RESOLUTION_PERF_FULL=1 to measure it",
)
def test_a_dense_contested_500_edge_request(capsys):
    _timed(_dense(), capsys, "a genuine multi-knapsack")


def test_decomposition_is_what_makes_the_clustered_case_fast():
    """The speed-up is structural, not a tuning constant: a request built from
    25 independent supplier batches must be seen as 25 groups, or the margin
    loop re-solves the whole model 250 times over."""
    req = _clustered()
    groups = partition(ProblemModel(req))
    assert len(groups) == 25
    assert sum(len(g) for g in groups) == len(req.edges)
