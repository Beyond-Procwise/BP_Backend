"""The adversarial graph corpus the property tests run on.

A generator that makes sparse graphs where every source has one obvious target
passes every property in the brief without testing anything. So this one is
built to be hard on purpose, and the corpus it produces is itself asserted on by
``test_corpus_is_adversarial`` — if the generator ever drifts back towards easy
data, that test fails before the properties start passing for the wrong reason.

Three things make a graph hard, and each is injected deliberately rather than
hoped for:

* **near-ties** — a source with three or more candidates whose evidence differs
  by less than 0.05 log-odds, so the choice between them is nearly arbitrary and
  any sloppiness in the tie-break shows up as non-determinism;
* **contested resources** — a resource whose claimants together want more than
  it holds, so somebody has to lose and greedy per-source choices go wrong;
* **constructed infeasibility** — a source with nowhere it could possibly go, so
  the layer has to fail closed with a certificate instead of quietly dropping it.

Sizes span 2 to 200 nodes in four equally-weighted bands, so half the corpus is
above 30 nodes. The traits reported for a graph are *observed* from the built
request, never taken on trust from the intent that built it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.services.resolution import (
    CandidateEdge,
    CardinalityRule,
    ResolutionRequest,
    ResourceCapacity,
)

GRAPHS = 1000
SHAPES = ("1:1", "N:1", "1:N", "N:M")
PROFILE = "p"

# A near-tie is evidence this close together. The brief's threshold.
TIE_BAND = 0.05

# Node-count bands, drawn with equal weight so the corpus is not clustered at
# the small end. The first band is what the brute-force oracles can enumerate.
BANDS = ((2, 8), (9, 30), (31, 90), (91, 200))

# How often each trait is attempted. Injection can fail — a two-node graph has
# no room for three competing candidates — so the realised rates are lower, and
# the corpus test asserts on the realised ones.
P_NEAR_TIE = 0.55
P_CONTESTED = 0.45
P_INFEASIBLE = 0.13
# Of the resources that were not deliberately squeezed, how many still bind.
P_ALSO_BINDING = 0.25


@dataclass(frozen=True)
class Graph:
    request: ResolutionRequest
    traits: frozenset
    nodes: int

    @property
    def small(self) -> bool:
        """Small enough to enumerate every valid assignment by hand."""
        return self.nodes <= 8


def _shape_bounds(shape: str) -> tuple[int | None, int | None]:
    """(max_sources_per_target, max_targets_per_source) for a bare rule."""
    if shape == "1:1":
        return 1, 1
    if shape == "N:1":
        return None, 1
    if shape == "1:N":
        return 1, None
    return None, None


def _sizes(rng: random.Random) -> tuple[int, int]:
    lo, hi = BANDS[rng.randrange(len(BANDS))]
    nodes = rng.randint(lo, hi)
    n_tgt = max(1, min(nodes - 1, round(nodes * rng.uniform(0.25, 0.6))))
    return nodes - n_tgt, n_tgt


def _graph(seed: int) -> Graph:
    rng = random.Random(seed)
    n_src, n_tgt = _sizes(rng)
    shape = SHAPES[rng.randrange(len(SHAPES))]
    metered = rng.random() < 0.65

    sources = [f"S{i}" for i in range(n_src)]
    targets = [f"T{j}" for j in range(n_tgt)]

    # --- edges -------------------------------------------------------------
    edges: list[CandidateEdge] = []
    tie_source = (
        rng.choice(sources) if n_tgt >= 3 and rng.random() < P_NEAR_TIE else None
    )
    for s in sources:
        if s == tie_source:
            # Three or more candidates that the evidence cannot separate. Half of
            # them are exact ties rather than near ones: only an exact tie leaves
            # more than one optimal assignment, and only then is there anything
            # for the tie-break rule to decide. Near ties test something else —
            # that a 0.01 difference is still allowed to decide.
            picked = rng.sample(targets, rng.randint(3, min(n_tgt, 5)))
            base = round(rng.uniform(-1.0, 4.0), 2)
            if rng.random() < 0.5:
                weights = [base] * len(picked)
            else:
                weights = [round(base + rng.uniform(0, TIE_BAND * 0.7), 4) for _ in picked]
        else:
            picked = rng.sample(targets, rng.randint(1, min(n_tgt, 4)))
            weights = [round(rng.uniform(-3.0, 5.0), 2) for _ in picked]
        for t, w in zip(picked, weights):
            consumes = {f"res:{t}": float(rng.randint(1, 10))} if metered else {}
            edges.append(CandidateEdge(s, t, w, 0.5, PROFILE, consumes))

    # --- capacities --------------------------------------------------------
    # Capacity always covers the largest single claim, so no document is
    # accidentally impossible: a resource is contested because its claimants
    # together want too much, which is the case worth testing, and infeasibility
    # only ever arrives on purpose below.
    claims: dict[str, list[float]] = {}
    for e in edges:
        for r, q in e.consumes.items():
            claims.setdefault(r, []).append(float(q))

    contestable = sorted(r for r, qs in claims.items() if len(qs) >= 2)
    squeeze_on = (
        rng.choice(contestable)
        if contestable and rng.random() < P_CONTESTED
        else None
    )
    capacities: list[ResourceCapacity] = []
    for r in sorted(claims):
        biggest, total = max(claims[r]), sum(claims[r])
        if r == squeeze_on:
            # Every claimant fits alone; together they do not. Somebody has to
            # lose, and losing a competition is a resolution, not a contradiction.
            cap = max(biggest, round(rng.uniform(biggest, total - 0.01), 2))
            tol = 0.0
        elif rng.random() < P_ALSO_BINDING:
            # Contention is not confined to the one resource that was squeezed:
            # a quarter of the rest bind too, so a graph can be contested in
            # several places at once.
            cap = max(biggest, round(rng.uniform(biggest, total), 2))
            tol = float(rng.choice((0, 1, 2)))
        else:
            # Generous: this resource is not what the assignment turns on. Every
            # resource binding at once makes each graph a multi-knapsack and buys
            # no extra coverage — the contested cases above are the ones that
            # test anything.
            cap = round(total + rng.uniform(0, total), 2)
            tol = float(rng.choice((0, 1, 2)))
        capacities.append(ResourceCapacity(r, cap, tol))

    # --- constructed infeasibility ----------------------------------------
    made_infeasible = False
    if rng.random() < P_INFEASIBLE and edges:
        victim = rng.choice(sources)
        mine = [e for e in edges if e.source_id == victim]
        by_id = {c.resource_id: c for c in capacities}
        rebuilt = []
        for e in mine:
            # Every one of this source's candidates alone outstrips its resource,
            # so no assignment — not even the empty one — can place it.
            t = e.target_id
            res = f"res:{t}"
            cap = by_id.get(res)
            bound = cap.bound if cap is not None else float(rng.randint(0, 20))
            if cap is None:
                capacities.append(ResourceCapacity(res, bound, 0.0))
                by_id[res] = capacities[-1]
                bound = by_id[res].bound
            rebuilt.append(CandidateEdge(e.source_id, t, e.log_odds, e.confidence,
                                         e.profile_id, {res: bound + rng.randint(1, 50)}))
        edges = [e for e in edges if e.source_id != victim] + rebuilt
        made_infeasible = True

    request = ResolutionRequest(
        request_id=f"G{seed}",
        edges=tuple(edges),
        capacities=tuple(capacities),
        rules=(CardinalityRule(PROFILE, shape),),
        profile_registry_version="test-v1",
    )
    return Graph(request, _observe(request, made_infeasible), n_src + n_tgt)


def _observe(req: ResolutionRequest, made_infeasible: bool) -> frozenset:
    """What the built request actually is, re-derived rather than assumed.

    ``made_infeasible`` records only that the construction was attempted; the
    ``infeasible`` trait is granted on the independently checked structure.
    """
    traits = set()

    by_source: dict[str, list[CandidateEdge]] = {}
    for e in req.edges:
        by_source.setdefault(e.source_id, []).append(e)

    for cand in by_source.values():
        weights = sorted(e.log_odds for e in cand)
        if any(weights[i + 2] - weights[i] <= TIE_BAND for i in range(len(weights) - 2)):
            traits.add("near_tie")
            break

    bounds = {c.resource_id: c.capacity + c.tolerance for c in req.capacities}
    claims: dict[str, float] = {}
    for e in req.edges:
        for r, q in e.consumes.items():
            claims[r] = claims.get(r, 0.0) + float(q)
    if any(total > bounds.get(r, float("inf")) for r, total in claims.items()):
        traits.add("contested")

    per_target, per_source = _shape_bounds(req.rules[0].shape)
    for cand in by_source.values():
        placeable = False
        for e in cand:
            if any(float(q) > bounds.get(r, float("inf")) for r, q in e.consumes.items()):
                continue
            if (per_target is not None and per_target < 1) or (
                per_source is not None and per_source < 1
            ):
                continue
            placeable = True
            break
        if not placeable:
            traits.add("infeasible")
            break
    if any(b < 0 for b in bounds.values()):
        traits.add("infeasible")
    if made_infeasible and "infeasible" not in traits:  # pragma: no cover
        raise AssertionError("infeasibility was constructed but is not observable")

    return frozenset(traits)


CORPUS: tuple[Graph, ...] = tuple(_graph(s) for s in range(GRAPHS))
