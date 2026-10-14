"""Property tests over the adversarial corpus in ``generator.py``.

Hand-built graphs do not catch the failures that matter here, so these run over
1000 generated ones. The generator is seeded per graph rather than from a global
RNG, so a failure names the exact graph that broke and re-running reproduces it.

The corpus is asserted on before the properties are: see
``test_corpus_is_adversarial``. A property that passes on easy data has proved
nothing, and that test is what stops this file quietly becoming that.
"""
import collections
import itertools
import random
from functools import lru_cache

import pytest

from src.services.resolution import (
    CandidateEdge,
    ResolutionRequest,
    resolve,
)
from src.services.resolution.model import DEGENERACY_FLOOR, ProblemModel
from src.services.resolution.solver import PENALTY_LOG_ODDS, Program
from tests.services.resolution.generator import (
    BANDS,
    CORPUS,
    GRAPHS,
    _shape_bounds,
)

SEEDS = range(GRAPHS)
SMALL = [i for i, g in enumerate(CORPUS) if g.small]
LARGE = [i for i, g in enumerate(CORPUS) if not g.small]
INFEASIBLE = [i for i, g in enumerate(CORPUS) if "infeasible" in g.traits]

# How many links to verify per large graph. Every link is verified on the small
# graphs by full enumeration; above that the cost is a re-solve per link, so a
# fixed, deterministically chosen sample keeps the suite finite without ever
# substituting an estimate for the real thing.
LARGE_MARGIN_SAMPLE = 5


@lru_cache(maxsize=None)
def _resolved(seed: int):
    """One solve per graph, shared by every property that starts from it."""
    return resolve(CORPUS[seed].request)


def _brute_force_optima(req, sources=None):
    """Every cheapest valid assignment, by enumeration. Same cost function the solver
    minimises: -log_odds per taken edge, plus the penalty per unassigned source.

    ``sources`` names the sources that must be accounted for. It matters when an
    edge has been removed to measure a margin: a source that loses its last
    candidate is still unassigned, and still costs the penalty.

    Returns ``(best_cost, [set of (source_id, target_id) per optimal assignment])``.
    More than one optimum means the evidence ties, and which of them comes back
    is then the tie-break rule's job rather than the solver's.
    """
    per_target, per_source = _shape_bounds(req.rules[0].shape)
    caps = {c.resource_id: c.capacity + c.tolerance for c in req.capacities}
    by_source: dict[str, list[CandidateEdge]] = {}
    for e in req.edges:
        by_source.setdefault(e.source_id, []).append(e)
    orphaned = len(set(sources) - set(by_source)) if sources is not None else 0

    options_per_source = []
    for s in sorted(by_source):
        opts = []
        cand = by_source[s]
        limit = len(cand) if per_source is None else min(per_source, len(cand))
        for size in range(0, limit + 1):
            opts.extend(itertools.combinations(cand, size))
        options_per_source.append(opts)

    best = None
    optima: list[frozenset] = []
    for combo in itertools.product(*options_per_source):
        taken = [e for group in combo for e in group]
        counts: dict[str, int] = {}
        used: dict[str, float] = {}
        for e in taken:
            counts[e.target_id] = counts.get(e.target_id, 0) + 1
            for r, q in e.consumes.items():
                used[r] = used.get(r, 0.0) + float(q)
        if per_target is not None and any(v > per_target for v in counts.values()):
            continue
        if any(used.get(r, 0.0) > caps.get(r, float("inf")) + 1e-9 for r in used):
            continue
        unassigned = sum(1 for group in combo if not group) + orphaned
        cost = sum(-e.log_odds for e in taken) + PENALTY_LOG_ODDS * unassigned
        picked = frozenset((e.source_id, e.target_id) for e in taken)
        if best is None or cost < best - 1e-9:
            best, optima = cost, [picked]
        elif abs(cost - best) <= 1e-9 and picked not in optima:
            optima.append(picked)
    return best, optima


def _brute_force_best(req, sources=None) -> float:
    return _brute_force_optima(req, sources)[0]


def _without(req: ResolutionRequest, link) -> ResolutionRequest:
    return ResolutionRequest(
        request_id=req.request_id,
        edges=tuple(e for e in req.edges
                    if (e.source_id, e.target_id) != (link.source_id, link.target_id)),
        capacities=req.capacities,
        rules=req.rules,
        profile_registry_version=req.profile_registry_version,
    )


# ---------------------------------------------------------------------------
# The corpus itself
# ---------------------------------------------------------------------------
def test_corpus_is_adversarial(capsys):
    """The generated set has to be hard enough to be worth running properties on.

    A generator that emits sparse graphs where every source has one obvious
    target passes all seven properties while testing nothing, so the thresholds
    the brief sets for the corpus are asserted here, and the realised
    distribution is printed so a reader can see what actually ran.
    """
    n = len(CORPUS)
    traits = collections.Counter(t for g in CORPUS for t in g.traits)
    statuses = collections.Counter(_resolved(i).status for i in SEEDS)
    bands = collections.Counter()
    for g in CORPUS:
        for lo, hi in BANDS:
            if lo <= g.nodes <= hi:
                bands[f"{lo}-{hi}"] += 1
                break
    edges = [len(g.request.edges) for g in CORPUS]

    with capsys.disabled():
        print(f"\n  corpus: {n} graphs, {sum(edges)} edges "
              f"(min {min(edges)}, max {max(edges)})")
        print(f"  nodes {min(g.nodes for g in CORPUS)}-{max(g.nodes for g in CORPUS)}: "
              + ", ".join(f"{k} {bands[k]} ({100 * bands[k] / n:.0f}%)"
                          for k in (f"{lo}-{hi}" for lo, hi in BANDS)))
        for t in ("near_tie", "contested", "infeasible"):
            print(f"  {t:11s} {traits[t]:4d} ({100 * traits[t] / n:.1f}%)")
        print("  statuses: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))

    # A source with three or more candidates the evidence cannot separate.
    assert traits["near_tie"] >= 0.30 * n, traits["near_tie"]
    # A resource whose claimants together want more than it holds.
    assert traits["contested"] >= 0.20 * n, traits["contested"]
    # Built so that no assignment, not even the empty one, can satisfy them.
    assert traits["infeasible"] >= 0.10 * n, traits["infeasible"]
    # Spanning the range, not clustered at the small end: every band is real and
    # at least a third of the corpus is above 30 nodes.
    assert all(bands[f"{lo}-{hi}"] >= 0.10 * n for lo, hi in BANDS), bands
    assert max(g.nodes for g in CORPUS) >= 190
    assert sum(v for k, v in bands.items() if k in ("31-90", "91-200")) >= 0.33 * n
    # And every status is actually exercised, so no property passes by
    # returning early on a corpus that never solves anything.
    assert all(statuses[s] >= 25 for s in ("RESOLVED", "DEGENERATE", "INFEASIBLE")), statuses


# ---------------------------------------------------------------------------
# 1. No over-consumption
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_no_resource_is_consumed_beyond_capacity_plus_tolerance(seed):
    req = CORPUS[seed].request
    result = _resolved(seed)
    if result.status == "INFEASIBLE":
        return
    caps = {c.resource_id: c for c in req.capacities}
    edge_of = {(e.source_id, e.target_id): e for e in req.edges}
    used: dict[str, float] = {}
    for link in result.links:
        for r, q in edge_of[(link.source_id, link.target_id)].consumes.items():
            used[r] = used.get(r, 0.0) + float(q)
    for r, total in used.items():
        assert total <= caps[r].bound + 1e-9, f"{req.request_id}: {r} over-consumed"


# ---------------------------------------------------------------------------
# 2. Cardinality respected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_no_end_exceeds_the_cardinality_its_profile_allows(seed):
    req = CORPUS[seed].request
    result = _resolved(seed)
    if result.status == "INFEASIBLE":
        return
    per_target, per_source = _shape_bounds(req.rules[0].shape)
    per_src = collections.Counter(l.source_id for l in result.links)
    per_tgt = collections.Counter(l.target_id for l in result.links)
    if per_source is not None:
        assert max(per_src.values(), default=0) <= per_source, req.request_id
    if per_target is not None:
        assert max(per_tgt.values(), default=0) <= per_target, req.request_id


# ---------------------------------------------------------------------------
# 3. Optimality
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SMALL)
def test_solver_objective_matches_brute_force(seed):
    """Under 8 nodes every valid assignment is enumerated and the best one has
    to be the one the solver returned."""
    result = _resolved(seed)
    if result.status == "INFEASIBLE":
        return
    assert result.objective == pytest.approx(
        _brute_force_best(CORPUS[seed].request), abs=1e-6
    ), CORPUS[seed].request.request_id


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_shuffling_the_inputs_changes_nothing(seed):
    req = CORPUS[seed].request
    baseline = _resolved(seed)
    rng = random.Random(seed + 10_000)
    edges = list(req.edges)
    caps = list(req.capacities)
    rng.shuffle(edges)
    rng.shuffle(caps)
    shuffled = ResolutionRequest(
        request_id=req.request_id, edges=tuple(edges), capacities=tuple(caps),
        rules=req.rules, profile_registry_version=req.profile_registry_version,
    )
    assert resolve(shuffled) == baseline


def test_the_same_request_solved_a_hundred_times_is_byte_identical():
    """The brief's wording, tested literally: shuffle the inputs between runs and
    the whole result — link order included — must not move. The graph is the
    hardest contested one in the corpus, not an arbitrary pick."""
    contested = [i for i in SEEDS
                 if "contested" in CORPUS[i].traits and "near_tie" in CORPUS[i].traits
                 and CORPUS[i].nodes <= 30 and _resolved(i).links]
    seed = max(contested, key=lambda i: len(CORPUS[i].request.edges))
    req = CORPUS[seed].request
    baseline = _resolved(seed)
    rng = random.Random(0)
    for _ in range(100):
        edges = list(req.edges)
        caps = list(req.capacities)
        rng.shuffle(edges)
        rng.shuffle(caps)
        again = resolve(ResolutionRequest(
            request_id=req.request_id, edges=tuple(edges), capacities=tuple(caps),
            rules=req.rules, profile_registry_version=req.profile_registry_version,
        ))
        assert again == baseline


# ---------------------------------------------------------------------------
# 5. Margin sign
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_margin_is_never_negative_and_a_tie_is_degenerate(seed):
    result = _resolved(seed)
    if result.status == "INFEASIBLE":
        return
    assert all(l.margin >= 0.0 for l in result.links), CORPUS[seed].request.request_id
    if any(l.margin == 0.0 for l in result.links):
        assert result.status == "DEGENERATE"
    if result.status == "DEGENERATE":
        assert any(l.margin_normalised < DEGENERACY_FLOOR for l in result.links)


# ---------------------------------------------------------------------------
# 6. Monotonicity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_raising_a_chosen_edge_never_removes_it(seed):
    req = CORPUS[seed].request
    result = _resolved(seed)
    if not result.links:
        return
    chosen = result.links[0]
    raised = tuple(
        CandidateEdge(e.source_id, e.target_id, e.log_odds + 2.5, e.confidence,
                      e.profile_id, e.consumes)
        if (e.source_id, e.target_id) == (chosen.source_id, chosen.target_id) else e
        for e in req.edges
    )
    after = resolve(ResolutionRequest(
        request_id=req.request_id, edges=raised, capacities=req.capacities,
        rules=req.rules, profile_registry_version=req.profile_registry_version,
    ))
    assert (chosen.source_id, chosen.target_id) in {
        (l.source_id, l.target_id) for l in after.links
    }, req.request_id


# ---------------------------------------------------------------------------
# 7. Infeasibility detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", INFEASIBLE)
def test_constructed_infeasible_graphs_are_always_detected(seed):
    """A source whose every candidate alone outstrips its resource cannot be
    placed by any assignment, and the layer must say so rather than drop it."""
    result = _resolved(seed)
    assert result.status == "INFEASIBLE", CORPUS[seed].request.request_id
    assert result.infeasibility_certificate
    assert result.links == ()
    assert set(result.unassigned_sources) == {e.source_id
                                              for e in CORPUS[seed].request.edges}


# ---------------------------------------------------------------------------
# The margin is the real thing, not an estimate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SMALL)
def test_every_margin_matches_a_brute_force_re_solve(seed):
    """Any shortcut in the margin computation has to survive full enumeration."""
    req = CORPUS[seed].request
    result = _resolved(seed)
    if result.status == "INFEASIBLE":
        return
    all_sources = {e.source_id for e in req.edges}
    for link in result.links:
        expected = _brute_force_best(_without(req, link), sources=all_sources) - result.objective
        assert link.margin == pytest.approx(expected, abs=1e-6), (
            f"{req.request_id}: margin for {link.source_id}->{link.target_id}"
        )


def _undecomposed_optimum(req: ResolutionRequest) -> float:
    """The optimum with none of the margin loop's shortcuts in play.

    One program over every edge of the request at once — no split into
    independent groups, no closed form for isolated candidates, no re-solve
    confined to a group. That is deliberately the pre-optimisation code path, so
    a bug in any of those shortcuts shows up as a disagreement with this.
    """
    return Program(ProblemModel(req)).solve().cost


@pytest.mark.parametrize("seed", LARGE)
def test_margins_on_large_graphs_match_an_undecomposed_re_solve(seed):
    """Above enumeration size the oracle is the same model solved in one piece."""
    req = CORPUS[seed].request
    result = _resolved(seed)
    if result.status == "INFEASIBLE" or not result.links:
        return
    n = len(result.links)
    picks = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})[:LARGE_MARGIN_SAMPLE]
    before = {e.source_id for e in req.edges}
    for k in picks:
        link = result.links[k]
        reduced = _without(req, link)
        # A source that loses its last candidate drops out of the reduced
        # request entirely, but it is still unassigned and still costs.
        vanished = before - {e.source_id for e in reduced.edges}
        expected = (_undecomposed_optimum(reduced)
                    + PENALTY_LOG_ODDS * len(vanished) - result.objective)
        assert link.margin == pytest.approx(expected, abs=1e-6), (
            f"{req.request_id}: margin for {link.source_id}->{link.target_id}"
        )


@pytest.mark.parametrize("seed", LARGE)
def test_splitting_a_request_into_groups_does_not_change_the_optimum(seed):
    """The decomposition is exact: the same objective comes back whether the
    model is solved in one piece or group by group."""
    result = _resolved(seed)
    if result.status == "INFEASIBLE":
        return
    assert result.objective == pytest.approx(
        _undecomposed_optimum(CORPUS[seed].request), abs=1e-6
    ), CORPUS[seed].request.request_id


@pytest.mark.parametrize("seed", SMALL)
def test_ties_are_broken_by_canonical_rank_not_by_the_solver(seed):
    """When the evidence ties, an explicit rule picks the winner — not HiGHS.

    Canonical order is `(source_id, target_id, profile_id)`, and each edge
    carries its position in that order as a sub-unit term in the objective. So
    among assignments that tie on evidence, the one whose chosen edges have the
    smallest total position wins. Every optimum is enumerated here and the
    returned one has to be that one.

    Without the rank term the canonical sort still makes the *model* identical
    under a shuffled input, so the determinism property would stay green while
    the choice among tied optima quietly became the solver's to make. This is the
    test that notices.
    """
    req = CORPUS[seed].request
    result = _resolved(seed)
    if result.status == "INFEASIBLE":
        return
    _, optima = _brute_force_optima(req)
    if len(optima) < 2:
        return

    order = {
        (e.source_id, e.target_id): i
        for i, e in enumerate(sorted(req.edges,
                                     key=lambda e: (e.source_id, e.target_id, e.profile_id)))
    }
    ranked = sorted((sum(order[k] for k in opt), sorted(opt)) for opt in optima)
    if ranked[0][0] == ranked[1][0]:
        return  # equal total rank: the rule does not separate these two

    assert {(l.source_id, l.target_id) for l in result.links} == set(
        map(tuple, ranked[0][1])
    ), req.request_id


# Exact ties are rare in a random corpus and rarer still on graphs small enough
# to enumerate, so the rule also gets cases built to be nothing but ties.
_TIE_CASES = {
    "one source, three identical candidates": (
        [("INV-1", "PO-3"), ("INV-1", "PO-1"), ("INV-1", "PO-2")], "1:1", {}, ()
    ),
    "two identical claimants, room for one": (
        [("INV-1", "PO-1"), ("INV-2", "PO-1")], "N:1",
        {"line:PO-1": 100.0}, (("line:PO-1", 100.0, 0.0),)
    ),
    # The three below are the shapes that actually notice when the rank term is
    # removed: partial capacity, where which claimants get in is a free choice
    # among many equally good ones. Left to HiGHS, the last claimants win.
    "four identical claimants, room for two": (
        [(f"INV-{i}", "PO-1") for i in range(1, 5)], "N:1",
        {"line:PO-1": 100.0}, (("line:PO-1", 200.0, 0.0),)
    ),
    "five identical claimants, room for three": (
        [(f"INV-{i}", "PO-1") for i in range(1, 6)], "N:1",
        {"line:PO-1": 100.0}, (("line:PO-1", 300.0, 0.0),)
    ),
    "three sources over two orders, capacity for two links": (
        [(f"INV-{i}", f"PO-{j}") for i in (1, 2, 3) for j in (1, 2)], "N:1",
        {"line": 100.0}, (("line", 200.0, 0.0),)
    ),
}

# Shapes so symmetric that every optimal assignment carries the same total rank.
# The rule cannot reach these — see DECISIONS.md, "what the rank term does not
# decide" — so what is asserted is what is actually true of them: the answer is
# one of the optima and it does not move between runs.
_SYMMETRIC_TIE_CASES = {
    "three sources, three targets, every pairing identical": (
        [(s, t) for s in ("INV-1", "INV-2", "INV-3")
         for t in ("PO-1", "PO-2", "PO-3")], "1:1"
    ),
    "four identical sources, two targets, two must go unplaced": (
        [(s, t) for s in ("INV-1", "INV-2", "INV-3", "INV-4")
         for t in ("PO-1", "PO-2")], "1:1"
    ),
}


def _tie_request(name, pairs, shape, consumes=(), caps=()):
    from src.services.resolution import CardinalityRule, ResourceCapacity

    return ResolutionRequest(
        request_id=f"tie:{name}",
        edges=tuple(CandidateEdge(s, t, 2.5, 0.9, "p", dict(consumes)) for s, t in pairs),
        capacities=tuple(ResourceCapacity(*c) for c in caps),
        rules=(CardinalityRule("p", shape),),
        profile_registry_version="test-v1",
    )


def _rank_order(req):
    return {(e.source_id, e.target_id): i
            for i, e in enumerate(sorted(req.edges,
                                         key=lambda e: (e.source_id, e.target_id, e.profile_id)))}


@pytest.mark.parametrize("name", sorted(_TIE_CASES))
def test_a_pure_tie_is_decided_by_canonical_rank(name):
    """Nothing but the rule can choose here, and it has to be the rule that does."""
    pairs, shape, consumes, caps = _TIE_CASES[name]
    req = _tie_request(name, pairs, shape, consumes, caps)
    result = resolve(req)
    _, optima = _brute_force_optima(req)
    assert len(optima) > 1, f"{name} is not actually a tie"

    order = _rank_order(req)
    ranked = sorted((sum(order[k] for k in opt), sorted(opt)) for opt in optima)
    assert ranked[0][0] != ranked[1][0], f"{name}: rank does not separate the optima"
    assert {(l.source_id, l.target_id) for l in result.links} == set(map(tuple, ranked[0][1]))
    assert result.status == "DEGENERATE"


@pytest.mark.parametrize("name", sorted(_SYMMETRIC_TIE_CASES))
def test_a_symmetric_tie_is_stable_even_though_the_rule_cannot_reach_it(name):
    """The known edge of the tie-break, pinned so it cannot drift unnoticed.

    A rank term added to the objective minimises the *total* rank of the chosen
    edges, which orders most ties but not a symmetric one: every perfect matching
    of a 3x3 block of identical edges carries the same total, so the rule is
    silent and HiGHS picks. What survives is that the model is identical for
    identical inputs, so the pick does not move — the result is reproducible even
    where it is not rule-determined.

    A symmetric tie is one where the *smallest* total rank is itself achieved by
    more than one assignment, which is what these two cases check they are.
    """
    pairs, shape = _SYMMETRIC_TIE_CASES[name]
    req = _tie_request(name, pairs, shape)
    _, optima = _brute_force_optima(req)
    order = _rank_order(req)
    ranked = sorted((sum(order[k] for k in opt), sorted(opt)) for opt in optima)
    assert len(optima) > 1, f"{name} is not a tie"
    assert ranked[0][0] == ranked[1][0], f"{name} is separable after all"

    first = resolve(req)
    assert frozenset((l.source_id, l.target_id) for l in first.links) in set(optima)
    assert first.status == "DEGENERATE"
    for _ in range(10):
        assert resolve(req) == first
