"""Property tests over generated graphs.

Hand-built graphs do not catch the failures that matter here, so these generate
1000 of them. The generator is seeded per graph rather than by a global RNG, so
a failure names the exact graph that broke and re-running reproduces it.
"""
import itertools
import random

import pytest

from src.services.resolution import (
    CandidateEdge,
    CardinalityRule,
    ResolutionRequest,
    ResourceCapacity,
    resolve,
)
from src.services.resolution.model import DEGENERACY_FLOOR
from src.services.resolution.solver import PENALTY_LOG_ODDS

GRAPHS = 1000
SHAPES = ["1:1", "N:1", "1:N", "N:M"]


def _graph(seed: int) -> ResolutionRequest:
    rng = random.Random(seed)
    n_src = rng.randint(1, 3)
    n_tgt = rng.randint(1, 3)
    shape = rng.choice(SHAPES)
    profile = "p"
    sources = [f"S{i}" for i in range(n_src)]
    targets = [f"T{j}" for j in range(n_tgt)]

    metered = rng.random() < 0.5
    edges = []
    for s in sources:
        picked = rng.sample(targets, rng.randint(1, n_tgt))
        for t in picked:
            consumes = {f"res:{t}": float(rng.randint(1, 10))} if metered else {}
            edges.append(CandidateEdge(
                source_id=s, target_id=t,
                log_odds=round(rng.uniform(-3.0, 5.0), 2),
                confidence=0.5, profile_id=profile, consumes=consumes,
            ))
    capacities = []
    if metered:
        for t in targets:
            capacities.append(ResourceCapacity(f"res:{t}", float(rng.randint(0, 20)),
                                               float(rng.choice([0, 1, 2]))))
    return ResolutionRequest(
        request_id=f"G{seed}", edges=tuple(edges), capacities=tuple(capacities),
        rules=(CardinalityRule(profile, shape),),
        profile_registry_version="test-v1",
    )


def _bounds(req):
    rule = req.rules[0]
    per_target, per_source = rule.max_sources_per_target, rule.max_targets_per_source
    if rule.shape == "1:1":
        per_target, per_source = 1, 1
    elif rule.shape == "N:1":
        per_source = 1
    elif rule.shape == "1:N":
        per_target = 1
    return per_target, per_source


def _brute_force_best(req, sources=None) -> float:
    """Cheapest valid assignment, by enumeration. Same cost function the solver
    minimises: -log_odds per taken edge, plus the penalty per unassigned source.

    ``sources`` names the sources that must be accounted for. It matters when an
    edge has been removed to measure a margin: a source that loses its last
    candidate is still unassigned, and still costs the penalty. Leaving it out
    was a bug in this helper that made 612 margins look wrong.
    """
    per_target, per_source = _bounds(req)
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
        if best is None or cost < best:
            best = cost
    return best


@pytest.mark.parametrize("seed", range(GRAPHS))
def test_generated_graphs_respect_capacity_cardinality_and_margin_sign(seed):
    req = _graph(seed)
    result = resolve(req)
    if result.status == "INFEASIBLE":
        assert result.infeasibility_certificate
        return

    # 1. no over-consumption
    caps = {c.resource_id: c for c in req.capacities}
    edge_of = {(e.source_id, e.target_id): e for e in req.edges}
    used: dict[str, float] = {}
    for link in result.links:
        for r, q in edge_of[(link.source_id, link.target_id)].consumes.items():
            used[r] = used.get(r, 0.0) + float(q)
    for r, total in used.items():
        assert total <= caps[r].bound + 1e-9, f"{req.request_id}: {r} over-consumed"

    # 2. cardinality respected
    per_target, per_source = _bounds(req)
    per_src_count: dict[str, int] = {}
    per_tgt_count: dict[str, int] = {}
    for link in result.links:
        per_src_count[link.source_id] = per_src_count.get(link.source_id, 0) + 1
        per_tgt_count[link.target_id] = per_tgt_count.get(link.target_id, 0) + 1
    if per_source is not None:
        assert max(per_src_count.values(), default=0) <= per_source
    if per_target is not None:
        assert max(per_tgt_count.values(), default=0) <= per_target

    # 5. margin sign, and a tie is DEGENERATE
    assert all(l.margin >= 0.0 for l in result.links)
    if any(l.margin == 0.0 for l in result.links):
        assert result.status == "DEGENERATE"
    if result.status == "DEGENERATE":
        assert any(l.margin_normalised < DEGENERACY_FLOOR for l in result.links)


@pytest.mark.parametrize("seed", range(GRAPHS))
def test_solver_objective_matches_brute_force(seed):
    req = _graph(seed)
    result = resolve(req)
    if result.status == "INFEASIBLE":
        return
    assert result.objective == pytest.approx(_brute_force_best(req), abs=1e-6)


@pytest.mark.parametrize("seed", range(GRAPHS))
def test_shuffling_the_inputs_changes_nothing(seed):
    req = _graph(seed)
    baseline = resolve(req)
    rng = random.Random(seed + 10_000)
    for _ in range(3):
        edges = list(req.edges)
        rng.shuffle(edges)
        caps = list(req.capacities)
        rng.shuffle(caps)
        shuffled = ResolutionRequest(
            request_id=req.request_id, edges=tuple(edges), capacities=tuple(caps),
            rules=req.rules, profile_registry_version=req.profile_registry_version,
        )
        assert resolve(shuffled) == baseline


@pytest.mark.parametrize("seed", range(GRAPHS))
def test_raising_a_chosen_edge_never_removes_it(seed):
    req = _graph(seed)
    result = resolve(req)
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
    }


@pytest.mark.parametrize("seed", range(GRAPHS))
def test_constructed_infeasible_graphs_are_always_detected(seed):
    """Every source's every candidate claims more than its resource can ever
    hold, so no assignment places it."""
    rng = random.Random(seed + 500_000)
    n_src = rng.randint(1, 3)
    edges, caps = [], []
    for i in range(n_src):
        t = f"T{i}"
        cap = float(rng.randint(0, 50))
        tol = float(rng.randint(0, 5))
        edges.append(CandidateEdge(f"S{i}", t, round(rng.uniform(-2, 5), 2), 0.5, "p",
                                   {f"res:{t}": cap + tol + rng.randint(1, 100)}))
        caps.append(ResourceCapacity(f"res:{t}", cap, tol))
    req = ResolutionRequest(
        request_id=f"X{seed}", edges=tuple(edges), capacities=tuple(caps),
        rules=(CardinalityRule("p", "N:1"),), profile_registry_version="test-v1",
    )

    result = resolve(req)

    assert result.status == "INFEASIBLE"
    assert result.infeasibility_certificate
    assert result.links == ()


@pytest.mark.parametrize("seed", range(GRAPHS))
def test_every_margin_matches_a_brute_force_re_solve(seed):
    """The margin is the real cost of doing without the link, not an estimate.
    Any shortcut in the margin computation has to survive this."""
    req = _graph(seed)
    result = resolve(req)
    if result.status == "INFEASIBLE":
        return
    all_sources = {e.source_id for e in req.edges}
    for link in result.links:
        without = ResolutionRequest(
            request_id=req.request_id,
            edges=tuple(e for e in req.edges
                        if (e.source_id, e.target_id) != (link.source_id, link.target_id)),
            capacities=req.capacities, rules=req.rules,
            profile_registry_version=req.profile_registry_version,
        )
        expected = (_brute_force_best(without, sources=all_sources)
                    - result.objective)
        assert link.margin == pytest.approx(expected, abs=1e-6), (
            f"{req.request_id}: margin for {link.source_id}->{link.target_id}"
        )


def test_the_same_request_solved_a_hundred_times_is_byte_identical():
    """The brief's wording, tested literally: shuffle the inputs between runs and
    the whole result — link order included — must not move."""
    req = _graph(42)
    baseline = resolve(req)
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
        assert [(l.source_id, l.target_id) for l in again.links] == \
               [(l.source_id, l.target_id) for l in baseline.links]
        assert again.inputs_hash == baseline.inputs_hash
