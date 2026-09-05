"""The single entry point: a ResolutionRequest in, a ResolutionResult out."""
from __future__ import annotations

from .certificate import certify
from .contracts import ResolutionRequest, ResolutionResult, ResolvedLink
from .fingerprint import inputs_hash
from .model import DEGENERACY_FLOOR, ProblemModel
from .solver import PENALTY_LOG_ODDS, SOLVER_VERSION, Program, Solution


def _normalise(margin: float, scale: float) -> float:
    """Margin as a fraction of the link's own evidence, in [0, 1].

    Not `margin / |objective|` as the brief specifies. The objective carries the
    unassignment penalty, which is deliberately an order of magnitude larger than
    any edge, so in any request where one document goes unplaced the penalty
    dominates the denominator, every margin divides down to nearly nothing, and
    the whole result reads DEGENERATE however decisive it was. Measured on the
    first real caller: two quotes competing for one order at F=95 and F=90 — a
    clear win — normalised to 0.008 and was reported contested.

    Normalising against the whole solution's evidence has the same fault in
    reverse: in a 200-link batch every individual margin is small next to the
    total. So the denominator is local — this link's own weight — which is what
    a margin is meaningfully a fraction of, and is independent of batch size.

    A link at even odds carries no evidence to compare against, so any positive
    margin there is fully decisive rather than dividing by nothing.
    """
    if margin == float("inf"):
        return 1.0
    if scale == 0.0:
        return 1.0 if margin > 0 else 0.0
    return min(1.0, max(0.0, margin / abs(scale)))


def _link(pm: ProblemModel, i: int, best: Solution, alt: Solution) -> ResolvedLink:
    edge = pm.edges[i]
    margin = float("inf") if not alt.feasible else alt.cost - best.cost
    # Floating-point subtraction of two equal optima can leave a -1e-16 crumb.
    if margin != float("inf") and abs(margin) < 1e-9:
        margin = 0.0
    displaced = (
        ()
        if not alt.feasible
        else tuple(sorted({pm.edges[j].target_id for j in alt.chosen
                           if pm.edges[j].source_id == edge.source_id}))
    )
    return ResolvedLink(
        source_id=edge.source_id,
        target_id=edge.target_id,
        log_odds=edge.log_odds,
        margin=margin,
        margin_normalised=_normalise(margin, abs(edge.log_odds)),
        displaced_by=displaced,
    )


def _is_isolated(pm: ProblemModel, i: int, claimants: dict[str, int]) -> bool:
    """True when forbidding edge i can change nothing except its own source.

    That needs three things: the source has no other candidate, so it can only
    become unassigned; no other edge draws on the resources this one frees; and
    nobody else is queuing for the target's cardinality slot. When all three
    hold, the alternative solution is the current one minus this link, and its
    cost follows in closed form — no re-solve, and no approximation.
    """
    edge = pm.edges[i]
    if len(pm.edges_by_source[edge.source_id]) != 1:
        return False
    if any(claimants.get(r, 0) > 1 for r in edge.consumes):
        return False
    if (pm.target_bounds[edge.target_id] is not None
            and len(pm.edges_by_target[edge.target_id]) > 1):
        return False
    return True


def _isolated_link(pm: ProblemModel, i: int) -> ResolvedLink:
    """The closed-form link for an isolated single candidate: its only
    alternative is leaving the source unassigned."""
    edge = pm.edges[i]
    margin = PENALTY_LOG_ODDS + edge.log_odds
    return ResolvedLink(
        source_id=edge.source_id,
        target_id=edge.target_id,
        log_odds=edge.log_odds,
        margin=margin,
        margin_normalised=_normalise(margin, abs(edge.log_odds)),
        displaced_by=(),
    )


def resolve(request: ResolutionRequest) -> ResolutionResult:
    pm = ProblemModel(request)

    # Fail closed, and say why in domain terms. This runs before the objective
    # is considered at all: a contradiction in the constraints is a finding
    # about the documents, not a solver outcome to be worked around.
    reasons = certify(pm)
    if reasons:
        return ResolutionResult(
            request_id=request.request_id,
            status="INFEASIBLE",
            links=(),
            unassigned_sources=pm.sources,
            objective=0.0,
            infeasibility_certificate=reasons,
            inputs_hash=inputs_hash(request),
            solver_version=SOLVER_VERSION,
        )

    program = Program(pm)
    best = program.solve()
    if not best.feasible:
        return ResolutionResult(
            request_id=request.request_id,
            status="INFEASIBLE",
            links=(),
            unassigned_sources=pm.sources,
            objective=0.0,
            infeasibility_certificate=(
                "the constraint set admits no assignment; no single constraint "
                "explains it, so the combination is contradictory",
            ),
            inputs_hash=inputs_hash(request),
            solver_version=SOLVER_VERSION,
        )

    claimants: dict[str, int] = {}
    for e in pm.edges:
        for r in e.consumes:
            claimants[r] = claimants.get(r, 0) + 1

    needs_solve = [i for i in best.chosen if not _is_isolated(pm, i, claimants)]
    alternatives = dict(zip(needs_solve, program.solve_many(needs_solve)))

    links = []
    for i in best.chosen:
        if i in alternatives:
            links.append(_link(pm, i, best, alternatives[i]))
        else:
            links.append(_isolated_link(pm, i))
    links = tuple(links)
    degenerate = any(l.margin_normalised < DEGENERACY_FLOOR for l in links)

    return ResolutionResult(
        request_id=request.request_id,
        status="DEGENERATE" if degenerate else "RESOLVED",
        links=links,
        unassigned_sources=best.unassigned,
        objective=best.cost,
        infeasibility_certificate=None,
        inputs_hash=inputs_hash(request),
        solver_version=SOLVER_VERSION,
    )
