# Resolution layer — decisions

Phase 1 turns the pairwise scorer's verdicts into a globally coherent assignment
and reports how forced that assignment was. This file records the choices that
are not obvious from the code, so the next person does not have to re-derive
them.

## Solver: scipy / HiGHS, not OR-Tools

The brief asked for OR-Tools `SimpleMinCostFlow`, falling back to a MIP solver
where value tolerances make capacity continuous. OR-Tools is not installed in
either virtualenv in this repo; `scipy` (with `scipy.optimize.milp`, which is
HiGHS) is, in both. HiGHS is exact, deterministic, and covers both the integer
assignment case and the continuous-capacity case in one formulation, so the
whole layer ships with no new dependency.

The consequence: there is one model, not two. A pure assignment problem is just
the case where no edge declares any `consumes`.

`scipy` is now declared in `requirements.txt` with a `>=1.9` floor, the release
`scipy.optimize.milp` arrived in. It was previously present in both virtualenvs
only as a transitive dependency and was never named, which left this layer one
unrelated dependency bump away from disappearing.

No upper bound is pinned. The tie-break is an explicit term in the objective
rather than a reliance on the solver's own ordering, so a HiGHS version change
cannot move an answer the rule reaches — with the one exception recorded under
the tie-break below, where the rule is silent and the solver picks. The two
virtualenvs here run different versions (1.16.3 under `venv`, 1.17.1 under
`.venv`) and the whole package's tests pass identically on both, which is the
evidence for that rather than the argument.

## Cost scaling multiplier: 10,000

Edge costs must be integers for the tie-break below to be exact. `log_odds`
arrives as a float, so it is multiplied by `COST_SCALE = 10_000` and rounded.

Four decimal places is the precision the scorer itself reports — `linking_engine`
rounds its log-odds `L` to six decimals and its final `F` to four — so nothing
meaningful is lost. Two log-odds values that differ by less than 1e-4 tie, and
are then separated by the tie-break rule rather than by floating-point noise.

The multiplier is fixed. It is not tuned per run, and changing it changes which
near-identical edges tie, so it is part of the reproducibility record.

## Unassignment penalty: 1,000,000 scaled units (100 log-odds)

Leaving a source unassigned costs `UNASSIGNMENT_PENALTY`. Real log-odds from the
scorer sit in roughly [-12, +12], so the worst edge in a realistic request costs
`12 * 10_000 = 120_000`. The penalty is an order of magnitude above that, which
makes assignment always preferred wherever one is feasible — the behaviour the
brief asks for.

It can never force an infeasible solve: the unassignment variable is always free
to take the value 1, so the model with it is always satisfiable.

It is also why a single-candidate link's margin is ~100 plus its own log-odds:
the alternative to that link is not a worse PO, it is no PO at all, and the
margin says so.

## Tie-break: canonical rank, not solver order

Edges are sorted by `(source_id, target_id, profile_id)` before the model is
built, and every index downstream — variable position, cost vector, rank —
derives from that order and nothing else. A shuffled input therefore produces a
byte-identical model.

Among solutions that tie on true cost, the chosen one is decided as far as
possible by an explicit term in the objective rather than by solver internals:
each edge variable carries its canonical rank as a sub-unit cost. How far that
reaches is set out below, and it is not all the way. Because the true cost is first multiplied by
`tiebreak_scale = n² + 1`, and the largest achievable sum of ranks is
`n(n-1)/2 < n² + 1`, the rank term can only ever separate solutions that are
already equal on evidence. It never changes which assignment is optimal.

**Scope of the rank term.** Every solve carries it — the primary one and every
margin re-solve — so the layer never hands a choice among tied optima to HiGHS
where the rule can reach.

An earlier version ranked only the affected source's edges during a re-solve, on
the theory that ranking everything splits objective ties HiGHS would otherwise
prune, and recorded that as measured 3x cheaper. Re-measured interleaved on a
500-edge request, that is backwards: 60 re-solves take 0.44s ranking every edge
against 0.71s ranking one source's, on a single-group model, and the two are
level (1.02x) on a clustered one. The simpler and more determined option is also
the cheaper one, so it is the one in the code.

**What the rank term decides, and what it does not.** The rule is not quite
"lexicographic on `(source_id, target_id)`". A term added to a sum minimises the
*total* rank of the chosen edges, so among tied optima the winner is the one
whose edges carry the smallest rank sum in canonical order. That orders most
ties. It does not order a symmetric one: every perfect matching of a 3×3 block of
identical edges has the same total, so the rule is silent there and HiGHS picks.
The pick is still stable — identical inputs build an identical model — but that
is stability, not a rule, and the difference matters because the brief asks for
the latter.

`test_a_symmetric_tie_is_stable_even_though_the_rule_cannot_reach_it` pins that
boundary so it cannot drift unnoticed, and
`test_a_pure_tie_is_decided_by_canonical_rank` plus
`test_ties_are_broken_by_canonical_rank_not_by_the_solver` check the cases the
rule does reach, by enumerating every optimum and demanding the min-rank one
back. Closing the gap entirely would need rank weights growing geometrically
with the edge count, so that no two distinct edge sets could share a sum; that is
not representable in float64 for a 500-edge request, and buying it with a
per-edge sequence of re-solves would cost more than the margin loop itself.

## Independent groups are solved independently

A request is rarely one problem. Two edges can only interfere if a constraint row
can hold them both: they share a source (every source has a coverage row), they
share a target that carries a cardinality limit, or they draw on the same
declared-capacity resource. `model.partition` unions exactly those three
relations and hands the solver one group at a time.

Nothing about the answer changes. The objective is a plain sum over edges, no row
spans two groups, and the rank tie-break is also a per-edge sum, so the minimum
of the whole is the sum of the minima — and the smallest total rank over the
whole is achieved exactly when each group's own total rank is smallest, so the
tie-break survives the split as well. A target nobody bounded and a resource
nobody gave a capacity write no row at all, so they couple nothing — which is why
the union is over active constraints rather than over "shares a target".

What changes is the margin loop, which is where nearly all the time goes. An edge
can only perturb its own group, so forbidding it re-solves that group and carries
the other groups' costs over untouched. On the shape the live callers actually
produce — `deal_assignment_service` already batches by supplier — a 500-edge
request falls into 25 groups, and the loop goes from 251 solves of a 500-edge
model to 251 solves of a 20-edge one. Measured below.

`test_splitting_a_request_into_groups_does_not_change_the_optimum` checks the
claim directly on every large graph in the corpus: the objective is solved twice,
once group by group and once as a single undecomposed program, and the two must
agree. The margin oracle for large graphs uses the undecomposed program too, so
a bug in the split shows up as a wrong margin rather than passing quietly.

## What INFEASIBLE means

The brief is ambiguous here, and this is the one place a judgement call was
needed. Its regression list requires that five invoices, of which any four fit a
PO, **resolve** (flagged DEGENERATE) with one invoice dropped. So dropping a
source is a legal, ordinary outcome. But its certificate example describes a
group of invoices over-claiming one PO line, which is structurally the same
situation.

The rule implemented, and the reason for it:

- **Losing a competition for scarce capacity is not infeasible.** The loser is
  reported in `unassigned_sources`, the request resolves, and if the choice of
  loser was arbitrary the near-zero margins make it `DEGENERATE`. This is what
  the five-invoice case needs, and it is honest: the documents do not contradict
  each other, they compete.
- **INFEASIBLE means no assignment satisfies the constraints — including the
  assignment that places nothing.** Two structural causes are detected before
  the objective is ever considered: a resource whose `capacity + tolerance` is
  below zero, and a document every one of whose candidates is individually
  impossible (it alone claims more than the resource can ever hold, or its only
  targets have a cardinality limit below one).

Both produce a certificate in procurement terms naming the resource, its
capacity and tolerance, and every claimant — the wording the brief asked for.
Detection runs ahead of the solve and never relaxes a constraint to get an
answer.

## Where exactness was traded for speed

**Nowhere.** No margin is approximated, no optimality gap is tolerated
(`mip_rel_gap` is left at its exact default), and no constraint is relaxed. The
five things that make it fast are all exact:

1. **One equality row instead of two rows when a source may take one target.**
   `sum(edges) + unassigned = 1` says exactly what a coverage row plus a
   cardinality row said, and halves the model for the N:1 shape that dominates
   real invoice-to-PO work.
2. **Only edge variables are declared integral.** Each source's coverage row
   forces its unassignment variable to one minus an integral sum, so it lands on
   an integer without being branched on.
3. **Margin re-solves are batched across threads.** HiGHS releases the GIL, and
   each solve reads a shared immutable model and writes only its own result, so
   the answer does not depend on scheduling. Below eight re-solves the pool
   costs more than it saves and the loop runs serially.
4. **A closed-form margin for isolated single candidates.** Where a source has
   exactly one candidate, no other edge draws on the resources that link would
   free, and nobody else is queuing for its target's cardinality slot, forbidding
   it can change nothing but that source. The margin is then exactly
   `penalty + log_odds` and no re-solve is run. All three conditions are
   load-bearing: dropping either of the last two makes
   `test_every_margin_matches_a_brute_force_re_solve` fail.
5. **Only the affected group is re-solved for a margin.** See the section above.
   This is the largest of the five by a wide margin, and the only one that
   changes the shape of the cost curve rather than its constant.

## The corpus the properties run on

The properties are only worth as much as the graphs they run on, and a generator
that emits sparse graphs where every source has one obvious target passes all
seven while testing nothing. So `tests/services/resolution/generator.py` builds
hardness in deliberately, and `test_corpus_is_adversarial` asserts on the corpus
itself before any property is checked, printing what it actually produced.

Three things are injected rather than hoped for: a source with three or more
candidates within 0.05 log-odds of each other, so the tie-break is genuinely
load-bearing; a resource whose claimants together want more than it holds, so
somebody has to lose; and a source with nowhere it could possibly go, so the
layer has to fail closed. Node counts are drawn from four equally weighted bands
spanning 2 to 200, so the corpus is not clustered at the small end.

The realised distribution, which is what the assertions are written against:

| | count | share | floor |
|---|---|---|---|
| near-tie | 450 | 45.0% | 30% |
| contested resource | 622 | 62.2% | 20% |
| constructed infeasible | 130 | 13.0% | 10% |
| 2–8 nodes | 257 | 25.7% | 10% |
| 9–30 nodes | 246 | 24.6% | 10% |
| 31–90 nodes | 252 | 25.2% | 10% |
| 91–200 nodes | 245 | 24.5% | 10% |

1000 graphs, 82,275 edges, 1 to 366 edges each. Solving them returns 683
RESOLVED, 187 DEGENERATE and 130 INFEASIBLE, and the corpus test asserts that
all three statuses are exercised — otherwise a property could pass by returning
early on a corpus that never actually solves anything.

Traits are **observed** from the built request, never taken on trust from the
intent that built it: the near-tie trait is granted by re-reading the edge
weights, the contested trait by re-adding the claims, and infeasibility by an
independently written placeability check rather than by asking the layer.

Two oracles check the answers. Under 8 nodes every valid assignment is
enumerated, and the solver's objective and every one of its margins must match.
Above that, enumeration is hopeless, so the oracle is the same model solved in
one undecomposed piece — the pre-optimisation code path — which shares none of
the margin loop's shortcuts.

## Measured cost

The brief asks for solve plus margin computation on a 500-edge graph in under two
seconds, and reports of one number would hide the thing that actually decides it:
whether the graph falls into independent groups. Three shapes, each 500 edges,
timed on this dev box (4 cores, otherwise idle, `venv` / scipy 1.16.3), median of
three runs, before and after groups were solved independently:

| 500-edge shape | groups | before | after |
|---|---|---|---|
| clustered by supplier — what the live callers batch | 25 | 2.42s | **0.42s** |
| spread over one pool of 80 orders, sparse | 1 | 1.69s | **1.36s** |
| dense contested, a genuine multi-knapsack | 1 | 19.17s | **19.28s** |

Two of the three meet the target. The clustered shape is the one that matters —
`deal_assignment_service` already groups by supplier before it resolves, and
`deal_clustering` resolves one batch at a time — and it is 5.7x faster than it
was. The spread shape has no decomposition to find and sits on the line: 1.36s
idle, about 2.1s with the cores busy, so machine load alone moves it across.

The dense case misses by an order of magnitude and is not fixable by any exact
means available here. It is 167 branch-and-bound solves of a 501-edge model whose
capacity rows genuinely bind — a multi-knapsack, not an assignment problem. The
LP relaxation is 100x faster but is never integral on these (measured: 0/12), so
there is nothing to fall back to, and scipy's fixed per-call cost is only 0.88ms,
so the time is real solver work rather than overhead. Cutting it would mean
either an approximate margin, which the brief rules out and which would be worse
than a slow honest one, or a residual-graph shortest-path formulation that only
exists for the pure assignment case with no capacities. It is recorded, excluded
from the default test run, and left alone.

`tests/services/resolution/test_performance.py` builds all three and prints the
numbers on every run, so this table can be re-derived rather than believed.

## Mutation check

A suite that passes proves nothing on its own; what matters is whether it fails
when the code is wrong. Four bugs were introduced one at a time, the whole
package's tests run against each, and the bug reverted. None was committed, and
none was applied to this checkout — the runs happen in a detached `git worktree`
so a shared working tree is never left holding a deliberately broken solver.

| bug introduced | tests that caught it | failures |
|---|---|---|
| cost sign inverted, so the model maximises | 9, incl. `test_solver_objective_matches_brute_force`, `test_raising_a_chosen_edge_never_removes_it`, `test_the_optimal_pair_beats_the_per_source_best_choices` | 1436 |
| one resource capacity row dropped from the model | 8, incl. `test_no_resource_is_consumed_beyond_capacity_plus_tolerance`, `test_five_invoices_where_any_four_fit_are_degenerate`, `test_splitting_a_request_into_groups_does_not_change_the_optimum` | 55 |
| tie-break left to the solver (rank term removed) | 2: `test_a_pure_tie_is_decided_by_canonical_rank`, `test_ties_are_broken_by_canonical_rank_not_by_the_solver` | 5 |
| second-best assignment returned instead of the best | 15, incl. `test_solver_objective_matches_brute_force`, `test_every_margin_matches_a_brute_force_re_solve`, `test_four_invoices_summing_exactly_to_one_po_resolve_as_a_set` | 2061 |

The third one is the reason this exercise was worth doing. On the first run it
was **not caught at all**: the whole suite stayed green with the tie-break gone.
Canonical sorting already makes the *model* identical under a shuffled input, so
the determinism property never notices — it is testing that the inputs are
normalised, not that ties are ruled on. Nothing else looked at which of several
tied optima came back.

Two things were added to close it. `test_ties_are_broken_by_canonical_rank_not_by_the_solver`
enumerates every optimum of each small graph and demands the smallest-rank one
back; and because exact ties turned out to be almost absent from a corpus of
random weights (1 small graph in 257 had more than one optimum), the generator
now makes half of its near-tie injections *exact* ties, which is the only kind
that leaves a tie-break anything to do. That took the corpus from 1 graph with
competing optima to 7.

`test_a_pure_tie_is_decided_by_canonical_rank` adds five shapes built to be
nothing but ties. Three of them — partial capacity, where which of several
identical claimants gets in is a free choice — are the ones that actually
separate; with the rank term removed the last claimants win instead of the
first. The other two ties HiGHS happens to break the same way the rule would, so
they prove nothing on their own and are kept only as documentation of the shape.

## The margin is normalised against evidence, not against the objective

The brief specifies `margin_normalised = margin / |objective|`. Implemented
literally, that is unusable. The objective carries the unassignment penalty,
which is deliberately an order of magnitude larger than any edge, so in any
request where a single document goes unplaced the penalty dominates the
denominator and every margin divides down to nearly nothing — the whole result
then reads DEGENERATE however decisive it actually was.

Measured on the first real wiring: two quotes competing for one purchase order,
one scoring F=95 and the other F=90. A clear win. It normalised to 0.008 and was
reported contested.

Normalising against the whole solution's evidence has the same fault in reverse:
in a 200-link batch every individual margin is small next to the total, so every
large batch would read degenerate instead.

The denominator is therefore local — the link's own weight, `|log_odds|`. That is
what a margin is meaningfully a fraction of, and it does not move with batch
size. `margin` itself is unchanged and is still the exact objective difference.

## Tolerance is absolute here, relative everywhere else

`ResourceCapacity.tolerance` is an absolute quantity in the same unit as
`capacity`. Every tolerance in the existing engine is relative —
`linking_engine.cmp_numeric_tol` uses 1% drift, `three_way_match` uses
`max(0.01, 0.5%)` — and the scoring profiles carry no tolerance band at all.
Callers must convert. This layer does not guess a percentage.

## What this layer does not do

It emits the margin. It does not decide what to do with it: no routing
threshold, no promotion gate, no band. The scorer, the gap layer and the hard
gates are untouched.

## First caller: deal_clustering award detection

`deal_clustering.awarded_pos` scores every bid against every purchase order
exactly as before — same `quote_po` profile, same `min_score` gate — and then
hands the surviving pairs here under a 1:1 rule, because an order is placed with
one supplier and a quote wins at most one order.

Edge weight is the logit of the engine's own `F`, **not** the scorer's raw
pre-sigmoid `L`. Every gate in that codebase decides on F, which is L put
through the sigmoid and then multiplied by the coverage and cap terms, so
ranking on F is what preserves today's answers. The logit is monotone in F, so a
batch of one quote resolves to exactly the order the old argmax returned.

`awarded_po_scored` keeps its single-bid contract and goes through the same
resolver, so there is one tie-break rule rather than two. Its only behaviour
change is on an exact tie, which now resolves to the lexicographically first
`po_id` instead of whichever order the caller happened to pass the orders in.


## Second caller: proposing a parent for documents that reference none

`link_proposals` puts the documents nobody ever scored in front of the scorer.
An invoice that cites a purchase order is matched against that order and nothing
else, and one that cites nothing has never been matched against anything at all —
1,964 sit in bp_testdb's `_trgt` linked to no order, with 194 more citing an order
that does not exist. They are scored against their supplier's orders and resolved
as a set under N:1, and the margin is what routing acts on: `suggested` where the
evidence separates the chosen order from every alternative, `contested` where it
does not.

Two things this caller establishes about the layer, both measured.

**The floor a caller applies is its own, not the promotion path's.** Blanking the
reference on 400 invoices that DO name their order and re-scoring each against
its supplier's whole order set:

| | n | min | median | max |
|---|---|---|---|---|
| true order | 320 | 22.1 | 56.9 | 60.6 |
| wrong order, same supplier | 305 | 0.3 | 1.7 | 21.4 |

The populations do not overlap, and the true order ranked first for all 320 — so
the layer's job here is not choosing an order, it is deciding whether to speak.
The promotion path's review floor of 65 sits above *both* populations: reusing it
returned an empty list on every run, which reads as "nothing to review" rather
than "the bar is unreachable". A missing reference costs the profile's heaviest
signal, so a document with no parent is simply scored on a different scale.

**Capacity was the wrong constraint for this problem, and that is worth
recording, because this looked like its first real user.** An order absorbing
only the value it authorised is exactly the model, and enforcing it withheld 108
of 253 known-true parents: an invoice that alone bills more than its order, or a
set that together outruns it, lost its candidate and got no proposal.

Over-billing is real — 617 of 10,251 same-currency invoice/order pairs in
bp_testdb bill beyond their order — but it is a *finding*, not an impossibility.
`three_way_match` already ruled on this shape: it raises `po_over_consumed` and
never holds the document, because a buyer needs to see the over-billing. An
unlinked invoice is not over-billing anything; it is invisible. So the value
question is answered on the proposal (`claim`, `order_remaining`,
`within_order_value`) and left out of the solve.

What that costs is the ability to prefer an order with room over one without, for
a document that has both — and no document in the corpus has two candidates above
the floor, so the benefit was unmeasurable while the cost was measured. The
capacity model is not wrong; a purchase order and its invoices is just not a
resource that runs out. It still has no live caller.

(A caveat on the corpus, since these numbers are only worth their provenance:
bp_testdb is seeded, and the seeding shows — the median single invoice bills
exactly 1.000x its order's total and the median order carries 2.03 invoices
summing to 2.00x. It cannot say what real part-billing looks like. What it can
say is which of the two designs withholds true parents, and that is what decided
this.)
