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

## Tie-break: lexicographic on (source_id, target_id)

Edges are sorted by `(source_id, target_id, profile_id)` before the model is
built, and every index downstream — variable position, cost vector, rank —
derives from that order and nothing else. A shuffled input therefore produces a
byte-identical model.

Among solutions that tie on true cost, the chosen one is decided by an explicit
term in the objective, not by solver internals: each edge variable carries its
canonical rank as a sub-unit cost. Because the true cost is first multiplied by
`tiebreak_scale = n² + 1`, and the largest achievable sum of ranks is
`n(n-1)/2 < n² + 1`, the rank term can only ever separate solutions that are
already equal on evidence. It never changes which assignment is optimal.

**Scope of the rank term.** The primary solve ranks every edge, so the returned
`links` are globally determined. A margin re-solve ranks only the edges of the
source whose link was forbidden. That is enough: a re-solve contributes two
things to the output, the objective value (independent of the rank term, since
cost is recomputed from `log_odds`) and `displaced_by` (which concerns only that
source). Ranking every edge in every re-solve is not free — it splits objective
ties that HiGHS would otherwise prune, and measured **3x slower** across the
margin loop. So the documented rule is:

> Among alternative optima, `displaced_by` reports the placement in which the
> affected source takes its lexicographically earliest targets.

Ties elsewhere in an alternative solution are not separated, because nothing in
the output depends on them.

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
four things that make it fast are all exact:

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
