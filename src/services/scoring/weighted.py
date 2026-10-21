"""Weighted scoring of supplier evaluations (SC-022): pure, deterministic, nothing hidden.

Turns evaluator scores and criterion weights into ranked totals, with knockout gates, a named
normalisation method, evaluator and criterion consistency flags, and weight sensitivity. It never
alters a score or a weight and never picks a winner: the ranking is arithmetic, the award is a
person's decision.
"""
from __future__ import annotations

import random
import statistics
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

METHOD_VERSION = "1.0.0"
NORMALISATIONS = ("none", "ratio_to_best", "min_max", "price_interpolation", "rank_based")
AGGREGATIONS = ("consensus_if_present", "mean", "median", "trimmed_mean")
_TOL = Decimal("1e-9")
_GATE, _GROUP = "gate_pass_fail", "group"
_METRIC_TYPES = ("price", "numeric_metric")
NOT_COMPUTED = ["krippendorff_alpha", "icc", "evaluator_rank_reversal"]


class ScoringError(ValueError):
    """The inputs cannot be scored; the message says what to fix."""


def _num(value: Any, what: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ScoringError(f"{what} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ScoringError(f"{what} is not a number") from None
    if out != out or abs(out) == float("inf"):
        raise ScoringError(f"{what} is not a finite number")
    return out


def _dec(value: Any, what: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ScoringError(f"{what} is not a number") from None


def _tree(criteria: Sequence[Mapping[str, Any]]):
    byid: Dict[str, Mapping[str, Any]] = {}
    for c in criteria:
        cid = str(c.get("criterion_id"))
        if cid in byid:
            raise ScoringError(f"criterion {cid} appears twice")
        byid[cid] = c
    children: Dict[Optional[str], List[str]] = {}
    for cid, c in byid.items():
        parent = c.get("parent_id")
        parent = None if parent in (None, "") else str(parent)
        if parent is not None and parent not in byid:
            raise ScoringError(f"criterion {cid} names a parent {parent} that does not exist")
        children.setdefault(parent, []).append(cid)
    sibs: Dict[Optional[str], Dict[str, float]] = {}
    for parent, ids in children.items():
        live = [i for i in ids if byid[i].get("type") != _GATE]
        if not live:
            continue
        weights = {i: _dec(byid[i].get("weight"), f"weight of {i}") for i in live}
        if any(v < 0 for v in weights.values()):
            raise ScoringError("a weight is negative")
        if abs(sum(weights.values()) - 1) > _TOL:
            raise ScoringError(f"weights under {parent or 'the top level'} sum to {sum(weights.values())}, "
                               "they must sum to 1")
        sibs[parent] = {i: float(v) for i, v in weights.items()}
    return byid, children, sibs


def _global_weights(byid, children, sibs) -> Dict[str, float]:
    out: Dict[str, float] = {}

    def walk(parent: Optional[str], factor: float) -> None:
        for cid, wt in sibs.get(parent, {}).items():
            if children.get(cid):
                walk(cid, factor * wt)
            else:
                out[cid] = factor * wt
    walk(None, 1.0)
    return out


def _aggregate(values: List[float], how: str) -> float:
    if how == "median":
        return float(statistics.median(values))
    if how == "trimmed_mean" and len(values) >= 5:
        v = sorted(values)[1:-1]
        return sum(v) / len(v)
    return sum(values) / len(values)


def _normalise_metric(values: Dict[str, float], method: str, direction: str, ref: Optional[float],
                      cid: str, warnings: List[str]) -> Dict[str, float]:
    low = direction == "lower_better"
    lo, hi = min(values.values()), max(values.values())
    best = lo if low else hi
    if method == "ratio_to_best":
        if any(v <= 0 for v in values.values()):
            raise ScoringError(f"{cid}: a price or metric is zero or negative, so a ratio to the best cannot be taken")
        return {s: (best / v if low else v / best) for s, v in values.items()}
    if method == "min_max":
        if hi == lo:
            warnings.append(f"{cid}: all suppliers are equal, so min_max gives every supplier 1")
            return {s: 1.0 for s in values}
        return {s: ((hi - v) / (hi - lo) if low else (v - lo) / (hi - lo)) for s, v in values.items()}
    if method == "price_interpolation":
        if not low:
            raise ScoringError(f"{cid}: price_interpolation is for lower-is-better prices")
        if ref is None or ref <= best:
            raise ScoringError(f"{cid}: price_reference must be given and above the lowest price")
        return {s: min(1.0, max(0.0, (ref - v) / (ref - best))) for s, v in values.items()}
    if method == "rank_based":
        n = len(values)
        order = sorted(values.items(), key=lambda kv: kv[1], reverse=not low)
        ranks: Dict[str, float] = {}
        i = 0
        while i < n:
            j = i
            while j + 1 < n and order[j + 1][1] == order[i][1]:
                j += 1
            for k in range(i, j + 1):
                ranks[order[k][0]] = (i + j) / 2 + 1
            i = j + 1
        return {s: (1.0 if n == 1 else (n - r) / (n - 1)) for s, r in ranks.items()}
    raise ScoringError(f"{cid}: a price or metric criterion needs a normalisation method, not '{method}'")


def _rank(totals: Mapping[str, float]) -> List[Tuple[str, float, int, bool]]:
    rounded = {s: round(t, 6) for s, t in totals.items()}
    order = sorted(rounded, key=lambda s: (-rounded[s], s))
    out = []
    for s in order:
        higher = sum(1 for o in rounded.values() if o > rounded[s])
        tied = sum(1 for o in rounded.values() if o == rounded[s]) > 1
        out.append((s, totals[s], higher + 1, tied))
    return out


def _with_weight(sibs, parent, cid, new) -> Dict[Optional[str], Dict[str, float]]:
    """Set one sibling's weight and rescale the others in proportion so the level still sums to 1."""
    group = dict(sibs[parent])
    others = {k: v for k, v in group.items() if k != cid}
    room = 1.0 - new
    rest = sum(others.values())
    if rest == 0:
        if abs(room) > 1e-12:
            return {}
        scaled = {k: 0.0 for k in others}
    else:
        scaled = {k: v * room / rest for k, v in others.items()}
    scaled[cid] = new
    out = {p: dict(g) for p, g in sibs.items()}
    out[parent] = scaled
    return out


def weighted_scoring(
    criteria: Sequence[Mapping[str, Any]], scale_min: float, scale_max: float,
    scores: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]] = (),
    consensus_scores: Sequence[Mapping[str, Any]] = (), normalisation: str = "ratio_to_best",
    aggregation: str = "consensus_if_present", price_reference: Optional[float] = None,
    sensitivity_steps_pp: Sequence[float] = (5, 10, 20), monte_carlo_n: int = 0,
    seed: Optional[int] = None, dirichlet_alpha: float = 50.0,
) -> Dict[str, Any]:
    if normalisation not in NORMALISATIONS:
        raise ScoringError(f"normalisation must be one of {', '.join(NORMALISATIONS)}")
    if aggregation not in AGGREGATIONS:
        raise ScoringError(f"aggregation must be one of {', '.join(AGGREGATIONS)}")
    smin, smax = _num(scale_min, "scale_min"), _num(scale_max, "scale_max")
    if smax <= smin:
        raise ScoringError("scale_max must be above scale_min")
    if monte_carlo_n and seed is None:
        raise ScoringError("monte carlo needs a seed so the result can be reproduced")
    byid, children, sibs = _tree(criteria)
    gweights = _global_weights(byid, children, sibs)
    leaves = [c for c in byid if not children.get(c)]
    gates = [c for c in leaves if byid[c].get("type") == _GATE]
    scored = [c for c in gweights if byid[c].get("type") == "scored"]
    metric_c = [c for c in gweights if byid[c].get("type") in _METRIC_TYPES]
    for c in gweights:
        if byid[c].get("type") not in ("scored",) + _METRIC_TYPES:
            raise ScoringError(f"criterion {c} has type {byid[c].get('type')}; leaves must be scored, price or numeric_metric")

    by_sc: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in scores:
        s, c, e = str(r.get("supplier_id")), str(r.get("criterion_id")), str(r.get("evaluator_id"))
        if c not in byid:
            raise ScoringError(f"a score names criterion {c}, which is not in the criteria")
        v = _num(r.get("score") if not isinstance(r.get("score"), str) else (1 if r.get("score").lower() == "pass" else 0), f"score for {s} on {c}")
        if byid[c].get("type") != _GATE and not (smin <= v <= smax):
            raise ScoringError(f"score {v} for {s} on {c} is outside the scale {smin} to {smax}")
        by_sc.setdefault((s, c), {})[e] = v
    consensus = {(str(r.get("supplier_id")), str(r.get("criterion_id"))): _num(r.get("score"), "a consensus score")
                 for r in consensus_scores}
    for (s, c), v in consensus.items():
        if not (smin <= v <= smax):
            raise ScoringError(f"consensus score {v} for {s} on {c} is outside the scale {smin} to {smax}")
    values: Dict[str, Dict[str, float]] = {}
    for r in metrics:
        values.setdefault(str(r.get("criterion_id")), {})[str(r.get("supplier_id"))] = _num(
            r.get("value"), f"value for {r.get('supplier_id')} on {r.get('criterion_id')}")
    suppliers = sorted({s for s, _ in by_sc} | {s for m in values.values() for s in m})
    if not suppliers:
        raise ScoringError("there are no suppliers to score")

    excluded: List[Dict[str, str]] = []
    alive: List[str] = []
    for s in suppliers:
        failed = None
        for g in gates:
            got = by_sc.get((s, g))
            if not got:
                raise ScoringError(f"no score for {s} on gate {g}")
            if min(got.values()) < 1:
                failed = g
                break
        if failed:
            excluded.append({"supplier_id": s, "reason": f"failed gate {failed}"})
        else:
            alive.append(s)

    warnings: List[str] = []
    aggregated: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    norm: Dict[str, Dict[str, float]] = {s: {} for s in alive}
    for c in scored:
        for s in alive:
            got = by_sc.get((s, c))
            if not got and (s, c) not in consensus:
                raise ScoringError(f"no score for {s} on {c}")
            individual = _aggregate(list(got.values()), "mean" if aggregation == "consensus_if_present" else aggregation) if got else None
            cons = consensus.get((s, c))
            used = cons if (cons is not None and aggregation == "consensus_if_present") else individual
            if used is None:
                raise ScoringError(f"no score for {s} on {c}")
            aggregated.setdefault(s, {})[c] = {"individual": individual, "consensus": cons, "used": used}
            norm[s][c] = (used - smin) / (smax - smin)
    for c in metric_c:
        have = values.get(c, {})
        missing = [s for s in alive if s not in have]
        if missing:
            raise ScoringError(f"no value for {', '.join(missing)} on {c}")
        direction = byid[c].get("direction") or ("lower_better" if byid[c].get("type") == "price" else "higher_better")
        per = _normalise_metric({s: have[s] for s in alive}, normalisation, direction, price_reference, c, warnings)
        for s in alive:
            norm[s][c] = per[s]
    if normalisation in ("min_max", "ratio_to_best") and metric_c:
        warnings.append(f"{normalisation} can reverse the order of two suppliers when a third is added or removed")

    def totals_for(sib_weights) -> Dict[str, float]:
        gw = _global_weights(byid, children, sib_weights)
        return {s: sum(gw[c] * norm[s][c] for c in gw) for s in alive}

    totals = totals_for(sibs)
    ranked = _rank(totals)
    ranking = [{"supplier_id": s, "total": t, "rank": r, "tied": tied,
                "contributions": {c: gweights[c] * norm[s][c] for c in gweights}}
               for s, t, r, tied in ranked]

    # Evaluator and criterion consistency.
    evaluator_flags: List[Dict[str, Any]] = []
    criterion_flags: List[Dict[str, Any]] = []
    evaluators = sorted({e for d in by_sc.values() for e in d})
    for e in evaluators:
        same = total = 0
        for c in scored:
            vals = [by_sc[(s, c)][e] for s in alive if e in by_sc.get((s, c), {})]
            if len(vals) >= 2:
                total += 1
                same += len(set(vals)) == 1
        if total and same / total >= 0.8:
            evaluator_flags.append({"evaluator_id": e, "flag": "identical_scores",
                                    "detail": f"gave every supplier the same score on {same} of {total} criteria"})
    if len(evaluators) >= 3:
        devs = {}
        for e in evaluators:
            d = []
            for (s, c), got in by_sc.items():
                if e in got and c in scored and s in alive and len(got) >= 3:
                    d.append(abs(got[e] - statistics.median(got.values())))
            if d:
                devs[e] = sum(d) / len(d)
        if len(devs) >= 3:
            mu, sd = statistics.mean(devs.values()), statistics.pstdev(devs.values())
            for e, v in devs.items():
                if sd > 0 and abs((v - mu) / sd) > 2:
                    evaluator_flags.append({"evaluator_id": e, "flag": "outlier", "detail": "scores sit far from the panel"})
    for c in scored:
        for s in alive:
            got = by_sc.get((s, c), {})
            if len(got) >= 2 and max(got.values()) - min(got.values()) > 0.4 * (smax - smin):
                criterion_flags.append({"criterion_id": c, "supplier_id": s, "flag": "score_range_over_40_percent",
                                        "detail": "the panel disagrees by more than 40 percent of the scale; moderate"})

    # Sensitivity.
    sensitivity: Dict[str, Any] = {"one_at_a_time": [], "break_even": [], "monte_carlo": None}
    if len(alive) >= 2:
        base_leader = ranked[0][0]
        for parent, group in sibs.items():
            for cid, w0 in group.items():
                for step in sensitivity_steps_pp:
                    for sign in (1, -1):
                        new = min(1.0, max(0.0, w0 + sign * step / 100.0))
                        alt = _with_weight(sibs, parent, cid, new)
                        if not alt:
                            continue
                        r = _rank(totals_for(alt))
                        sensitivity["one_at_a_time"].append({
                            "criterion_id": cid, "step_pp": sign * step, "new_weight": new, "rank_1": r[0][0],
                            "rank_1_changed": r[0][0] != base_leader, "ranks": {s: rk for s, _, rk, _ in r}})
                runner = ranked[1][0]
                swap = None
                if ranked[0][2] != ranked[1][2] and w0 < 1.0:
                    lo, hi = _with_weight(sibs, parent, cid, 0.0), _with_weight(sibs, parent, cid, 1.0)
                    if lo and hi:
                        t0, t1 = totals_for(lo), totals_for(hi)
                        d0, d1 = t0[base_leader] - t0[runner], t1[base_leader] - t1[runner]
                        if d0 * d1 < 0:
                            swap = d0 / (d0 - d1)
                sensitivity["break_even"].append({"criterion_id": cid, "weight_now": w0, "weight_at_swap": swap})
        if monte_carlo_n:
            rng = random.Random(int(seed))
            wins = {s: 0 for s in alive}
            rank_sum = {s: 0.0 for s in alive}
            for _ in range(int(monte_carlo_n)):
                draw = {}
                for parent, group in sibs.items():
                    g = {k: (rng.gammavariate(dirichlet_alpha * v, 1.0) if v > 0 else 0.0) for k, v in group.items()}
                    tot = sum(g.values()) or 1.0
                    draw[parent] = {k: v / tot for k, v in g.items()}
                r = _rank(totals_for(draw))
                wins[r[0][0]] += 1
                for s, _, rk, _ in r:
                    rank_sum[s] += rk
            n = int(monte_carlo_n)
            sensitivity["monte_carlo"] = {
                "n": n, "seed": int(seed), "alpha": dirichlet_alpha,
                "probability_rank_1": {s: wins[s] / n for s in alive},
                "mean_rank": {s: rank_sum[s] / n for s in alive}}

    return {
        "ranking": ranking, "excluded": excluded, "global_weights": gweights,
        "aggregated_scores": aggregated, "normalised_scores": norm,
        "normalisation": normalisation, "aggregation": aggregation,
        "evaluator_flags": evaluator_flags, "criterion_flags": criterion_flags,
        "sensitivity": sensitivity, "warnings": warnings, "not_computed": list(NOT_COMPUTED),
        "method_version": METHOD_VERSION,
    }
