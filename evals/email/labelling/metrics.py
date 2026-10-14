"""The numbers a filled-in set produces. Pure functions: no model, no database, nothing here guesses.

A statistic that cannot be computed (no data, a rater who skipped) is returned as ``None`` and says why;
it is never replaced by a default that would read as a result.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


def consensus(labels_by_rater: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """{id: {"label": majority label or None on a tie, "votes": Counter, "agreement": 0..1}}."""

    ids = sorted({i for per in labels_by_rater.values() for i in per})
    out = {}
    for i in ids:
        votes = Counter(per[i] for per in labels_by_rater.values() if per.get(i))
        top = votes.most_common()
        tie = len(top) > 1 and top[0][1] == top[1][1]
        out[i] = {"label": None if (not top or tie) else top[0][0], "votes": dict(votes),
                  "agreement": (top[0][1] / sum(votes.values())) if votes else 0.0}
    return out


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    """Agreement between two raters beyond chance. None if there is nothing to compare or chance agreement is 1."""

    if not a or len(a) != len(b):
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return None if pe >= 1 else round((po - pe) / (1 - pe), 3)


def intent_vs_labels(key: Dict[str, Dict[str, Any]], cons: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Where my intended family and the team's consensus differ: the family definitions are the suspect, not the model."""

    diffs = [{"id": i, "intended": k["intended"], "team": cons.get(i, {}).get("label"), "kind": k["kind"]}
             for i, k in sorted(key.items()) if cons.get(i, {}).get("label") != k["intended"]]
    return {"compared": len(key), "disagreements": diffs, "agree_rate": round(1 - len(diffs) / len(key), 3) if key else None}


def classifier_report(consensus_labels: Dict[str, Optional[str]], predictions: Dict[str, Dict[str, Any]],
                      key: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Score model predictions against the team's consensus.

    ``predictions[id]`` = {"status": captured|invalid|unavailable, "family": str|None, "asked": bool (a clarification
    question was raised), "lookup_keys": {...}, "latency_s": float}. An item with no consensus (a tie) is excluded and
    counted, not forced into a label. A model that ASKS on an item the team called 'unclear' is correct; asking on a clear
    item is an over-ask and is reported separately from a wrong answer.
    """

    scored = {i: l for i, l in consensus_labels.items() if l is not None}
    excluded = sorted(i for i, l in consensus_labels.items() if l is None)
    right = wrong = over_ask = right_ask = missed_ask = invalid = 0
    confusion: Dict[str, Dict[str, int]] = {}
    for i, label in scored.items():
        p = predictions.get(i) or {"status": "unavailable"}
        if p.get("status") != "captured":
            invalid += 1
            continue
        if label == "unclear":
            right_ask, missed_ask = (right_ask + 1, missed_ask) if p.get("asked") else (right_ask, missed_ask + 1)
            continue
        if p.get("asked"):
            over_ask += 1
            continue
        confusion.setdefault(label, {}).setdefault(p.get("family") or "none", 0)
        confusion[label][p.get("family") or "none"] += 1
        if p.get("family") == label:
            right += 1
        else:
            wrong += 1
    answered = right + wrong
    n_clear = sum(1 for l in scored.values() if l != "unclear")
    n_unclear = len(scored) - n_clear
    invented = 0
    if key:                       # lookup keys the model produced that are in neither the request's expected set nor absent
        for i, p in predictions.items():
            exp = (key.get(i) or {}).get("lookup_keys") or {}
            invented += sum(1 for k, v in (p.get("lookup_keys") or {}).items() if exp.get(k) != v)
    lat = sorted(p["latency_s"] for p in predictions.values() if isinstance(p.get("latency_s"), (int, float)))
    return {
        "scored": len(scored), "excluded_no_consensus": excluded,
        "accuracy_when_answered": round(right / answered, 3) if answered else None,
        "answered": answered, "wrong": wrong, "over_ask": over_ask,
        "unclear_items": n_unclear, "asked_on_unclear": right_ask, "answered_on_unclear": missed_ask,
        "ask_recall_on_unclear": round(right_ask / n_unclear, 3) if n_unclear else None,
        "over_ask_rate": round(over_ask / n_clear, 3) if n_clear else None,
        "not_usable_output": invalid, "confusion": confusion,
        "lookup_keys_wrong_or_invented": invented if key else None,
        "latency_s": ({"median": lat[len(lat) // 2], "max": lat[-1]} if lat else None),
    }


def _rank(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) != len(b) or len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return None if va == 0 or vb == 0 else round(cov / (va * vb) ** 0.5, 3)


def human_mean(scores_by_rater: Dict[str, Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, float]]:
    """{rater: {id: {criterion: score}}} -> {id: {criterion: mean over raters}}."""

    acc: Dict[str, Dict[str, List[int]]] = {}
    for per in scores_by_rater.values():
        for i, crit in per.items():
            for c, v in crit.items():
                acc.setdefault(i, {}).setdefault(c, []).append(v)
    return {i: {c: sum(v) / len(v) for c, v in crit.items()} for i, crit in acc.items()}


def judge_report(human: Dict[str, Dict[str, float]], model: Dict[str, Dict[str, Any]], key: Dict[str, Dict[str, Any]],
                 *, low: float = 2.5) -> Dict[str, Any]:
    """Compare the model judge with the team's scores, and check both against the deliberately flawed controls.

    ``model[id]`` = {"status": scored|invalid|unavailable, "scores": {criterion: n}, "overall": x, "latency_s": s}.
    """

    ids = [i for i in human if (model.get(i) or {}).get("status") == "scored"]
    unusable = sorted(i for i in human if i not in ids)
    diffs: Dict[str, List[float]] = {}
    for i in ids:
        for c, v in human[i].items():
            if c == "overall":
                continue
            m = (model[i].get("scores") or {}).get(c)
            if m is not None:
                diffs.setdefault(c, []).append(abs(m - v))
    all_diffs = [d for ds in diffs.values() for d in ds]
    h_over = [human[i].get("overall") for i in ids if human[i].get("overall") is not None and model[i].get("overall") is not None]
    m_over = [model[i]["overall"] for i in ids if human[i].get("overall") is not None and model[i].get("overall") is not None]

    def split(scores: Dict[str, float]) -> Dict[str, Any]:
        good = [v for i, v in scores.items() if key.get(i, {}).get("kind") == "good"]
        bad = [v for i, v in scores.items() if key.get(i, {}).get("kind") == "flawed"]
        gm, bm = (sum(good) / len(good) if good else None), (sum(bad) / len(bad) if bad else None)
        caught = [v for v in bad if gm is not None and v <= gm - 1]
        return {"good_mean": None if gm is None else round(gm, 2), "flawed_mean": None if bm is None else round(bm, 2),
                "flawed_scored_a_point_or_more_below_good": f"{len(caught)}/{len(bad)}" if bad else None,
                "separates": (gm is not None and bm is not None and gm - bm >= 1)}

    human_overall = {i: human[i]["overall"] for i in human if human[i].get("overall") is not None}
    model_overall = {i: model[i]["overall"] for i in ids if model[i].get("overall") is not None}
    lat = sorted(model[i]["latency_s"] for i in model if isinstance(model[i].get("latency_s"), (int, float)))
    return {
        "compared": len(ids), "not_usable_output": unusable,
        "mean_abs_difference": round(sum(all_diffs) / len(all_diffs), 3) if all_diffs else None,
        "within_one_point": round(sum(d <= 1 for d in all_diffs) / len(all_diffs), 3) if all_diffs else None,
        "mean_abs_difference_by_criterion": {c: round(sum(d) / len(d), 3) for c, d in sorted(diffs.items())},
        "rank_agreement_overall": spearman(h_over, m_over),
        "controls_the_team_found": split(human_overall),
        "controls_the_model_found": split(model_overall),
        "latency_s": ({"median": lat[len(lat) // 2], "max": lat[-1]} if lat else None),
    }
