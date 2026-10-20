"""A free-text request in, a proposed report or workflow out (audit rec 8, 2026-10-20).

"Rank these suppliers and email them" used to reach nobody: the UI named the workflow, or a person
did. This reads a request and proposes the report type or workflow it asks for, with the inputs it
could find and the inputs still missing.

It only PROPOSES. Nothing here runs anything; a person confirms. Rules decide, because they are
exact and explainable (the same stance as ``ask_router``): each clause of the request is scored
against every target, and a clause routes only if the best score clears ``MIN_SCORE`` AND beats the
runner-up by ``MARGIN``. Anything else is "clarify" (a near tie, with the candidates) or "none" (with
what the router can do) -- never a guess. A part of the request that routes nowhere is returned in
``unrouted``, never dropped.

Today and the supplier lookup are injected, so a period resolves the same in every run and the router
itself touches no database.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MIN_SCORE = 3
MARGIN = 2


@dataclass(frozen=True)
class Target:
    kind: str                                  # "report" | "workflow"
    id: str
    label: str
    phrases: List[Tuple[str, int]]             # (regex, weight)
    inputs: List[str]                          # required: "supplier_id" | "period" | "deal_id"
    examples: List[str] = field(default_factory=list)


TARGETS: List[Target] = [
    Target("report", "supplier_qbr", "Supplier business review (QBR)",
           [(r"\bqbrs?\b", 5), (r"quarterly\s+business\s+review", 5), (r"\bbusiness\s+review", 4),
            (r"supplier\s+(?:performance\s+)?review", 3), (r"quarterly\s+review", 3)],
           ["supplier_id", "period"],
           ["Prepare a QBR for Acme Ltd for Q3 2026", "I need the quarterly business review for Acme Ltd"]),
    Target("report", "exec_procurement_summary", "Executive procurement summary",
           [(r"\bexec(?:utive)?\s+(?:procurement\s+)?summary", 5), (r"procurement\s+summary", 4),
            (r"spend\s+summary", 3), (r"summary\s+of\s+(?:the\s+)?(?:period|quarter|spend)", 3)],
           ["period"], ["Give me the executive procurement summary for last quarter"]),
    Target("report", "board_paper", "Board paper",
           [(r"board\s+(?:paper|pack)", 5), (r"paper\s+for\s+the\s+board", 5),
            (r"committee\s+paper", 4), (r"approval\s+paper", 4)],
           ["deal_id"], ["Write a board paper for DEALV2-001696"]),
    Target("report", "supplier_criticality_review", "Supplier criticality review",
           [(r"criticality", 5), (r"critical\s+suppliers?", 4), (r"supplier\s+concentration", 4),
            (r"single[-\s]?source", 3)],
           ["period"], ["Run a supplier criticality review for last quarter"]),
    Target("workflow", "supplier_ranking", "Rank suppliers",
           [(r"\brank(?:ing|ings|ed)?\b", 3), (r"\brank\s+(?:the\s+|these\s+|our\s+)?suppliers?", 5),
            (r"shortlist", 4), (r"score\s+(?:the\s+)?suppliers?", 4), (r"best\s+supplier", 4),
            (r"supplier\s+ranking", 5)],
           [], ["Rank the suppliers on this deal"]),
    Target("workflow", "quote_evaluation", "Evaluate quotes",
           [(r"(?:compare|evaluate|assess|analy[sz]e)\s+(?:the\s+|these\s+|all\s+)?(?:\w+\s+)?"
             r"(?:quotes?|bids?|quotations?)", 5), (r"quote\s+(?:comparison|evaluation)", 5),
            (r"which\s+quote", 4), (r"\bbid\s+(?:comparison|evaluation)", 4)],
           [], ["Compare the quotes we received"]),
    Target("workflow", "opportunity_mining", "Find savings opportunities",
           [(r"(?:find|identify|look\s+for|spot|surface)\s+(?:\w+\s+)?(?:savings?|opportunit\w+)", 5),
            (r"savings?\s+opportunit\w+", 5), (r"where\s+(?:can|could)\s+we\s+save", 5),
            (r"opportunity\s+mining", 5), (r"cost\s+saving", 3)],
           [], ["Find savings opportunities in our spend"]),
    Target("workflow", "supplier_interaction", "Contact suppliers (RFQ, negotiation)",
           [(r"\be-?mail\b", 3), (r"\b(?:rfq|rfp|rfi)\b", 5), (r"request\s+(?:a\s+|for\s+)?quot(?:e|ation)s?", 5),
            (r"ask\s+(?:the\s+)?suppliers?", 4), (r"contact\s+(?:the\s+)?suppliers?", 4),
            (r"negotiat\w+", 4), (r"supplier\s+interaction", 5), (r"for\s+a\s+quote", 3)],
           [], ["Email the suppliers and ask for a quote"]),
    Target("workflow", "document_extraction", "Extract data from documents",
           [(r"\bextract\w*", 4),
            (r"(?:read|parse|process|ingest)\s+(?:the\s+)?(?:uploaded\s+)?"
             r"(?:invoices?|documents?|pdfs?|files?|quotes?|purchase\s+orders?)", 4),
            (r"document\s+extraction", 5)],
           [], ["Extract the data from the uploaded invoices"]),
    Target("workflow", "requirements_to_ranking", "From requirements to a supplier ranking",
           # Weighted above supplier_ranking's generic words: a request that starts from requirements
           # contains "rank" too, and the more specific phrase has to win over the words it contains.
           [(r"requirements?\s+(?:to|and|into)\s+rank\w*", 10),
            (r"from\s+(?:the\s+|our\s+)?requirements?\b.*\brank", 10)],
           [], ["Go from the requirements to a supplier ranking"]),
    Target("workflow", "approvals", "Approvals",
           [(r"\bapprovals?\b", 2), (r"\bapprove\b", 3), (r"sign[-\s]?off", 3),
            (r"approval\s+(?:workflow|queue|request)", 5)],
           [], ["Start the approval workflow"]),
]

#: Registered but deliberately not offered by the router (it has its own flow). The drift test names
#: every type and workflow that is in neither this set nor TARGETS.
NOT_ROUTABLE_REPORTS = {"sales_quote"}
NOT_ROUTABLE_WORKFLOWS: set = set()


# --- clauses -------------------------------------------------------------------------------------

_SPLIT = re.compile(r"\s*(?:[.?!;](?=\s|$)|,|\band\s+then\b|\bthen\b|\bafterwards?\b|\bafter\s+that\b|\band\b)\s*",
                    re.I)


def _clauses(text: str) -> List[str]:
    return [c.strip() for c in _SPLIT.split(text) if c and c.strip()]


def _score(clause: str, target: Target) -> Tuple[int, List[str]]:
    total, matched = 0, []
    for pattern, weight in target.phrases:
        m = re.search(pattern, clause, re.I)
        if m:
            total += weight
            matched.append(m.group(0).strip())
    return total, matched


# --- inputs --------------------------------------------------------------------------------------

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]
_ISO = r"\d{4}-\d{2}-\d{2}"


def _quarter(year: int, q: int) -> Tuple[date, date, str]:
    start = date(year, 3 * q - 2, 1)
    end = date(year + (q == 4), 1 if q == 4 else 3 * q + 1, 1) - timedelta(days=1)
    return start, end, f"Q{q} {year}"


def _period(text: str, today: date) -> Tuple[Optional[Dict[str, str]], List[str], bool]:
    """(period inputs, assumptions, impossible). ``impossible`` = a period was given but cannot be one."""
    t = text
    m = re.search(rf"({_ISO})\s*(?:to|until|through|-)\s*({_ISO})", t)
    if m:
        try:
            a, b = date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
        except ValueError:
            return None, [], True
        if a > b:
            return None, [], True
        return _p(a, b, f"{a.isoformat()} to {b.isoformat()}"), [], False
    t = re.sub(_ISO, " ", t)
    m = re.search(r"\bQ([1-4])\s*(?:of\s+|FY\s*)?(\d{4})\b", t, re.I)
    if m:
        a, b, label = _quarter(int(m.group(2)), int(m.group(1)))
        return _p(a, b, label), [], False
    m = re.search(r"\bH([12])\s*(?:of\s+)?(\d{4})\b", t, re.I)
    if m:
        y, h = int(m.group(2)), int(m.group(1))
        return _p(date(y, 1 if h == 1 else 7, 1), date(y, 6, 30) if h == 1 else date(y, 12, 31),
                  f"H{h} {y}"), [], False
    m = re.search(r"\bQ([1-4])\b", t, re.I)
    if m:
        q = int(m.group(1))
        year = today.year if _quarter(today.year, q)[0] <= today else today.year - 1
        a, b, label = _quarter(year, q)
        return _p(a, b, label), [f"Q{q} was taken to mean {label}: no year was given"], False
    cur_q = (today.month - 1) // 3 + 1
    if re.search(r"\blast\s+quarter\b", t, re.I):
        y, q = (today.year, cur_q - 1) if cur_q > 1 else (today.year - 1, 4)
        a, b, label = _quarter(y, q)
        return _p(a, b, label), [], False
    if re.search(r"\bthis\s+quarter\b", t, re.I):
        a, b, label = _quarter(today.year, cur_q)
        return _p(a, b, label), [], False
    if re.search(r"\blast\s+month\b", t, re.I):
        first = today.replace(day=1)
        end = first - timedelta(days=1)
        return _p(end.replace(day=1), end, f"{_MONTHS[end.month - 1]} {end.year}"), [], False
    if re.search(r"\bthis\s+month\b", t, re.I):
        first = today.replace(day=1)
        nxt = date(first.year + (first.month == 12), 1 if first.month == 12 else first.month + 1, 1)
        return _p(first, nxt - timedelta(days=1), f"{_MONTHS[first.month - 1]} {first.year}"), [], False
    if re.search(r"\blast\s+year\b", t, re.I):
        y = today.year - 1
        return _p(date(y, 1, 1), date(y, 12, 31), str(y)), [], False
    if re.search(r"\bthis\s+year\b", t, re.I):
        return _p(date(today.year, 1, 1), date(today.year, 12, 31), str(today.year)), [], False
    m = re.search(r"(?<![\w-])(20\d{2})(?![\w-])", t)
    if m:
        y = int(m.group(1))
        return _p(date(y, 1, 1), date(y, 12, 31), str(y)), [], False
    return None, [], False


def _p(a: date, b: date, label: str) -> Dict[str, str]:
    return {"period_start": a.isoformat(), "period_end": b.isoformat(), "period_label": label}


def _suppliers(text: str, lookup: Optional[Callable[[str], Sequence[Tuple[str, str]]]]):
    """Suppliers named in the text: only whole names or ids, the longest name winning over one it contains."""
    if lookup is None:
        return []
    try:
        candidates = list(lookup(text))
    except Exception:  # noqa: BLE001 - an unreachable lookup is a missing input, not a crash
        logger.exception("intent router: supplier lookup failed")
        return []
    spans = []
    for sid, name in candidates:
        for needle in (name, sid):
            m = re.search(rf"(?<![\w-]){re.escape(str(needle))}(?![\w-])", text, re.I) if needle else None
            if m:
                spans.append((m.start(), m.end(), sid, name))
                break
    kept = [s for s in spans
            if not any(o is not s and o[0] <= s[0] and s[1] <= o[1] and (o[1] - o[0]) > (s[1] - s[0])
                       for o in spans)]
    seen, out = set(), []
    for _a, _b, sid, name in sorted(kept):
        if sid not in seen:
            seen.add(sid)
            out.append({"id": sid, "name": name})
    return out


def _only_names(clause: str, today: date, lookup) -> bool:
    """Does the clause name a supplier, a deal or a period (so it is an input, not a request)?"""
    return bool(_suppliers(clause, lookup) or re.search(r"\bDEAL[A-Z0-9]*-\d+\b", clause, re.I)
                or _period(clause, today)[0])


def _inputs(target: Target, text: str, today: date, lookup) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    missing: List[str] = []
    assumptions: List[str] = []
    candidates: Dict[str, Any] = {}
    needs = target.inputs

    period, assumed, impossible = _period(text, today)
    if "period" in needs:
        if period and not impossible:
            inputs.update(period)
            assumptions += assumed
        else:
            missing.append("period")
    if "supplier_id" in needs:
        found = _suppliers(text, lookup)
        if len(found) == 1:
            inputs["supplier_id"] = found[0]["id"]
        else:
            missing.append("supplier_id")
            if len(found) > 1:
                candidates["supplier_id"] = found
    if "deal_id" in needs:
        m = re.search(r"\b(DEAL[A-Z0-9]*-\d+)\b", text, re.I)
        if m:
            inputs["deal_id"] = m.group(1).upper()
        else:
            missing.append("deal_id")
    cur = re.search(r"\bin\s+(GBP|USD|EUR|INR|AED|NZD|AUD|CAD|CHF|JPY|CNY|SEK|NOK|DKK)\b", text, re.I)
    if cur and target.kind == "report":
        inputs["currency"] = cur.group(1).upper()
    return {"inputs": inputs, "missing": missing, "assumptions": assumptions, "candidates": candidates}


# --- the decision ---------------------------------------------------------------------------------

def _options(limit: Optional[int] = None) -> List[Dict[str, str]]:
    out = [{"kind": t.kind, "id": t.id, "label": t.label} for t in TARGETS]
    return out[:limit] if limit else out


def route(text: Any, *, today: Optional[date] = None,
          supplier_lookup: Optional[Callable[[str], Sequence[Tuple[str, str]]]] = None) -> Dict[str, Any]:
    """Propose the report(s) or workflow(s) a request asks for. Never raises, never executes."""
    today = today or date.today()
    base: Dict[str, Any] = {"decision": "none", "steps": [], "alternatives": [], "options": [],
                            "unrouted": [], "reason": "", "proposal_only": True}
    try:
        if not isinstance(text, str) or not text.strip():
            return {**base, "options": _options(), "reason": "the request is empty"}
        steps: List[Dict[str, Any]] = []
        unrouted: List[str] = []
        alternatives: List[Dict[str, Any]] = []
        seen: set = set()
        for clause in _clauses(text):
            scored = sorted(((*_score(clause, t), t) for t in TARGETS), key=lambda x: -x[0])
            best = scored[0] if scored else (0, [], None)
            second = scored[1][0] if len(scored) > 1 else 0
            if best[0] < MIN_SCORE:
                # A tail that asks for nothing of its own and only names a supplier, a deal or a
                # period ("... and Redstone Works 9", "... and Q3 2026") belongs to the request
                # before it; anything else that routes nowhere is reported, not dropped.
                if steps and best[0] == 0 and (len(clause.split()) <= 3 or _only_names(
                        clause, today, supplier_lookup)):
                    continue
                unrouted.append(clause)
                continue
            if best[0] - second < MARGIN:
                tied = [s for s in scored if best[0] - s[0] < MARGIN and s[0] >= MIN_SCORE]
                alternatives += [{"kind": s[2].kind, "id": s[2].id, "label": s[2].label,
                                  "score": s[0], "matched": s[1]} for s in tied]
                continue
            score, matched, target = best
            if (target.kind, target.id) in seen:
                continue
            seen.add((target.kind, target.id))
            found = _inputs(target, text, today, supplier_lookup)
            steps.append({"kind": target.kind, "id": target.id, "label": target.label,
                          "score": score, "matched": matched, "inputs": found["inputs"],
                          "missing": found["missing"], "ready": not found["missing"],
                          "assumptions": found["assumptions"], "candidates": found["candidates"]})
        if alternatives:
            return {**base, "decision": "clarify", "steps": steps, "alternatives": alternatives,
                    "unrouted": unrouted,
                    "reason": "more than one action fits the request equally well: which did you mean?"}
        if not steps:
            return {**base, "options": _options(), "unrouted": unrouted,
                    "reason": "no request this router can act on was recognised"}
        decision = "plan" if len(steps) > 1 else "route"
        reason = "; ".join(f"{s['label']}: matched {', '.join(repr(m) for m in s['matched'])}"
                           for s in steps)
        return {**base, "decision": decision, "steps": steps, "unrouted": unrouted, "reason": reason}
    except Exception:  # noqa: BLE001
        logger.exception("intent router failed")
        return {**base, "options": _options(), "reason": "the router failed; nothing was proposed"}
