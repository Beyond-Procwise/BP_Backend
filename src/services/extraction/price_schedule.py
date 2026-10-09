"""How a proposal's price changes over its term, read from the document and checked.

A three-year order form prices each line by year:

    | Line item                                  | Year 1 (£) | Year 2 (£) | Year 3 (£) | 3-yr subtotal (£) |
    | Platform licence — Enterprise (240 seats)  | 1008000    | 1058400    | 1111320    | 3177720           |

and states, in its header or terms, "Annual uplift (clause 6.2): 3.5% per annum". Extraction kept
the Year 1 figure as the line's cost (the user's ruling: a quote's total is Year 1) and dropped the
rest, so a bid cheapest in Year 1 and dearest over the term looked cheapest, and a schedule rising
~5% a year under a 3.5% clause (Meridia MCP-Q-7740 V3: ~£53,000 over the term) passed unseen.

This reads, from the document's own text (the parser's full_text, so extraction and the backfill
use one path):
  - schedules(text): each line's price per period, and its term total;
  - pricing_terms(text): the term length and the uplift the document states (a % per annum), or
    the index it names (CPI / RPI) when it states no figure;
  - escalation_findings(...): where a line rises faster than the stated uplift, or rises with no
    uplift stated at all — each with the £ it adds over the term.
Nothing here edits a figure the document printed.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

_PERIOD_RE = re.compile(r"^\s*(?:year|yr|y)\s*(\d{1,2})\b", re.IGNORECASE)
_SUBTOTAL_RE = re.compile(r"(\d+)\s*-?\s*(?:yr|year)s?\b.*total|\b(?:sub)?total\b.*\bterm\b|\btcv\b", re.IGNORECASE)
_TCV_ROW_RE = re.compile(r"total contract value|\btcv\b", re.IGNORECASE)
_MONEY_RE = re.compile(r"^-?[£$€]?\s*-?\d[\d,]*(?:\.\d+)?$")


def _cells(row: str) -> list[str]:
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _money(cell: str) -> Optional[float]:
    c = cell.replace(" ", " ").strip()
    if not c or not _MONEY_RE.match(c.replace(" ", "")):
        return None
    return float(re.sub(r"[£$€,\s]", "", c))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def schedules(text: Optional[str]) -> dict[str, Any]:
    """{'lines': [{description, periods: [{period, label, amount}], term_total}], 'stated_tcv'}
    from every table in `text` whose header names two or more periods (Year 1, Year 2, ...)."""
    out: dict[str, Any] = {"lines": [], "stated_tcv": None}
    rows = [ln for ln in (text or "").splitlines() if ln.strip().startswith("|")]
    i = 0
    while i < len(rows):
        head = _cells(rows[i])
        periods = {j: int(m.group(1)) for j, h in enumerate(head) if (m := _PERIOD_RE.match(h))}
        if len(periods) < 2:
            i += 1
            continue
        sub_col = next((j for j, h in enumerate(head) if j not in periods and _SUBTOTAL_RE.search(h)), None)
        desc_col = min(j for j in range(len(head)) if j not in periods and j != sub_col)
        i += 1
        while i < len(rows):
            cells = _cells(rows[i])
            if set("".join(cells)) <= set("-: "):          # the |---|---| separator
                i += 1
                continue
            if len(cells) != len(head) or (len([j for j, h in enumerate(cells) if _PERIOD_RE.match(h)]) >= 2):
                break                                          # a different table, or the next header
            desc = cells[desc_col] if desc_col < len(cells) else ""
            amounts = {j: _money(cells[j]) for j in periods}
            sub = _money(cells[sub_col]) if sub_col is not None else None
            if all(v is None for v in amounts.values()):
                if sub is not None and _TCV_ROW_RE.search(desc):
                    out["stated_tcv"] = sub
                i += 1
                continue
            plist = [{"period": periods[j], "label": head[j].split("(")[0].strip() or f"Year {periods[j]}",
                      "amount": amounts[j]} for j in sorted(periods, key=lambda k: periods[k]) if amounts[j] is not None]
            out["lines"].append({
                "description": desc,
                "periods": plist,
                "term_total": sub if sub is not None else round(sum(p["amount"] for p in plist), 2),
            })
            i += 1
    return out


_TERM_RE = re.compile(r"\bterm\b[^|\n]{0,20}?:\s*(\d{1,3})\s*months?", re.IGNORECASE)
_TERM_YEARS_RE = re.compile(r"\bterm\b[^|\n]{0,20}?:\s*(\d{1,2})\s*years?", re.IGNORECASE)
_UPLIFT_RE = re.compile(r"(?:uplift|escalation|increase)[^|\n%]{0,60}?(\d{1,2}(?:\.\d{1,2})?)\s*%"
                        r"|(\d{1,2}(?:\.\d{1,2})?)\s*%\s*(?:annual|per annum|p\.a\.|a year|yearly)[^|\n]{0,20}?(?:uplift|increase|escalation)",
                        re.IGNORECASE)
_INDEX_RE = re.compile(r"\b(CPIH|CPI|RPI)\b(?:\s*\+\s*(\d{1,2}(?:\.\d{1,2})?)\s*%)?")
_LABEL_RE = re.compile(r"^(?:annual\s+)?(?:uplift|escalation|indexation|price review|term)\b", re.IGNORECASE)
# Reviewer commentary printed on the document ("RISK — ... worsened from CPI+2.0%") is about the
# terms, not the terms: an old rate quoted in a note must never be read as the one stated.
_COMMENTARY_RE = re.compile(r"^\s*(?:RISK|WATCH|NOTE|NB)\b|^\s*[•\-*]", re.IGNORECASE)


def _statements(text: str) -> list[str]:
    """The document's own statements, one per cell — with a label cell joined to the value
    beside it ("Uplift | CPI + 4.5% per annum" -> "Uplift: CPI + 4.5% per annum"). Commentary
    rows are left out."""
    out = []
    for ln in text.splitlines():
        if not ln.strip().startswith("|"):
            if ln.strip() and not _COMMENTARY_RE.match(ln):
                out.append(ln.strip())
            continue
        cells = [c for c in _cells(ln)]
        first = next((c for c in cells if c), "")
        if _COMMENTARY_RE.match(first):
            continue
        for k, c in enumerate(cells):
            if not c:
                continue
            if _LABEL_RE.match(c) and k + 1 < len(cells) and cells[k + 1] and ":" not in c:
                out.append(f"{c}: {cells[k + 1]}")
            else:
                out.append(c)
    return out


def pricing_terms(text: Optional[str]) -> dict[str, Any]:
    """{'term_months', 'uplift_pct', 'uplift_text', 'indexation', 'uplift_conflict'} as the
    document states them. "CPI + 4.5%" is an index plus a margin, recorded as indexation, never as
    a fixed rate. A document stating two different fixed rates has none recorded (uplift_pct None,
    uplift_conflict True): which one governs is a question for a person, not a guess."""
    term, rates, first_text, index = None, [], None, None
    for st in _statements(text or ""):
        if term is None and (m := _TERM_RE.search(st)):
            term = int(m.group(1))
        elif term is None and (m := _TERM_YEARS_RE.search(st)):
            term = int(m.group(1)) * 12
        im = _INDEX_RE.search(st)
        if im and index is None:
            index = im.group(1) + (f" + {im.group(2)}%" if im.group(2) else "")
        fixed = _INDEX_RE.sub("", st) if im else st
        for m in _UPLIFT_RE.finditer(fixed):
            rates.append(float(m.group(1) or m.group(2)))
            first_text = first_text or st
    distinct = sorted(set(rates))
    return {
        "term_months": term,
        "uplift_pct": distinct[0] if len(distinct) == 1 else None,
        "uplift_conflict": len(distinct) > 1,
        "uplift_text": (first_text or "")[:200] or None,
        "indexation": index,
    }


def escalation_findings(lines: list[dict], terms: dict, tolerance_pp: float) -> list[dict]:
    """What the schedule does against what the document states, one finding per document:
      - 'uplift_above_stated': lines rising faster than the stated uplift, with the £ the extra
        rise adds over the term (each period's price against Year 1 compounded at the stated rate);
      - 'price_rises_unstated': lines rising over the term with no uplift and no index stated,
        with the £ the rises add.
    A one-off charge (later periods 0) is not a rise. tolerance_pp is the governed allowance
    (reconciliation_tolerances.uplift_tolerance_pp) for rounding in a printed schedule."""
    stated = terms.get("uplift_pct")
    out = []
    over, unstated = [], []
    for li in lines:
        ps = [p for p in li.get("periods") or [] if p.get("amount") is not None]
        ps.sort(key=lambda p: p["period"])
        if len(ps) < 2 or not ps[0]["amount"] or ps[0]["amount"] <= 0:
            continue
        rises = [(b["amount"] / a["amount"] - 1) * 100 for a, b in zip(ps, ps[1:]) if a["amount"] > 0 and b["amount"] > 0]
        if not rises:
            continue
        top = max(rises)
        base = ps[0]["amount"]
        if stated is not None:
            if top > stated + tolerance_pp:
                extra = sum(p["amount"] - base * (1 + stated / 100) ** (p["period"] - ps[0]["period"]) for p in ps[1:] if p["amount"] > 0)
                over.append({"description": li.get("description"), "actual_pct": round(top, 2), "extra": round(extra, 2)})
        elif not terms.get("indexation") and top > tolerance_pp:
            extra = sum(p["amount"] - base for p in ps[1:] if p["amount"] > 0)
            unstated.append({"description": li.get("description"), "actual_pct": round(top, 2), "extra": round(extra, 2)})
    if over:
        out.append({"issue_type": "uplift_above_stated", "stated_pct": stated, "lines": over,
                    "extra": round(sum(x["extra"] for x in over), 2)})
    if unstated:
        out.append({"issue_type": "price_rises_unstated", "stated_pct": None, "lines": unstated,
                    "extra": round(sum(x["extra"] for x in unstated), 2)})
    return out


def attach_schedules(line_items: list[dict], sched_lines: list[dict]) -> list[dict]:
    """Each line item with `price_schedule` (JSON text) set from the schedule row that names it
    (by its description, case and spacing aside). A line with no schedule row is left as it is."""
    by = {}
    for s in sched_lines:
        by.setdefault(_norm(s["description"]), s)
    out = []
    for li in line_items:
        row = dict(li)
        s = by.get(_norm(row.get("item_description")))
        if s and row.get("price_schedule") is None:
            # JSON text: the promotions copy row values as read (see the migration's note).
            row["price_schedule"] = json.dumps({"periods": s["periods"], "term_total": s["term_total"]})
        out.append(row)
    return out


def finding_notes(f: dict) -> str:
    """The finding in plain words, with its figures."""
    names = "; ".join(f"{x['description']} rises {x['actual_pct']:.1f}% a year" for x in f["lines"])
    if f["issue_type"] == "uplift_above_stated":
        return (f"The schedule rises faster than the stated {f['stated_pct']:g}% uplift: {names}. "
                f"Over the term that adds £{f['extra']:,.0f} against the stated rate.")
    return (f"Prices rise over the term but no uplift or index is stated: {names}. "
            f"The rises add £{f['extra']:,.0f} over the term.")
