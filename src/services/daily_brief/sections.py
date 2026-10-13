"""The readings of the daily brief, as functions of rows. No database here.

One rule runs through all of them, the same one Today's brief already keeps: a
reading with nothing behind it is ABSENT (``None``), never a row of zeros. A
counted zero is different and is kept -- "nothing was closed this week" is news
when something could have been.

Money is only ever added up where it can be added honestly. A figure in a
currency with no rate, or an issue with no readable amount, is left out of the
total and the total says it is partial.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

# How many examples a list names when the settings give that list no number of
# its own. The count beside it is always whole.
MAX_EXAMPLES = 6

Row = Dict[str, Any]


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def age_days(stamp: Any, now: datetime) -> Optional[int]:
    """Whole days since ``stamp``; None when there is no date to count from."""

    if not isinstance(stamp, datetime):
        return None
    return max((now - _aware(stamp)).days, 0)


def _by_type(rows: Iterable[Row], key: str) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[str(row.get(key) or "other")] = counts.get(str(row.get(key) or "other"), 0) + 1
    return [{"type": k, "count": n} for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _money_first(value_of: Callable[[Row], Optional[float]], then: Callable[[Row], Any]):
    """Sort key: largest known amount first, the unvalued after them, by ``then``."""

    def key(row: Row):
        value = value_of(row)
        return (value is None, -(value or 0.0), then(row))
    return key


# ── momentum ─────────────────────────────────────────────────────────────────

def momentum(*, window_days: int, issues: Optional[Row], decisions: Optional[Row],
             recovered: Optional[Row], opportunities: Optional[Row]) -> Optional[dict]:
    """This window against the one before it. Each measure whose source answered."""

    metrics: List[dict] = []

    def add(key: str, source: Optional[Row], current: str, prior: str, unit: str = "count") -> None:
        if source is None or source.get(current) is None:
            return
        cast = float if unit == "gbp" else int
        metrics.append({"key": key, "unit": unit, "current": cast(source[current]),
                        "prior": cast(source.get(prior) or 0)})

    add("issues_closed", issues, "closed_current", "closed_prior")
    add("issues_raised", issues, "raised_current", "raised_prior")
    add("decisions_made", decisions, "made_current", "made_prior")
    add("decisions_raised", decisions, "raised_current", "raised_prior")
    add("recovered", recovered, "current", "prior", unit="gbp")
    add("opportunities_progressed", opportunities, "current", "prior")
    if not metrics:
        return None
    return {"windowDays": window_days, "metrics": metrics}


# ── critical and aged ────────────────────────────────────────────────────────

def aged_issues(issues: Iterable[Row], priced: Dict[Any, Row], *, now: datetime,
                min_days: int, severities: Iterable[str], top_n: int) -> Optional[dict]:
    """Open issues at these grades that have sat for at least ``min_days``.

    ``priced`` maps an issue's id to what the value calculation made of it. An
    issue that calculation marked as the same money as another (``superseded_by``)
    is still COUNTED -- it is still open -- but it is not named and not added
    up, or one overbilled order would fill the list with its own lines.
    """

    grades = {str(s).lower() for s in severities}
    aged = []
    for row in issues:
        days = age_days(row.get("created_at"), now)
        if str(row.get("severity") or "").lower() in grades and days is not None and days >= min_days:
            aged.append({**row, "_days": days, "_p": priced.get(row.get("discrepancy_id")) or {}})
    if not aged:
        return None

    distinct = [r for r in aged if not r["_p"].get("superseded_by")]
    amounts = [r["_p"].get("amount_gbp") for r in distinct]
    known = [a for a in amounts if a is not None]
    ranked = sorted(distinct, key=_money_first(lambda r: r["_p"].get("amount_gbp"),
                                               lambda r: -r["_days"]))
    return {
        "count": len(aged),
        "oldestDays": max(r["_days"] for r in aged),
        "valueGbp": round(sum(known), 2) if known else None,
        "valuePartial": bool(known) and len(known) < len(distinct),
        "items": [{
            "name": r.get("doc_pk_candidate"),
            "issue": r.get("issue_type"),
            "supplier": r.get("supplier_name"),
            "dealId": r.get("deal_id"),
            "days": r["_days"],
            "valueGbp": r["_p"].get("amount_gbp"),
        } for r in ranked[:top_n]],
    }


def aged_decisions(decisions: Iterable[Row], *, now: datetime, min_days: int) -> Optional[dict]:
    """Decisions still open after ``min_days``. Counted by kind, never summed in
    money: what is at stake is recorded per decision in its own currency."""

    aged = []
    for row in decisions:
        days = age_days(row.get("created_at"), now)
        if days is not None and days >= min_days:
            aged.append({**row, "_days": days})
    if not aged:
        return None
    return {"minDays": min_days, "count": len(aged),
            "oldestDays": max(r["_days"] for r in aged),
            "byType": _by_type(aged, "subject_type")}


def critical_aged(*, issues: Optional[dict], decisions: Optional[dict],
                  expired_contracts: Optional[dict], min_days: int) -> Optional[dict]:
    """The three kinds of thing that have been left: issues, decisions, contracts
    past their end date. Each present only when there is one."""

    out: Dict[str, Any] = {"minDays": min_days}
    if issues:
        out["issues"] = issues
    if decisions:
        out["decisions"] = decisions
    if expired_contracts and expired_contracts.get("count"):
        out["expiredContracts"] = expired_contracts
    return out if len(out) > 1 else None


# ── upcoming ─────────────────────────────────────────────────────────────────

def upcoming(*, alerts: Iterable[Any], contracts: Dict[str, Row], name_of: Callable[[Row], str],
             gbp_of: Callable[[Row], Optional[float]], horizon_days: int, top_n: int,
             min_value_gbp: Optional[float]) -> Optional[dict]:
    """Contracts ending inside the horizon with no renewal under way, largest first.

    ``alerts`` are the expiry rule's own (already net of contracts a demand
    covers). With a value floor set, a contract whose value cannot be stated in
    GBP is left out and counted in ``unvalued``: it cannot be shown to clear
    the floor, and it must not vanish without a trace either.
    """

    inside = [a for a in alerts
              if not a.suppressed_by and a.days_to_end is not None
              and 0 <= a.days_to_end <= horizon_days and a.contract_id in contracts]
    rows = [{"alert": a, "row": contracts[a.contract_id], "gbp": gbp_of(contracts[a.contract_id])}
            for a in inside]
    unvalued = sum(1 for r in rows if r["gbp"] is None)
    if min_value_gbp is not None:
        rows = [r for r in rows if r["gbp"] is not None and r["gbp"] >= min_value_gbp]
    if not rows:
        return None

    known = [r["gbp"] for r in rows if r["gbp"] is not None]
    ranked = sorted(rows, key=_money_first(lambda r: r["gbp"], lambda r: r["alert"].days_to_end))
    return {
        "horizonDays": horizon_days,
        "minValueGbp": min_value_gbp,
        "count": len(rows),
        "unvalued": unvalued,
        "valueGbp": round(sum(known), 2) if known else None,
        "valuePartial": bool(known) and len(known) < len(rows),
        "nearestDays": min(r["alert"].days_to_end for r in rows),
        "items": [{
            "name": name_of(r["row"]),
            "due": r["alert"].end_date.isoformat() if isinstance(r["alert"].end_date, date) else None,
            "days": r["alert"].days_to_end,
            "valueGbp": r["gbp"],
        } for r in ranked[:top_n]],
    }


# ── approvals ────────────────────────────────────────────────────────────────

def approvals(decisions: Iterable[Row], *, now: datetime) -> Optional[dict]:
    """Every decision waiting on a person, by kind."""

    rows = list(decisions)
    if not rows:
        return None
    ages = [d for d in (age_days(r.get("created_at"), now) for r in rows) if d is not None]
    return {"count": len(rows), "oldestDays": max(ages) if ages else None,
            "byType": _by_type(rows, "subject_type")}


# ── a buyer's pipeline ───────────────────────────────────────────────────────

def pipeline(issues: Iterable[Row], *, now: datetime, severities: Iterable[str], top_n: int,
             deals_in_scope: Optional[int]) -> Optional[dict]:
    """The reader's deals that have something open on them, worst first."""

    grades = {str(s).lower() for s in severities}
    open_issues = [r for r in issues if str(r.get("severity") or "").lower() in grades]
    deals: Dict[str, dict] = {}
    for row in open_issues:
        deal_id = row.get("deal_id")
        if not deal_id:
            continue
        deal = deals.setdefault(str(deal_id), {
            "dealId": str(deal_id), "name": row.get("deal_name"),
            "supplier": row.get("supplier_name"), "issues": 0, "critical": 0, "oldestDays": 0})
        deal["issues"] += 1
        if str(row.get("severity") or "").lower() == "critical":
            deal["critical"] += 1
        deal["oldestDays"] = max(deal["oldestDays"], age_days(row.get("created_at"), now) or 0)

    if not open_issues and not deals_in_scope:
        return None
    ranked = sorted(deals.values(),
                    key=lambda d: (-d["critical"], -d["issues"], -d["oldestDays"], d["dealId"]))
    return {"deals": deals_in_scope, "dealsNeedingAttention": len(deals),
            "openIssues": len(open_issues), "items": ranked[:top_n]}


# ── supplier emails ──────────────────────────────────────────────────────────

def supplier_emails(replies: Iterable[Row], awaiting_decision: int, *, now: datetime,
                    window_days: int) -> Optional[dict]:
    """Replies suppliers have sent inside the window, newest first."""

    rows = sorted((r for r in replies if isinstance(r.get("responded_at"), datetime)),
                  key=lambda r: _aware(r["responded_at"]), reverse=True)
    if not rows and not awaiting_decision:
        return None
    return {
        "windowDays": window_days,
        "replies": len(rows),
        "awaitingDecision": int(awaiting_decision or 0),
        "items": [{
            "supplier": r.get("supplier_name") or r.get("supplier_id"),
            "subject": r.get("subject"),
            "daysAgo": age_days(r["responded_at"], now),
        } for r in rows[:MAX_EXAMPLES]],
    }


# ── opportunities ────────────────────────────────────────────────────────────

def opportunities(rows: Iterable[Row], *, top_n: int) -> Optional[dict]:
    """Open opportunities nobody has taken up, largest first."""

    found = list(rows)
    if not found:
        return None

    def value(row: Row) -> Optional[float]:
        v = row.get("financial_impact_gbp")
        return float(v) if v is not None else None

    known = [value(r) for r in found if value(r) is not None]
    ranked = sorted(found, key=_money_first(value, lambda r: str(r.get("opportunity_id"))))
    return {
        "count": len(found),
        "valueGbp": round(sum(known), 2) if known else None,
        "valuePartial": bool(known) and len(known) < len(found),
        "items": [{
            "id": r.get("opportunity_ref_id") or r.get("opportunity_id"),
            "name": r.get("item_description") or r.get("detector_type"),
            "kind": r.get("detector_type"),
            "supplier": r.get("supplier_name"),
            "dealId": r.get("deal_id"),
            "valueGbp": value(r),
        } for r in ranked[:top_n]],
    }
