"""Contract expiry buckets: which window an end date falls in, and what to record.

Everything here is pure -- dates and dicts in, dates and dicts out -- so the rules
about boundaries and suppression can be tested without a database.

Bucket edges come from the rule (``proc.bp_rule``, ``bucket_months``), never from
this file. With the shipped edges [3, 6, 9, 12, 18] the labels are
EXPIRED, 0-3, 3-6, 6-9, 9-12, 12-18, and a contract further out than the last
edge gets no alert (``None``).

Boundary rule: an end date exactly ON an edge belongs to the SOONER bucket. A
contract ending on the day that is exactly three months away is in "0-3".
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dateutil.relativedelta import relativedelta

EXPIRED = "EXPIRED"
NO_END_DATE = "NO_END_DATE"


def add_months(as_of: date, months: int) -> date:
    """as_of plus whole calendar months; a day that does not exist is clamped
    to the month's last day (31 Aug + 6 months = 28 Feb, or 29 in a leap year)."""
    return as_of + relativedelta(months=months)


def edges(bucket_months: Sequence[int]) -> List[int]:
    clean = sorted({int(m) for m in bucket_months})
    if not clean or clean[0] <= 0:
        raise ValueError(f"bucket_months must be positive whole months, got {list(bucket_months)!r}")
    return clean


def bucket_labels(bucket_months: Sequence[int]) -> List[str]:
    e = edges(bucket_months)
    return [f"{lo}-{hi}" for lo, hi in zip([0] + e[:-1], e)]


def classify(end_date: date, as_of: date, bucket_months: Sequence[int]) -> Optional[str]:
    """EXPIRED, a bucket label, or None when the end is beyond the last edge."""
    if end_date < as_of:
        return EXPIRED
    e = edges(bucket_months)
    lows = [0] + e[:-1]
    for lo, hi in zip(lows, e):
        if end_date <= add_months(as_of, hi):
            return f"{lo}-{hi}"
    return None


@dataclass(frozen=True)
class Desired:
    """The alert a contract should carry right now."""
    contract_id: str
    bucket: str
    end_date: Optional[date]
    days_to_end: Optional[int]
    suppressed_by: Optional[str]      # demand id when an active demand covers it


@dataclass
class Plan:
    insert: List[Desired]             # no row yet -> a NEW alert fires
    reopen: List[Tuple[int, Desired]]  # was suppressed, demand gone -> fires again
    suppress: List[Tuple[int, Desired]]
    touch: List[int]                  # still true, unchanged
    clear: List[int]                  # no longer true


def is_active_demand(status: Optional[str], inactive: Iterable[str]) -> bool:
    """A demand item suppresses an alert only while it is live. Closed, cancelled,
    rejected, completed (and drafts, which nobody has submitted) do not count.
    An item with NO status is treated as active -- it exists and nothing says it ended."""
    return (status or "").strip().lower() not in {s.strip().lower() for s in inactive}


def desired_alerts(
    contracts: Iterable[Dict],
    active_demand_by_contract: Dict[str, str],
    as_of: date,
    cfg: Dict,
) -> List[Desired]:
    """What each in-scope contract should be alerting as, per the rule's config."""
    months = cfg["bucket_months"]
    status_wanted = str(cfg.get("lifecycle_status", "active")).lower()
    out: List[Desired] = []
    for c in contracts:
        if str(c.get("contract_lifecycle_status") or "").strip().lower() != status_wanted:
            continue
        cid = str(c["contract_id"])
        end = c.get("contract_end_date")
        if end is None:
            if not cfg.get("flag_missing_end_date", True):
                continue
            bucket, days = NO_END_DATE, None
        else:
            bucket = classify(end, as_of, months)
            days = (end - as_of).days
            if bucket is None:
                continue
            if bucket == EXPIRED and not cfg.get("alert_expired", True):
                continue
        out.append(Desired(cid, bucket, end, days, active_demand_by_contract.get(cid)))
    return out


def plan_changes(desired: Iterable[Desired], existing: Iterable[Dict]) -> Plan:
    """Compare what should be true with the open/suppressed rows already stored.

    A row is identified by (contract, bucket, end date) -- the table's unique key
    -- so a contract that moves bucket, or whose end date is amended, gets a new
    alert and the old one is cleared, while one that stays put never re-fires.
    """
    key = lambda cid, b, e: (cid, b, e)  # noqa: E731
    have = {key(r["contract_id"], r["bucket"], r["end_date"]): r for r in existing}
    want = {key(d.contract_id, d.bucket, d.end_date): d for d in desired}
    plan = Plan([], [], [], [], [])
    for k, d in want.items():
        row = have.get(k)
        if row is None:
            plan.insert.append(d)
        elif d.suppressed_by and row["status"] == "open":
            plan.suppress.append((row["alert_id"], d))
        elif not d.suppressed_by and row["status"] == "suppressed":
            plan.reopen.append((row["alert_id"], d))
        else:
            plan.touch.append(row["alert_id"])
    for k, row in have.items():
        if k not in want:
            plan.clear.append(row["alert_id"])
    return plan
