"""Today's brief — the signals the home page cannot derive from what it fetches.

The brief on the front door is composed in the UI from a registry of READINGS
(ProcurementHome/briefReadings.js). Four of its signals had no backend at all
and shipped as a literal called MOCK_BRIEF_SIGNALS: a savings trend, a spend
trend, contracts expiring undecided, and the caller's own requests. This module
is their source.

WHAT IS NOT HERE, AND WHY. The mock's fifth signal — missedOpportunity, "£86.0K
went by on Northvale freight" — has no measurable counterpart. Every row in
proc.bp_opportunity sits at stage 'identified'; none has ever been retired or
advanced, so there is no lapse, no window and no missed state to read. It is
omitted rather than approximated, and SIGNAL_KEYS is the whole list on purpose.

THE RULE THROUGHOUT: a signal the corpus cannot ground is ABSENT from the
response, never zero. Each reading's build() already returns null for a missing
signal and the brief drops that line, so the brief gets shorter and stays true.
A zero is only ever returned where the zero was actually counted — realised
savings is a measured £0 in this corpus, and that is news, not an absence.

FX. Rates are USD-quoted (rates[ccy] = units of ccy per USD), so
amount(ccy) -> GBP is amount / rates[ccy] * rates["GBP"]. The conversion and
the rate fetch are imported from value_summary_service rather than restated, so
there is one FX convention in the service layer and not two that can drift.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from src.services.db import get_conn
# Single-sourced on purpose — see the FX note above. These are the same two
# helpers GET /spendiq/value-summary converts with, so a figure in the brief
# and the same figure in Value Found cannot be converted two different ways.
from src.services.value_summary_service import _get_rates as get_rates
from src.services.value_summary_service import _to_gbp as to_gbp

log = logging.getLogger(__name__)

# Every signal this endpoint can serve. missedOpportunity is deliberately not
# one of them (see the module docstring).
SIGNAL_KEYS = ("savingsTrend", "spendTrend", "expiringUndecided", "myRequests")

# The words `sources` may carry. OutputSafetyMiddleware withholds any field that
# names an internal route or table, so these stay plain.
SOURCE_STATES = ("ok", "empty", "unavailable", "failed")

# How far ahead a contract counts as expiring. 60 days is the window the brief's
# own reading was written against.
EXPIRY_WINDOW_DAYS = 60

# The savings window. The brief is a DAILY one, and quarters cannot answer "what
# changed" on the day — the current calendar quarter is one day old. Thirty days
# against the thirty before it is the movement a person can act on.
SAVINGS_WINDOW_DAYS = 30

# A hand-off counts as recent news for a week.
HANDOFF_RECENT_DAYS = 7

# How many examples a signal names. The reading prints one line per item, and a
# card listing 43 contracts is the report the brief exists to replace. The COUNT
# stays whole — only the examples are a sample.
MAX_ITEMS = 6

# A trailing quarter holding a sliver of the previous one's invoices is the shape
# of a corpus that stopped being loaded, not of spend that collapsed. Below this
# share of the preceding quarter's row count, a quarter is not a period.
TREND_MIN_SHARE = 0.25

# proc.bp_requirement's status vocabulary, in the words the brief speaks.
OPEN_REQUIREMENT_STAGES = {
    "gathering": "being specified",
    "complete": "ready for sourcing",
}
HANDED_OFF_STATUS = "handed_off"

# proc.bp_value_outcome types that mean money actually landed. 'claimed' is money
# asked for and not yet received; 'claim_dropped' is money given up. Neither is a
# saving, and counting them as one is how a ledger starts lying.
REALISED_OUTCOME_TYPES = ("recovered", "realised_saving", "avoided")


# ── periods ──────────────────────────────────────────────────────────────────

def period_label(day: date) -> str:
    """"Q2 2026". The brief prints this verbatim as the period it is comparing
    against, so the comparison can never be read as a different quarter."""
    return f"Q{(day.month - 1) // 3 + 1} {day.year}"


def pick_trend_periods(buckets: list[dict]) -> Optional[tuple[dict, dict]]:
    """Choose the latest period that genuinely has data, and the one before it.

    Today is 2026-10-01 and the newest invoice in the corpus is 2026-07-18, so
    the current calendar quarter is empty. Comparing it against the last one
    would report "spend is down 100%" — a fact about when loading stopped,
    dressed as a fact about spend. Returns None when there are not two periods
    to compare, rather than inventing a baseline.
    """
    ordered = sorted(buckets, key=lambda b: b["period"])
    if len(ordered) < 2:
        return None
    for i in range(len(ordered) - 1, 0, -1):
        current, prior = ordered[i], ordered[i - 1]
        if current.get("n", 0) >= TREND_MIN_SHARE * max(prior.get("n", 0), 1):
            return current, prior
    return None


# ── spend ────────────────────────────────────────────────────────────────────

def spend_signal(rows: Iterable[dict], rates: Optional[dict]) -> Optional[dict]:
    """Net-of-VAT spend for the latest populated quarter against the one before.

    ``rows`` are per-period, per-currency sums. Five currencies sit in the
    invoice corpus and only ten of 12,408 rows carry a persisted conversion, so
    the totals are built here at one live rate set — and a leg that cannot be
    converted is excluded and NAMED, never folded in at 1:1.
    """
    if not rates:
        return None   # no rate set, no honest cross-currency total

    per_period: dict[date, dict] = {}
    excluded: dict[date, set[str]] = {}
    for row in rows:
        period = row["period"]
        bucket = per_period.setdefault(period, {"period": period, "amount": 0.0, "n": 0})
        bucket["n"] += int(row.get("n") or 0)
        gbp, _ = to_gbp(float(row["amount"] or 0.0), row.get("currency"), rates)
        if gbp is None:
            excluded.setdefault(period, set()).add(row.get("currency") or "unknown")
            continue
        bucket["amount"] += gbp

    picked = pick_trend_periods(list(per_period.values()))
    if picked is None:
        return None
    current, prior = picked

    return {
        "current": round(current["amount"], 2),
        "prior": round(prior["amount"], 2),
        # The UI prints `period` as what it is comparing against, and
        # `currentPeriod` as the row the current figure belongs to. Naming both
        # is what stops "this period" meaning the empty quarter we are in.
        "period": period_label(prior["period"]),
        "currentPeriod": period_label(current["period"]),
        "excludedCurrencies": sorted(
            excluded.get(current["period"], set()) | excluded.get(prior["period"], set())
        ),
        "rateDate": rates.get("_fetched_at"),
    }


# ── contracts expiring with nothing decided ──────────────────────────────────

def _contract_name(row: dict) -> str:
    parts = [p for p in (row.get("supplier_name"), row.get("contract_title")) if p]
    return " · ".join(parts) if parts else str(row.get("contract_id") or "")


def expiring_signal(rows: Iterable[dict], today: date, within_days: int,
                    rates: Optional[dict]) -> Optional[dict]:
    """Contracts running out inside the window with no decision recorded.

    A value that cannot be converted never removes a contract from the COUNT:
    the contract expires either way, and dropping it to keep the arithmetic
    tidy would hide the exposure the reading exists to show. The total then
    says it is partial instead.
    """
    items: list[dict] = []
    total_gbp = 0.0
    converted_any = False
    unconverted_any = False

    for row in rows:
        end = row.get("contract_end_date")
        if not end:
            continue
        if isinstance(end, datetime):
            end = end.date()
        days = (end - today).days
        items.append({
            "name": _contract_name(row),
            "supplier": row.get("supplier_name"),
            "due": f"{end.day} {end:%b}",
            "days": days,
            "autoRenew": str(row.get("auto_renew_flag") or "").strip().lower() in ("yes", "true"),
        })
        value = row.get("total_contract_value")
        if value is None:
            unconverted_any = True
            continue
        gbp, _ = to_gbp(float(value), row.get("currency"), rates) if rates else (None, None)
        if gbp is None:
            unconverted_any = True
        else:
            total_gbp += gbp
            converted_any = True

    if not items:
        return None

    items.sort(key=lambda i: i["days"])
    nearest = items[0]
    return {
        "count": len(items),
        "withinDays": within_days,
        "valueGbp": round(total_gbp, 2) if converted_any else None,
        "valuePartial": converted_any and unconverted_any,
        "nearest": {"name": nearest["supplier"] or nearest["name"], "days": nearest["days"]},
        "items": [{"name": i["name"], "due": i["due"], "days": i["days"]}
                  for i in items[:MAX_ITEMS]],
    }


# ── the caller's own requests ────────────────────────────────────────────────

def identities_for(*, username: Optional[str], email: Optional[str],
                   subject: Optional[str]) -> list[str]:
    """Every spelling of one person that proc.bp_requirement.created_by holds.

    The column carries 'nick', 'p.keerthana' and 'p.keerthana@dhsit.co.uk' side
    by side, so a principal has to match its username, its email and that
    email's local part or a person's own requests belong to nobody.

    Returns [] when there is no principal at all — with auth off, nobody owns
    anybody's requests, which is the only safe reading of "my".
    """
    out: list[str] = []
    for value in (username, email, subject):
        if not value:
            continue
        lowered = str(value).strip().lower()
        if lowered and lowered not in out:
            out.append(lowered)
        if "@" in lowered:
            local = lowered.split("@", 1)[0]
            if local and local not in out:
                out.append(local)
    return out


def my_requests_signal(rows: Iterable[dict], now: datetime) -> Optional[dict]:
    """What the caller raised that is still moving, and what has just moved on."""
    open_items: list[dict] = []
    handed_off: list[dict] = []

    for row in rows:
        status = str(row.get("status") or "").strip().lower()
        if status in OPEN_REQUIREMENT_STAGES:
            open_items.append(row)
        elif status == HANDED_OFF_STATUS:
            moved = row.get("updated_at") or row.get("created_at")
            if moved and (now - _aware(moved)).days <= HANDOFF_RECENT_DAYS:
                handed_off.append(row)

    if not open_items and not handed_off:
        return None

    by_recency = sorted(open_items, key=lambda r: _aware(r["created_at"]), reverse=True)
    signal: dict[str, Any] = {
        "open": len(open_items),
        "items": [{"name": r.get("title"),
                   "stage": OPEN_REQUIREMENT_STAGES[str(r["status"]).lower()]}
                  for r in by_recency],
    }
    if by_recency:
        oldest = min(_aware(r["created_at"]) for r in open_items)
        signal["oldestDays"] = (now - oldest).days
        signal["latest"] = {
            "name": by_recency[0].get("title"),
            "stage": OPEN_REQUIREMENT_STAGES[str(by_recency[0]["status"]).lower()],
        }
    if handed_off:
        signal["approvedRecently"] = len(handed_off)
        signal["approvedPeriod"] = "this week"
        signal["approvedNames"] = [r.get("title") for r in handed_off]
    return signal


def _aware(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)


# ── savings: what actually moved ─────────────────────────────────────────────

def savings_signal(realised: dict, identified: dict, open_total: Optional[float],
                   window_days: int) -> Optional[dict]:
    """Realised and identified savings over the window, and the window before it.

    This is NOT a target, and it is not a percentage dressed as progress. It
    reports what happened: what came back, and what was found. Realised savings
    across this corpus is a measured £0 — that zero is counted, so it is
    reported; the brief can say plainly that nothing was realised and still name
    what was identified. Only when nothing moved on EITHER count is there no
    reading to make, and then the signal is withheld rather than printed as a
    row of zeroes.
    """
    moved = any((realised.get("current"), realised.get("prior"),
                 identified.get("current"), identified.get("prior")))
    if not moved:
        return None
    return {
        "windowDays": window_days,
        "realised": round(float(realised.get("current") or 0.0), 2),
        "realisedPrior": round(float(realised.get("prior") or 0.0), 2),
        "identified": round(float(identified.get("current") or 0.0), 2),
        "identifiedPrior": round(float(identified.get("prior") or 0.0), 2),
        "identifiedCount": int(identified.get("count") or 0),
        "identifiedOpen": round(float(open_total), 2) if open_total is not None else None,
    }


# ── the SQL, one loader per signal ───────────────────────────────────────────

_SPEND_SQL = """
    SELECT date_trunc('quarter', invoice_date)::date AS period,
           currency,
           SUM(invoice_amount)                       AS amount,
           COUNT(*)                                  AS n
      FROM proc.bp_invoice_trgt
     WHERE invoice_date IS NOT NULL
       AND invoice_amount IS NOT NULL
     GROUP BY 1, 2
     ORDER BY 1
"""

# "Not decided" is the absence of a decision RECORD, not a guess from the
# contract's own status: proc.bp_decision is where a decision is written down,
# and its subject_type/subject_id is the generic handle a contract hangs on.
# An auto-renewing contract is not decided either — a silent roll-on is the
# exposure this reading is about, so auto_renew_flag is reported, not filtered.
#
# The supplier join currently matches NOTHING, and is kept rather than dropped.
# bp_contract_master carries 2,545 distinct supplier_ids in an 'S1045' format
# that reconciles to neither supplier register (bp_supplier is keyed
# 'SUP-<Name>', bp_supplier_master 'SI000001'), so a contract cannot be named by
# its supplier today. The reading falls back to the contract's own title and
# invents nobody; the join starts paying the moment those ids are crosswalked.
_EXPIRING_SQL = """
    SELECT c.contract_id, c.contract_title, c.contract_end_date,
           c.total_contract_value, c.currency, c.auto_renew_flag,
           s.supplier_name
      FROM proc.bp_contract_master c
      LEFT JOIN proc.bp_supplier s ON s.supplier_id = c.supplier_id
     WHERE c.contract_end_date BETWEEN CURRENT_DATE AND CURRENT_DATE + %s
       AND COALESCE(c.contract_lifecycle_status, '') ILIKE 'active'
       AND NOT EXISTS (
             SELECT 1 FROM proc.bp_decision d
              WHERE d.subject_type = 'contract'
                AND d.subject_id = c.contract_id::text
           )
     ORDER BY c.contract_end_date
"""

_REQUIREMENTS_SQL = """
    SELECT requirement_id, title, status, created_by, created_at, updated_at
      FROM proc.bp_requirement
     WHERE lower(created_by) = ANY(%s)
     ORDER BY created_at DESC
     LIMIT 200
"""

# Superseded rows are corrections that have been replaced; counting both sides
# double-counts the money. value_ledger records a correction as a new row whose
# supersedes_id points at the one it replaces.
_REALISED_SQL = """
    SELECT COALESCE(SUM(amount_gbp) FILTER (
               WHERE valid_from > CURRENT_DATE - %s), 0)                    AS current,
           COALESCE(SUM(amount_gbp) FILTER (
               WHERE valid_from > CURRENT_DATE - %s
                 AND valid_from <= CURRENT_DATE - %s), 0)                   AS prior
      FROM proc.bp_value_outcome o
     WHERE o.outcome_type = ANY(%s)
       AND o.amount_gbp IS NOT NULL
       AND NOT EXISTS (SELECT 1 FROM proc.bp_value_outcome s
                        WHERE s.supersedes_id = o.outcome_id)
"""

_IDENTIFIED_SQL = """
    SELECT COALESCE(SUM(financial_impact_gbp) FILTER (
               WHERE detected_on > now() - %s::interval), 0)                AS current,
           COUNT(*) FILTER (WHERE detected_on > now() - %s::interval)       AS n,
           COALESCE(SUM(financial_impact_gbp) FILTER (
               WHERE detected_on > now() - %s::interval
                 AND detected_on <= now() - %s::interval), 0)               AS prior,
           COALESCE(SUM(financial_impact_gbp) FILTER (
               WHERE retired_at IS NULL), 0)                                AS open_total
      FROM proc.bp_opportunity
"""


def _rows(cur, sql: str, params: tuple = ()) -> list[dict]:
    cur.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def build_brief_signals(*, username: Optional[str] = None, email: Optional[str] = None,
                        subject: Optional[str] = None, conn=None) -> dict:
    """Assemble every signal the brief can ground, and say what became of each.

    Each signal loads in isolation: one failing query costs its own line and
    nothing else. ``sources`` reports the outcome per signal in plain words —
    never naming a table or a route, which OutputSafetyMiddleware would withhold.
    """
    signals: dict[str, Any] = {}
    sources: dict[str, str] = {}
    today = date.today()
    now = datetime.now(timezone.utc)

    def attempt(key: str, fn) -> None:
        try:
            value = fn()
        except Exception:
            log.exception("brief signals: %s failed", key)
            sources[key] = "failed"
            return
        if value is None:
            sources[key] = "empty"
            return
        signals[key] = value
        sources[key] = "ok"

    rates = get_rates()

    ctx = get_conn() if conn is None else _Borrowed(conn)
    with ctx as active:
        cur = active.cursor()

        attempt("spendTrend", lambda: spend_signal(_rows(cur, _SPEND_SQL), rates))

        attempt("expiringUndecided", lambda: expiring_signal(
            _rows(cur, _EXPIRING_SQL, (EXPIRY_WINDOW_DAYS,)),
            today, EXPIRY_WINDOW_DAYS, rates))

        identities = identities_for(username=username, email=email, subject=subject)
        if identities:
            attempt("myRequests", lambda: my_requests_signal(
                _rows(cur, _REQUIREMENTS_SQL, (identities,)), now))
        else:
            # No principal means no "my". Returning everyone's requests here
            # would put another person's demands in this person's brief.
            sources["myRequests"] = "unavailable"

        attempt("savingsTrend", lambda: _load_savings(cur))

    return {
        "asOf": now.isoformat(),
        "sources": sources,
        # Said outright so a reader of the payload is not left wondering whether
        # the brief simply failed to mention it.
        "missedOpportunity": None,
        **signals,
    }


def _load_savings(cur) -> Optional[dict]:
    window = f"{SAVINGS_WINDOW_DAYS} days"
    realised = _rows(cur, _REALISED_SQL, (
        timedelta(days=SAVINGS_WINDOW_DAYS),
        timedelta(days=SAVINGS_WINDOW_DAYS * 2),
        timedelta(days=SAVINGS_WINDOW_DAYS),
        list(REALISED_OUTCOME_TYPES),
    ))[0]
    identified = _rows(cur, _IDENTIFIED_SQL, (
        window, window, f"{SAVINGS_WINDOW_DAYS * 2} days", window,
    ))[0]
    return savings_signal(
        realised={"current": realised["current"], "prior": realised["prior"]},
        identified={"current": identified["current"], "prior": identified["prior"],
                    "count": identified["n"]},
        open_total=identified["open_total"],
        window_days=SAVINGS_WINDOW_DAYS,
    )


class _Borrowed:
    """Use a caller's connection without closing it on the way out."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False
