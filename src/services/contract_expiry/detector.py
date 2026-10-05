"""The contract_expiry_bucket_check rule: read the rule, evaluate, record alerts.

The rule's thresholds live in ``proc.bp_rule`` (``conditions``). This module reads
them through the RuleBook, so changing the edges is an UPDATE, not a deploy. If
the rule row is missing, ``run`` raises: a sweep that silently watches nothing is
the failure this exists to remove.

Alerts are recorded in ``proc.bp_contract_expiry_alert``; see the migration for
what makes "once per bucket" a database guarantee.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from src.engines.rule_book import RuleBook
from src.services.db import get_conn

from .buckets import Desired, desired_alerts, is_active_demand, plan_changes

log = logging.getLogger(__name__)

RULE_SLUG = "contract_expiry_bucket_check"

# The brief names the supplier. Same two-route join the brief used before it read
# buckets (see tests/services/test_brief_signals_contract_supplier.py for why it
# is kept though it currently matches nothing). A crosswalk can in principle map
# one id to several suppliers, so rows are de-duplicated by contract after load.
_CONTRACTS_WITH_SUPPLIER_SQL = """
    SELECT c.contract_id, c.supplier_id, c.contract_title, c.contract_end_date,
           c.contract_lifecycle_status, c.total_contract_value, c.currency,
           c.auto_renew_flag, c.renewal_term, c.spend_category,
           COALESCE(s.supplier_name, sx.supplier_name) AS supplier_name
      FROM proc.bp_contract_master c
      LEFT JOIN proc.bp_supplier s ON s.supplier_id = c.supplier_id
      LEFT JOIN proc.bp_supplier_id_crosswalk x
             ON x.uicanvas_supplier_id = c.supplier_id
      LEFT JOIN proc.bp_supplier sx ON sx.supplier_id = x.bp_supplier_id
"""

# The demand item names its contract in payload->>key. The key is the rule's
# (demand_contract_key), so where the link lives is configuration, not code.
_DEMAND_SQL = """
    SELECT demand_id, status, payload->>%s AS contract_id
      FROM proc.bp_demand
     WHERE payload->>%s IS NOT NULL
"""

_OPEN_SQL = """
    SELECT alert_id, contract_id, bucket, end_date, status
      FROM proc.bp_contract_expiry_alert
     WHERE status IN ('open', 'suppressed')
"""

_UPSERT_SQL = """
    INSERT INTO proc.bp_contract_expiry_alert
        (contract_id, bucket, end_date, status, suppressed_by, days_to_end, rule_id, rule_version)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (contract_id, bucket, COALESCE(end_date, DATE '0001-01-01'))
    DO UPDATE SET status = EXCLUDED.status, suppressed_by = EXCLUDED.suppressed_by,
                  days_to_end = EXCLUDED.days_to_end, cleared_at = NULL,
                  last_evaluated_at = now(), first_fired_at = now()
    RETURNING alert_id
"""


def load_rule(rule_book: Optional[RuleBook] = None):
    book = rule_book or RuleBook(connection_factory=get_conn)
    rule = book.rule_for(RULE_SLUG)
    if rule is None:
        raise RuntimeError(f"proc.bp_rule has no active rule {RULE_SLUG!r}")
    return rule


def _rows(cur) -> List[Dict[str, Any]]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def evaluate(cur, as_of: Optional[date] = None, *, conditions: Optional[Dict[str, Any]] = None,
             rule_book: Optional[RuleBook] = None) -> Dict[str, Any]:
    """Read-only: what every contract should be alerting as, right now.

    Writes nothing, so a page can call it on every load and never go stale. The
    stored alerts (``run``) are the record of what FIRED; this is what is TRUE.
    """
    rule = load_rule(rule_book)
    cfg = {**rule.conditions, **(conditions or {})}
    as_of = as_of or date.today()
    key = cfg.get("demand_contract_key", "contract_id")
    inactive = cfg.get("inactive_demand_statuses", [])

    cur.execute(_CONTRACTS_WITH_SUPPLIER_SQL)
    contracts: Dict[str, Dict[str, Any]] = {}
    for row in _rows(cur):
        contracts.setdefault(str(row["contract_id"]), row)
    cur.execute(_DEMAND_SQL, (key, key))
    active_demand: Dict[str, str] = {}
    for d in _rows(cur):
        if is_active_demand(d["status"], inactive):
            active_demand.setdefault(str(d["contract_id"]), d["demand_id"])
    return {
        "rule": rule, "cfg": cfg, "as_of": as_of, "contracts": contracts,
        "desired": desired_alerts(contracts.values(), active_demand, as_of, cfg),
    }


def run(as_of: Optional[date] = None, *, conditions: Optional[Dict[str, Any]] = None,
        rule_book: Optional[RuleBook] = None, dry_run: bool = False) -> Dict[str, Any]:
    """Evaluate every contract and bring the alert table in line.

    Returns ``fired`` -- the alerts that are NEW this run (first time in a bucket,
    or re-entering after a demand closed) -- plus counts. ``dry_run`` computes and
    returns the same answer without writing.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        ev = evaluate(cur, as_of, conditions=conditions, rule_book=rule_book)
        rule, cfg, as_of = ev["rule"], ev["cfg"], ev["as_of"]
        contracts, desired = ev["contracts"], ev["desired"]
        cur.execute(_OPEN_SQL)
        existing = _rows(cur)
        plan = plan_changes(desired, existing)

        if not dry_run:
            def upsert(d: Desired):
                status = "suppressed" if d.suppressed_by else "open"
                cur.execute(_UPSERT_SQL, (d.contract_id, d.bucket, d.end_date, status,
                                          d.suppressed_by, d.days_to_end, rule.rule_id, rule.version))
            for d in plan.insert:
                upsert(d)
            for _, d in plan.reopen:
                upsert(d)
            for aid, d in plan.suppress:
                cur.execute("UPDATE proc.bp_contract_expiry_alert SET status='suppressed', suppressed_by=%s, "
                            "days_to_end=%s, last_evaluated_at=now() WHERE alert_id=%s",
                            (d.suppressed_by, d.days_to_end, aid))
            if plan.touch:
                cur.execute("UPDATE proc.bp_contract_expiry_alert SET last_evaluated_at=now() "
                            "WHERE alert_id = ANY(%s)", (plan.touch,))
            if plan.clear:
                cur.execute("UPDATE proc.bp_contract_expiry_alert SET status='cleared', cleared_at=now() "
                            "WHERE alert_id = ANY(%s)", (plan.clear,))
        fired = [d for d in plan.insert if not d.suppressed_by] + [d for _, d in plan.reopen]

    return {
        "as_of": as_of.isoformat(),
        "rule_id": rule.rule_id,
        "bucket_months": cfg["bucket_months"],
        "evaluated": len(contracts),
        "fired": [{"contract": contracts[d.contract_id], "desired": d} for d in fired],
        "counts": {"new": len(plan.insert), "reopened": len(plan.reopen),
                   "suppressed": len(plan.suppress), "unchanged": len(plan.touch),
                   "cleared": len(plan.clear)},
        "dry_run": dry_run,
    }
