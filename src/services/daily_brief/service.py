"""One person's daily brief: who they are decides the view, the view decides the rest.

The view comes from the saved settings (``settings``), the reader's records
from the same assignment reports already use (``report_data.scope``): a Buyer
sees the deals of the buyer codes an Admin gave them, and somebody given none
sees NO records -- never everyone's. A section that needs that assignment and
has none says ``unassigned`` rather than showing an empty pipeline, because
"nothing needs you" and "we do not know which deals are yours" are different
things to be told.

Each section loads in isolation, as Today's brief signals already do: one
failing query costs its own section and nothing else, and the payload says
which.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from src.services import brief_signals_service as signals
from src.services import governed_limits, rbac
from src.services import value_summary_service as value_summary
from src.services.contract_expiry import detector as expiry_detector
from src.services.contract_expiry.buckets import EXPIRED
from src.services.daily_brief import sections, settings
from src.services.db import get_conn
from src.services.report_data import scope as report_scope

log = logging.getLogger(__name__)

# The words a section's state may carry. Plain on purpose: OutputSafetyMiddleware
# withholds any field that names an internal route or table.
STATES = ("ok", "empty", "unassigned", "unavailable", "failed")

#: Sections that read the reader's own deals, so need a buyer assignment when
#: the view covers only the reader's records.
_NEEDS_ASSIGNMENT = ("critical_aged", "pipeline", "supplier_emails", "opportunities")

# One row per document, so an issue is tied to one deal and its buyer. Verified
# 2026-10-13: no document number sits on more than one deal (0 of 38,549).
#
# MATERIALIZED is load-bearing. Joined as an ordinary subquery, the planner
# re-plans around the case-insensitive severity test and this takes 5.8s on
# bp_testdb; computing the document map once takes 0.23s for the same rows.
_ISSUES_SQL = """
    WITH dd AS MATERIALIZED (
        SELECT DISTINCT ON (doc_pk) doc_pk, deal_id, deal_name, buyer_id, supplier_name
          FROM proc.bp_deal_documents
         ORDER BY doc_pk, deal_id)
    SELECT x.discrepancy_id, x.doc_type, x.doc_pk_candidate, x.issue_type, x.severity,
           x.created_at, dd.deal_id, dd.deal_name, dd.buyer_id, dd.supplier_name
      FROM proc.bp_extraction_discrepancy x
      LEFT JOIN dd ON dd.doc_pk = x.doc_pk_candidate
     WHERE x.status = 'open'
       AND lower(x.severity) = ANY(%s)
"""

_ISSUE_MOVEMENT_SQL = """
    SELECT count(*) FILTER (WHERE created_at > now() - %(w)s)              AS raised_current,
           count(*) FILTER (WHERE created_at <= now() - %(w)s
                              AND created_at > now() - %(w2)s)             AS raised_prior,
           count(*) FILTER (WHERE status = 'resolved'
                              AND resolved_at > now() - %(w)s)             AS closed_current,
           count(*) FILTER (WHERE status = 'resolved'
                              AND resolved_at <= now() - %(w)s
                              AND resolved_at > now() - %(w2)s)            AS closed_prior
      FROM proc.bp_extraction_discrepancy
     WHERE lower(severity) = ANY(%(grades)s)
"""

_DECISION_MOVEMENT_SQL = """
    SELECT count(*) FILTER (WHERE created_at > now() - %(w)s)              AS raised_current,
           count(*) FILTER (WHERE created_at <= now() - %(w)s
                              AND created_at > now() - %(w2)s)             AS raised_prior,
           count(*) FILTER (WHERE actioned_at > now() - %(w)s)             AS made_current,
           count(*) FILTER (WHERE actioned_at <= now() - %(w)s
                              AND actioned_at > now() - %(w2)s)            AS made_prior
      FROM proc.bp_decision
"""

# An opportunity has "progressed" when somebody moved it off the stage it was
# found at. Re-detection does not touch stage_updated_at, so this cannot be
# inflated by the detector running again.
_OPPORTUNITY_MOVEMENT_SQL = """
    SELECT count(*) FILTER (WHERE stage_updated_at > now() - %(w)s)        AS current,
           count(*) FILTER (WHERE stage_updated_at <= now() - %(w)s
                              AND stage_updated_at > now() - %(w2)s)       AS prior
      FROM proc.bp_opportunity
     WHERE stage <> 'identified'
"""

_OPEN_DECISIONS_SQL = """
    SELECT decision_id, subject_type, subject_id, supplier_id, created_at
      FROM proc.bp_decision
     WHERE status = 'open'
"""

_DEALS_IN_SCOPE_SQL = "SELECT count(*) AS n FROM proc.bp_deal_overview WHERE buyer_id = ANY(%s)"
_ALL_DEALS_SQL = "SELECT count(*) AS n FROM proc.bp_deal_overview"

_MY_SUPPLIERS_SQL = """
    SELECT DISTINCT supplier_id, supplier_name
      FROM proc.bp_deal_overview
     WHERE buyer_id = ANY(%s)
"""

# Who sent an email is not recorded (draft_rfq_emails.requested_by is empty on
# every row, verified 2026-10-13), so a reply is tied to a buyer through the
# supplier: it is "a reply from a supplier on one of your deals".
_REPLIES_SQL = """
    SELECT t.unique_id, t.supplier_id, t.subject, t.responded_at
      FROM proc.workflow_email_tracking t
     WHERE t.responded_at > now() - %s
"""

_SUPPLIER_NAMES_SQL = "SELECT supplier_id, supplier_name FROM proc.bp_supplier WHERE supplier_id = ANY(%s)"

_OPPORTUNITIES_SQL = """
    SELECT o.opportunity_id, o.opportunity_ref_id, o.detector_type, o.supplier_name,
           o.item_description, o.financial_impact_gbp, o.deal_id, d.buyer_id
      FROM proc.bp_opportunity o
      LEFT JOIN proc.bp_deal_overview d ON d.deal_id = o.deal_id
     WHERE o.retired_at IS NULL
       AND o.stage = 'identified'
"""


def _rows(cur, sql: str, params: Any = ()) -> List[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _grades(values) -> List[str]:
    return sorted({str(v).strip().lower() for v in values})


def _priced(cur, rates: Optional[dict]) -> Dict[Any, dict]:
    """What each issue is worth, by id, through the SAME steps Value Found takes
    (classify, convert, then the rules that stop one sum being counted twice), so
    a figure in the brief cannot differ from the same issue's figure there."""

    findings = []
    for row in value_summary._load_discrepancies(cur):
        finding = value_summary.classify_discrepancy(row)
        if finding is not None:
            finding["_id"] = row.get("discrepancy_id")
            findings.append(value_summary._apply_discrepancy_fx(finding, rates))
    return {f["_id"]: f for f in value_summary.supersede_overlaps(findings)}


class _Brief:
    """One build: the settings, the reader, and each shared read done at most once."""

    def __init__(self, cur, rules: Dict[str, Any], *, buyers: Optional[tuple],
                 principal: Any, now: datetime, today: date) -> None:
        self.cur, self.rules, self.buyers = cur, rules, buyers
        self.principal, self.now, self.today = principal, now, today
        self._cache: Dict[str, Any] = {}

    def _once(self, key: str, load: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = load()
        return self._cache[key]

    def _mine(self, rows: List[dict]) -> List[dict]:
        if self.buyers is None:
            return rows
        return [r for r in rows if r.get("buyer_id") in self.buyers]

    # shared reads ------------------------------------------------------------

    def rates(self) -> Optional[dict]:
        return self._once("rates", signals.get_rates)

    def issues(self) -> List[dict]:
        grades = _grades([*self.rules["issue_severities"], *self.rules["critical_aged_severities"]])
        return self._mine(self._once("issues", lambda: _rows(self.cur, _ISSUES_SQL, (grades,))))

    def open_decisions(self) -> List[dict]:
        return self._once("decisions", lambda: _rows(self.cur, _OPEN_DECISIONS_SQL))

    def expiry(self) -> dict:
        return self._once("expiry", lambda: expiry_detector.evaluate(
            self.cur, self.today, with_supplier=True))

    def my_suppliers(self) -> Optional[set]:
        if self.buyers is None:
            return None

        def load() -> set:
            names: set = set()
            for row in _rows(self.cur, _MY_SUPPLIERS_SQL, (list(self.buyers),)):
                names.update(str(v).strip().lower() for v in row.values() if v)
            return names
        return self._once("suppliers", load)

    def _windows(self, days: int) -> dict:
        return {"w": timedelta(days=days), "w2": timedelta(days=days * 2)}

    # sections ----------------------------------------------------------------

    def momentum(self) -> Optional[dict]:
        days = self.rules["momentum_window_days"]
        w = self._windows(days)
        issues = _rows(self.cur, _ISSUE_MOVEMENT_SQL,
                       {**w, "grades": _grades(self.rules["issue_severities"])})[0]
        decisions = _rows(self.cur, _DECISION_MOVEMENT_SQL, w)[0]
        recovered = _rows(self.cur, signals._REALISED_SQL, (
            w["w"], w["w2"], w["w"], list(signals.REALISED_OUTCOME_TYPES)))[0]
        moved = _rows(self.cur, _OPPORTUNITY_MOVEMENT_SQL, w)[0]
        return sections.momentum(window_days=days, issues=issues, decisions=decisions,
                                 recovered=recovered, opportunities=moved)

    def critical_aged(self) -> Optional[dict]:
        rules = self.rules
        issues = sections.aged_issues(
            self.issues(), _priced(self.cur, self.rates()), now=self.now,
            min_days=rules["critical_aged_min_days"],
            severities=rules["critical_aged_severities"], top_n=rules["critical_aged_top_n"])
        decisions = expired = None
        if self.buyers is None:
            # Neither a decision nor a contract carries a buyer, so in a view of the
            # reader's own records they are left out rather than shown as theirs.
            decisions = sections.aged_decisions(
                self.open_decisions(), now=self.now, min_days=rules["approval_aged_min_days"])
            evaluation = self.expiry()
            late = [evaluation["contracts"][a.contract_id] for a in evaluation["desired"]
                    if a.bucket == EXPIRED and not a.suppressed_by]
            value, partial = signals._total_gbp(late, self.rates())
            expired = {"count": len(late), "valueGbp": value, "valuePartial": partial}
        return sections.critical_aged(issues=issues, decisions=decisions,
                                      expired_contracts=expired,
                                      min_days=rules["critical_aged_min_days"])

    def upcoming(self) -> Optional[dict]:
        evaluation, rates = self.expiry(), self.rates()

        def gbp_of(row: dict) -> Optional[float]:
            if row.get("total_contract_value") is None:
                return None
            return signals.to_gbp(float(row["total_contract_value"]), row.get("currency"), rates)[0]

        return sections.upcoming(
            alerts=evaluation["desired"], contracts=evaluation["contracts"],
            name_of=signals._contract_name, gbp_of=gbp_of,
            horizon_days=self.rules["upcoming_horizon_days"], top_n=self.rules["upcoming_top_n"],
            min_value_gbp=self.rules["upcoming_min_value_gbp"])

    def approvals(self) -> Optional[dict]:
        return sections.approvals(self.open_decisions(), now=self.now)

    def pipeline(self) -> Optional[dict]:
        if self.buyers is None:
            deals = _rows(self.cur, _ALL_DEALS_SQL)[0]["n"]
        else:
            deals = _rows(self.cur, _DEALS_IN_SCOPE_SQL, (list(self.buyers),))[0]["n"]
        return sections.pipeline(self.issues(), now=self.now,
                                 severities=self.rules["issue_severities"],
                                 top_n=self.rules["pipeline_top_n"], deals_in_scope=int(deals))

    def supplier_emails(self) -> Optional[dict]:
        days = self.rules["supplier_email_window_days"]
        replies = _rows(self.cur, _REPLIES_SQL, (timedelta(days=days),))
        waiting = [d for d in self.open_decisions() if d.get("subject_type") == "email_reply"]
        mine = self.my_suppliers()
        if mine is not None:
            def on_my_deals(row: dict) -> bool:
                return str(row.get("supplier_id") or "").strip().lower() in mine
            replies = [r for r in replies if on_my_deals(r)]
            waiting = [d for d in waiting if on_my_deals(d)]
        ids = sorted({str(r["supplier_id"]) for r in replies if r.get("supplier_id")})
        if ids:
            names = {r["supplier_id"]: r["supplier_name"]
                     for r in _rows(self.cur, _SUPPLIER_NAMES_SQL, (ids,))}
            replies = [{**r, "supplier_name": names.get(r.get("supplier_id"))} for r in replies]
        return sections.supplier_emails(replies, len(waiting), now=self.now, window_days=days)

    def opportunities(self) -> Optional[dict]:
        rows = self._mine(_rows(self.cur, _OPPORTUNITIES_SQL))
        return sections.opportunities(rows, top_n=self.rules["opportunity_top_n"])

    def my_requests(self) -> Optional[dict]:
        identities = signals.identities_for(
            username=getattr(self.principal, "username", None),
            email=getattr(self.principal, "email", None),
            subject=getattr(self.principal, "subject", None))
        if not identities:
            return None
        return signals.my_requests_signal(
            _rows(self.cur, signals._REQUIREMENTS_SQL, (identities,)), self.now)


def reader(principal: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
    """Who is reading: their role, the view the settings give them and why, and
    which records are theirs. ``buyers`` is None when they see everyone's."""

    scope = report_scope.resolve(principal)
    identities = [getattr(principal, name, None) for name in ("subject", "email", "username")]
    view, basis = settings.resolve_view(
        rules, role=scope.role, groups=rbac.role_basis(principal)["groups"],
        identities=[i for i in identities if i])
    own = rules["views"][view]["scope"] == settings.SCOPE_OWN and not scope.all_rows
    return {"role": scope.role, "view": view, "basis": basis,
            "buyers": tuple(scope.buyers) if own else None}


def build(principal: Any, *, conn=None) -> dict:
    """The brief for ``principal``. Raises ``BriefSettingsUnavailable`` when the
    settings cannot be read: no view is guessed for anybody."""

    rules = settings.load()
    who = reader(principal, rules)
    buyers = who["buyers"]
    unassigned = buyers is not None and not buyers
    now = datetime.now(timezone.utc)
    out: List[dict] = []

    ctx = get_conn() if conn is None else signals._Borrowed(conn)
    with ctx as active:
        brief = _Brief(active.cursor(), rules, buyers=buyers, principal=principal,
                       now=now, today=date.today())
        for key in rules["views"][who["view"]]["sections"]:
            if unassigned and key in _NEEDS_ASSIGNMENT:
                out.append({"key": key, "state": "unassigned"})
                continue
            if key == "my_requests" and principal is None:
                # No sign-in means no "my"; everyone's requests are nobody's brief.
                out.append({"key": key, "state": "unavailable"})
                continue
            try:
                data = getattr(brief, key)()
            except Exception:
                log.exception("daily brief: %s failed", key)
                out.append({"key": key, "state": "failed"})
                continue
            out.append({"key": key, "state": "ok", "data": data} if data is not None
                       else {"key": key, "state": "empty"})

    return {
        "asOf": now.isoformat(),
        "view": who["view"],
        "viewBasis": who["basis"],
        "role": who["role"],
        "scope": settings.SCOPE_OWN if buyers is not None else settings.SCOPE_EVERYONE,
        "assigned": not unassigned,
        # Whether to OFFER the settings. The settings endpoints ask the gate themselves.
        "canConfigure": rbac.may(who["role"], "configure"),
        "sections": out,
    }


# ── saving the settings ──────────────────────────────────────────────────────

class SettingsConflict(RuntimeError):
    """Somebody else saved the settings first. Nothing was written."""


def read_settings() -> Dict[str, Any]:
    """The saved settings with their version, valid or not, for the settings screen.

    Unlike ``settings.load`` this does not refuse invalid settings: the screen
    is where they get repaired, so it must be able to show them and say what is
    wrong.
    """

    with get_conn() as conn:
        row = _settings_row(conn.cursor())
    rules = row["rules"]
    return {"rules": rules, "version": row["version"],
            "lastModifiedBy": row["last_modified_by"],
            "lastModifiedAt": row["last_modified_date"].isoformat()
            if isinstance(row["last_modified_date"], datetime) else None,
            "problems": settings.validate(rules), "catalogue": settings.catalogue()}


def _settings_row(cur) -> dict:
    rows = _rows(cur, """
        SELECT policy_id, version, last_modified_by, last_modified_date,
               policy_details->'rules' AS rules
          FROM proc.bp_policy
         WHERE policy_details->>'policy_identifier' = %s
           AND COALESCE(policy_status, 1) = 1
         ORDER BY policy_id""", (settings.POLICY,))
    if len(rows) != 1:
        raise settings.BriefSettingsUnavailable(
            "the daily brief settings were not found" if not rows
            else "more than one set of daily brief settings is active")
    return rows[0]


def save_settings(changes: Dict[str, Any], *, by: str, expected_version: Optional[int] = None) -> Dict[str, Any]:
    """Apply ``changes`` over the saved settings and store the result, if it holds together.

    ``changes`` replaces whole settings by name (a view map is replaced, not
    merged entry by entry), so what is saved is exactly what the screen showed.
    Raises ``ValueError`` carrying every problem when the result is not valid,
    and ``SettingsConflict`` when ``expected_version`` is no longer the saved one.
    """

    if not isinstance(changes, dict) or not changes:
        raise ValueError("no settings were given to change")

    with get_conn() as conn:
        cur = conn.cursor()
        row = _settings_row(cur)
        if expected_version is not None and int(row["version"] or 0) != int(expected_version):
            raise SettingsConflict("the settings were changed by somebody else; reload and try again")
        merged = {**(row["rules"] or {}), **changes}
        problems = settings.validate(merged)
        if problems:
            raise ValueError("; ".join(problems))
        # The version is in the WHERE as well as the check above: get_conn() is
        # autocommit, so two saves can interleave between the read and the write,
        # and the second must change nothing rather than overwrite the first.
        cur.execute("""
            UPDATE proc.bp_policy
               SET policy_details = jsonb_set(policy_details, '{rules}', %s::jsonb),
                   version = COALESCE(version, 0) + 1,
                   last_modified_date = now(), last_modified_by = %s
             WHERE policy_id = %s AND COALESCE(version, 0) = %s""",
                    (json.dumps(merged), by, row["policy_id"], int(row["version"] or 0)))
        if cur.rowcount != 1:
            raise SettingsConflict("the settings were changed by somebody else; reload and try again")

    governed_limits.reset_cache()
    rbac.reset_policy_cache()
    return read_settings()
