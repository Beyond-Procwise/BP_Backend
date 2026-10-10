"""Learning (brief §6): read what people decided, and store what that supports as suggestions.

The only learning module that touches the database. It reads the firing log and writes rows in
proc.bp_agent_policy_suggestion, nothing else. This module never writes a policy version: a
suggestion is a row for an owner to review, and only a person's save changes a policy.

get_conn() is AUTOCOMMIT: each policy's rows are written in one explicit transaction (_tx).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.agent_policy import approvals, learning_draft, learning_patterns, learning_stats, settings
from services.agent_policy.learning_stats import Decided

logger = logging.getLogger(__name__)

#: The most decisions read for one policy in one scan.
MAX_DECISIONS = 5000
COUNTERS = ("policies", "raised", "updated", "reopened", "weakened", "superseded", "errors")


@dataclass
class Target:
    """A live approve policy to look at: its live form, and every live version of it that had the
    identical condition (their firings are evidence for the same condition)."""
    key: str
    live_version: int
    form: Dict[str, Any]
    versions: List[int]


# One row per approval case. A repeated agent call is linked to the same case and one decision
# settles every linked row, so rows are not decisions: DISTINCT ON the case. A reversal recorded
# on any row of the case reverses the decision. Timeouts, system deciders, a missing or blank
# decider and still-paused rows are not people's decisions.
_DECIDED_SQL = """
    SELECT DISTINCT ON (COALESCE(f.decision_id, -f.firing_id))
           f.firing_id, f.matched_values, f.result, f.decided_by, f.created_at, f.decided_at,
           EXISTS (SELECT 1 FROM proc.bp_policy_firing_reversal r
                     JOIN proc.bp_policy_firing o ON o.firing_id = r.firing_id
                    WHERE o.firing_id = f.firing_id
                       OR (o.decision_id IS NOT NULL AND o.decision_id = f.decision_id)) AS reversed
      FROM proc.bp_policy_firing f
     WHERE f.policy_key = %s AND f.policy_version = ANY(%s)
       AND f.outcome = 'approve' AND f.result IN ('approved', 'rejected')
       AND f.decided_at IS NOT NULL AND f.decided_by IS NOT NULL
       AND f.decided_by NOT LIKE 'system:%%'
       AND btrim(f.decided_by) <> ''
     ORDER BY COALESCE(f.decision_id, -f.firing_id), f.firing_id
     LIMIT %s
"""

_TARGETS_SQL = """
    SELECT p.policy_key, p.live_version, v.form_state, v.compiled
      FROM proc.bp_agent_policy p
      JOIN proc.bp_agent_policy_version v ON v.policy_key = p.policy_key AND v.version = p.live_version
      LEFT JOIN proc.bp_business_area a ON a.area_name = p.area_name
     WHERE p.status = 'live' AND v.form_state ->> 'outcome' = 'approve'
       AND COALESCE(a.never_suggest, FALSE) = FALSE
       AND COALESCE((v.compiled -> 'learning' ->> 'eligible')::boolean, FALSE)
"""
_SAME_CONDITION_SQL = """
    SELECT version FROM proc.bp_agent_policy_version
     WHERE policy_key = %s AND saved_as = 'live' AND form_state ->> 'outcome' = 'approve'
       AND form_state -> 'hidden' -> 'condition' = %s::jsonb
     ORDER BY version
"""

_LOCK_SQL = """
    SELECT suggestion_id, change_key, status, n_at_action, proposal, confident
      FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s FOR UPDATE
"""
_INSERT_SQL = """
    INSERT INTO proc.bp_agent_policy_suggestion
           (policy_key, change_key, policy_version, kind, proposal, stats, confident, status, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, TRUE, 'open', %s, %s)
    ON CONFLICT (policy_key, change_key) DO NOTHING
"""
# The same statement refreshes an open card and brings a hidden one back.
_OPEN_SQL = """
    UPDATE proc.bp_agent_policy_suggestion
       SET status = 'open', confident = TRUE, proposal = %s::jsonb, stats = %s::jsonb,
           policy_version = %s, kind = %s, n_at_action = CASE WHEN status = 'open' THEN n_at_action END,
           updated_at = %s
     WHERE suggestion_id = %s
"""
_SUPERSEDE_SQL = """
    UPDATE proc.bp_agent_policy_suggestion SET status = 'superseded', updated_at = %s WHERE suggestion_id = %s
"""
_WEAKEN_SQL = """
    UPDATE proc.bp_agent_policy_suggestion SET stats = %s::jsonb, confident = FALSE, updated_at = %s
     WHERE suggestion_id = %s
"""
# Cards of policies the scan no longer looks at (retired, no longer approve, area now "never
# suggest"). Real keys only, so the rows of a test running beside the scan are never touched.
_GONE_SQL = """
    UPDATE proc.bp_agent_policy_suggestion s SET status = 'superseded', updated_at = %s
     WHERE s.status IN ('open','not_yet','dismissed') AND s.policy_key ~ '^[A-Z]{3}-[0-9]{4,}$'
       AND NOT (s.policy_key = ANY(%s))
"""


def _json(raw: Any) -> Any:
    if isinstance(raw, (str, bytes)):
        try:
            return json.loads(raw)
        except ValueError:
            return None
    return raw


def decisions(cur, key: str, versions: List[int]) -> List[Decided]:
    """What people decided under these versions of the policy, one per approval case."""
    cur.execute(_DECIDED_SQL, (key, list(versions), MAX_DECISIONS))
    rows = cur.fetchall()
    if len(rows) >= MAX_DECISIONS:
        logger.warning("learning: %s has more than %d decisions; only the first %d are read",
                       key, MAX_DECISIONS, MAX_DECISIONS)
    out: List[Decided] = []
    for firing_id, values, result, decided_by, asked_at, decided_at, reversed_ in rows:
        values = _json(values)
        out.append(Decided(firing_id=firing_id, values=values if isinstance(values, dict) else {},
                           approved=result == "approved", decided_by=decided_by, asked_at=asked_at,
                           decided_at=decided_at, reversed=bool(reversed_)))
    return out


def targets(cur) -> List[Target]:
    """Every live approve policy learning may suggest on. The area's "never suggest" setting is
    read here, at scan time, as well as the eligibility compiled when the version was saved."""
    cur.execute(_TARGETS_SQL)
    out: List[Target] = []
    for key, live_version, form, _compiled in cur.fetchall():
        form = _json(form)
        condition = ((form or {}).get("hidden") or {}).get("condition") if isinstance(form, dict) else None
        cur.execute(_SAME_CONDITION_SQL, (key, json.dumps(condition)))
        versions = [r[0] for r in cur.fetchall()] or [live_version]
        out.append(Target(key=key, live_version=live_version, form=form, versions=versions))
    return out


def _store(cur, target: Target, proposals: List[Dict[str, Any]], decided: List[Decided],
           cfg: Dict[str, Any], now: datetime) -> Dict[str, int]:
    """Bring the policy's suggestion rows in line with this scan's proposals.

    The stored proposal is the change as it was last supported (its own `stats`, `plain` and
    `evidence` included); the `stats` column is the evidence for that change as of this scan.
    """
    counts = {k: 0 for k in COUNTERS}
    condition = target.form["hidden"]["condition"]
    cur.execute(_LOCK_SQL, (target.key,))
    existing = {r[1]: r for r in cur.fetchall()}
    proposed = set()
    for p in proposals:
        change_key, n = p["changeKey"], p["stats"]["decisions"]
        proposed.add(change_key)
        row = existing.get(change_key)
        args = (json.dumps(p), json.dumps(p["stats"]), target.live_version, p["kind"], now)
        if row is None:
            cur.execute(_INSERT_SQL, (target.key, change_key, target.live_version, p["kind"],
                                      json.dumps(p), json.dumps(p["stats"]), now, now))
            counts["raised"] += cur.rowcount
            continue
        suggestion_id, _key, status, n_at_action = row[0], row[1], row[2], row[3]
        if status == "saved":
            continue                                   # once saved, never suggested again
        if status in ("not_yet", "dismissed"):
            more = learning_stats._num(cfg, "not_yet_more" if status == "not_yet" else "dismiss_more")
            if n < (n_at_action or 0) + more:
                continue
        cur.execute(_OPEN_SQL, args + (suggestion_id,))
        counts[{"superseded": "raised", "open": "updated"}.get(status, "reopened")] += 1
    for change_key, (suggestion_id, _key, status, _n, proposal, confident) in existing.items():
        if change_key in proposed or status not in ("open", "not_yet", "dismissed"):
            continue
        proposal = _json(proposal)
        if not isinstance(proposal, dict) or not learning_patterns.targets(proposal, condition):
            cur.execute(_SUPERSEDE_SQL, (now, suggestion_id))        # the condition has moved on
            counts["superseded"] += 1
        elif status == "open":                                       # "Not enough yet"
            current = learning_stats.stats(learning_patterns.group_for(proposal, decided))
            cur.execute(_WEAKEN_SQL, (json.dumps(current), now, suggestion_id))
            counts["weakened"] += 1 if confident else 0              # counted when it turns, not hourly
    return counts


def _scan_one(conn, target: Target, cfg: Dict[str, Any], now: datetime) -> Dict[str, int]:
    with approvals._tx(conn):
        with conn.cursor() as cur:
            condition = target.form["hidden"]["condition"]
            decided = decisions(cur, target.key, target.versions)
            proposals = learning_patterns.propose(condition, decided, cfg)
            for p in proposals:
                p["plain"] = learning_draft.plain(p, target.form)
                p["evidence"] = learning_draft.evidence(p, target.form)
                p["inputName"] = learning_draft._input(target.form, p["field"]).get("name") or p["field"]
            return _store(cur, target, proposals, decided, cfg, now)


def scan(conn, now: datetime, *, only: Optional[List[Target]] = None) -> Dict[str, int]:
    """The hourly look at every live approve policy. Each policy is its own transaction and its
    own failure: one that cannot be read is counted in `errors` and the rest still run. Never
    raises. `only` narrows the scan to the given targets (tests)."""
    counts = {k: 0 for k in COUNTERS}
    try:
        cfg = settings.load_settings(conn)["learning"]
        if only is None:
            with conn.cursor() as cur:
                todo = targets(cur)
        else:
            todo = list(only)
    except Exception:  # noqa: BLE001 - nothing was read, so nothing is judged or superseded
        logger.exception("learning scan could not load its policies")
        counts["errors"] += 1
        return counts
    for target in todo:
        counts["policies"] += 1
        try:
            for name, n in _scan_one(conn, target, cfg, now).items():
                counts[name] += n
        except Exception:  # noqa: BLE001 - one policy must not stop the scan
            counts["errors"] += 1
            logger.exception("learning scan failed for %s", getattr(target, "key", "?"))
    if only is None:
        try:
            with approvals._tx(conn):
                with conn.cursor() as cur:
                    cur.execute(_GONE_SQL, (now, [t.key for t in todo]))
                    counts["superseded"] += cur.rowcount
        except Exception:  # noqa: BLE001
            counts["errors"] += 1
            logger.exception("learning scan could not close the cards of policies no longer scanned")
    return counts
