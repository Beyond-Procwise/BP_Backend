"""Learning (brief §6): read what people decided, and store what that supports as suggestions.

The only learning module that touches the database. It reads the firing log and writes rows in
proc.bp_agent_policy_suggestion, nothing else. This module never writes a policy version: a
suggestion is a row for an owner to review, and only a person's save changes a policy.

The hourly scan raises and refreshes the rows; a person hides one (Not yet, Dismiss) or asks for
it as an unsaved form (Draft this change, which writes nothing); a save that holds the change
marks it saved.

What a row's status means, and what moves it:
  open        shown. Raised by the scan; `confident` false = "Not enough yet" (kept, not drafted).
  not_yet,    hidden by a person. The scan brings it back only after that many more decisions
  dismissed   (company settings), and only while the evidence still supports the change.
  saved       the change is in a version IN FORCE: the policy's live version or its latest one.
              Set by any save whose form holds the change (a person's or a machine draft's). While
              either version still holds it, the scan never touches the row. Once neither does (the
              draft was saved back, or replaced) and the evidence still supports the change, the
              scan reopens it: nobody dismissed it, so it must not be silenced for good.
  superseded  the policy's condition no longer has the state the change starts from.

Locking. Every writer that works on ONE policy (the scan's per-policy step, act, note_save) first
takes all of that policy's suggestion rows `ORDER BY suggestion_id FOR UPDATE` (_LOCK_SQL), so for
one policy a person's action and the scan run one after the other, never interleaved. The scan's
closing statement (_GONE_SQL) works across policies; it takes its rows in the same suggestion_id
order before changing them, so it cannot deadlock with the per-policy writers. `draft` takes no
lock: it is one read statement and writes nothing.

get_conn() is AUTOCOMMIT: each policy's rows are written in one explicit transaction (_tx).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.agent_policy import approvals, learning_draft, learning_patterns, learning_stats, settings
from services.agent_policy.compiler import on_missing_for
from services.agent_policy.learning_stats import Decided

logger = logging.getLogger(__name__)

#: The most decisions read for one policy in one scan: past it, the newest are the ones read.
MAX_DECISIONS = 5000
COUNTERS = ("policies", "raised", "updated", "reopened", "weakened", "superseded", "errors")
#: Every company setting the scan reads. Checked once, before anything is judged.
SETTINGS = ("min_decisions", "min_days", "min_approvers", "wilson_lower", "median_seconds_floor",
            "not_yet_more", "dismiss_more")


@dataclass
class Target:
    """A live approve policy to look at: its live form, and every live version of it that had the
    identical condition (their firings are evidence for the same condition). `latest_condition`
    is the condition of the policy's latest version (a draft beside the live one, when there is
    one); None means there is nothing newer to read and the live condition stands for it."""
    key: str
    live_version: int
    form: Dict[str, Any]
    versions: List[int]
    latest_condition: Any = None


# One row per approval case. A repeated agent call is linked to the same case and one decision
# settles every linked row, so rows are not decisions: DISTINCT ON the case. A reversal recorded
# on any row of the case reverses the decision. Timeouts, system deciders, a missing or blank
# decider and still-paused rows are not people's decisions. The newest come first, so past the
# cap it is the oldest that go unread: a new rejection or reversal is always seen.
_DECIDED_SQL = """
    SELECT d.firing_id, d.matched_values, d.result, d.decided_by, d.created_at, d.decided_at, d.reversed
      FROM (
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
       AND lower(btrim(f.decided_by)) NOT LIKE 'system:%%'
       AND btrim(f.decided_by) <> ''
     ORDER BY COALESCE(f.decision_id, -f.firing_id), f.firing_id
           ) d
     ORDER BY d.decided_at DESC, d.firing_id DESC
     LIMIT %s
"""

_TARGETS_SQL = """
    SELECT p.policy_key, p.live_version, v.form_state, v.compiled,
           lv.form_state -> 'hidden' -> 'condition' AS latest_condition
      FROM proc.bp_agent_policy p
      JOIN proc.bp_agent_policy_version v ON v.policy_key = p.policy_key AND v.version = p.live_version
      LEFT JOIN proc.bp_agent_policy_version lv ON lv.policy_key = p.policy_key AND lv.version = p.latest_version
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
      FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s ORDER BY suggestion_id FOR UPDATE
"""
_INSERT_SQL = """
    INSERT INTO proc.bp_agent_policy_suggestion
           (policy_key, change_key, policy_version, kind, proposal, stats, confident, status, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, TRUE, 'open', %s, %s)
    ON CONFLICT (policy_key, change_key) DO NOTHING
"""
# The same statement refreshes an open card and brings a hidden or no-longer-saved one back.
_OPEN_SQL = """
    UPDATE proc.bp_agent_policy_suggestion
       SET status = 'open', confident = TRUE, proposal = %s::jsonb, stats = %s::jsonb,
           policy_version = %s, kind = %s, n_at_action = CASE WHEN status = 'open' THEN n_at_action END,
           saved_version = NULL, updated_at = %s
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
# The rows are locked in suggestion_id order before any is changed, the order _LOCK_SQL uses.
_GONE_SQL = """
    UPDATE proc.bp_agent_policy_suggestion s SET status = 'superseded', updated_at = %s
      FROM (SELECT g.suggestion_id FROM proc.bp_agent_policy_suggestion g
             WHERE g.status IN ('open','not_yet','dismissed') AND g.policy_key ~ '^[A-Z]{3}-[0-9]{4,}$'
               AND NOT (g.policy_key = ANY(%s))
             ORDER BY g.suggestion_id FOR UPDATE) gone
     WHERE s.suggestion_id = gone.suggestion_id
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
    cur.execute(_DECIDED_SQL, (key, list(versions), MAX_DECISIONS + 1))     # one extra: is there more?
    rows = cur.fetchall()
    if len(rows) > MAX_DECISIONS:
        logger.warning("learning: %s has more than %d decisions; only the newest %d are read",
                       key, MAX_DECISIONS, MAX_DECISIONS)
        rows = rows[:MAX_DECISIONS]
    out: List[Decided] = []
    for firing_id, values, result, decided_by, asked_at, decided_at, reversed_ in rows:
        values = _json(values)
        out.append(Decided(firing_id=firing_id, values=values if isinstance(values, dict) else {},
                           approved=result == "approved", decided_by=decided_by, asked_at=asked_at,
                           decided_at=decided_at, reversed=bool(reversed_)))
    return out


def _targets(cur) -> Tuple[List[Target], List[str]]:
    """The targets, and the keys of policies that are targets but could not be read."""
    cur.execute(_TARGETS_SQL)
    out: List[Target] = []
    unread: List[str] = []
    for key, live_version, form, _compiled, latest_condition in cur.fetchall():
        try:
            form = _json(form)
            condition = form["hidden"]["condition"]
            cur.execute(_SAME_CONDITION_SQL, (key, json.dumps(condition)))
            versions = [r[0] for r in cur.fetchall()] or [live_version]
        except Exception:  # noqa: BLE001 - one policy must not hide the others
            logger.exception("learning: %s could not be read; it is skipped this time", key)
            unread.append(key)
            continue
        out.append(Target(key=key, live_version=live_version, form=form, versions=versions,
                          latest_condition=_json(latest_condition)))
    return out, unread


def targets(cur) -> List[Target]:
    """Every live approve policy learning may suggest on. The area's "never suggest" setting is
    read here, at scan time, as well as the eligibility compiled when the version was saved.
    A policy whose row cannot be read is logged by key and left out; the rest are returned."""
    return _targets(cur)[0]


_HIDDEN = ("not_yet", "dismissed")


def _latest_condition(target: Target) -> Any:
    """The condition of the policy's latest version; the live one when nothing newer was read."""
    return target.form["hidden"]["condition"] if target.latest_condition is None else target.latest_condition


def _in_force(proposal: Dict[str, Any], target: Target) -> bool:
    """Whether a version in force, the live one or the latest, holds the proposal's change."""
    return any(learning_patterns.applied(proposal, c)
               for c in (target.form["hidden"]["condition"], _latest_condition(target)))


def _wait(cfg: Dict[str, Any], status: str) -> float:
    """How many more decisions bring a hidden card back."""
    waits = {"not_yet": learning_stats._num(cfg, "not_yet_more"), "dismissed": learning_stats._num(cfg, "dismiss_more")}
    return waits.get(status, max(waits.values()))


def _checked_settings(conn) -> Dict[str, Any]:
    """The learning settings, every one a usable number, or TypeError/ValueError naming the first that is not."""
    cfg = settings.load_settings(conn)["learning"]
    for name in SETTINGS:
        learning_stats._num(cfg, name)
    return cfg


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
        if status == "saved" and _in_force(p, target):
            continue                                   # saved, and a version in force still holds it
        # A saved row whose change is in neither the live nor the latest version any more (the
        # draft was saved back, or replaced) returns: nobody dismissed it.
        counter = {"superseded": "raised", "open": "updated"}.get(status, "reopened")
        # A card a person hid comes back only on that many more decisions. Superseding keeps the
        # count, so a hidden card whose condition went away and came back is still hidden; which
        # of the two it was is no longer known, so the longer wait applies.
        if status in _HIDDEN or (status == "superseded" and n_at_action is not None):
            if n_at_action is None:
                logger.warning("learning: suggestion %s is %s with no decision count; it stays hidden",
                               suggestion_id, status)
                continue
            if n < n_at_action + _wait(cfg, status):
                continue
            counter = "reopened"
        cur.execute(_OPEN_SQL, args + (suggestion_id,))
        counts[counter] += 1
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
    own failure: one that cannot be read is counted in `errors` and the rest still run. A company
    setting that is not a usable number stops the whole scan with one warning and one error:
    nothing is judged on a guess. Never raises. `only` narrows the scan to the given targets (tests)."""
    counts = {k: 0 for k in COUNTERS}
    try:
        cfg = _checked_settings(conn)
    except Exception as exc:  # noqa: BLE001 - a setting that is not a number: no suggestion, never a guess
        logger.warning("learning scan not run, nothing written: %s", exc)
        counts["errors"] += 1
        return counts
    unread: List[str] = []
    try:
        if only is None:
            with conn.cursor() as cur:
                todo, unread = _targets(cur)
        else:
            todo = list(only)
    except Exception:  # noqa: BLE001 - nothing was read, so nothing is judged or superseded
        logger.exception("learning scan could not load its policies")
        counts["errors"] += 1
        return counts
    counts["errors"] += len(unread)
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
                    # an unread policy is still a policy: its cards are not closed
                    cur.execute(_GONE_SQL, (now, [t.key for t in todo] + unread))
                    counts["superseded"] += cur.rowcount
        except Exception:  # noqa: BLE001
            counts["errors"] += 1
            logger.exception("learning scan could not close the cards of policies no longer scanned")
    return counts


# ---------------------------------------------------------------- the card, the queue, a person's actions

class Refused(Exception):
    """A person's request on a suggestion that cannot be done; `message` is for that person."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


#: What a person may do to an open card, and the status it then has.
VERBS = {"not_yet": "not_yet", "dismiss": "dismissed"}
#: The most cards one read of the queue returns (newest first).
MAX_CARDS = 500
_CARD_STATS = ("decisions", "approved", "approvers", "days", "medianSeconds", "wilsonLower")
_NOT_FOUND = ("not_found", "No such suggestion.")
_NOT_OPEN = ("not_open", "This suggestion is no longer open.")
_NOT_CONFIDENT = ("not_confident", "There is not enough evidence yet.")
_POLICY_CHANGED = ("policy_changed", "This policy has changed since the suggestion was made.")

# A suggestion with its policy's latest form (NULL when the policy row is gone) and where the
# policy stands. One statement, so a read is one consistent picture without a lock.
_CARD_SQL = """
    SELECT s.suggestion_id, s.policy_key, s.status, s.kind, s.proposal, s.stats, s.confident, s.updated_at,
           v.form_state, p.status, p.latest_version, lv.form_state -> 'hidden' -> 'condition'
      FROM proc.bp_agent_policy_suggestion s
      LEFT JOIN proc.bp_agent_policy p ON p.policy_key = s.policy_key
      LEFT JOIN proc.bp_agent_policy_version v ON v.policy_key = p.policy_key AND v.version = p.latest_version
      LEFT JOIN proc.bp_agent_policy_version lv ON lv.policy_key = p.policy_key AND lv.version = p.live_version
"""
_CARD_ORDER = " ORDER BY s.updated_at DESC, s.suggestion_id DESC"
_KEY_OF_SQL = "SELECT policy_key FROM proc.bp_agent_policy_suggestion WHERE suggestion_id = %s"
_ACT_SQL = """
    UPDATE proc.bp_agent_policy_suggestion
       SET status = %s, n_at_action = %s, acted_by = %s, acted_at = %s, updated_at = %s
     WHERE suggestion_id = %s
"""
_SAVED_SQL = """
    UPDATE proc.bp_agent_policy_suggestion SET status = 'saved', saved_version = %s, updated_at = now()
     WHERE suggestion_id = %s
"""


def _card_row(r) -> Dict[str, Any]:
    form = _json(r[8])
    return {"suggestion_id": r[0], "policy_key": r[1], "status": r[2], "kind": r[3], "proposal": _json(r[4]),
            "stats": _json(r[5]), "confident": r[6], "updated_at": r[7],
            "form": form if isinstance(form, dict) else None, "policy_status": r[9], "latest_version": r[10],
            "live_condition": _json(r[11])}


def card(row: Dict[str, Any]) -> Dict[str, Any]:
    """What a person is shown. The change and its evidence are worded now, from the policy's form
    and the `stats` column (the evidence as of the latest scan): the sentence stored inside the
    proposal is from when the change was last supported, which on a weak card is out of date.
    Without the policy's form, or when the wording cannot be built, the stored wording is used."""
    proposal = row.get("proposal") if isinstance(row.get("proposal"), dict) else {}
    stats = row.get("stats") if isinstance(row.get("stats"), dict) else {}
    form = row.get("form")
    change, evidence = proposal.get("plain"), proposal.get("evidence")
    if isinstance(form, dict):
        try:
            change, evidence = (learning_draft.plain(proposal, form),
                                learning_draft.evidence({**proposal, "stats": stats}, form))
        except Exception as exc:  # noqa: BLE001 - one unreadable row must not empty the queue
            logger.warning("learning: the wording of suggestion %s could not be built (%s: %s); "
                           "the wording stored with it is shown", row.get("suggestion_id"), type(exc).__name__, exc)
    when = row.get("updated_at")
    confident = bool(row.get("confident"))
    return {"id": row["suggestion_id"], "policyKey": row["policy_key"],
            "policyName": (form.get("name") or "") if isinstance(form, dict) else "",
            "status": row["status"], "kind": row["kind"], "change": change or "", "evidence": evidence or "",
            "stats": {k: stats.get(k) for k in _CARD_STATS},
            # "suggestion confidence": never the same name as stage 1's extraction confidence
            "suggestionConfidence": "High" if confident else "Not enough yet",
            "canDraft": confident and row["status"] == "open",
            "updatedAt": when.astimezone(timezone.utc).isoformat() if isinstance(when, datetime) else None}


def list_cards(conn, *, status: str = "open") -> List[Dict[str, Any]]:
    """The queue, newest first: the open cards, or (`all`) every suggestion whatever became of it."""
    if status not in ("open", "all"):
        raise ValueError(status)
    where = " WHERE s.status = 'open'" if status == "open" else ""
    with conn.cursor() as cur:
        cur.execute(_CARD_SQL + where + _CARD_ORDER + " LIMIT %s", (MAX_CARDS,))
        return [card(_card_row(r)) for r in cur.fetchall()]


def for_policy(cur, key: str) -> List[Dict[str, Any]]:
    """The open cards of one policy, for its page."""
    cur.execute(_CARD_SQL + " WHERE s.policy_key = %s AND s.status = 'open'" + _CARD_ORDER, (key,))
    return [card(_card_row(r)) for r in cur.fetchall()]


def _read(cur, suggestion_id: int) -> Dict[str, Any]:
    cur.execute(_CARD_SQL + " WHERE s.suggestion_id = %s", (suggestion_id,))
    r = cur.fetchone()
    if r is None:
        raise Refused(*_NOT_FOUND)
    return _card_row(r)


def act(conn, suggestion_id: int, verb: str, *, actor: str, now: datetime) -> Dict[str, Any]:
    """Not yet / Dismiss: hide an open card, recording who, when and on how many decisions. The
    scan brings it back only after that many more, so the count is always written: the decisions
    in the `stats` column, the evidence as of the latest scan."""
    if verb not in VERBS:
        raise ValueError(verb)
    with approvals._tx(conn):
        with conn.cursor() as cur:
            cur.execute(_KEY_OF_SQL, (suggestion_id,))
            found = cur.fetchone()
            if found is None:
                raise Refused(*_NOT_FOUND)
            cur.execute(_LOCK_SQL, (found[0],))          # the policy's rows, in the scan's order
            cur.fetchall()
            row = _read(cur, suggestion_id)              # read again, now that it cannot change
            if row["status"] != "open":
                raise Refused(*_NOT_OPEN)
            decided = (row["stats"] or {}).get("decisions") if isinstance(row["stats"], dict) else None
            if isinstance(decided, bool) or not isinstance(decided, int):
                # a hidden card with no count would stay hidden for ever
                raise ValueError(f"suggestion {suggestion_id} has no decision count")
            cur.execute(_ACT_SQL, (VERBS[verb], decided, actor, now, now, suggestion_id))
            return card(_read(cur, suggestion_id))


def draft(conn, suggestion_id: int) -> Dict[str, Any]:
    """'Draft this change': the policy's next version as an UNSAVED form. Writes nothing; only the
    owner's save, and then the normal activate step, changes the policy. Built on the policy's
    latest version, so a newer draft is carried forward rather than overwritten."""
    with conn.cursor() as cur:
        row = _read(cur, suggestion_id)
    # saved: a version already holds the change. superseded: the condition is no longer the one
    # the suggestion was about. Either way the policy has moved on from the card.
    if row["status"] in ("saved", "superseded"):
        raise Refused(*_POLICY_CHANGED)
    if row["status"] != "open":
        raise Refused(*_NOT_OPEN)
    if not row["confident"]:
        raise Refused(*_NOT_CONFIDENT)
    form, proposal = row["form"], row["proposal"]
    if row["policy_status"] != "live" or form is None or not isinstance(proposal, dict):
        raise Refused(*_POLICY_CHANGED)
    try:
        # The evidence is about the policy IN FORCE: its live condition must still have the state
        # the change starts from, whatever a newer draft says. And only an approve policy is loosened.
        if form.get("outcome") != "approve" or not learning_patterns.targets(proposal, row["live_condition"]):
            raise Refused(*_POLICY_CHANGED)
        built = learning_draft.build(form, proposal, on_missing_for(form, settings.load_settings(conn)))
    except Refused:
        raise
    except Exception:  # noqa: BLE001 - a form that can no longer be read is a policy that has changed
        logger.exception("learning: suggestion %s could not be drafted", suggestion_id)
        built = None
    if built is None:
        raise Refused(*_POLICY_CHANGED)
    return {"policyKey": row["policy_key"], "baseVersion": row["latest_version"], "form": built["form"],
            "changeNote": built["changeNote"], "checkWording": built["checkWording"],
            "changedWording": built["changedWording"], "suggestionId": row["suggestion_id"]}


def note_save(cur, key: str, version: int, form: Dict[str, Any]) -> int:
    """A person saved `version` of the policy. Every open or snoozed suggestion whose change that
    form actually holds is marked saved (and is never suggested again). Returns how many.

    Runs inside the save's transaction, on its cursor: the mark and the version are one write."""
    hidden = form.get("hidden") if isinstance(form, dict) else None
    condition = hidden.get("condition") if isinstance(hidden, dict) else None
    cur.execute(_LOCK_SQL, (key,))
    marked = 0
    for suggestion_id, _change, status, _n, proposal, _confident in cur.fetchall():
        proposal = _json(proposal)
        if status not in ("open", "not_yet") or not isinstance(proposal, dict):
            continue
        try:
            holds = learning_patterns.applied(proposal, condition)
        except Exception as exc:  # noqa: BLE001 - one unreadable proposal must not block the others
            logger.warning("learning: suggestion %s could not be compared with version %s of %s (%s: %s); "
                           "it is left as it is", suggestion_id, version, key, type(exc).__name__, exc)
            continue
        if holds:
            cur.execute(_SAVED_SQL, (version, suggestion_id))
            marked += 1
    return marked
