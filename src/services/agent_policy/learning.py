"""Learning (brief §6): read what people decided, and store what that supports as suggestions.

The only learning module that touches the database. It reads the firing log and writes rows in
proc.bp_agent_policy_suggestion (and a person's reversal records). It never writes a policy version: a
suggestion is a row for an owner to review, and only a person's save changes a policy.

The hourly scan raises and refreshes the rows; a person hides one (Not yet, Dismiss: only the
policy's owner or an Approver, `Caller.may_hide`) or asks for
it as an unsaved form (Draft this change, which writes nothing); a save that holds the change
marks it saved. An Approver's record that an approved request was later reversed is the one other
row this module writes (record_reversal, proc.bp_policy_firing_reversal, append-only).

What a row's status means, and what moves it:
  open        shown. Raised by the scan; `confident` false = "Not enough yet" (kept, not drafted).
  not_yet,    hidden by a person. The scan brings it back only after that many more decisions
  dismissed   (company settings), and only while the evidence still supports the change.
  saved       the change is in a version IN FORCE: the policy's live version or its latest one.
              Set by any save whose form holds the change (a person's or a machine draft's). While
              either version still holds it, the scan never touches the row. Once neither does (the
              draft was saved back, or replaced) and the evidence still supports the change, the
              scan reopens it: nobody dismissed it, so it must not be silenced for good.
  superseded  the policy's condition no longer has the state the change starts from, or the live
              form now marks the change's input sensitive (its value must not set a figure on a card).

Locking. Every writer that works on ONE policy (the scan's per-policy step, act, note_save) first
takes all of that policy's suggestion rows `ORDER BY suggestion_id FOR UPDATE` (_LOCK_SQL), so for
one policy a person's action and the scan run one after the other, never interleaved. The scan
read its policies before it took that lock, so under the lock it reads the policy's live and
latest conditions again (_FRESH_SQL) and judges "in force" on those: a save that landed in
between is seen. If the live condition itself moved, the policy waits for the next scan. The scan's
closing statement (_GONE_SQL) works across policies; it takes its rows in the same suggestion_id
order before changing them, so it cannot deadlock with the per-policy writers. `draft` takes no
lock: it is one read statement and writes nothing.

get_conn() is AUTOCOMMIT: each policy's rows are written in one explicit transaction (_tx).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from services.agent_policy import approvals, learning_draft, learning_patterns, learning_stats, settings
from services.agent_policy.compiler import on_missing_for
from services.agent_policy.learning_stats import Decided

logger = logging.getLogger(__name__)

#: The most decisions read for one policy in one scan: past it, the newest are the ones read.
MAX_DECISIONS = 5000
#: `errors`: a policy's scan step failed (or the scan could not start). `unreadable`: one stored
#: suggestion row could not be read and was skipped; the rest of its policy was still processed.
#: `skipped`: the policy changed (or could not be re-read) between the scan reading it and locking
#: it; nothing was judged, the next scan picks it up.
COUNTERS = ("policies", "raised", "updated", "reopened", "weakened", "superseded", "errors", "unreadable",
            "skipped")
#: Every company setting the scan reads. Checked once, before anything is judged.
SETTINGS = ("min_decisions", "min_days", "min_approvers", "wilson_lower", "median_seconds_floor",
            "not_yet_more", "dismiss_more")


@dataclass
class Target:
    """A live approve policy to look at: its live form, and every live version of it that had the
    identical condition (their firings are evidence for the same condition). `latest_condition`
    is the condition of the policy's latest version (a draft beside the live one, when there is
    one); None means there is nothing newer to read and the live condition stands for it.
    Anything else that is not a condition (not a dict) means the latest version cannot be read."""
    key: str
    live_version: int
    form: Dict[str, Any]
    versions: List[int]
    latest_condition: Any = None


# One row per approval case. A repeated agent call is linked to the same case and one decision
# settles every linked row, so rows are not decisions: DISTINCT ON the case. A reversal recorded
# on any row of the case that is this policy's reverses the decision (a case can also link other
# policies' rows: notify rows, a clash's losers). Timeouts, system deciders, a missing or blank
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
                       OR (o.decision_id IS NOT NULL AND o.decision_id = f.decision_id
                           AND o.policy_key = f.policy_key)) AS reversed
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

# Three reasons a live policy is left out, each on its own: it does not ask for approval, its
# business area is "never suggest" NOW, or its compiled document does not say it is eligible.
# `eligible` is compared as text: a stored value that is not a boolean then simply does not match,
# where a cast would fail the statement and stop the scan for every policy.
_TARGETS_SQL = """
    SELECT p.policy_key, p.live_version, v.form_state,
           lv.form_state -> 'hidden' -> 'condition' AS latest_condition
      FROM proc.bp_agent_policy p
      JOIN proc.bp_agent_policy_version v ON v.policy_key = p.policy_key AND v.version = p.live_version
      LEFT JOIN proc.bp_agent_policy_version lv ON lv.policy_key = p.policy_key AND lv.version = p.latest_version
      LEFT JOIN proc.bp_business_area a ON a.area_name = p.area_name
     WHERE p.status = 'live' AND v.form_state ->> 'outcome' = 'approve'
       AND COALESCE(a.never_suggest, FALSE) = FALSE
       AND v.compiled -> 'learning' ->> 'eligible' = 'true'
"""
_SAME_CONDITION_SQL = """
    SELECT version FROM proc.bp_agent_policy_version
     WHERE policy_key = %s AND saved_as = 'live' AND form_state ->> 'outcome' = 'approve'
       AND form_state -> 'hidden' -> 'condition' = %s::jsonb
     ORDER BY version
"""

# The policy as it stands now, read under the lock: its live version and condition, whether it has
# a latest version, and that version's condition. One statement, so it is one consistent picture.
_FRESH_SQL = """
    SELECT v.form_state -> 'hidden' -> 'condition', lv.version IS NOT NULL,
           lv.form_state -> 'hidden' -> 'condition', p.live_version
      FROM proc.bp_agent_policy p
      LEFT JOIN proc.bp_agent_policy_version v ON v.policy_key = p.policy_key AND v.version = p.live_version
      LEFT JOIN proc.bp_agent_policy_version lv ON lv.policy_key = p.policy_key AND lv.version = p.latest_version
     WHERE p.policy_key = %s
"""
#: A latest version that exists but holds no condition: it cannot be judged.
_UNREADABLE = "unreadable"
_REAL_KEY = re.compile(r"^[A-Z]{3}-[0-9]{4,}$")

_LOCK_SQL = """
    SELECT suggestion_id, change_key, status, n_at_action, proposal, confident
      FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s ORDER BY suggestion_id FOR UPDATE
"""
_INSERT_SQL = """
    INSERT INTO proc.bp_agent_policy_suggestion
           (policy_key, change_key, policy_version, kind, proposal, stats, confident, status, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, TRUE, 'open', %s, %s)
    ON CONFLICT (policy_key, change_key) DO NOTHING
    RETURNING suggestion_id
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
    for key, live_version, form, latest_condition in cur.fetchall():
        try:
            form = _json(form)
            condition = form["hidden"]["condition"]
            cur.execute(_SAME_CONDITION_SQL, (key, json.dumps(condition)))
            versions = [r[0] for r in cur.fetchall()] or [live_version]
        except Exception as exc:  # noqa: BLE001 - one policy must not hide the others
            # one line, no stack trace: it is said again every hour until the policy is put right
            logger.warning("learning: %s could not be read (%s: %s); it is skipped this time",
                           key, type(exc).__name__, exc)
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


def _in_force(proposal: Dict[str, Any], target: Target, told: List[str]) -> bool:
    """Whether a version in force, the live one or the latest, holds the proposal's change.
    When that cannot be told (the latest version's condition is not a condition, or cannot be
    compared) the answer is yes: a saved row is left alone rather than reopened on a guess.
    Said once per policy (`told` remembers)."""
    try:
        latest = _latest_condition(target)
        if not isinstance(latest, dict):
            raise TypeError("the latest version's condition is not a condition")
        return any(learning_patterns.applied(proposal, c) for c in (target.form["hidden"]["condition"], latest))
    except Exception as exc:  # noqa: BLE001 - one unreadable version must not fail the policy's scan step
        if not told:
            told.append(target.key)
            logger.warning("learning: whether a saved change of %s is in force cannot be told (%s: %s); "
                           "its saved suggestions are left as they are", target.key, type(exc).__name__, exc)
        return True


def _fresh(cur, target: Target) -> Optional[Target]:
    """The target with its conditions as they are NOW (call it under the policy's lock), or None
    when the policy is no longer the one the target was read as: its live version or its live
    condition moved (the versions whose firings are the evidence were chosen for those), or its
    row cannot be found. Nothing is judged then; the next scan reads it afresh."""
    cur.execute(_FRESH_SQL, (target.key,))
    found = cur.fetchone()
    if found is None:
        # No policy row. For a real policy key that is a policy that cannot be re-read: skip it.
        # Any other key is a target a test injected (TST-<hex>a keys are never real policies and
        # have no row): it stands as given.
        return None if _REAL_KEY.match(str(target.key)) else target
    live, has_latest, latest = _json(found[0]), found[1], _json(found[2])
    if found[3] != target.live_version or live != target.form["hidden"]["condition"]:
        return None
    return replace(target, latest_condition=(_UNREADABLE if latest is None else latest) if has_latest else None)


def _sensitive_fields(form: Dict[str, Any]) -> set:
    """The fields the policy's live form marks sensitive. Enforcement masks such a value in the
    log from the version that marks it; firings recorded BEFORE that still hold it in the clear."""
    inputs = (form.get("hidden") or {}).get("inputs") or []
    return {i.get("field") for i in inputs if isinstance(i, dict) and i.get("sensitive") and i.get("field")}


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


def _tell_owner(cur, target: Target, suggestion_id: int, proposal: Dict[str, Any], decided: List[Decided]) -> None:
    """One in-product notification to the policy's owner (the name its LIVE form gives) that a card
    was raised or has come back; never for a plain refresh (the caller decides that). Written on the
    policy's own cursor, so it is committed with the card or not at all, and inside a savepoint: a
    notification that cannot be written is logged and undone on its own, and the card stays.

    The notification table needs a firing to hang on (its check constraint allows no firing only
    for a conflict link), so the row names the newest decision of the change's evidence: a firing
    of this policy, which is also how the reader learns the policy key. Nobody is told when the
    form names no owner."""
    owner = str(target.form.get("owner") or "").strip()
    if not owner:
        return
    cur.execute("SAVEPOINT learning_notify")
    try:
        from services.agent_policy import gate   # lazily: the gate pulls in the decision engine

        evidence = learning_patterns.group_for(proposal, decided) or decided
        newest = max(evidence, key=lambda d: (d.decided_at, d.firing_id))
        gate._notify(cur, newest.firing_id, [owner],
                     f"Learning suggests a change to {target.key}: {proposal['plain']}", f"suggestion:{suggestion_id}")
    except Exception as exc:  # noqa: BLE001 - type only: a driver message can quote stored values
        cur.execute("ROLLBACK TO SAVEPOINT learning_notify")
        logger.warning("learning: the owner of %s could not be told about suggestion %s (%s); the card is kept",
                       target.key, suggestion_id, type(exc).__name__)
    cur.execute("RELEASE SAVEPOINT learning_notify")


def _store(cur, target: Target, locked: List[tuple], proposals: List[Dict[str, Any]], decided: List[Decided],
           cfg: Dict[str, Any], now: datetime) -> Dict[str, int]:
    """Bring the policy's suggestion rows (`locked`: what _LOCK_SQL returned) in line with this
    scan's proposals.

    The stored proposal is the change as it was last supported (its own `stats`, `plain` and
    `evidence` included); the `stats` column is the evidence for that change as of this scan.
    A stored row that cannot be read is logged, counted in `unreadable` and left exactly as it
    is; the policy's other rows are still processed.

    A value marked sensitive never sets a figure on a card: a proposal on a field the live form
    marks sensitive is not stored, and a card already shown or hidden on that field is superseded
    (older firings, recorded before the input was marked, hold the value unmasked and could
    otherwise put it in the "to" figure everyone who can read the queue sees).

    The owner is told (`_tell_owner`) each time a card is raised or comes back to `open`, once;
    a refresh of a card that is already open tells nobody.
    """
    counts = {k: 0 for k in COUNTERS}
    condition = target.form["hidden"]["condition"]
    sensitive = _sensitive_fields(target.form)
    existing = {r[1]: r for r in locked}
    proposed = set()
    told: List[str] = []
    for p in proposals:
        if p["field"] in sensitive:
            continue
        change_key, n = p["changeKey"], p["stats"]["decisions"]
        proposed.add(change_key)
        row = existing.get(change_key)
        args = (json.dumps(p), json.dumps(p["stats"]), target.live_version, p["kind"], now)
        if row is None:
            cur.execute(_INSERT_SQL, (target.key, change_key, target.live_version, p["kind"],
                                      json.dumps(p), json.dumps(p["stats"]), now, now))
            raised = cur.fetchone()                    # None: someone else's row got there first
            if raised is not None:
                counts["raised"] += 1
                _tell_owner(cur, target, raised[0], p, decided)
            continue
        suggestion_id, _key, status, n_at_action = row[0], row[1], row[2], row[3]
        if status == "saved" and _in_force(p, target, told):
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
        if counter != "updated":                       # raised again or reopened, not a refresh
            _tell_owner(cur, target, suggestion_id, p, decided)
    for change_key, (suggestion_id, _key, status, _n, proposal, confident) in existing.items():
        if change_key in proposed or status not in ("open", "not_yet", "dismissed"):
            continue
        proposal = _json(proposal)
        group: List[Decided] = []
        try:
            still = (isinstance(proposal, dict) and proposal.get("field") not in sensitive
                     and learning_patterns.targets(proposal, condition))
            if still and status == "open":
                group = learning_patterns.group_for(proposal, decided)
        except Exception as exc:  # noqa: BLE001 - one unreadable row must not stop the policy's others
            logger.warning("learning: suggestion %s of %s could not be read (%s: %s); it is left as it is",
                           suggestion_id, target.key, type(exc).__name__, exc)
            counts["unreadable"] += 1
            continue
        if not still:
            cur.execute(_SUPERSEDE_SQL, (now, suggestion_id))        # the condition has moved on, or
                                                                     # its input is now sensitive
            counts["superseded"] += 1
        elif status == "open":                                       # "Not enough yet"
            # outside the guard: the arithmetic does not depend on the stored row, so a failure
            # here is a bug and fails the policy's step loudly (`errors`)
            current = learning_stats.stats(group)
            cur.execute(_WEAKEN_SQL, (json.dumps(current), now, suggestion_id))
            counts["weakened"] += 1 if confident else 0              # counted when it turns, not hourly
    return counts


def _scan_one(conn, target: Target, cfg: Dict[str, Any], now: datetime) -> Dict[str, int]:
    with approvals._tx(conn):
        with conn.cursor() as cur:
            cur.execute(_LOCK_SQL, (target.key,))          # first: nothing below is judged unlocked
            locked = cur.fetchall()
            fresh = _fresh(cur, target)
            if fresh is None:
                logger.info("learning: %s changed (or could not be re-read) while it was being scanned; "
                            "it is left for the next scan", target.key)
                return {**{k: 0 for k in COUNTERS}, "skipped": 1}
            target = fresh
            condition = target.form["hidden"]["condition"]
            decided = decisions(cur, target.key, target.versions)
            proposals = learning_patterns.propose(condition, decided, cfg)
            for p in proposals:
                p["plain"] = learning_draft.plain(p, target.form)
                p["evidence"] = learning_draft.evidence(p, target.form)
                p["inputName"] = learning_draft._input(target.form, p["field"]).get("name") or p["field"]
            return _store(cur, target, locked, proposals, decided, cfg, now)


def scan(conn, now: datetime, *, only: Optional[List[Target]] = None) -> Dict[str, int]:
    """The hourly look at every live approve policy. Each policy is its own transaction and its
    own failure: one that cannot be read is counted in `errors` and the rest still run (one stored
    row that cannot be read is counted in `unreadable`, and the rest of its policy still runs). A company
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
_CARD_STATS = ("decisions", "approved", "approvers", "days", "medianSeconds", "wilsonLower", "reversed")
_NOT_FOUND = ("not_found", "No such suggestion.")
_NOT_OPEN = ("not_open", "This suggestion is no longer open.")
_NOT_CONFIDENT = ("not_confident", "There is not enough evidence yet.")
_POLICY_CHANGED = ("policy_changed", "This policy has changed since the suggestion was made.")
NOT_ALLOWED = ("not_allowed", "Only the policy's owner or an approver can hide a suggestion.")
#: Why an open card's change cannot be drafted right now (`blocked` on the card).
DRAFT_IN_THE_WAY = "The latest draft of this policy already changes this. Activate or discard that draft first."
NOT_ACTIVE = "This policy is no longer Active."


@dataclass(frozen=True)
class Caller:
    """Who is reading the queue or acting on a card, as far as learning needs to know.
    `names`: the decider names the person is linked to in the decider map. `editor`: has the role
    that edits drafts (Buyer) or higher. `approver`: has the Approver role or higher. `known`
    False: the decider map could not be read, so `names` is empty and nothing can be called theirs."""
    names: FrozenSet[str] = field(default_factory=frozenset)
    editor: bool = False
    approver: bool = False
    known: bool = True

    def owns(self, owner: str) -> bool:
        """Is the person linked to this owner name? A blank owner is nobody's."""
        return bool(owner) and owner in self.names

    def may_hide(self, owner: str) -> bool:
        """Not yet / Dismiss: the role that edits drafts, AND the policy's owner or an Approver."""
        return self.editor and (self.approver or self.owns(owner))


def caller_of(conn, principal, *, editor: bool, approver: bool, mapping: Any = None) -> Caller:
    """The person as learning sees them: their roles (worked out by the endpoint) and the decider
    names they are linked to, from `mapping` when the endpoint has already read the decider map,
    else read here. Never raises: an unreadable decider map is `known` False."""
    from services.agent_policy import approval_views, deciders   # lazily: approval_views reaches the gate

    try:
        names = frozenset(approval_views.caller_names(
            principal, deciders.load_map(conn) if mapping is None else mapping))
    except Exception as exc:  # noqa: BLE001 - the queue is still shown; nothing is called "mine"
        logger.warning("learning: the decider map could not be read (%s: %s); no card is shown as the caller's",
                       type(exc).__name__, exc)
        return Caller(editor=editor, approver=approver, known=False)
    return Caller(names=names, editor=editor, approver=approver)


# A suggestion with its policy's latest form (NULL when the policy row is gone), where the policy
# stands (its status, and its latest and live version numbers: a latest that is not the live one
# is a draft), and the owner named by its LIVE version (the owner in force: a draft cannot rename
# its way to a card). One statement, so a read is one consistent picture without a lock.
_CARD_SQL = """
    SELECT s.suggestion_id, s.policy_key, s.status, s.kind, s.proposal, s.stats, s.confident, s.updated_at,
           v.form_state, p.status, p.latest_version, lv.form_state -> 'hidden' -> 'condition',
           lv.form_state ->> 'owner', p.live_version
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
            "live_condition": _json(r[11]), "owner": r[12], "live_version": r[13]}


def _owner(row: Dict[str, Any]) -> str:
    """The owner in force: the one the policy's live version names ("" when it is not live)."""
    return str(row.get("owner") or "").strip()


def _blocked(row: Dict[str, Any], proposal: Dict[str, Any]) -> Optional[str]:
    """Why the change of an OPEN card cannot be drafted right now, in words for the person, or
    None. Judged as `draft` judges it, from what the card's own row already holds (no read here):
    the policy is no longer live; or its latest version is a draft whose condition no longer has
    the state the change starts from (one that cannot be read included: `draft` refuses that too).
    A row read without its policy row (only a test's injected suggestion) says nothing."""
    status = row.get("policy_status")
    if status is None:
        return None
    if status != "live":
        return NOT_ACTIVE
    latest, live = row.get("latest_version"), row.get("live_version")
    if latest is None or live is None or latest == live:
        return None
    try:
        starts_there = learning_patterns.targets(proposal, row["form"]["hidden"]["condition"])
    except Exception:  # noqa: BLE001 - a draft that cannot be read is a draft in the way
        starts_there = False
    return None if starts_there else DRAFT_IN_THE_WAY


def card(row: Dict[str, Any], caller: Optional[Caller] = None) -> Dict[str, Any]:
    """What a person is shown. The change and its evidence are worded now, from the policy's form
    and the `stats` column (the evidence as of the latest scan): the sentence stored inside the
    proposal is from when the change was last supported, which on a weak card is out of date.
    Without the policy's form, or when the wording cannot be built, the stored wording is used.

    `caller` is who is looking: `mine` says they are linked to the policy's owner, `canAct` that
    they may hide the card now (`Caller.may_hide`, on an open card). Both are false when nobody
    says who is looking.

    `blocked` is there only when an open card's change cannot be drafted right now (`_blocked`):
    the sentence to show, with `canDraft` false, so nobody has to click to find out."""
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
    # A reversal is what weakens a card whose approvals still read as full support: say so.
    undone = stats.get("reversed")
    if evidence and isinstance(undone, int) and not isinstance(undone, bool) and undone > 0:
        evidence += (" 1 of them was later marked as reversed." if undone == 1
                     else f" {undone} of them were later marked as reversed.")
    when = row.get("updated_at")
    confident = bool(row.get("confident"))
    owner = _owner(row)
    blocked = _blocked(row, proposal) if row["status"] == "open" else None
    return {"id": row["suggestion_id"], "policyKey": row["policy_key"],
            "policyName": (form.get("name") or "") if isinstance(form, dict) else "",
            "status": row["status"], "kind": row["kind"], "change": change or "", "evidence": evidence or "",
            "stats": {k: stats.get(k) for k in _CARD_STATS},
            # "suggestion confidence": never the same name as stage 1's extraction confidence
            "suggestionConfidence": "High" if confident else "Not enough yet",
            "canDraft": confident and row["status"] == "open" and blocked is None,
            **({"blocked": blocked} if blocked else {}),
            "ownerName": owner, "mine": caller is not None and caller.owns(owner),
            "canAct": caller is not None and row["status"] == "open" and caller.may_hide(owner),
            "updatedAt": when.astimezone(timezone.utc).isoformat() if isinstance(when, datetime) else None}


def list_cards(conn, *, status: str = "open", caller: Optional[Caller] = None) -> List[Dict[str, Any]]:
    """The queue, newest first: the open cards, or (`all`) every suggestion whatever became of it.
    Each card says whether it is `caller`'s and whether they may hide it."""
    if status not in ("open", "all"):
        raise ValueError(status)
    where = " WHERE s.status = 'open'" if status == "open" else ""
    with conn.cursor() as cur:
        cur.execute(_CARD_SQL + where + _CARD_ORDER + " LIMIT %s", (MAX_CARDS,))
        return [card(_card_row(r), caller) for r in cur.fetchall()]


def for_policy(cur, key: str, caller: Optional[Caller] = None) -> List[Dict[str, Any]]:
    """The open cards of one policy, for its page, as `caller` sees them."""
    cur.execute(_CARD_SQL + " WHERE s.policy_key = %s AND s.status = 'open'" + _CARD_ORDER, (key,))
    return [card(_card_row(r), caller) for r in cur.fetchall()]


def _read(cur, suggestion_id: int) -> Dict[str, Any]:
    cur.execute(_CARD_SQL + " WHERE s.suggestion_id = %s", (suggestion_id,))
    r = cur.fetchone()
    if r is None:
        raise Refused(*_NOT_FOUND)
    return _card_row(r)


def act(conn, suggestion_id: int, verb: str, *, actor: str, now: datetime, caller: Caller) -> Dict[str, Any]:
    """Not yet / Dismiss: hide an open card, recording who, when and on how many decisions. The
    scan brings it back only after that many more, so the count is always written: the decisions
    in the `stats` column, the evidence as of the latest scan.

    Only the policy's owner (someone linked to the owner its live version names) or an Approver
    may hide a card, and never anyone below the role that edits drafts: judged here, on the row as
    it is under the lock, whatever the endpoint checked. Anyone else is Refused `not_allowed`,
    before they learn whether the card is still open."""
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
            if not caller.may_hide(_owner(row)):
                raise Refused(*NOT_ALLOWED)
            if row["status"] != "open":
                raise Refused(*_NOT_OPEN)
            decided = (row["stats"] or {}).get("decisions") if isinstance(row["stats"], dict) else None
            if isinstance(decided, bool) or not isinstance(decided, int):
                # a hidden card with no count would stay hidden for ever
                raise ValueError(f"suggestion {suggestion_id} has no decision count")
            cur.execute(_ACT_SQL, (VERBS[verb], decided, actor, now, now, suggestion_id))
            return card(_read(cur, suggestion_id), caller)


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


# ---------------------------------------------------------------- an approved request, later reversed

_NO_FIRING = ("not_found", "No such request.")
_NOT_APPROVED = ("not_approved", "Only an approved request can be marked as reversed.")
_NOT_A_PERSON = ("not_approved", "Only a request a person approved can be marked as reversed.")
_ALREADY_REVERSED = ("already_reversed", "This request is already marked as reversed.")
_REASON_REQUIRED = ("reason_required", "Say why it was reversed.")

# The clicked row; the earliest row of its approval case that is THIS policy's own approved request
# (itself when it is not linked to a case); and whether any of this policy's rows of the case
# already carries a reversal. One decision settles every linked row of the policy, so that is what
# is reversed. A case's decision_id is also carried by rows that are not this request: other
# policies' notify rows and a clash's losers (gate.link_paused_notifies, settled 'approved' with
# the group), and a replay's re-check rows. One case is opened per approve policy
# (approvals.open_case), and a repeat call reuses the case of its own policy, so "same policy_key,
# approve, approved" is exactly the request and its repeats.
_CASE_OF_SQL = """
    SELECT f.outcome, f.result, f.decided_by,
           (SELECT min(o.firing_id) FROM proc.bp_policy_firing o
             WHERE (o.firing_id = f.firing_id OR (f.decision_id IS NOT NULL AND o.decision_id = f.decision_id))
               AND o.policy_key = f.policy_key AND o.outcome = 'approve' AND o.result = 'approved'),
           EXISTS (SELECT 1 FROM proc.bp_policy_firing_reversal r
                     JOIN proc.bp_policy_firing s ON s.firing_id = r.firing_id
                    WHERE s.firing_id = f.firing_id
                       OR (f.decision_id IS NOT NULL AND s.decision_id = f.decision_id
                           AND s.policy_key = f.policy_key))
      FROM proc.bp_policy_firing f
     WHERE f.firing_id = %s
"""
_REVERSE_SQL = """
    INSERT INTO proc.bp_policy_firing_reversal (firing_id, reversed_by, reversed_by_email, reversed_at, reason)
    VALUES (%s, %s, %s, %s, %s) ON CONFLICT (firing_id) DO NOTHING RETURNING firing_id
"""


def record_reversal(conn, firing_id: int, *, actor: str, reason: str, now: datetime,
                    email: Optional[str] = None) -> Dict[str, Any]:
    """An Approver's record that an approved request was later undone (a refund clawed back).

    Only a request a PERSON approved. Stored on the policy's EARLIEST approved firing row of the
    approval case, so one decision has at most one reversal whichever linked row was clicked, and
    never on another policy's row that shares the case. Append-only: it cannot be undone, and the scan
    counts it against every band that holds the decision. `email` is the actor's sign-in email, kept
    (trimmed; NULL when the sign-in has none) so the activity list can show who by name."""
    reason = reason.strip() if isinstance(reason, str) else ""
    if not reason:
        raise Refused(*_REASON_REQUIRED)
    with approvals._tx(conn):
        with conn.cursor() as cur:
            cur.execute(_CASE_OF_SQL, (firing_id,))
            found = cur.fetchone()
            if found is None:
                raise Refused(*_NO_FIRING)
            outcome, result, decided_by, first, reversed_ = found
            if (outcome, result) != ("approve", "approved") or first is None:
                raise Refused(*_NOT_APPROVED)
            # timeouts, group and precedent settlements are not a person's approval (as the scan reads them)
            if not (decided_by or "").strip() or decided_by.strip().lower().startswith("system:"):
                raise Refused(*_NOT_A_PERSON)
            if reversed_:
                raise Refused(*_ALREADY_REVERSED)
            cur.execute(_REVERSE_SQL, (first, actor, str(email or "").strip() or None, now, reason))
            if cur.fetchone() is None:               # someone else recorded it a moment ago
                raise Refused(*_ALREADY_REVERSED)
    return {"firingId": int(first), "reversedBy": actor, "reversedAt": now.astimezone(timezone.utc).isoformat(),
            "reason": reason}
