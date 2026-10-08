"""Stage 6: learn from what reviewers do to a draft, and route each lesson to a queue.

This module NEVER changes a prompt, a policy, a style profile, an exemplar set or a family config. It fills
queues and candidate lists; a person decides every one. It still reads only the edit score, the figures that
changed, the word counts and the model's own draft. The sent text and its diff ARE now stored
(bp_draft_sent_text, retention-governed, writer-only) but this job does not consume them yet.

Routes (an edit can take several):

  fact      a Postgres-backed figure was changed   -> bp_dq_item, with the row it came from. The reviewer's
                                                       value is stored THERE ONLY and is never learned from.
  reasoned  one of our judgements was changed       -> bp_eval_candidate; 3+ distinct reviewers making the
                                                       same correction opens a bp_review_item
  wording   words changed beyond the figures        -> counts toward the reviewer's style signals
  family    the reviewer answered the classifier's
            question with the other family          -> bp_classifier_example

What is NOT possible without the sent text, and is therefore not claimed: rules about WHICH words a person
prefers, and detecting "the same wording correction" from several people. The job reports "many people
rewrite this family" instead, and says so in the item.
"""

from __future__ import annotations

import json
import logging
import statistics
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from . import validator as V

logger = logging.getLogger(__name__)
SLUG = "email_learning_rules"
INT_KEYS = ("min_distinct_users", "window_days", "style_window_edits", "min_edits_for_style_rule", "max_style_rules",
            "wording_edit_min_words", "exemplar_review_months", "batch_size")
NUM_KEYS = ("length_shorten_ratio", "length_lengthen_ratio", "high_divergence", "exemplar_max_distance", "exemplar_min_judge")


class LearningRulesUnavailable(RuntimeError):
    """A threshold is missing or unusable. The job does not run on a number it made up."""


def load_rules(policy_engine: Any) -> Dict[str, Any]:
    if policy_engine is None:
        raise LearningRulesUnavailable("no policy engine available")
    try:
        policy = policy_engine.get_policy(SLUG)
    except Exception as exc:  # noqa: BLE001
        raise LearningRulesUnavailable(f"policy store unreadable: {exc}") from exc
    rules = ((policy or {}).get("details") or {}).get("rules") if isinstance(policy, dict) else None
    if not isinstance(rules, dict):
        raise LearningRulesUnavailable("no learning rules are defined")
    out: Dict[str, Any] = {}
    for key in INT_KEYS + NUM_KEYS:
        v = rules.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise LearningRulesUnavailable(f"{key} is missing or not a positive number")
        out[key] = int(v) if key in INT_KEYS else float(v)
    return out


# --- classifying one edit -------------------------------------------------------------------------------

def _dec(v: Any) -> Optional[Decimal]:
    return V.to_decimal(v) if v not in (None, "") else None


def _keys_with_value(pool: Dict[str, Any], value: str) -> List[str]:
    d = _dec(value)
    return [k for k, f in (pool or {}).items() if d is not None and _dec((f or {}).get("value")) == d]


def classify_edit(outcome: Dict[str, Any], capture: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    removed = [r for r in (outcome.get("removed_figures") or [])]
    added = [a for a in (outcome.get("added_figures") or [])]
    facts, reasoned = capture.get("facts") or {}, capture.get("reasoned") or {}
    fact_changes, reasoned_changes = [], []
    for i, r in enumerate(removed):
        # A figure that simply disappears (the sentence was cut) is an omission, not a correction. A change
        # needs a figure that took its place; they are paired in the order they appear.
        if i >= len(added):
            continue
        now = added[i]["value"]
        if r.get("class") == "fact":
            for key in _keys_with_value(facts, r["value"]) or ["unknown"]:
                fact_changes.append({"fact_key": key, "was": r["value"], "now": now, "source": (facts.get(key) or {})})
        elif r.get("class") == "reasoned":
            for key in _keys_with_value(reasoned, r["value"]) or ["unknown"]:
                a, b = _dec(r["value"]), _dec(now)
                direction = "changed" if a is None or b is None else "raised" if b > a else "lowered" if b < a else "changed"
                reasoned_changes.append({"key": key, "direction": direction, "from": r["value"], "to": now})
    distance = float(outcome.get("edit_distance") or 0)
    longer = max(int(outcome.get("drafted_words") or 0), int(outcome.get("sent_words") or 0))
    figure_words = max(len(removed), len(added))
    words_changed = max(0, round(distance * longer) - figure_words)
    wording = words_changed >= rules["wording_edit_min_words"]

    res = capture.get("assumptions_resolution") or {}
    items = capture.get("assumption_items") or []
    all_confirmed = all((res.get(a["id"]) or {}).get("action") == "confirm" for a in items)
    judge = capture.get("judge") or {}
    exemplar = bool(
        distance < rules["exemplar_max_distance"]
        and judge.get("status") == "scored" and float(judge.get("overall") or 0) >= rules["exemplar_min_judge"]
        and not (capture.get("carried_unverified") or {}) and not (capture.get("unverified_figures") or [])
        # Beyond the four criteria in the spec: a draft whose price or deadline a person just overruled is not
        # an example of a good one, however few words changed.
        and all_confirmed and not fact_changes and not reasoned_changes
        and capture.get("assurance_status") != "unassured")
    routes = [r for r, on in (("fact", bool(fact_changes)), ("reasoned", bool(reasoned_changes)), ("wording", wording)) if on]
    return {"fact": fact_changes, "reasoned": reasoned_changes, "wording": wording, "words_changed": words_changed,
            "exemplar": exemplar, "routes": routes}


# --- the job ------------------------------------------------------------------------------------------------

def _rows(cur) -> List[Dict[str, Any]]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _j(v: Any) -> Optional[str]:
    return None if v is None else json.dumps(v, default=str)


def run_learning(conn: Any, policy_engine: Any, now: Optional[datetime] = None) -> Dict[str, int]:
    """Process every sent draft not yet learned from. Idempotent: a second run finds nothing to do."""

    rules = load_rules(policy_engine)
    now = now or datetime.now(timezone.utc)
    report = {"outcomes": 0, "dq_items": 0, "eval_candidates": 0, "exemplar_candidates": 0, "classifier_examples": 0,
              "review_items": 0, "style_rule_batches": 0, "exemplars_expired": 0}
    touched_users = set()
    with conn.cursor() as cur:
        cur.execute("""SELECT o.outcome_id, o.capture_id, o.sent_by, o.reviewed_by, o.edit_distance, o.edit_class,
                              o.removed_figures, o.added_figures, o.drafted_words, o.sent_words,
                              c.family_id, c.facts, c.reasoned, c.tone_variables, c.judge, c.assumption_items,
                              c.assumptions_resolution, c.carried_unverified, c.unverified_figures, c.draft_text,
                              c.assurance_status, c.brief
                       FROM email_agent.bp_draft_outcome o JOIN email_agent.bp_draft_capture c USING (capture_id)
                       WHERE o.outcome = 'sent' AND o.learning_processed_at IS NULL
                       ORDER BY o.outcome_id LIMIT %s""", (rules["batch_size"],))
        batch = _rows(cur)
        for row in batch:
            verdict = classify_edit(row, row, rules)
            for f in verdict["fact"]:
                cur.execute("""INSERT INTO email_agent.bp_dq_item (outcome_id, capture_id, family_id, fact_key, source,
                                  value_in_postgres, value_from_reviewer, sent_by)
                               VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s) ON CONFLICT (outcome_id, fact_key) DO NOTHING""",
                            (row["outcome_id"], row["capture_id"], row["family_id"], f["fact_key"],
                             _j({k: f["source"].get(k) for k in ("table", "column", "row_id", "retrieved_at")}),
                             str((f["source"] or {}).get("value", f["was"])), f["now"], row["sent_by"]))
                report["dq_items"] += cur.rowcount
            # The fact rule: a fact edit teaches nothing else. No eval candidate, exemplar or style signal.
            if not verdict["fact"]:
                for rc in verdict["reasoned"]:
                    snap = {"facts": {k: v.get("value") for k, v in (row["facts"] or {}).items()},
                            "reasoned": row["reasoned"], "tone": row["tone_variables"], "brief": row["brief"]}
                    cur.execute("""INSERT INTO email_agent.bp_eval_candidate (outcome_id, capture_id, family_id, correction_key,
                                      direction, from_value, to_value, snapshot, draft_text, sent_by)
                                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s) ON CONFLICT (outcome_id, correction_key) DO NOTHING""",
                                (row["outcome_id"], row["capture_id"], row["family_id"], rc["key"], rc["direction"],
                                 rc["from"], rc["to"], _j(snap), row["draft_text"], row["sent_by"]))
                    report["eval_candidates"] += cur.rowcount
                if verdict["exemplar"]:
                    review_after = _add_months(now.date(), rules["exemplar_review_months"])
                    cur.execute("""INSERT INTO email_agent.bp_exemplar_candidate (capture_id, outcome_id, family_id, tone_variables,
                                      author, reviewed_by, edit_distance, judge_overall, draft_text)
                                   VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s) ON CONFLICT (capture_id) DO NOTHING""",
                                (row["capture_id"], row["outcome_id"], row["family_id"], _j(row["tone_variables"]),
                                 row["sent_by"], row["reviewed_by"], row["edit_distance"],
                                 (row["judge"] or {}).get("overall"), row["draft_text"]))
                    report["exemplar_candidates"] += cur.rowcount
                if row["sent_by"] and not verdict["reasoned"]:
                    touched_users.add(row["sent_by"])
            cur.execute("UPDATE email_agent.bp_draft_outcome SET learning_processed_at = %s, learning_routes = %s::jsonb "
                        "WHERE outcome_id = %s", (now, _j(verdict["routes"]), row["outcome_id"]))
            report["outcomes"] += 1
        report["classifier_examples"] = _classifier_examples(cur)
        report["review_items"] = _review_items(cur, rules, now)
        for user in sorted(touched_users):
            report["style_rule_batches"] += _style_rules(cur, user, rules, now)
        cur.execute("""UPDATE email_agent.bp_exemplar_candidate SET status = 'expired'
                       WHERE status = 'approved' AND review_after < %s""", (now.date(),))
        report["exemplars_expired"] = cur.rowcount
    return report


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    last = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return date(year, month, min(d.day, last))


def _classifier_examples(cur) -> int:
    """A reviewer answered the classifier's question with the OTHER family: a labelled example."""
    cur.execute("""SELECT capture_id, request_text, family_id, assumptions_resolution, assumption_items
                   FROM email_agent.bp_draft_capture
                   WHERE request_text IS NOT NULL AND assumptions_resolution -> 'clarification' ->> 'action' = 'edit'""")
    n = 0
    for r in _rows(cur):
        res = r["assumptions_resolution"]["clarification"]
        item = next((a for a in (r["assumption_items"] or []) if a.get("id") == "clarification"), {})
        label = res.get("value")
        if label not in (item.get("options") or []):
            continue                                   # not one of the two families offered: not a label
        cur.execute("""INSERT INTO email_agent.bp_classifier_example (capture_id, request_text, predicted_family, labeled_family, labeled_by)
                       VALUES (%s,%s,%s,%s,%s) ON CONFLICT (capture_id) DO NOTHING""",
                    (r["capture_id"], r["request_text"], r["family_id"], label, res.get("by")))
        n += cur.rowcount
    return n


def _review_items(cur, rules: Dict[str, Any], now: datetime) -> int:
    opened = 0
    cur.execute("""SELECT family_id, correction_key, direction, count(DISTINCT sent_by) AS users, array_agg(outcome_id) AS outcomes
                   FROM email_agent.bp_eval_candidate
                   WHERE sent_by IS NOT NULL AND created_at >= %s - make_interval(days => %s)
                   GROUP BY 1,2,3 HAVING count(DISTINCT sent_by) >= %s""", (now, rules["window_days"], rules["min_distinct_users"]))
    for r in _rows(cur):
        cur.execute("""INSERT INTO email_agent.bp_review_item (family_id, kind, signature, evidence)
                       VALUES (%s,'reasoning_guidance',%s,%s::jsonb) ON CONFLICT (family_id, kind, signature) WHERE status = 'open' DO NOTHING""",
                    (r["family_id"], f"{r['correction_key']}:{r['direction']}",
                     _j({"distinct_reviewers": r["users"], "outcome_ids": r["outcomes"],
                         "meaning": f"{r['users']} different reviewers each {r['direction']} {r['correction_key']}: review this family's reasoning guidance"})))
        opened += cur.rowcount
    cur.execute("""SELECT c.family_id, count(DISTINCT o.sent_by) AS users, array_agg(o.outcome_id) AS outcomes
                   FROM email_agent.bp_draft_outcome o JOIN email_agent.bp_draft_capture c USING (capture_id)
                   WHERE o.outcome = 'sent' AND o.sent_by IS NOT NULL AND o.learning_routes ? 'wording'
                     AND o.edit_distance >= %s AND o.recorded_at >= %s - make_interval(days => %s)
                   GROUP BY 1 HAVING count(DISTINCT o.sent_by) >= %s""",
                (rules["high_divergence"], now, rules["window_days"], rules["min_distinct_users"]))
    for r in _rows(cur):
        cur.execute("""INSERT INTO email_agent.bp_review_item (family_id, kind, signature, evidence)
                       VALUES (%s,'wording_review','heavy_rewrite',%s::jsonb) ON CONFLICT (family_id, kind, signature) WHERE status = 'open' DO NOTHING""",
                    (r["family_id"], _j({"distinct_reviewers": r["users"], "outcome_ids": r["outcomes"],
                                         "meaning": f"{r['users']} different reviewers heavily rewrote this family's drafts: review its config, "
                                                    "rubric and guardrails. This detects 'many people rewrite it', NOT 'the same correction' "
                                                    "- that needs the edit text, which is not stored."})))
        opened += cur.rowcount
    return opened


def _style_rules(cur, user: str, rules: Dict[str, Any], now: datetime) -> int:
    # Style is what a person does to the WORDING. An edit that changed a fact or one of our judgements was a
    # content correction, and its length change says nothing about their voice, so it is left out.
    cur.execute("""SELECT drafted_words, sent_words, edit_distance FROM email_agent.bp_draft_outcome
                   WHERE outcome = 'sent' AND sent_by = %s AND learning_processed_at IS NOT NULL
                     AND NOT (learning_routes ? 'fact') AND NOT (learning_routes ? 'reasoned')
                   ORDER BY outcome_id DESC LIMIT %s""", (user, rules["style_window_edits"]))
    rows = _rows(cur)
    if len(rows) < rules["min_edits_for_style_rule"]:
        return 0
    ratios = [r["sent_words"] / r["drafted_words"] for r in rows if (r["drafted_words"] or 0) > 0 and r["sent_words"] is not None]
    if not ratios:
        return 0
    median, mean_dist = statistics.median(ratios), statistics.fmean(float(r["edit_distance"] or 0) for r in rows)
    ev = {"edits_considered": len(rows), "median_length_ratio": round(median, 3), "mean_edit_distance": round(mean_dist, 3)}
    proposed: List[Dict[str, Any]] = []
    if median < rules["length_shorten_ratio"]:
        proposed.append({"rule_key": "shorter", "rule_text": f"Draft shorter: your sent emails are usually about {round((1 - median) * 100)}% shorter than drafted."})
    elif median > rules["length_lengthen_ratio"]:
        proposed.append({"rule_key": "longer", "rule_text": f"Draft a little fuller: your sent emails are usually about {round((median - 1) * 100)}% longer than drafted."})
    if mean_dist >= rules["high_divergence"]:
        proposed.append({"rule_key": "rewrites_heavily", "rule_text": "You rewrite drafts heavily, so the drafted voice is probably not yours: consider refreshing your style profile."})
    proposed = proposed[: rules["max_style_rules"]]
    cur.execute("SELECT rule_key, rule_text FROM email_agent.bp_style_rule WHERE sent_by = %s AND status = 'proposed' ORDER BY rule_key", (user,))
    current = [(r["rule_key"], r["rule_text"]) for r in _rows(cur)]
    if current == sorted((p["rule_key"], p["rule_text"]) for p in proposed):
        return 0                                       # nothing new to say
    cur.execute("UPDATE email_agent.bp_style_rule SET status = 'superseded' WHERE sent_by = %s AND status = 'proposed'", (user,))
    batch = uuid.uuid4().hex            # not the clock: two batches may share a timestamp
    for p in proposed:
        cur.execute("""INSERT INTO email_agent.bp_style_rule (sent_by, batch_id, rule_key, rule_text, evidence)
                       VALUES (%s,%s,%s,%s,%s::jsonb)""", (user, batch, p["rule_key"], p["rule_text"], _j(ev)))
    return 1 if proposed else 0


# --- decisions a person makes (the job never makes these) ---------------------------------------------------------

def decide_style_rule(conn: Any, rule_id: int, by: str, action: str, text: Optional[str] = None) -> Dict[str, Any]:
    """The reviewer a rule is ABOUT approves, edits or rejects it. Nobody else may."""
    if action not in ("approve", "edit", "reject") or (action == "edit" and not (text or "").strip()):
        return {"ok": False, "error": "action must be approve, reject, or edit with text"}
    status = {"approve": "approved", "edit": "edited", "reject": "rejected"}[action]
    with conn.cursor() as cur:
        cur.execute("""UPDATE email_agent.bp_style_rule SET status = %s, edited_text = %s, decided_by = %s, decided_at = now()
                       WHERE rule_id = %s AND sent_by = %s AND status IN ('proposed','approved','edited') RETURNING rule_id""",
                    (status, text.strip() if action == "edit" else None, by, rule_id, by))
        row = cur.fetchone()
    return {"ok": bool(row)} if row else {"ok": False, "error": "no such rule for you"}


def approve_exemplar(conn: Any, exemplar_id: int, by: str, rules: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
    """A person approves an exemplar candidate. It is then due for re-review after the configured months.

    The author cannot approve their own: promotion needs a second pair of eyes.
    """
    now = now or datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute("""UPDATE email_agent.bp_exemplar_candidate
                       SET status = 'approved', approved_by = %s, approved_at = %s, review_after = %s
                       WHERE exemplar_id = %s AND status = 'candidate' AND author IS DISTINCT FROM %s RETURNING exemplar_id""",
                    (by, now, _add_months(now.date(), rules["exemplar_review_months"]), exemplar_id, by))
        return {"ok": bool(cur.fetchone())}
