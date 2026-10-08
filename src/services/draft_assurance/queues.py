"""What the learning job leaves for people, as lists a screen can show.

Every function takes a connection and returns plain dicts. Two rules shape what they return:

* Internal names stay inside: a data-quality item says "Supplier current offer" and a row id, never a table or
  column. (The output-safety filter would withhold such a field anyway; better that it is never produced.)
* Raw text stays out of listings. An exemplar's text is a separate call (``exemplar_detail``) that the API gates
  harder, and an eval candidate's draft text is never returned at all.

An unknown status is an error, not an empty list: "nothing waiting" and "you asked for a state that does not
exist" must not look the same.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

MAX_LIMIT = 200
STATUSES = {
    "dq": ("open", "resolved", "dismissed"),
    "review": ("open", "accepted", "dismissed"),
    "exemplar": ("candidate", "approved", "rejected", "expired"),
    "eval": ("candidate", "exported", "rejected"),
    "classifier": ("candidate", "exported", "rejected"),
    "flag": ("open", "confirmed_fraud", "cleared"),
}


def _limit(limit: Any) -> int:
    try:
        return max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        return 50


def _status(kind: str, status: Optional[str], default: str) -> Optional[str]:
    if status is None:
        return None
    status = status or default
    if status not in STATUSES[kind]:
        raise ValueError(f"status must be one of {list(STATUSES[kind])}, got {status!r}")
    return status


def _label(key: Any) -> str:
    text = str(key or "").replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def _rows(cur, sql: str, params: tuple) -> List[tuple]:
    cur.execute(sql, params)
    return cur.fetchall()


def _iso(v: Any) -> Optional[str]:
    return v.isoformat() if hasattr(v, "isoformat") else v


def counts(conn: Any, user: Optional[str]) -> Dict[str, int]:
    """How much is waiting in each queue. Style rules are the CALLER's own."""

    q = {
        "data_quality": ("SELECT count(*) FROM email_agent.bp_dq_item WHERE status = 'open'", ()),
        "review_items": ("SELECT count(*) FROM email_agent.bp_review_item WHERE status = 'open'", ()),
        "style_rules": ("SELECT count(*) FROM email_agent.bp_style_rule WHERE status = 'proposed' AND sent_by = %s", (user or "",)),
        "exemplars": ("SELECT count(*) FROM email_agent.bp_exemplar_candidate WHERE status = 'candidate'", ()),
        "eval_candidates": ("SELECT count(*) FROM email_agent.bp_eval_candidate WHERE status = 'candidate'", ()),
        "classifier_examples": ("SELECT count(*) FROM email_agent.bp_classifier_example WHERE status = 'candidate'", ()),
        "inbound_flags": ("SELECT count(*) FROM email_agent.bp_inbound_flag WHERE status = 'open'", ()),
    }
    out = {}
    with conn.cursor() as cur:
        for name, (sql, params) in q.items():
            out[name] = int(_rows(cur, sql, params)[0][0])
    return out


def list_data_quality(conn: Any, status: Optional[str] = "open", limit: Any = 50) -> List[Dict[str, Any]]:
    st = _status("dq", status, "open")
    where, params = ("WHERE status = %s", (st,)) if st else ("", ())
    with conn.cursor() as cur:
        rows = _rows(cur, f"SELECT dq_id, family_id, fact_key, source, value_in_postgres, value_from_reviewer, sent_by, status, note, "
                          f"created_at, resolved_by, resolved_at FROM email_agent.bp_dq_item {where} ORDER BY dq_id DESC LIMIT %s", params + (_limit(limit),))
    out = []
    for r in rows:
        src = r[3] if isinstance(r[3], dict) else {}
        out.append({"id": r[0], "family": r[1], "fact": _label(r[2]), "row_id": src.get("row_id"), "retrieved_at": src.get("retrieved_at"),
                    "value_in_postgres": r[4],
                    "value_from_reviewer": {"value": r[5], "verified": False},      # a reviewer's replacement is a claim, never a fact
                    "sent_by": r[6], "status": r[7], "note": r[8], "created_at": _iso(r[9]), "resolved_by": r[10], "resolved_at": _iso(r[11])})
    return out


def list_review_items(conn: Any, status: Optional[str] = "open", limit: Any = 50) -> List[Dict[str, Any]]:
    st = _status("review", status, "open")
    where, params = ("WHERE status = %s", (st,)) if st else ("", ())
    with conn.cursor() as cur:
        rows = _rows(cur, f"SELECT review_id, family_id, kind, signature, evidence, status, opened_at, decided_by, decided_at "
                          f"FROM email_agent.bp_review_item {where} ORDER BY review_id DESC LIMIT %s", params + (_limit(limit),))
    return [{"id": r[0], "family": r[1], "kind": _label(r[2]), "signature": r[3], "evidence": r[4], "status": r[5],
             "opened_at": _iso(r[6]), "decided_by": r[7], "decided_at": _iso(r[8])} for r in rows]


def list_style_rules(conn: Any, user: Optional[str], limit: Any = 50) -> List[Dict[str, Any]]:
    """The caller's OWN rules (any state they can still act on or have decided). Nobody lists anybody else's."""

    if not user:
        return []
    with conn.cursor() as cur:
        rows = _rows(cur, "SELECT rule_id, rule_key, rule_text, edited_text, status, evidence, generated_at, decided_at "
                          "FROM email_agent.bp_style_rule WHERE sent_by = %s AND status <> 'superseded' ORDER BY rule_id DESC LIMIT %s",
                     (str(user), _limit(limit)))
    return [{"id": r[0], "key": r[1], "text": (r[3] if r[4] == "edited" and r[3] else r[2]), "original_text": r[2],
             "status": r[4], "evidence": r[5], "generated_at": _iso(r[6]), "decided_at": _iso(r[7])} for r in rows]


def list_exemplars(conn: Any, status: Optional[str] = "candidate", limit: Any = 50) -> List[Dict[str, Any]]:
    st = _status("exemplar", status, "candidate")
    where, params = ("WHERE status = %s", (st,)) if st else ("", ())
    with conn.cursor() as cur:
        rows = _rows(cur, f"SELECT exemplar_id, family_id, author, reviewed_by, edit_distance, judge_overall, status, approved_by, approved_at, "
                          f"review_after FROM email_agent.bp_exemplar_candidate {where} ORDER BY exemplar_id DESC LIMIT %s", params + (_limit(limit),))
    return [{"id": r[0], "family": r[1], "author": r[2], "reviewed_by": r[3],
             "edit_distance": float(r[4]) if r[4] is not None else None, "judge_overall": float(r[5]) if r[5] is not None else None,
             "status": r[6], "approved_by": r[7], "approved_at": _iso(r[8]), "review_after": _iso(r[9])} for r in rows]


def exemplar_detail(conn: Any, exemplar_id: int) -> Optional[Dict[str, Any]]:
    """One exemplar WITH its text. The text is raw email content: the API gates this call harder than the listing."""

    with conn.cursor() as cur:
        rows = _rows(cur, "SELECT exemplar_id, family_id, author, status, draft_text FROM email_agent.bp_exemplar_candidate WHERE exemplar_id = %s",
                     (int(exemplar_id),))
    if not rows:
        return None
    r = rows[0]
    return {"id": r[0], "family": r[1], "author": r[2], "status": r[3], "text": r[4]}


def list_eval_candidates(conn: Any, status: Optional[str] = "candidate", limit: Any = 50) -> List[Dict[str, Any]]:
    st = _status("eval", status, "candidate")
    where, params = ("WHERE status = %s", (st,)) if st else ("", ())
    with conn.cursor() as cur:
        rows = _rows(cur, f"SELECT eval_id, family_id, correction_key, direction, from_value, to_value, sent_by, status, created_at "
                          f"FROM email_agent.bp_eval_candidate {where} ORDER BY eval_id DESC LIMIT %s", params + (_limit(limit),))
    return [{"id": r[0], "family": r[1], "correction": str(r[2]).replace("_", " "), "direction": r[3], "from_value": r[4], "to_value": r[5],
             "sent_by": r[6], "status": r[7], "created_at": _iso(r[8])} for r in rows]


def list_classifier_examples(conn: Any, status: Optional[str] = "candidate", limit: Any = 50) -> List[Dict[str, Any]]:
    st = _status("classifier", status, "candidate")
    where, params = ("WHERE status = %s", (st,)) if st else ("", ())
    with conn.cursor() as cur:
        rows = _rows(cur, f"SELECT example_id, request_text, predicted_family, labeled_family, labeled_by, status, created_at "
                          f"FROM email_agent.bp_classifier_example {where} ORDER BY example_id DESC LIMIT %s", params + (_limit(limit),))
    return [{"id": r[0], "request": r[1], "predicted_family": r[2], "labeled_family": r[3], "labeled_by": r[4], "status": r[5],
             "created_at": _iso(r[6])} for r in rows]


_FLAG_LABELS = {"payment_detail_change": "Asks for new or changed payment details",
                "bank_details_with_pressure": "Bank details with pressure language",
                "auth_failed": "The sender failed authentication",
                "auth_missing": "The sender could not be authenticated",
                "domain_mismatch": "The sender's domain is not the supplier's"}


def list_inbound_flags(conn: Any, status: Optional[str] = "open", limit: Any = 50) -> List[Dict[str, Any]]:
    """Replies a person must look at. Names the message by id and dispatch; carries signals and keywords, never email text."""

    st = _status("flag", status, "open")
    where, params = ("WHERE status = %s", (st,)) if st else ("", ())
    with conn.cursor() as cur:
        rows = _rows(cur, f"SELECT flag_id, workflow_id, unique_id, supplier_id, response_message_id, kinds, terms, status, created_at, "
                          f"decided_by, decided_at, note FROM email_agent.bp_inbound_flag {where} ORDER BY flag_id DESC LIMIT %s",
                     params + (_limit(limit),))
    out = []
    for r in rows:
        kinds = r[5] if isinstance(r[5], list) else []
        signals = [_FLAG_LABELS.get(k, _label(k)) for k in kinds]
        out.append({"id": r[0], "workflow_id": r[1], "dispatch_id": r[2], "supplier_id": r[3], "message_id": r[4],
                    "what": signals[0] if signals else "Needs review", "signals": signals, "keywords": r[6], "status": r[7],
                    "created_at": _iso(r[8]), "decided_by": r[9], "decided_at": _iso(r[10]), "note": r[11]})
    return out
