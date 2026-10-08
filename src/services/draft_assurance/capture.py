"""Record what an assured draft rested on, and what became of it. Never raises.

Two writes, both into ``email_agent`` and nowhere else:

* ``record_draft``  when a draft is stored: the assurance record, and the MODEL'S text.
* ``record_sent``   when it is sent: an edit score and the figures that changed.

The sent text itself is never stored. It is compared here and dropped.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import validator as V

logger = logging.getLogger(__name__)

_TAG = re.compile(r"<[^>]+>")
_MARKER = re.compile(r"<!--.*?-->", re.DOTALL)


def plain(text: Optional[str]) -> str:
    """Markup, hidden dispatch markers and entities removed, so draft and sent compare."""

    out = _MARKER.sub(" ", text or "")
    out = re.sub(r"<(br|/p|/div|/li)\s*/?>", "\n", out, flags=re.IGNORECASE)
    return re.sub(r"[ \t]+", " ", html.unescape(_TAG.sub(" ", out))).strip()


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


def record_draft(conn: Any, draft: Dict[str, Any]) -> Optional[int]:
    """Store the capture row for a draft that carries an assurance record. Returns its id."""

    a = draft.get("assurance")
    if not isinstance(a, dict) or not draft.get("unique_id"):
        return None
    # Masked before it is stored OR hashed: bank details are never kept, not even in the model's own draft.
    text = mask_bank_details(plain(draft.get("body") or draft.get("text")))[0]
    meta = draft.get("metadata") if isinstance(draft.get("metadata"), dict) else {}
    tone = a.get("tone")
    ex = a.get("exemplars")
    acc = a.get("accountability") or {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO email_agent.bp_draft_capture
                   (unique_id, workflow_id, supplier_id, path, family_id, family_version, mode,
                    assurance_status, request_text, facts, carried_unverified, conflicts, reasoned,
                    assumptions, unverified_figures, violations, repaired, draft_text, draft_hash,
                    initiated_by, initiated_by_kind, family_source, classification, clarification,
                    lookup_keys, user_instruction, tone_variables, tone_sources, exemplar_ids, exemplar_scope,
                    brief, assumption_items, judge, authority, stage_status, ready, steering)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,
                           %s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,
                           %s,%s,%s,%s::jsonb,%s::jsonb,
                           %s::jsonb,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,
                           %s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s::jsonb)
                   RETURNING capture_id""",
                (str(draft["unique_id"]), draft.get("workflow_id"), draft.get("supplier_id"),
                 meta.get("intent") or draft.get("intent"),
                 a.get("family_id") or "unknown", a.get("family_version"), a.get("mode"),
                 a.get("status") or "unassured", (a.get("request_text") or None),
                 _json(a.get("facts")), _json(a.get("carried_unverified")), _json(a.get("conflicts")),
                 _json(a.get("reasoned")), _json(a.get("assumptions")),
                 _json(a.get("unverified_figures")), _json(a.get("violations")),
                 a.get("repaired"), text, hashlib.sha256(text.encode()).hexdigest(),
                 acc.get("initiated_by"), acc.get("kind"), a.get("family_source"),
                 _nj(a.get("classification")), _nj(a.get("clarification")),
                 _nj(a.get("lookup_keys")), a.get("user_instruction"),
                 _nj(tone["variables"]) if tone else None, _nj(tone["sources"]) if tone else None,
                 _nj(ex["ids"]) if ex else None, ex["scope"] if ex else None,
                 _nj(a.get("brief")), _nj(a.get("assumption_items")), _nj(a.get("judge")),
                 _nj(a.get("authority")), _nj(a.get("stage_status")), a.get("ready"), _nj(a.get("steering"))),
            )
            row = cur.fetchone()
        return int(row[0]) if row else None
    except Exception:  # noqa: BLE001 - capture must never break drafting
        logger.exception("could not record draft capture for %s", draft.get("unique_id"))
        return None


def _nj(value: Any) -> Optional[str]:
    """JSON for a column, keeping None as SQL NULL ("not captured") and [] / {} as captured-empty."""
    return None if value is None else _json(value)


def _words(text: str) -> List[str]:
    return re.findall(r"[\w'’]+", (text or "").lower())


def word_distance(drafted: str, sent: str) -> float:
    """Word-level edit distance over the longer side: 0.0 identical, 1.0 total rewrite.

    The same measure as ``style.feedback.divergence_score``, kept here because that
    module imports the database layer and a send must not depend on it. A test pins
    the two to the same answers.
    """

    a, b = _words(drafted), _words(sent)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, wa in enumerate(a, start=1):
        current = [i]
        for j, wb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (wa != wb)))
        previous = current
    return round(previous[-1] / max(len(a), len(b)), 3)


_BANK_MASK = "[BANK DETAILS REMOVED]"
_BANK_PATTERNS = (
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("sort_code", re.compile(r"\b\d{2}-\d{2}-\d{2}\b")),
    ("account_number", re.compile(r"(?i)\b(account\s*(?:number|no\.?)\s*:?\s*)\d{6,12}\b")),
)


def mask_bank_details(text: str) -> "tuple[str, Dict[str, int]]":
    """Bank details are never stored. Returns the text with each one replaced, and a count per kind."""

    counts: Dict[str, int] = {}
    for kind, pattern in _BANK_PATTERNS:
        if kind == "account_number":
            text, n = pattern.subn(lambda m: m.group(1) + _BANK_MASK, text)
        else:
            text, n = pattern.subn(_BANK_MASK, text)
        if n:
            counts[kind] = n
    return text, counts


def _pieces(text: str) -> List[str]:
    return re.findall(r"\S+|\s+", text or "")


def diff_ops(drafted: str, sent: str) -> List[List[Any]]:
    """Ops that turn ``drafted`` into ``sent``: ["eq", n] keep n pieces, ["del", text, n], ["ins", text].

    Pieces are words and the whitespace between them, so ``apply_diff`` rebuilds the sent text exactly.
    """

    import difflib

    a, b = _pieces(drafted), _pieces(sent)
    ops: List[List[Any]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            ops.append(["eq", i2 - i1])
            continue
        if i2 > i1:
            ops.append(["del", "".join(a[i1:i2]), i2 - i1])
        if j2 > j1:
            ops.append(["ins", "".join(b[j1:j2])])
    return ops


def apply_diff(drafted: str, ops: List[List[Any]]) -> str:
    """Rebuild the sent text from the draft and its ops (the proof that the stored diff is complete)."""

    a, i, out = _pieces(drafted), 0, []
    for op in ops:
        if op[0] == "eq":
            out.extend(a[i:i + op[1]])
            i += op[1]
        elif op[0] == "del":
            i += op[2]
        elif op[0] == "ins":
            out.append(op[1])
    return "".join(out)


def _tokens(text: str) -> Dict[str, str]:
    """Figures, dates and references in a text, as {normalised value: kind}."""

    out: Dict[str, str] = {}
    for m in V._REF.finditer(text):
        out[re.sub(r"[-/ ]", "", m.group(0)).upper()] = "reference"
    for m in V._DATE.finditer(text):
        out[m.group(0).lower()] = "date"
    for n in V.figures_in(text):
        out[format(n.normalize(), "f")] = "figure"   # "132500", never "1.325E+5"
    return out


def _class(value: str, assurance: Dict[str, Any]) -> str:
    facts = {str(v.get("value")) for v in (assurance.get("facts") or {}).values()}
    reasoned = {str(v.get("value")) for v in (assurance.get("reasoned") or {}).values()}

    def match(pool):
        d = V.to_decimal(value)
        return any((V.to_decimal(p) == d and d is not None) or str(p).lower() == value for p in pool)

    if match(facts):
        return "fact"
    if match(reasoned):
        return "reasoned"
    return "other"


def measure_edit(drafted: str, sent: str, assurance: Dict[str, Any]) -> Dict[str, Any]:
    """Distance, class and changed figures between a draft and what was sent. No text out."""

    # Bank details are masked BEFORE anything is measured: a changed account number would otherwise be stored
    # as a "changed figure" in bp_draft_outcome, which every role that can read outcomes may see.
    a, b = mask_bank_details(plain(drafted))[0], mask_bank_details(plain(sent))[0]
    ta, tb = _tokens(a), _tokens(b)
    removed = [{"value": v, "kind": ta[v], "class": _class(v, assurance)} for v in ta if v not in tb]
    added = [{"value": v, "kind": tb[v], "class": _class(v, assurance)} for v in tb if v not in ta]
    distance = word_distance(a, b)
    classes = {c["class"] for c in removed + added}
    if distance == 0:
        edit_class = "none"
    elif "fact" in classes:
        edit_class = "fact"
    elif "reasoned" in classes:
        edit_class = "reasoned"
    elif classes:
        edit_class = "figure_other"
    else:
        edit_class = "wording"
    return {"edit_distance": distance, "edit_class": edit_class, "removed": removed, "added": added,
            "drafted_words": len(re.findall(r"[\w'’]+", a)), "sent_words": len(re.findall(r"[\w'’]+", b))}


def record_sent(conn: Any, unique_id: Optional[str], sent_body: Optional[str], *,
                reviewed_by: Optional[str] = None, sent_by: Optional[str] = None,
                retention_days: Optional[int] = None) -> Optional[int]:
    """Compare what was sent with the latest capture for ``unique_id`` and store the result.

    With a positive ``retention_days`` the sent text and its diff are kept too (bank details masked), in the
    separately-granted ``bp_draft_sent_text`` table. Without one, no raw text is stored.
    """

    if not unique_id:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT capture_id, draft_text, captured_at,
                          (SELECT count(*) FROM email_agent.bp_draft_capture c2 WHERE c2.unique_id = c.unique_id),
                          facts, reasoned, text_expired_at
                   FROM email_agent.bp_draft_capture c WHERE unique_id = %s
                   ORDER BY captured_at DESC LIMIT 1""",
                (str(unique_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            capture_id, draft_text, captured_at, captures, facts, reasoned, text_expired_at = row
            parse = lambda v: v if isinstance(v, dict) else (json.loads(v) if v else {})  # noqa: E731
            have_draft = text_expired_at is None and bool(draft_text)
            if have_draft:
                m = measure_edit(draft_text, sent_body or "", {"facts": parse(facts), "reasoned": parse(reasoned)})
            else:
                # The model's draft aged out: there is nothing to measure against. NULL = not captured; a made-up
                # distance of 1.0 would teach the learning job that every late send was a total rewrite.
                m = {"edit_distance": None, "edit_class": None, "removed": None, "added": None,
                     "drafted_words": None, "sent_words": len(re.findall(r"[\w'’]+", plain(sent_body)))}
            seconds = None
            if captured_at is not None:
                now = datetime.now(captured_at.tzinfo or timezone.utc)
                seconds = max(0, int((now - captured_at).total_seconds()))
            cur.execute(
                """INSERT INTO email_agent.bp_draft_outcome
                   (capture_id, outcome, edit_distance, edit_class, removed_figures, added_figures,
                    drafted_words, sent_words, regeneration_count, time_to_send_s, reviewed_by, sent_by)
                   VALUES (%s,'sent',%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT DO NOTHING RETURNING outcome_id""",
                (capture_id, m["edit_distance"], m["edit_class"], _nj(m["removed"]), _nj(m["added"]),
                 m["drafted_words"], m["sent_words"], max(0, int(captures) - 1), seconds,
                 reviewed_by, sent_by),
            )
            out = cur.fetchone()
            if out and _keeps_text(retention_days):
                _store_sent_text(conn, cur, int(out[0]), capture_id, draft_text if have_draft else None, sent_body)
        return int(out[0]) if out else None
    except Exception:  # noqa: BLE001 - never block or fail a send over bookkeeping
        logger.exception("could not record send outcome for %s", unique_id)
        return None


def _keeps_text(retention_days: Any) -> bool:
    return isinstance(retention_days, int) and not isinstance(retention_days, bool) and retention_days > 0


def _store_sent_text(conn: Any, cur: Any, outcome_id: int, capture_id: int, draft_text: Optional[str],
                     sent_body: Optional[str]) -> None:
    """Keep the sent text and its diff. A failure here is logged and must not lose the outcome row."""

    use_savepoint = not getattr(conn, "autocommit", True)       # inside a transaction one error would poison the rest
    try:
        if use_savepoint:
            cur.execute("SAVEPOINT sent_text")
        sent_plain, redactions = mask_bank_details(plain(sent_body))
        diff = None
        if draft_text is not None:
            draft_plain, more = mask_bank_details(plain(draft_text))
            for k, v in more.items():
                redactions[k] = redactions.get(k, 0) + v
            diff = diff_ops(draft_plain, sent_plain)
        cur.execute(
            """INSERT INTO email_agent.bp_draft_sent_text (outcome_id, capture_id, sent_text, diff, text_hash, redactions)
               VALUES (%s,%s,%s,%s::jsonb,%s,%s::jsonb) ON CONFLICT DO NOTHING""",
            (outcome_id, capture_id, sent_plain, _nj(diff), hashlib.sha256(sent_plain.encode()).hexdigest(),
             _json(redactions)),
        )
        if use_savepoint:
            cur.execute("RELEASE SAVEPOINT sent_text")
    except Exception:  # noqa: BLE001
        logger.exception("could not store the sent text for outcome %s", outcome_id)
        if use_savepoint:
            try:
                cur.execute("ROLLBACK TO SAVEPOINT sent_text")
            except Exception:  # noqa: BLE001
                pass


# --- the events that follow a draft: abandon, confirm, readiness ---------------------------------

def _latest(cur: Any, unique_id: str, columns: str):
    cur.execute(f"SELECT {columns} FROM email_agent.bp_draft_capture WHERE unique_id = %s "
                "ORDER BY captured_at DESC, capture_id DESC LIMIT 1", (str(unique_id),))
    return cur.fetchone()


def record_abandoned(conn: Any, unique_id: Optional[str], by: Optional[str], reason: Optional[str] = None) -> Optional[int]:
    """The draft was closed without being sent. A draft already sent cannot be abandoned."""

    if not unique_id or not by:
        return None
    try:
        with conn.cursor() as cur:
            row = _latest(cur, unique_id, "capture_id")
            if not row:
                return None
            cur.execute("SELECT 1 FROM email_agent.bp_draft_outcome WHERE capture_id = %s AND outcome = 'sent'", (row[0],))
            if cur.fetchone():
                return None
            cur.execute("INSERT INTO email_agent.bp_draft_outcome (capture_id, outcome, abandoned_by, abandon_reason) "
                        "VALUES (%s, 'abandoned', %s, %s) RETURNING outcome_id",
                        (row[0], str(by), (reason or "")[:500] or None))
            out = cur.fetchone()
        return int(out[0]) if out else None
    except Exception:  # noqa: BLE001
        logger.exception("could not record abandon for %s", unique_id)
        return None


ACTIONS = ("confirm", "edit", "reject")


def confirm_assumptions(conn: Any, unique_id: str, confirmations: List[Dict[str, Any]], by: str) -> Dict[str, Any]:
    """Apply a person's answers to the draft's assumptions and recompute whether it is ready.

    confirm  the assumption stands.     reject  it does not; the draft is not ready.
    edit     the value changes, so the text rests on something stale: needs_redraft, not ready.
    An unknown id, an unknown action, or an edit with no value is refused whole -- nothing is half applied.
    """

    if not by:
        return {"ok": False, "error": "no reviewer identified"}
    with conn.cursor() as cur:
        row = _latest(cur, unique_id, "capture_id, assumption_items, assumptions_resolution")
        if not row:
            return {"ok": False, "error": "no such draft"}
        cid, items, res = row
        parse = lambda v, d: v if isinstance(v, type(d)) else (json.loads(v) if v else d)  # noqa: E731
        items, res = parse(items, []), parse(res, {})
        ids = {a["id"] for a in items}
        now = datetime.now(timezone.utc).isoformat()
        staged: Dict[str, Any] = {}
        for c in confirmations or []:
            cid_, action = (c or {}).get("id"), (c or {}).get("action")
            if cid_ not in ids:
                return {"ok": False, "error": f"unknown assumption {cid_!r}"}
            if action not in ACTIONS:
                return {"ok": False, "error": f"action must be one of {list(ACTIONS)}"}
            if action == "edit" and (c.get("value") in (None, "")):
                return {"ok": False, "error": "an edit needs a value"}
            staged[cid_] = {"action": action, "by": str(by), "at": now,
                            **({"value": c["value"]} if action == "edit" else {})}
        res.update(staged)
        unresolved = [a["id"] for a in items if a["id"] not in res]
        blocked = [k for k, v in res.items() if v["action"] in ("reject", "edit")]
        ready = not unresolved and not blocked      # the classifier's question is one of the items
        cur.execute("UPDATE email_agent.bp_draft_capture SET assumptions_resolution = %s::jsonb, ready = %s, "
                    "needs_redraft = %s, ready_at = CASE WHEN %s THEN now() ELSE NULL END WHERE capture_id = %s",
                    (_json(res), ready, bool(blocked), ready, cid))
    return {"ok": True, "ready": ready, "unresolved": unresolved, "needs_redraft": bool(blocked)}


def readiness(conn: Any, unique_id: Optional[str]) -> Dict[str, Any]:
    """Whether the latest capture for ``unique_id`` is ready to send. ``ready`` is None if it cannot be known."""

    if not unique_id:
        return {"ready": None, "reason": "no draft id"}
    try:
        with conn.cursor() as cur:
            row = _latest(cur, unique_id, "ready, needs_redraft")
    except Exception as exc:  # noqa: BLE001
        return {"ready": None, "reason": f"readiness unreadable: {type(exc).__name__}"}
    if not row:
        return {"ready": None, "reason": "draft was never captured"}
    return {"ready": bool(row[0]), "needs_redraft": bool(row[1])}


# --- the reviewer-facing view ---------------------------------------------------------------------

_VIEW_COLUMNS = ("capture_id", "unique_id", "family_id", "family_version", "mode", "assurance_status", "family_source",
                 "facts", "carried_unverified", "conflicts", "reasoned", "unverified_figures", "violations",
                 "brief", "assumption_items", "assumptions_resolution", "judge", "authority", "tone_variables",
                 "tone_sources", "exemplar_ids", "exemplar_scope", "clarification", "classification", "stage_status",
                 "ready", "needs_redraft", "initiated_by", "initiated_by_kind", "repaired")


def load_raw(conn: Any, unique_id: str) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        row = _latest(cur, unique_id, ", ".join(_VIEW_COLUMNS))
    if not row:
        return None
    out = dict(zip(_VIEW_COLUMNS, row))
    for k, v in list(out.items()):
        if isinstance(v, str) and k not in ("unique_id", "family_id", "mode", "assurance_status", "family_source",
                                            "exemplar_scope", "initiated_by", "initiated_by_kind"):
            try:
                out[k] = json.loads(v)
            except ValueError:
                pass
    return out


def to_view(raw: Dict[str, Any], reviewed_by: Optional[str] = None) -> Dict[str, Any]:
    """What a reviewer's screen may see. Facts carry a human label and a row id, never an internal table or column name."""

    res = raw.get("assumptions_resolution") or {}
    facts = {k: {"value": f.get("value"), "label": f.get("label") or k.replace("_", " "), "source": "postgres",
                 "row_id": f.get("row_id"), "retrieved_at": f.get("retrieved_at")}
             for k, f in (raw.get("facts") or {}).items()}
    for k, v in (raw.get("carried_unverified") or {}).items():
        facts.setdefault(k, {"value": v, "label": k.replace("_", " "), "source": "carried_unverified",
                             "row_id": None, "retrieved_at": None})
    items = []
    for a in raw.get("assumption_items") or []:
        items.append({**a, "resolution": res.get(a["id"], a.get("resolution"))})
    brief = raw.get("brief")
    if brief and isinstance(brief.get("reasoned"), dict):
        # The client may not format numbers, so a 0-1 confidence reaches it as a band.
        for r in brief["reasoned"].values():
            c = r.get("confidence")
            r["confidence_label"] = (None if c is None else "high" if c >= 0.8 else "medium" if c >= 0.5 else "low")
    return {
        "unique_id": raw["unique_id"], "status": raw["assurance_status"], "ready": bool(raw.get("ready")),
        "needs_redraft": bool(raw.get("needs_redraft")),
        "family_id": raw["family_id"], "family_source": raw.get("family_source"), "mode": raw.get("mode"),
        "brief": ({**(raw.get("brief") or {}), "assumptions": items} if raw.get("brief") else None),
        "assumptions": items,
        "facts": facts,
        "unverified_figures": raw.get("unverified_figures") or [],
        "conflicts": [{"fact": c.get("fact"), "label": (facts.get(c.get("fact")) or {}).get("label"),
                       "postgres": c.get("postgres"), "supplied": c.get("supplied"),
                       "resolution": c.get("resolution")} for c in (raw.get("conflicts") or [])],
        "violations": [{"kind": v.get("kind"), "detail": v.get("detail"), "severity": v.get("severity")}
                       for v in (raw.get("violations") or [])],
        "judge": raw.get("judge"), "authority": raw.get("authority"),
        "tone": ({"variables": raw["tone_variables"], "sources": raw["tone_sources"]}
                 if raw.get("tone_variables") else None),
        "clarification": raw.get("clarification"),
        "stage_status": raw.get("stage_status"),
        "accountability": {"initiated_by": raw.get("initiated_by"), "reviewed_by": reviewed_by},
    }
