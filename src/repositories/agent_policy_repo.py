"""Agent policies and their versions.

    draft --activate--> live --retire--> retired
      ^                  |  (a new draft beside a live version leaves the live one live)
      +----- save -------+

Every save inserts a version row; rows are immutable (DB trigger). get_conn() is AUTOCOMMIT,
so each transition switches autocommit off and commits once: the version row and the
pointer move together or not at all. base_version is the optimistic lock: a save based on
anything but the latest version is refused, never merged silently.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services import output_safety as osafe
from services.agent_policy import conflict_cases, contract, readiness
from services.agent_policy.compiler import compile_policy
from services.agent_policy.registry import load_registry
from services.agent_policy.settings import load_settings


class StaleVersion(Exception):
    pass


class NotReady(Exception):
    def __init__(self, problems: List[Dict[str, Any]]):
        super().__init__(f"{len(problems)} problem(s)")
        self.problems = problems


class NotFound(Exception):
    pass


class InvalidTransition(Exception):
    pass


_activation_problems = readiness.activation_problems

CONTRACT_FAILED = ("The generated policy failed its final check. "
                   "An administrator can see the details in the technical view.")
WITHHELD = [{"field": "form", "code": "withheld_text",
             "message": "Some text could not be shown, so nothing was saved. Reload the policy and try again."}]


def _holds_withheld_text(value: Any) -> bool:
    """Does any string anywhere in the form equal what the output-safety scrubber puts in place
    of text it would not show? Such a form was loaded through a scrubbed answer; saving it
    would write the marker into an immutable version as if a person had typed it."""
    if isinstance(value, str):
        return value in (osafe.SAFE_FIELD, osafe.SAFE_REPLY)
    if isinstance(value, dict):
        return any(_holds_withheld_text(k) or _holds_withheld_text(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_holds_withheld_text(v) for v in value)
    return False


def _refuse_withheld(form: Dict[str, Any]) -> None:
    if _holds_withheld_text(form):
        raise NotReady([dict(p) for p in WITHHELD])
_contract_problems = contract.validate


def attribute_confirmation(new_form: Dict[str, Any], previous_form: Optional[Dict[str, Any]], *,
                           actor: str, now_iso: str) -> Dict[str, Any]:
    """Return a copy of new_form whose `checked` is what the server records, never what the browser sent.

    No confirmation -> None. The very same confirmation carried forward unchanged (and not cleared by an
    edit to what it vouches for) -> the earlier one is kept. A stale confirmation resent after a clearing edit -> None (the edit cleared it). Anything else is a fresh
    tick, attributed to the authenticated caller.
    """
    out = dict(new_form)
    sent = new_form.get("checked")
    prev = (previous_form or {}).get("checked")
    if not sent:
        out["checked"] = None
    elif prev and sent == prev:
        # the browser resent the earlier confirmation: keep it, unless the edit cleared it
        out["checked"] = None if readiness.confirmation_cleared(previous_form, new_form) else prev
    else:
        out["checked"] = {"by": actor, "at": now_iso}
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _area(cur, name: Optional[str]) -> Dict[str, Any]:
    if name:
        cur.execute("SELECT area_name, never_suggest FROM proc.bp_business_area WHERE area_name = %s", (name,))
        row = cur.fetchone()
        if row:
            return {"area_name": row[0], "never_suggest": row[1]}
    cur.execute("SELECT area_name, never_suggest FROM proc.bp_business_area WHERE is_unassigned")
    row = cur.fetchone()
    return {"area_name": row[0], "never_suggest": row[1]}


def never_suggest_for(conn, area_name: Optional[str]) -> bool:
    """The business area's never_suggest, resolved exactly as a save resolves it."""
    return bool(_area(conn.cursor(), area_name)["never_suggest"])


def _real_area(cur, name: Optional[str]) -> Optional[str]:
    """The area's name if it names a real business area, else None (never trust the raw form value)."""
    if not name:
        return None
    cur.execute("SELECT area_name FROM proc.bp_business_area WHERE area_name = %s", (name,))
    row = cur.fetchone()
    return row[0] if row else None


def _allocate(cur, area_name: str) -> str:
    cur.execute("UPDATE proc.bp_business_area SET last_number = last_number + 1 "
                "WHERE area_name = %s RETURNING id_prefix, last_number", (area_name,))
    prefix, number = cur.fetchone()
    return f"{prefix}-{number:04d}"


def _compile(cur, key, version, saved_as, form, document_text):
    """Compile and contract-check a form without writing anything."""
    settings = load_settings(cur.connection)
    registry = load_registry(cur.connection)
    area = _area(cur, form.get("businessArea"))
    compiled = compile_policy(form, policy_key=key, version=version, status=saved_as,
                              settings=settings, never_suggest=area["never_suggest"])
    problems = _contract_problems(compiled, registry)
    confidence = readiness.extraction_confidence(form, document_text, registry, settings)
    return compiled, problems, confidence


def _write_version(cur, key, version, saved_as, form, actor, note, document_text):
    compiled, problems, confidence = _compile(cur, key, version, saved_as, form, document_text)
    cur.execute(
        "INSERT INTO proc.bp_agent_policy_version (policy_key, version, saved_as, form_state, compiled,"
        " problems, confidence, change_note, saved_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (key, version, saved_as, json.dumps(form), json.dumps(compiled), json.dumps(problems),
         json.dumps(confidence) if confidence else None, note, actor))
    return compiled, problems


def _source_text(cur, form: Dict[str, Any], document_id: Optional[int] = None) -> Optional[str]:
    """The stored text of the document version the form says it came from, so a person's save
    is judged against the same document as the agent's (confidence stays what it was).

    The document is the policy's ``source_document_id`` when known, else the one titled as
    the form's source. Only text already parsed is used: a save never reads S3.
    """
    src = form.get("source") or {}
    try:
        version = int(src.get("documentVersion"))
    except (TypeError, ValueError):
        return None
    if document_id is None:
        if not src.get("document"):
            return None
        cur.execute("SELECT document_id FROM proc.bp_policy_document WHERE title = %s"
                    " ORDER BY document_id LIMIT 1", (src.get("document"),))
        row = cur.fetchone()
        if not row:
            return None
        document_id = row[0]
    cur.execute("SELECT parsed_text FROM proc.bp_policy_document_version WHERE document_id = %s AND version = %s",
                (document_id, version))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def _txn(conn):
    conn.autocommit = False
    return conn.cursor()


def create_draft(conn, form: Dict[str, Any], *, actor: str, source: Optional[Dict[str, Any]] = None,
                 document_text: Optional[str] = None) -> Dict[str, Any]:
    """``source`` ({"documentId", "reference", "split"}) records which document clause the policy
    was extracted from, in the same transaction; ``document_text`` lets the save compute the
    extraction confidence. A person's save passes neither, and the text is then the stored
    text of the form's source document version (``_source_text``)."""
    _refuse_withheld(form)
    src = source or {}
    cur = _txn(conn)
    try:
        form = attribute_confirmation(form, None, actor=actor, now_iso=_now_iso())
        if document_text is None:
            document_text = _source_text(cur, form, src.get("documentId"))
        area = _area(cur, form.get("businessArea"))
        key = _allocate(cur, area["area_name"])
        cur.execute("INSERT INTO proc.bp_agent_policy (policy_key, area_name, status, latest_version, created_by,"
                    " source_document_id, source_reference, source_split) VALUES (%s,%s,'draft',1,%s,%s,%s,%s)",
                    (key, _real_area(cur, form.get("businessArea")), actor,
                     src.get("documentId"), src.get("reference"), src.get("split")))
        _write_version(cur, key, 1, "draft", form, actor, form.get("changeNote") or "", document_text)
        conn.commit()
        return {"policyKey": key, "version": 1}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True


def _lock(cur, key):
    cur.execute("SELECT status, live_version, latest_version FROM proc.bp_agent_policy WHERE policy_key = %s FOR UPDATE", (key,))
    row = cur.fetchone()
    if not row:
        raise NotFound(key)
    return {"status": row[0], "live_version": row[1], "latest_version": row[2]}


def _set_source(cur, policy_key: str, source: Optional[Dict[str, Any]]) -> None:
    if source is not None:
        cur.execute("UPDATE proc.bp_agent_policy SET source_reference = %s, source_split = %s WHERE policy_key = %s",
                    (source.get("reference"), source.get("split"), policy_key))


def update_source(conn, policy_key: str, *, reference: Optional[str], split: Optional[str]) -> None:
    """Point a policy at its clause's current number (a renumbered clause, matched unchanged).
    One statement: atomic under autocommit."""
    _set_source(conn.cursor(), policy_key, {"reference": reference, "split": split})


def save_version(conn, policy_key: str, form: Dict[str, Any], *, base_version: int, intent: str,
                 actor: str, change_note: str, document_text: Optional[str] = None,
                 source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``source`` ({"reference", "split"}), from an extraction run, moves the policy's clause
    reference with the new version in the same transaction (a renumbered clause)."""
    if intent not in ("draft", "activate"):
        raise ValueError(intent)
    _refuse_withheld(form)
    cur = _txn(conn)
    try:
        row = _lock(cur, policy_key)
        if row["latest_version"] != base_version:
            raise StaleVersion(f"latest is {row['latest_version']}, edit was based on {base_version}")
        cur.execute("SELECT form_state FROM proc.bp_agent_policy_version WHERE policy_key=%s AND version=%s",
                    (policy_key, base_version))
        prev = cur.fetchone()
        if not prev:
            raise NotFound(policy_key)
        prev_form = json.loads(prev[0]) if isinstance(prev[0], str) else prev[0]
        form = attribute_confirmation(form, prev_form, actor=actor, now_iso=_now_iso())
        if document_text is None:
            cur.execute("SELECT source_document_id FROM proc.bp_agent_policy WHERE policy_key = %s", (policy_key,))
            document_text = _source_text(cur, form, cur.fetchone()[0])
        version = base_version + 1
        if intent == "activate":
            settings, registry = load_settings(conn), load_registry(conn)
            problems = list(_activation_problems(form, registry, settings))
            if not problems:
                # The final check only speaks when the form itself is ready: its findings restate
                # the form's problems in the compiled document's terms, and a person can act on
                # neither. One problem, counted; the details are in the Admin technical view.
                _, contract_problems, _ = _compile(cur, policy_key, version, "live", form, document_text)
                if contract_problems:
                    problems.append({"field": "registry", "code": "contract", "count": len(set(contract_problems)),
                                     "routeTo": "administrator", "message": CONTRACT_FAILED})
            if problems:
                raise NotReady(problems)   # the full list, in one go; nothing written
            _write_version(cur, policy_key, version, "live", form, actor, change_note, document_text)
            cur.execute("UPDATE proc.bp_agent_policy SET status='live', live_version=%s, latest_version=%s,"
                        " area_name=COALESCE(%s, area_name) WHERE policy_key=%s",
                        (version, version, _real_area(cur, form.get("businessArea")), policy_key))
        else:
            _write_version(cur, policy_key, version, "draft", form, actor, change_note, document_text)
            cur.execute("UPDATE proc.bp_agent_policy SET latest_version=%s, area_name=COALESCE(%s, area_name)"
                        " WHERE policy_key=%s", (version, _real_area(cur, form.get("businessArea")), policy_key))
        _set_source(cur, policy_key, source)
        conn.commit()
        return {"policyKey": policy_key, "version": version}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True


def retire(conn, policy_key: str, *, base_version: int, actor: str, change_note: str) -> Dict[str, Any]:
    cur = _txn(conn)
    try:
        row = _lock(cur, policy_key)
        if row["latest_version"] != base_version:
            raise StaleVersion(f"latest is {row['latest_version']}, retire was based on {base_version}")
        if row["status"] == "retired":
            raise InvalidTransition(f"{policy_key} is already retired")
        # record the form that was actually live; a draft-only policy has its latest form
        source_version = row["live_version"] if row["live_version"] is not None else base_version
        cur.execute("SELECT form_state FROM proc.bp_agent_policy_version WHERE policy_key=%s AND version=%s",
                    (policy_key, source_version))
        form = cur.fetchone()[0]
        form = json.loads(form) if isinstance(form, str) else form
        version = base_version + 1
        _write_version(cur, policy_key, version, "retired", form, actor, change_note, None)
        cur.execute("UPDATE proc.bp_agent_policy SET status='retired', live_version=NULL, latest_version=%s"
                    " WHERE policy_key=%s", (version, policy_key))
        conn.commit()
        return {"policyKey": policy_key, "version": version}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True


def _j(v):
    return json.loads(v) if isinstance(v, str) else v


def get_policy(conn, policy_key: str) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT status, live_version, latest_version, area_name FROM proc.bp_agent_policy WHERE policy_key=%s", (policy_key,))
    head = cur.fetchone()
    if not head:
        raise NotFound(policy_key)
    cur.execute("SELECT version, saved_as, saved_by, saved_at, change_note, form_state, compiled, problems, confidence"
                " FROM proc.bp_agent_policy_version WHERE policy_key=%s ORDER BY version", (policy_key,))
    versions = [{"version": r[0], "savedAs": r[1], "savedBy": r[2], "savedAt": r[3].isoformat(),
                 "changeNote": r[4], "form": _j(r[5]), "compiled": _j(r[6]), "problems": _j(r[7]),
                 "confidence": _j(r[8])} for r in cur.fetchall()]
    for v in versions:
        if head[1] is not None and v["version"] == head[1] and isinstance(v["compiled"], dict):
            v["compiled"] = conflict_cases.overlay(cur, [v["compiled"]])[0]   # what the orchestrator is fed
    history = conflict_cases.history_for(cur, policy_key, head[2], head[0])
    return {"policyKey": policy_key, "status": head[0], "liveVersion": head[1], "latestVersion": head[2],
            "areaName": head[3], "versions": versions, "conflicts": history["conflicts"],
            "pendingConflictAction": history["pendingAction"]}


def list_policies(conn) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT p.policy_key, p.status, p.live_version, p.latest_version, v.form_state, v.confidence, v.problems"
        " FROM proc.bp_agent_policy p JOIN proc.bp_agent_policy_version v"
        "   ON v.policy_key = p.policy_key AND v.version = p.latest_version ORDER BY p.policy_key")
    rows = cur.fetchall()
    open_cases = conflict_cases.open_cases_by_policy(cur)
    out = []
    for key, status, live, latest, form, conf, problems in rows:
        form = _j(form) or {}
        src = form.get("source") or {}
        out.append({"policyKey": key, "status": status, "liveVersion": live, "latestVersion": latest,
                    "name": form.get("name"), "category": form.get("category"),
                    "businessArea": form.get("businessArea"), "subArea": form.get("subArea"),
                    "outcome": form.get("outcome"),
                    "source": {"document": src.get("document"), "documentVersion": src.get("documentVersion"),
                               "reference": src.get("reference")},
                    "confidence": _j(conf), "problemsCount": len(_j(problems) or []),
                    "openConflicts": list(open_cases.get(key, []))})
    return out


def live_documents(conn) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT v.compiled FROM proc.bp_agent_policy p JOIN proc.bp_agent_policy_version v"
                " ON v.policy_key = p.policy_key AND v.version = p.live_version"
                " WHERE p.status = 'live' ORDER BY p.policy_key")
    # standing rules are overlaid at read time: version rows are immutable (the orchestrator feed
    # and live_policies.load both read through here)
    return conflict_cases.overlay(cur, [_j(r[0]) for r in cur.fetchall()])
