"""The extraction run's work: read documents, propose policies, save them as drafts.

``run_extract`` and ``run_fix`` are the ``work(conn, run, emit)`` functions run_runner runs.
Every finding is emitted as soon as it is decided, so the screen's list fills during the run.

The agent's output is a proposal: every save is a draft (``intent="draft"``), ``checked``
is None and ``hidden.setBy`` is "extraction_agent", and the actor is the person who started
the run. One policy, one chunk or one document going wrong is an ``error`` item and the run
goes on; only a missing governed prompt fails the whole run.

Revised documents (the user's ruling): an unchanged clause writes nothing; a changed clause
gets a new draft and the live version stays live; a new clause gets a new draft; a clause
that is gone is a "Proposed retire" item and is never retired here.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Callable, Dict, List, Optional

from repositories import agent_policy_repo as repo
from services.agent_policy import converter, documents, extractor, matching
from services.agent_policy.registry import load_registry
from services.agent_policy.sections import chunk_sections, split_sections
from services.agent_policy.settings import load_settings
from services.obligations.grounding import is_quote_grounded

UNGROUNDED_NOTE = "The excerpt was not found word for word in the document."
RETIRE_HELD_NOTE = ("Some sections or policies could not be read, so no policy from this document was proposed "
                    "for retirement. Run the extraction again to check for removed clauses.")
_ACTOR_FALLBACK = "extraction_agent"
_COUNT_KEYS = ("documents", "chunks", "policies", "new", "changed", "unchanged", "proposedRetire",
               "notEnforceable", "errors")

Emit = Callable[..., int]


def _one_line(exc: BaseException) -> str:
    return (" ".join(str(exc).split()) or type(exc).__name__)[:300]


def load_taxonomy(conn) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT area_name, sub_areas, is_unassigned FROM proc.bp_business_area"
                " ORDER BY is_unassigned, area_name")
    return [{"areaName": r[0], "subAreas": list(r[1] or []), "unassigned": bool(r[2])}
            for r in cur.fetchall()]


def _document(conn, document_id) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT title FROM proc.bp_policy_document WHERE document_id = %s", (document_id,))
    row = cur.fetchone()
    return {"title": row[0]} if row else None


def _existing(conn, document_id) -> List[Dict[str, Any]]:
    """The document's current policies. A retired policy is no longer current: it is neither
    matched nor proposed for retirement again.

    ``form`` is the agent's baseline -- the last version an extraction run saved (its
    ``policy`` items record which) -- so a person's later edit is never mistaken for a change
    in the document. A policy with no such item falls back to its latest version. The change
    note and ``hidden.setBy`` are not used: a person's save carries both forward.
    ``latestForm``/``latestVersion`` are what a new draft is based on.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT p.policy_key, p.source_reference, p.source_split, p.latest_version, lv.form_state,"
        "       av.version, av.form_state"
        "  FROM proc.bp_agent_policy p"
        "  JOIN proc.bp_agent_policy_version lv"
        "    ON lv.policy_key = p.policy_key AND lv.version = p.latest_version"
        "  LEFT JOIN LATERAL ("
        "       SELECT i.saved_version FROM proc.bp_policy_extraction_item i"
        "        WHERE i.policy_key = p.policy_key AND i.kind = 'policy'"
        "          AND i.decision IN ('new', 'changed') AND i.saved_version IS NOT NULL"
        "        ORDER BY i.saved_version DESC LIMIT 1) a ON TRUE"
        "  JOIN proc.bp_agent_policy_version av"
        "    ON av.policy_key = p.policy_key AND av.version = COALESCE(a.saved_version, p.latest_version)"
        " WHERE p.source_document_id = %s AND p.status <> 'retired' ORDER BY p.policy_key",
        (document_id,))
    out = []
    for key, ref, split, latest, latest_form, agent_version, agent_form in cur.fetchall():
        out.append({"policyKey": key, "reference": ref, "split": split,
                    "latestVersion": latest, "latestForm": _j(latest_form),
                    "agentVersion": agent_version, "form": _j(agent_form)})
    return out


def _j(value: Any) -> Dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else (value or {})


# Fields a person sets that say nothing about what the document requires: kept when the
# document changes and the policy is re-extracted.
_PERSON_FIELDS = ("owner", "effectiveFrom", "reviewBy")
_NOT_AN_EDIT = ("checked", "changeNote") + _PERSON_FIELDS


def _carry_person_fields(form: Dict[str, Any], old: Dict[str, Any]) -> None:
    for f in _PERSON_FIELDS:
        if old["latestForm"].get(f) != old["form"].get(f):
            form[f] = old["latestForm"].get(f)


def _replaces_edits(old: Dict[str, Any]) -> bool:
    """Did a person edit the policy after the agent's last save (beyond the carried fields)?"""
    if old["agentVersion"] == old["latestVersion"]:
        return False
    strip = lambda f: {k: v for k, v in f.items() if k not in _NOT_AN_EDIT}  # noqa: E731
    return strip(old["latestForm"]) != strip(old["form"])


def _as_proposal(form: Dict[str, Any]) -> Dict[str, Any]:
    """What the code guarantees about every form it saves, whatever the converter returned."""
    out = copy.deepcopy(form)
    out["checked"] = None
    out.setdefault("hidden", {})
    out["hidden"] = dict(out["hidden"] or {}, setBy="extraction_agent")
    return out


def _summary(form: Dict[str, Any]) -> Dict[str, Any]:
    src = form.get("source") or {}
    hidden = form.get("hidden") or {}
    return {"name": form.get("name"), "situation": form.get("situation"), "outcome": form.get("outcome"),
            "reference": src.get("reference"), "excerpt": src.get("excerpt"),
            "split": matching.split_key(form), "agentNotes": list(hidden.get("agentNotes") or []),
            "unknownNames": list(hidden.get("unknownNames") or [])}


def _new_counts() -> Dict[str, int]:
    return {k: 0 for k in _COUNT_KEYS}


def _extract_document(conn, run: Dict[str, Any], emit: Emit, req: Dict[str, Any], counts: Dict[str, int],
                      *, registry, taxonomy) -> None:
    document_id, version = req.get("documentId"), req.get("version")
    at = {"document_id": document_id, "document_version": version}
    actor = run.get("started_by") or _ACTOR_FALLBACK
    counts["documents"] += 1

    doc = _document(conn, document_id)
    if doc is None:
        counts["errors"] += 1
        emit("error", {"message": f"Document {document_id} does not exist."}, **at)
        return
    title = doc["title"]
    try:
        text = documents.document_text(conn, document_id, version)
    except documents.DocumentUnreadable as exc:
        counts["errors"] += 1
        emit("error", {"message": f"The document could not be read: {exc.reason}"}, **at)
        return
    except Exception as exc:  # noqa: BLE001 - one document must not end the run
        counts["errors"] += 1
        emit("error", {"message": f"The document could not be read: {_one_line(exc)}"}, **at)
        return

    chunks = chunk_sections(split_sections(text))
    proposed: List[Dict[str, Any]] = []
    incomplete = False   # a chunk or a policy was lost: removed clauses cannot be told apart
    for chunk in chunks:
        counts["chunks"] += 1
        refs = [s.get("reference") for s in chunk]
        try:
            result = extractor.extract_chunk(chunk, document={"title": title, "version": version},
                                             taxonomy=taxonomy, registry=registry)
        except extractor.PromptUnavailable:
            raise  # a governed prompt is missing: the whole run fails, as the ruling says
        except Exception as exc:  # noqa: BLE001 - one chunk must not end the run
            incomplete = True
            counts["errors"] += 1
            named = ", ".join(r or "(no number)" for r in refs)
            emit("error", {"message": f"Sections {named} could not be read by the agent: {_one_line(exc)}",
                           "references": refs}, reference=refs[0] if refs else None, **at)
            continue
        for ne in result.not_enforceable:
            counts["notEnforceable"] += 1
            emit("not_enforceable", ne.model_dump(), reference=ne.reference, **at)
        for p in result.policies:
            try:
                form = converter.to_form(p, document_title=title, document_version=version,
                                         registry=registry, taxonomy=taxonomy)
            except Exception as exc:  # noqa: BLE001
                incomplete = True
                counts["errors"] += 1
                emit("error", {"message": f"A policy in section {p.reference} could not be converted: "
                                          f"{_one_line(exc)}", "name": p.name},
                     reference=p.reference, **at)
                continue
            form = _as_proposal(form)
            if not is_quote_grounded(p.excerpt, text, min_words=4):
                form["hidden"]["agentNotes"] = list(form["hidden"].get("agentNotes") or []) + [UNGROUNDED_NOTE]
            counts["policies"] += 1
            proposed.append(form)

    try:
        _settle(conn, emit, proposed, counts, incomplete=incomplete, actor=actor, title=title,
                version=version, document_id=document_id, text=text, at=at)
    except Exception as exc:  # noqa: BLE001 - one document must not end the run
        counts["errors"] += 1
        emit("error", {"message": f"The policies from this document could not be matched and saved: "
                                  f"{_one_line(exc)}"}, **at)


def _settle(conn, emit: Emit, proposed: List[Dict[str, Any]], counts: Dict[str, int], *, incomplete: bool,
            actor, title, version, document_id, text, at) -> None:
    """Match the document's proposals with its existing policies and act on each decision."""
    existing = _existing(conn, document_id)
    decisions = matching.match(existing, proposed)
    by_key = {e["policyKey"]: e for e in existing}
    for form, d in zip(proposed, decisions[:len(proposed)]):
        _decide(conn, emit, form, d, by_key, counts, actor=actor, title=title, version=version,
                document_id=document_id, text=text, at=at)

    retire = decisions[len(proposed):]
    if retire and incomplete:
        # A clause in a section the agent could not read (or a policy that could not be
        # converted) is missing from `proposed`, not from the document: proposing its policy
        # for retirement would be a false finding.
        emit("note", {"message": RETIRE_HELD_NOTE,
                      "policyKeys": [d["policyKey"] for d in retire]}, **at)
        return
    for d in retire:
        old = by_key[d["policyKey"]]
        src = (old["form"].get("source") or {})
        counts["proposedRetire"] += 1
        emit("proposed_retire",
             {"policyKey": d["policyKey"], "name": old["form"].get("name"), "excerpt": src.get("excerpt"),
              "reference": old["reference"], "latestVersion": old["latestVersion"]},
             reference=old["reference"], policy_key=d["policyKey"], decision="proposed_retire", **at)


def _decide(conn, emit: Emit, form: Dict[str, Any], d: Dict[str, Any], by_key, counts, *, actor, title,
            version, document_id, text, at) -> None:
    ref = (form.get("source") or {}).get("reference")
    payload = _summary(form)
    decision = d["decision"]
    try:
        if decision == "new":
            form["changeNote"] = f"Extracted from {title} v{version}, section {ref}."
            saved = repo.create_draft(conn, form, actor=actor, document_text=text,
                                      source={"documentId": document_id, "reference": ref,
                                              "split": matching.split_key(form)})
            key, saved_version = saved["policyKey"], saved["version"]
        elif decision == "changed":
            old = by_key[d["policyKey"]]
            note = f"Re-extracted from {title} v{version}, section {ref}."
            if _replaces_edits(old):
                note += f" Replaces the edits in v{old['latestVersion']}."  # history keeps them
            _carry_person_fields(form, old)
            form["changeNote"] = note
            saved = repo.save_version(conn, d["policyKey"], form, base_version=old["latestVersion"],
                                      intent="draft", actor=actor, change_note=note, document_text=text)
            key, saved_version = saved["policyKey"], saved["version"]
        else:  # unchanged: nothing is written
            key, saved_version = d["policyKey"], None
            payload["latestVersion"] = by_key[key]["latestVersion"]
    except repo.StaleVersion:
        counts["errors"] += 1
        emit("error", {"message": f"{d['policyKey']} was edited while the agent was reading the document, "
                                  "so its new draft was not saved. Run the extraction again.",
                       "name": form.get("name")}, reference=ref, policy_key=d["policyKey"], **at)
        return
    except Exception as exc:  # noqa: BLE001 - one policy must not end the run
        counts["errors"] += 1
        emit("error", {"message": f"The policy from section {ref} could not be saved: {_one_line(exc)}",
                       "name": form.get("name")}, reference=ref, policy_key=d.get("policyKey"), **at)
        return
    counts[decision] += 1
    emit("policy", payload, reference=ref, policy_key=key, decision=decision, saved_version=saved_version,
         **at)


def run_extract(conn, run: Dict[str, Any], emit: Emit) -> Dict[str, int]:
    # Settings are not loaded here: each save reads them itself (compile + confidence). The
    # registry and taxonomy are read once so every chunk is judged against one snapshot.
    registry = load_registry(conn)
    taxonomy = load_taxonomy(conn)
    counts = _new_counts()
    for req in (run.get("request") or {}).get("documents") or []:
        _extract_document(conn, run, emit, req, counts, registry=registry, taxonomy=taxonomy)
    return counts


_FIX_FIELDS = ("situation", "hidden", "examples", "outcome", "deciders", "notify")


def run_fix(conn, run: Dict[str, Any], emit: Emit) -> Dict[str, int]:
    """Ask the agent to fix one policy so the flipped examples come out as the reviewer expects.

    Emits one ``fix`` item and never saves: the reviewer accepts it in the form, and the
    normal save follows.
    """
    req = run.get("request") or {}
    key, base = req.get("policyKey"), req.get("baseVersion")
    policy = repo.get_policy(conn, key)
    form = next((v["form"] for v in policy["versions"] if v["version"] == base), None)
    if form is None:
        raise repo.NotFound(f"{key} version {base}")
    registry = load_registry(conn)
    settings = load_settings(conn)
    taxonomy = load_taxonomy(conn)
    proposal = extractor.fix_policy(form, list(req.get("flipped") or []), registry=registry,
                                    taxonomy=taxonomy, settings=settings)
    src = form.get("source") or {}
    fixed = _as_proposal(converter.to_form(proposal, document_title=src.get("document"),
                                           document_version=src.get("documentVersion"),
                                           registry=registry, taxonomy=taxonomy))
    proposed = {k: fixed.get(k) for k in _FIX_FIELDS}
    proposed["messages"] = {"messageForAgent": fixed.get("messageForAgent"),
                            "messageForPerson": fixed.get("messageForPerson")}
    emit("fix", {"proposed": proposed, "basedOn": base}, policy_key=key)
    return {"fixes": 1}
