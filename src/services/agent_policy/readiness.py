"""What stands between a policy and Active, said in the form's own words.

activation_problems() returns EVERY problem at once, ordered as the form is laid out, so the
UI can show one summary and focus problems[0]. A Draft never calls this: a Draft saves with
only a name.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.agent_policy import conditions
from services.agent_policy import deciders as deciders_mod
from services.agent_policy.compiler import response_time
from services.agent_policy.registry import RegistrySnapshot

# Problem messages use product wording only. The screen renders the brief's own sentences
# (which name the orchestrator) from each problem's `code`; variable data lives in keys
# that are not prose (`names`, `missing`), so the output-safety scrubber leaves them alone.
UNKNOWN_NAME = "This policy refers to something that is not recognised. An administrator needs to add it."
CANT_ENFORCE = "Can't be enforced yet: some information is not available when this policy is checked."
_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")
_CONFIRM_KEYS = ("situation", "outcome", "hidden", "messageForPerson", "deciders", "notify")


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, (list, dict)) and not v)


def _seconds(duration: Optional[str]) -> Optional[int]:
    m = _DURATION.match(duration or "")
    if not m or duration in ("P", "PT"):
        return None
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def _human(duration: str) -> str:
    secs = _seconds(duration) or 0
    hours, rem = divmod(secs, 3600)
    if rem == 0 and hours:
        return f"{hours} hour" + ("s" if hours != 1 else "")
    return duration


def _cant_enforce_parts(form: Dict[str, Any], registry: RegistrySnapshot):
    """(plain names of inputs not received at the checkpoint, plain name of a checkpoint not yet checked)."""
    h = form.get("hidden") or {}
    cp = h.get("checkpoint")
    names: List[str] = []
    for i in h.get("inputs") or []:
        if not registry.available(cp, i.get("field")):
            names.append(i.get("name") or i.get("field"))
    listed = {i.get("field") for i in h.get("inputs") or []}
    for f in sorted(conditions.condition_fields(h.get("condition"))):
        row = registry.input_row(cp, f)
        # A field with no registry row is reported as unknown by activation_problems.
        if f not in listed and row and not registry.available(cp, f):
            names.append(row.get("plain") or f)
    for m in h.get("missingInputs") or []:
        names.append(m.get("name"))
    checkpoint = registry.plain(cp) if cp and registry.knows_checkpoint(cp) and not registry.checkpoint_live(cp) else None
    return names, checkpoint


def _cant_enforce(form: Dict[str, Any], registry: RegistrySnapshot) -> List[str]:
    names, checkpoint = _cant_enforce_parts(form, registry)
    out = [f"Can't be enforced yet: the orchestrator does not receive {n} at this point" for n in names]
    if checkpoint:
        out.append(f"Can't be enforced yet: nothing checks policies {checkpoint} yet")
    return out


def how_enforced(form: Dict[str, Any], registry: RegistrySnapshot, settings: Dict[str, Any]) -> Dict[str, Any]:
    cant = _cant_enforce(form, registry)
    if cant:
        names, checkpoint = _cant_enforce_parts(form, registry)
        # cantEnforce keeps the full sentences; the screen builds its own from the names
        # (its wording names the orchestrator, which this service may not say to a person).
        return {"ok": False, "cantEnforce": cant, "cantEnforceNames": names,
                "cantEnforceCheckpoint": checkpoint}
    h = form.get("hidden") or {}
    cp = h.get("checkpoint")
    plain = (h.get("actions") or {}).get("plain") or "act"
    needs = ", ".join(
        f"{i.get('name')} ({i['unit']}, from the action)" if i.get("unit") else f"{i.get('name')} (from the action)"
        for i in h.get("inputs") or [])
    outcome = form.get("outcome")
    person = form.get("messageForPerson")
    tail = f' The agent tells the person: "{person}"' if person else ""
    if outcome == "approve":
        deciders = [d for d in form.get("deciders") or [] if str(d).strip()]
        within, _ = response_time(form, settings)
        if len(deciders) > 1:
            then = f"the action pauses and {deciders[0]} is asked to approve, then {', then '.join(deciders[1:])} if there is no answer within {_human(within)}; if the last level does not answer, the action is rejected."
        else:
            then = f"the action pauses and {deciders[0] if deciders else "nobody yet"} is asked to approve within {_human(within)}; with no answer it is rejected."
    elif outcome == "block":
        then = "the action is refused. Nobody is asked to approve it."
    elif outcome == "notify":
        then = "the action goes ahead and " + ", ".join(form.get("notify") or ["nobody yet"]) + " is told."
    else:
        then = "nothing yet: choose what happens."
    return {"ok": True,
            "checkedWhen": f"the agent is about to do this: {plain} ({registry.plain(cp)})",
            "needsToKnow": needs,
            "then": then + tail}


UNMAPPED = ("Nobody is linked to {names} yet. An administrator must link them before this policy "
            "can be Active.")


def _load_deciders():
    from services.db import get_conn

    with get_conn() as conn:
        return deciders_mod.load_map(conn)


def activation_problems(form: Dict[str, Any], registry: RegistrySnapshot,
                        settings: Dict[str, Any], deciders: Optional[Dict[str, Any]] = None
                        ) -> List[Dict[str, Any]]:
    p: List[Dict[str, Any]] = []
    add = lambda f, c, m, **kw: p.append({"field": f, "code": c, "message": m, **kw})  # noqa: E731
    h = form.get("hidden") or {}
    outcome = form.get("outcome")

    # Form order (brief §3.1): Identity, The policy (situation, how enforced, examples),
    # What happens, Applies to, Governance. problems[0] is therefore the first field to fix.
    if _blank(form.get("name")): add("name", "name_required", "Name is required.")
    if _blank(form.get("businessArea")): add("businessArea", "business_area_required", "Business area is required.")
    if _blank(form.get("subArea")): add("subArea", "sub_area_required", "Sub-area is required.")
    if _blank(form.get("situation")): add("situation", "situation_required", "The situation is required.")

    cp = h.get("checkpoint")
    if _blank(cp): add("checkpoint", "checkpoint_required", "The policy has no checkpoint.")
    cond = h.get("condition")
    unknown = [t for t in sorted(conditions.tool_names(cond)) if not registry.knows_action(cp, t)]
    unknown += [t for t in (h.get("actions") or {}).get("tools") or [] if not registry.knows_action(cp, t)]
    unknown += [f for f in sorted(conditions.condition_fields(cond)) if not registry.input_row(cp, f)]
    unknown += list(h.get("unknownNames") or [])
    if cp and not registry.knows_checkpoint(cp):
        unknown.append(cp)
    if unknown:
        add("registry", "unknown_name", UNKNOWN_NAME, names=sorted(set(unknown)), routeTo="administrator")
    missing, not_checked = _cant_enforce_parts(form, registry)
    if missing:
        add("inputs", "cant_enforce", CANT_ENFORCE, missing=missing, routeTo="administrator")
    if not_checked:
        add("inputs", "checkpoint_not_live", f"Can't be enforced yet: nothing checks policies {not_checked} yet.",
            checkpoint=not_checked, routeTo="administrator")
    inputs = h.get("inputs") or []
    if any(i.get("isAmount") for i in inputs):
        currency = (h.get("units") or {}).get("currency")
        if _blank(currency) or any(i.get("isAmount") and i.get("unit") != currency for i in inputs):
            add("units", "currency_required", "Amounts need a stated currency.")
    tw = h.get("timeWindow")
    if tw and _blank(tw.get("timeZone")):
        add("timeWindow", "time_zone_required", "A time window needs a time zone.")

    rows = conditions.reviewer_view(form, settings)
    if not rows:
        add("examples", "examples_missing", "There are no examples to check.")
    elif any(r["flipped"] for r in rows):
        add("examples", "example_flipped", "An example was marked wrong, so the condition is wrong. Ask the agent to fix it.")
    elif any(r["computed"] == "invalid" for r in rows):
        add("examples", "condition_unreadable", "The condition could not be read. Ask the agent to fix it.")
    if not form.get("checked"):
        add("checked", "not_confirmed", "Confirm the examples and how it is enforced.")

    if outcome not in ("approve", "block", "notify"): add("outcome", "outcome_required", "Choose what happens.")
    if outcome == "approve":
        if not [d for d in form.get("deciders") or [] if str(d).strip()]:
            add("deciders", "deciders_required", "Add at least one person or role who decides.")
        if form.get("responseTime") is not None and (_seconds(form["responseTime"]) or 0) < 1:
            add("responseTime", "response_time_invalid", "A custom response time must be at least 1.")
    if outcome == "notify" and not [n for n in form.get("notify") or [] if str(n).strip()]:
        add("notify", "notify_required", "Add at least one person to tell.")
    if outcome in ("approve", "block") and _blank(form.get("messageForAgent")):
        add("messageForAgent", "message_for_agent_required", "Write the message the agent receives.")

    # Every name a person must answer for has to be linked to someone, or the policy could
    # pause an action that nobody is allowed to release.
    wanted = []
    if outcome == "approve":
        wanted.append(("deciders", [d for d in form.get("deciders") or [] if str(d).strip()]))
    if outcome in ("block", "notify"):
        wanted.append(("notify", [n for n in form.get("notify") or [] if str(n).strip()]))
    if any(names for _, names in wanted):
        mapping = deciders
        if mapping is None:
            try:
                mapping = _load_deciders()
            except Exception:  # noqa: BLE001 - never pass silently when the map can't be read
                mapping = None
                add(wanted[0][0] if wanted else "deciders", "decider_map_unavailable",
                    "Can't check who is linked to the people named here right now. Try again shortly.",
                    routeTo="administrator")
        if mapping is not None:
            for field, names in wanted:
                missing_names = deciders_mod.unmapped(names, mapping)
                if missing_names:
                    add(field, "decider_unmapped", UNMAPPED.format(names=", ".join(missing_names)),
                        names=missing_names, routeTo="administrator")

    limit = form.get("limit") or {}
    if limit.get("on") and _blank(limit.get("text")):
        add("limit", "limit_text_required", "Say how the policy is limited, or turn Limit it off.")

    if _blank(form.get("owner")): add("owner", "owner_required", "Owner is required.")
    return p


def confirmation_cleared(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    if any(old.get(k) != new.get(k) for k in _CONFIRM_KEYS):
        return True
    return [bool(e.get("flipped")) for e in old.get("examples") or []] != \
           [bool(e.get("flipped")) for e in new.get("examples") or []] or \
           [e.get("input") for e in old.get("examples") or []] != [e.get("input") for e in new.get("examples") or []]


# The one word-for-word check for an excerpt: extraction confidence and the agent's
# "not found word for word" note must never disagree. Whole-quote containment only
# (services.obligations.grounding), never extraction_v3's digit fallback.
EXCERPT_MIN_WORDS = 4


def excerpt_grounded(excerpt: Optional[str], document_text: Optional[str]) -> bool:
    from services.obligations.grounding import is_quote_grounded

    return is_quote_grounded(excerpt or "", document_text or "", min_words=EXCERPT_MIN_WORDS)


def extraction_confidence(form: Dict[str, Any], document_text: Optional[str], registry: RegistrySnapshot,
                          settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not form.get("source"):
        return None
    failed: List[str] = []
    if not excerpt_grounded((form.get("source") or {}).get("excerpt"), document_text):
        failed.append("The excerpt does not appear word for word in the document")
    required = ("name", "category", "businessArea", "subArea", "situation", "outcome", "owner")
    if any(_blank(form.get(k)) for k in required) or _blank((form.get("hidden") or {}).get("condition")):
        failed.append("Not every field is filled")
    probs = activation_problems(form, registry, settings, deciders={})   # only registry problems are read; no DB
    if any(p["field"] == "registry" for p in probs):
        failed.append("The condition uses names the orchestrator does not recognise")
    rows = conditions.reviewer_view(form, settings)
    disagree = sum(1 for r in rows if r["agent_expected"] != r["computed"])
    if disagree:
        failed.append(f"The agent's expected result differs from the computed one for {disagree} example"
                      + ("s" if disagree != 1 else ""))
    if _cant_enforce(form, registry):
        failed.append("An input is not available when the policy is checked")
    level = "High" if not failed else "Medium" if len(failed) == 1 else "Low"
    return {"level": level, "failed": failed}
