"""The brief: what a draft is for, and what each judgement in it rests on.

Two producers, one schema. A planner model writes one for free text (``from_prompt``); for the
negotiation paths the negotiation agent has already done the planning, so ``counter_brief``
MAPS its output into the same shape instead of asking a second model to re-plan (a second
planner could contradict the first one's numbers).

``normalise_brief`` is the gate both go through. The rule it enforces is the spec's: a reasoned
value whose basis names nothing real is not accepted, it becomes an assumption a person must
confirm. A basis is never invented here -- where the source gave none, the answer is "assumption"
or "missing".
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

TEXT_KEYS = ("goal", "explicit_ask", "deadline", "tone_rationale")
LIST_KEYS = ("key_points", "risks_to_avoid")


class BriefInvalid(ValueError):
    """The planner's output is not a brief. The caller records it as invalid; it is never repaired by guessing."""


def _assumption(n: int, key: Optional[str], text: str) -> Dict[str, Any]:
    return {"id": key or f"a{n}", "key": key, "text": text, "resolution": None}


def normalise_brief(raw: Any, *, fact_keys: Iterable[str], context_keys: Iterable[str]) -> Dict[str, Any]:
    """Validate a planner brief. Returns ``{"status": "ready"|"missing", ...}`` or raises ``BriefInvalid``."""

    if not isinstance(raw, dict):
        raise BriefInvalid("the brief is not a JSON object")
    if set(raw) == {"missing"}:
        missing = raw["missing"]
        if not isinstance(missing, list) or not all(isinstance(m, str) for m in missing):
            raise BriefInvalid("'missing' must be a list of fact keys")
        return {"status": "missing", "missing": missing}
    for key in TEXT_KEYS:
        if not isinstance(raw.get(key), (str, type(None))):
            raise BriefInvalid(f"{key} must be text")
    for key in LIST_KEYS:
        if not isinstance(raw.get(key), list) or not all(isinstance(x, str) for x in raw[key]):
            raise BriefInvalid(f"{key} must be a list of text")
    reasoned_in = raw.get("reasoned")
    if not isinstance(reasoned_in, dict):
        raise BriefInvalid("reasoned must be an object")
    if not isinstance(raw.get("assumptions", []), list):
        raise BriefInvalid("assumptions must be a list")

    known = set(fact_keys) | set(context_keys)
    reasoned: Dict[str, Dict[str, Any]] = {}
    assumptions: List[Dict[str, Any]] = []
    for key, item in reasoned_in.items():
        if not isinstance(item, dict) or "value" not in item:
            raise BriefInvalid(f"reasoned.{key} must carry a value")
        basis_in = item.get("basis", [])
        if not isinstance(basis_in, list):
            raise BriefInvalid(f"reasoned.{key}.basis must be a list")
        basis = [b for b in basis_in if isinstance(b, str) and b in known]
        conf = item.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) and 0 <= conf <= 1 else None
        reasoned[key] = {"value": item["value"], "basis": basis, "confidence": conf}
        if not basis:
            dropped = [b for b in basis_in if b not in known]
            why = f" (named {dropped}, which are not facts or context)" if dropped else ""
            assumptions.append(_assumption(len(assumptions) + 1, key,
                                           f"{key} = {item['value']} has no verified basis{why}"))
    for extra in raw.get("assumptions", []):
        text = extra.get("text") if isinstance(extra, dict) else extra
        if isinstance(text, str) and text.strip() and not any(text == a["text"] for a in assumptions):
            assumptions.append(_assumption(len(assumptions) + 1, None, text.strip()))
    brief = {k: raw.get(k) for k in TEXT_KEYS}
    brief.update({k: list(raw[k]) for k in LIST_KEYS})
    brief.update({"status": "ready", "reasoned": reasoned, "assumptions": assumptions, "missing": []})
    return brief


def _points(value: Any) -> List[str]:
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            text = next((item[k] for k in ("play", "name", "title", "description") if isinstance(item.get(k), str)), None)
            if text:
                out.append(text.strip())
    return out


def counter_brief(data: Dict[str, Any], inputs: Any, tone: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map what the negotiation agent decided into the standard brief. Nothing is invented."""

    missing: List[str] = []
    goal = next((str(data[k]).strip() for k in ("rationale", "strategy") if str(data.get(k) or "").strip()), None)
    if goal is None:
        missing.append("goal")
    asks = [str(a).strip() for a in (data.get("asks") or []) if str(a).strip()] if isinstance(data.get("asks"), list) else []
    key_points = asks + [p for p in _points(data.get("play_recommendations")) if p not in asks]
    if not asks:
        missing.append("explicit_ask")
    reasoned = {k: {"value": r["value"], "basis": list(r["basis"]), "confidence": None}
                for k, r in inputs.reasoned.items()}
    deadline = reasoned.get("response_deadline", {}).get("value")
    if deadline is None:
        missing.append("deadline")
    assumptions = [_assumption(i + 1, k, f"{k} = {r['value']} has no verified basis")
                   for i, (k, r) in enumerate(reasoned.items()) if not r["basis"]]
    rationale = None
    if tone and tone.get("status") == "captured":
        rationale = "; ".join(f"{k} {tone['values'][k]} ({tone['sources'][k]['source']})" for k in tone["values"])
    else:
        missing.append("tone_rationale")
    fam = inputs.family
    return {"status": "ready", "goal": goal, "key_points": key_points,
            "explicit_ask": asks[0] if asks else None,
            "deadline": str(deadline) if deadline is not None else None,
            "tone_rationale": rationale,
            "risks_to_avoid": sorted(fam.never_state) + sorted(fam.forbidden_patterns),
            "reasoned": reasoned, "assumptions": assumptions, "missing": missing}
