"""The six tone variables, each with the source it came from. No model guesses anything here.

For every variable the answer is one of three, recorded with it: derived from a Postgres row,
set by the person's own words, or the variable's declared default because there was no data.
A person's instruction overrides everything, including what Postgres said.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VARIABLES = ("relationship_tier", "escalation_level", "leverage", "recipient_seniority",
             "relationship_health", "region_formality", "warmth", "directness")
GAP_RULES = ("default", "assume")      # `ask` is refused: nothing yet lets a reviewer answer a question
SUPPLIER_COLUMNS = {"is_preferred_supplier", "contact_role_1", "country"}   # derivations may read only these
KINDS = {"supplier_flag", "supplier_keyword", "supplier_map", "prior_contacts"}


class ToneRulesUnavailable(RuntimeError):
    """The tone rules are missing or malformed; nothing is derived from a guess in their place."""


@dataclass(frozen=True)
class ToneRules:
    variables: Dict[str, Dict[str, Any]]
    overrides: List[Dict[str, Any]]
    version: Any = None
    cues: Optional[str] = None       # regex of words that mark an instruction as a tone request
    directives: Optional[Dict[str, Dict[str, str]]] = None   # variable -> value -> the sentence it steers the drafter with


def parse_rules(rules: Any, version: Any = None) -> ToneRules:
    if not isinstance(rules, dict) or not isinstance(rules.get("variables"), dict):
        raise ToneRulesUnavailable("tone rules carry no variables")
    variables = rules["variables"]
    missing = [v for v in VARIABLES if v not in variables]
    if missing:
        raise ToneRulesUnavailable(f"tone rules lack: {missing}")
    for name, spec in variables.items():
        allowed = spec.get("allowed")
        if not isinstance(allowed, list) or not allowed:
            raise ToneRulesUnavailable(f"{name}: allowed values missing")
        if "unknown_default" not in spec or spec["unknown_default"] not in allowed:
            raise ToneRulesUnavailable(f"{name}: unknown_default missing or not an allowed value")
        if spec.get("on_gap") not in GAP_RULES:
            raise ToneRulesUnavailable(f"{name}: on_gap must be one of {list(GAP_RULES)}, got {spec.get('on_gap')!r}")
        derive = spec.get("derive")
        if derive is not None:
            if derive.get("kind") not in KINDS:
                raise ToneRulesUnavailable(f"{name}: unknown derivation {derive.get('kind')!r}")
            if derive["kind"].startswith("supplier") and derive.get("column") not in SUPPLIER_COLUMNS:
                raise ToneRulesUnavailable(f"{name}: column {derive.get('column')!r} is not readable here")
            outs = (list((derive.get("map") or {}).values())
                    + [r.get("value") for r in derive.get("rules") or []]
                    + [lv.get("value") for lv in derive.get("levels") or []])
            bad = [o for o in outs if o not in allowed]
            if bad:
                raise ToneRulesUnavailable(f"{name}: derivation yields values outside allowed: {bad}")
    overrides = rules.get("instruction_overrides") or []
    for o in overrides:
        try:
            re.compile(o["match"])
        except (KeyError, re.error, TypeError) as exc:
            raise ToneRulesUnavailable(f"bad instruction override: {exc}") from exc
        for var, val in (o.get("set") or {}).items():
            if var not in variables or val not in variables[var]["allowed"]:
                raise ToneRulesUnavailable(f"override sets {var}={val!r}, which is not allowed")
    cues = rules.get("tone_cues")
    if cues is not None:
        try:
            re.compile(cues)
        except (re.error, TypeError) as exc:
            raise ToneRulesUnavailable(f"bad tone_cues: {exc}") from exc
    directives = rules.get("directives")
    if directives is not None:
        if not isinstance(directives, dict):
            raise ToneRulesUnavailable("directives must be an object")
        for var, by_value in directives.items():
            if var not in variables or not isinstance(by_value, dict):
                raise ToneRulesUnavailable(f"directives name {var!r}, which is not a tone variable")
            allowed = {str(a) for a in variables[var]["allowed"]}
            for value, text in by_value.items():
                if str(value) not in allowed:
                    raise ToneRulesUnavailable(f"directive for {var}={value!r}, which is not an allowed value")
                if not isinstance(text, str) or not text.strip():
                    raise ToneRulesUnavailable(f"directive for {var}={value!r} is empty")
    return ToneRules(variables=variables, overrides=list(overrides), version=version, cues=cues, directives=directives)


def load_rules(policy_engine: Optional[Any], slug: str = "email_tone_rules") -> ToneRules:
    if policy_engine is None:
        raise ToneRulesUnavailable("no policy engine available")
    try:
        policy = policy_engine.get_policy(slug)
    except Exception as exc:  # noqa: BLE001
        raise ToneRulesUnavailable(f"policy store unreadable: {exc}") from exc
    if not isinstance(policy, dict):
        raise ToneRulesUnavailable("no tone rules are defined")
    details = policy.get("details") or {}
    version = policy.get("version")
    if version is None and isinstance(policy.get("raw_row"), dict):
        version = policy["raw_row"].get("version")
    return parse_rules(details.get("rules"), version)


def _query(conn: Any, sql: str, params: List[Any]) -> List[Any]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def _supplier_value(conn: Any, column: str, supplier_id: Optional[str]) -> Any:
    if not supplier_id:
        return None
    rows = _query(conn, f"SELECT {column} FROM proc.bp_supplier WHERE supplier_id = %s", [supplier_id])
    return rows[0][0] if rows else None


def _prior_contacts(conn: Any, workflow_id: Optional[str], supplier_id: Optional[str]) -> Optional[int]:
    """How many times we have already written to this supplier in this workflow, or None if unknowable."""
    if not workflow_id or not supplier_id:
        return None
    rows = _query(conn, "SELECT count(*) FROM proc.workflow_email_tracking "
                        "WHERE workflow_id = %s AND supplier_id = %s", [workflow_id, supplier_id])
    return int(rows[0][0]) if rows else None


def _derive(conn: Any, spec: Dict[str, Any], supplier_id, workflow_id) -> Any:
    d = spec.get("derive")
    if not d:
        return None, None
    kind = d["kind"]
    if kind == "prior_contacts":
        n = _prior_contacts(conn, workflow_id, supplier_id)
        if n is None:
            return None, None
        value = None
        for lv in sorted(d["levels"], key=lambda x: x["min"]):
            if n >= lv["min"]:
                value = lv["value"]
        return value, f"{n} earlier contact(s) on this thread"
    raw = _supplier_value(conn, d["column"], supplier_id)
    if raw is None or str(raw).strip() == "":
        return None, None
    if kind == "supplier_flag":
        return (d["map"].get(str(bool(raw)).lower()) if isinstance(raw, bool) else d["map"].get(str(raw).lower())), f"supplier {d['column']}"
    if kind == "supplier_map":
        return d["map"].get(str(raw).strip()), f"supplier {d['column']}"
    text = str(raw).lower()
    for r in d["rules"]:
        if any(k in text for k in r["contains"]):
            return r["value"], f"supplier {d['column']}"
    return None, None


def derive_tone(conn: Any, rules: ToneRules, *, supplier_id: Optional[str], workflow_id: Optional[str],
                instruction: Optional[str] = None) -> Dict[str, Any]:
    """{"values": {...}, "sources": {var: {"source", "detail"}}, "status": "captured"}."""

    values: Dict[str, Any] = {}
    sources: Dict[str, Dict[str, Any]] = {}
    for name in VARIABLES:
        spec = rules.variables[name]
        try:
            value, detail = _derive(conn, spec, supplier_id, workflow_id)
        except Exception:  # noqa: BLE001 - a failed lookup is "no data", and the default says so
            logger.exception("tone derivation failed for %s", name)
            value, detail = None, None
        if value is not None and value in spec["allowed"]:
            values[name], sources[name] = value, {"source": "postgres", "detail": detail}
        else:
            values[name] = spec["unknown_default"]
            sources[name] = {"source": "default", "detail": "no stored data to derive it from"}
    text = (instruction or "").strip()
    covered: List[tuple] = []
    if text:
        for o in rules.overrides:
            hits = list(re.finditer(o["match"], text, re.IGNORECASE))
            if hits:
                covered += [h.span() for h in hits]
                for var, val in o["set"].items():
                    values[var] = val
                    sources[var] = {"source": "user_instruction", "detail": f"matched /{o['match']}/"}
    # A GAP is a variable that took its declared default. Whether that needs a person to confirm
    # it is the variable's own `on_gap` rule, not a guess made here.
    gaps = [{"variable": n, "default": values[n], "on_gap": rules.variables[n]["on_gap"]}
            for n in VARIABLES if sources[n]["source"] == "default"]
    # A tone word no override covers is a request we did not understand. It is reported, never dropped.
    unmapped: List[str] = []
    if text and rules.cues:
        for m in re.finditer(rules.cues, text, re.IGNORECASE):
            if not any(a <= m.start() and m.end() <= b for a, b in covered):
                unmapped.append(m.group(0).lower())
    return {"status": "captured", "version": rules.version, "values": values, "sources": sources,
            "gaps": gaps, "unmapped_instruction": sorted(set(unmapped))}
