"""One email family's definition, read from ``proc.bp_policy``.

The spec asks for families in versioned config. This platform's versioned config
is ``bp_policy`` (hot-reloadable, audited, one active row per name), so a family
is a policy row, not a YAML file. Adding a family is a row plus eval cases.

A missing or malformed row RAISES. It never falls back to a default family,
because a validator that quietly uses a different rulebook than the one a person
approved is the failure ``governed_limits`` was written to prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

DEFAULT_CARRIED = ("asks", "line_items", "lead_time_request")
_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")
_ORDER = re.compile(r"^[a-z_][a-z0-9_]*( (ASC|DESC))?(, [a-z_][a-z0-9_]*( (ASC|DESC))?)*$")


class FamilyConfigUnavailable(RuntimeError):
    """The family definition could not be read or is unusable; do not assure with a guess."""


@dataclass(frozen=True)
class FactSource:
    key: str
    table: str
    column: str
    row_id: str
    lookup: Dict[str, str]          # table column -> lookup_key name
    order_by: Optional[str] = None
    value_type: str = "text"        # text | number | date
    caller_keys: List[str] = field(default_factory=list)
    label: str = ""            # human wording; internal table/column names never leave the backend


@dataclass(frozen=True)
class FamilyConfig:
    family_id: str
    version: Any
    facts: Dict[str, FactSource]
    required_facts: List[str]
    context: List[str]
    reasoned: Dict[str, Dict[str, Any]]
    never_state: Dict[str, List[str]]   # internal figure -> payload keys that carry it
    required_elements: List[str]
    carried_keys: List[str]   # payload keys whose figures are accepted but reported unverified
    forbidden_patterns: Dict[str, str]
    length_target: int
    mode: str  # "shadow" records only; "enforce" lets callers refuse on failure
    rubric: List[str] = field(default_factory=list)
    authority_agent: Optional[str] = None
    description: str = ""
    classifiable: bool = True      # False = an assured path, not a kind of request: the classifier never offers it
    request_description: str = ""  # when to choose this family, in the words the classifier is shown
    request_label: str = ""        # a short name for it, used in the question put to the person


def _fail(msg: str) -> "FamilyConfigUnavailable":
    return FamilyConfigUnavailable(msg)


def parse_family(rules: Any, version: Any = None, description: str = "") -> FamilyConfig:
    if not isinstance(rules, dict):
        raise _fail("family rules are not an object")
    family_id = rules.get("family_id")
    if not isinstance(family_id, str) or not family_id:
        raise _fail("family_id missing")
    sources: Dict[str, FactSource] = {}
    for key, raw in (rules.get("fact_sources") or {}).items():
        if not isinstance(raw, dict):
            raise _fail(f"fact_sources.{key} is not an object")
        table, column, row_id = raw.get("table"), raw.get("column"), raw.get("row_id")
        lookup = raw.get("lookup")
        for label, ident in (("table", table), ("column", column), ("row_id", row_id)):
            if not isinstance(ident, str) or not _IDENT.match(ident):
                raise _fail(f"fact_sources.{key}.{label} is not a plain identifier")
        if not isinstance(lookup, dict) or not lookup:
            raise _fail(f"fact_sources.{key}.lookup must name at least one key")
        for col, name in lookup.items():
            if not _IDENT.match(str(col)) or not _IDENT.match(str(name)):
                raise _fail(f"fact_sources.{key}.lookup has a non-identifier")
        order_by = raw.get("order_by")
        if order_by is not None and not _ORDER.match(str(order_by)):
            raise _fail(f"fact_sources.{key}.order_by is not a plain ORDER BY list")
        vt = raw.get("value_type", "text")
        if vt not in ("text", "number", "date"):
            raise _fail(f"fact_sources.{key}.value_type unknown: {vt}")
        sources[key] = FactSource(
            key=key, table=table, column=column, row_id=row_id,
            lookup={str(c): str(n) for c, n in lookup.items()},
            order_by=order_by, value_type=vt,
            caller_keys=[str(k) for k in (raw.get("caller_keys") or [])],
            label=str(raw.get("label") or key.replace("_", " ")),
        )
    required = [str(k) for k in (rules.get("required_facts") or [])]
    unknown = [k for k in required if k not in sources]
    if unknown:
        raise _fail(f"required_facts without a fact_source: {unknown}")
    patterns = rules.get("forbidden_patterns") or {}
    if not isinstance(patterns, dict):
        raise _fail("forbidden_patterns must be an object")
    if "request_description" in rules and not (isinstance(rules["request_description"], str) and rules["request_description"].strip()):
        raise _fail("request_description must be non-empty text")
    if "request_label" in rules and not (isinstance(rules["request_label"], str) and rules["request_label"].strip()):
        raise _fail("request_label must be non-empty text")
    if "classifiable" in rules and not isinstance(rules["classifiable"], bool):
        raise _fail("classifiable must be true or false")
    mode = rules.get("mode", "shadow")
    if mode not in ("shadow", "enforce"):
        raise _fail(f"mode must be shadow or enforce, got {mode!r}")
    return FamilyConfig(
        family_id=family_id, version=version, facts=sources, required_facts=required,
        context=[str(c) for c in (rules.get("context") or [])],
        reasoned={str(k): dict(v or {}) for k, v in (rules.get("reasoned") or {}).items()},
        never_state={str(k): [str(c) for c in (v or [])]
                     for k, v in (rules.get("never_state") or {}).items()},
        required_elements=[str(k) for k in (rules.get("required_elements") or [])],
        carried_keys=[str(k) for k in (rules.get("carried_keys") or DEFAULT_CARRIED)],
        forbidden_patterns={str(k): str(v) for k, v in patterns.items()},
        length_target=int(rules.get("length_target") or 0),
        mode=mode,
        rubric=[str(r) for r in (rules.get("rubric") or [])],
        authority_agent=(str(rules["authority_agent"]) if rules.get("authority_agent") else None),
        classifiable=rules.get("classifiable", True),
        request_description=str(rules.get("request_description") or "").strip(),
        request_label=str(rules.get("request_label") or "").strip(),
    )


def load_family(slug: str, policy_engine: Optional[Any]) -> FamilyConfig:
    """The family named ``slug`` from the policy store, or raise."""

    if policy_engine is None:
        raise _fail("no policy engine available")
    try:
        policy = policy_engine.get_policy(slug)
    except Exception as exc:  # noqa: BLE001 - an outage is not "no rules"
        raise _fail(f"policy store unreadable: {exc}") from exc
    if not isinstance(policy, dict):
        raise _fail(f"no policy named {slug}")
    details = policy.get("details")
    rules = details.get("rules") if isinstance(details, dict) else None
    # PolicyEngine keeps the stored row under raw_row; a bare dict may carry it directly.
    version = policy.get("version")
    if version is None and isinstance(policy.get("raw_row"), dict):
        version = policy["raw_row"].get("version")
    return replace(parse_family(rules, version=version), description=str(policy.get("policy_desc") or ""))


def list_families(policy_engine: Optional[Any]) -> Dict[str, str]:
    """{family_id: description} for every active email family, read from config at run time.

    The classifier is shown exactly this list, so a family added as a row is classifiable with
    no code change, and a family that is not a row cannot be named.
    """

    out: Dict[str, str] = {}
    if policy_engine is None:
        return out
    try:
        policies = policy_engine.list_policies()
    except Exception:  # noqa: BLE001
        return out
    for p in policies or []:
        if p.get("policy_type") != "email_family":
            continue
        rules = (p.get("details") or {}).get("rules") or {}
        fid = rules.get("family_id")
        if rules.get("classifiable") is False:     # an assured path (RFQ batch, human-written), not a kind of request
            continue
        if isinstance(fid, str) and fid:
            out[fid] = str(rules.get("request_description") or p.get("policy_desc") or fid)
    return out


def list_labels(policy_engine: Optional[Any]) -> Dict[str, str]:
    """{family_id: short label} for the classifiable families that declare one. The question put to a person uses these;
    a family without one is named by the first sentence of its description instead."""

    out: Dict[str, str] = {}
    if policy_engine is None:
        return out
    try:
        policies = policy_engine.list_policies()
    except Exception:  # noqa: BLE001
        return out
    for p in policies or []:
        if p.get("policy_type") != "email_family":
            continue
        rules = (p.get("details") or {}).get("rules") or {}
        fid, label = rules.get("family_id"), rules.get("request_label")
        if rules.get("classifiable") is False or not (isinstance(fid, str) and fid):
            continue
        if isinstance(label, str) and label.strip():
            out[fid] = label.strip()
    return out
