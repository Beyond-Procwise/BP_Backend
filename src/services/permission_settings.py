"""Onboarding permission settings: who may do a sensitive action is an administrator's choice.

The hardening work put a permission check in front of dangerous actions. What that check ASKS
(which role is enough) must not be compiled into the code: the product owner decides it at
onboarding and can change it later. Each choice here is one named *setting*; it is stored as a
``proc.bp_policy`` row, which is exactly what ``guardrail.authorize`` already reads, so saving a
setting changes the gate with no second mechanism.

  * A setting names the actions it covers and a starting role (the safe default). Dangerous
    actions start at Admin.
  * Creating a draft (an agent, a prompt, a policy) stays open; it is *using* one that needs the
    approver the administrator names (decision D-34).
  * Saving is one transaction with an optimistic lock and a history line; a stale save changes
    nothing, a role the policy does not define is refused, and every change is audited.
  * Seeding only fills in settings that have no row; it never overwrites an administrator's choice.

Rows take effect when the policy cache reloads (restart or ``policy.reload``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.agent_actions import record_action_or_fail
from services.db import get_conn

POLICY_TYPE = "security"


class UnknownSetting(KeyError):
    """The key is not in the catalogue."""


class InvalidRole(ValueError):
    """The role is not one the role definition knows."""


class SettingConflict(RuntimeError):
    """Somebody saved this setting between the read and the write."""


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    description: str            # plain English, shown to the administrator
    policy_name: str
    policy_identifier: str
    applies_to: Tuple[str, ...]
    default_role: str


CATALOGUE: Tuple[Setting, ...] = (
    Setting("draft_authoring", "Who may create drafts",
            "Write a draft agent, prompt or policy. A draft does nothing until it is approved for use.",
            "DraftAuthoringAuthorityPolicy", "draft_authoring_authority",
            ("prompt.author", "agent_policy.write"), "Buyer"),
    Setting("agent_use", "Who approves the use of agents and workflows",
            "Start an agent or run a workflow. Anyone can create one; this is who lets it run.",
            "AgentRunAuthorityPolicy", "agent_run_authority",
            ("agent.run", "workflow.run"), "Admin"),
    Setting("prompt_use", "Who approves the use of prompts",
            "Promote a draft prompt so every agent uses it, or roll one back.",
            "PromptPromotionAuthorityPolicy", "prompt_promotion_authority",
            ("prompt.promote",), "Admin"),
    Setting("agent_policy_use", "Who approves the use of agent policies",
            "Activate an agent policy so it governs real runs.",
            "AgentPolicyActivationAuthorityPolicy", "agent_policy_activation_authority",
            ("agent_policy.activate",), "Admin"),
    Setting("supplier_master", "Who may change the supplier master",
            "Merge or split suppliers, scan for duplicates, apply or reject an enrichment proposal.",
            "SupplierMasterAuthorityPolicy", "supplier_master_authority",
            ("supplier.write",), "Admin"),
    Setting("screening_record", "Who may record compliance screening results",
            "Record what a sanctions, financial-health, ESG, forced-labour, conflict-minerals or risk-event check "
            "found on a supplier, with the provider and date. Agents only read these results.",
            "ScreeningRecordAuthorityPolicy", "screening_record_authority",
            ("supplier.screening.record",), "Admin"),
    Setting("index_series_record", "Who may load commodity index series",
            "Load a published index or price series (code, publisher, dated values) into the index register that "
            "index-linked pricing, price-increase splits and savings neutralisation read. Agents only read it.",
            "IndexSeriesRecordAuthorityPolicy", "index_series_record_authority",
            ("index_series.record",), "Admin"),
    Setting("po_renumber", "Who may renumber purchase orders in bulk",
            "Rewrite the purchase-order number across all final records.",
            "PoRenumberAuthorityPolicy", "po_renumber_authority",
            ("document.renumber",), "Admin"),
    Setting("opportunity_mining", "Who may write mined opportunities back",
            "Load mined findings into the opportunity list and re-link them to deals in bulk.",
            "OpportunityMiningAuthorityPolicy", "opportunity_mining_authority",
            ("opportunity.mine",), "Admin"),
)

_BY_KEY = {s.key: s for s in CATALOGUE}


@dataclass(frozen=True)
class ChoiceSetting:
    """A setting whose value is one of a fixed list of named choices, not a role.

    Stored in the same ``proc.bp_policy`` table, with the chosen value under ``rules.mode``. It has no
    ``applies_to``: it is read by name by the code that needs it, not weighed by ``guardrail.authorize``
    (the same treatment the governed limits get).
    """
    key: str
    label: str
    description: str
    policy_name: str
    policy_identifier: str
    choices: Tuple[str, ...]
    default: str


CHOICES: Tuple[ChoiceSetting, ...] = (
    ChoiceSetting(
        "workflow_question_answer", "Who may answer a paused workflow's question",
        "When a workflow stops and asks a person something, who may answer. approver_only: only people "
        "with the approver permission, and never the person who started the run when the question is an "
        "approval. starter_or_approver: the person who started the run, or an approver. any_signed_in: "
        "anyone signed in.",
        "WorkflowQuestionAnswerPolicy", "workflow_question_answer",
        ("approver_only", "starter_or_approver", "any_signed_in"), "approver_only"),
)

_CHOICE_BY_KEY = {c.key: c for c in CHOICES}


class InvalidChoice(ValueError):
    """The value is not one of the setting's choices."""


def setting(key: str) -> Setting:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise UnknownSetting(f"{key!r} is not a permission setting") from None


def choice_setting(key: str) -> ChoiceSetting:
    try:
        return _CHOICE_BY_KEY[key]
    except KeyError:
        raise UnknownSetting(f"{key!r} is not a choice setting") from None


def _known_roles(roles: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if roles is not None:
        return roles
    from services import rbac
    table = rbac._roles_table(rbac.policy_engine())  # noqa: SLF001 - the one source of role names
    return dict(table)


def _details(s: Setting, role: str, history: Optional[list] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "policy_identifier": s.policy_identifier,
        "required_role": role,
        "applies_to": list(s.applies_to),
        "rules": {"effect": "allow"},
    }
    if history:
        d["history"] = history
    return d


def _record(conn, **kw) -> None:
    record_action_or_fail(conn=conn, **kw)


def _load(cur, names: List[str], for_update: bool = False) -> Dict[str, Tuple[int, int, Dict[str, Any]]]:
    cur.execute(
        "SELECT policy_id, policy_name, version, policy_details FROM proc.bp_policy "
        "WHERE policy_name = ANY(%s) AND policy_status = 1" + (" FOR UPDATE" if for_update else ""),
        (names,))
    out = {}
    for pid, name, version, details in cur.fetchall():
        if isinstance(details, str):
            details = json.loads(details)
        out[name] = (pid, int(version or 0), details if isinstance(details, dict) else {})
    return out


def _insert(cur, s: Setting, role: str, by: str, history: Optional[list] = None) -> None:
    cur.execute(
        "INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details, "
        "policy_linked_agents, policy_status, version, created_by, created_date) "
        "VALUES (%s, %s, %s, %s::jsonb, '', 1, 1, %s, now())",
        (s.policy_name, POLICY_TYPE, s.description, json.dumps(_details(s, role, history)), by))


def _choice_details(c: ChoiceSetting, value: str, history: Optional[list] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"policy_identifier": c.policy_identifier, "rules": {"mode": value}}
    if history:
        d["history"] = history
    return d


def _insert_choice(cur, c: ChoiceSetting, value: str, by: str, history: Optional[list] = None) -> None:
    cur.execute(
        "INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details, "
        "policy_linked_agents, policy_status, version, created_by, created_date) "
        "VALUES (%s, %s, %s, %s::jsonb, '', 1, 1, %s, now())",
        (c.policy_name, POLICY_TYPE, c.description, json.dumps(_choice_details(c, value, history)), by))


def _in_transaction(conn, work):
    # get_conn() connections are autocommit: the lock, write, history and audit must stand together.
    previous = getattr(conn, "autocommit", True)
    conn.autocommit = False
    try:
        try:
            result = work(conn.cursor())
            conn.commit()
            return result
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.autocommit = previous


def current(conn=None) -> List[Dict[str, Any]]:
    """Every setting with the role now in force; ``configured`` is False if it has no row yet."""
    if conn is None:
        with get_conn() as own:
            return current(own)
    rows = _load(conn.cursor(), [s.policy_name for s in CATALOGUE])
    out = []
    for s in CATALOGUE:
        got = rows.get(s.policy_name)
        details = got[2] if got else {}
        out.append({
            "key": s.key, "label": s.label, "description": s.description,
            "actions": list(s.applies_to),
            "role": details.get("required_role", s.default_role) if got else s.default_role,
            "default_role": s.default_role,
            "configured": got is not None,
            "version": got[1] if got else 0,
            "kind": "role",
        })
    chosen = _load(conn.cursor(), [c.policy_name for c in CHOICES])
    for c in CHOICES:
        got = chosen.get(c.policy_name)
        value = ((got[2].get("rules") or {}).get("mode") if got else None) or c.default
        out.append({
            "key": c.key, "label": c.label, "description": c.description,
            "actions": [], "kind": "choice", "choices": list(c.choices),
            "value": value, "default_value": c.default,
            "configured": got is not None, "version": got[1] if got else 0,
        })
    return out


def choice_value(key: str, conn=None) -> str:
    """The value now chosen for choice setting ``key``; its default if no row exists yet.

    RAISES if the row cannot be read: the caller decides what an outage means (see
    ``services.workflow_answer_policy``, which fails closed). A stored value that is not one of the
    choices is returned as stored, so the caller can refuse it rather than quietly widen it.
    """
    c = choice_setting(key)
    if conn is None:
        with get_conn() as own:
            return choice_value(key, own)
    got = _load(conn.cursor(), [c.policy_name]).get(c.policy_name)
    if not got:
        return c.default
    return str((got[2].get("rules") or {}).get("mode") or c.default)


def minimum_role(key: str, conn=None, fallback: str = "Admin") -> str:
    """The role now required by setting ``key``; ``fallback`` if the row cannot be read.

    Callers that gate on it use the strictest role as the fallback: an outage must not look
    like "anyone may".
    """
    try:
        for row in current(conn):
            if row["key"] == key:
                return str(row["role"])
    except Exception:  # noqa: BLE001
        pass
    return fallback


def seed_defaults(by: str, conn=None, roles: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Create a row for every setting that has none. Never touches one that exists."""
    if conn is None:
        with get_conn() as own:
            return seed_defaults(by, own, roles)
    known = _known_roles(roles)
    for s in CATALOGUE:
        if s.default_role not in known:
            raise InvalidRole(f"default role {s.default_role!r} for {s.key} is not a defined role")

    def work(cur):
        have = _load(cur, [s.policy_name for s in CATALOGUE], for_update=True)
        created = []
        for s in CATALOGUE:
            if s.policy_name not in have:
                _insert(cur, s, s.default_role, by)
                created.append(s.key)
        have_c = _load(cur, [c.policy_name for c in CHOICES], for_update=True)
        for c in CHOICES:
            if c.policy_name not in have_c:
                _insert_choice(cur, c, c.default, by)
                created.append(c.key)
        if created:
            _record(conn, phase="configure", action_type="permission_settings_seeded",
                    summary=f"onboarding defaults written for {', '.join(created)}",
                    details={"changed_by": by, "settings": created})
        return {"created": created}
    return _in_transaction(conn, work)


def set_role(key: str, role: str, by: str, conn=None,
             roles: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make ``role`` the minimum for setting ``key``. A no-op save writes and audits nothing."""
    s = setting(key)
    if conn is None:
        with get_conn() as own:
            return set_role(key, role, by, own, roles)
    if role not in _known_roles(roles):
        raise InvalidRole(f"{role!r} is not a defined role")

    def work(cur):
        have = _load(cur, [s.policy_name], for_update=True)
        if s.policy_name not in have:
            _insert(cur, s, role, by, [_history_line(1, by, None, role)])
            _record(conn, phase="configure", action_type="permission_setting_changed",
                    summary=f"{s.key}: set to {role} by {by}",
                    details={"changed_by": by, "setting": s.key, "before": None, "after": role, "version": 1})
            return {"key": s.key, "role": role, "changed": True, "version": 1}
        pid, version, details = have[s.policy_name]
        before = details.get("required_role")
        if before == role:
            return {"key": s.key, "role": role, "changed": False, "version": version}
        history = list(details.get("history") or []) + [_history_line(version + 1, by, before, role)]
        merged = {**details, "required_role": role, "history": history}
        cur.execute(
            "UPDATE proc.bp_policy SET policy_details = %s::jsonb, version = version + 1, "
            "last_modified_by = %s, last_modified_date = now() WHERE policy_id = %s AND version = %s",
            (json.dumps(merged), by, pid, version))
        if cur.rowcount != 1:
            raise SettingConflict(f"{s.label} was changed by somebody else; reload and try again")
        _record(conn, phase="configure", action_type="permission_setting_changed",
                summary=f"{s.key}: {before} -> {role} by {by}",
                details={"changed_by": by, "setting": s.key, "before": before, "after": role,
                         "version": version + 1})
        return {"key": s.key, "role": role, "changed": True, "version": version + 1}
    return _in_transaction(conn, work)


def _history_line(version: int, by: str, before: Optional[str], after: str) -> Dict[str, Any]:
    return {"version": version, "changed_by": by, "changed_at": datetime.now(timezone.utc).isoformat(),
            "field": "required_role", "before": before, "after": after}


def set_choice(key: str, value: str, by: str, conn=None) -> Dict[str, Any]:
    """Make ``value`` the choice for ``key``. A value outside the list is refused; a no-op save writes nothing."""
    c = choice_setting(key)
    if value not in c.choices:
        raise InvalidChoice(f"{value!r} is not one of: {', '.join(c.choices)}")
    if conn is None:
        with get_conn() as own:
            return set_choice(key, value, by, own)

    def work(cur):
        have = _load(cur, [c.policy_name], for_update=True)
        if c.policy_name not in have:
            _insert_choice(cur, c, value, by, [_choice_history(1, by, None, value)])
            _record(conn, phase="configure", action_type="permission_setting_changed",
                    summary=f"{c.key}: set to {value} by {by}",
                    details={"changed_by": by, "setting": c.key, "before": None, "after": value, "version": 1})
            return {"key": c.key, "value": value, "changed": True, "version": 1}
        pid, version, details = have[c.policy_name]
        before = (details.get("rules") or {}).get("mode")
        if before == value:
            return {"key": c.key, "value": value, "changed": False, "version": version}
        history = list(details.get("history") or []) + [_choice_history(version + 1, by, before, value)]
        merged = {**details, "rules": {**(details.get("rules") or {}), "mode": value}, "history": history}
        cur.execute(
            "UPDATE proc.bp_policy SET policy_details = %s::jsonb, version = version + 1, "
            "last_modified_by = %s, last_modified_date = now() WHERE policy_id = %s AND version = %s",
            (json.dumps(merged), by, pid, version))
        if cur.rowcount != 1:
            raise SettingConflict(f"{c.label} was changed by somebody else; reload and try again")
        _record(conn, phase="configure", action_type="permission_setting_changed",
                summary=f"{c.key}: {before} -> {value} by {by}",
                details={"changed_by": by, "setting": c.key, "before": before, "after": value,
                         "version": version + 1})
        return {"key": c.key, "value": value, "changed": True, "version": version + 1}
    return _in_transaction(conn, work)


def _choice_history(version: int, by: str, before: Optional[str], after: str) -> Dict[str, Any]:
    return {"version": version, "changed_by": by, "changed_at": datetime.now(timezone.utc).isoformat(),
            "field": "mode", "before": before, "after": after}
