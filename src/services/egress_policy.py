"""What may leave the system is an administrator's policy row, not code (1.20 b, D-6 / D-62).

``EgressPolicy`` (``proc.bp_policy``, identifier ``egress``) holds two rule lists:

    rules = {"allow": [ {purpose, hosts: [pattern...], agents: [name|"*"...]} ... ],
             "deny":  [ {purpose, hosts, agents, note?} ... ]}

A call matches a rule when its purpose equals the rule's, its destination host matches one of the
host patterns (exact, case-insensitive; ``*.suffix``; or ``*`` for any host) and its agent is named
or the rule says ``*``. A matching DENY wins over any ALLOW. Nothing matching means deny. LOCAL
destinations (loopback, private addresses) are always allowed and never consult the row: the model
host and the in-VPC services must keep working whatever the row says.

SHADOW. This module only says what the gate WOULD do. ``services.egress._evaluate`` records the
answer on every event row and still permits the call; the per-purpose enforce switch is step (d).

FAIL CLOSED. If the row cannot be read, every external destination would be denied, and the
reason says so, so a week of shadow cannot show "all allowed" because the database was down.
NO ROW. The built-in default rule set applies (local allow, external deny) and the event names it,
so an unseeded database is visible as such.

The row is read at most once per ``_TTL`` seconds; a save through this module invalidates the cache
in this process (other processes see it within the TTL).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.services.agent_actions import record_action_or_fail
from src.services.db import get_conn
from src.services import egress
from src.services.egress import Purpose

logger = logging.getLogger(__name__)

POLICY_NAME = "EgressPolicy"
IDENTIFIER = "egress"
POLICY_TYPE = "security"
DESCRIPTION = ("Which outside destinations each purpose and agent may call. Local destinations are always "
               "allowed. A deny rule beats an allow rule; anything unmatched is denied. In shadow mode the "
               "decision is recorded on every egress event and nothing is blocked.")

DEFAULT_VERSION = egress.SHADOW_RULES_VERSION   # no row: the built-in rule set (one literal, in egress)
UNREADABLE_VERSION = "egress-policy-unreadable"
_TTL = 30.0            # a good read is reused this long
_FAIL_TTL = 5.0        # a failed read is remembered this long: an outage costs one connect attempt per 5 s, not one per call

#: Starting rules: the external callers seen in the last week's egress events, plus the D-62 denial.
SEED_RULES: Dict[str, List[Dict[str, Any]]] = {
    "allow": [
        {"purpose": "fx_rates", "hosts": ["open.er-api.com"], "agents": ["*"],
         "note": "currency reference rates; no company text in the request"},
        {"purpose": "mailbox", "hosts": ["graph.microsoft.com", "login.microsoftonline.com"], "agents": ["*"],
         "note": "the company's own Microsoft 365 mailbox"},
        {"purpose": "supplier_research", "hosts": ["*"], "agents": ["*"],
         "note": "public web for supplier research; the address guard still refuses private hosts"},
    ],
    "deny": [
        {"purpose": "model_inference", "hosts": ["api.ollama.com"], "agents": ["*"],
         "note": "D-62: company text never goes to Ollama cloud"},
    ],
}

_PURPOSES = frozenset(p.value for p in Purpose)


class InvalidRules(ValueError):
    """The rules are not in the shape above, or name an unknown purpose."""


class RulesConflict(RuntimeError):
    """Somebody saved the rules between the read and the write."""


@dataclass(frozen=True)
class Verdict:
    effect: str          # "allow" | "deny"
    reason: str
    version: str         # which rule set judged it


def _host_matches(pattern: str, host: str) -> bool:
    pattern, host = pattern.strip().lower(), host.strip().lower()
    if pattern == "*":
        return True
    if pattern.startswith("*."):
        suffix = pattern[1:]                       # ".amazonaws.com"
        return host.endswith(suffix) and len(host) > len(suffix)
    return pattern == host


def _rule_ok(rule: Any) -> bool:
    return (isinstance(rule, dict)
            and rule.get("purpose") in _PURPOSES
            and isinstance(rule.get("hosts"), list) and rule["hosts"]
            and all(isinstance(h, str) and h.strip() for h in rule["hosts"])
            and isinstance(rule.get("agents"), list) and rule["agents"]
            and all(isinstance(a, str) and a.strip() for a in rule["agents"]))


def _matches(rule: Dict[str, Any], purpose: Purpose, host: str, agent: str) -> bool:
    if rule.get("purpose") != purpose.value:
        return False
    if not any(_host_matches(p, host) for p in rule["hosts"]):
        return False
    agents = [a.strip() for a in rule["agents"]]
    return "*" in agents or agent.strip() in agents


@dataclass(frozen=True)
class Policy:
    version: int
    rules: Dict[str, Any]

    @property
    def label(self) -> str:
        return f"egress-policy-v{self.version}"

    def _rules(self, kind: str) -> List[Dict[str, Any]]:
        got = self.rules.get(kind) if isinstance(self.rules, dict) else None
        # A malformed rule is skipped, never read as "allow everything".
        return [r for r in (got if isinstance(got, list) else []) if _rule_ok(r)]

    def decide(self, purpose: Purpose, host: str, agent: str) -> Verdict:
        for rule in self._rules("deny"):
            if _matches(rule, purpose, host, agent):
                note = rule.get("note")
                return Verdict("deny", f"deny rule for {purpose.value} to {host}"
                                       + (f": {note}" if note else ""), self.label)
        for rule in self._rules("allow"):
            if _matches(rule, purpose, host, agent):
                return Verdict("allow", f"allow rule for {purpose.value} to {host} (agent {agent})", self.label)
        return Verdict("deny", f"no rule allows {purpose.value} to {host} for agent {agent}", self.label)


def _clean_host(pattern: str, where: str) -> str:
    p = pattern.strip().lower()
    if p == "*":
        return p
    if any(ch in p for ch in "/:@ \t") or p in ("*.", ".") or p.startswith("*.") and len(p) < 4:
        raise InvalidRules(f"{where}: {pattern!r} is not a host pattern (a bare host name, '*.suffix' or '*')")
    if "*" in p[1:] or (p.startswith("*") and not p.startswith("*.")):
        raise InvalidRules(f"{where}: {pattern!r} may only use '*' alone or as '*.suffix'")
    return p


def validate(rules: Any) -> Dict[str, List[Dict[str, Any]]]:
    """The rules as they will be stored, or ``InvalidRules``. Only ``allow`` and ``deny`` are known.

    Hosts are lower-cased and stripped; agents stripped. A pattern with a scheme, path, port or
    space can never match a host name and is refused rather than stored dead."""
    if not isinstance(rules, dict):
        raise InvalidRules("rules must be an object with 'allow' and 'deny' lists")
    extra = set(rules) - {"allow", "deny"}
    if extra:
        raise InvalidRules(f"unknown key(s): {', '.join(sorted(extra))}")
    out: Dict[str, List[Dict[str, Any]]] = {}
    for kind in ("allow", "deny"):
        got = rules.get(kind, [])
        if not isinstance(got, list):
            raise InvalidRules(f"'{kind}' must be a list of rules")
        for i, rule in enumerate(got):
            if not _rule_ok(rule):
                raise InvalidRules(
                    f"{kind}[{i}] must be {{purpose: one of {sorted(_PURPOSES)}, hosts: [non-empty list], "
                    f"agents: [non-empty list]}}")
        out[kind] = [{"purpose": r["purpose"],
                      "hosts": [_clean_host(h, f"{kind}[{i}].hosts") for h in r["hosts"]],
                      "agents": [a.strip() for a in r["agents"]],
                      **({"note": r["note"]} if r.get("note") else {})} for i, r in enumerate(got)]
    return out


# ---- the row ---------------------------------------------------------------------------------------

def _record(conn, **kw) -> None:
    record_action_or_fail(conn=conn, **kw)


def _load(cur, for_update: bool = False):
    cur.execute(
        "SELECT policy_id, policy_name, version, policy_details FROM proc.bp_policy "
        "WHERE policy_name = ANY(%s) AND policy_status = 1" + (" FOR UPDATE" if for_update else ""),
        ([POLICY_NAME],))
    for pid, _name, version, details in cur.fetchall():
        if isinstance(details, str):
            details = json.loads(details)
        return pid, int(version or 0), details if isinstance(details, dict) else {}
    return None


def _load_row() -> Optional[Policy]:
    """The row as a Policy, or None when there is no row. Raises when it cannot be read."""
    with get_conn() as conn:
        got = _load(conn.cursor())
    if not got:
        return None
    _pid, version, details = got
    return Policy(version=version, rules=details.get("rules") or {})


_cache_lock = threading.Lock()
_cached: Optional[Policy] = None
_cached_error: Optional[BaseException] = None
_cached_at = 0.0
_cache_has = False


def invalidate() -> None:
    global _cache_has, _cached, _cached_error, _cached_at
    with _cache_lock:
        _cache_has, _cached, _cached_error, _cached_at = False, None, None, 0.0


def _policy() -> Optional[Policy]:
    """The cached row. Raises the cached (or fresh) read error so the caller fails closed.

    The read happens under the lock: one thread loads while the others wait for its result (no
    thundering herd at expiry), and a failed read is remembered for ``_FAIL_TTL`` so a database
    outage does not put a connection attempt in front of every external call.
    """
    global _cache_has, _cached, _cached_error, _cached_at
    with _cache_lock:
        age = time.monotonic() - _cached_at
        if _cache_has and age < (_FAIL_TTL if _cached_error is not None else _TTL):
            if _cached_error is not None:
                raise _cached_error
            return _cached
        try:
            loaded = _load_row()
        except Exception as exc:  # noqa: BLE001 - remembered, then re-raised: the caller fails closed
            _cache_has, _cached, _cached_error, _cached_at = True, None, exc, time.monotonic()
            raise
        _cache_has, _cached, _cached_error, _cached_at = True, loaded, None, time.monotonic()
        return loaded


def shadow_for(purpose: Purpose, host: str, agent: str, *, local: bool) -> Verdict:
    """What the gate WOULD decide for this call."""
    if local:
        return Verdict("allow", f"local destination {host}", DEFAULT_VERSION)
    try:
        policy = _policy()
    except Exception as exc:  # noqa: BLE001 - any failure to read must close, not open
        logger.warning("egress policy could not be read (%s): external destinations would be denied",
                       type(exc).__name__)
        return Verdict("deny", f"the egress policy could not be read, so {purpose.value} to {host} "
                               f"would be denied", UNREADABLE_VERSION)
    if policy is None:
        return Verdict("deny", f"no egress rule allows {purpose.value} to {host} for agent {agent} "
                               f"({DEFAULT_VERSION}: external destinations are denied until a rule allows them)",
                       DEFAULT_VERSION)
    return policy.decide(purpose, host, agent)


# ---- administration --------------------------------------------------------------------------------

def _details(rules: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"policy_identifier": IDENTIFIER, "rules": rules, "history": history}


def _history_line(version: int, by: str) -> Dict[str, Any]:
    return {"version": version, "changed_by": by, "changed_at": datetime.now(timezone.utc).isoformat(),
            "field": "rules"}


def _in_transaction(conn, work):
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
        try:
            conn.autocommit = previous
        except Exception:  # noqa: BLE001
            pass


def seed(by: str, conn=None) -> Dict[str, Any]:
    """Write the starting rules if there is no row. Never overwrites an administrator's rules."""
    if conn is None:
        with get_conn() as own:
            return seed(by, own)

    def work(cur):
        if _load(cur, for_update=True):
            return {"changed": False}
        try:
            cur.execute(
                "INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details, "
                "policy_linked_agents, policy_status, version, created_by, created_date) "
                "VALUES (%s, %s, %s, %s::jsonb, '', 1, 1, %s, now())",
                (POLICY_NAME, POLICY_TYPE, DESCRIPTION, json.dumps(_details(SEED_RULES, [_history_line(1, by)])), by))
        except Exception as exc:  # noqa: BLE001 - two seeds at once: the active-row unique index refuses the second
            raise RulesConflict("the egress policy was seeded by somebody else at the same moment; reload") from exc
        _record(conn, phase="configure", action_type="egress_policy_changed",
                summary=f"egress policy seeded by {by}",
                details={"changed_by": by, "version": 1, "rules": SEED_RULES})
        return {"changed": True, "version": 1}
    out = _in_transaction(conn, work)
    invalidate()
    return out


def replace_rules(rules: Any, by: str, expected_version: int, conn=None) -> Dict[str, Any]:
    """Make ``rules`` the policy. Validated first; refused if the row moved since ``expected_version``."""
    clean = validate(rules)
    if conn is None:
        with get_conn() as own:
            return replace_rules(clean, by, expected_version, own)

    def work(cur):
        got = _load(cur, for_update=True)
        if not got:
            raise RulesConflict("the egress policy has not been seeded yet")
        pid, version, details = got
        if version != expected_version:
            raise RulesConflict(f"the egress policy was changed by somebody else (now v{version}); reload")
        if details.get("rules") == clean:
            return {"version": version, "changed": False}
        history = list(details.get("history") or []) + [_history_line(version + 1, by)]
        merged = {**details, "rules": clean, "history": history}
        cur.execute(
            "UPDATE proc.bp_policy SET policy_details = %s::jsonb, version = version + 1, "
            "last_modified_by = %s, last_modified_date = now() WHERE policy_id = %s AND version = %s",
            (json.dumps(merged), by, pid, version))
        if cur.rowcount != 1:
            raise RulesConflict("the egress policy was changed by somebody else; reload and try again")
        _record(conn, phase="configure", action_type="egress_policy_changed",
                summary=f"egress policy rules replaced by {by} (v{version + 1})",
                details={"changed_by": by, "version": version + 1, "before": details.get("rules"),
                         "after": clean})
        return {"version": version + 1, "changed": True}
    out = _in_transaction(conn, work)
    invalidate()
    return out


def current(conn=None) -> Dict[str, Any]:
    """The rules in force for the administrator's screen."""
    if conn is None:
        with get_conn() as own:
            return current(own)
    got = _load(conn.cursor())
    if not got:
        # No row: the gate's shadow verdict is "deny everything external" (the built-in default), so the
        # rules in force are EMPTY. The seed is offered separately as what POST /seed would write.
        return {"configured": False, "version": 0, "rules": {"allow": [], "deny": []},
                "default_rules": True, "seed": SEED_RULES, "purposes": sorted(_PURPOSES)}
    _pid, version, details = got
    return {"configured": True, "version": version, "rules": details.get("rules") or {},
            "default_rules": False, "purposes": sorted(_PURPOSES),
            "history": details.get("history") or []}
