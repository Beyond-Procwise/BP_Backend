"""Who may decide for a decider name (a role such as "Finance Manager").

proc.bp_policy_decider_map links each name a policy refers to with the people who answer for it:
Cognito groups and/or individual email addresses. Eligibility is by link only. Admin is NOT
automatically eligible: whoever administers the system must not also approve its exceptions
(separation of duties).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

Mapping = Dict[str, Dict[str, List[str]]]


def load_map(conn) -> Mapping:
    """{decider_name: {"groups": [...], "emails": [...]}} from proc.bp_policy_decider_map."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT decider_name, groups, emails FROM proc.bp_policy_decider_map")
        rows = cur.fetchall()
    finally:
        try:
            cur.close()
        except Exception:  # noqa: BLE001
            pass
    out: Mapping = {}
    for name, groups, emails in rows or []:
        out[str(name)] = {"groups": [str(g) for g in groups or []],
                          "emails": [str(e).strip().lower() for e in emails or []]}
    return out


def _is_linked(entry: Optional[Dict[str, Any]]) -> bool:
    return bool(entry and (entry.get("groups") or entry.get("emails")))


def eligible(principal, decider_name: str, mapping: Mapping) -> bool:
    entry = (mapping or {}).get(decider_name)
    if not _is_linked(entry) or principal is None:
        return False
    claims = getattr(principal, "claims", None) or {}
    groups = claims.get("cognito:groups") or []
    if isinstance(groups, str):
        groups = [groups]
    if set(groups) & set(entry.get("groups") or []):
        return True
    email = (getattr(principal, "email", None) or "").strip().lower()
    return bool(email) and email in {e.lower() for e in entry.get("emails") or []}


def unmapped(names: Iterable[str], mapping: Mapping) -> List[str]:
    """The names nobody is linked to (absent, or present with no groups and no emails)."""
    seen: List[str] = []
    for n in names or []:
        n = str(n).strip()
        if n and n not in seen and not _is_linked((mapping or {}).get(n)):
            seen.append(n)
    return seen
