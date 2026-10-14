"""Who started a draft, who reviewed it, who sent it. Three roles, never conflated.

Authority checks and style learning key off the human reviewer / sender, never the initiator.
An agent-initiated draft with no human reviewer cannot be sent.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def initiated(user_id: Optional[str], agent: str, agent_ids: Iterable[str] = ()) -> Dict[str, str]:
    """A person if the request came from one, otherwise the agent. Known agent ids are never persons."""

    agents = {a for a in agent_ids if a}
    if user_id and user_id not in agents:
        return {"id": str(user_id), "kind": "user"}
    return {"id": agent, "kind": "agent"}


def reviewer_problem(reviewed_by: Optional[str], initiator: Optional[Dict[str, str]],
                     autonomous: bool = False) -> Optional[str]:
    """Why this send has no usable human reviewer, or None if it does."""

    if autonomous:
        return "no human reviewed this draft"
    name = str(reviewed_by or "").strip()
    if not name:
        return "the approval names no person"
    if initiator and initiator.get("kind") == "agent" and name == initiator.get("id"):
        return "the reviewer is the agent that wrote the draft"
    return None


def mailbox_owner(conn: Any, mailbox: Optional[str]) -> Optional[str]:
    """The user a bound mailbox belongs to (an ACTIVE binding only), or None."""

    if not mailbox:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_ref FROM proc.bp_mailbox_binding "
                        "WHERE lower(mailbox_address) = lower(%s) AND is_active LIMIT 2", [mailbox])
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001
        return None
    return str(rows[0][0]) if len(rows) == 1 and rows[0][0] else None
