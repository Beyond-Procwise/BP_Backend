"""Questions the platform asks of the people graph. All tenant-scoped, all accept ``as_of``, and every
result carries a ``reason`` naming the edge(s) that justified it and a ``basis`` of asserted | derived | inferred.

Graph filters, embeddings rank: an optional ``ranker`` may only REORDER people the graph already returned;
anything it adds is dropped, so a vector store can never widen who is considered.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from services.people.graph.model import Edge, NodeRef, utc
from services.people.graph.store import GraphStore

U = lambda k: NodeRef("User", k)
ENTITY = lambda k: NodeRef("Entity", k)
ACTED = ("APPROVED", "REJECTED", "SIGNED", "ESCALATED")
BASIS = {"hr": "asserted", "rbac": "asserted", "user_action": "asserted", "derived": "derived", "inferred": "inferred"}


def _now(as_of: Optional[datetime]) -> datetime:
    return utc(as_of) if as_of else datetime.now(timezone.utc)


def _name(store: GraphStore, tenant: str, subject: str) -> str:
    n = store.node(tenant, U(subject)) or {}
    return n.get("display_name") or subject


def _covering(store: GraphStore, tenant: str, holders: Dict[str, str], t: datetime) -> Dict[str, str]:
    """Add people who cover for a holder through a delegation that is valid at ``t``."""
    out = dict(holders)
    for e in store.edges(tenant, type="DELEGATES_TO", as_of=t):
        if e.src.key in holders and e.dst.key not in out:
            out[e.dst.key] = f"covering for {_name(store, tenant, e.src.key)} (delegation)"
    return out


# ------------------------------------------------------------------ who can act
def who_can_act(store: GraphStore, tenant: str, action: str, *, amount: Optional[float] = None,
                as_of: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Walk Permission <- Role <- User, plus delegation. Deterministic; informs, never authorises."""
    t = _now(as_of)
    holders: Dict[str, str] = {}
    perm = NodeRef("Permission", action)
    roles = [e.src.key for e in store.edges(tenant, type="GRANTS", dst=perm, as_of=t)]
    for role in roles:
        for e in store.edges(tenant, type="HAS_ROLE", dst=NodeRef("Role", role), as_of=t):
            holders.setdefault(e.src.key, f"holds role {role}, which grants {action}")
    if action == "approve" or action.endswith(".approve"):
        for e in store.edges(tenant, type="HAS_AUTHORITY", as_of=t):
            if e.props.get("action") == "approve":
                holders.setdefault(e.src.key, "has approval authority")
    if amount is not None:
        covered = {}
        for e in store.edges(tenant, type="HAS_AUTHORITY", as_of=t):
            if e.props.get("action") == "approve" and (e.props.get("unlimited") or e.props.get("max_amount", 0) >= amount):
                covered[e.src.key] = e
        holders = {u: why + (" with unlimited authority" if covered[u].props.get("unlimited") else
                             f" with authority up to {covered[u].props['max_amount']:g}")
                   for u, why in holders.items() if u in covered}
    holders = _covering(store, tenant, holders, t)
    return [{"subject": u, "name": _name(store, tenant, u), "reason": why, "basis": "asserted"}
            for u, why in sorted(holders.items())]


# ------------------------------------------------------------------ who has acted
def who_has_acted(store: GraphStore, tenant: str, key: str, *, as_of: Optional[datetime] = None,
                  limit: int = 20) -> List[Dict[str, Any]]:
    """Decision edges on an entity, or on anything tagged with this supplier/category. Most recent first."""
    t = _now(as_of)
    seen: List[Edge] = []
    for typ in ACTED:
        for e in store.edges(tenant, type=typ, as_of=t, source="user_action"):
            if e.dst.key == key or e.props.get("supplier_key") == key or e.props.get("category_key") == key:
                seen.append(e)
    seen.sort(key=lambda e: utc(e.props["decided_at"]), reverse=True)
    return [{"subject": e.src.key, "name": _name(store, tenant, e.src.key), "action": e.type.lower(),
             "on": e.dst.key, "at": utc(e.props["decided_at"]).isoformat(), "basis": "asserted",
             "reason": f"{e.type.lower()} {e.dst.key} on {utc(e.props['decided_at']).date().isoformat()}"}
            for e in seen[:limit]]


# ------------------------------------------------------------------ who knows
def who_knows_about(store: GraphStore, tenant: str, key: str, *, k: int = 5,
                    as_of: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """From OWNS, ACTIVE_IN and decision edges. Never from messages."""
    t = _now(as_of)
    score: Dict[str, float] = {}
    why: Dict[str, List[str]] = {}
    basis: Dict[str, str] = {}

    def add(u, pts, text, b):
        score[u] = score.get(u, 0) + pts
        why.setdefault(u, []).append(text)
        basis[u] = "asserted" if basis.get(u, "asserted") == "asserted" and b == "asserted" else b

    for e in store.edges(tenant, type="OWNS", dst=ENTITY(key), as_of=t):
        primary = e.props.get("ownership_type") == "primary"
        add(e.src.key, 100 if primary else 50, f"{'primary' if primary else 'backup'} owner", BASIS[e.source])
    for e in store.edges(tenant, type="ACTIVE_IN", dst=ENTITY(key), as_of=t):
        add(e.src.key, 10 * float(e.props.get("confidence", 0)), f"has been deciding here (derived, "
            f"confidence {float(e.props.get('confidence', 0)):.2f})", BASIS[e.source])
    for a in who_has_acted(store, tenant, key, as_of=t, limit=1000):
        add(a["subject"], 1, f"{a['action']} {a['on']}", "asserted")
    ranked = sorted(score, key=lambda u: (-score[u], u))[:k]
    return [{"subject": u, "name": _name(store, tenant, u), "score": round(score[u], 2), "basis": basis[u],
             "reason": "; ".join(dict.fromkeys(why[u]))} for u in ranked]


# ------------------------------------------------------------------ bottlenecks
def find_bottlenecks(store: GraphStore, tenant: str, *, window_days: int = 90, as_of: Optional[datetime] = None,
                     min_decisions: int = 3, slow_ms: int = 24 * 3600 * 1000, min_escalations: int = 2) -> List[Dict[str, Any]]:
    from datetime import timedelta
    t = _now(as_of)
    start = t - timedelta(days=window_days)
    durations: Dict[str, List[int]] = {}
    escalations: Dict[str, int] = {}
    for typ in ("APPROVED", "REJECTED"):
        for e in store.edges(tenant, type=typ, as_of=t, source="user_action"):
            if utc(e.props["decided_at"]) >= start and e.props.get("duration_ms") is not None:
                durations.setdefault(e.src.key, []).append(int(e.props["duration_ms"]))
    for e in store.edges(tenant, type="ESCALATED", as_of=t, source="user_action"):
        if utc(e.props["decided_at"]) >= start:
            escalations[e.src.key] = escalations.get(e.src.key, 0) + 1
    out = []
    for u in set(durations) | set(escalations):
        d = durations.get(u, [])
        med = statistics.median(d) if len(d) >= min_decisions else None
        slow = med is not None and med >= slow_ms
        esc = escalations.get(u, 0) >= min_escalations
        if slow or esc:
            out.append({"subject": u, "name": _name(store, tenant, u), "decisions": len(d), "median_ms": med,
                        "escalations": escalations.get(u, 0), "basis": "asserted",
                        "reason": " and ".join(x for x in (
                            f"median {med / 3600000:.1f}h to decide over {len(d)} decisions" if slow else "",
                            f"{escalations[u]} escalations" if esc else "") if x)})
    return sorted(out, key=lambda r: (-(r["median_ms"] or 0), -r["escalations"], r["subject"]))


# ------------------------------------------------------------------ recipient suggestions (tiered)
def suggest_recipients(store: GraphStore, tenant: str, actor: str, *, entity_key: Optional[str] = None,
                       category_key: Optional[str] = None, supplier_key: Optional[str] = None,
                       amount: Optional[float] = None, k: int = 10, graph_tiers: bool = False,
                       ranker: Optional[Callable[[Sequence[str]], Sequence[str]]] = None,
                       as_of: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Ordered tiers, each result labelled. Tier 1 (asserted) always; tiers 2-4 only when ``graph_tiers`` is on,
    which is off by default per tenant."""
    t = _now(as_of)
    out: List[Dict[str, Any]] = []
    seen = {actor}

    def add(tier, subject, reason, basis):
        if subject not in seen:
            seen.add(subject)
            out.append({"tier": tier, "subject": subject, "name": _name(store, tenant, subject), "reason": reason, "basis": basis})

    for e in store.edges(tenant, src=U(actor), type="REPORTS_TO", as_of=t):
        add(1, e.dst.key, "your manager", "asserted")
    for a in who_can_act(store, tenant, "approve", amount=amount if amount is not None else None, as_of=t):
        if a["subject"] != actor:
            add(1, a["subject"], f"approver: {a['reason']}", "asserted")
    for key in (entity_key, category_key, supplier_key):
        if key:
            for e in store.edges(tenant, type="OWNS", dst=ENTITY(key), as_of=t):
                add(1, e.src.key, f"{e.props.get('ownership_type', 'owner')} owner of {key}", BASIS[e.source])
    if graph_tiers:
        tier23: List[Dict[str, Any]] = []
        for key in (entity_key, category_key, supplier_key):
            if key:
                for a in who_has_acted(store, tenant, key, as_of=t):
                    if a["subject"] not in seen:
                        seen.add(a["subject"]); tier23.append(
                            {"tier": 2, "subject": a["subject"], "name": a["name"],
                             "reason": f"has {a['action']} {a['on']}", "basis": "asserted"})
        for e in store.edges(tenant, src=U(actor), type="WORKS_WITH", as_of=t):
            if e.dst.key not in seen:
                seen.add(e.dst.key); tier23.append(
                    {"tier": 3, "subject": e.dst.key, "name": _name(store, tenant, e.dst.key),
                     "reason": f"often works with you (confidence {float(e.props.get('confidence', 0)):.2f})", "basis": BASIS[e.source]})
        if ranker and tier23:
            by_subject = {r["subject"]: r for r in tier23}
            order = [x for x in ranker(list(by_subject)) if x in by_subject]       # anything extra is dropped
            order += [x for x in by_subject if x not in order]
            tier23 = [dict(by_subject[x], ranked=True) for x in order]
        out.extend(tier23)
    return out[:k]
