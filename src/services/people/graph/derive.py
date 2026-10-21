"""Derivation: the only path from behaviour into the graph, and it admits just two advisory edges.

WORKS_WITH(a,b)  only when, inside the window, a and b addressed each other at least N times across at least
                 M distinct entities. Chat volume on one entity (or none) never qualifies.
ACTIVE_IN(u,x)   only from DECISION edges (approve / reject / sign) on a category or supplier, weighted by
                 outcome, never from views or messages.
Both carry source=derived, confidence, evidence_count, computed_at, window_days. Confidence decays with a
half-life, falls with each dismissed suggestion that pointed at it, and an edge below the floor is not kept.
Derived edges are rebuilt every run (they are advisory, not history) and every consumer must treat them so.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from services.people.graph.model import NodeRef, utc
from services.people.graph.writer import GraphWriter

OUTCOME_WEIGHT = {"APPROVED": 1.0, "SIGNED": 1.0, "REJECTED": 0.5}
DISMISS_PENALTY = 0.8


@dataclass(frozen=True)
class DeriveConfig:
    n: int = 8                     # interactions
    m: int = 3                     # distinct entities
    window_days: int = 90
    half_life_days: float = 42.0   # six weeks
    floor: float = 0.2
    active_min: int = 3            # decisions on a category/supplier before ACTIVE_IN is stated


@dataclass
class Interaction:
    a: str
    b: str
    at: datetime
    entity: Optional[str] = None


def decayed(at: datetime, now: datetime, half_life_days: float) -> float:
    age = max((now - at).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age / half_life_days)


def confidence(weighted: float, n: int) -> float:
    """0.5 when the decayed evidence equals the threshold N, rising toward 1."""
    return 1.0 - 0.5 ** (weighted / max(n, 1))


def works_with(interactions: Iterable[Interaction], dismissals: Dict[Tuple[str, str], int], now: datetime,
               cfg: DeriveConfig) -> List[dict]:
    start = now - timedelta(days=cfg.window_days)
    pairs: Dict[Tuple[str, str], List[Interaction]] = defaultdict(list)
    for i in interactions:
        if i.a == i.b or not (start <= utc(i.at) <= now):
            continue
        pairs[tuple(sorted((i.a, i.b)))].append(i)
    out = []
    for (a, b), items in pairs.items():
        entities = {i.entity for i in items if i.entity}
        if len(items) < cfg.n or len(entities) < cfg.m:
            continue
        w = sum(decayed(utc(i.at), now, cfg.half_life_days) for i in items)
        conf = confidence(w, cfg.n) * DISMISS_PENALTY ** (dismissals.get((a, b), 0) + dismissals.get((b, a), 0))
        if conf >= cfg.floor:
            out.append({"a": a, "b": b, "confidence": round(conf, 4), "evidence_count": len(items)})
    return out


def active_in(decisions: Iterable[Tuple[str, str, str, datetime]], dismissals: Dict[Tuple[str, str], int],
              now: datetime, cfg: DeriveConfig) -> List[dict]:
    """decisions: (user, edge_type, target_key like 'supplier:S1' / 'category:IT', decided_at)."""
    start = now - timedelta(days=cfg.window_days)
    groups: Dict[Tuple[str, str], List[Tuple[str, datetime]]] = defaultdict(list)
    for user, etype, target, at in decisions:
        if etype in OUTCOME_WEIGHT and start <= utc(at) <= now:
            groups[(user, target)].append((etype, utc(at)))
    out = []
    for (user, target), items in groups.items():
        if len(items) < cfg.active_min:
            continue
        w = sum(OUTCOME_WEIGHT[t] * decayed(at, now, cfg.half_life_days) for t, at in items)
        conf = confidence(w, cfg.n) * DISMISS_PENALTY ** dismissals.get((user, target), 0)
        if conf >= cfg.floor:
            out.append({"user": user, "target": target, "confidence": round(conf, 4), "evidence_count": len(items)})
    return out


def rebuild(w: GraphWriter, interactions: Iterable[Interaction], decisions: Iterable[Tuple[str, str, str, datetime]],
            dismissals: Dict[Tuple[str, str], int], now: Optional[datetime] = None,
            cfg: DeriveConfig = DeriveConfig()) -> Dict[str, int]:
    """Replace this tenant's derived edges with the freshly computed set."""
    now = now or datetime.now(timezone.utc)
    ww = works_with(interactions, dismissals, now, cfg)
    ai = active_in(decisions, dismissals, now, cfg)
    w.store.delete_edges(w.tenant, source="derived", type="WORKS_WITH")
    w.store.delete_edges(w.tenant, source="derived", type="ACTIVE_IN")
    common = dict(source="derived", source_ref="derivation:daily", asserted_by="system:derivation", valid_from=now,
                  computed_at=now, window_days=cfg.window_days)
    for r in ww:
        for x, y in ((r["a"], r["b"]), (r["b"], r["a"])):          # both directions so either person's query finds it
            w.assert_edge(NodeRef("User", x), "WORKS_WITH", NodeRef("User", y), confidence=r["confidence"],
                          evidence_count=r["evidence_count"], **common)
    for r in ai:
        kind = r["target"].split(":")[0]
        w.assert_edge(NodeRef("User", r["user"]), "ACTIVE_IN", NodeRef("Entity", r["target"]),
                      confidence=r["confidence"], evidence_count=r["evidence_count"], **common)
    return {"works_with": len(ww), "active_in": len(ai)}


# ------------------------------------------------------------------ reading the inputs from Postgres / graph
def load_interactions(conn, tenant: str, since: datetime) -> List[Interaction]:
    """Addressed interactions: direct messages and mentions/comments aimed at a person, with the entity if any."""
    with conn.cursor() as cur:
        cur.execute("SELECT actor_id, target_user_id, occurred_at, CASE WHEN entity_type IS NULL THEN NULL "
                    "ELSE entity_type || ':' || entity_id END FROM proc.bp_activity WHERE tenant_id=%s AND occurred_at >= %s "
                    "AND target_user_id IS NOT NULL AND verb IN ('message.sent','mention','comment')", (tenant, since))
        return [Interaction(a, b, at, ent) for a, b, at, ent in cur.fetchall()]


def load_dismissals(conn, tenant: str, since: datetime) -> Dict[Tuple[str, str], int]:
    with conn.cursor() as cur:
        cur.execute("SELECT actor_id, target_user_id, metadata->>'target_key', count(*) FROM proc.bp_activity "
                    "WHERE tenant_id=%s AND verb='suggestion.dismissed' AND occurred_at >= %s GROUP BY 1,2,3", (tenant, since))
        out: Dict[Tuple[str, str], int] = {}
        for actor, target_user, target_key, n in cur.fetchall():
            key = (actor, target_user or target_key)
            if key[1]:
                out[key] = out.get(key, 0) + int(n)
        return out


def load_decisions(store, tenant: str, since: datetime) -> List[Tuple[str, str, str, datetime]]:
    out = []
    for t in OUTCOME_WEIGHT:
        for e in store.edges(tenant, type=t, source="user_action", as_of=datetime.now(timezone.utc), include_closed=True):
            at = e.props.get("decided_at")
            if at is None or utc(at) < since:
                continue
            for k in ("supplier_key", "category_key"):
                if e.props.get(k):
                    out.append((e.src.key, t, e.props[k], at))
    return out
