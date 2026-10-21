"""Runtime glue: the live graph store, the refresh job, the per-tenant tier switch and the learning loop."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from services.people.admin import PeopleError, TENANT
from services.people.graph import activity, derive, legacy_link, sync
from services.people.graph.model import DEFAULT_TENANT
from services.people.graph.store import GraphStore, Neo4jGraph
from services.people.graph.writer import GraphWriter

logger = logging.getLogger(__name__)
TIERS_KEY = "people.graph_tiers"
_store: Optional[GraphStore] = None


def get_store() -> GraphStore:
    """The app's own Neo4j connection (same config path as the rest of the app)."""
    global _store
    if _store is None:
        from services.platform_kg import _driver
        _store = Neo4jGraph(_driver())
    return _store


def writer(store: Optional[GraphStore] = None) -> GraphWriter:
    return GraphWriter(store or get_store(), TENANT)


def refresh(conn, store: Optional[GraphStore] = None, *, event_days: int = 30, now: Optional[datetime] = None,
            retention_days: int = activity.DEFAULT_RETENTION_DAYS, cfg: derive.DeriveConfig = derive.DeriveConfig()) -> Dict[str, Any]:
    """The daily job: HR + RBAC mirrors, decision edges (recent events AND history), derivation, retention."""
    now = now or datetime.now(timezone.utc)
    store = store or get_store()
    w = GraphWriter(store, TENANT, now=lambda: now)
    out: Dict[str, Any] = {"hr": sync.sync_hr(w, sync.load_people(conn), now=now)}
    try:
        out["rbac"] = sync.sync_rbac(w, sync.load_rbac_snapshot())
    except Exception:
        logger.exception("RBAC mirror failed; graph keeps its previous copy")
        out["rbac"] = {"error": True}
    facts = sync.facts_from_history(conn) + sync.facts_from_events(conn, now - timedelta(days=event_days))
    out["decisions"] = sync.apply_decisions(w, facts)
    since = now - timedelta(days=cfg.window_days)
    out["derived"] = derive.rebuild(w, derive.load_interactions(conn, TENANT, since),
                                    derive.load_decisions(store, TENANT, since), derive.load_dismissals(conn, TENANT, since), now, cfg)
    try:                                       # a missing/ambiguous older node is reported, never fatal
        out["legacy_links"] = legacy_link.link_all(store, TENANT)
    except Exception:
        logger.warning("linking stubs to existing nodes failed", exc_info=True)
        out["legacy_links"] = {"error": True}
    out["pruned_activity"] = activity.prune(conn, retention_days, now)
    return out


# ------------------------------------------------------------------ the tier switch (off by default)
def tiers_enabled(conn) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT config_value FROM proc.bp_admin_config WHERE config_key=%s", (TIERS_KEY,))
            r = cur.fetchone()
        return bool(r and isinstance(r[0], dict) and r[0].get("enabled") is True)
    except Exception:
        return False                       # unreadable means off, never on


def set_tiers(conn, enabled: bool, by: str, *, force: bool = False, min_shown: int = 30, min_rate: float = 0.2) -> Dict[str, Any]:
    """Tiers 2-4 go live for the tenant only once the tier-1 suggestions have been seen and used.
    Without that data the switch refuses (409) unless an administrator forces it."""
    if enabled and not force:
        report = acceptance_report(conn)
        t1 = report["tiers"].get("1", {"shown": 0, "rate": 0.0})
        if t1["shown"] < min_shown or t1["rate"] < min_rate:
            raise PeopleError(409, f"not enough evidence yet: tier 1 has been shown {t1['shown']} times "
                                   f"(need {min_shown}) with an acceptance rate of {t1['rate']:.0%} (need {min_rate:.0%}). "
                                   "An administrator can switch it on anyway with force.")
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by) VALUES (%s,%s::jsonb,%s) "
                    "ON CONFLICT (config_key) DO UPDATE SET config_value=EXCLUDED.config_value, "
                    "last_modified_by=EXCLUDED.last_modified_by, last_modified_date=now()",
                    (TIERS_KEY, json.dumps({"enabled": bool(enabled), "forced": bool(force and enabled)}), by))
    return {"enabled": bool(enabled)}


def acceptance_report(conn, days: int = 90) -> Dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(metadata->>'tier','?'), verb, count(*) FROM proc.bp_activity WHERE tenant_id=%s "
                    "AND occurred_at >= %s AND verb IN ('suggestion.shown','suggestion.accepted','suggestion.dismissed') "
                    "GROUP BY 1,2", (TENANT, since))
        rows = cur.fetchall()
    tiers: Dict[str, Dict[str, Any]] = {}
    for tier, verb, n in rows:
        t = tiers.setdefault(tier, {"shown": 0, "accepted": 0, "dismissed": 0})
        t[verb.split(".")[1]] = int(n)
    for t in tiers.values():
        t["rate"] = (t["accepted"] / t["shown"]) if t["shown"] else 0.0
    return {"window_days": days, "tiers": tiers, "graph_tiers_enabled": tiers_enabled(conn)}
