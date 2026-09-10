"""The ordered resolution pass.

Stages run in the dependency order the spec declares (section 4.5): identity
first, then items, then contract succession and coverage. Each stage writes its
edges before the next reads them, because a later profile reads earlier edges as
signals.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.services.resolution import (
    CandidateEdge, CardinalityRule, ResolutionRequest, resolve,
)
from .edge_writer import DerivedEdge, write_edges
from .observations import observation_digest
from .profiles import supplier_identity as si

log = logging.getLogger(__name__)


def to_candidate_edges(scored: List[dict], profile_id: str) -> tuple:
    """Pairwise verdicts as MILP input. log_odds travels; F does not.

    CandidateEdge documents log_odds as "from the existing scorer, pre-sigmoid"
    and confidence as "post-sigmoid, for reporting only". Honour that: the
    solver reasons in log-odds.
    """
    return tuple(
        CandidateEdge(
            source_id=s["source_id"], target_id=s["target_id"],
            log_odds=s["result"]["L"], confidence=s["result"]["P_raw"],
            profile_id=profile_id,
        )
        for s in scored
    )


def band_for_resolution(band: str, status: Optional[str]) -> str:
    """A near-tie is not an auto-link, however high F went.

    DEGENERATE means the solver found the assignment barely forced -- another
    answer was nearly as good. Recording that as a certainty would be the
    precise thing this layer exists to prevent.
    """
    if status == "INFEASIBLE":
        return "weak_relation"
    if status == "DEGENERATE" and band == "auto_link":
        return "review"
    return band


def run_supplier_identity(conn: Any, driver: Any, limit: Optional[int] = None) -> dict:
    """Score supplier pairs, resolve globally, write SAME_ENTITY edges.

    bp_supplier_master is keyed SI###### (the uicanvas keyspace); the graph's
    Supplier nodes are keyed SUP-* (bp_supplier's keyspace). Those are
    different tables with different ids for the same company, so every row
    is bridged through bp_supplier_id_crosswalk to carry bp_supplier's id as
    `supplier_id` -- that is the identifier this function must put on
    source_id/target_id and therefore on the written edge, or write_edges's
    MATCH finds no node and silently writes nothing. The master's own
    attributes (name, VAT, etc.) still come from bp_supplier_master; only the
    identifier is swapped for the one the graph actually uses.
    """
    import psycopg2.extras
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT s.supplier_id AS supplier_id, m.supplier_name, m.vat_number,
                  m.registration_number, m.duns_number, m.postal_code,
                  m.country, m.bank_account_number
           FROM proc.bp_supplier_master m
           JOIN proc.bp_supplier_id_crosswalk x
             ON x.uicanvas_supplier_id = m.supplier_id
           JOIN proc.bp_supplier s ON s.supplier_id = x.bp_supplier_id
           ORDER BY m.supplier_id""" + (f" LIMIT {int(limit)}" if limit else "")
    )
    rows = [dict(r) for r in cur.fetchall()]

    scored = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            result = si.score(a, b)
            if result["F"] < 45.0:      # block_or_exception: nothing is said
                continue
            scored.append({"source_id": a["supplier_id"],
                           "target_id": b["supplier_id"],
                           "result": result, "src": a, "tgt": b})

    if not scored:
        return {"scored": 0, "written": 0, "status": None}

    request = ResolutionRequest(
        request_id="supplier_identity",
        edges=to_candidate_edges(scored, si.PROFILE),
        capacities=(),
        rules=(CardinalityRule(profile_id=si.PROFILE, shape="N:1",
                               max_targets_per_source=1),),
        profile_registry_version=si.VERSION,
    )
    outcome = resolve(request)
    kept = {(l.source_id, l.target_id): l for l in outcome.links}

    edges = []
    for s in scored:
        link = kept.get((s["source_id"], s["target_id"]))
        if link is None:
            continue
        r = s["result"]
        edges.append(DerivedEdge(
            rel_type="SAME_ENTITY",
            from_label="Supplier", from_key="supplier_id", from_value=s["source_id"],
            to_label="Supplier", to_key="supplier_id", to_value=s["target_id"],
            F=r["F"], band=band_for_resolution(r["decision"], outcome.status),
            P_raw=r["P_raw"], L_evidence=r["L_evidence"], profile=si.PROFILE,
            profile_version=si.VERSION, signals=r["signals"],
            observations=observation_digest(
                o for obs in si.observations_for(s["src"], s["tgt"]).values() for o in obs
            ),
            resolution=outcome.status, margin=link.margin_normalised,
        ))

    written = write_edges(driver, edges)
    log.info("supplier_identity: scored=%d kept=%d written=%d status=%s",
             len(scored), len(edges), written, outcome.status)
    return {"scored": len(scored), "written": written, "status": outcome.status}
