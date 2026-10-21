"""The proposed_edge queue: the only door through which an inference (for example from embeddings) can reach the graph.

An inference is written here with confidence, model version and evidence. It is invisible to every graph
query until an administrator (or the entity owner) confirms it; confirming writes ONE edge tagged
source=inferred, confirmed_by=<person>. The graph never learns from vectors silently.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.people.admin import PeopleError, TENANT
from services.people.graph.model import EDGE_RULES, NodeRef, AdmissionError
from services.people.graph.writer import GraphWriter

INFERABLE = {"WORKS_WITH", "ACTIVE_IN", "OWNS"}


def propose(conn, src: NodeRef, edge_type: str, dst: NodeRef, *, confidence: float, model_version: str,
            evidence: Dict[str, Any], props: Optional[Dict[str, Any]] = None) -> Optional[int]:
    if edge_type not in INFERABLE:
        raise PeopleError(422, f"only {', '.join(sorted(INFERABLE))} can be proposed")
    if not (0 <= confidence <= 1) or not model_version:
        raise PeopleError(422, "a proposal needs a confidence between 0 and 1 and a model_version")
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_proposed_edge (tenant_id, src_label, src_key, edge_type, dst_label, dst_key, props,"
                    " confidence, model_version, evidence) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb) "
                    "ON CONFLICT DO NOTHING RETURNING id",
                    (TENANT, src.label, src.key, edge_type, dst.label, dst.key, json.dumps(props or {}), confidence,
                     model_version, json.dumps(evidence, default=str)))
        r = cur.fetchone()
        return r[0] if r else None


def list_proposals(conn, status: str = "proposed", limit: int = 100) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, src_label, src_key, edge_type, dst_label, dst_key, props, confidence, model_version, evidence, "
                    "status, proposed_at, decided_by FROM proc.bp_proposed_edge WHERE tenant_id=%s AND status=%s "
                    "ORDER BY proposed_at DESC LIMIT %s", (TENANT, status, limit))
        return [{"id": r[0], "from": {"label": r[1], "key": r[2]}, "type": r[3], "to": {"label": r[4], "key": r[5]},
                 "props": r[6], "confidence": float(r[7]), "model_version": r[8], "evidence": r[9], "status": r[10],
                 "proposed_at": r[11].isoformat(), "decided_by": r[12]} for r in cur.fetchall()]


def decide(conn, w: GraphWriter, proposal_id: int, by: str, confirm: bool) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT src_label, src_key, edge_type, dst_label, dst_key, props, confidence, model_version, status "
                    "FROM proc.bp_proposed_edge WHERE id=%s AND tenant_id=%s FOR UPDATE", (proposal_id, TENANT))
        r = cur.fetchone()
        if r is None:
            raise PeopleError(404, "no such proposal")
        if r[8] != "proposed":
            raise PeopleError(409, f"this proposal was already {r[8]}")
        edge = None
        if confirm:
            props = dict(r[5] or {})
            if r[2] == "OWNS":
                props.setdefault("ownership_type", "backup")
            if r[2] in ("WORKS_WITH", "ACTIVE_IN"):
                now = datetime.now(timezone.utc)
                props.update(evidence_count=int((r[5] or {}).get("evidence_count", 1)), computed_at=now, window_days=0)
            try:
                edge = w.assert_edge(NodeRef(r[0], r[1]), r[2], NodeRef(r[3], r[4]), source="inferred",
                                     source_ref=f"bp_proposed_edge:{proposal_id}", asserted_by=by, confirmed_by=by,
                                     model_version=r[7], confidence=float(r[6]), **props)
            except AdmissionError as exc:
                raise PeopleError(422, str(exc))
        cur.execute("UPDATE proc.bp_proposed_edge SET status=%s, decided_by=%s, decided_at=now() WHERE id=%s",
                    ("confirmed" if confirm else "rejected", by, proposal_id))
    return {"id": proposal_id, "status": "confirmed" if confirm else "rejected", "edge_id": edge.id if edge else None}
