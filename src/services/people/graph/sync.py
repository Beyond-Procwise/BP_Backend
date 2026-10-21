"""Sync jobs that fill the graph from systems of record. All idempotent: a second run changes nothing.

HR / IdP  -> bp_person: reporting line, title, department, delegation, spending authority.
RBAC      -> bp_role_assignment + RoleDefinitionPolicy: user roles, role->permission grants.
Decisions -> approval events and the existing approval tables: one edge per explicit action.
Runtime permission checks stay in the auth layer; this copy exists for reasoning and explanation.
If the two disagree the auth layer is right and this copy is stale: the sync reports what it had to change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from services.people.graph.model import NodeRef, utc
from services.people.graph.writer import GraphWriter

logger = logging.getLogger(__name__)
U = lambda k: NodeRef("User", k)


# ------------------------------------------------------------------ HR
def sync_hr(w: GraphWriter, people: Iterable[Dict[str, Any]], now: Optional[datetime] = None) -> Dict[str, int]:
    """``people``: dicts with subject, display_name, title, department_id, manager_subject, approval_limit,
    is_active, delegate_subject, delegate_until. Inactive people lose their org edges (their decision
    history stays)."""
    now = now or datetime.now(timezone.utc)
    stats = {"people": 0, "changed": 0}
    for p in people:
        stats["people"] += 1
        sub = p["subject"]
        ref = U(sub)
        w.node(ref, display_name=p.get("display_name") or sub, active=bool(p.get("is_active", True)))
        active = bool(p.get("is_active", True))
        meta = dict(source="hr", source_ref=f"bp_person:{sub}", asserted_by="sync:hr")
        verdicts = []
        verdicts.append(w.set_single(ref, "REPORTS_TO", U(p["manager_subject"]) if active and p.get("manager_subject") else None, **meta))
        title = (p.get("title") or "").strip()
        verdicts.append(w.set_single(ref, "HOLDS_TITLE", NodeRef("Title", title) if active and title else None, **meta))
        dept = p.get("department_id")
        verdicts.append(w.set_single(ref, "MEMBER_OF", NodeRef("Department", dept) if active and dept else None, **meta))
        until = p.get("delegate_until")
        if active and p.get("delegate_subject") and until and utc(until) > now:
            verdicts.append(w.set_single(ref, "DELEGATES_TO", U(p["delegate_subject"]), valid_to=until, **meta))
        else:
            verdicts.append(w.set_single(ref, "DELEGATES_TO", None, **meta))
        limit = p.get("approval_limit")
        auth = {}
        if active and limit is not None:
            et = NodeRef("EntityType", "approval")
            auth = {et: {"action": "approve", **({"unlimited": True} if float(limit) == -1 else {"max_amount": float(limit)})}}
        verdicts.append(w.sync_set(ref, "HAS_AUTHORITY", auth, **meta))
        stats["changed"] += sum(v == "changed" for v in verdicts)
    return stats


# ------------------------------------------------------------------ RBAC
@dataclass
class RbacSnapshot:
    assignments: Dict[str, List[str]]          # subject -> roles (direct, unrevoked)
    roles: Dict[str, List[str]]                # role -> allowed action classes
    actions: Dict[str, str]                    # action name -> class


def load_rbac_snapshot() -> RbacSnapshot:
    from services import rbac
    from services.actions import ACTIONS
    table = rbac._roles_table(None) or {}
    roles = {r: [str(c) for c in (e.get("allow") or [])] for r, e in table.items() if isinstance(e, dict)}
    return RbacSnapshot(rbac._load_role_assignments(), roles, dict(ACTIONS))


def sync_rbac(w: GraphWriter, snap: RbacSnapshot) -> Dict[str, int]:
    meta = dict(source="rbac", source_ref="RoleDefinitionPolicy", asserted_by="sync:rbac")
    changed = 0
    for action, cls in snap.actions.items():
        perm = NodeRef("Permission", action)
        w.node(perm, action_class=cls)
        changed += w.set_single(perm, "ON", NodeRef("EntityType", action.split(".")[0]), **meta) == "changed"
    for role, classes in snap.roles.items():
        r = NodeRef("Role", role)
        w.node(r)
        want = {NodeRef("Permission", a): {} for a, c in snap.actions.items() if c in classes}
        changed += w.sync_set(r, "GRANTS", want, **meta) == "changed"
    assign_meta = dict(source="rbac", source_ref="bp_role_assignment", asserted_by="sync:rbac")
    holders = {e.src.key for e in w.store.edges(w.tenant, type="HAS_ROLE", source="rbac")} | set(snap.assignments)
    for sub in holders:
        w.node(U(sub))
        changed += w.sync_set(U(sub), "HAS_ROLE", {NodeRef("Role", r): {} for r in snap.assignments.get(sub, [])},
                              **assign_meta) == "changed"
    if changed:
        logger.warning("people graph: RBAC mirror was out of date, %d edge set(s) changed", changed)
    return {"users": len(holders), "roles": len(snap.roles), "changed": int(changed)}


# ------------------------------------------------------------------ decisions
@dataclass
class DecisionFact:
    verb: str                     # approve | reject | request | escalate
    actor: str
    entity_key: str               # e.g. decision:123, approval:7
    event_id: str                 # stable across the live consumer and the backfill
    decided_at: datetime
    source_ref: str
    duration_ms: Optional[int] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    supplier_key: Optional[str] = None
    category_key: Optional[str] = None
    escalated_to: Optional[str] = None


_TYPE = {"approve": "APPROVED", "reject": "REJECTED", "request": "REQUESTED", "escalate": "ESCALATED"}


def apply_decisions(w: GraphWriter, facts: Iterable[DecisionFact]) -> Dict[str, int]:
    out = {"written": 0, "skipped": 0}
    for f in facts:
        extras = {k: v for k, v in dict(duration_ms=f.duration_ms, amount=f.amount, currency=f.currency,
                                        supplier_key=f.supplier_key, category_key=f.category_key,
                                        escalated_to=f.escalated_to).items() if v is not None}
        e = w.record_decision(U(f.actor), _TYPE[f.verb], NodeRef("Entity", f.entity_key), event_id=f.event_id,
                              decided_at=f.decided_at, asserted_by=f.actor, source_ref=f.source_ref, **extras)
        out["written" if e else "skipped"] += 1
        w.node(NodeRef("Entity", f.entity_key), entity_type=f.entity_key.split(":")[0])
    return out


def facts_from_events(conn, since: datetime) -> List[DecisionFact]:
    """approval.* domain events -> facts. Duration = decision time minus the request's own timestamp."""
    with conn.cursor() as cur:
        cur.execute("SELECT event_id, event_type, actor_id, entity_id, occurred_at, data FROM proc.bp_domain_event "
                    "WHERE event_type IN ('approval.requested','approval.approved','approval.rejected') "
                    "AND occurred_at >= %s AND actor_id NOT LIKE 'agent:%%' ORDER BY occurred_at", (since,))
        rows = cur.fetchall()
        requested_at = {}
        for eid, et, actor, ent, at, data in rows:
            if et == "approval.requested":
                requested_at[ent] = at
        out = []
        for eid, et, actor, ent, at, data in rows:
            verb = {"approval.requested": "request", "approval.approved": "approve", "approval.rejected": "reject"}[et]
            req = requested_at.get(ent)
            if req is None and verb != "request":
                cur.execute("SELECT occurred_at FROM proc.bp_domain_event WHERE event_id=%s", (f"approval.requested:{ent}",))
                r = cur.fetchone(); req = r[0] if r else None
            dur = int((at - req).total_seconds() * 1000) if verb != "request" and req else None
            out.append(DecisionFact(verb, actor, f"decision:{ent}", eid, at, f"bp_domain_event:{eid}", duration_ms=dur))
    return out


def facts_from_history(conn) -> List[DecisionFact]:
    """Backfill from the tables that already record who decided: bp_approval and policy approval cases."""
    out: List[DecisionFact] = []
    with conn.cursor() as cur:
        cur.execute("SELECT approval_id, status, actioned_by, actioned_at, amount, currency, supplier_id FROM proc.bp_approval "
                    "WHERE actioned_by IS NOT NULL AND actioned_at IS NOT NULL AND status IN ('approved','refused')")
        for aid, status, by, at, amount, cur_, sup in cur.fetchall():
            out.append(DecisionFact("approve" if status == "approved" else "reject", by, f"approval:{aid}",
                                    f"bp_approval:{aid}", at, f"bp_approval:{aid}",
                                    amount=float(amount) if amount is not None else None, currency=cur_,
                                    supplier_key=f"supplier:{sup}" if sup else None))
        cur.execute(
            "SELECT a.decision_id, a.decision, a.actioned_by, a.actioned_at, a.current_level, o.decision_id, o.created_at "
            "FROM proc.bp_decision a JOIN proc.bp_decision o ON o.subject_type=a.subject_type AND o.subject_id=a.subject_id "
            "AND o.actioned_by IS NULL AND o.decision='approve_or_reject' "
            "WHERE a.subject_type='agent_policy_approval' AND a.status='actioned' AND a.decision IN ('approve','reject') "
            "AND a.actioned_by IS NOT NULL AND a.actioned_by NOT LIKE 'system:%%'")
        for did, verb, by, at, level, orig, created in cur.fetchall():
            name = "approved" if verb == "approve" else "rejected"
            dur = int((at - created).total_seconds() * 1000) if created and at else None
            out.append(DecisionFact(verb, by, f"decision:{orig}", f"approval.{name}:{orig}:{level}", at,
                                    f"bp_decision:{did}", duration_ms=dur))
    return out


#: What a finding escalation is addressed to. The action records no person: the finding stays open in the queue it was
#: raised into ("a request for a human", decision_engine.STATUS_FOR_ACTION), so that is the honest destination.
FINDINGS_QUEUE = "findings_queue"


def facts_from_findings(conn) -> List[DecisionFact]:
    """A person escalating a finding on a PURCHASE ORDER -> an ESCALATED edge to the stub ``po:<po_id>``.

    Only the escalate verb: other finding outcomes (confirm, dismiss, flag, ...) are not decision edges until ruled on.
    The PO is the finding's own document (bp_extraction_discrepancy.doc_type = 'purchase_order', doc_pk_candidate);
    findings on invoices, quotes, contracts, or with no document id produce nothing. People only: system and agent
    actors are skipped, as for approvals."""
    out: List[DecisionFact] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT d.decision_id, d.actioned_by, d.actioned_at, d.supplier_id, f.doc_pk_candidate "
            "FROM proc.bp_decision d JOIN proc.bp_extraction_discrepancy f ON f.discrepancy_id::text = d.subject_id "
            "WHERE d.subject_type='finding' AND d.decision='escalate' AND d.status IN ('actioned','overridden') "
            "AND d.actioned_by IS NOT NULL AND d.actioned_at IS NOT NULL "
            "AND d.actioned_by NOT LIKE 'system:%%' AND d.actioned_by NOT LIKE 'agent:%%' "
            "AND f.doc_type='purchase_order' AND btrim(coalesce(f.doc_pk_candidate,'')) <> ''")
        for did, by, at, sup, po in cur.fetchall():
            out.append(DecisionFact("escalate", by, f"po:{po.strip()}", f"finding.escalated:{did}", at, f"bp_decision:{did}",
                                    supplier_key=f"supplier:{sup}" if sup else None, escalated_to=FINDINGS_QUEUE))
    return out


def load_people(conn) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT subject, display_name, title, department_id, manager_subject, approval_limit, is_active, "
                    "delegate_subject, delegate_until FROM proc.bp_person WHERE tenant_id=%s", (_tenant(),))
        keys = ["subject", "display_name", "title", "department_id", "manager_subject", "approval_limit", "is_active",
                "delegate_subject", "delegate_until"]
        return [dict(zip(keys, r)) for r in cur.fetchall()]


def _tenant() -> str:
    from services.people.admin import TENANT
    return TENANT
