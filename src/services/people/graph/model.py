"""The people knowledge graph: what may enter it, and in what shape.

The knowledge graph is the system of **asserted facts**: identity, titles, roles, permissions, ownership, reporting lines, and recorded decisions, each with a source and a validity window. The activity store is the system of **observed behaviour**: messages, views, mentions, response times. The embedding index is the system of **meaning**: semantic similarity over content. Behaviour may be summarised into the graph only through explicit, thresholded derivation jobs; meaning may be proposed to the graph only through a confirmed inference with recorded confidence. Runtime authorisation is never taken from the graph. Any retrieval that involves people is filtered by the graph before it is ranked by embeddings.

ADMISSION RULE. An edge enters the graph only if
  1. it has an authoritative source (HR/IdP, the RBAC configuration, or an explicit user action) that
     can be named: ``source`` + ``source_ref`` (the table row or event it came from);
  2. it is a true/false statement, not an impression (so no edge exists because of frequency alone);
  3. it carries a lifecycle: ``asserted_by``, ``valid_from``, ``valid_to`` (open = None). A change
     closes the old edge and opens a new one; nothing is overwritten.
``validate_edge`` enforces this at the single write path; an edge that fails is rejected with
AdmissionError, never stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, Optional, Tuple

DEFAULT_TENANT = "default"

NODE_LABELS = frozenset({"User", "Title", "Role", "Permission", "Department", "Team", "Entity", "EntityType"})

SOURCES = frozenset({"hr", "rbac", "user_action", "derived", "inferred"})

#: edge type -> (allowed sources, allowed source labels, allowed target labels, required extra props)
EDGE_RULES: Dict[str, Tuple[FrozenSet[str], FrozenSet[str], FrozenSet[str], FrozenSet[str]]] = {
    # identity & organisation (HR / IdP / user admin)
    "REPORTS_TO": (frozenset({"hr"}), frozenset({"User"}), frozenset({"User"}), frozenset()),
    "HOLDS_TITLE": (frozenset({"hr"}), frozenset({"User"}), frozenset({"Title"}), frozenset()),
    "MEMBER_OF": (frozenset({"hr"}), frozenset({"User"}), frozenset({"Department", "Team"}), frozenset()),
    "DELEGATES_TO": (frozenset({"hr"}), frozenset({"User"}), frozenset({"User"}), frozenset()),
    # spending authority as set on the person (max_amount, or unlimited)
    "HAS_AUTHORITY": (frozenset({"hr"}), frozenset({"User"}), frozenset({"EntityType"}), frozenset({"action"})),
    # RBAC, mirrored from the RBAC configuration; never authored here
    "HAS_ROLE": (frozenset({"rbac"}), frozenset({"User"}), frozenset({"Role"}), frozenset()),
    "GRANTS": (frozenset({"rbac"}), frozenset({"Role"}), frozenset({"Permission"}), frozenset()),
    "ON": (frozenset({"rbac"}), frozenset({"Permission"}), frozenset({"EntityType"}), frozenset()),
    # ownership: explicit assignment in the product
    "OWNS": (frozenset({"user_action", "inferred"}), frozenset({"User"}), frozenset({"Entity"}), frozenset({"ownership_type"})),
    "WATCHES": (frozenset({"user_action"}), frozenset({"User"}), frozenset({"Entity"}), frozenset()),
    # decisions: one edge per explicit action, never aggregated away
    "REQUESTED": (frozenset({"user_action"}), frozenset({"User"}), frozenset({"Entity"}), frozenset({"event_id", "decided_at"})),
    "APPROVED": (frozenset({"user_action"}), frozenset({"User"}), frozenset({"Entity"}), frozenset({"event_id", "decided_at"})),
    "REJECTED": (frozenset({"user_action"}), frozenset({"User"}), frozenset({"Entity"}), frozenset({"event_id", "decided_at"})),
    "SIGNED": (frozenset({"user_action"}), frozenset({"User"}), frozenset({"Entity"}), frozenset({"event_id", "decided_at"})),
    "ESCALATED": (frozenset({"user_action"}), frozenset({"User"}), frozenset({"Entity"}), frozenset({"event_id", "decided_at", "escalated_to"})),
    "ASSIGNED": (frozenset({"user_action"}), frozenset({"User"}), frozenset({"Entity"}), frozenset({"event_id", "decided_at"})),
    "ASSIGNED_TO": (frozenset({"user_action"}), frozenset({"User"}), frozenset({"User"}), frozenset({"event_id", "decided_at"})),
    # the ONLY two edges behaviour may produce, and only through the derivation job (advisory)
    "WORKS_WITH": (frozenset({"derived", "inferred"}), frozenset({"User"}), frozenset({"User"}),
                   frozenset({"confidence", "evidence_count", "computed_at", "window_days"})),
    "ACTIVE_IN": (frozenset({"derived", "inferred"}), frozenset({"User"}), frozenset({"Entity"}),
                  frozenset({"confidence", "evidence_count", "computed_at", "window_days"})),
}

DECISION_TYPES = frozenset({"REQUESTED", "APPROVED", "REJECTED", "SIGNED", "ESCALATED", "ASSIGNED", "ASSIGNED_TO"})

#: Frequency is not a fact. These are refused by name with a message that says why.
EXCLUDED_EDGE_TYPES = frozenset({"MESSAGED", "VIEWED", "COLLABORATED_ON", "MENTIONED", "COMMENTED_ON", "SENT_MESSAGE",
                                 "CHATTED_WITH", "READ"})


class AdmissionError(ValueError):
    """The edge does not satisfy the admission rule and was not written."""


@dataclass(frozen=True)
class NodeRef:
    label: str
    key: str

    def __post_init__(self):
        if self.label not in NODE_LABELS:
            raise AdmissionError(f"unknown node label {self.label!r}")
        if not str(self.key).strip():
            raise AdmissionError("a node needs a key")


@dataclass
class Edge:
    tenant: str
    src: NodeRef
    type: str
    dst: NodeRef
    props: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    @property
    def source(self) -> str:
        return self.props.get("source", "")

    @property
    def valid_from(self) -> Optional[datetime]:
        return self.props.get("valid_from")

    @property
    def valid_to(self) -> Optional[datetime]:
        return self.props.get("valid_to")

    def current_at(self, t: datetime) -> bool:
        vf, vt = self.valid_from, self.valid_to
        return vf is not None and vf <= t and (vt is None or vt > t)


def utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _primitive(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool, datetime)) or (
        isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v))


def validate_edge(edge: Edge) -> None:
    t = edge.type
    if t in EXCLUDED_EDGE_TYPES:
        raise AdmissionError(f"{t} is excluded: views, messages, mentions and similar behaviour belong in the "
                             "activity store, never as graph edges")
    rule = EDGE_RULES.get(t)
    if rule is None:
        raise AdmissionError(f"unknown edge type {t!r}")
    sources, src_labels, dst_labels, extras = rule
    p = edge.props
    src = p.get("source")
    if src not in SOURCES:
        raise AdmissionError(f"{t}: an edge needs a source, one of {sorted(SOURCES)}")
    if src not in sources:
        raise AdmissionError(f"{t} cannot come from source {src!r}; allowed: {sorted(sources)}")
    if not str(p.get("source_ref") or "").strip():
        raise AdmissionError(f"{t}: source_ref is required (the table row or event this came from)")
    if not str(p.get("asserted_by") or "").strip():
        raise AdmissionError(f"{t}: asserted_by is required")
    if not isinstance(p.get("valid_from"), datetime):
        raise AdmissionError(f"{t}: valid_from is required")
    vt = p.get("valid_to")
    if vt is not None and (not isinstance(vt, datetime) or vt <= p["valid_from"]):
        raise AdmissionError(f"{t}: valid_to must be after valid_from")
    if edge.src.label not in src_labels or edge.dst.label not in dst_labels:
        raise AdmissionError(f"{t} cannot join {edge.src.label} to {edge.dst.label}")
    missing = sorted(k for k in extras if p.get(k) is None)
    if missing:
        raise AdmissionError(f"{t}: missing {', '.join(missing)}")
    if src == "inferred" and not (p.get("confirmed_by") and p.get("model_version") and p.get("confidence") is not None):
        raise AdmissionError(f"{t}: an inferred edge needs confirmed_by, model_version and confidence")
    if src == "derived" and not isinstance(p.get("confidence"), (int, float)):
        raise AdmissionError(f"{t}: a derived edge needs a numeric confidence")
    bad = [k for k, v in p.items() if not _primitive(v)]
    if bad:
        raise AdmissionError(f"{t}: properties must be plain values ({', '.join(bad)})")
    if not edge.tenant:
        raise AdmissionError("every edge is tenant-scoped")
