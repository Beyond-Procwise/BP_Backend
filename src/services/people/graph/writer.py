"""The only write path into the people graph. Every method goes through validate_edge (the admission rule)
and keeps the lifecycle: a change closes the old edge and opens a new one; nothing is overwritten."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from services.people.graph.model import DEFAULT_TENANT, DECISION_TYPES, Edge, NodeRef, utc
from services.people.graph.store import GraphStore


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GraphWriter:
    def __init__(self, store: GraphStore, tenant: str = DEFAULT_TENANT, now: Callable[[], datetime] = _now) -> None:
        self.store, self.tenant, self._now = store, tenant, now

    def node(self, ref: NodeRef, **props: Any) -> NodeRef:
        self.store.upsert_node(self.tenant, ref, props)
        return ref

    def assert_edge(self, src: NodeRef, type: str, dst: NodeRef, *, source: str, source_ref: str, asserted_by: str,
                    valid_from: Optional[datetime] = None, valid_to: Optional[datetime] = None, **extras: Any) -> Edge:
        props = {"source": source, "source_ref": source_ref, "asserted_by": asserted_by,
                 "valid_from": utc(valid_from or self._now()), **extras}
        if valid_to is not None:
            props["valid_to"] = utc(valid_to)
        return self.store.add_edge(Edge(self.tenant, src, type, dst, props))

    # ---- lifecycle helpers -------------------------------------------------------------------
    def _current(self, src: NodeRef, type: str, source: str):
        return self.store.edges(self.tenant, src=src, type=type, source=source, as_of=self._now())

    def set_single(self, src: NodeRef, type: str, dst: Optional[NodeRef], *, source: str, source_ref: str,
                   asserted_by: str, valid_to: Optional[datetime] = None, **extras: Any) -> str:
        """The (at most one) current target of ``type`` from ``src``. Returns 'unchanged'|'changed'|'cleared'."""
        return self.sync_set(src, type, {dst: dict(extras, **({"valid_to": valid_to} if valid_to else {}))} if dst else {},
                             source=source, source_ref=source_ref, asserted_by=asserted_by)

    def sync_set(self, src: NodeRef, type: str, desired: Dict[NodeRef, Dict[str, Any]], *, source: str,
                 source_ref: str, asserted_by: str) -> str:
        """Make the current edges of ``type`` from ``src`` (for this ``source``) equal ``desired``
        ({target: extra props}). Missing -> opened; surplus -> closed; changed extras -> closed + reopened.
        Idempotent: running it twice changes nothing the second time."""
        now = self._now()
        verdict = "unchanged"
        current = {}
        for e in self._current(src, type, source):
            if e.dst in desired and e.dst not in current:
                current[e.dst] = e
            else:
                self.store.close_edge(self.tenant, e.id, now); verdict = "changed"
        for dst, extras in desired.items():
            e = current.get(dst)
            extras = {k: v for k, v in extras.items() if v is not None}
            if e is not None:
                same = all(e.props.get(k) == (utc(v) if isinstance(v, datetime) else v) for k, v in extras.items())
                extra_keys = {k for k in e.props if k not in extras and k not in
                              ("source", "source_ref", "asserted_by", "valid_from", "valid_to")}
                if same and not extra_keys:
                    continue
                self.store.close_edge(self.tenant, e.id, now)
            vt = extras.pop("valid_to", None)
            self.assert_edge(src, type, dst, source=source, source_ref=source_ref, asserted_by=asserted_by,
                             valid_from=now, valid_to=vt, **extras)
            verdict = "changed"
        return verdict

    def record_decision(self, src: NodeRef, type: str, dst: NodeRef, *, event_id: str, decided_at: datetime,
                        asserted_by: str, source_ref: str, **extras: Any) -> Optional[Edge]:
        """One edge per explicit user action, idempotent by event_id. Returns None when already recorded."""
        assert type in DECISION_TYPES, type
        for e in self.store.edges(self.tenant, src=src, type=type, dst=dst, include_closed=True):
            if e.props.get("event_id") == event_id:
                return None
        return self.assert_edge(src, type, dst, source="user_action", source_ref=source_ref, asserted_by=asserted_by,
                                valid_from=decided_at, event_id=event_id, decided_at=utc(decided_at), **extras)
