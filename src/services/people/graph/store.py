"""GraphStore: the few operations the people graph needs, with an in-memory and a Neo4j implementation.

All reads and writes are tenant-scoped. Nodes of the spec's own labels (User, Role, ...) are matched on
(pw_tenant, pw_key). Anything people act on is an ``Entity`` stub keyed ``type:id`` so the pre-existing
Supplier / PurchaseOrder nodes that other code queries are never touched.
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Protocol

from services.people.graph.model import (EDGE_RULES, NODE_LABELS, AdmissionError, Edge, NodeRef, utc, validate_edge)


# Pre-existing node types a stub may be linked to, and the property each is keyed on elsewhere in the app.
LEGACY_PROPS = {"Supplier": "supplier_id", "PurchaseOrder": "po_id"}


class GraphStore(Protocol):
    def upsert_node(self, tenant: str, ref: NodeRef, props: Optional[dict] = None) -> None: ...
    def add_edge(self, edge: Edge) -> Edge: ...
    def edges(self, tenant: str, *, src: Optional[NodeRef] = None, type: Optional[str] = None,
              dst: Optional[NodeRef] = None, source: Optional[str] = None, as_of: Optional[datetime] = None,
              include_closed: bool = False) -> List[Edge]: ...
    def close_edge(self, tenant: str, edge_id: str, at: datetime) -> None: ...
    def delete_edges(self, tenant: str, *, source: Optional[str] = None, type: Optional[str] = None,
                     src: Optional[NodeRef] = None, dst: Optional[NodeRef] = None) -> int: ...
    def node(self, tenant: str, ref: NodeRef) -> Optional[dict]: ...
    def rename_node(self, tenant: str, ref: NodeRef, new_key: str, drop_props: Iterable[str] = ()) -> bool: ...
    def entity_keys(self, tenant: str, prefix: str) -> List[str]:
        """Keys of Entity stubs starting with prefix that are not yet linked to a pre-existing node."""
    def link_legacy(self, tenant: str, ref: NodeRef, label: str, prop: str, value: str) -> str:
        """Link a stub to the ONE pre-existing node with label.prop == value: 'linked' | 'missing' | 'ambiguous'.
        Never creates or changes the pre-existing node."""


def _match(e: Edge, src, type, dst, source) -> bool:
    return ((src is None or e.src == src) and (type is None or e.type == type) and (dst is None or e.dst == dst)
            and (source is None or e.source == source))


class InMemoryGraph:
    def __init__(self) -> None:
        self._nodes: Dict[tuple, dict] = {}
        self._edges: Dict[str, Edge] = {}
        self._ids = itertools.count(1)
        self.legacy: Dict[tuple, int] = {}          # (label, prop, value) -> how many such pre-existing nodes
        self._links: Dict[tuple, tuple] = {}

    def upsert_node(self, tenant, ref, props=None):
        self._nodes.setdefault((tenant, ref), {}).update(props or {})

    def add_edge(self, edge):
        validate_edge(edge)
        for ref in (edge.src, edge.dst):
            self._nodes.setdefault((edge.tenant, ref), {})
        stored = Edge(edge.tenant, edge.src, edge.type, edge.dst, dict(edge.props), id=str(next(self._ids)))
        self._edges[stored.id] = stored
        return stored

    def edges(self, tenant, *, src=None, type=None, dst=None, source=None, as_of=None, include_closed=False):
        out = [e for e in self._edges.values() if e.tenant == tenant and _match(e, src, type, dst, source)]
        if as_of is not None:
            out = [e for e in out if e.current_at(utc(as_of))]
        elif not include_closed:
            from datetime import datetime as _dt, timezone
            now = _dt.now(timezone.utc)
            out = [e for e in out if e.current_at(now)]
        return sorted(out, key=lambda e: int(e.id))

    def close_edge(self, tenant, edge_id, at):
        e = self._edges[edge_id]
        assert e.tenant == tenant
        e.props["valid_to"] = max(at, e.props["valid_from"])

    def delete_edges(self, tenant, *, source=None, type=None, src=None, dst=None):
        ids = [i for i, e in self._edges.items() if e.tenant == tenant and _match(e, src, type, dst, source)]
        for i in ids:
            del self._edges[i]
        return len(ids)

    def node(self, tenant, ref):
        return self._nodes.get((tenant, ref))

    def rename_node(self, tenant, ref, new_key, drop_props=()):
        props = self._nodes.pop((tenant, ref), None)
        if props is None:
            return False
        for k in drop_props:
            props.pop(k, None)
        new = NodeRef(ref.label, new_key)
        self._nodes[(tenant, new)] = props
        for e in self._edges.values():
            if e.tenant == tenant:
                if e.src == ref:
                    e.src = new
                if e.dst == ref:
                    e.dst = new
        return True

    def entity_keys(self, tenant, prefix):
        return sorted(r.key for (t, r) in self._nodes
                      if t == tenant and r.label == "Entity" and r.key.startswith(prefix) and (t, r) not in self._links)

    def link_legacy(self, tenant, ref, label, prop, value):
        if label not in LEGACY_PROPS or LEGACY_PROPS[label] != prop:
            raise AdmissionError(f"{label}.{prop} is not a linkable legacy key")
        n = self.legacy.get((label, prop, value), 0)
        if n != 1:
            return "missing" if n == 0 else "ambiguous"
        self._links[(tenant, ref)] = (label, prop, value)
        return "linked"

    def linked_to(self, tenant, ref):
        return self._links.get((tenant, ref))


_DT_KEYS = ("valid_from", "valid_to", "decided_at", "computed_at")


def _to_neo(props: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in props.items():
        if v is None:
            continue
        out[k] = utc(v).isoformat(timespec="microseconds") if isinstance(v, datetime) else (list(v) if isinstance(v, tuple) else v)
    return out


def _from_neo(props: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in props.items() if k != "pw_tenant"}
    for k in _DT_KEYS:
        if isinstance(out.get(k), str):
            out[k] = datetime.fromisoformat(out[k])
    out.setdefault("valid_to", None)
    return out


class Neo4jGraph:
    """Relationship types and labels are interpolated into Cypher only after checking them against the
    closed vocabularies in model.py; every VALUE is a query parameter."""

    def __init__(self, driver, database: Optional[str] = None) -> None:
        self._driver, self._db = driver, database

    def _run(self, cypher: str, **params):
        with self._driver.session(database=self._db) as s:
            return [r for r in s.run(cypher, **params)]

    @staticmethod
    def _lbl(ref: NodeRef) -> str:
        assert ref.label in NODE_LABELS
        return ref.label

    def upsert_node(self, tenant, ref, props=None):
        self._run(f"MERGE (n:{self._lbl(ref)} {{pw_tenant:$t, pw_key:$k}}) SET n += $p",
                  t=tenant, k=ref.key, p=_to_neo(props or {}))

    def add_edge(self, edge):
        validate_edge(edge)
        assert edge.type in EDGE_RULES
        for ref in (edge.src, edge.dst):
            self.upsert_node(edge.tenant, ref)
        rows = self._run(
            f"MATCH (a:{self._lbl(edge.src)} {{pw_tenant:$t, pw_key:$a}}), (b:{self._lbl(edge.dst)} {{pw_tenant:$t, pw_key:$b}}) "
            f"CREATE (a)-[r:{edge.type}]->(b) SET r = $p, r.pw_tenant = $t RETURN elementId(r) AS id",
            t=edge.tenant, a=edge.src.key, b=edge.dst.key, p=_to_neo(edge.props))
        return Edge(edge.tenant, edge.src, edge.type, edge.dst, dict(edge.props), id=rows[0]["id"])

    def edges(self, tenant, *, src=None, type=None, dst=None, source=None, as_of=None, include_closed=False):
        if type is not None and type not in EDGE_RULES:
            raise AdmissionError(f"unknown edge type {type!r}")
        a = f":{self._lbl(src)}" if src else ""
        b = f":{self._lbl(dst)}" if dst else ""
        rel = f":{type}" if type else ""
        where = ["r.pw_tenant = $t"]
        params: Dict[str, Any] = {"t": tenant}
        if src:
            where.append("a.pw_key = $a"); params["a"] = src.key
        if dst:
            where.append("b.pw_key = $b"); params["b"] = dst.key
        if source:
            where.append("r.source = $s"); params["s"] = source
        rows = self._run(f"MATCH (a{a} {{pw_tenant:$t}})-[r{rel}]->(b{b} {{pw_tenant:$t}}) WHERE {' AND '.join(where)} "
                         "RETURN elementId(r) AS id, type(r) AS type, labels(a) AS al, a.pw_key AS ak, labels(b) AS bl, "
                         "b.pw_key AS bk, properties(r) AS p", **params)
        out = []
        for r in rows:
            al = next(l for l in r["al"] if l in NODE_LABELS)
            bl = next(l for l in r["bl"] if l in NODE_LABELS)
            out.append(Edge(tenant, NodeRef(al, r["ak"]), r["type"], NodeRef(bl, r["bk"]), _from_neo(dict(r["p"])), id=r["id"]))
        if as_of is not None:
            out = [e for e in out if e.current_at(utc(as_of))]
        elif not include_closed:
            from datetime import datetime as _dt, timezone
            now = _dt.now(timezone.utc)
            out = [e for e in out if e.current_at(now)]
        return sorted(out, key=lambda e: (e.valid_from, e.id))

    def close_edge(self, tenant, edge_id, at):
        self._run("MATCH ()-[r]->() WHERE elementId(r) = $id AND r.pw_tenant = $t "
                  "SET r.valid_to = CASE WHEN r.valid_from > $at THEN r.valid_from ELSE $at END",
                  id=edge_id, t=tenant, at=utc(at).isoformat(timespec="microseconds"))

    def delete_edges(self, tenant, *, source=None, type=None, src=None, dst=None):
        rel = f":{type}" if type else ""
        if type is not None and type not in EDGE_RULES:
            raise AdmissionError(f"unknown edge type {type!r}")
        where, params = ["r.pw_tenant = $t"], {"t": tenant}
        if source:
            where.append("r.source = $s"); params["s"] = source
        if src:
            where.append("a.pw_key = $a"); params["a"] = src.key
        if dst:
            where.append("b.pw_key = $b"); params["b"] = dst.key
        a = f":{self._lbl(src)}" if src else ""
        b = f":{self._lbl(dst)}" if dst else ""
        rows = self._run(f"MATCH (a{a} {{pw_tenant:$t}})-[r{rel}]->(b{b} {{pw_tenant:$t}}) WHERE {' AND '.join(where)} "
                         "WITH r, count(r) AS c DELETE r RETURN count(*) AS n", **params)
        return sum(r["n"] for r in rows)

    def node(self, tenant, ref):
        rows = self._run(f"MATCH (n:{self._lbl(ref)} {{pw_tenant:$t, pw_key:$k}}) RETURN properties(n) AS p", t=tenant, k=ref.key)
        return dict(rows[0]["p"]) if rows else None

    def rename_node(self, tenant, ref, new_key, drop_props=()):
        drops = "".join(f", n.{p} = null" for p in drop_props if p.isidentifier())
        rows = self._run(f"MATCH (n:{self._lbl(ref)} {{pw_tenant:$t, pw_key:$k}}) SET n.pw_key = $new{drops} RETURN count(n) AS c",
                         t=tenant, k=ref.key, new=new_key)
        return bool(rows and rows[0]["c"])

    def entity_keys(self, tenant, prefix):
        rows = self._run("MATCH (e:Entity {pw_tenant:$t}) WHERE e.pw_key STARTS WITH $p AND NOT (e)-[:REPRESENTS]->() "
                         "RETURN e.pw_key AS k ORDER BY k", t=tenant, p=prefix)
        return [r["k"] for r in rows]

    def link_legacy(self, tenant, ref, label, prop, value):
        if LEGACY_PROPS.get(label) != prop:             # label/prop reach Cypher text only from this closed map
            raise AdmissionError(f"{label}.{prop} is not a linkable legacy key")
        n = self._run(f"MATCH (l:{label}) WHERE l.{prop} = $v RETURN count(l) AS c", v=value)[0]["c"]
        if n != 1:
            return "missing" if n == 0 else "ambiguous"
        rows = self._run(f"MATCH (e:Entity {{pw_tenant:$t, pw_key:$k}}), (l:{label}) WHERE l.{prop} = $v "
                         "MERGE (e)-[r:REPRESENTS]->(l) RETURN count(r) AS c", t=tenant, k=ref.key, v=value)
        return "linked" if rows and rows[0]["c"] else "missing"
