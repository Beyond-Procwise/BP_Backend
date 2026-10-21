"""Link Entity stubs to the nodes the rest of the app already keeps in the graph.

A stub ``supplier:SUP-X`` stands for the person-facing "thing people act on"; the older ingestion code
keeps ``(:Supplier {supplier_id})`` and ``(:PurchaseOrder {po_id})``. This adds ``(stub)-[:REPRESENTS]->(old)``
so a walk can go from who-acted-on-it to what-it-is (its invoices, quotes, contract).

Rules: the older node is only ever MATCHED, never created or changed; a stub links only when exactly one older
node carries that id (zero = not in the graph yet, several = ambiguous: both are reported and left alone);
the link has no pw_tenant and the older node has none, so the asserted-edge queries never see it.
Idempotent; re-run by the daily refresh so stubs created later pick their node up once it exists.
"""
from __future__ import annotations

import logging
from typing import Dict

from services.people.graph.model import NodeRef
from services.people.graph.store import GraphStore

logger = logging.getLogger(__name__)

# stub type prefix -> (older label, older key property)
LINKS = {"supplier": ("Supplier", "supplier_id"), "po": ("PurchaseOrder", "po_id")}


def link_all(store: GraphStore, tenant: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for kind, (label, prop) in LINKS.items():
        counts = {"linked": 0, "missing": 0, "ambiguous": 0}
        prefix = f"{kind}:"
        for key in store.entity_keys(tenant, prefix):
            counts[store.link_legacy(tenant, NodeRef("Entity", key), label, prop, key[len(prefix):])] += 1
        out[kind] = counts
    return out
