"""The only place a derived edge reaches Neo4j.

One writer is what makes the redaction rule enforceable: a bank account is
Tier-1 identity evidence and must never sit in an edge property in cleartext.
The score survives; the value does not.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, List, Optional

log = logging.getLogger(__name__)

#: Signals whose raw `value` is never persisted. query_engine already excludes
#: banking detail from the default supplier projection; that exclusion holds here.
REDACTED_SIGNALS = frozenset({"bank_account", "bank_iban", "bank_swift"})

#: Profiles whose parameters are declared unmeasured (spec section 9). Their
#: edges never reach auto_link, whatever F says, until a labelled sample exists.
UNCALIBRATED_PROFILES = frozenset({
    "contract_coverage", "contract_succession", "supplier_identity",
    "item_equivalence",
})

#: Cypher structural identifiers (labels, property keys, relationship types) are
#: interpolated into the query text -- they cannot be bound as parameters. This
#: module is the only writer specifically so that this validation happens in one
#: place, rather than trusting every future caller to only ever pass hardcoded
#: strings.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _validate_identifier(field_name: str, value: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"invalid Cypher identifier for {field_name}: {value!r} "
            f"(refusing to interpolate into query text)"
        )


@dataclass(frozen=True)
class DerivedEdge:
    rel_type: str
    from_label: str
    from_key: str
    from_value: str
    to_label: str
    to_key: str
    to_value: str
    # Optional, not merely nullable: a membership can arise without ever being
    # scored (item_key's exact-match consolidation -- see `basis` below). F,
    # band, P_raw, L_evidence and signals are a *measurement*; None means no
    # measurement was taken, and cypher_for must write that as an absent
    # property, never a fabricated 0.0/"[]" that would misrepresent a real
    # (if low) score.
    F: Optional[float]
    band: Optional[str]
    P_raw: Optional[float]
    L_evidence: Optional[float]
    profile: str
    profile_version: str
    signals: Optional[List[dict]]
    observations: str
    resolution: Optional[str] = None
    margin: Optional[float] = None
    #: How this edge's membership was established. "scored" -- the profile's
    #: pairwise scorer linked the two lines. "exact_item_id" -- the class's
    #: item_key matched this line's own item_id exactly, independent of any
    #: score (deterministic, not inferred). None for edge kinds that predate
    #: this distinction (e.g. SAME_ENTITY), which are always scored.
    basis: Optional[str] = None


def redact_signals(signals: Optional[List[dict]]) -> Optional[List[dict]]:
    if signals is None:
        return None
    out = []
    for s in signals:
        if s.get("id") in REDACTED_SIGNALS and "value" in s:
            s = {**s, "value": "[redacted]"}
        out.append(s)
    return out


def cypher_for(edge: DerivedEdge) -> tuple[str, dict]:
    for field_name, value in (
        ("rel_type", edge.rel_type),
        ("from_label", edge.from_label),
        ("from_key", edge.from_key),
        ("to_label", edge.to_label),
        ("to_key", edge.to_key),
    ):
        _validate_identifier(field_name, value)
    if edge.profile in UNCALIBRATED_PROFILES and edge.band == "auto_link":
        raise ValueError(
            f"{edge.profile} is capped at review until calibrated "
            f"(spec section 9); refusing to write band=auto_link"
        )
    props = {
        # F/band/P_raw/L_evidence/signals stay explicit None, never a
        # fabricated default, when the edge carries no measurement -- Cypher's
        # `SET r += $props` removes a property whose value is null, so a
        # rewrite genuinely clears a stale score rather than leaving one
        # stranded from an earlier run.
        "F": edge.F, "band": edge.band, "P_raw": edge.P_raw,
        "L_evidence": edge.L_evidence, "profile": edge.profile,
        "profile_version": edge.profile_version,
        "signals": (json.dumps(redact_signals(edge.signals))
                    if edge.signals is not None else None),
        "observations": edge.observations,
        "resolution": edge.resolution, "margin": edge.margin,
        "basis": edge.basis,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    q = (
        f"MATCH (a:{edge.from_label} {{{edge.from_key}: $from_value}}) "
        f"MATCH (b:{edge.to_label} {{{edge.to_key}: $to_value}}) "
        f"MERGE (a)-[r:{edge.rel_type}]->(b) "
        f"SET r += $props "
        f"RETURN count(r) AS cnt"
    )
    return q, {"from_value": edge.from_value, "to_value": edge.to_value,
               "props": props}


def write_edges(driver: Any, edges: List[DerivedEdge]) -> int:
    """Write derived edges. Never raises: the graph is a downstream side effect
    and _trgt remains the source of truth for documents.

    A `ValueError` out of `cypher_for` (the uncalibrated-profile guard, or the
    structural-identifier validation) is a *refusal*, not a fault: it is
    logged distinctly at ERROR and that one edge is skipped, but the rest of
    the batch still gets written. Anything else (a driver/session error, a
    Neo4j connectivity failure) is a transport-level fault, logged at
    WARNING, and aborts whatever remains of the batch -- it says nothing
    about whether the remaining edges themselves are safe to write.
    """
    written = 0
    try:
        with driver.session() as session:
            for edge in edges:
                try:
                    q, params = cypher_for(edge)
                except ValueError as exc:
                    log.error(
                        "edge_writer: refusing edge %s(%s)->%s(%s): %s",
                        edge.from_label, edge.from_value,
                        edge.to_label, edge.to_value, exc,
                    )
                    continue
                result = session.run(q, **params)
                written += (result.single() or {}).get("cnt", 0)
    except Exception as exc:  # noqa: BLE001
        log.warning("edge_writer: %d/%d written before transport failure: %s",
                    written, len(edges), exc)
    return written
