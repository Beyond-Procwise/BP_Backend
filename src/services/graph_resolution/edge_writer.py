"""The only place a derived edge reaches Neo4j.

One writer is what makes the redaction rule enforceable: a bank account is
Tier-1 identity evidence and must never sit in an edge property in cleartext.
The score survives; the value does not.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, List, Optional

log = logging.getLogger(__name__)

#: Signals whose raw `value` is never persisted. query_engine already excludes
#: banking detail from the default supplier projection; that exclusion holds here.
REDACTED_SIGNALS = frozenset({"bank_account", "bank_iban", "bank_swift"})

#: Profiles whose parameters are declared unmeasured (spec section 9). Their
#: edges never reach auto_link, whatever F says, until a labelled sample exists.
UNCALIBRATED_PROFILES = frozenset({"contract_coverage", "contract_succession"})


@dataclass(frozen=True)
class DerivedEdge:
    rel_type: str
    from_label: str
    from_key: str
    from_value: str
    to_label: str
    to_key: str
    to_value: str
    F: float
    band: str
    P_raw: float
    L_evidence: float
    profile: str
    profile_version: str
    signals: List[dict]
    observations: str
    resolution: Optional[str] = None
    margin: Optional[float] = None


def redact_signals(signals: List[dict]) -> List[dict]:
    out = []
    for s in signals:
        if s.get("id") in REDACTED_SIGNALS and "value" in s:
            s = {**s, "value": "[redacted]"}
        out.append(s)
    return out


def cypher_for(edge: DerivedEdge) -> tuple[str, dict]:
    if edge.profile in UNCALIBRATED_PROFILES and edge.band == "auto_link":
        raise ValueError(
            f"{edge.profile} is capped at review until calibrated "
            f"(spec section 9); refusing to write band=auto_link"
        )
    props = {
        "F": edge.F, "band": edge.band, "P_raw": edge.P_raw,
        "L_evidence": edge.L_evidence, "profile": edge.profile,
        "profile_version": edge.profile_version,
        "signals": json.dumps(redact_signals(edge.signals)),
        "observations": edge.observations,
        "resolution": edge.resolution, "margin": edge.margin,
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
    and _trgt remains the source of truth for documents."""
    written = 0
    try:
        with driver.session() as session:
            for edge in edges:
                q, params = cypher_for(edge)
                result = session.run(q, **params)
                written += (result.single() or {}).get("cnt", 0)
    except Exception as exc:  # noqa: BLE001
        log.warning("edge_writer: %d/%d written before failure: %s",
                    written, len(edges), exc)
    return written
