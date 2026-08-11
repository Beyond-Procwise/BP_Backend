#!/usr/bin/env python
"""Merge duplicate KG nodes onto one survivor, then make duplication impossible.

``_create_indexes`` issues ``CREATE INDEX``, not ``CREATE CONSTRAINT ... IS
UNIQUE``. An index makes MERGE fast; it does not make it safe. The graph
accumulated nodes sharing a primary key — measured 2026-08-11: Invoice 2,
PurchaseOrder 1, Quote 16, with ORB-Q-6612 present seven times. The source
tables are clean (bp_quote_trgt holds 21,049 rows and 21,049 distinct
quote_ids), so this is a graph artefact, not bad data.

Two steps, in this order, because Neo4j REFUSES a uniqueness constraint while
duplicates exist:

  1. dedupe — pick one survivor per key, move the others' relationships onto it,
     delete the others
  2. constrain — replace the per-label indexes with uniqueness constraints, so
     a duplicate can no longer be created

Survivor choice: the node with the most relationships, tie-broken by the lowest
element id so a re-run is deterministic. Relationships are re-pointed with MERGE
so an edge the survivor already has is not duplicated.

APOC is not installed on this instance (checked), so ``apoc.refactor.mergeNodes``
is unavailable and the re-pointing is done explicitly. Relationship types are
read from the database and validated against ``[A-Z_]+`` before being
interpolated — they cannot be interpolated as parameters in Cypher.

Usage:
    set -a && . ./.env && set +a
    ./.venv/bin/python scripts/kg_dedupe_and_constrain.py --dry-run
    ./.venv/bin/python scripts/kg_dedupe_and_constrain.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("kg_dedupe")
logging.getLogger("neo4j").setLevel(logging.ERROR)

# (label, primary key) — mirrors ENTITY_TABLE_MAP, imported so the two cannot
# drift apart.
def _labels():
    from src.services.procurement_kg_builder import ENTITY_TABLE_MAP
    return [(label, pk) for _e, (_t, pk, label) in ENTITY_TABLE_MAP.items()]


_REL_TYPE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _duplicate_keys(session, label: str, pk: str) -> list:
    return [
        r["k"] for r in session.run(
            f"MATCH (n:{label}) WHERE n.{pk} IS NOT NULL "
            f"WITH n.{pk} AS k, count(*) AS c WHERE c > 1 RETURN k",
        )
    ]


def _merge_one_key(session, label: str, pk: str, key) -> int:
    """Collapse every node with this key onto one survivor. Returns nodes removed."""
    rows = list(session.run(
        f"MATCH (n:{label} {{{pk}: $k}}) "
        "OPTIONAL MATCH (n)-[r]-() "
        "WITH n, count(r) AS rels "
        "RETURN elementId(n) AS eid, rels ORDER BY rels DESC, eid ASC",
        k=key,
    ))
    if len(rows) < 2:
        return 0

    keep, drop = rows[0]["eid"], [r["eid"] for r in rows[1:]]

    for eid in drop:
        # Outgoing, then incoming. Types come from the database; validated
        # before interpolation because Cypher cannot parameterise a type.
        for direction in ("out", "in"):
            pattern = "(d)-[r]->(o)" if direction == "out" else "(o)-[r]->(d)"
            types = [
                t["t"] for t in session.run(
                    f"MATCH (d) WHERE elementId(d) = $eid "
                    f"MATCH {pattern} RETURN DISTINCT type(r) AS t",
                    eid=eid,
                )
            ]
            for rel in types:
                if not _REL_TYPE.match(rel):
                    log.warning("skipping unexpected relationship type %r", rel)
                    continue
                new = (f"MERGE (k)-[:{rel}]->(o)" if direction == "out"
                       else "MERGE (o)-[:%s]->(k)" % rel)
                old = (f"MATCH (d)-[r:{rel}]->(o)" if direction == "out"
                       else f"MATCH (o)-[r:{rel}]->(d)")
                session.run(
                    "MATCH (d) WHERE elementId(d) = $eid "
                    "MATCH (k) WHERE elementId(k) = $keep "
                    f"{old} {new} DELETE r",
                    eid=eid, keep=keep,
                )
        session.run("MATCH (d) WHERE elementId(d) = $eid DETACH DELETE d", eid=eid)

    return len(drop)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    removed_total = 0
    with driver.session() as s:
        plan = {}
        for label, pk in _labels():
            keys = _duplicate_keys(s, label, pk)
            if keys:
                n = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                dis = s.run(
                    f"MATCH (n:{label}) RETURN count(DISTINCT n.{pk}) AS c"
                ).single()["c"]
                plan[label] = {"pk": pk, "keys": keys, "extra": n - dis}
                log.info("%-16s %d duplicated key(s), %d extra node(s)",
                         label, len(keys), n - dis)
        if not plan:
            log.info("no duplicates")
        if args.dry_run:
            for label, info in plan.items():
                log.info("  %s would collapse: %s", label, info["keys"][:10])
            log.info("dry run — nothing changed")
            driver.close()
            return 0

        if plan:
            stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
            out = Path("backups/neo4j") / f"duplicate_nodes_{stamp}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            snapshot = {}
            for label, info in plan.items():
                snapshot[label] = [
                    dict(r["n"]) for r in s.run(
                        f"MATCH (n:{label}) WHERE n.{info['pk']} IN $keys RETURN n",
                        keys=info["keys"],
                    )
                ]
            out.write_text(json.dumps(snapshot, default=str))
            log.info("backed up the affected nodes -> %s", out)

            for label, info in plan.items():
                for key in info["keys"]:
                    removed_total += _merge_one_key(s, label, info["pk"], key)
            log.info("removed %d duplicate node(s)", removed_total)

        # --- constraints -------------------------------------------------
        #
        # A plain index on the same property BLOCKS the constraint: Neo4j
        # answers "There already exists an index (:Label {pk}). A constraint
        # cannot be created..." — the constraint brings its own index, so the
        # old one has to go first. That is why _create_indexes must also stop
        # creating these; see procurement_kg_builder.
        log.info("applying uniqueness constraints")
        existing = {
            (tuple(r["labelsOrTypes"] or []), tuple(r["properties"] or [])): r["name"]
            for r in s.run(
                "SHOW INDEXES YIELD name, labelsOrTypes, properties, owningConstraint "
                "WHERE owningConstraint IS NULL RETURN name, labelsOrTypes, properties"
            )
        }
        for label, pk in _labels():
            name = f"uniq_{label.lower()}_{pk}"
            stale = existing.get(((label,), (pk,)))
            try:
                if stale:
                    s.run(f"DROP INDEX {stale} IF EXISTS")
                    log.info("   %-16s dropped plain index %s", label, stale)
                s.run(
                    f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{pk} IS UNIQUE"
                )
                log.info("   %-16s constraint applied", label)
            except Exception as exc:
                log.error("   %-16s FAILED: %s", label, str(exc)[:160])

        log.info("verifying")
        ok = True
        for label, pk in _labels():
            n = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            dis = s.run(
                f"MATCH (n:{label}) RETURN count(DISTINCT n.{pk}) AS c"
            ).single()["c"]
            if n != dis:
                log.error("   %-16s STILL DUPLICATED: %d nodes, %d keys",
                          label, n, dis)
                ok = False
        if ok:
            log.info("   every label has one node per key")

    driver.close()
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
