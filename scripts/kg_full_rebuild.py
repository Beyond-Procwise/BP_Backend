#!/usr/bin/env python
"""Rebuild the procurement knowledge graph from the _trgt tier, supervised.

The scheduled job (backend_scheduler._run_kg_sync) is gated OFF by
KG_FULL_REBUILD_ENABLED because a full rebuild is a mass write, not an
increment: roughly 232,000 nodes on the current corpus. This runs the same
builder deliberately, with progress and before/after counts, so the first one
is watched rather than discovered.

Idempotent — every write is a Neo4j MERGE on the primary key, so re-running
updates nodes rather than duplicating them. Safe to interrupt and resume.

Usage:
    set -a && . ./.env && set +a
    ./.venv/bin/python scripts/kg_full_rebuild.py            # entities + relationships
    ./.venv/bin/python scripts/kg_full_rebuild.py --entities # nodes only
    ./.venv/bin/python scripts/kg_full_rebuild.py --dry-run  # report, write nothing

Note this MUST run on .venv, not venv: the neo4j driver is installed in the
runtime environment only.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kg_rebuild")
for noisy in ("neo4j", "urllib3", "botocore", "boto3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _agent_nick():
    """Minimal stand-in carrying the two things the builder actually uses."""
    import psycopg2
    from config.settings import settings

    class _N:
        def __init__(self) -> None:
            self.settings = settings

        def get_db_connection(self):
            return psycopg2.connect(
                host=os.environ["DB_HOST"], dbname=os.environ["DB_NAME"],
                user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"],
                port=os.environ.get("DB_PORT", "5432"),
            )

    return _N()


def _graph_counts(driver) -> dict[str, int]:
    labels = ("Supplier", "Contract", "Invoice", "InvoiceLine",
              "PurchaseOrder", "POLine", "Quote", "QuoteLine", "Policy")
    out = {}
    with driver.session() as s:
        for lbl in labels:
            out[lbl] = s.run(
                f"MATCH (n:{lbl}) RETURN count(n) AS c"
            ).single()["c"]
        out["_total"] = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    return out


def _source_counts(agent_nick, table_map) -> dict[str, int]:
    out = {}
    conn = agent_nick.get_db_connection()
    try:
        with conn.cursor() as cur:
            for entity, (table, _pk, _lbl) in table_map.items():
                try:
                    cur.execute(f"SELECT count(*) FROM {table}")
                    out[entity] = cur.fetchone()[0]
                except Exception:
                    conn.rollback()
                    out[entity] = -1     # -1 means the table is unreadable
    finally:
        conn.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", action="store_true",
                    help="load nodes only, skip relationship building")
    ap.add_argument("--dry-run", action="store_true",
                    help="report source vs graph counts and exit")
    args = ap.parse_args()

    from src.services.procurement_kg_builder import (
        ENTITY_TABLE_MAP, ProcurementKGBuilder,
    )

    nick = _agent_nick()
    builder = ProcurementKGBuilder(nick)
    if not builder._driver:
        log.error("Neo4j is not reachable — nothing done")
        return 2

    src = _source_counts(nick, ENTITY_TABLE_MAP)
    before = _graph_counts(builder._driver)

    log.info("%-16s %10s %10s", "entity", "source", "graph")
    for entity, (table, _pk, label) in ENTITY_TABLE_MAP.items():
        n = src.get(entity, -1)
        log.info("%-16s %10s %10s%s", entity,
                 "UNREADABLE" if n < 0 else n, before.get(label, 0),
                 "   <-- source unreadable" if n < 0 else "")
    log.info("graph total before: %d", before["_total"])

    unreadable = [e for e, n in src.items() if n < 0]
    if unreadable:
        log.error("these source tables cannot be read: %s", unreadable)
        log.error("fix ENTITY_TABLE_MAP before rebuilding; refusing to run")
        return 3

    if args.dry_run:
        log.info("dry run — nothing written")
        builder.close()
        return 0

    started = time.time()
    if args.entities:
        counts = {}
        for entity, (table, pk, label) in ENTITY_TABLE_MAP.items():
            t0 = time.time()
            counts[entity] = builder._load_entity(table, pk, label)
            log.info("  %-16s %7d nodes in %5.1fs", entity, counts[entity],
                     time.time() - t0)
    else:
        counts = builder.build_full_graph()

    after = _graph_counts(builder._driver)
    builder.close()

    log.info("--- done in %.1f minutes ---", (time.time() - started) / 60)
    log.info("%-16s %10s %10s %10s", "label", "before", "after", "delta")
    for lbl in sorted(set(before) | set(after)):
        if lbl.startswith("_"):
            continue
        b, a = before.get(lbl, 0), after.get(lbl, 0)
        log.info("%-16s %10d %10d %+10d", lbl, b, a, a - b)
    log.info("graph total: %d -> %d", before["_total"], after["_total"])

    # The check the scheduled job now makes too: a rebuild that loaded no
    # documents is a failure however cheerfully it finished.
    docs = ("Invoice", "InvoiceLine", "PurchaseOrder", "POLine",
            "Quote", "QuoteLine")
    if not any(after.get(d, 0) for d in docs):
        log.error("no document nodes in the graph after a full rebuild")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
