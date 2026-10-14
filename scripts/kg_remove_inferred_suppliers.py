#!/usr/bin/env python
"""Remove Supplier nodes the graph builder invented, rather than read.

``ProcurementKGBuilder._infer_suppliers`` was documented as running "if
bp_supplier is empty" and the check was never written, so it ran on every
rebuild against a populated supplier master. Its PO branch is the damaging one:

    MERGE (s:Supplier {supplier_name: p.supplier_name})
    ON CREATE SET s.supplier_id = p.supplier_name

That sets a supplier_id to a company NAME. Such a node can never join to
proc.bp_supplier, and it inflates every supplier figure taken from the graph.
The invoice/quote/contract branches do the mirror image, setting supplier_name
to the supplier_id, so a supplier ends up named "SUP-Northgate".

The guard is now in place, so this does not recur — but the nodes already
written have to come out.

WHAT IDENTIFIES THEM: ``ON CREATE SET s.source = 'inferred_from_*'``. Because it
is ON CREATE, a MERGE that matched a real supplier node left ``source`` unset.
So the property is an exact discriminator: every node carrying it was invented
by this code, and no node read from proc.bp_supplier carries it. That is a far
safer test than guessing from the shape of an id.

The relationships pointing at these nodes go with them. Nothing is lost that the
graph does not still hold: the Invoice/PO/Quote nodes keep their own
``supplier_id`` / ``supplier_name`` properties, so the claim each document made
survives — only the fabricated node and edge disappear.

Usage:
    set -a && . ./.env && set +a
    ./.venv/bin/python scripts/kg_remove_inferred_suppliers.py --dry-run
    ./.venv/bin/python scripts/kg_remove_inferred_suppliers.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("kg_cleanup")
logging.getLogger("neo4j").setLevel(logging.ERROR)

_INFERRED = "MATCH (s:Supplier) WHERE s.source STARTS WITH 'inferred_from'"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    with driver.session() as s:
        total = s.run("MATCH (n:Supplier) RETURN count(n) AS c").single()["c"]
        log.info("Supplier nodes in the graph: %d", total)

        log.info("by source:")
        for r in s.run(
            "MATCH (n:Supplier) RETURN coalesce(n.source,'(read from bp_supplier)') "
            "AS src, count(*) AS n ORDER BY n DESC"
        ):
            log.info("   %-34s %6d", r["src"], r["n"])

        doomed = s.run(f"{_INFERRED} RETURN count(s) AS c").single()["c"]
        rels = s.run(
            f"{_INFERRED} MATCH (s)-[r]-() RETURN count(r) AS c"
        ).single()["c"]
        log.info("fabricated: %d node(s), %d relationship(s)", doomed, rels)

        if not doomed:
            log.info("nothing to remove")
            driver.close()
            return 0

        # Anything carrying an inferred marker that ALSO exists in the supplier
        # master would be a real supplier wrongly labelled. Report it and refuse
        # rather than delete something that turns out to be real.
        overlap = s.run(
            f"{_INFERRED} RETURN s.supplier_id AS sid LIMIT 100000"
        )
        ids = {r["sid"] for r in overlap if r["sid"]}

        import psycopg2
        conn = psycopg2.connect(
            host=os.environ["DB_HOST"], dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"],
            port=os.environ.get("DB_PORT", "5432"),
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT supplier_id FROM proc.bp_supplier "
                    "WHERE supplier_id = ANY(%s)", (list(ids),)
                )
                real = {r[0] for r in cur.fetchall()}
        finally:
            conn.close()

        if real:
            log.error(
                "%d node(s) marked inferred also exist in proc.bp_supplier: %s",
                len(real), sorted(real)[:10],
            )
            log.error("refusing to delete — these may be real suppliers")
            driver.close()
            return 3
        log.info("none of them exist in proc.bp_supplier — safe to remove")

        if args.dry_run:
            log.info("dry run — nothing deleted")
            driver.close()
            return 0

        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
        out = Path("backups/neo4j") / f"inferred_suppliers_{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = [dict(r["s"]) for r in s.run(f"{_INFERRED} RETURN s")]
        out.write_text(json.dumps(rows, default=str))
        log.info("backed up %d node(s) -> %s", len(rows), out)

        removed = 0
        while True:
            n = s.run(
                f"{_INFERRED} WITH s LIMIT 1000 DETACH DELETE s "
                "RETURN count(*) AS n"
            ).single()["n"]
            removed += n
            if n == 0:
                break
        after = s.run("MATCH (n:Supplier) RETURN count(n) AS c").single()["c"]
        log.info("removed %d; Supplier %d -> %d", removed, total, after)

    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
