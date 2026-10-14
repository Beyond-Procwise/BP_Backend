"""Run the whole resolution pass against the live database and graph.

    ./.venv/bin/python -m scripts.graph_resolution.run_pass [--limit N]

The runtime venv is the one with the neo4j driver; ./venv (the test venv) does
not have it. Prints the per-stage counts and the review backlog as JSON, so a
run that produces nothing says so in numbers rather than in silence.
"""
from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the rows each stage reads (default: no cap)")
    args = ap.parse_args()

    load_dotenv(".env")
    import psycopg2
    from neo4j import GraphDatabase

    from src.services.graph_resolution.pass_runner import run_all

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"), dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
    )
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    try:
        print(json.dumps(run_all(conn, driver, args.limit), indent=2, default=str))
    finally:
        driver.close()
        conn.close()


if __name__ == "__main__":
    main()
