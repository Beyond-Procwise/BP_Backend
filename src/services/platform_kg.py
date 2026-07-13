"""The platform's model of itself, in the knowledge graph.

Neo4j already held ~10k relationships about the DATA — suppliers, invoices, POs, quotes,
line items. It held nothing about the SYSTEM: no process, no stage, no agent, no screen.
So AgentNick could reason about a supplier and not about how a document reaches that
supplier, which agent decides what, or where a buyer would see the answer.

This loads `resources/knowledge/platform_ontology.yaml` into the same graph, so the two
halves connect: an Agent WRITES_TO a table, a Screen READS the same table, a Stage
PRODUCES it. Ask "where does a buyer see a promoted quote?" and the graph can answer by
walking edges instead of by someone remembering.

Why the graph and not the system prompt: the platform description is a few thousand
tokens. In a SYSTEM block that is charged on every single call and eats a third of an
8k context — it would slow every extraction to teach the model something extraction does
not need. Retrieved on demand it costs nothing until it is wanted.

Idempotent: MERGE on id, so re-running updates in place rather than duplicating.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)

ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "knowledge" / "platform_ontology.yaml"
)


def _driver():
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not uri:
        raise RuntimeError("NEO4J_URI is not configured")
    return GraphDatabase.driver(uri, auth=(user, pwd))


def load_ontology(path: Optional[Path] = None) -> dict[str, Any]:
    with open(path or ONTOLOGY_PATH) as fh:
        return yaml.safe_load(fh) or {}


def sync(path: Optional[Path] = None) -> dict[str, int]:
    """Push the platform ontology into Neo4j. Returns node/edge counts written."""
    doc = load_ontology(path)
    counts = {"Process": 0, "Stage": 0, "Agent": 0, "Screen": 0, "Model": 0, "Gap": 0, "edges": 0}
    drv = _driver()
    try:
        with drv.session() as s:
            for p in doc.get("processes") or []:
                s.run(
                    "MERGE (p:Process {id:$id}) "
                    "SET p.name=$name, p.summary=$summary, p.component=$component",
                    id=p["id"], name=p.get("name"), summary=p.get("summary"),
                    component=p.get("component"),
                )
                counts["Process"] += 1
                prev = None
                for st in p.get("stages") or []:
                    s.run(
                        "MERGE (x:Stage {id:$id}) "
                        "SET x.name=$name, x.detail=$detail, x.component=$component "
                        "WITH x MATCH (p:Process {id:$pid}) MERGE (p)-[:HAS_STAGE]->(x)",
                        id=f"{p['id']}::{st['id']}", name=st.get("name"),
                        detail=st.get("detail"), component=st.get("component"), pid=p["id"],
                    )
                    counts["Stage"] += 1
                    counts["edges"] += 1
                    if prev:
                        # Stage order is knowledge in itself: "what happens next" is a walk.
                        s.run(
                            "MATCH (a:Stage {id:$a}), (b:Stage {id:$b}) MERGE (a)-[:NEXT]->(b)",
                            a=prev, b=f"{p['id']}::{st['id']}",
                        )
                        counts["edges"] += 1
                    prev = f"{p['id']}::{st['id']}"

            for a in doc.get("agents") or []:
                s.run(
                    "MERGE (n:Agent {id:$id}) SET n.name=$name, n.does=$does",
                    id=a["id"], name=a.get("name"), does=a.get("does"),
                )
                counts["Agent"] += 1
                for t in a.get("writes") or []:
                    s.run(
                        "MERGE (t:DataTable {name:$t}) WITH t "
                        "MATCH (n:Agent {id:$id}) MERGE (n)-[:WRITES_TO]->(t)",
                        t=t, id=a["id"],
                    )
                    counts["edges"] += 1
                for t in a.get("reads") or []:
                    s.run(
                        "MERGE (t:DataTable {name:$t}) WITH t "
                        "MATCH (n:Agent {id:$id}) MERGE (n)-[:READS]->(t)",
                        t=t, id=a["id"],
                    )
                    counts["edges"] += 1

            for sc in doc.get("screens") or []:
                s.run(
                    "MERGE (n:Screen {id:$id}) "
                    "SET n.name=$name, n.shows=$shows, n.known_gap=$gap",
                    id=sc["id"], name=sc.get("name"), shows=sc.get("shows"),
                    gap=sc.get("known_gap"),
                )
                counts["Screen"] += 1
                for ep in sc.get("backed_by") or []:
                    s.run(
                        "MERGE (e:Endpoint {path:$p}) WITH e "
                        "MATCH (n:Screen {id:$id}) MERGE (n)-[:CALLS]->(e)",
                        p=ep, id=sc["id"],
                    )
                    counts["edges"] += 1
                for t in sc.get("reads_tables") or []:
                    # This is the edge that joins the platform graph to the DATA graph:
                    # the same table an Agent WRITES_TO is the one a Screen SURFACES.
                    s.run(
                        "MERGE (t:DataTable {name:$t}) WITH t "
                        "MATCH (n:Screen {id:$id}) MERGE (n)-[:SURFACES]->(t)",
                        t=t, id=sc["id"],
                    )
                    counts["edges"] += 1

            for m in doc.get("models") or []:
                s.run(
                    "MERGE (n:Model {id:$id}) SET n.name=$name, n.role=$role",
                    id=m["id"], name=m.get("name"), role=m.get("role"),
                )
                counts["Model"] += 1

            # Record what is BROKEN as first-class knowledge. A graph that only holds the
            # happy path teaches the model to expect one.
            for g in doc.get("known_gaps") or []:
                s.run(
                    "MERGE (n:Gap {id:$id}) SET n.what=$what",
                    id=g["id"], what=g.get("what"),
                )
                counts["Gap"] += 1
    finally:
        drv.close()
    log.info("platform_kg: synced %s", counts)
    return counts


_STOPWORDS = {"the", "a", "an", "of", "for", "to", "in", "on", "is", "and", "what", "how", "does"}


def describe(topic: str, limit: int = 8) -> list[dict[str, Any]]:
    """What does the platform know about `topic`? Retrieval for AgentNick.

    Matches on ANY significant word, ranked by how many of them a node hits. Requiring the
    whole phrase as a substring found nothing for "quote promotion" or "invoices screen" —
    the exact shape of question this exists to answer.
    """
    words = [w for w in (topic or "").lower().split() if w not in _STOPWORDS and len(w) > 2]
    if not words:
        words = [(topic or "").lower()]

    q = """
    CALL {
        MATCH (p:Process)
        WITH p, toLower(p.name + ' ' + coalesce(p.summary,'') + ' ' + coalesce(p.id,'')) AS hay
        RETURN 'Process' AS kind, p.name AS name, p.summary AS detail, hay
      UNION
        MATCH (s:Stage)
        WITH s, toLower(s.name + ' ' + coalesce(s.detail,'') + ' ' + coalesce(s.id,'')) AS hay
        RETURN 'Stage' AS kind, s.name AS name, s.detail AS detail, hay
      UNION
        MATCH (a:Agent)
        WITH a, toLower(a.name + ' ' + coalesce(a.does,'') + ' ' + coalesce(a.id,'')) AS hay
        RETURN 'Agent' AS kind, a.name AS name, a.does AS detail, hay
      UNION
        MATCH (c:Screen)
        WITH c, toLower(c.name + ' ' + coalesce(c.shows,'') + ' ' + coalesce(c.known_gap,'') + ' ' + coalesce(c.id,'')) AS hay,
             trim(coalesce(c.shows,'') + ' ' + coalesce(c.known_gap,'')) AS det
        RETURN 'Screen' AS kind, c.name AS name, det AS detail, hay
      UNION
        MATCH (g:Gap)
        WITH g, toLower(g.id + ' ' + coalesce(g.what,'')) AS hay
        RETURN 'Known gap' AS kind, g.id AS name, g.what AS detail, hay
    }
    WITH kind, name, detail, hay,
         size([w IN $words WHERE hay CONTAINS w]) AS hits
    WHERE hits > 0
    RETURN kind, name, detail, hits
    ORDER BY hits DESC, kind
    LIMIT $lim
    """
    drv = _driver()
    try:
        with drv.session() as s:
            return [
                {"kind": r["kind"], "name": r["name"], "detail": r["detail"]}
                for r in s.run(q, words=words, lim=limit)
            ]
    finally:
        drv.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv

    load_dotenv()
    print(sync())
