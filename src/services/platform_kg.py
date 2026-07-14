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
                    "SET p.name=$name, p.summary=$summary, p.say=$say, p.component=$component",
                    id=p["id"], name=p.get("name"), summary=p.get("summary"),
                    say=p.get("say"), component=p.get("component"),
                )
                counts["Process"] += 1
                prev = None
                for st in p.get("stages") or []:
                    s.run(
                        "MERGE (x:Stage {id:$id}) "
                        "SET x.name=$name, x.detail=$detail, x.say=$say, x.component=$component "
                        "WITH x MATCH (p:Process {id:$pid}) MERGE (p)-[:HAS_STAGE]->(x)",
                        id=f"{p['id']}::{st['id']}", name=st.get("name"),
                        detail=st.get("detail"), say=st.get("say"),
                        component=st.get("component"), pid=p["id"],
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
                    "MERGE (n:Agent {id:$id}) SET n.name=$name, n.does=$does, n.say=$say",
                    id=a["id"], name=a.get("name"), does=a.get("does"), say=a.get("say"),
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
                    "SET n.name=$name, n.shows=$shows, n.known_gap=$gap, n.how_to=$how_to, "
                    "n.say=$say",
                    id=sc["id"], name=sc.get("name"), shows=sc.get("shows"),
                    gap=sc.get("known_gap"), how_to=sc.get("how_to"), say=sc.get("say"),
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
                    "MERGE (n:Model {id:$id}) SET n.name=$name, n.role=$role, n.say=$say",
                    id=m["id"], name=m.get("name"), role=m.get("role"), say=m.get("say"),
                )
                counts["Model"] += 1

            # Record what is BROKEN as first-class knowledge. A graph that only holds the
            # happy path teaches the model to expect one.
            for g in doc.get("known_gaps") or []:
                s.run(
                    "MERGE (n:Gap {id:$id}) SET n.what=$what, n.say=$say",
                    id=g["id"], what=g.get("what"), say=g.get("say"),
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

    Each fact comes back in BOTH registers, and the split is the point:

      * ``say``      — the fact as the user experiences it. This is what an answer is built
                       from. Safe to paraphrase, safe to quote.
      * ``internal`` — the same fact as an engineer states it, naming tables and routes. It
                       is here so the agent can REASON. It must never reach a user.

    Returning only ``internal`` (which is all this used to do) is what broke it in both
    directions: the agent repeated the internals to whoever asked, and once that was blocked
    it had nothing left to say and went mute. Measured coverage fell to 0.258 WITH the graph
    versus 0.439 without it — the knowledge was actively making it worse, because the only
    words it had for the truth were words it was not allowed to use.

    ``say`` falls back to ``internal`` when a fact has no user-facing wording yet, so a
    half-migrated ontology degrades to the old behaviour rather than to silence.
    """
    words = [w for w in (topic or "").lower().split() if w not in _STOPWORDS and len(w) > 2]
    if not words:
        words = [(topic or "").lower()]

    q = """
    CALL {
        MATCH (p:Process)
        WITH p, toLower(p.name + ' ' + coalesce(p.summary,'') + ' ' + coalesce(p.say,'') + ' ' + coalesce(p.id,'')) AS hay
        RETURN 'Process' AS kind, p.name AS name, p.summary AS detail, p.say AS say, hay
      UNION
        MATCH (s:Stage)
        WITH s, toLower(s.name + ' ' + coalesce(s.detail,'') + ' ' + coalesce(s.say,'') + ' ' + coalesce(s.id,'')) AS hay
        RETURN 'Stage' AS kind, s.name AS name, s.detail AS detail, s.say AS say, hay
      UNION
        MATCH (a:Agent)
        WITH a, toLower(a.name + ' ' + coalesce(a.does,'') + ' ' + coalesce(a.say,'') + ' ' + coalesce(a.id,'')) AS hay
        RETURN 'Agent' AS kind, a.name AS name, a.does AS detail, a.say AS say, hay
      UNION
        MATCH (c:Screen)
        WITH c, toLower(c.name + ' ' + coalesce(c.shows,'') + ' ' + coalesce(c.known_gap,'') + ' ' + coalesce(c.how_to,'') + ' ' + coalesce(c.say,'') + ' ' + coalesce(c.id,'')) AS hay,
             trim(coalesce(c.shows,'') + ' ' + coalesce(c.known_gap,'')) AS det,
             trim(coalesce(c.say,'') + ' ' + coalesce(c.how_to,'')) AS sy
        RETURN 'Screen' AS kind, c.name AS name, det AS detail, sy AS say, hay
      UNION
        MATCH (g:Gap)
        WITH g, toLower(g.id + ' ' + coalesce(g.what,'') + ' ' + coalesce(g.say,'')) AS hay
        RETURN 'Known gap' AS kind, g.id AS name, g.what AS detail, g.say AS say, hay
    }
    WITH kind, name, detail, say, hay,
         size([w IN $words WHERE hay CONTAINS w]) AS hits
    WHERE hits > 0
    RETURN kind, name, detail, say, hits
    ORDER BY hits DESC, kind
    LIMIT $lim
    """
    drv = _driver()
    try:
        with drv.session() as s:
            out: list[dict[str, Any]] = []
            for r in s.run(q, words=words, lim=limit):
                say = (r["say"] or "").strip()
                internal = (r["detail"] or "").strip()
                out.append(
                    {
                        "kind": r["kind"],
                        "name": r["name"],
                        # What the user may be told. The answer is built from this.
                        "say": say or internal,
                        # Context to reason with. Never repeat it.
                        "internal": internal,
                        # Kept for callers written against the old shape.
                        "detail": say or internal,
                    }
                )
            return out
    finally:
        drv.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv

    load_dotenv()
    print(sync())
