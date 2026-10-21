import os
from datetime import datetime, timedelta, timezone

import pytest

from services.people.graph.store import InMemoryGraph, Neo4jGraph

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
TENANT = "zt_graph"


class Clock:
    def __init__(self, t=T0):
        self.t = t
    def __call__(self):
        return self.t
    def advance(self, **kw):
        self.t += timedelta(**kw)
        return self.t


def _neo():
    if os.environ.get("PROCWISE_TEST_LIVE_DB") != "1":
        pytest.skip("needs the live graph (PROCWISE_TEST_LIVE_DB=1)")
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    return d


@pytest.fixture(params=["memory", "neo4j"])
def store(request):
    if request.param == "memory":
        yield InMemoryGraph()
        return
    d = _neo()
    def wipe():
        with d.session() as s:
            s.run("MATCH (n) WHERE n.pw_tenant STARTS WITH 'zt_' DETACH DELETE n")
    wipe()
    try:
        yield Neo4jGraph(d)
    finally:
        wipe(); d.close()


@pytest.fixture()
def clock():
    return Clock()
