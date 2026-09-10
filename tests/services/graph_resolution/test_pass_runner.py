import re

from src.services.graph_resolution.pass_runner import (
    to_candidate_edges, band_for_resolution, run_supplier_identity,
)
from src.services.graph_resolution.profiles import supplier_identity as si
from src.services.resolution import CandidateEdge


def test_candidate_edges_carry_log_odds_not_F():
    scored = [{"source_id": "SUP-A", "target_id": "S1",
               "result": {"L": 3.2, "L_evidence": 7.1, "P_raw": 0.96, "F": 94.0}}]
    edges = to_candidate_edges(scored, profile_id="supplier_identity")
    assert isinstance(edges[0], CandidateEdge)
    assert edges[0].log_odds == 3.2   # L, prior included: the solver wants full log-odds
    assert edges[0].confidence == 0.96


def test_degenerate_resolution_is_capped_below_auto_link():
    assert band_for_resolution("auto_link", "DEGENERATE") == "review"


def test_resolved_status_leaves_the_band_alone():
    assert band_for_resolution("auto_link", "RESOLVED") == "auto_link"


def test_infeasible_never_yields_an_actionable_band():
    assert band_for_resolution("auto_link", "INFEASIBLE") == "weak_relation"


# --- Fakes for run_supplier_identity: no live DB, no live Neo4j -------------

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed_sql = None

    def execute(self, sql, *args, **kwargs):
        self.executed_sql = sql

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    """Records the last cursor it handed out, so a test can inspect the SQL."""

    def __init__(self, rows):
        self._rows = rows
        self.last_cursor = None

    def cursor(self, cursor_factory=None):
        self.last_cursor = _FakeCursor(self._rows)
        return self.last_cursor


class _FakeResult:
    def single(self):
        return {"cnt": 1}


class _FakeSession:
    def __init__(self, calls):
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self._calls.append(params)
        return _FakeResult()


class _FakeDriver:
    def __init__(self):
        self.calls = []

    def session(self):
        return _FakeSession(self.calls)


# Two supplier_master rows sharing a VAT number: identical, well-formed input
# for the scorer. What is under test is the *identifier* the code carries
# through onto the written edge, not the scoring arithmetic.
_ROWS = [
    {"supplier_id": "SUP-Acme", "supplier_name": "Acme Ltd",
     "vat_number": "GB123456789", "registration_number": "REG1",
     "duns_number": "DUNS1", "postal_code": "SW1A 1AA", "country": "GB",
     "bank_account_number": None},
    {"supplier_id": "SUP-Acme-Duplicate", "supplier_name": "Acme Ltd",
     "vat_number": "GB123456789", "registration_number": "REG1",
     "duns_number": "DUNS1", "postal_code": "SW1A 1AA", "country": "GB",
     "bank_account_number": None},
]


def test_run_supplier_identity_writes_edges_keyed_in_the_graphs_supplier_keyspace(
    monkeypatch,
):
    """Regression guard for the SI###### vs SUP-* keyspace mismatch.

    bp_supplier_master is keyed SI######; Neo4j Supplier nodes are keyed
    SUP-*. If run_supplier_identity ever again puts the master's own id
    straight onto an edge instead of bridging through the crosswalk to
    bp_supplier's id, `write_edges`'s MATCH would silently match no node
    and every edge would vanish -- exactly the bug this guards against.
    """
    monkeypatch.setattr(
        si, "score",
        lambda a, b: {
            "F": 70.0, "decision": "review", "P_raw": 0.9,
            "L": 2.0, "L_evidence": 2.0, "signals": [],
        },
    )

    conn = _FakeConn(_ROWS)
    driver = _FakeDriver()

    result = run_supplier_identity(conn, driver, limit=200)

    assert result["scored"] == 1
    assert result["written"] == 1
    assert driver.calls, "no edge reached the driver"
    for call in driver.calls:
        assert re.match(r"^SUP-", call["from_value"]), call["from_value"]
        assert re.match(r"^SUP-", call["to_value"]), call["to_value"]

    # The query itself must actually bridge through the crosswalk -- a fake
    # cursor can't prove the join resolves rows correctly, but it can prove
    # the query no longer asks bp_supplier_master alone for an id it does
    # not carry in the SUP-* keyspace.
    sql = conn.last_cursor.executed_sql
    assert "bp_supplier_id_crosswalk" in sql
    assert "uicanvas_supplier_id" in sql
    assert "bp_supplier_id" in sql
