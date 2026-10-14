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

    # Presence of those tokens alone doesn't prove the join runs the right
    # way round -- x.bp_supplier_id = m.supplier_id (the reversed, wrong-way
    # join that returns 0 rows against the real database) would satisfy every
    # substring check above too. Pin the actual equality pairing instead, with
    # whitespace normalised so reformatting the SQL can't break the regex.
    normalised = re.sub(r"\s+", " ", sql)
    assert re.search(
        r"x\.uicanvas_supplier_id\s*=\s*m\.supplier_id"
        r"|m\.supplier_id\s*=\s*x\.uicanvas_supplier_id",
        normalised,
    ), "crosswalk's uicanvas_supplier_id must be equated with the master's supplier_id"
    assert re.search(
        r"s\.supplier_id\s*=\s*x\.bp_supplier_id"
        r"|x\.bp_supplier_id\s*=\s*s\.supplier_id",
        normalised,
    ), "bp_supplier's supplier_id must be equated with the crosswalk's bp_supplier_id"


# --- Task 8: equivalence_classes / run_item_equivalence --------------------

from src.services.graph_resolution.pass_runner import equivalence_classes


def test_equivalence_classes_are_transitive():
    linked = [("L1", "L2"), ("L2", "L3"), ("L9", "L10")]
    classes = equivalence_classes(["L1", "L2", "L3", "L9", "L10", "L11"], linked)
    as_sets = sorted([sorted(c) for c in classes])
    assert ["L1", "L2", "L3"] in as_sets
    # Lexicographic sort of ["L9", "L10"] is ["L10", "L9"] ("1" < "9"), not
    # ["L9", "L10"] -- compare sorted-to-sorted so the check is order-
    # independent instead of pinned to a string-sort artefact.
    assert sorted(["L9", "L10"]) in [sorted(c) for c in classes]
    assert ["L11"] in as_sets, "a line linked to nothing is its own class"


def test_every_line_appears_in_exactly_one_class():
    lines = ["A", "B", "C"]
    classes = equivalence_classes(lines, [("A", "B")])
    flat = [m for c in classes for m in c]
    assert sorted(flat) == sorted(lines)
    assert len(flat) == len(set(flat))


from src.services.graph_resolution.pass_runner import run_item_equivalence
from src.services.graph_resolution.profiles import item_equivalence as ie

_ITEM_ROWS = [
    {"invoice_line_id": "IL1", "item_id": "ITM1", "item_description": "Widget A",
     "unit_of_measure": "EA", "unit_price": 10.0, "invoice_id": "INV1"},
    {"invoice_line_id": "IL2", "item_id": "ITM2", "item_description": "Widget B",
     "unit_of_measure": "EA", "unit_price": 20.0, "invoice_id": "INV2"},
]


def test_run_item_equivalence_singleton_classes_never_carry_the_refused_band(
    monkeypatch,
):
    """item_equivalence is uncalibrated (edge_writer.UNCALIBRATED_PROFILES) and
    write_edges refuses band="auto_link" for it outright. A singleton (a line
    matched to nothing) has no pairwise score to report -- defaulting that
    absence to F=100/band="auto_link" (as a naive implementation might) would
    make write_edges silently refuse every such edge, so a correctly-placed
    class of one would vanish from the graph while run_item_equivalence
    reported success. Pin that it does not: every unlinked line still reaches
    the driver.

    It must also not default to a fabricated F=0.0/band="block_or_exception"
    -- that asserts a measurement that was never taken. No score field is
    written at all (None, which cypher_for/Neo4j treats as "no property"),
    and the membership is instead told apart by basis="exact_item_id": this
    line's own item_id is the class's canonical id, not a scored link to
    anything else.
    """
    monkeypatch.setattr(ie, "score", lambda a, b: {
        "F": 37.63, "decision": "block_or_exception", "P_raw": 0.05,
        "L": -2.0, "L_evidence": -1.0, "signals": [],
    })

    conn = _FakeConn(_ITEM_ROWS)
    driver = _FakeDriver()

    result = run_item_equivalence(conn, driver, limit=200)

    assert result["lines"] == 2
    assert result["classes"] == 2, "no pair reaches F>=80, so both lines are singletons"
    assert result["multi_member_classes"] == 0
    assert result["written"] == 2, "singleton edges must still reach the graph"

    edge_calls = [c for c in driver.calls if "props" in c]
    assert len(edge_calls) == 2
    for call in edge_calls:
        props = call["props"]
        assert props["band"] != "auto_link", \
            "write_edges refuses auto_link for an uncalibrated profile -- this would vanish"
        # No scoring occurred for these lines -- assert absence, not a
        # fabricated zero/empty value standing in for "no evidence".
        assert props["F"] is None
        assert props["band"] is None
        assert props["P_raw"] is None
        assert props["L_evidence"] is None
        assert props["signals"] is None
        assert props["basis"] == "exact_item_id"


def test_run_item_equivalence_links_a_pair_that_clears_the_threshold(monkeypatch):
    """When the profile does score a pair at or above 80 (auto_link_with_warning
    and up), the two lines land in one class, sharing one item_key, and the
    edges carry the real pairwise evidence rather than the no-evidence default.
    """
    monkeypatch.setattr(ie, "score", lambda a, b: {
        "F": 85.0, "decision": "auto_link_with_warning", "P_raw": 0.9,
        "L": 1.5, "L_evidence": 1.2, "signals": [{"id": "item_id"}],
    })

    conn = _FakeConn(_ITEM_ROWS)
    driver = _FakeDriver()

    result = run_item_equivalence(conn, driver, limit=200)

    assert result["classes"] == 1
    assert result["multi_member_classes"] == 1
    assert result["written"] == 2, "both lines get an OF_ITEM edge to the same Item"

    edge_calls = [c for c in driver.calls if "props" in c]
    to_values = {c["to_value"] for c in edge_calls}
    assert len(to_values) == 1, "both lines must point at the same minted Item"
    for call in edge_calls:
        assert call["props"]["band"] == "auto_link_with_warning"
        assert call["props"]["F"] == 85.0
        assert call["props"]["P_raw"] == 0.9
        assert call["props"]["L_evidence"] == 1.2, \
            "L_evidence must come from the prior-free L_evidence field, not L"
        assert call["props"]["basis"] == "scored", \
            "this membership came from a pairwise link, not item_key's exact-id rule"

    merge_calls = [c for c in driver.calls if "k" in c]
    assert len(merge_calls) == 1, "one Item MERGE per class, not per line"
