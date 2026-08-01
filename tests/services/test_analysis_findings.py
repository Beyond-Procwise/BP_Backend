"""The frozen findings blob. Fail-closed is the whole point: a source that
errored must be recorded as 'error', never as an empty list, because an empty
list reads as 'we looked and found nothing'."""
from src.services import analysis_findings


class FakeCursor:
    def __init__(self, script):
        self.script = script
        self.last = ""
        self.calls = []

    def execute(self, sql, params=None):
        self.last = " ".join(sql.split())
        self.calls.append((self.last, params))
        for marker, behaviour in self.script.items():
            if marker in self.last:
                if isinstance(behaviour, Exception):
                    raise behaviour
                self._rows = behaviour
                return
        self._rows = []

    @property
    def description(self):
        return [type("C", (), {"name": n})
                for n in (self._rows[0].keys() if self._rows else [])]

    def fetchall(self):
        return [tuple(r.values()) for r in self._rows]


class FakeConn:
    def __init__(self, script):
        self._cur = FakeCursor(script)

    def cursor(self):
        return self._cur


def _stub_benchmark(calls=None, fail_for=()):
    """A drop-in for analysis_findings._benchmark_for_deal that never touches
    the real benchmark engine or the DB. Records every deal_id it was called
    with, and raises for any deal_id listed in fail_for, so tests can prove
    both the success path and the fail-closed path without importing
    src.api.routers.benchmark or executing production code in a unit test."""
    def _run(deal_id):
        if calls is not None:
            calls.append(deal_id)
        if deal_id in fail_for:
            raise RuntimeError(f"benchmark failed for {deal_id}")
        return {"deal_id": deal_id, "stub": True}
    return _run


def test_capture_records_every_source_as_ok_when_all_succeed(monkeypatch):
    calls = []
    monkeypatch.setattr(analysis_findings, "_benchmark_for_deal", _stub_benchmark(calls))
    conn = FakeConn({
        "bp_deal_overview": [{"deal_id": "D-1", "currency": "GBP"}],
        "bp_extraction_discrepancy": [],
        "bp_opportunity": [],
        "bp_analysis_summary": [],
    })
    got = analysis_findings.capture(["D-1"], file_paths=["documents/invoice/x.xlsx"],
                                     conn=conn)

    assert set(got["sources"]) == {"deals", "discrepancies", "opportunities",
                                   "summaries", "benchmarks"}
    assert got["sources"]["deals"] == "ok"
    assert got["deals"][0]["deal_id"] == "D-1"
    assert got["sources"]["benchmarks"] == "ok"
    assert calls == ["D-1"]


def test_a_failing_source_is_recorded_as_error_not_as_empty(monkeypatch):
    monkeypatch.setattr(analysis_findings, "_benchmark_for_deal", _stub_benchmark())
    conn = FakeConn({
        "bp_deal_overview": [{"deal_id": "D-1"}],
        "bp_opportunity": RuntimeError("relation does not exist"),
    })
    got = analysis_findings.capture(["D-1"], conn=conn)

    assert got["sources"]["opportunities"] == "error"
    assert got["opportunities"] == []


def test_capture_with_no_deals_still_returns_a_usable_blob():
    """A one-off analysis that produced no deal still has a findings record.

    deals, opportunities, summaries and benchmarks are all keyed on deal_ids,
    so all four are skipped the same way - [] with status 'ok', not merely
    the one the original test happened to check.
    """
    got = analysis_findings.capture([], conn=FakeConn({}))

    for name in ("deals", "opportunities", "summaries", "benchmarks"):
        assert got[name] == [], name
        assert got["sources"][name] == "ok", name


# --- deals ------------------------------------------------------------

def test_deals_source_is_ok_when_it_succeeds(monkeypatch):
    monkeypatch.setattr(analysis_findings, "_benchmark_for_deal", _stub_benchmark())
    conn = FakeConn({"bp_deal_overview": [{"deal_id": "D-1", "currency": "GBP"}]})

    got = analysis_findings.capture(["D-1"], conn=conn)

    assert got["sources"]["deals"] == "ok"
    assert got["deals"] == [{"deal_id": "D-1", "currency": "GBP"}]


def test_deals_source_is_error_and_empty_when_it_fails(monkeypatch):
    monkeypatch.setattr(analysis_findings, "_benchmark_for_deal", _stub_benchmark())
    conn = FakeConn({"bp_deal_overview": RuntimeError("relation does not exist")})

    got = analysis_findings.capture(["D-1"], conn=conn)

    assert got["sources"]["deals"] == "error"
    assert got["deals"] == []


# --- summaries ----------------------------------------------------------

def test_summaries_source_is_ok_when_it_succeeds(monkeypatch):
    monkeypatch.setattr(analysis_findings, "_benchmark_for_deal", _stub_benchmark())
    conn = FakeConn({"bp_analysis_summary": [{"deal_id": "D-1", "summary": "ok"}]})

    got = analysis_findings.capture(["D-1"], conn=conn)

    assert got["sources"]["summaries"] == "ok"
    assert got["summaries"] == [{"deal_id": "D-1", "summary": "ok"}]


def test_summaries_source_is_error_and_empty_when_it_fails(monkeypatch):
    monkeypatch.setattr(analysis_findings, "_benchmark_for_deal", _stub_benchmark())
    conn = FakeConn({"bp_analysis_summary": RuntimeError("relation does not exist")})

    got = analysis_findings.capture(["D-1"], conn=conn)

    assert got["sources"]["summaries"] == "error"
    assert got["summaries"] == []


# --- benchmarks -----------------------------------------------------------

def test_benchmarks_source_is_ok_when_every_deal_succeeds(monkeypatch):
    calls = []
    monkeypatch.setattr(analysis_findings, "_benchmark_for_deal", _stub_benchmark(calls))
    conn = FakeConn({})

    got = analysis_findings.capture(["D-1", "D-2"], conn=conn)

    assert got["sources"]["benchmarks"] == "ok"
    assert [row["deal_id"] for row in got["benchmarks"]] == ["D-1", "D-2"]
    assert calls == ["D-1", "D-2"]


def test_benchmarks_partial_failure_records_error_and_discards_the_partial_list(monkeypatch):
    """One deal's benchmark call succeeds, the next raises. The invariant is
    error-or-empty, never error-with-a-partial-list: a consumer seeing a
    non-empty list under sources['benchmarks'] == 'ok' has no signal that a
    sibling deal's benchmark silently dropped out, and would render the
    partial list as if it were complete."""
    calls = []
    monkeypatch.setattr(analysis_findings, "_benchmark_for_deal",
                         _stub_benchmark(calls, fail_for={"D-2"}))
    conn = FakeConn({})

    got = analysis_findings.capture(["D-1", "D-2"], conn=conn)

    assert got["sources"]["benchmarks"] == "error"
    assert got["benchmarks"] == []
    assert calls == ["D-1", "D-2"]


def test_discrepancies_are_scoped_by_file_paths_not_deal_ids(monkeypatch):
    """discrepancies has no deal_id to key on, so it runs off file_paths: when
    given, the query fires with the file-path list as its param."""
    monkeypatch.setattr(analysis_findings, "_benchmark_for_deal", _stub_benchmark())
    conn = FakeConn({
        "bp_deal_overview": [],
        "bp_extraction_discrepancy": [{"discrepancy_id": "DX-1"}],
    })
    file_paths = ["documents/invoice/ORB-INV-9901_INVOICE.xlsx"]

    got = analysis_findings.capture(["D-1"], file_paths=file_paths, conn=conn)

    assert got["sources"]["discrepancies"] == "ok"
    assert got["discrepancies"] == [{"discrepancy_id": "DX-1"}]

    disc_calls = [params for sql, params in conn._cur.calls
                  if "bp_extraction_discrepancy" in sql]
    assert len(disc_calls) == 1
    assert disc_calls[0] == (file_paths,)


def test_discrepancies_query_does_not_run_when_file_paths_is_empty(monkeypatch):
    """No file_paths (a one-off analysis with no documents recorded, or the
    caller simply not asking) means the query never fires - that is not the
    same as running it and getting nothing back."""
    monkeypatch.setattr(analysis_findings, "_benchmark_for_deal", _stub_benchmark())
    conn = FakeConn({
        "bp_extraction_discrepancy": [{"discrepancy_id": "should-not-appear"}],
    })

    got = analysis_findings.capture(["D-1"], conn=conn)

    assert got["discrepancies"] == []
    assert got["sources"]["discrepancies"] == "ok"
    disc_calls = [sql for sql, _ in conn._cur.calls if "bp_extraction_discrepancy" in sql]
    assert disc_calls == []


def test_a_failing_discrepancies_source_records_error_same_as_others():
    conn = FakeConn({
        "bp_extraction_discrepancy": RuntimeError("relation does not exist"),
    })
    got = analysis_findings.capture([], file_paths=["documents/invoice/x.xlsx"],
                                     conn=conn)

    assert got["sources"]["discrepancies"] == "error"
    assert got["discrepancies"] == []


def test_headline_sums_opportunity_impact_and_recovered_amounts():
    findings = {
        "opportunities": [{"financial_impact_gbp": 10000},
                          {"financial_impact_gbp": 2400}],
        "discrepancies": [{"recovered_amount": 500}],
        "deals": [{"currency": "GBP"}],
    }
    assert analysis_findings.headline(findings) == (12900.0, "GBP")


def test_headline_is_none_when_there_is_nothing_to_add_up():
    """Never 0 — zero means 'we found nothing', None means 'no figure'."""
    findings = {"opportunities": [], "discrepancies": [], "deals": []}
    assert analysis_findings.headline(findings) == (None, None)


def test_headline_defaults_currency_to_gbp_when_no_deal_carries_one():
    """There is a figure to report, but nothing in 'deals' supplies a
    currency (e.g. discrepancies-only findings with no linked deal) - the
    list view still needs a currency to render, so it defaults to GBP
    rather than crashing or reporting None."""
    findings = {
        "opportunities": [],
        "discrepancies": [{"recovered_amount": 100}],
        "deals": [],
    }
    assert analysis_findings.headline(findings) == (100.0, "GBP")
