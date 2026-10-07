"""The registry is the allowlist: nothing outside it is interpreted, and no request value is ever SQL."""
import datetime as dt

import pytest

from src.services.report_data import live, registry as R
from src.services.report_data.scope import Scope
from src.services.report_data.spec import SpecRejected, parse_tile, validate_tile

ADMIN = Scope("Admin", True, ())
BUYER = Scope("Buyer", False, ("CC000109", "CC000021"))
F, T = dt.date(2026, 1, 1), dt.date(2026, 4, 1)


def _t(**kw):
    base = {"id": "t", "metric": "committed_spend", "viz": "kpi"}
    base.update(kw)
    return parse_tile(base)


@pytest.mark.parametrize("raw,why", [
    ({"metric": "no_such_metric"}, "unknown metric"),
    ({"metric": "committed_spend", "groupBy": ["no_such_dim"]}, "unknown dimension"),
    ({"metric": "quote_volume", "groupBy": ["detector_type"]}, "cannot be grouped"),
    ({"metric": "committed_spend", "groupBy": ["category"]}, "no live data"),
    ({"metric": "committed_spend", "filters": {"month": ["2026-01-01"]}}, "filter by period"),
    ({"metric": "committed_spend", "groupBy": ["month"], "viz": "kpi", "derive": None}, None),
])
def test_the_registry_refuses_what_it_does_not_list(raw, why):
    if why is None:
        validate_tile(_t(**raw), "live")          # a KPI with a group-by is allowed to be built; it just shows the first row
        return
    with pytest.raises(SpecRejected, match=why):
        validate_tile(_t(**raw), "live")


@pytest.mark.parametrize("raw", [
    {"metric": "committed_spend", "viz": "pie"}, {"metric": "committed_spend", "comparison": "last_tuesday"},
    {"metric": "committed_spend", "groupBy": ["month", "supplier", "region"]},
    {"metric": "committed_spend", "groupBy": ["month", "month"]}, {"metric": "committed_spend", "top": 0},
    {"metric": "committed_spend", "top": "5; DROP TABLE x"}, {"metric": "committed_spend", "sort": "random"},
    {"metric": "committed_spend", "derive": {"op": "ratio", "a": "x", "b": "y"}}, {"viz": "kpi"},
    {"derive": {"op": "ratio", "a": "committed_spend"}}, {"derive": {"op": "divide", "a": "a", "b": "b"}},
])
def test_a_malformed_spec_is_rejected_whole(raw):
    with pytest.raises(SpecRejected):
        parse_tile({"id": "t", **raw})


def test_a_presentation_only_metric_is_rejected_nowhere_but_has_no_live_source():
    m = R.METRICS["spend_by_category"]
    assert m.availability == R.PRESENTATION_ONLY and m.measure is None
    with pytest.raises(LookupError):
        live.build_query(m, [], {}, F, T, ADMIN)


def test_request_values_are_bound_never_concatenated():
    evil = "x'); DROP TABLE proc.bp_invoice_trgt; --"
    sql, params = live.build_query(R.METRICS["committed_spend"], ["supplier"], {"region": [evil]}, F, T, BUYER)
    assert evil not in sql and "DROP" not in sql
    assert evil in next(v for k, v in params.items() if k.startswith("flt_"))
    # every placeholder has a parameter, and nothing else is bound
    import re
    assert set(re.findall(r"%\((\w+)\)s", sql)) == set(params)


def test_every_registered_live_metric_builds_for_every_dimension_it_lists():
    for m in R.live_metrics():
        for d in m.dimensions:
            sql, params = live.build_query(m, [d], {}, F, T, BUYER)
            assert "GROUP BY" in sql and "%(t_from)s" in sql
        assert m.source in R.SOURCES and R.SOURCES[m.source].requires


def test_a_buyer_query_carries_the_scope_and_an_admin_query_does_not():
    b, bp = live.build_query(R.METRICS["committed_spend"], [], {}, F, T, BUYER)
    a, ap = live.build_query(R.METRICS["committed_spend"], [], {}, F, T, ADMIN)
    assert "i.buyer_id = ANY(%(buyers)s)" in b and bp["buyers"] == ["CC000109", "CC000021"]
    assert "buyers" not in ap and "ANY(%(buyers)s)" not in a


def test_findings_are_scoped_through_their_document_to_the_deals_buyer():
    sql, params = live.build_query(R.METRICS["duplicate_risk"], [], {}, F, T, BUYER)
    assert "o.buyer_id = ANY(%(buyers)s)" in sql and "bp_deal_documents" in sql


def test_a_buyer_with_nothing_assigned_runs_no_query(monkeypatch):
    monkeypatch.setattr(live, "run_query", lambda *a, **k: pytest.fail("must not query"))
    assert live.series(R.METRICS["committed_spend"], [], {}, F, T, Scope("Buyer", False, ())) == []
