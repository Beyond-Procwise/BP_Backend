"""Per-tile states, the checks layer, and the presentation provider's promises."""
import datetime as dt
from types import SimpleNamespace as NS

import pytest

from src.services.report_data import live, presentation, registry as R, service
from src.services.report_data.live import Row
from src.services.report_data.scope import Scope
from src.services.report_data.spec import SpecRejected

TODAY = dt.date(2026, 10, 7)
ADMIN = Scope("Admin", True, ())
P = NS(subject="u1")


def run(tiles, scope=ADMIN, mode="live", allow=lambda a: True, mode_ok=None, f="2026-01-01", t="2026-10-31"):
    return service.compute(P, {"from": f, "to": t, "tiles": tiles, "data_mode": mode}, scope=scope,
                           authorise=allow, mode_ok=mode_ok, today=TODAY)


def months(vals):
    return [Row((dt.date(2026, i + 1, 1),), (dt.date(2026, i + 1, 1).strftime("%b %Y"),), v) for i, v in enumerate(vals)]


@pytest.fixture
def fake_live(monkeypatch):
    calls = []

    def series(metric, group_by, filters, start, end, scope):
        calls.append((metric.key, tuple(group_by), start, end))
        if group_by == ["month"]:
            return months([10.0, 20.0, 30.0])
        return [Row((), (), 60.0)]
    monkeypatch.setattr(live, "series", series)
    monkeypatch.setattr(live, "unattributed_findings", lambda *a, **k: 7)
    return calls


# ---- per-tile states --------------------------------------------------------------------------

def test_a_tile_the_caller_may_not_read_is_forbidden_not_zero_and_not_omitted(fake_live):
    out = run([{"id": "a", "metric": "committed_spend"}, {"id": "b", "metric": "duplicate_risk"}],
              allow=lambda action: action != "invoice.read")
    by = {t["id"]: t for t in out["tiles"]}
    assert by["a"]["status"] == "forbidden" and "result" not in by["a"]
    assert by["b"]["status"] == "ok"                    # one denied source marks only its own tiles
    assert len(out["tiles"]) == 2


def test_a_buyer_with_nothing_assigned_sees_an_explicit_empty_state_not_everything(fake_live):
    out = run([{"id": "a", "metric": "committed_spend"}], scope=Scope("Buyer", False, ()))
    assert out["tiles"][0]["status"] == "empty_scope" and fake_live == []


def test_in_live_mode_an_unready_tile_says_no_data_source_and_never_returns_presentation_values(fake_live):
    out = run([{"id": "a", "metric": "spend_by_category", "groupBy": ["category"], "viz": "table"}])
    t = out["tiles"][0]
    assert t["status"] == "rejected" or t["status"] == "no_data_source"
    assert "result" not in t and t["data_mode"] == "live" and t["marker"] is None
    out = run([{"id": "a", "metric": "compliance_rate"}])
    assert out["tiles"][0]["status"] == "no_data_source" and "result" not in out["tiles"][0]


def test_a_rejected_spec_is_reported_on_its_own_tile(fake_live):
    out = run([{"id": "ok", "metric": "committed_spend"}, {"id": "bad", "metric": "committed_spend", "groupBy": ["nope"]}])
    assert [t["status"] for t in out["tiles"]] == ["ok", "rejected"] and "nope" in out["tiles"][1]["reason"]


# ---- checks -----------------------------------------------------------------------------------

def test_groups_that_do_not_sum_to_the_total_are_withheld_not_published(monkeypatch):
    monkeypatch.setattr(live, "series", lambda m, g, *a: months([10.0, 20.0, 30.0]) if g else [Row((), (), 999.0)])
    out = run([{"id": "a", "metric": "committed_spend", "groupBy": ["month"], "viz": "line"}])
    t = out["tiles"][0]
    assert t["status"] == "withheld" and "result" not in t
    assert any(c["code"] == "groups_do_not_sum" for c in t["checks"])


def test_groups_that_tie_are_published_and_say_so(fake_live):
    out = run([{"id": "a", "metric": "committed_spend", "groupBy": ["month"], "viz": "line"}])
    t = out["tiles"][0]
    assert t["status"] == "ok" and any(c["code"] == "groups_sum_to_total" for c in t["checks"])


def test_empty_months_appear_as_zero_for_a_sum_and_as_a_gap_for_a_rate(monkeypatch):
    two = lambda v: [Row((dt.date(2026, 1, 1),), ("Jan 2026",), v), Row((dt.date(2026, 3, 1),), ("Mar 2026",), v)]
    monkeypatch.setattr(live, "series", lambda m, g, *a: two(5.0) if g else [Row((), (), 10.0)])
    out = run([{"id": "s", "metric": "committed_spend", "groupBy": ["month"], "viz": "line"},
               {"id": "r", "metric": "cycle_time_to_po", "groupBy": ["month"], "viz": "line"}], t="2026-03-31")
    data = {t["id"]: t["result"]["series"][0]["data"] for t in out["tiles"]}
    assert data["s"] == [5.0, 0.0, 5.0]               # February is a true zero spend
    assert data["r"] == [5.0, None, 5.0]              # February has no average: not "0 days"


def test_a_running_period_is_measured_to_today_and_compared_like_for_like(fake_live):
    out = run([{"id": "a", "metric": "committed_spend", "comparison": "prior_period"}], f="2026-10-01", t="2026-10-31")
    t = out["tiles"][0]
    assert t["params"]["period"]["partial"] is True and t["params"]["period"]["effective_to"] == "2026-10-07"
    cmp_ = t["comparison"]["window"]
    assert (cmp_["from"], cmp_["to"]) == ("2026-09-24", "2026-09-30")     # 7 days vs 7 days, not a whole September
    assert any(c["code"] == "partial_period" for c in t["checks"])


def test_every_tile_resolves_the_same_anchor(fake_live):
    out = run([{"id": "a", "metric": "committed_spend"}, {"id": "b", "metric": "quote_volume"}], f="2026-10-01", t="2026-10-31")
    assert {t["params"]["period"]["as_of"] for t in out["tiles"]} == {"2026-10-07"}
    assert {t["params"]["period"]["anchor"] for t in out["tiles"]} == {"CURRENT_DATE"}


def test_top_n_keeps_an_other_row_so_the_total_still_ties(monkeypatch):
    rows = [Row((f"s{i}",), (f"S{i}",), float(10 - i)) for i in range(6)]
    monkeypatch.setattr(live, "series", lambda m, g, *a: rows if g else [Row((), (), 45.0)])
    out = run([{"id": "a", "metric": "committed_spend", "groupBy": ["supplier"], "viz": "table", "top": 3}])
    r = out["tiles"][0]["result"]
    assert [x[0] for x in r["rows"]] == ["S0", "S1", "S2", "Other"] and r["total"] == 45.0


def test_a_derived_ratio_is_computed_from_registered_metrics_never_typed(monkeypatch):
    vals = {"off_contract_spend": 25.0, "committed_spend": 100.0}
    monkeypatch.setattr(live, "series", lambda m, g, *a: [Row((), (), vals[m.key])])
    out = run([{"id": "d", "derive": {"op": "ratio", "a": "off_contract_spend", "b": "committed_spend"}, "viz": "kpi"}])
    assert out["tiles"][0]["result"]["value"] == 0.25
    out = run([{"id": "d", "derive": {"op": "ratio", "a": "off_contract_spend", "b": "committed_spend"}, "viz": "kpi", "comparison": "none"}])
    assert out["tiles"][0]["status"] == "ok"
    with pytest.raises(SpecRejected):
        run([{"id": "d", "metric": "committed_spend", "derive": {"op": "ratio", "a": "x", "b": "y"}}])


def test_admin_sees_the_count_of_findings_no_buyer_can_see(fake_live):
    out = run([{"id": "f", "metric": "duplicate_risk"}])
    assert any(c["code"] == "unattributed_findings" and c["count"] == 7 for c in out["tiles"][0]["checks"])
    out = run([{"id": "f", "metric": "duplicate_risk"}], scope=Scope("Buyer", False, ("CC1",)))
    assert not any(c["code"] == "unattributed_findings" for c in out["tiles"][0]["checks"])


def test_three_way_match_with_nothing_assessed_is_not_a_zero(monkeypatch):
    monkeypatch.setattr(live, "series", lambda m, g, *a: [Row((), (), None, 0)])
    t = run([{"id": "a", "metric": "three_way_match_rate", "comparison": "none"}])["tiles"][0]
    assert t["result"]["value"] is None and t["result"]["assessed"] == 0


# ---- presentation -----------------------------------------------------------------------------

def test_presentation_data_is_refused_unless_the_caller_is_allowed(fake_live):
    with pytest.raises(service.ModeRefused):
        run([{"id": "a", "metric": "committed_spend"}], mode="presentation", mode_ok=lambda: False)
    with pytest.raises(service.ModeRefused):
        run([{"id": "a", "metric": "committed_spend"}], mode="presentation")       # no checker at all: refused
    assert fake_live == []


def test_a_report_never_mixes_live_and_presentation(fake_live):
    with pytest.raises(SpecRejected, match="mix"):
        run([{"id": "a", "metric": "committed_spend", "data_mode": "presentation"}], mode="live")
    with pytest.raises(SpecRejected, match="mix"):
        run([{"id": "a", "metric": "committed_spend", "data_mode": "live"}], mode="presentation", mode_ok=lambda: True)


def test_every_presentation_tile_carries_the_marker_and_the_fictional_organisation():
    tiles = [{"id": k, "metric": k} for k in R.METRICS if k != "supplier_risk_profile"] + \
            [{"id": "risk", "metric": "supplier_risk_profile", "groupBy": ["risk_dimension"], "viz": "bar"}]
    out = run(tiles, mode="presentation", mode_ok=lambda: True)
    assert out["marker"] == service.MARKER and "Fictional" in out["org"]
    assert len(out["tiles"]) == len(R.METRICS)
    for t in out["tiles"]:
        assert t["status"] == "ok", (t["id"], t.get("reason"))
        assert t["data_mode"] == "presentation" and t["marker"] == "PRESENTATION DATA - NOT REAL"


def test_presentation_covers_the_five_tiles_with_no_live_source():
    for k in ("tail_spend_visibility", "compliance_rate", "tail_spend_breakdown", "spend_by_category", "supplier_risk_profile"):
        assert R.METRICS[k].availability == R.PRESENTATION_ONLY
    out = run([{"id": "c", "metric": "spend_by_category", "groupBy": ["category"], "viz": "table"}],
              mode="presentation", mode_ok=lambda: True)
    assert out["tiles"][0]["status"] == "ok" and len(out["tiles"][0]["result"]["rows"]) == 6


@pytest.mark.parametrize("metric,dim", [("committed_spend", "supplier"), ("committed_spend", "month"),
                                        ("spend_by_category", "category"), ("tail_spend_breakdown", "tail_band"),
                                        ("committed_spend", "region")])
def test_presentation_groups_tie_to_their_totals(metric, dim):
    m = R.METRICS[metric]
    f, t = dt.date(2025, 3, 1), dt.date(2026, 6, 30)
    total = presentation.series(m, [], {}, f, t, None, TODAY)[0].value
    assert sum(r.value for r in presentation.series(m, [dim], {}, f, t, None, TODAY)) == pytest.approx(total)


def test_presentation_periods_tie_to_the_whole():
    m = R.METRICS["committed_spend"]
    whole = presentation.series(m, [], {}, dt.date(2025, 1, 1), dt.date(2025, 12, 31), None, TODAY)[0].value
    halves = sum(presentation.series(m, [], {}, a, b, None, TODAY)[0].value
                 for a, b in ((dt.date(2025, 1, 1), dt.date(2025, 6, 30)), (dt.date(2025, 7, 1), dt.date(2025, 12, 31))))
    assert whole == pytest.approx(halves)


def test_presentation_refuses_a_breakdown_it_did_not_generate():
    out = run([{"id": "a", "metric": "duplicate_risk", "groupBy": ["finding_type"], "viz": "table"}],
              mode="presentation", mode_ok=lambda: True)
    assert out["tiles"][0]["status"] == "rejected" and "result" not in out["tiles"][0]


def test_presentation_names_are_synthetic():
    names = {presentation._sup(i)[1] for i in range(1, presentation.N_SUPPLIERS + 1)}
    assert all(n.startswith("Synthetic Supplier") for n in names)
    assert all(b.startswith("Synthetic Buyer") for b in presentation.BUYERS)
