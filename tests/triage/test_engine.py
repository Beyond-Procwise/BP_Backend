from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.services.governed_limits import LimitUnavailable
from src.services.triage import engine
from tests.triage.helpers import deal, inv, line, make_cfg, po, quote

CFG = make_cfg()


def test_triage_set_on_a_clean_deal():
    out = engine.triage_set(deal(po(), inv()), CFG)
    assert out.findings == [] and out.verdict.verdict == "Matched"


def test_triage_set_on_a_quote_only_deal_does_not_crash():
    assert engine.triage_set(deal(quote()), CFG).verdict.verdict == "Incomplete"


@pytest.fixture
def fakes(monkeypatch):
    sets = {
        "D-OK": deal(po(), inv(), deal_id="D-OK"),
        "D-BAD": deal(po(), inv(currency="EUR"), deal_id="D-BAD"),
        "D-BOOM": deal(po(), inv(), deal_id="D-BOOM"),
    }
    calls = []

    @contextmanager
    def connect():
        yield SimpleNamespace(cursor=lambda: None)

    monkeypatch.setattr(engine.loader, "load_deal_sets",
                        lambda cur, ids: {d: sets[d] for d in ids if d in sets})
    monkeypatch.setattr(engine.writer, "start_run", lambda conn, mode, cfg: calls.append("start") or "RUN-1")
    monkeypatch.setattr(engine.writer, "write_batch",
                        lambda conn, run_id, outs: calls.append([o.deal_id for o in outs]) or {"inserted": 1})
    monkeypatch.setattr(engine.writer, "finish_run", lambda conn, run_id, report: calls.append("finish"))
    real = engine.triage_set

    def maybe_boom(ds, cfg):
        if ds.deal_id == "D-BOOM":
            raise ValueError("boom")
        return real(ds, cfg)

    monkeypatch.setattr(engine, "triage_set", maybe_boom)
    return SimpleNamespace(connect=connect, calls=calls)


def test_one_failing_deal_does_not_stop_the_run(fakes):
    report = engine.run_triage(["D-OK", "D-BOOM", "D-BAD", "D-NONE"], "backfill",
                               cfg=CFG, connect=fakes.connect)
    assert report.deals_done == 2 and list(report.failed) == ["D-BOOM"]
    assert report.deals_without_documents == 1
    assert fakes.calls == ["start", ["D-OK", "D-BAD"], "finish"]
    assert report.verdicts["Blocked"] == 1 and report.shown == 1
    assert 0 < report.noise_ratio <= 1


def test_dry_run_writes_nothing(fakes):
    report = engine.run_triage(["D-OK", "D-BAD"], "backfill", dry_run=True, cfg=CFG,
                               connect=fakes.connect)
    assert fakes.calls == [] and report.deals_done == 2


def test_missing_policy_aborts_before_any_write(fakes, monkeypatch):
    def refuse():
        raise LimitUnavailable("no triage_tolerances")
    monkeypatch.setattr(engine, "load_config", refuse)
    with pytest.raises(LimitUnavailable):
        engine.run_triage(["D-OK"], "backfill", connect=fakes.connect)
    assert fakes.calls == []


def test_report_renders_the_scale_measures(fakes):
    report = engine.run_triage(["D-OK", "D-BAD"], "backfill", cfg=CFG, connect=fakes.connect,
                               known_gaps=["Invoices with no deal_id are not triaged: 7"])
    text = report.render()
    for needle in ("Noise ratio", "deals/s", "Blocked", "Payee bank details",
                   "Invoices with no deal_id are not triaged: 7"):
        assert needle in text
    assert report.to_dict()["verdicts"]["Blocked"] == 1


def test_view_dict_lists_only_s1_and_s2():
    out = engine.triage_set(deal(po(), inv(currency="EUR", terms="60 days")), CFG)
    view = engine.view_dict(out, {})
    assert view["verdict"] == "Blocked"
    assert [f["severity"] for f in view["findings"]] == ["S1", "S2"]
