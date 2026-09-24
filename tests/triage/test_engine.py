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


# --- final review F1/F2: scheduled re-triage by content fingerprint ---------------

@pytest.fixture
def sched(monkeypatch):
    """run_changed over fake sets and a fake bp_triage_deal_state (deal -> (hash, cfg))."""
    sets = {"D-A": deal(po(), inv(), deal_id="D-A"),
            "D-B": deal(po(), inv(currency="EUR"), deal_id="D-B")}
    state: dict = {}
    written: list = []
    boom: set = set()

    @contextmanager
    def connect():
        yield SimpleNamespace(cursor=lambda: None)

    def write_batch(conn, run_id, outs):
        for o in outs:
            written.append(o)
            if o.content_hash:
                state[o.deal_id] = (o.content_hash, CFG.fingerprint)
            else:
                state.pop(o.deal_id, None)
        return {"inserted": 0}

    monkeypatch.setattr(engine, "_has_baseline", lambda cur: True)
    monkeypatch.setattr(engine.writer, "deal_state", lambda cur: dict(state))
    monkeypatch.setattr(engine.loader, "list_deal_ids", lambda cur: sorted(sets))
    monkeypatch.setattr(engine.loader, "load_deal_sets",
                        lambda cur, ids: {d: sets[d] for d in ids if d in sets})
    monkeypatch.setattr(engine.writer, "start_run", lambda conn, mode, cfg: "RUN-S")
    monkeypatch.setattr(engine.writer, "write_batch", write_batch)
    monkeypatch.setattr(engine.writer, "finish_run", lambda conn, run_id, report: None)
    real = engine.triage_set

    def maybe_boom(ds, cfg):
        if ds.deal_id in boom:
            raise ValueError("boom")
        return real(ds, cfg)

    monkeypatch.setattr(engine, "triage_set", maybe_boom)

    def run(cfg=CFG):
        written.clear()
        return engine.run_changed(cfg=cfg, connect=connect)

    return SimpleNamespace(sets=sets, state=state, written=written, boom=boom, run=run)


def _ids(outs):
    return sorted(o.deal_id for o in outs)


def test_new_deals_with_no_state_are_selected_then_unchanged_ones_are_not(sched):
    report = sched.run()
    assert report.mode == "scheduled" and _ids(sched.written) == ["D-A", "D-B"]
    assert sched.run() is None and sched.written == []      # nothing changed: no run at all


def test_a_changed_deal_is_selected(sched):
    sched.run()
    sched.sets["D-A"].invoices[0].payment_terms = "60 days"
    sched.run()
    assert _ids(sched.written) == ["D-A"]


def test_a_tolerance_change_reselects_every_deal(sched):
    sched.run()
    sched.run(cfg=make_cfg(quantity_over_pct="7"))
    assert _ids(sched.written) == ["D-A", "D-B"]


def test_a_vanished_deal_reaches_the_writer_as_an_empty_output(sched):
    sched.run()
    del sched.sets["D-B"]
    report = sched.run()
    (out,) = sched.written
    assert (out.deal_id, out.results, out.findings, out.content_hash) == ("D-B", [], [], "")
    assert "D-B" not in sched.state and report.deals_vanished == 1
    assert sched.run() is None                               # and it is not chased again


def test_a_failed_deal_keeps_no_state_and_is_retried_next_pass(sched):
    sched.boom.add("D-B")
    report = sched.run()
    assert list(report.failed) == ["D-B"] and "D-B" not in sched.state
    sched.boom.clear()
    sched.run()
    assert _ids(sched.written) == ["D-B"]


def test_no_full_backfill_means_nothing_runs(sched, monkeypatch):
    monkeypatch.setattr(engine, "_has_baseline", lambda cur: False)
    assert sched.run() is None and sched.written == []


def test_baseline_counts_only_full_backfills():
    assert "mode = 'backfill'" in engine._BASELINE_SQL
    assert "scheduled" not in engine._BASELINE_SQL and "single" not in engine._BASELINE_SQL


def test_triage_set_carries_the_content_hash():
    from src.services.triage.fingerprint import deal_content_hash
    ds = deal(po(), inv())
    assert engine.triage_set(ds, CFG).content_hash == deal_content_hash(ds)
