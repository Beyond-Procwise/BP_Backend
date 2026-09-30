"""The sweep: what it counts, and what it refuses to do."""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services.playbooks import sweep as mod  # noqa: E402
from src.services.playbooks.store import Playbook, PlaybookStore  # noqa: E402


class FakeCursor:
    """Yields each source's rows once, then an empty page to end the loop."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.description = None
        self._rows = []

    def execute(self, sql, params=None):
        page = self._pages.pop(0) if self._pages else []
        self.description = [(c,) for c in page[0]] if page else [("finding_id",)]
        self._rows = page[1] if page else []

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


DF_COLS = ("finding_id", "rule_id", "category", "severity", "doc_type",
           "blocks_promotion", "deal_id")
OPP_COLS = ("opportunity_id", "detector_type", "supplier_id", "category_id", "deal_id")


def pages(detection_rows=(), opportunity_rows=()):
    """One full page then an empty one, per source, in sweep order."""
    return [
        (DF_COLS, list(detection_rows)), (DF_COLS, []),
        (OPP_COLS, list(opportunity_rows)), (OPP_COLS, []),
    ]


def store_with(*playbooks):
    s = PlaybookStore(playbook_rows=[])
    s._playbooks = list(playbooks)
    return s


def pb(pid, match, source="detection_finding", name=None):
    return Playbook(playbook_id=pid, playbook_name=name or f"pb{pid}",
                    trigger_source=source, trigger_match=match,
                    agent_workflow_id=958, params={}, version=1)


@pytest.fixture(autouse=True)
def captured(monkeypatch):
    calls = {"proposed": [], "ambiguous": []}
    monkeypatch.setattr(mod.proposer, "propose",
                        lambda f, s, conn=None: calls["proposed"].append((f.finding_id, s.playbook.playbook_id)) or len(calls["proposed"]))
    monkeypatch.setattr(mod.proposer, "record_ambiguous",
                        lambda f, tied: calls["ambiguous"].append(f.finding_id))
    return calls


def test_a_matching_finding_is_proposed(captured):
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "duplicate", "duplicate", "critical", "invoice", True, "D-900")]
    )))
    report = mod.sweep(store=store_with(pb(7, {"rule_id": "duplicate"})), conn=conn)
    assert captured["proposed"] == [("4211", 7)]
    assert (report.scanned, report.proposed, report.unmatched) == (1, 1, 0)


def test_an_unmatched_finding_is_counted_not_proposed(captured):
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "quantity", "quantity", "critical", "invoice", True, "D-900")]
    )))
    report = mod.sweep(store=store_with(pb(7, {"rule_id": "duplicate"})), conn=conn)
    assert captured["proposed"] == []
    assert (report.scanned, report.proposed, report.unmatched) == (1, 0, 1)


def test_an_ambiguous_match_proposes_nothing_and_is_recorded(captured):
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "duplicate", "duplicate", "critical", "invoice", True, "D-900")]
    )))
    report = mod.sweep(
        store=store_with(pb(1, {"rule_id": "duplicate"}), pb(2, {"severity": "critical"})),
        conn=conn,
    )
    assert captured["proposed"] == []
    assert captured["ambiguous"] == ["4211"]
    assert report.ambiguous == 1


def test_both_sources_are_swept_and_never_crossed(captured):
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "duplicate", "duplicate", "critical", "invoice", True, "D-900")],
        opportunity_rows=[("OPP-1", "Invoice Overbilling", "SUP-3", None, "D-901")],
    )))
    report = mod.sweep(
        store=store_with(
            pb(1, {"rule_id": "duplicate"}),
            pb(2, {"detector_type": "Invoice Overbilling"}, source="opportunity"),
        ),
        conn=conn,
    )
    assert sorted(captured["proposed"]) == [("4211", 1), ("OPP-1", 2)]
    assert report.scanned == 2


def test_an_already_queued_finding_counts_separately(monkeypatch):
    """propose() returning None is the ordinary case on every sweep after the
    first, and must not be reported as work done."""
    monkeypatch.setattr(mod.proposer, "propose", lambda f, s, conn=None: None)
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "duplicate", "duplicate", "critical", "invoice", True, "D-900")]
    )))
    report = mod.sweep(store=store_with(pb(7, {"rule_id": "duplicate"})), conn=conn)
    assert (report.proposed, report.already_queued) == (0, 1)


def test_an_empty_playbook_table_sweeps_to_a_clean_zero(captured):
    """Shipping day. Not an error, and the count still gets logged."""
    conn = FakeConn(FakeCursor(pages(
        detection_rows=[(4211, "duplicate", "duplicate", "critical", "invoice", True, "D-900")]
    )))
    report = mod.sweep(store=store_with(), conn=conn)
    assert (report.scanned, report.proposed) == (1, 0)
    assert "0 proposed" in report.render()


def test_the_report_renders_every_count():
    text = mod.SweepReport(scanned=10, proposed=2, already_queued=7,
                           ambiguous=1, unmatched=0).render()
    for fragment in ("10 scanned", "2 proposed", "7 already queued",
                     "1 ambiguous", "0 unmatched"):
        assert fragment in text


def test_one_failing_proposal_does_not_abort_the_whole_sweep(monkeypatch, caplog):
    """A dropped connection on finding 3,000 of 4,838 used to skip every
    finding after it -- in both sources -- and, because the exception escaped
    before the log line, the run reported no count at all. _pages restarts from
    the beginning each tick, so a deterministic failure repeated for ever."""
    boom = {"n": 0}

    def sometimes(finding, selection, conn=None):
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("connection reset by peer")
        return 99

    monkeypatch.setattr(mod.proposer, "propose", sometimes)
    conn = FakeConn(FakeCursor(pages(detection_rows=[
        (1, "duplicate", "duplicate", "critical", "invoice", True, "D-1"),
        (2, "duplicate", "duplicate", "critical", "invoice", True, "D-2"),
    ])))
    report = mod.sweep(store=store_with(pb(7, {"rule_id": "duplicate"})), conn=conn)
    assert (report.scanned, report.proposed, report.failed) == (2, 1, 1)
    assert "1 failed" in report.render()


def test_the_count_is_logged_even_when_the_sweep_blows_up(monkeypatch, caplog):
    """The sweep's whole promise is that zero is visible. A run that dies
    without a count is indistinguishable from one that stopped running."""
    import logging

    def explode(*a, **kw):
        raise RuntimeError("the store vanished")

    monkeypatch.setattr(mod, "_run", explode)
    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError):
            mod.sweep(store=store_with(), conn=FakeConn(FakeCursor(pages())))
    assert "playbook sweep:" in caplog.text
