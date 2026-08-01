"""The freeze runs between linking and broadcasting, and never blocks the
broadcast. A spinner that never resolves is a worse failure than a report
with no findings."""
import importlib

mod = importlib.import_module("src.services.session_notify_listener")


def _listener():
    return mod.SessionNotifyListener.__new__(mod.SessionNotifyListener)


def test_freeze_runs_after_linking_and_before_broadcasting(monkeypatch):
    order = []
    lis = _listener()
    monkeypatch.setattr(lis, "_link_session", lambda s: order.append("link"),
                        raising=False)
    monkeypatch.setattr(lis, "_freeze_analysis", lambda s: order.append("freeze"),
                        raising=False)
    monkeypatch.setattr(lis, "_broadcast", lambda s, p: order.append("cast"),
                        raising=False)

    lis._link_then_broadcast("ses-1", {})

    assert order == ["link", "freeze", "cast"]


def test_a_failing_freeze_still_broadcasts(monkeypatch):
    order = []
    lis = _listener()
    monkeypatch.setattr(lis, "_link_session", lambda s: None, raising=False)

    def boom(_):
        raise RuntimeError("db gone")

    monkeypatch.setattr(lis, "_freeze_analysis", boom, raising=False)
    monkeypatch.setattr(lis, "_broadcast", lambda s, p: order.append("cast"),
                        raising=False)

    lis._link_then_broadcast("ses-1", {})

    assert order == ["cast"]


def test_freeze_passes_the_captured_findings_to_the_store(monkeypatch):
    captured = {}
    store = importlib.import_module("src.services.analysis_store")
    findings = importlib.import_module("src.services.analysis_findings")

    monkeypatch.setattr(store, "deal_ids_for_session", lambda s: ["D-1"],
                        raising=False)
    monkeypatch.setattr(store, "document_count_for_session", lambda s: 4,
                        raising=False)
    monkeypatch.setattr(store, "file_paths_for_session",
                        lambda s: ["documents/invoice/ORB-INV-9901.xlsx"],
                        raising=False)
    monkeypatch.setattr(findings, "capture",
                        lambda ids, **k: {"deals": [{"currency": "GBP"}],
                                          "opportunities": [
                                              {"financial_impact_gbp": 900}],
                                          "discrepancies": []})
    monkeypatch.setattr(store, "freeze",
                        lambda sid, **kw: captured.update(kw) or "aid")

    _listener()._freeze_analysis("ses-1")

    assert captured["value_found"] == 900.0
    assert captured["currency"] == "GBP"
    assert captured["document_count"] == 4


def test_freeze_passes_file_paths_through_to_capture(monkeypatch):
    """This is the part that would silently produce empty discrepancies if
    wired wrong: file_paths must reach capture(), not just deal_ids."""
    captured_capture_kwargs = {}
    store = importlib.import_module("src.services.analysis_store")
    findings = importlib.import_module("src.services.analysis_findings")

    paths = ["documents/invoice/ORB-INV-9901.xlsx", "documents/po/PO-42.pdf"]

    monkeypatch.setattr(store, "deal_ids_for_session", lambda s: ["D-1"],
                        raising=False)
    monkeypatch.setattr(store, "document_count_for_session", lambda s: 2,
                        raising=False)
    monkeypatch.setattr(store, "file_paths_for_session", lambda s: paths,
                        raising=False)

    def fake_capture(ids, **kwargs):
        captured_capture_kwargs.update(kwargs)
        return {"deals": [], "opportunities": [], "discrepancies": []}

    monkeypatch.setattr(findings, "capture", fake_capture)
    monkeypatch.setattr(store, "freeze", lambda sid, **kw: "aid")

    _listener()._freeze_analysis("ses-1")

    assert captured_capture_kwargs.get("file_paths") == paths
