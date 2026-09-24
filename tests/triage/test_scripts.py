import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Cur:
    def execute(self, *a):
        pass

    def fetchone(self):
        return (7,)


@contextmanager
def _conn():
    yield SimpleNamespace(cursor=lambda: _Cur())


def test_backfill_dry_run_passes_ids_and_the_no_deal_gap(monkeypatch, capsys):
    mod = _load("triage_backfill")
    seen = {}

    def run_triage(ids, mode, **kw):
        seen.update(ids=ids, mode=mode, **kw)
        return SimpleNamespace(failed={}, render=lambda: "REPORT", to_dict=lambda: {})

    monkeypatch.setattr(mod, "get_conn", _conn)
    monkeypatch.setattr(mod, "run_triage", run_triage)
    assert mod.main(["--deals", "A, B", "--dry-run"]) == 0
    assert seen["ids"] == ["A", "B"] and seen["mode"] == "backfill" and seen["dry_run"] is True
    assert seen["known_gaps"] == ["Invoices with no deal_id are not triaged: 7"]
    assert "REPORT" in capsys.readouterr().out


def test_rollback_prints_what_it_did(monkeypatch, capsys):
    mod = _load("triage_rollback")
    monkeypatch.setattr(mod, "get_conn", _conn)
    monkeypatch.setattr(mod.writer, "rollback_run", lambda conn, run_id: {
        "findings_removed": 3, "findings_kept": 1, "audit_rows_removed": 40})
    assert mod.main(["--run-id", "RUN-1"]) == 0
    out = capsys.readouterr().out
    assert "removed 3" in out and "kept 1" in out and "40 audit rows" in out
