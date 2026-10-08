"""scripts/health_alerts.py: one evaluation per timer firing, one JSON line
to the journal, and a non-zero exit only when the alerting itself crashed."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "health_alerts.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("health_alerts", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_outcome_is_one_json_line(script, monkeypatch, capsys):
    import src.services.signal_alerts as sa
    monkeypatch.setattr(sa, "run_from_env", lambda: {
        "active": ["index_empty"], "raised": ["index_empty"], "cleared": [], "notified": False})

    assert script.main() == 0
    line = json.loads(capsys.readouterr().out.strip())
    assert line["event"] == "signal_alerts" and line["active"] == ["index_empty"]


def test_a_crash_exits_non_zero_so_systemd_marks_the_unit_failed(script, monkeypatch, capsys):
    import src.services.signal_alerts as sa

    def _boom():
        raise RuntimeError("mail away")

    monkeypatch.setattr(sa, "run_from_env", _boom)

    assert script.main() == 1
    assert json.loads(capsys.readouterr().out.strip())["event"] == "signal_alerts_error"
