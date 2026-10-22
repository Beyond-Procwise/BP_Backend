"""The export tools as the agent sees them: content in, an export id out, refusals and outages said plainly."""
from types import SimpleNamespace as NS

import pytest

from orchestration import agentnick_control as control
from src.services import export_store as ES
from src.services.contract_tools import store


@pytest.fixture()
def tools(monkeypatch):
    saved = []

    def fake_save(kind, *, title, content, created_by, source="agent", filename=None, rows=None):
        saved.append((kind, title, len(content), created_by, source))
        return {"export_id": "e1", "filename": f"{title}.{kind}", "size_bytes": len(content)}
    monkeypatch.setattr(ES, "save", fake_save)
    t = {x.name: x for x in control.build_tools(NS(agents={"opportunity_miner": NS()}), user_id="carol")}
    return t, saved


def test_the_four_export_tools_are_offered_and_build_under_the_callers_name(tools):
    t, saved = tools
    assert {"build_docx", "build_xlsx", "build_pptx", "render_chart"} <= set(t)
    out = t["build_xlsx"].handler(title="Spend", sheets=[{"name": "s", "columns": ["a"], "rows": [[1]]}])
    assert out["found"] and out["written"] and out["export_id"] == "e1" and out["filename"] == "Spend.xlsx"
    assert saved[0][0] == "xlsx" and saved[0][3] == "carol" and saved[0][2] > 0
    assert "export id e1" in out["summary"] and "nothing else was changed" in out["summary"]
    out = t["render_chart"].handler(title="Top", chart_type="bar", data=[{"label": "a", "value": 1}], unit="GBP")
    assert out["found"] and saved[-1][0] == "svg"


def test_content_outside_the_limits_comes_back_as_a_plain_refusal_not_a_file(tools):
    t, saved = tools
    out = t["build_docx"].handler(title="Note", sections=[])
    assert out["found"] is False and "Nothing was built" in out["note"] and saved == []


def test_an_unreachable_store_is_said_and_the_answer_stays_in_text(tools, monkeypatch):
    t, _ = tools

    def down(*a, **k):
        raise store.StoreUnavailable("down")
    monkeypatch.setattr(ES, "save", down)
    out = t["build_pptx"].handler(title="Deck", slides=[{"title": "s", "bullets": ["b"]}])
    assert out["found"] is False and out["available"] is False and "no file was saved" in out["note"]


def test_the_export_tools_are_on_the_ask_path_and_in_the_registry():
    from services import ask_routing
    from scripts.agent_policy import seed_registry
    for name in ("build_docx", "build_xlsx", "build_pptx", "render_chart"):
        assert name in ask_routing.READ_ONLY_TOOLS and name in seed_registry.FIXED_TOOLS
