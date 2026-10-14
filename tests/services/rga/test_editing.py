"""The light editor: words and layout only, never a number.

An edit is re-drawn from the report's STORED Fact Pack (no re-query), post-checked on both
files exactly as the agent's own draft was, and saved as a new version only if it passes.
The store is faked: what is under test is the rule, not the table (the table's half is in
test_job_store_live.py).
"""
from __future__ import annotations

import copy

import pytest

from src.services.rga import editing
from src.services.rga.render import html as page_renderer

JOB = "rpt-edit-1"


class FakeStore:
    def __init__(self, pack, ast, *, status="released", editable=True):
        self.job = {"job_id": JOB, "report_type": "exec_procurement_summary",
                    "status": status, "run_id": "FP-fixture0001", "requested_by": "asker",
                    "editable": editable, "current_version": 1 if editable else None,
                    "title": "Executive procurement summary"}
        self.stored = {"version": 1, "title": "Executive procurement summary",
                       "ast": ast.model_dump(mode="json"),
                       "fact_pack": pack.model_dump(mode="json", exclude_computed_fields=True)}
        self.saved = []
        self.stale = None

    def get(self, job_id):
        return dict(self.job) if job_id == JOB else None

    def draft(self, job_id):
        return copy.deepcopy(self.stored) if job_id == JOB and self.job["editable"] else None

    def save_version(self, job_id, **kw):
        if self.stale is not None:
            raise editing.StaleVersion(self.stale)
        new = kw["base_version"] + 1
        kw["before_commit"](new)
        self.saved.append((job_id, kw))
        return new


@pytest.fixture
def store(monkeypatch, pack, ast):
    s = FakeStore(pack, ast)
    monkeypatch.setattr(editing, "job_store", s)
    return s


@pytest.fixture(autouse=True)
def events(monkeypatch):
    """bp_agent_actions is append-only: every emit is captured, never written."""
    seen = []
    monkeypatch.setattr(editing.audit, "emit", lambda action, **k: seen.append((action, k)))
    return seen


def _ast(store):
    return copy.deepcopy(store.stored["ast"])


def _save(store, ast_json, *, title="Executive procurement summary", base=1):
    return editing.save(JOB, base_version=base, title=title, ast_json=ast_json,
                        by="editor", summary="tightened the wording")


def test_the_editor_is_offered_each_figure_without_its_workings(store, pack):
    d = editing.draft(JOB)
    assert d["version"] == 1 and d["editable"] is True and d["reason"] is None
    assert d["title"] == "Executive procurement summary" and d["ast"]["sections"]
    first = d["facts"][0]
    assert set(first) == {"fact_id", "label", "display", "confidence", "origin"}
    assert first == {"fact_id": "F0001", "label": "Invoiced spend (GBP)",
                     "display": pack.fact("F0001").display, "confidence": "CORROBORATED",
                     "origin": pack.fact("F0001").origin.value}


def test_a_report_from_before_the_editor_says_why_it_cannot_be_edited(store):
    store.job.update(editable=False)
    d = editing.draft(JOB)
    assert d["editable"] is False and "made before editing existed" in d["reason"]
    with pytest.raises(editing.NotEditable, match="made before editing existed"):
        _save(store, _ast(store))


@pytest.mark.parametrize("status", ["blocked", "failed", "running"])
def test_only_a_released_report_can_be_edited(store, status):
    store.job.update(status=status)
    with pytest.raises(editing.NotEditable, match="only a released report"):
        _save(store, _ast(store))


def test_a_typed_number_in_a_sentence_is_refused_in_words(store):
    a = _ast(store)
    a["sections"][0]["blocks"][2]["text"] = "Spend rose by 40 percent to {{F0001}}."
    with pytest.raises(editing.EditRefused) as exc:
        _save(store, a)
    assert any("Sentences can't contain typed numbers — insert a figure instead" in r
               and "Executive summary" in r for r in exc.value.reasons)
    assert store.saved == []


def test_a_typed_number_in_a_table_cell_is_refused_in_words(store):
    a = _ast(store)
    a["sections"][1]["blocks"][0]["rows"][0][1] = "12,000"
    with pytest.raises(editing.EditRefused) as exc:
        _save(store, a)
    assert any("Table cells can't contain typed numbers" in r for r in exc.value.reasons)
    assert store.saved == []


def test_a_figure_that_is_not_in_the_report_is_refused(store):
    a = _ast(store)
    a["sections"][0]["blocks"][2]["text"] = "Spend was {{F0099}}."
    with pytest.raises(editing.EditRefused) as exc:
        _save(store, a)
    assert any("F0099 isn't one of this report's figures" in r for r in exc.value.reasons)
    assert store.saved == []


def test_a_number_typed_into_a_title_is_refused(store):
    with pytest.raises(editing.EditRefused) as exc:
        _save(store, _ast(store), title="Summary for board meeting 4471")
    assert any("The report title" in r and "can't contain typed numbers" in r
               for r in exc.value.reasons)
    assert store.saved == []


def test_a_recommendation_resting_on_an_unmeasured_figure_is_refused(store):
    a = _ast(store)
    a["sections"][0]["blocks"].append({"type": "narrative", "role": "recommendation",
                                       "text": "Chase {{F0004}} harder.", "fact_refs": ["F0004"]})
    with pytest.raises(editing.EditRefused) as exc:
        _save(store, a)
    assert any("recommendation can't rest on" in r and "Realised savings" in r
               for r in exc.value.reasons)
    assert store.saved == []


def test_an_empty_title_is_refused(store):
    with pytest.raises(editing.EditRefused) as exc:
        _save(store, _ast(store), title="   ")
    assert any("needs a title" in r for r in exc.value.reasons)


def test_a_good_edit_is_redrawn_from_the_stored_pack_and_saved(store, events, pack):
    a = _ast(store)
    a["sections"][0]["blocks"][2]["text"] = "Across {{F0002}} deals the spend came to {{F0001}}."
    a["sections"].reverse()                                   # layout: sections reordered
    new = _save(store, a, title="Quarterly spend summary")
    assert new == 2
    (job_id, kw), = store.saved
    assert job_id == JOB and kw["base_version"] == 1 and kw["by"] == "editor"
    assert kw["title"] == "Quarterly spend summary" and kw["summary"] == "tightened the wording"
    text = " ".join(page_renderer.extract_text(kw["page"]))
    assert "Across 376 deals the spend came to" in text
    assert text.index("Control coverage") < text.index("Executive summary")
    assert kw["deck"][:2] == b"PK"                           # the deck is redrawn too
    assert f"hash {pack.hash[:12]}" in kw["page"].decode()   # the stored pack, unchanged
    (action, ev), = events
    assert action == editing.audit.EDITED
    assert ev["details"]["version"] == 2 and ev["details"]["base_version"] == 1
    assert ev["details"]["edited_by"] == "editor"
    assert ev["details"]["page_sha256"] and ev["details"]["deck_sha256"]
    assert ev["pack_hash"] == pack.hash


def test_someone_else_saving_first_is_passed_on(store, events):
    store.stale = 3
    with pytest.raises(editing.StaleVersion) as exc:
        _save(store, _ast(store), base=2)
    assert exc.value.current == 3 and events == []


def test_a_preview_draws_the_page_and_saves_nothing(store):
    a = _ast(store)
    a["sections"][0]["blocks"][2]["text"] = "Spend came to {{F0001}}."
    html, reasons = editing.preview(JOB, title="Draft title", ast_json=a)
    assert reasons == [] and b"Spend came to" in html and store.saved == []


def test_a_preview_of_a_failing_edit_still_shows_the_page_and_the_reasons(store):
    a = _ast(store)
    a["sections"][0]["blocks"].append({"type": "narrative", "role": "recommendation",
                                       "text": "Chase {{F0004}} harder.", "fact_refs": ["F0004"]})
    html, reasons = editing.preview(JOB, title="T", ast_json=a)
    assert html and any("recommendation can't rest on" in r for r in reasons)


def test_a_preview_that_cannot_be_drawn_gives_only_reasons(store):
    a = _ast(store)
    a["sections"][0]["blocks"][2]["text"] = "Up 40 percent."
    html, reasons = editing.preview(JOB, title="T", ast_json=a)
    assert html is None and any("typed numbers" in r for r in reasons)


def test_an_oversized_report_is_refused(store):
    a = _ast(store)
    a["sections"] = a["sections"] * 30
    with pytest.raises(editing.EditRefused) as exc:
        _save(store, a)
    assert any("too many sections" in r for r in exc.value.reasons)


def test_every_mistake_is_reported_at_once(store):
    a = _ast(store)
    a["sections"][0]["blocks"][2]["text"] = "Up 40 percent."
    a["sections"][1]["blocks"][0]["rows"][0][1] = "12,000"
    with pytest.raises(editing.EditRefused) as exc:
        _save(store, a, title="")
    joined = " | ".join(exc.value.reasons)
    assert "needs a title" in joined and "Sentences can't" in joined and "Table cells" in joined


# -- final review fixes ----------------------------------------------------------------------

@pytest.mark.parametrize("title", ["Summary</style>", "Summary {x}", "a > b"])
def test_a_title_may_not_carry_markup(store, title):
    with pytest.raises(editing.EditRefused) as exc:
        _save(store, _ast(store), title=title)
    assert any("can't contain" in r and "< > { }" in r for r in exc.value.reasons)


def test_a_section_title_may_not_carry_markup(store):
    a = _ast(store)
    a["sections"][0]["title"] = "Exec </style>"
    with pytest.raises(editing.EditRefused):
        _save(store, a)


@pytest.mark.parametrize("where", ["section", "column", "series", "title"])
def test_a_number_typed_into_a_heading_is_refused_even_if_it_matches_a_figure(store, where):
    """Final review I1: '376' is F0002's value, so the post-check let 'Savings of 376
    suppliers' through as a section title. Headings and labels refuse a NEW number outright."""
    a, title = _ast(store), "Executive procurement summary"
    if where == "section":
        a["sections"][0]["title"] = "Savings of 376 suppliers"
    elif where == "column":
        a["sections"][1]["blocks"][0]["columns"][0] = "Measure 376"
    elif where == "series":
        a["sections"][1]["blocks"][1]["series"][0]["label"] = "Spend ½"
    else:
        title = "Board pack 376"
    with pytest.raises(editing.EditRefused) as exc:
        _save(store, a, title=title)
    assert any("Titles and labels can't contain typed numbers" in r for r in exc.value.reasons)
    assert store.saved == []


def test_a_number_the_agent_itself_wrote_in_a_heading_may_stay(store):
    base = store.stored["ast"]
    base["sections"][0]["title"] = "Executive summary 2026 Q1"      # as the agent wrote it
    a = _ast(store)
    a["sections"].reverse()
    assert _save(store, a) == 2


def test_the_preview_is_marked_as_a_draft_and_the_saved_page_is_not(store):
    html, _ = editing.preview(JOB, title="T", ast_json=_ast(store))
    assert b"DRAFT \xe2\x80\x94 not signed off" in html
    _save(store, _ast(store))
    assert b"DRAFT" not in store.saved[0][1]["page"]
