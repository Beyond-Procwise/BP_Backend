"""Phase 6: retrieval grading with bounded retries, an explicit "not found", and the comparison branch (tests B1, B2)."""
import types

import pytest

from src.services import ask_compare, ask_flags, ask_grade
from src.services.analytics import ask as analytics_ask

MODEL = "test-reranker"


# --- ask_compare -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,entities,topic", [
    ("Compare Ironbridge and Meridia", ("Ironbridge", "Meridia"), ""),
    ("Compare Ironbridge Managed IT Ltd and Meridia Cloud Platforms on payment terms",
     ("Ironbridge Managed IT Ltd", "Meridia Cloud Platforms"), "payment terms"),
    ("Please compare the Ironbridge quote with the Meridia quote.", ("Ironbridge quote", "Meridia quote"), ""),
    ("What are the differences between Ironbridge and Meridia?", ("Ironbridge", "Meridia"), ""),
    ("How does Ironbridge compare to Meridia for delivery?", ("Ironbridge", "Meridia"), "delivery"),
    ("Ironbridge vs Meridia", ("Ironbridge", "Meridia"), ""),
])
def test_comparison_questions_are_detected(text, entities, topic):
    c = ask_compare.detect(text)
    assert c is not None and c.entities == entities and c.topic == topic


@pytest.mark.parametrize("text", [
    "compare them", "Compare this and that", "Who is the cheapest supplier?", "What is in the Ironbridge contract?",
    "Compare Ironbridge and Ironbridge", "Show me both", "", None,
])
def test_other_questions_are_not_comparisons(text):
    assert ask_compare.detect(text) is None


def test_each_side_gets_its_own_query_and_the_prompt_names_a_side_with_nothing():
    c = ask_compare.detect("Compare Ironbridge and Meridia on payment terms")
    assert c.search_queries() == ["Ironbridge payment terms", "Meridia payment terms"]
    line = ask_compare.instruction(c, ["Meridia"])
    assert "Ironbridge with Meridia" in line and "No relevant records were found for Meridia" in line
    assert "No relevant records" not in ask_compare.instruction(c, [])


# --- ask_grade -------------------------------------------------------------------------------------------------------------------

def _hit(score, reranked=True, doc="d"):
    return types.SimpleNamespace(id=doc, payload={"document_type": "contract", "content": f"text of {doc}", "record_id": doc},
                                 rerank_score=score, combined_score=score, aggregated=0.01, score=score, reranked=reranked)


def test_grade_uses_only_documents_the_cross_encoder_scored(monkeypatch):
    monkeypatch.setitem(ask_grade.THRESHOLDS, MODEL, 0.5)
    hits = [_hit(0.02, reranked=False), _hit(0.3), _hit(0.4)]
    g = ask_grade.grade(hits, MODEL)
    assert g.graded and g.score == 0.4 and not g.relevant and g.threshold == 0.5
    assert ask_grade.grade([_hit(0.9)], MODEL).relevant


def test_no_threshold_or_no_scored_documents_means_ungraded_and_passed_through(monkeypatch):
    assert ask_grade.grade([_hit(0.0)], "some-other-model") == ask_grade.Grade(0.0, None, False, True)
    monkeypatch.setitem(ask_grade.THRESHOLDS, MODEL, 0.5)
    g = ask_grade.grade([_hit(0.01, reranked=False)], MODEL)
    assert not g.graded and g.relevant


def test_retry_queries_are_distinct_bounded_and_keep_names():
    q = "Does our contract with Ironbridge Managed IT Ltd have a force majeure clause?"
    retries = ask_grade.retry_queries(q)
    assert 1 <= len(retries) <= ask_grade.MAX_RETRIES and len(set(retries)) == len(retries) and q not in retries
    assert all("Ironbridge" in r for r in retries)
    assert "IT" in retries[0].split()                       # the acronym is not mistaken for the pronoun
    assert ask_grade.retry_queries("") == []
    assert ask_grade.retry_queries("contract") == []        # nothing to reword


def test_not_found_text_says_how_many_were_searched():
    assert "6 documents" in ask_grade.not_found_text(6) and "1 document " in ask_grade.not_found_text(1)


# --- wired into the pipeline -----------------------------------------------------------------------------------------------------

class _History:
    def __init__(self):
        self.saved = None

    def get_history(self, user_id):
        return []

    def save_history(self, user_id, history):
        self.saved = history


def _pipeline(monkeypatch, by_query, *, corpus=None):
    """A pipeline whose retriever answers from ``by_query`` (a function query -> hits) and whose model call is recorded."""
    from src.services.model_selector import RAGPipeline
    p = RAGPipeline.__new__(RAGPipeline)
    p.default_llm_model = "m"
    p.history_manager = _History()
    p.agent_nick = types.SimpleNamespace(agents={}, get_db_connection=lambda: (_ for _ in ()).throw(RuntimeError("no db")))
    p.searches, p.prompts = [], []

    class Rag:
        embedder = None
        primary_collection = "c"
        uploaded_collection = "u"
        reranker_model_name = MODEL

        def search(self, q, **k):
            p.searches.append((q, k.get("top_k")))
            return by_query(q)

    p.rag = Rag()
    p._session_uploads, p._uploaded_context, p._nltk_processor = {}, {}, None
    p._enable_static_qa, p._static_agent = False, None
    p._citation_guidelines = {"fallback": "nothing found"}

    def compose(self, query, context, draft, nltk, adhoc, corpus_context="", comparison_note=""):
        p.prompts.append({"query": query, "context": context, "corpus": corpus_context, "note": comparison_note})
        return "PROMPT"

    monkeypatch.setattr(type(p), "_compose_llm_prompt", compose)
    monkeypatch.setattr(type(p), "_generate_response", lambda self, *a, **k: {"answer": "model answer", "follow_ups": []})
    monkeypatch.setattr("src.services.model_selector.corpus_facts.fetch_facts", lambda *a, **k: corpus)
    for name in ("_get_cached_response", "_store_cached_response"):
        monkeypatch.setattr(type(p), name, lambda self, *a, **k: None)
    monkeypatch.setattr(type(p), "_build_cache_key", lambda self, *a, **k: None)
    monkeypatch.setattr(type(p), "_try_static_answer", lambda self, *a, **k: None)
    return p


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(analytics_ask, "analytic_answer", lambda *a, **k: None)
    monkeypatch.setattr("src.services.model_selector.record_ask_route", lambda **k: None)
    monkeypatch.setattr("src.services.model_selector.record_action", lambda **k: None)
    monkeypatch.setitem(ask_grade.THRESHOLDS, MODEL, 0.5)
    for name in (ask_flags.GRADE, ask_flags.COMPARE, ask_flags.CONDENSE, ask_flags.ROUTER, ask_flags.EXACT_ID, ask_flags.INPUT_GUARD):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


Q = "Does our contract with Ironbridge Managed IT Ltd have a force majeure clause?"


def test_a_weak_search_is_retried_and_the_better_attempt_is_used(monkeypatch):
    retry = ask_grade.retry_queries(Q)[0]
    p = _pipeline(monkeypatch, lambda q: [_hit(0.9, doc="good")] if q == retry else [_hit(0.1, doc="weak")])
    events = []
    result = p.answer_question(query=Q, user_id="u", trace=True, on_event=lambda k, pl: events.append((k, pl)))
    assert [s[0] for s in p.searches] == [Q, retry]
    assert "text of good" in p.prompts[0]["context"] and "text of weak" not in p.prompts[0]["context"]
    assert any(k == "stage" and pl.get("stage") == "retrying" and pl.get("attempt") == 1 for k, pl in events)
    assert [g["attempt"] for g in result["trace"]["grading"]] == [0, 1] and result["trace"]["retrieval_count"] == 2
    assert "model answer" in result["answer"]


def test_b1_still_weak_after_two_retries_is_an_explicit_not_found_with_no_model_call(monkeypatch):
    p = _pipeline(monkeypatch, lambda q: [_hit(0.1, doc="a"), _hit(0.2, doc="b")])
    result = p.answer_question(query=Q, user_id="u", trace=True)
    assert len(p.searches) == 1 + ask_grade.MAX_RETRIES                     # never more than two retries
    assert p.prompts == []                                                  # nothing was written from weak documents
    assert result["answer"] == ask_grade.not_found_text(2) and result["retrieved_documents"] == []
    assert p.history_manager.saved[-1]["query"] == Q and p.history_manager.saved[-1]["answer"] == result["answer"]
    assert all(g["relevant"] is False for g in result["trace"]["grading"])


def test_database_facts_keep_a_weak_search_from_becoming_not_found(monkeypatch):
    p = _pipeline(monkeypatch, lambda q: [_hit(0.1)], corpus={"intent": "x", "rows": [{"a": 1}]})
    monkeypatch.setattr("src.services.model_selector.corpus_facts.render_facts", lambda facts: "COUNTED ROWS")
    result = p.answer_question(query=Q, user_id="u")
    assert "model answer" in result["answer"] and p.prompts[0]["corpus"] == "COUNTED ROWS"


def test_an_ungraded_reranker_is_never_retried_or_refused(monkeypatch):
    monkeypatch.delitem(ask_grade.THRESHOLDS, MODEL)
    p = _pipeline(monkeypatch, lambda q: [_hit(0.0)])
    assert "model answer" in p.answer_question(query=Q, user_id="u")["answer"] and len(p.searches) == 1


def test_the_flag_turns_grading_off(monkeypatch):
    monkeypatch.setenv(ask_flags.GRADE, "0")
    p = _pipeline(monkeypatch, lambda q: [_hit(0.0)])
    assert "model answer" in p.answer_question(query=Q, user_id="u")["answer"] and len(p.searches) == 1


def test_a_relevant_first_search_is_not_retried(monkeypatch):
    p = _pipeline(monkeypatch, lambda q: [_hit(0.95)])
    p.answer_question(query=Q, user_id="u")
    assert len(p.searches) == 1


def _named(doc, text):
    h = _hit(0.9, doc=doc)
    h.payload["content"] = text
    return h


def test_b2_a_comparison_searches_each_side_and_labels_the_results(monkeypatch):
    def by_query(q):
        return [_named("IMI-MSA", "Ironbridge terms: net 30")] if q.startswith("Ironbridge") else [_named("MCP-Q", "Meridia terms: net 45")]
    p = _pipeline(monkeypatch, by_query)
    result = p.answer_question(query="Compare Ironbridge and Meridia on payment terms", user_id="u", trace=True)
    assert p.searches == [("Ironbridge payment terms", 4), ("Meridia payment terms", 4)]
    ctx = p.prompts[0]["context"]
    assert "Ironbridge:" in ctx and "Meridia:" in ctx and "No relevant records" not in p.prompts[0]["note"]
    assert "compares Ironbridge with Meridia" in p.prompts[0]["note"]
    assert result["trace"]["comparison"] == {"entities": ["Ironbridge", "Meridia"], "documents_found": {"Ironbridge": 1, "Meridia": 1}}


def test_a_side_whose_documents_never_name_it_is_missing_not_filled_with_another_documents_text(monkeypatch):
    """Live B2 check: 'Meridia payment terms' returned three generic policy documents; they must not be labelled Meridia's."""
    def by_query(q):
        if q.startswith("Ironbridge"):
            return [_named("IMI-MSA", "Ironbridge Managed IT Ltd master agreement, net 30")]
        return [_named("POLICY-1", "General payment terms policy: suppliers are paid in 30 days")]
    p = _pipeline(monkeypatch, by_query)
    p.answer_question(query="Compare Ironbridge and Meridia on payment terms", user_id="u")
    assert "No relevant records were found for Meridia" in p.prompts[0]["note"]
    assert "General payment terms policy" not in p.prompts[0]["context"] and "master agreement" in p.prompts[0]["context"]
    assert [s[0] for s in p.searches] == ["Ironbridge payment terms", "Meridia payment terms", "Meridia"]   # one entity-only second try


def test_the_entity_only_second_try_can_rescue_a_side(monkeypatch):
    def by_query(q):
        if q.startswith("Ironbridge"):
            return [_named("IMI-MSA", "Ironbridge agreement")]
        return [_named("MCP-Q", "Meridia Cloud Platforms quote MCP-Q-7740")] if q == "Meridia" else [_named("POLICY-1", "generic policy")]
    p = _pipeline(monkeypatch, by_query)
    p.answer_question(query="Compare Ironbridge and Meridia on payment terms", user_id="u")
    assert "No relevant records" not in p.prompts[0]["note"] and "Meridia:" in p.prompts[0]["context"]


@pytest.mark.parametrize("entity,text,expected", [
    ("Meridia Cloud Platforms", "Quote from Meridia Cloud Platforms Ltd", True),
    ("Meridia Cloud Platforms", "Payment terms policy for all suppliers", False),
    ("Ironbridge Managed IT Ltd", "ironbridge master services agreement", True),
    ("Ironbridge quote", "A quote for office chairs", False),          # 'quote' is a document word, not the entity
    ("quotes", "anything at all", True),                                # nothing distinctive to check: not filtered
])
def test_mentions(entity, text, expected):
    assert ask_compare.mentions(entity, {"content": text}) is expected


def test_the_compare_flag_restores_single_retrieval(monkeypatch):
    monkeypatch.setenv(ask_flags.COMPARE, "0")
    p = _pipeline(monkeypatch, lambda q: [_hit(0.9)])
    p.answer_question(query="Compare Ironbridge and Meridia", user_id="u")
    assert len(p.searches) == 1 and p.prompts[0]["note"] == ""


def test_no_threshold_is_shipped_because_the_reranker_does_not_separate_answerable_from_absent(monkeypatch):
    """Grading is inert by design until a threshold is calibrated (see the note in ask_grade.THRESHOLDS)."""
    monkeypatch.delitem(ask_grade.THRESHOLDS, MODEL)        # the autouse fixture's test-only value
    assert ask_grade.THRESHOLDS == {}
    assert ask_grade.grade([_hit(0.0)], "BAAI/bge-reranker-large").graded is False
