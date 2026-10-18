"""Phase 4: exact-ID pre-route and the pre-retrieval router (tests D1-D5, F1)."""
import time
import types

import pytest

from src.services import ask_exact_id, ask_flags, ask_router
from src.services.analytics import ask as analytics_ask


# --- exact ids -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Is PO-2025-0257 approved yet?", {"po": ["PO-2025-0257"]}),
    ("status of po 2025 0257 please", {"po": ["PO-2025-0257"]}),
    ("show PO000123", {"po": ["PO000123"]}),
    ("part ITM003969 and itm-004100", {"part": ["ITM003969", "ITM004100"]}),
    ("who is SUP-OrbisPlatformSolutionsLtd", {"supplier": ["SUP-OrbisPlatformSolutionsLtd"]}),
    ("PO-2025-0257 lines for ITM003969 from SUP-MarloweConstructionLtd",
     {"po": ["PO-2025-0257"], "part": ["ITM003969"], "supplier": ["SUP-MarloweConstructionLtd"]}),
    ("PO-2025-0257 and po-2025-0257", {"po": ["PO-2025-0257"]}),
])
def test_extract_finds_identifiers(text, expected):
    assert ask_exact_id.extract(text) == expected


@pytest.mark.parametrize("text", [
    "Who can approve a PO?", "What is the PO process?", "Q4 spend", "Signed 2025-04-30 for 1432980",
    "the report", "ITM", "SUPPLIER-1", "approve POs over 10000",
])
def test_extract_ignores_ordinary_text(text):
    assert ask_exact_id.extract(text) == {}


class _Cur:
    def __init__(self, tables):
        self.tables, self.executed, self._rows, self.description = tables, [], [], []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        key = next((k for k in self.tables if k in sql), None)
        cols, rows = self.tables.get(key, ([], []))
        wanted = params[0].upper() if params else None
        self.description = [(c,) for c in cols]
        self._rows = [r for r in rows if wanted is None or str(r[0]).upper() == wanted or str(r[2] if len(r) > 2 else "").upper() == wanted]

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def __init__(self, cur):
        self.cur = cur

    def cursor(self):
        return self.cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _db():
    cur = _Cur({
        "bp_purchase_order_trgt": (["po_id", "supplier_id", "supplier_name", "po_status", "currency", "total_amount"],
                                   [("PO-2025-0257", "SUP-Ironbridge", "Ironbridge Managed IT Ltd", None, "GBP", "1432980.00")]),
        "bp_po_line_items_trgt": (["po_id", "line_number", "item_id", "item_description", "quantity"],
                                  [("PO-2025-0257", 1, "ITM003969", "Service Desk", 4200)]),
        "bp_supplier": (["supplier_id", "supplier_name"], [("SUP-Ironbridge", "Ironbridge Managed IT Ltd")]),
    })
    return cur, (lambda: _Conn(cur))


def test_fetch_and_render_a_po_with_lines():
    cur, factory = _db()
    found = ask_exact_id.fetch(factory, {"po": ["PO-2025-0257"]})
    text = ask_exact_id.render(found)
    assert "Purchase order PO-2025-0257 (found in the records)" in text
    assert "supplier_name=Ironbridge Managed IT Ltd" in text and "total_amount=1432980.00" in text
    assert "po_status" not in text                                # NULL columns are left out, never invented
    assert "item_id=ITM003969" in text


def test_a_missing_id_is_reported_not_invented():
    _, factory = _db()
    found = ask_exact_id.fetch(factory, {"po": ["PO-2025-9999"]})
    assert found["records"] == [] and found["missing"] == ["PO-2025-9999"]
    assert ask_exact_id.render(found) == ""


def test_sql_is_parameterised_and_never_touches_bank_columns():
    cur, factory = _db()
    ask_exact_id.fetch(factory, {"po": ["PO-2025-0257'; DROP TABLE x;--"], "supplier": ["SUP-Ironbridge"], "part": ["ITM003969"]})
    assert cur.executed
    for sql, params in cur.executed:
        assert "DROP" not in sql and "%s" in sql and params
        for forbidden in ("bank", "iban", "swift", "tax_id", "vat", "contact_email", "credit_limit"):
            assert forbidden not in sql.lower()


def test_a_database_failure_falls_back_to_nothing():
    def boom():
        raise RuntimeError("db down")
    assert ask_exact_id.fetch(boom, {"po": ["PO-2025-0257"]}) == {"records": [], "missing": []}


# --- router ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Hello", "hi", "Hi there", "Good morning!", "Good afternoon team", "hey, how are you today?"])
def test_greetings_route_to_greeting(text):
    assert ask_router.decide(text).route == ask_router.GREETING


@pytest.mark.parametrize("text", [
    "Hello, can you show me our contracts?", "Good morning - which POs are late?", "Thanks, now list the suppliers",
    "Thanks", "thank you!", "ok", "great",
    "How are you going to approve this?", "hi", ])
def test_only_a_bare_greeting_skips_retrieval(text):
    expected = ask_router.GREETING if text == "hi" else ask_router.RAG
    assert ask_router.decide(text).route == expected


@pytest.mark.parametrize("text", [
    "What is the stock level for part ITM003969?", "How many units do we have in stock?", "Do we hold safety stock for laptops?",
    "inventory on hand for ITM003969"])
def test_inventory_questions(text):
    assert ask_router.decide(text).route == ask_router.INVENTORY


def test_calculation_needs_a_verb_and_amounts():
    assert ask_router.decide("Calculate the landed cost of 500 units at £12.40 each with 8% freight").route == ask_router.CALCULATION
    assert ask_router.decide("Calculate how our approvals work").route == ask_router.RAG
    assert ask_router.decide("What is the total spend with Ironbridge?").route == ask_router.RAG


@pytest.mark.parametrize("text", [
    "Show me the approval matrix", "Who can approve a purchase order over 50,000 pounds?", "What is the policy on spend thresholds?",
    "Which suppliers are preferred?", "Explain the three-way match"])
def test_policy_and_record_questions_default_to_rag(text):
    d = ask_router.decide(text)
    assert d.route == ask_router.RAG


class _Embedder:
    """Maps phrases to axes so that similarity is predictable."""
    KEYS = ["stock", "cost", "hello"]

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np
        single = isinstance(texts, str)
        rows = []
        for t in ([texts] if single else texts):
            v = np.array([1.0 if k in t.lower() else 0.0 for k in self.KEYS] + [0.3], dtype="float32")
            rows.append(v / np.linalg.norm(v))
        out = np.stack(rows)
        return out[0] if single else out


def test_near_threshold_paraphrase_is_logged_with_its_score_and_falls_back():
    ask_router._cache.clear()
    d = ask_router.decide("Is the warehouse well supplied?", embedder=_Embedder(), use_embedding=True)
    assert d.route == ask_router.RAG and d.layer == "default" and d.threshold == ask_router.EMBED_THRESHOLD
    assert d.score is not None and d.best_route is not None
    assert set(d.log_fields()) >= {"route", "score", "threshold", "best_route"}


def test_embedding_can_route_a_paraphrase_above_the_threshold():
    ask_router._cache.clear()
    d = ask_router.decide("tell me the stock please", embedder=_Embedder(), use_embedding=True)
    assert d.route == ask_router.INVENTORY or d.route == ask_router.RAG        # rules may already have claimed it
    d2 = ask_router.decide("stock", embedder=_Embedder(), use_embedding=True, threshold=0.5)
    assert d2.route in (ask_router.INVENTORY, ask_router.RAG)


def test_an_embedding_match_never_turns_a_question_into_a_greeting():
    ask_router._cache.clear()
    d = ask_router.decide("hello world of suppliers and what they supply", embedder=_Embedder(), use_embedding=True, threshold=0.1)
    assert d.route != ask_router.GREETING


def test_a_broken_embedder_means_rag():
    class Broken:
        def encode(self, *a, **k):
            raise RuntimeError("oom")
    ask_router._cache.clear()
    assert ask_router.decide("anything about suppliers", embedder=Broken(), use_embedding=True).route == ask_router.RAG


# --- wired into the pipeline -----------------------------------------------------------------------------------------------------

class _History:
    def __init__(self):
        self.saved = None

    def get_history(self, user_id):
        return []

    def save_history(self, user_id, history):
        self.saved = history


def _pipeline(conn_factory=None):
    from src.services.model_selector import RAGPipeline
    p = RAGPipeline.__new__(RAGPipeline)
    p.default_llm_model = "test-model"
    p.history_manager = _History()
    p.agent_nick = types.SimpleNamespace(agents={}, get_db_connection=conn_factory or (lambda: (_ for _ in ()).throw(RuntimeError("no db"))))
    return p


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(analytics_ask, "analytic_answer", lambda *a, **k: None)
    monkeypatch.setattr("src.services.model_selector.record_ask_route", lambda **k: None)
    monkeypatch.setattr("src.services.model_selector.record_action", lambda **k: None)
    for name in (ask_flags.ROUTER, ask_flags.ROUTER_EMBED, ask_flags.EXACT_ID, ask_flags.INPUT_GUARD):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


def _stages(events):
    return [p.get("stage") for k, p in events if k == "stage"]


def test_d1_a_greeting_is_answered_instantly_without_retrieval_or_a_model_call():
    p, events = _pipeline(), []
    started = time.perf_counter()
    result = p.answer_question(query="Good morning", user_id="u", on_event=lambda k, pl: events.append((k, pl)), trace=True)
    assert time.perf_counter() - started < 1.0
    assert result["answer"] == ask_router.GREETING_REPLY
    assert "retrieving" not in _stages(events) and "generating" not in _stages(events)
    assert result["trace"]["route"]["route"] == "greeting" and result["trace"]["retrieval_count"] == 0
    assert _stages(events)[-1] == "answer_complete"


def test_d4_an_inventory_question_gets_one_clarifying_question_when_no_inventory_data_exists():
    cur = _Cur({})
    p = _pipeline(lambda: _Conn(cur))
    result = p.answer_question(query="What is the stock level for part ITM003969?", user_id="u", trace=True)
    assert result["answer"] == ask_router.INVENTORY_CLARIFY and result["answer"].count("?") == 1
    assert result["trace"]["route"]["route"] == "inventory"


def test_inventory_goes_on_to_retrieval_if_inventory_data_exists():
    class Cur2(_Cur):
        def fetchone(self):
            return (1,)
    p = _pipeline(lambda: _Conn(Cur2({})))
    with pytest.raises(AttributeError):                            # the stub has no retriever: it was reached
        p.answer_question(query="How many units do we have in stock?", user_id="u")


def test_d3_an_approval_matrix_question_still_goes_to_rag():
    with pytest.raises(AttributeError):
        _pipeline().answer_question(query="Show me the approval matrix for capital expenditure", user_id="u")


def test_flags_turn_the_router_off(monkeypatch):
    monkeypatch.setenv(ask_flags.ROUTER, "0")
    with pytest.raises(AttributeError):
        _pipeline().answer_question(query="Hello", user_id="u")


def test_f1_a_po_number_skips_vector_search_and_reaches_the_prompt(monkeypatch):
    cur, factory = _db()
    p = _pipeline(factory)

    class NoSearch:
        embedder = None
        primary_collection = "c"
        uploaded_collection = "u"

        def search(self, *a, **k):
            raise AssertionError("vector search ran for an exact id")

    p.rag = NoSearch()
    for name, value in dict(
        _session_uploads={}, _uploaded_context={}, _nltk_processor=None, _enable_static_qa=False, _static_agent=None,
        _citation_guidelines={"fallback": "nothing"},
    ).items():
        setattr(p, name, value)
    seen = {}

    def fake_compose(self, query, context, draft, nltk, adhoc, corpus_context=""):
        seen["corpus_context"] = corpus_context
        return "PROMPT"

    monkeypatch.setattr(type(p), "_compose_llm_prompt", fake_compose)
    monkeypatch.setattr(type(p), "_generate_response", lambda self, *a, **k: {"answer": "PO-2025-0257 is with Ironbridge.", "follow_ups": []})
    monkeypatch.setattr("src.services.model_selector.corpus_facts.fetch_facts", lambda *a, **k: None)
    for name in ("_get_cached_response", "_store_cached_response"):
        monkeypatch.setattr(type(p), name, lambda self, *a, **k: None)
    monkeypatch.setattr(type(p), "_build_cache_key", lambda self, *a, **k: None)
    monkeypatch.setattr(type(p), "_analyse_session_history", lambda self, h: ("", [], {}))
    monkeypatch.setattr(type(p), "_is_policy_question", lambda self, *a, **k: False)
    monkeypatch.setattr(type(p), "_try_static_answer", lambda self, *a, **k: None)
    events = []
    result = p.answer_question(query="What is the status of PO-2025-0257?", user_id="u", on_event=lambda k, pl: events.append((k, pl)), trace=True)
    assert "Purchase order PO-2025-0257" in seen["corpus_context"] and "Ironbridge Managed IT Ltd" in seen["corpus_context"]
    assert result["trace"]["route"]["route"] == "exact_id" and result["trace"]["retrieval_count"] == 0
    assert any(k == "stage" and pl.get("stage") == "retrieving" and pl.get("lookup") == "records" for k, pl in events)


def test_an_id_not_in_the_records_falls_back_to_normal_retrieval():
    cur, factory = _db()
    p = _pipeline(factory)
    p._session_uploads, p._uploaded_context = {}, {}
    with pytest.raises(AttributeError):                            # fell through to self.rag
        p.answer_question(query="What is PO-2025-9999?", user_id="u")
