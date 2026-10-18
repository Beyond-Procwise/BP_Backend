"""Phase 7: date maths in code (F2) and PO status read from several sources at once (F1; F3 is not built)."""
import datetime as dt
import threading
import time
import types

import pytest

from src.services import ask_dates, ask_exact_id, ask_flags, ask_po_status
from src.services.analytics import ask as analytics_ask

NOW = dt.date(2026, 10, 18)


# --- ask_dates -------------------------------------------------------------------------------------------------------------------

def test_f2_a_contract_signed_15_june_2024_with_a_21_month_term_expired_on_2026_03_15():
    f = ask_dates.from_question("The contract was signed on 15 June 2024 with a 21 month term. Has it expired?", NOW)
    assert f.expiry == dt.date(2026, 3, 15) and f.expired and f.days_from_today == -217
    text = ask_dates.render(f)
    assert "Expiry date: 2026-03-15" in text and "EXPIRED" in text and "217 days before today" in text and "2026-10-18" in text


@pytest.mark.parametrize("q,expiry", [
    ("Signed June 15, 2024, term 3 years", dt.date(2027, 6, 15)),
    ("It started 2024-06-15 and runs for 21 months", dt.date(2026, 3, 15)),
    ("effective 1st March 2026 for 12 months", dt.date(2027, 3, 1)),
    ("signed 31 January 2025, 1 month term", dt.date(2025, 2, 28)),           # clamped to the end of a shorter month
    ("commenced 29 February 2024 for 1 year", dt.date(2025, 2, 28)),
    ("starts 1 Oct 2026 for 90 days", dt.date(2026, 12, 30)),
])
def test_expiry_from_a_date_and_a_term(q, expiry):
    assert ask_dates.from_question(q, NOW).expiry == expiry


def test_a_future_expiry_is_not_expired():
    f = ask_dates.from_question("signed 1 March 2026 for 24 months", NOW)
    assert not f.expired and "NOT YET EXPIRED" in ask_dates.render(f)


@pytest.mark.parametrize("q", [
    "Signed 03/04/2026 for 12 months",                    # numeric day/month order is not guessed
    "When was the contract signed? Term 12 months.",      # no date
    "signed 1 March 2026 and again 5 May 2026 for 12 months",   # two dates: which is the start?
    "signed 1 March 2026 for 12 months or 24 months",     # two terms
    "The 12 month plan", "", None,
])
def test_anything_ambiguous_is_left_alone(q):
    assert ask_dates.from_question(q, NOW) is None
    assert ask_dates.render(None) == ""


def test_a_contract_record_uses_its_end_date_else_start_plus_term():
    a = ask_dates.from_contract_row({"contract_start_date": dt.date(2026, 3, 1), "contract_end_date": dt.date(2027, 2, 28)}, NOW)
    assert a.expiry == dt.date(2027, 2, 28) and not a.expired and a.source == "the contract record"
    b = ask_dates.from_contract_row({"contract_start_date": dt.date(2024, 6, 15), "contract_end_date": None, "term_months": 21}, NOW)
    assert b.expiry == dt.date(2026, 3, 15) and b.expired
    assert ask_dates.from_contract_row({"contract_start_date": None, "contract_end_date": None}, NOW) is None


def test_add_months_clamps_and_crosses_years():
    assert ask_dates.add_months(dt.date(2025, 11, 30), 3) == dt.date(2026, 2, 28)
    assert ask_dates.add_months(dt.date(2024, 1, 31), 1) == dt.date(2024, 2, 29)
    assert ask_dates.add_months(dt.date(2024, 12, 15), 12) == dt.date(2025, 12, 15)


# --- ids -------------------------------------------------------------------------------------------------------------------------

def test_contract_ids_are_extracted_without_clashing_with_po_ids():
    assert ask_exact_id.extract("When does SA-2026-0310 expire?") == {"contract": ["SA-2026-0310"]}
    assert ask_exact_id.extract("contract C00003 and PO-2025-0257") == {"po": ["PO-2025-0257"], "contract": ["C00003"]}
    assert ask_exact_id.extract("Is there a PO-2025-0257 for OF-2026-0117?") == {"po": ["PO-2025-0257"], "contract": ["OF-2026-0117"]}


# --- a fake database -------------------------------------------------------------------------------------------------------------

class FakeDB:
    """Hands out one connection per call; each query is answered from ``tables`` by a fragment of its SQL."""

    def __init__(self, tables, *, delay=None, fail=()):
        self.tables, self.delay, self.fail = tables, delay or {}, set(fail)
        self.queries, self.threads, self._lock = [], set(), threading.Lock()

    def __call__(self):
        return _Conn(self)


class _Conn:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _Cur(self.db)


class _Cur:
    def __init__(self, db):
        self.db, self.description, self.rows = db, [], []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        with self.db._lock:
            self.db.queries.append((sql, params))
            self.db.threads.add(threading.current_thread().name)
        key = next((k for k in self.db.tables if k in sql), None)
        if key in self.db.fail:
            raise RuntimeError("connection reset")
        time.sleep(self.db.delay.get(key, 0))
        cols, rows = self.db.tables.get(key, ([], []))
        self.description = [(c,) for c in cols]
        wanted = params[0].upper() if params else None
        self.rows = [r for r in rows if wanted is None or str(r[0]).upper() == wanted or key not in ("bp_purchase_order_trgt",) or True]

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


def _tables():
    return {
        "FROM proc.bp_purchase_order_trgt WHERE": (["po_id", "supplier_name", "currency", "total_amount"], [("PO-2025-0257", "Ironbridge Managed IT Ltd", "GBP", "1432980.00")]),
        "FROM proc.bp_invoice_trgt": (["invoice_id", "invoice_date", "invoice_status"], [("IMI-INV-Q2", "2025-07-01", "Paid"), ("IMI-INV-Q3", "2025-10-01", "Open")]),
        "FROM proc.bp_goods_receipt_trgt": (["grn_id", "receipt_date"], []),
        "FROM proc.bp_approval a": (["approval_id", "status", "decision"], [("A1", "closed", "approved")]),
    }


# --- ask_po_status ---------------------------------------------------------------------------------------------------------------

def test_status_questions_are_told_from_what_a_po_is():
    assert ask_po_status.is_status_question("Where is PO-2025-0257?", ["PO-2025-0257"])
    assert ask_po_status.is_status_question("Has PO-2025-0257 been approved?", ["PO-2025-0257"])
    assert not ask_po_status.is_status_question("What is the supplier on PO-2025-0257?", ["PO-2025-0257"])
    assert not ask_po_status.is_status_question("Where is the approval policy?", [])


def test_the_sources_are_read_in_parallel_and_joined():
    db = FakeDB(_tables(), delay={k: 0.3 for k in _tables()})
    started = time.perf_counter()
    res = ask_po_status.gather(db, "PO-2025-0257", timeout_s=3)
    took = time.perf_counter() - started
    assert took < 0.9, took                                              # four 0.3 s reads took about one
    assert len(db.threads) == 4 and {r["state"] for r in res.values()} == {"ok", "empty"}
    assert res["Goods receipts"]["state"] == "empty" and len(res["Invoices"]["rows"]) == 2


def test_a_slow_source_is_named_missing_and_the_rest_still_answer():
    db = FakeDB(_tables(), delay={"FROM proc.bp_invoice_trgt": 2.0})
    started = time.perf_counter()
    res = ask_po_status.gather(db, "PO-2025-0257", timeout_s=0.3)
    assert time.perf_counter() - started < 1.0                           # did not wait for the slow one
    assert res["Invoices"]["state"] == "unavailable" and "0.3 seconds" in res["Invoices"]["why"]
    assert res["Purchase order record"]["state"] == "ok" and res["Approvals"]["state"] == "ok"
    text = ask_po_status.render("PO-2025-0257", res)
    assert "NOT AVAILABLE" in text and "Invoices (did not answer within 0.3 seconds)" in text
    assert "Ironbridge Managed IT Ltd" in text and "Goods receipts: nothing recorded" in text


def test_a_failing_source_is_unavailable_not_empty():
    db = FakeDB(_tables(), fail={"FROM proc.bp_goods_receipt_trgt"})
    res = ask_po_status.gather(db, "PO-2025-0257", timeout_s=2)
    assert res["Goods receipts"]["state"] == "unavailable"
    text = ask_po_status.render("PO-2025-0257", res)
    assert "Goods receipts (could not be read)" in text and "Goods receipts: nothing recorded" not in text


def test_carrier_tracking_is_reported_as_not_connected_never_invented():
    text = ask_po_status.render("PO-2025-0257", ask_po_status.gather(FakeDB(_tables()), "PO-2025-0257", timeout_s=2))
    assert "Carrier tracking: not connected" in text


# --- wired into the pipeline -----------------------------------------------------------------------------------------------------

class _History:
    def __init__(self):
        self.saved = None

    def get_history(self, user_id):
        return []

    def save_history(self, user_id, history):
        self.saved = history


def _pipeline(monkeypatch, db):
    from src.services.model_selector import RAGPipeline
    p = RAGPipeline.__new__(RAGPipeline)
    p.default_llm_model = "m"
    p.history_manager = _History()
    p.agent_nick = types.SimpleNamespace(agents={}, get_db_connection=db)
    p.searches, p.prompts = [], []

    class Rag:
        embedder = None
        primary_collection = "c"
        uploaded_collection = "u"
        reranker_model_name = "x"

        def search(self, q, **k):
            p.searches.append(q)
            return []

    p.rag = Rag()
    p._session_uploads, p._uploaded_context, p._nltk_processor = {}, {}, None
    p._enable_static_qa, p._static_agent = False, None
    p._citation_guidelines = {"fallback": "nothing found"}

    def compose(self, query, context, draft, nltk, adhoc, corpus_context="", comparison_note=""):
        p.prompts.append(corpus_context)
        return "PROMPT"

    monkeypatch.setattr(type(p), "_compose_llm_prompt", compose)
    monkeypatch.setattr(type(p), "_generate_response", lambda self, *a, **k: {"answer": "ok", "follow_ups": []})
    monkeypatch.setattr("src.services.model_selector.corpus_facts.fetch_facts", lambda *a, **k: None)
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
    monkeypatch.setattr(ask_dates, "today", lambda: NOW)
    for name in (ask_flags.DATE_FACTS, ask_flags.PO_STATUS, ask_flags.EXACT_ID, ask_flags.ROUTER, ask_flags.CONDENSE, ask_flags.INPUT_GUARD):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


def test_f2_the_prompt_carries_the_computed_expiry_and_retrieval_still_runs(monkeypatch):
    p = _pipeline(monkeypatch, FakeDB({}))
    p.answer_question(query="The contract was signed on 15 June 2024 with a 21 month term. Has it expired?", user_id="u")
    assert "Expiry date: 2026-03-15" in p.prompts[0] and "EXPIRED" in p.prompts[0] and "today is 2026-10-18" in p.prompts[0]
    assert len(p.searches) == 1


def test_the_date_flag_turns_it_off(monkeypatch):
    monkeypatch.setenv(ask_flags.DATE_FACTS, "0")
    p = _pipeline(monkeypatch, FakeDB({}))
    p.answer_question(query="The contract was signed on 15 June 2024 with a 21 month term. Has it expired?", user_id="u")
    assert p.prompts == [] or "Expiry date" not in "".join(p.prompts)


def test_f1_a_po_status_question_is_read_from_every_source_without_vector_search(monkeypatch):
    db = FakeDB(_tables())
    p = _pipeline(monkeypatch, db)
    result = p.answer_question(query="Where is PO-2025-0257 up to? Has it been approved?", user_id="u", trace=True)
    ctx = p.prompts[0]
    for needle in ("Purchase order record:", "Invoices:", "Goods receipts: nothing recorded", "Approvals:", "Carrier tracking: not connected"):
        assert needle in ctx, needle
    assert p.searches == [] and result["trace"]["retrieval_count"] == 0
    assert len(db.queries) == 4                                          # one read per source (parallelism is proved by the timing test)


def test_a_source_that_fails_is_named_in_the_prompt_and_the_answer_still_goes_ahead(monkeypatch):
    p = _pipeline(monkeypatch, FakeDB(_tables(), fail={"FROM proc.bp_invoice_trgt"}))
    p.answer_question(query="What is the status of PO-2025-0257?", user_id="u")
    assert "NOT AVAILABLE" in p.prompts[0] and "Invoices (could not be read)" in p.prompts[0] and "Ironbridge Managed IT Ltd" in p.prompts[0]


def test_a_po_that_is_not_known_falls_back_to_normal_retrieval(monkeypatch):
    empty = {k: (c, []) for k, (c, _) in _tables().items()}
    p = _pipeline(monkeypatch, FakeDB(empty))
    p.answer_question(query="What is the status of PO-2025-9999?", user_id="u")
    assert p.searches                                                   # retrieval ran


def test_the_po_flag_falls_back_to_the_plain_lookup(monkeypatch):
    monkeypatch.setenv(ask_flags.PO_STATUS, "0")
    db = FakeDB({"bp_purchase_order_trgt": (["po_id", "supplier_name"], [("PO-2025-0257", "Ironbridge Managed IT Ltd")]),
                 "bp_po_line_items_trgt": (["po_id", "line_number"], [])})
    p = _pipeline(monkeypatch, db)
    p.answer_question(query="What is the status of PO-2025-0257?", user_id="u")
    assert "Carrier tracking" not in p.prompts[0] and "Purchase order PO-2025-0257" in p.prompts[0]


def test_a_contract_id_gets_its_expiry_worked_out_from_the_record(monkeypatch):
    db = FakeDB({"FROM proc.bp_contracts": (
        ["contract_id", "contract_title", "contract_start_date", "contract_end_date"],
        [("SA-2026-0310", "Service Agreement", dt.date(2026, 3, 1), dt.date(2027, 2, 28))])})
    p = _pipeline(monkeypatch, db)
    p.answer_question(query="When does SA-2026-0310 expire?", user_id="u")
    assert "Expiry date: 2027-02-28" in p.prompts[0] and "NOT YET EXPIRED" in p.prompts[0] and p.searches == []
