from src.services.negotiation_advice import store as st


class _Cur:
    def __init__(self, rec, rows=None):
        self._rec = rec
        self._rows = rows or []
        self.description = []

    def execute(self, sql, params=()):
        self._rec.append((sql, params))
        for needle, (cols, rows) in (self._rows or {}).items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._matched = list(rows)
                return
        self.description = []
        self._matched = []

    def fetchall(self):
        return getattr(self, "_matched", [])

    def fetchone(self):
        m = getattr(self, "_matched", [])
        return m[0] if m else None


class _Conn:
    def __init__(self, rows=None):
        self.rec = []
        self._cur = _Cur(self.rec, rows)
        self.committed = False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True


def test_save_advice_inserts_and_returns_an_id():
    conn = _Conn()
    out = st.save_advice(conn, deal_id="D-1", supplier_id="SUP-1",
                         quadrant="Leverage", quadrant_source="computed",
                         quadrant_confidence=0.8, style="Competitive",
                         style_source="computed", signals={"deal_value": 1},
                         plays=[{"lever": "Commercial"}], created_by="buyer")
    assert out["advice_id"]
    assert out["quadrant"] == "Leverage"
    sql = " ".join(s for s, _ in conn.rec)
    assert "INSERT INTO proc.bp_negotiation_advice" in sql
    assert conn.committed


def test_state_fact_records_provenance():
    conn = _Conn()
    st.state_fact(conn, advice_id="A-1", fact_key="alternative_supplier_count",
                  fact_value="2", stated_by="buyer")
    sql = " ".join(s for s, _ in conn.rec)
    assert "INSERT INTO proc.bp_negotiation_advice_fact" in sql
    assert conn.committed


def test_withdraw_sets_a_timestamp_rather_than_deleting():
    conn = _Conn()
    st.withdraw_fact(conn, advice_id="A-1", fact_key="alternative_supplier_count")
    sql = " ".join(s for s, _ in conn.rec)
    assert "withdrawn_at" in sql
    assert "DELETE" not in sql.upper()


def test_active_facts_excludes_withdrawn():
    conn = _Conn({"FROM proc.bp_negotiation_advice_fact":
                  (["fact_key", "fact_value"],
                   [("alternative_supplier_count", "2")])})
    facts = st.active_facts(conn, "A-1")
    assert facts == {"alternative_supplier_count": "2"}
    sql = " ".join(s for s, _ in conn.rec)
    assert "withdrawn_at IS NULL" in sql
