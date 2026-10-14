"""A database without the supplier crosswalk still gets its expiry alerts."""
import pytest

from src.services.contract_expiry import detector


class _Cur:
    def __init__(self, fail_on_join):
        self.fail_on_join, self.sql, self.description = fail_on_join, [], [("contract_id",)]
        self.connection = type("C", (), {"rolled_back": 0, "rollback": lambda s: None})()

    def execute(self, sql, *a):
        self.sql.append(sql)
        if self.fail_on_join and "bp_supplier_id_crosswalk" in sql:
            raise RuntimeError('relation "proc.bp_supplier_id_crosswalk" does not exist')

    def fetchall(self):
        return [("C1",)]


def test_falls_back_to_the_plain_query_when_the_crosswalk_is_missing():
    cur = _Cur(fail_on_join=True)
    rows = detector._load_contracts(cur, with_supplier=True)
    assert rows == [{"contract_id": "C1"}]
    assert "bp_supplier_id_crosswalk" not in cur.sql[-1]


def test_other_errors_are_not_swallowed():
    cur = _Cur(fail_on_join=False)
    cur.execute = lambda sql, *a: (_ for _ in ()).throw(RuntimeError("connection reset"))
    with pytest.raises(RuntimeError, match="connection reset"):
        detector._load_contracts(cur, with_supplier=True)


def test_the_recording_job_never_asks_for_supplier_names():
    cur = _Cur(fail_on_join=False)
    detector._load_contracts(cur, with_supplier=False)
    assert all("crosswalk" not in q for q in cur.sql)
