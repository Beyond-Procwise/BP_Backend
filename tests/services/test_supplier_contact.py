"""One place answers "how do we reach this supplier".

Before this there were four: the dispatch guard queried proc.bp_supplier by
supplier_id, and the drafting agent took an address from whatever the ranking
payload happened to carry, then from two more fallbacks. Three of those four
could be influenced by data flowing through the workflow; only the first read
the supplier master. These tests pin the one that survives.
"""

from __future__ import annotations

import pytest

from src.services import supplier_contact


class _Conn:
    """A connection that answers the way psycopg2 does."""

    def __init__(self, row):
        self._row = row
        self.executed = []

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return [] if self._row is None else [self._row]


def test_resolves_both_addresses_and_the_name_from_the_master():
    conn = _Conn(("first@supplier.test", "second@supplier.test", "Dana Okafor"))

    contact = supplier_contact.resolve_contact(conn, "SUP-1")

    assert contact.emails == ["first@supplier.test", "second@supplier.test"]
    assert contact.name == "Dana Okafor"


def test_queries_the_supplier_master_keyed_on_supplier_id():
    conn = _Conn(("a@supplier.test", None, None))

    supplier_contact.resolve_contact(conn, "SUP-7")

    sql, params = conn.executed[0]
    assert "proc.bp_supplier" in sql
    assert "supplier_id = %s" in sql
    assert params == ("SUP-7",)


def test_blank_and_missing_values_are_dropped_not_returned_as_empty_strings():
    conn = _Conn(("  real@supplier.test  ", "   ", None))

    contact = supplier_contact.resolve_contact(conn, "SUP-2")

    assert contact.emails == ["real@supplier.test"]
    assert contact.name is None


def test_no_supplier_id_resolves_to_nothing_rather_than_querying():
    conn = _Conn(("should@not.be.read", None, None))

    contact = supplier_contact.resolve_contact(conn, None)

    assert contact.emails == []
    assert contact.name is None
    assert conn.executed == []


def test_a_supplier_absent_from_the_master_resolves_to_nothing():
    contact = supplier_contact.resolve_contact(_Conn(None), "SUP-UNKNOWN")

    assert contact.emails == []
    assert contact.name is None


def test_a_test_connection_may_answer_the_lookup_directly():
    """The seam the existing guard tests already rely on."""

    class _Fake:
        def lookup_supplier_emails(self, supplier_id):
            return ["seam@supplier.test"]

    contact = supplier_contact.resolve_contact(_Fake(), "SUP-3")

    assert contact.emails == ["seam@supplier.test"]


def test_a_database_failure_resolves_to_nothing_rather_than_raising():
    """No address is a held draft. An exception here would be a send path crash."""

    class _Broken:
        def cursor(self):
            raise RuntimeError("connection reset")

    contact = supplier_contact.resolve_contact(_Broken(), "SUP-4")

    assert contact.emails == []
    assert contact.name is None
