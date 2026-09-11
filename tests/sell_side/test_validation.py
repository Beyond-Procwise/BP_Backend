"""Validation guards that raise before any cursor is used, so no database is needed.

Every case below calls the function with conn=None: if the guard did not fire before the
first `dict_cursor(conn)` / `conn.cursor(...)` call, the test would blow up with an
AttributeError on None rather than the ValueError it asserts -- that failure mode is itself
proof the guard runs first.
"""
import datetime as dt
from decimal import Decimal as D

import pytest

from src.services.sell_side import accounts, opportunities as opp, quotes

ACCOUNT_GUARDS = [
    pytest.param(
        lambda: accounts.create_account(None, account_name="Foo", bogus_field="x"),
        "bogus_field", id="create_account-unknown-field"),
    pytest.param(
        lambda: accounts.create_account(None, account_name="   "),
        "account_name", id="create_account-blank-name"),
    pytest.param(
        lambda: accounts.create_account(None, account_name="Foo", default_currency="nope"),
        "currency code", id="create_account-invalid-default-currency"),
    pytest.param(
        lambda: accounts.add_contact(None, "ACC-1", contact_name="  "),
        "contact_name", id="add_contact-blank-name"),
    pytest.param(
        lambda: accounts.set_history_scope(None, "ACC-1", source_kind="our_invoices",
                                           completeness="bogus"),
        "completeness", id="set_history_scope-invalid-completeness"),
]


@pytest.mark.parametrize("call, match", ACCOUNT_GUARDS)
def test_an_accounts_guard_raises_before_any_cursor_use(call, match):
    with pytest.raises(ValueError, match=match):
        call()


OPPORTUNITY_GUARDS = [
    pytest.param(
        lambda: opp.create_opportunity(None, account_id="LIVETEST-X",
                                       opportunity_type="bogus", currency="GBP"),
        "opportunity_type", id="create_opportunity-invalid-type"),
    pytest.param(
        lambda: opp.create_opportunity(None, account_id="LIVETEST-X",
                                       opportunity_type="upsell", currency="GBP",
                                       phase_id="sales.margin",
                                       subprocess_id="sales.approval.pricing-approval"),
        "belongs to", id="create_opportunity-invalid-phase-subprocess-pair"),
    pytest.param(
        lambda: opp.create_opportunity(None, account_id="LIVETEST-X",
                                       opportunity_type="upsell", currency="GBP",
                                       expected_quantity=D("0")),
        "expected_quantity", id="create_opportunity-quantity-not-positive"),
    pytest.param(
        lambda: opp.create_opportunity(None, account_id="LIVETEST-X",
                                       opportunity_type="upsell", currency="GBP",
                                       expected_unit_price=D("-1")),
        "expected_unit_price", id="create_opportunity-negative-unit-price"),
    pytest.param(
        lambda: opp.add_justification(None, 1, kind="bogus", claim="hi"),
        "kind", id="add_justification-invalid-kind"),
]


@pytest.mark.parametrize("call, match", OPPORTUNITY_GUARDS)
def test_an_opportunities_guard_raises_before_any_cursor_use(call, match):
    with pytest.raises(ValueError, match=match):
        call()


QUOTE_GUARDS = [
    pytest.param(
        lambda: quotes.create_draft(
            None, account_id="X", currency="GBP",
            valid_until=dt.date.today() + dt.timedelta(days=1), lines=[], created_by="a"),
        "at least one line", id="create_draft-no-lines"),
]


@pytest.mark.parametrize("call, match", QUOTE_GUARDS)
def test_a_quotes_guard_raises_before_any_cursor_use(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_dict_cursor_refuses_an_autocommit_connection():
    from src.services.sell_side._db import dict_cursor

    class _AutocommitConn:
        autocommit = True

    with pytest.raises(RuntimeError, match="transactional"):
        dict_cursor(_AutocommitConn())
