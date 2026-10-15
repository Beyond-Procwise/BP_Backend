"""The Quote builder's quote against a real database: what is stored, what one deal may
hold, who may write what, and what the customer's page can and cannot see."""
import datetime as dt
import uuid
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from src.services.sell_side import accounts, builder, quote_send, quotes
from src.services.sell_side._db import NotFound, StateConflict
from tests.sell_side.conftest import SENTINEL, live, seed_item

pytestmark = live
TODAY = dt.date.today()
DRAFTER, REVIEWER = f"{SENTINEL}-drafter", f"{SENTINEL}-reviewer"


def _deal(conn, name=None):
    deal_id = f"{SENTINEL}-DEAL-{name or uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_deal (deal_id) VALUES (%s)", (deal_id,))
    conn.commit()
    return deal_id


def _draft(**over):
    """The screen's own seeded quote (UI data/mockQuotes.js)."""
    return {
        "currency": "GBP", "category": "Platform / Enterprise",
        "issuedAt": TODAY.isoformat(), "validUntil": (TODAY + dt.timedelta(days=30)).isoformat(),
        "preparedBy": "A. Okafor", "customer": {"name": "Kestrel Freight", "contact": "A. Reyes"},
        "scopeHtml": "<p>Field service platform, 340 seats.</p>",
        "assumptionsHtml": "<p><b>Assumes:</b> 11 sites.</p>",
        "lines": [
            {"id": "l1", "tag": "product", "description": "Field service seats", "qty": 340,
             "unit": "seat", "cost": 125, "markupPct": 60, "unitPrice": 200, "discountPct": 4},
            {"id": "l2", "tag": "service", "description": "Onboarding", "qty": 1,
             "unit": "engagement", "cost": 12000, "markupPct": 50, "unitPrice": 18000,
             "discountPct": 0},
            {"id": "l3", "tag": "one-off", "description": "Device provisioning", "qty": 11,
             "unit": "site", "cost": None, "markupPct": None, "unitPrice": 950, "discountPct": 0},
        ],
        "terms": {"payment": "Net 30", "start": "2026-11-01", "lengthMonths": 12,
                  "renewal": "Auto, 60-day notice", "sla": "SLA-B"},
        **over,
    }


def _new(conn, deal_id="auto", **over):
    deal_id = _deal(conn) if deal_id == "auto" else deal_id
    return builder.create(conn, deal_id=deal_id, draft=_draft(**over), created_by=DRAFTER)


def _approved(conn, **over):
    q = _new(conn, **over)
    builder.submit(conn, int(q["id"]), actor=DRAFTER)
    quotes.approve(conn, int(q["id"]), approver=REVIEWER)
    return builder.get(conn, int(q["id"]))


# --- what is stored ----------------------------------------------------------------

def test_a_quote_on_a_deal_comes_back_as_the_screen_holds_it(live_db):
    conn, _ = live_db
    deal = _deal(conn)
    q = builder.create(conn, deal_id=deal, draft=_draft(), created_by=DRAFTER)

    assert q["dealId"] == deal and q["version"] == 1 and q["status"] == "draft"
    assert q["number"] is None, "a number is minted when the quote is issued, not before"
    assert q["customer"]["name"] == "Kestrel Freight"
    assert q["terms"]["lengthMonths"] == 12 and q["conditions"] == []
    assert [(l["tag"], l["qty"], l["unitPrice"], l["cost"], l["discountPct"]) for l in q["lines"]] == [
        ("product", D(340), D("200.0000"), D("125.0000"), D("4.0000")),
        ("service", D(1), D("18000.0000"), D("12000.0000"), D("0.0000")),
        ("one-off", D(11), D("950.0000"), None, D("0.0000"))]
    assert q["position"]["subprocessId"] == "sales.opportunity.quote-drafted"


def test_margin_is_over_the_costed_lines_and_says_how_many_it_left_out(live_db):
    conn, _ = live_db
    t = _new(conn)["totals"]
    # 340 x 200 less 4% = 65,280; + 18,000; + 10,450 uncosted
    assert t["exTax"] == D("93730.00")
    assert t["cost"] == D("54500.00")            # 42,500 + 12,000
    assert t["margin"] == D("28780.00")          # 83,280 costed revenue - 54,500
    assert t["marginFraction"] == D("0.3456")
    assert (t["costedLines"], t["uncostedLines"]) == (2, 1)


def test_a_typed_cost_of_zero_is_kept_and_not_counted(live_db):
    conn, _ = live_db
    q = _new(conn, lines=[{"tag": "service", "description": "Free?", "qty": 1,
                           "unitPrice": 100, "cost": 0}])
    assert q["lines"][0]["cost"] == D("0.0000")
    assert q["totals"]["margin"] is None and q["totals"]["uncostedLines"] == 1


def test_what_is_printed_for_a_customer_is_cleaned_when_it_is_stored(live_db):
    conn, _ = live_db
    q = _new(conn, scopeHtml='<p onclick="x()">Scope</p><script>steal()</script>')
    assert q["scopeHtml"] == "<p>Scope</p>"


# --- the deal ----------------------------------------------------------------------

def test_a_quote_may_be_started_before_its_deal_exists_and_is_linked_once(live_db):
    conn, _ = live_db
    q = builder.create(conn, deal_id=None, draft=_draft(), created_by=DRAFTER)
    assert q["dealId"] is None
    assert [u["id"] for u in builder.list_unlinked(conn, DRAFTER)] == [q["id"]]
    assert builder.list_unlinked(conn, f"{SENTINEL}-someone-else") == []

    deal = _deal(conn)
    linked = builder.save(conn, int(q["id"]), draft=_draft(), actor=DRAFTER, deal_id=deal)
    assert linked["dealId"] == deal
    assert builder.list_unlinked(conn, DRAFTER) == []

    with pytest.raises(StateConflict, match="set once"):
        builder.save(conn, int(q["id"]), draft=_draft(), actor=DRAFTER, deal_id=_deal(conn))
    assert builder.get(conn, int(q["id"]))["dealId"] == deal


def test_saving_without_naming_a_deal_leaves_the_link_alone(live_db):
    conn, _ = live_db
    q = _new(conn)
    assert builder.save(conn, int(q["id"]), draft=_draft(), actor=DRAFTER)["dealId"] == q["dealId"]


def test_one_deal_has_one_live_quote(live_db):
    conn, _ = live_db
    deal = _deal(conn)
    first = _new(conn, deal_id=deal)
    with pytest.raises(StateConflict, match="already has quote"):
        _new(conn, deal_id=deal)
    loose = builder.create(conn, deal_id=None, draft=_draft(), created_by=DRAFTER)
    with pytest.raises(StateConflict, match="already has quote"):
        builder.save(conn, int(loose["id"]), draft=_draft(), actor=DRAFTER, deal_id=deal)
    assert [q["id"] for q in builder.list_for_deal(conn, deal)] == [first["id"]]


def test_the_database_itself_refuses_a_second_live_quote_on_a_deal(live_db):
    """The service check is the message; the index is the control. Two people linking at
    the same moment both pass the check, and only the index stands between them."""
    conn, _ = live_db
    deal = _deal(conn)
    _new(conn, deal_id=deal)
    loose = builder.create(conn, deal_id=None, draft=_draft(), created_by=DRAFTER)
    import psycopg2
    with conn.cursor() as cur, pytest.raises(psycopg2.errors.UniqueViolation):
        cur.execute("UPDATE proc.bp_sales_quote SET deal_id = %s WHERE sales_quote_id = %s",
                    (deal, int(loose["id"])))
    conn.rollback()


def test_a_quote_cannot_go_on_a_deal_that_does_not_exist(live_db):
    conn, _ = live_db
    with pytest.raises(NotFound):
        builder.create(conn, deal_id=f"{SENTINEL}-DEAL-nope", draft=_draft(), created_by=DRAFTER)


# --- saving ------------------------------------------------------------------------

def test_a_save_replaces_the_lines_and_reprices(live_db):
    conn, _ = live_db
    q = _new(conn)
    saved = builder.save(conn, int(q["id"]), actor=DRAFTER, draft=_draft(lines=[
        {"tag": "service", "description": "Only this", "qty": 2, "unitPrice": 100, "cost": 60}]))
    assert [l["description"] for l in saved["lines"]] == ["Only this"]
    assert saved["totals"]["exTax"] == D("200.00") and saved["totals"]["marginFraction"] == D("0.4000")
    assert saved["totals"]["uncostedLines"] == 0
    assert saved["id"] == q["id"] and saved["version"] == 1


def test_a_save_over_somebody_elses_later_save_is_refused(live_db):
    conn, _ = live_db
    q = _new(conn)
    mine = builder.save(conn, int(q["id"]), draft=_draft(category="Mine"), actor=DRAFTER,
                        revision=q["revision"])
    with pytest.raises(StateConflict, match="someone else"):
        builder.save(conn, int(q["id"]), draft=_draft(category="Theirs"), actor="other",
                     revision=q["revision"])
    assert builder.get(conn, int(q["id"]))["category"] == "Mine"
    assert mine["revision"] != q["revision"]


def test_only_a_draft_is_edited(live_db):
    conn, _ = live_db
    q = _new(conn)
    builder.submit(conn, int(q["id"]), actor=DRAFTER)
    with pytest.raises(StateConflict, match="only a draft"):
        builder.save(conn, int(q["id"]), draft=_draft(category="Sneaky"), actor=DRAFTER)


def test_the_drafter_cannot_write_their_own_conditions(live_db):
    conn, _ = live_db
    q = _new(conn)
    saved = builder.save(conn, int(q["id"]), actor=DRAFTER,
                         draft=_draft(conditions=["Approved, honest"], status="approved",
                                      number="Q-2026-0001", version=9))
    assert saved["conditions"] == [] and saved["status"] == "draft"
    assert saved["number"] is None and saved["version"] == 1


def test_a_catalog_quote_is_not_edited_in_the_builder(live_db):
    conn, distributor = live_db
    item = seed_item(conn, distributor, f"{SENTINEL}-B1")
    accounts.create_account(conn, account_id=f"{SENTINEL}-B", account_name="Builder Ltd")
    catalog = quotes.create_draft(
        conn, account_id=f"{SENTINEL}-B", currency="GBP", created_by=DRAFTER,
        valid_until=TODAY + dt.timedelta(days=30),
        lines=[{"catalog_item_id": item, "quantity": D(1), "unit_price": D(16)}])
    with pytest.raises(StateConflict, match="catalog quote"):
        builder.save(conn, catalog["sales_quote_id"], draft=_draft(), actor=DRAFTER)
    assert quotes.get_quote(conn, catalog["sales_quote_id"])["lines"][0]["catalog_item_id"] == item


# --- review ------------------------------------------------------------------------

def test_a_quote_with_nothing_on_it_cannot_go_for_review(live_db):
    conn, _ = live_db
    q = builder.create(conn, deal_id=None, created_by=DRAFTER,
                       draft=_draft(lines=[], customer={"name": ""}))
    with pytest.raises(StateConflict, match="a customer, at least one line"):
        builder.submit(conn, int(q["id"]), actor=DRAFTER)


def test_review_moves_the_quote_along_the_ladder_and_into_the_queue(live_db):
    conn, _ = live_db
    q = _new(conn)
    sent = builder.submit(conn, int(q["id"]), actor=DRAFTER)
    assert sent["status"] == "in-review"
    assert sent["position"]["subprocessId"] == "sales.approval.deal-desk-review"
    assert q["id"] in [w["id"] for w in builder.review_queue(conn)]

    back = quotes.withdraw(conn, int(q["id"]), actor=DRAFTER)
    assert back["status"] == "draft"
    assert q["id"] not in [w["id"] for w in builder.review_queue(conn)]


def test_the_approver_writes_the_conditions_and_nobody_approves_their_own(live_db):
    conn, _ = live_db
    q = _new(conn)
    builder.submit(conn, int(q["id"]), actor=DRAFTER)
    with pytest.raises(StateConflict, match="their own"):
        quotes.approve(conn, int(q["id"]), approver=DRAFTER, conditions=["I approve me"])
    quotes.approve(conn, int(q["id"]), approver=REVIEWER,
                   conditions=["Price held to 31 December", "  "])
    after = builder.get(conn, int(q["id"]))
    assert after["status"] == "approved" and after["approvedBy"] == REVIEWER
    assert after["conditions"] == ["Price held to 31 December"]


def test_a_rejection_needs_a_reason_and_a_different_person(live_db):
    conn, _ = live_db
    q = _new(conn)
    builder.submit(conn, int(q["id"]), actor=DRAFTER)
    with pytest.raises(ValueError, match="reason"):
        quotes.reject(conn, int(q["id"]), approver=REVIEWER, reason="  ")
    with pytest.raises(StateConflict, match="their own"):
        quotes.reject(conn, int(q["id"]), approver=DRAFTER, reason="changed my mind")
    assert quotes.reject(conn, int(q["id"]), approver=REVIEWER,
                         reason="discount too deep")["status"] == "draft"
    with conn.cursor() as cur:
        cur.execute("SELECT details->>'reason', details->>'actor' FROM proc.bp_agent_actions "
                    "WHERE action_type = 'sales_quote.rejected' AND doc_pk = %s", (q["id"],))
        assert cur.fetchone() == ("discount too deep", REVIEWER)
    conn.rollback()


def test_a_quote_priced_below_its_entered_cost_is_not_routinely_approved(live_db):
    conn, _ = live_db
    q = _new(conn, lines=[{"tag": "service", "description": "Loss", "qty": 1,
                           "unitPrice": 90, "cost": 100}])
    builder.submit(conn, int(q["id"]), actor=DRAFTER)
    with pytest.raises(StateConflict, match="below cost"):
        quotes.approve(conn, int(q["id"]), approver=REVIEWER)


# --- after it has gone out ---------------------------------------------------------

def test_superseding_leaves_what_was_sent_alone_and_starts_the_next_version(live_db):
    conn, _ = live_db
    q = _approved(conn)
    issued = quotes.issue(conn, int(q["id"]), actor=REVIEWER)
    with pytest.raises(StateConflict, match="only a draft"):
        builder.save(conn, int(q["id"]), draft=_draft(category="After the fact"), actor=DRAFTER)

    v2 = builder.supersede(conn, int(q["id"]), actor=DRAFTER)
    assert (v2["version"], v2["status"], v2["number"], v2["conditions"]) == (2, "draft", None, [])
    assert v2["supersedesId"] == q["id"] and v2["dealId"] == q["dealId"]
    assert [l["description"] for l in v2["lines"]] == [l["description"] for l in q["lines"]]
    assert v2["totals"] == q["totals"]

    old = builder.get(conn, int(q["id"]))
    assert old["status"] == "superseded" and old["number"] == issued["quote_number"]
    assert [x["id"] for x in builder.list_for_deal(conn, q["dealId"])] == [v2["id"], q["id"]]


def test_a_draft_is_edited_not_superseded(live_db):
    conn, _ = live_db
    q = _new(conn)
    with pytest.raises(StateConflict, match="only a quote that has been sent"):
        builder.supersede(conn, int(q["id"]), actor=DRAFTER)


def test_a_quote_with_no_deal_fails_the_send_check_that_asks_for_one(live_db):
    conn, _ = live_db
    q = builder.create(conn, deal_id=None, draft=_draft(), created_by=DRAFTER)
    builder.submit(conn, int(q["id"]), actor=DRAFTER)
    quotes.approve(conn, int(q["id"]), approver=REVIEWER)
    prepared = quote_send.prepare(conn, int(q["id"]), sender="quotes@livetest.example",
                                  principal=SimpleNamespace(subject=REVIEWER))
    failed = {c["id"] for c in prepared.checks if not c["passed"]}
    assert "linked_to_deal" in failed
    conn.rollback()


# --- what the customer can see -----------------------------------------------------

def test_the_customer_record_of_a_builder_quote_has_no_cost_in_it(live_db):
    conn, _ = live_db
    q = _approved(conn)
    customer = quotes.get_quote(conn, int(q["id"]))
    assert customer["account_name"] == "Kestrel Freight" and customer["contact_name"] == "A. Reyes"
    flat = repr(customer)
    for internal in ("unit_cost", "markup", "total_margin", "margin_pct", "cost_source",
                     "125.0000", "12000.0000", "54500.00", "28780.00"):
        assert internal not in flat, f"{internal} is on the customer record"


def test_the_customers_page_is_produced_for_a_builder_quote_and_carries_no_cost(live_db):
    from src.services.rga import quote_pipeline

    conn, _ = live_db
    q = _approved(conn)
    page = quote_pipeline.produce(quotes.get_quote(conn, int(q["id"])),
                                  quotes.get_quote_internal(conn, int(q["id"])), fmt="html")
    html = page.content if isinstance(page.content, str) else page.content.decode()
    assert "Kestrel Freight" in html and "Field service seats" in html
    assert "£93,730" in html
    for internal in ("54,500", "54500", "28,780", "28780", "£12,000", "£125"):
        assert internal not in html, f"{internal} is on the customer's page"


# --- the cost record apart, the customer's email, the settings (ruled 2026-10-14) ---

def _customer(name=None, **more):
    return {"name": name or f"{SENTINEL} Customer {uuid.uuid4().hex[:6]}", **more}


def test_a_save_that_leaves_the_cost_record_off_keeps_it(live_db):
    conn, _ = live_db
    q = _new(conn)
    # the screen's customer quote: no tag, cost or markup on any line it read
    lines = [{k: v for k, v in l.items() if k not in ("tag", "cost", "markupPct", "costSource")}
             for l in q["lines"]]
    saved = builder.save(conn, int(q["id"]), draft=_draft(lines=lines), actor=DRAFTER)
    assert [(l["tag"], l["cost"], l["markupPct"]) for l in saved["lines"]] == [
        ("product", D("125.0000"), D("60.0000")), ("service", D("12000.0000"), D("50.0000")),
        ("one-off", None, None)]
    assert saved["totals"]["margin"] == q["totals"]["margin"]


def test_the_cost_record_is_written_on_its_own_and_the_price_does_not_move(live_db):
    conn, _ = live_db
    q = _new(conn)
    first, third = q["lines"][0]["id"], q["lines"][2]["id"]
    saved = builder.save_internal(conn, int(q["id"]), actor=DRAFTER, lines={
        first: {"cost": 150}, third: {"cost": 600, "markupPct": 58, "tag": "service"}})
    assert [(l["cost"], l["unitPrice"], l["tag"]) for l in saved["lines"]] == [
        (D("150.0000"), D("200.0000"), "product"), (D("12000.0000"), D("18000.0000"), "service"),
        (D("600.0000"), D("950.0000"), "service")]
    assert saved["totals"]["uncostedLines"] == 0
    with pytest.raises(ValueError, match="not lines of quote"):
        builder.save_internal(conn, int(q["id"]), actor=DRAFTER, lines={"999999999": {"cost": 1}})
    with pytest.raises(ValueError, match="only tag, cost, markupPct"):
        builder.save_internal(conn, int(q["id"]), actor=DRAFTER,
                              lines={saved["lines"][0]["id"]: {"unitPrice": 1}})


def test_a_line_keeps_its_id_through_a_save_so_its_cost_record_still_finds_it(live_db):
    conn, _ = live_db
    q = _new(conn)
    ids = [l["id"] for l in q["lines"]]
    # the screen moves the third line to the top, edits the first and adds a new one
    lines = [q["lines"][2], {**q["lines"][0], "description": "Seats, renamed"},
             q["lines"][1], {"tag": "service", "description": "Training", "qty": 1,
                             "unitPrice": 500}]
    saved = builder.save(conn, int(q["id"]), draft=_draft(lines=lines), actor=DRAFTER)
    assert [l["id"] for l in saved["lines"][:3]] == [ids[2], ids[0], ids[1]]
    assert saved["lines"][3]["id"] not in ids
    assert saved["lines"][1]["description"] == "Seats, renamed"
    # the cost record read before the save is still about these lines
    again = builder.save_internal(conn, int(q["id"]), actor=DRAFTER,
                                  lines={ids[0]: {"cost": 130}})
    assert [l["cost"] for l in again["lines"]][:3] == [None, D("130.0000"), D("12000.0000")]
    # a line the save leaves out is gone, and its id is not reused by the next one
    dropped = builder.save(conn, int(q["id"]), actor=DRAFTER,
                           draft=_draft(lines=[again["lines"][1], {**again["lines"][1], "id": ids[0]}]))
    assert [l["id"] for l in dropped["lines"]][0] == ids[0]
    assert dropped["lines"][1]["id"] not in ids, "the same id twice is one kept line and one new"


def test_a_save_that_sends_a_blank_cost_does_not_erase_it(live_db):
    conn, _ = live_db
    q = _new(conn)
    # a screen that does not hold the cost record sends every cost and markup as null
    blank = [{**l, "cost": None, "markupPct": None} for l in q["lines"]]
    saved = builder.save(conn, int(q["id"]), draft=_draft(lines=blank), actor=DRAFTER)
    assert [(l["cost"], l["markupPct"]) for l in saved["lines"]] == [
        (D("125.0000"), D("60.0000")), (D("12000.0000"), D("50.0000")), (None, None)]
    # clearing one is the cost record's own route
    cleared = builder.save_internal(conn, int(q["id"]), actor=DRAFTER,
                                    lines={q["lines"][0]["id"]: {"cost": None, "markupPct": None}})
    assert (cleared["lines"][0]["cost"], cleared["lines"][0]["markupPct"]) == (None, None)
    assert cleared["totals"]["uncostedLines"] == 2


def test_a_quote_given_to_another_customer_is_not_filed_under_the_first(live_db):
    conn, _ = live_db
    first = _customer(contact="A. Reyes", email="a.reyes@first.livetest.example")
    q = _new(conn, customer=first)
    other = _customer(contact="B. Lane", email="b.lane@second.livetest.example")
    saved = builder.save(conn, int(q["id"]), draft=_draft(customer=other), actor=DRAFTER)
    assert saved["customer"]["accountId"] != q["customer"]["accountId"]
    assert saved["customer"]["email"] == "b.lane@second.livetest.example"
    old = accounts.get_account(conn, q["customer"]["accountId"])
    assert [c["contact_email"] for c in old["contacts"]] == ["a.reyes@first.livetest.example"]
    assert quotes.get_quote(conn, int(q["id"]))["account_name"] == other["name"]


def test_a_customer_renamed_keeps_its_record_and_the_page_prints_the_typed_name(live_db):
    conn, _ = live_db
    who = _customer(contact="A. Reyes", email="a.reyes@renamed.livetest.example")
    q = _new(conn, customer=who)
    longer = {**who, "name": who["name"] + " Holdings Ltd"}
    saved = builder.save(conn, int(q["id"]), draft=_draft(customer=longer), actor=DRAFTER)
    assert saved["customer"]["accountId"] == q["customer"]["accountId"]
    assert quotes.get_quote(conn, int(q["id"]))["account_name"] == longer["name"]


def test_an_address_taken_off_the_quote_leaves_nobody_to_send_it_to(live_db):
    conn, _ = live_db
    who = _customer(contact="A. Reyes", email="a.reyes@removed.livetest.example")
    q = _new(conn, customer=who)
    saved = builder.save(conn, int(q["id"]), actor=DRAFTER,
                         draft=_draft(customer={k: v for k, v in who.items() if k != "email"}))
    assert saved["customer"]["email"] is None
    assert saved["customer"]["accountId"] == q["customer"]["accountId"], "same name, same customer"
    renamed = builder.save(conn, int(q["id"]), actor=DRAFTER,
                           draft=_draft(customer={"name": f"{SENTINEL} Someone Else", "contact": "X"}))
    assert renamed["customer"]["accountId"] is None, "another name with no address is not theirs"


def test_the_customer_fields_carry_nothing_that_is_ours(live_db):
    conn, _ = live_db
    q = _new(conn)
    flat = repr(builder.customer_fields(q))
    for ours in ("'cost'", "markupPct", "costSource", "'tag'", "'margin'", "marginFraction",
                 "125.0000", "12000.0000", "54500.00", "28780.00"):
        assert ours not in flat, f"{ours} is on the customer's quote"
    internal = builder.internal_fields(q, {"marginFloorPct": 20})
    assert set(internal["lines"]) == {l["id"] for l in q["lines"]}
    assert internal["totals"]["margin"] == D("28780.00")


def test_a_typed_email_is_saved_to_the_customer_and_used_for_the_send(live_db):
    conn, _ = live_db
    who = _customer(contact="A. Reyes", email="a.reyes@kestrel.livetest.example")
    q = _approved(conn, customer=who)
    assert q["customer"]["email"] == "a.reyes@kestrel.livetest.example"
    account = accounts.get_account(conn, q["customer"]["accountId"])
    assert account["account_name"] == who["name"]
    assert [(c["contact_name"], c["contact_email"]) for c in account["contacts"]] == [
        ("A. Reyes", "a.reyes@kestrel.livetest.example")]
    prepared = quote_send.prepare(conn, int(q["id"]), sender="quotes@livetest.example",
                                  principal=SimpleNamespace(subject=REVIEWER))
    conn.rollback()
    assert prepared.to == "a.reyes@kestrel.livetest.example"
    assert all(c["passed"] for c in prepared.checks if c["id"] in (
        "has_recipient", "linked_to_deal", "margin_policy_set", "tax_rate_set"))


def test_the_same_customer_gathers_on_one_record_as_details_come_in(live_db):
    conn, _ = live_db
    name = _customer()["name"]
    first = _new(conn, customer=_customer(name, email="ops@gull.livetest.example"))
    # the next quote finds them by the address, and their name is now known
    second = _new(conn, customer=_customer(name.upper() + " ", contact="R. Gull",
                                           email="OPS@gull.livetest.example"))
    assert second["customer"]["accountId"] == first["customer"]["accountId"]
    contacts = accounts.get_account(conn, first["customer"]["accountId"])["contacts"]
    assert [(c["contact_name"], c["contact_email"]) for c in contacts] == [
        ("R. Gull", "ops@gull.livetest.example")]


def test_a_builder_quote_with_an_account_but_no_deal_is_still_not_sent(live_db):
    conn, _ = live_db
    q = builder.create(conn, deal_id=None, created_by=DRAFTER,
                       draft=_draft(customer=_customer(email="x@nodeal.livetest.example")))
    assert q["customer"]["accountId"] is not None
    assert int(q["id"]) in {int(u["id"]) for u in builder.list_unlinked(conn, DRAFTER)}
    builder.submit(conn, int(q["id"]), actor=DRAFTER)
    quotes.approve(conn, int(q["id"]), approver=REVIEWER)
    prepared = quote_send.prepare(conn, int(q["id"]), sender="quotes@livetest.example",
                                  principal=SimpleNamespace(subject=REVIEWER))
    conn.rollback()
    assert "linked_to_deal" in {c["id"] for c in prepared.checks if not c["passed"]}


def test_a_new_quote_takes_the_policy_tax_rate_and_keeps_the_one_it_was_given(live_db):
    conn, _ = live_db
    q = _new(conn)
    assert q["taxRatePct"] == D("20.0000")                      # the stated default
    kept = builder.save(conn, int(q["id"]), draft=_draft(), actor=DRAFTER)
    assert kept["taxRatePct"] == D("20.0000")                   # left out: kept
    changed = builder.save(conn, int(q["id"]), draft=_draft(taxRatePct=5), actor=DRAFTER)
    assert changed["taxRatePct"] == D("5.0000")
    with pytest.raises(ValueError, match="taxRatePct"):
        builder.save(conn, int(q["id"]), draft=_draft(taxRatePct=120), actor=DRAFTER)


def test_while_pricing_policy_is_unset_a_quote_neither_goes_for_review_nor_issues(
        live_db, pricing_unset):
    conn, _ = live_db
    q = _new(conn)
    assert q["taxRatePct"] is None
    with pytest.raises(builder.PolicyUnset) as refused:
        builder.submit(conn, int(q["id"]), actor=DRAFTER)
    assert refused.value.codes == ["MARGIN_POLICY_UNSET", "TAX_POLICY_UNSET"]
    assert builder.get(conn, int(q["id"]))["status"] == "draft"


def test_a_quote_approved_before_the_policy_went_unset_is_not_issued(live_db, monkeypatch):
    from tests.sell_side.conftest import STATED_PRICING, state_pricing

    conn, _ = live_db
    q = _approved(conn)
    state_pricing(monkeypatch, {**STATED_PRICING, "margin_floor_pct": None})
    with pytest.raises(StateConflict, match="MARGIN_POLICY_UNSET"):
        quotes.issue(conn, int(q["id"]), actor=REVIEWER)
    assert builder.get(conn, int(q["id"]))["number"] is None


def test_a_quote_says_it_was_sent_only_once_the_send_is_recorded(live_db):
    conn, _ = live_db
    q = _approved(conn)
    assert q["sent"] is None
    quotes.record_event(conn, quote_send.SENT_EVENT,
                        {"sales_quote_id": int(q["id"]), "quote_ref": q["ref"]},
                        actor=REVIEWER, email_id="<m-1@livetest>")
    conn.commit()
    sent = builder.get(conn, int(q["id"]))["sent"]
    assert sent["emailId"] == "<m-1@livetest>" and sent["at"] is not None
