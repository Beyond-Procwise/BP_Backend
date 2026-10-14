"""SYNTHETIC sell-side quotes. Invented for tests; no customer, product or price here is real.

There is no real sell-side quote to test against (bp_sqldb holds none, and the labelled
corpus is buy-side), so these two stand in, with Nick's approval (2026-10-10). Each is
returned as the two records a quote is: the customer record, shaped as
quotes.get_quote returns it, and the internal record, shaped as
quotes.get_quote_internal returns it. Fresh objects on every call.

  goods_quote()         three lines of goods, every cost known. Line 2 is priced at exactly
                        what line 1 costs us (41.20), so a figure that is ours on one line
                        and the customer's on another is always in play.
  unknown_cost_quote()  two lines; the second has no cost on file, so its margin and the
                        quote's totals are unknown (None), never zero.
"""
import datetime as dt
from decimal import Decimal as D

SYNTHETIC = True


def _line(line_id, no, sku, description, qty, list_price, unit_price, total, discount):
    return {"sales_quote_line_id": line_id, "sales_quote_id": 9001, "line_no": no,
            "sales_opportunity_id": None, "catalog_item_id": 7000 + no,
            "distributor_sku": sku, "mpn": None, "item_description": description,
            "quantity": D(qty), "unit_of_measure": "EA", "currency": "GBP",
            "list_price_at_quote": D(list_price), "unit_price": D(unit_price),
            "discount_pct": D(discount), "line_total": D(total), "justification_id": None}


def _header(status, total):
    return {"sales_quote_id": 9001, "tenant_id": "default", "quote_ref": "SQ-SYNTHETIC-A",
            "quote_number": None,
            "status": status, "account_id": "SYNTHETIC-ACC", "contact_id": None,
            "account_name": "Synthetic Fixture Co <Ltd>", "contact_name": "Sam Sample",
            "currency": "GBP", "quote_date": dt.date(2026, 9, 14),
            "valid_until": dt.date(2026, 11, 13), "total_ex_tax": D(total),
            "created_by": "sub-author", "approved_by": None}


def goods_quote(status="approved"):
    quote = _header(status, "2177.00")
    quote["lines"] = [
        _line(501, 1, "SYN-A", "Synthetic bracket & plate", "25", "62.0000", "58.0000", "1450.00", "0.0645"),
        _line(502, 2, "SYN-B", "Synthetic fastener kit", "5", "45.0000", "41.2000", "206.00", "0.0844"),
        _line(503, 3, "SYN-C", "Synthetic enclosure", "3", "180.0000", "173.6667", "521.00", "0.0352"),
    ]
    quote["justifications"] = [{"justification_id": 801, "kind": "end_of_life",
                                "claim": "Line 3 replaces a withdrawn part",
                                "customer_safe": True}]
    internal = {
        "sales_quote_id": 9001, "total_cost": D("1543.75"), "total_margin": D("633.25"),
        "margin_pct": D("0.2909"),
        "lines": [
            {"sales_quote_line_id": 501, "line_no": 1, "unit_cost": D("41.2000"),
             "cost_tier_applied": D("10"), "line_margin": D("420.00"), "line_margin_pct": D("0.2897")},
            {"sales_quote_line_id": 502, "line_no": 2, "unit_cost": D("27.3500"),
             "cost_tier_applied": None, "line_margin": D("69.25"), "line_margin_pct": D("0.3362")},
            {"sales_quote_line_id": 503, "line_no": 3, "unit_cost": D("125.6667"),
             "cost_tier_applied": None, "line_margin": D("144.00"), "line_margin_pct": D("0.2764")},
        ],
        "justifications": quote["justifications"] + [
            {"justification_id": 802, "kind": "price_gap",
             "claim": "We clear 633.25 on this one", "customer_safe": False}],
    }
    return quote, internal


def unknown_cost_quote(status="approved"):
    quote = _header(status, "1656.00")
    quote["quote_ref"] = "SQ-SYNTHETIC-B"
    quote["lines"] = [
        _line(511, 1, "SYN-A", "Synthetic bracket & plate", "25", "62.0000", "58.0000", "1450.00", "0.0645"),
        _line(512, 2, "SYN-D", "Synthetic part, cost not on file", "2", "110.0000", "103.0000", "206.00", "0.0636"),
    ]
    quote["justifications"] = []
    internal = {
        "sales_quote_id": 9001, "total_cost": None, "total_margin": None, "margin_pct": None,
        "lines": [
            {"sales_quote_line_id": 511, "line_no": 1, "unit_cost": D("41.2000"),
             "cost_tier_applied": D("10"), "line_margin": D("420.00"), "line_margin_pct": D("0.2897")},
            {"sales_quote_line_id": 512, "line_no": 2, "unit_cost": None,
             "cost_tier_applied": None, "line_margin": None, "line_margin_pct": None},
        ],
        "justifications": [],
    }
    return quote, internal
