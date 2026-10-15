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


def builder_quote(status="approved", *, tax_rate="20", seller=True, sections=None):
    """A Quote builder quote: the seller's words (scope, assumptions, a text section of
    its own layout, terms), a discount on line 1, and tax at the quote's own rate. The
    seller is handed in as the pack would read it from the company profile."""
    from src.services.rga.builders.sales_quote import seller_from

    quote = _header(status, "3720.00")
    quote.update(quote_ref="SQ-SYNTHETIC-C", account_id=None,
                 account_name="Synthetic Haulage Ltd", contact_name="Robin Sample")
    quote["lines"] = [
        {**_line(601, 1, None, "Synthetic field seats", "10", "200.0000", "200.0000",
                 "1920.00", "0"), "catalog_item_id": None, "unit_of_measure": "seat",
         "entered_discount_pct": D("4.0000")},
        {**_line(602, 2, None, "Synthetic onboarding", "1", "1800.0000", "1800.0000",
                 "1800.00", "0"), "catalog_item_id": None, "unit_of_measure": "engagement",
         "entered_discount_pct": D("0.0000")},
    ]
    quote["justifications"] = []
    quote["tax_rate_pct"] = D(tax_rate) if tax_rate is not None else None
    quote["scope_html"] = "<p>Synthetic platform for <b>eleven</b> depots.</p><p>Second line<br>of scope.</p>"
    quote["assumptions_html"] = "<ul><li>Depots have power</li><li>Go-live in Q1</li></ul>"
    quote["terms"] = {"payment": "Net 30 from invoice", "start": "1 November 2026",
                      "lengthMonths": 12, "renewal": "Auto, 60-day notice", "sla": "SLA-B"}
    quote["sections"] = sections if sections is not None else [
        {"key": "letterhead", "kind": "letterhead", "heading": "Letterhead", "logoId": None},
        {"key": "customer", "kind": "customer", "heading": "Prepared for"},
        {"key": "scope", "kind": "scope", "heading": "What we will deliver"},
        {"key": "lines", "kind": "lines", "heading": "Pricing"},
        {"key": "totals", "kind": "totals", "heading": "Totals"},
        {"key": "terms", "kind": "terms", "heading": "Terms"},
        {"key": "why_us", "kind": "text", "heading": "Why Synthetic",
         "html": "<p>Seven years on the road.</p>"},
        {"key": "assumptions", "kind": "assumptions", "heading": "Assumptions"},
        {"key": "conditions", "kind": "conditions", "heading": "Conditions of approval"},
    ]
    quote["conditions"] = ["Do not discount further than 4%"]
    quote["seller"] = seller_from({
        "legalName": "Synthetic Seller Ltd", "tradingName": "SynSell",
        "address": "1 Fixture Street\nTestville TE5 7AA", "email": "sales@synthetic.example",
        "phone": "+44 20 7946 0000", "web": None, "companyNumber": "09876543",
        "vatNumber": "GB 123 4567 89", "footerNote": "Registered in England No. 09876543",
        "bank": {"name": "Synthetic Bank", "sortCode": "12-34-56",
                 "accountNumber": "87654321", "iban": None}}, _tiny_png()) if seller else None
    internal = {
        "sales_quote_id": 9001, "total_cost": D("2450.00"), "total_margin": D("1270.00"),
        "margin_pct": D("0.3414"),
        "lines": [
            {"sales_quote_line_id": 601, "line_no": 1, "unit_cost": D("125.0000"),
             "cost_tier_applied": None, "line_margin": D("670.00"), "line_margin_pct": D("0.3490")},
            {"sales_quote_line_id": 602, "line_no": 2, "unit_cost": D("1200.0000"),
             "cost_tier_applied": None, "line_margin": D("600.00"), "line_margin_pct": D("0.3333")},
        ],
        "justifications": [],
    }
    return quote, internal


def _tiny_png():
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (4, 2), (10, 42, 67)).save(out, format="PNG")
    return out.getvalue(), "image/png"
