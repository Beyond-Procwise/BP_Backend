"""template_reader: a seller's own quote in, a proposed template out -- structure only.

The model is a stand-in here: these tests pin what CODE does with whatever it says, which
is the part that keeps a price, an invented line or a made-up term off the proposal."""
import json
import re

import pytest

from src.services.sell_side import template_reader as tr
from src.services.sell_side import templates

CSV = (b"Acme Consulting - Quotation Q-1042\n"
       b"Description,Qty,Unit,Unit price,Total\n"
       b"Senior consultant,10,day,\"\xc2\xa3950.00\",\"\xc2\xa39,500.00\"\n"
       b"Project manager,5,day,\"\xc2\xa3800.00\",\"\xc2\xa34,000.00\"\n"
       b"Travel and expenses,1,,,\n"
       b"Payment terms: 30 days from invoice\n"
       b"Assumptions: Work is carried out remotely. Travel is billed at \xc2\xa3450 per trip.\n")

MONEY = re.compile(r"[£$€]\s?\d|\d[\d,]*\.\d{2}\b")


def _said(**over):
    said = {
        "name": "Acme Consulting quotation",
        "category": "Professional services",
        "columns": ["Description", "Qty", "Unit", "Unit price", "Total"],
        "sections": ["Payment terms", "Assumptions"],
        "lines": [
            {"description": "Senior consultant", "unit": "day", "tag": "service"},
            {"description": "Project manager", "unit": "day", "tag": "service"},
            {"description": "Travel and expenses", "unit": "", "tag": "one-off"},
        ],
        "termFields": ["payment"],
        "assumptions": ["Work is carried out remotely.", "Travel is billed at £450 per trip."],
    }
    said.update(over)
    return said


def _read(data=CSV, filename="acme-quote.csv", **said):
    seen = {}

    def call(prompt, **kw):
        seen["prompt"], seen["kw"] = prompt, kw
        return json.dumps(_said(**said))

    out = tr.read(data, filename, call=call, prompt=lambda: "Read {filename}:\n{document}",
                  busy=lambda: False)
    return out, seen


def test_the_reading_has_the_shape_the_quote_builder_renders():
    out, _ = _read()
    assert set(out) == {"name", "category", "lines", "termFields", "columns", "sections",
                        "pricesLeftOut", "notes", "assumptionsHtml", "sample"}
    assert out["sample"] is False
    assert [l["key"] for l in out["lines"]] == ["senior_consultant", "project_manager",
                                                "travel_and_expenses"]
    assert out["lines"][0] == {"key": "senior_consultant", "tag": "service",
                               "description": "Senior consultant", "unit": "day"}
    assert out["termFields"] == ["payment"]
    assert out["columns"] == ["Description", "Qty", "Unit", "Unit price", "Total"]


def test_the_body_passes_templates_clean_as_it_is():
    out, _ = _read()
    # `sections` here is the headings found, as written, for the seller to read; on a
    # saved template `sections` is the page layout (objects). It is not sent back.
    cleaned = templates.clean({k: v for k, v in out.items() if k != "sections"})
    assert cleaned["lines"] == out["lines"]
    assert cleaned["termFields"] == out["termFields"]
    assert cleaned["category"] == out["category"]
    assert cleaned["assumptionsHtml"] == out["assumptionsHtml"]
    with pytest.raises(ValueError, match="section 1 must be an object"):
        templates.clean(out)


def test_no_price_leaves_in_any_field_and_priced_lines_are_counted():
    out, _ = _read()
    assert not MONEY.search(json.dumps(out, ensure_ascii=False))
    assert out["pricesLeftOut"] == 2          # the two rows that carried a price
    # the sentence with a price in it is left out of the assumptions, the other is kept
    assert "remotely" in out["assumptionsHtml"]
    assert "450" not in out["assumptionsHtml"]


def test_a_price_the_model_slips_into_a_description_is_taken_out():
    out, _ = _read(lines=[{"description": "Senior consultant £950.00", "unit": "day",
                           "tag": "service"}])
    assert out["lines"][0]["description"] == "Senior consultant"
    assert not MONEY.search(json.dumps(out, ensure_ascii=False))


def test_a_line_that_is_not_on_the_document_is_dropped_and_said():
    out, _ = _read(lines=[{"description": "Senior consultant", "unit": "day", "tag": "service"},
                          {"description": "Cloud hosting", "unit": "month", "tag": "service"}])
    assert [l["description"] for l in out["lines"]] == ["Senior consultant"]
    assert any("Cloud hosting" in n for n in out["notes"])


def test_a_missing_or_unread_unit_is_unit_and_said():
    out, _ = _read(lines=[{"description": "Travel and expenses", "unit": "", "tag": "one-off"},
                          {"description": "Project manager", "unit": "fortnight", "tag": "service"}])
    assert [l["unit"] for l in out["lines"]] == ["unit", "unit"]
    assert any("unit" in n for n in out["notes"])


def test_a_term_the_document_does_not_mention_is_not_proposed():
    out, _ = _read(termFields=["payment", "sla", "renewal"])
    assert out["termFields"] == ["payment"]


def test_headings_not_on_the_document_are_left_out():
    out, _ = _read(columns=["Description", "Margin"], sections=["Assumptions", "Warranty"])
    assert out["columns"] == ["Description"]
    assert out["sections"] == ["Assumptions"]


def test_repeated_descriptions_get_distinct_keys():
    data = CSV + b"Senior consultant,2,day,\"\xc2\xa3950.00\",\n"
    out, _ = _read(data=data, lines=[{"description": "Senior consultant", "unit": "day", "tag": "service"},
                                     {"description": "Senior consultant", "unit": "day", "tag": "service"}])
    assert [l["key"] for l in out["lines"]] == ["senior_consultant", "senior_consultant_2"]


def test_the_name_falls_back_to_the_file_name_when_the_model_invents_one():
    out, _ = _read(name="Global Mega Deal")
    assert out["name"] == "Acme quote"


def test_the_model_is_given_the_document_and_no_price_field_to_answer_in():
    _, seen = _read()
    assert "Senior consultant" in seen["prompt"] and "acme-quote.csv" in seen["prompt"]
    props = json.dumps(seen["kw"]["format"])
    for word in ("price", "cost", "total", "amount", "discount"):
        assert f'"{word}' not in props.lower()


def test_no_line_found_is_a_readable_refusal():
    with pytest.raises(tr.Unreadable, match="pricing table"):
        _read(lines=[])


def test_a_document_with_no_text_is_a_readable_refusal():
    with pytest.raises(tr.Unreadable, match="no text"):
        _read(data=b"   \n", filename="empty.txt")


@pytest.mark.parametrize("name", ["quote.exe", "quote.zip", "quote"])
def test_an_unsupported_type_is_refused(name):
    with pytest.raises(tr.Unsupported):
        _read(filename=name)


def test_an_oversized_file_is_refused():
    with pytest.raises(tr.Unsupported, match="15 MB"):
        _read(data=b"x" * (tr.MAX_BYTES + 1), filename="big.txt")


def test_a_busy_or_silent_model_is_unavailable():
    with pytest.raises(tr.AgentUnavailable):
        tr.read(CSV, "q.csv", call=lambda *a, **k: None, prompt=lambda: "{document}",
                busy=lambda: False)
    with pytest.raises(tr.AgentUnavailable):
        tr.read(CSV, "q.csv", call=lambda *a, **k: "{}", prompt=lambda: "{document}",
                busy=lambda: True)
    with pytest.raises(tr.AgentUnavailable):
        tr.read(CSV, "q.csv", call=lambda *a, **k: "not json", prompt=lambda: "{document}",
                busy=lambda: False)


def test_document_text_lays_tables_out_as_rows():
    text = tr.document_text(CSV, "q.csv")
    assert "| Senior consultant | 10 | day | £950.00 | £9,500.00 |" in text


def test_the_word_invoice_alone_is_not_payment_terms():
    data = b"Invoice No. 3021\nDescription,Qty,Rate\nWebsite Design,1,2000.00\n"
    out, _ = _read(data=data, filename="inv.txt",
                   lines=[{"description": "Website Design", "unit": "", "tag": "service"}],
                   termFields=["payment"], assumptions=[])
    assert out["termFields"] == []
