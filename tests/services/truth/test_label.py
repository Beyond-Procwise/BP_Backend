from src.services.truth.label import label_record

SRC = "Invoice No: INV-2024-001\nFrom: TechNova Ltd\nSubtotal: £2,400.00\nDate: 15/03/2024"


def test_header_fields_are_each_given_a_verdict():
    rec = {"header": {"invoice_id": "INV-2024-001", "supplier_id": "TechNova Ltd",
                      "invoice_amount": 2400.00, "invoice_date": "2024-03-15"}}
    out = label_record(rec, SRC)
    assert out["counts"]["verified"] == 4
    assert out["fields"]["invoice_id"]["outcome"] == "verified"


def test_a_derived_field_is_unverifiable_even_with_source_text():
    rec = {"header": {"converted_amount_usd": 3100.00}}
    out = label_record(rec, SRC)
    assert out["fields"]["converted_amount_usd"]["outcome"] == "unverifiable"
    assert out["fields"]["converted_amount_usd"]["rule"] == "derived-field"


def test_a_hallucinated_value_is_unsupported():
    rec = {"header": {"supplier_id": "Nonexistent Holdings Ltd"}}
    out = label_record(rec, SRC)
    assert out["fields"]["supplier_id"]["outcome"] == "unsupported"


def test_line_item_fields_are_keyed_by_index():
    rec = {"header": {}, "line_items": [{"item_description": "Laptop Dell XPS 15"}]}
    out = label_record(rec, "Laptop Dell XPS 15  2  £1,200.00")
    assert out["fields"]["line_items[0].item_description"]["outcome"] == "verified"


def test_source_text_with_no_words_is_unverifiable_not_unsupported():
    # Review Focus 5: a scanned page with no text layer. The model may well be
    # right; there is simply nothing to check against.
    rec = {"header": {"invoice_id": "INV-2024-001"}}
    out = label_record(rec, "   \n\n \t ")
    assert out["fields"]["invoice_id"]["outcome"] == "unverifiable"
