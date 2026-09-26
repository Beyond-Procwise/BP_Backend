from src.services.truth.label import label_record

SRC = "Invoice No: INV-2024-001\nFrom: TechNova Ltd\nSubtotal: £2,400.00\nDate: 15/03/2024"


def test_header_fields_are_each_given_a_verdict():
    rec = {"header": {"invoice_id": "INV-2024-001", "supplier_name": "TechNova Ltd",
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
    # supplier_id is a pipeline-minted surrogate and therefore derived, so the
    # hallucination test uses a field the document really does carry.
    rec = {"header": {"supplier_name": "Nonexistent Holdings Ltd"}}
    out = label_record(rec, SRC)
    assert out["fields"]["supplier_name"]["outcome"] == "unsupported"


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


def test_the_number_of_fields_emitted_is_recorded():
    """A model that emits only the five fields it is confident about scores
    100% accuracy at 100% coverage. Neither number can see what was never
    emitted, so the count of emitted fields is reported alongside them."""
    out = label_record({"header": {"invoice_id": "INV-2024-001"}}, SRC)
    assert out["counts"]["emitted"] == 1

    richer = label_record(
        {"header": {"invoice_id": "INV-2024-001", "supplier_name": "TechNova Ltd",
                    "invoice_amount": 2400.00}}, SRC)
    assert richer["counts"]["emitted"] == 3


def test_a_line_items_value_that_is_not_a_list_does_not_crash():
    out = label_record({"header": {}, "line_items": "not a list"}, SRC)
    assert out["counts"]["emitted"] == 0
