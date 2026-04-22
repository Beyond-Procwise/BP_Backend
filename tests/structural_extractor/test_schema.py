from src.services.structural_extractor.discovery.schema import FieldType, fields_for, type_of


def test_invoice_field_types():
    assert type_of("Invoice", "invoice_date") == FieldType.DATE
    assert type_of("Invoice", "invoice_total_incl_tax") == FieldType.MONEY
    assert type_of("Invoice", "supplier_id") == FieldType.ORG
    assert type_of("Invoice", "currency") == FieldType.CURRENCY_CODE


def test_po_fields_contain_total():
    fs = fields_for("Purchase_Order")
    assert "total_amount" in fs
    assert "supplier_name" in fs


def test_field_type_has_address():
    assert FieldType.ADDRESS
