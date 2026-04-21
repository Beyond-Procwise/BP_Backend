from enum import Enum


class FieldType(str, Enum):
    ID = "id"
    DATE = "date"
    MONEY = "money"
    PERCENT = "percent"
    CURRENCY_CODE = "currency"
    ORG = "org"
    ADDRESS = "address"
    TEXT = "text"
    INTEGER = "integer"


FIELD_TYPES: dict[tuple[str, str], FieldType] = {
    # Invoice
    ("Invoice", "invoice_id"):             FieldType.ID,
    ("Invoice", "po_id"):                  FieldType.ID,
    ("Invoice", "supplier_id"):            FieldType.ORG,
    ("Invoice", "buyer_id"):               FieldType.ORG,
    ("Invoice", "invoice_date"):           FieldType.DATE,
    ("Invoice", "due_date"):               FieldType.DATE,
    ("Invoice", "invoice_amount"):         FieldType.MONEY,
    ("Invoice", "tax_amount"):             FieldType.MONEY,
    ("Invoice", "tax_percent"):            FieldType.PERCENT,
    ("Invoice", "invoice_total_incl_tax"): FieldType.MONEY,
    ("Invoice", "currency"):               FieldType.CURRENCY_CODE,
    ("Invoice", "payment_terms"):          FieldType.TEXT,
    # Purchase_Order
    ("Purchase_Order", "po_id"):                  FieldType.ID,
    ("Purchase_Order", "supplier_name"):          FieldType.ORG,
    ("Purchase_Order", "supplier_id"):            FieldType.ORG,
    ("Purchase_Order", "buyer_id"):               FieldType.ORG,
    ("Purchase_Order", "order_date"):             FieldType.DATE,
    ("Purchase_Order", "expected_delivery_date"): FieldType.DATE,
    ("Purchase_Order", "total_amount"):           FieldType.MONEY,
    ("Purchase_Order", "tax_amount"):             FieldType.MONEY,
    ("Purchase_Order", "tax_percent"):            FieldType.PERCENT,
    ("Purchase_Order", "total_amount_incl_tax"):  FieldType.MONEY,
    ("Purchase_Order", "currency"):               FieldType.CURRENCY_CODE,
    ("Purchase_Order", "payment_terms"):          FieldType.TEXT,
    ("Purchase_Order", "incoterm"):               FieldType.TEXT,
    ("Purchase_Order", "delivery_address_line1"): FieldType.ADDRESS,
    # Quote
    ("Quote", "quote_id"):               FieldType.ID,
    ("Quote", "supplier_id"):            FieldType.ORG,
    ("Quote", "buyer_id"):               FieldType.ORG,
    ("Quote", "quote_date"):             FieldType.DATE,
    ("Quote", "validity_date"):          FieldType.DATE,
    ("Quote", "total_amount"):           FieldType.MONEY,
    ("Quote", "tax_amount"):             FieldType.MONEY,
    ("Quote", "tax_percent"):            FieldType.PERCENT,
    ("Quote", "total_amount_incl_tax"):  FieldType.MONEY,
    ("Quote", "currency"):               FieldType.CURRENCY_CODE,
}


def fields_for(doc_type: str) -> list[str]:
    return [fname for (dt, fname) in FIELD_TYPES if dt == doc_type]


def type_of(doc_type: str, field_name: str) -> FieldType | None:
    return FIELD_TYPES.get((doc_type, field_name))
