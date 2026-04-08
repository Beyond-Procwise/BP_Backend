"""Tests for vendor profile service and bp_ schema mapping."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from utils.procurement_schema import (
    BP_PROCUREMENT_SCHEMAS,
    BP_DOC_TYPE_TO_TABLE,
    map_columns_to_schema,
)


class TestBPSchemas:
    def test_all_bp_doc_types_have_schemas(self):
        for doc_type, tables in BP_DOC_TYPE_TO_TABLE.items():
            header_table = tables[0]
            assert header_table in BP_PROCUREMENT_SCHEMAS, (
                f"BP doc_type '{doc_type}' header table '{header_table}' not in BP_PROCUREMENT_SCHEMAS"
            )
            line_table = tables[1]
            if line_table:
                assert line_table in BP_PROCUREMENT_SCHEMAS, (
                    f"BP doc_type '{doc_type}' line table '{line_table}' not in BP_PROCUREMENT_SCHEMAS"
                )

    def test_bp_invoice_schema_has_required_fields(self):
        schema = BP_PROCUREMENT_SCHEMAS["proc.bp_invoice"]
        assert "invoice_id" in schema.required
        assert "supplier_id" in schema.required
        assert "invoice_id" in schema.columns
        assert "invoice_total_incl_tax" in schema.columns

    def test_bp_purchase_order_schema(self):
        schema = BP_PROCUREMENT_SCHEMAS["proc.bp_purchase_order"]
        assert "po_id" in schema.required
        assert "supplier_name" in schema.required
        assert "total_amount" in schema.columns

    def test_bp_quote_schema(self):
        schema = BP_PROCUREMENT_SCHEMAS["proc.bp_quote"]
        assert "quote_id" in schema.required
        assert "supplier_id" in schema.required

    def test_bp_contracts_schema(self):
        schema = BP_PROCUREMENT_SCHEMAS["proc.bp_contracts"]
        assert "contract_id" in schema.required
        assert "supplier_id" in schema.required

    def test_map_columns_to_bp_schema(self):
        mapping = map_columns_to_schema(
            ["invoice_id", "supplier_id", "invoice_total_incl_tax", "currency"],
            "proc.bp_invoice",
        )
        assert mapping["invoice_id"] == "invoice_id"
        assert mapping["supplier_id"] == "supplier_id"
        assert mapping["currency"] == "currency"

    def test_map_columns_bp_synonym_match(self):
        mapping = map_columns_to_schema(
            ["invoice number", "vendor name"],
            "proc.bp_invoice",
        )
        assert mapping.get("invoice number") == "invoice_id"

    def test_bp_po_line_items_schema(self):
        schema = BP_PROCUREMENT_SCHEMAS["proc.bp_po_line_items"]
        assert "po_id" in schema.required
        assert "line_number" in schema.required
        assert "item_description" in schema.columns


class TestVendorProfile:
    def test_vendor_profile_dataclass(self):
        from services.vendor_profile_service import VendorProfile

        profile = VendorProfile(
            supplier_name="ACME Corp",
            doc_type="Invoice",
            currency_hint="USD",
            date_format_hint="MM/DD/YYYY",
            label_overrides={"invoice_total_incl_tax": ["Grand Total"]},
            extraction_count=5,
        )
        assert profile.supplier_name == "ACME Corp"
        assert profile.currency_hint == "USD"
        assert profile.extraction_count == 5
        assert "invoice_total_incl_tax" in profile.label_overrides

    def test_vendor_profile_defaults(self):
        from services.vendor_profile_service import VendorProfile

        profile = VendorProfile()
        assert profile.supplier_name == ""
        assert profile.label_overrides == {}
        assert profile.extraction_count == 0
