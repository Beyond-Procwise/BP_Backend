"""Tests for CSV/Excel tabular extraction pipeline."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.procurement_schema import (
    normalize_category,
    map_columns_to_schema,
    CATEGORY_TO_DOC_TYPE,
    DOC_TYPE_TO_TABLE,
)


class TestNormalizeCategory:
    def test_known_categories(self):
        assert normalize_category("invoice") == "Invoice"
        assert normalize_category("po") == "Purchase_Order"
        assert normalize_category("quote") == "Quote"
        assert normalize_category("quotes") == "Quote"
        assert normalize_category("contract") == "Contract"
        assert normalize_category("purchase_order") == "Purchase_Order"

    def test_case_insensitive(self):
        assert normalize_category("INVOICE") == "Invoice"
        assert normalize_category("Po") == "Purchase_Order"
        assert normalize_category("  Quote  ") == "Quote"

    def test_unknown_returns_none(self):
        assert normalize_category("spend") is None
        assert normalize_category("item") is None
        assert normalize_category("") is None
        assert normalize_category("unknown_type") is None


class TestMapColumnsToSchema:
    def test_exact_match(self):
        mapping = map_columns_to_schema(
            ["invoice_id", "supplier_name", "invoice_total_incl_tax"],
            "proc.invoice_agent",
        )
        assert mapping["invoice_id"] == "invoice_id"
        assert mapping["supplier_name"] == "supplier_name"
        assert mapping["invoice_total_incl_tax"] == "invoice_total_incl_tax"

    def test_synonym_match(self):
        mapping = map_columns_to_schema(
            ["invoice number", "vendor name"],
            "proc.invoice_agent",
        )
        assert mapping.get("invoice number") == "invoice_id"
        assert mapping.get("vendor name") == "vendor_name"

    def test_no_match_excluded(self):
        mapping = map_columns_to_schema(
            ["totally_random_column", "invoice_id"],
            "proc.invoice_agent",
        )
        assert "totally_random_column" not in mapping
        assert mapping["invoice_id"] == "invoice_id"

    def test_unknown_table_returns_empty(self):
        mapping = map_columns_to_schema(
            ["invoice_id"],
            "proc.nonexistent_table",
        )
        assert mapping == {}

    def test_po_columns(self):
        mapping = map_columns_to_schema(
            ["po_id", "supplier_name", "total_amount"],
            "proc.purchase_order_agent",
        )
        assert "po_id" in mapping
        assert "supplier_name" in mapping


class TestCategoryToDocTypeIntegration:
    def test_all_categories_have_tables(self):
        for category, doc_type in CATEGORY_TO_DOC_TYPE.items():
            assert doc_type in DOC_TYPE_TO_TABLE, (
                f"Category '{category}' maps to '{doc_type}' "
                f"which is not in DOC_TYPE_TO_TABLE"
            )


class TestKGIngestionService:
    def test_label_from_category(self):
        from services.kg_ingestion_service import _label_from_category

        assert _label_from_category("spend") == "Spend"
        assert _label_from_category("item_master") == "ItemMaster"
        assert _label_from_category("") == "UnknownData"

    def test_find_column(self):
        import re
        from services.kg_ingestion_service import KGIngestionService

        pattern = re.compile(r"^(supplier_id|vendor_id)$", re.IGNORECASE)
        assert KGIngestionService._find_column(
            ["name", "supplier_id", "amount"], pattern
        ) == "supplier_id"
        assert KGIngestionService._find_column(
            ["name", "amount"], pattern
        ) is None

    def test_ingest_returns_zero_for_empty_df(self):
        from services.kg_ingestion_service import KGIngestionService

        service = KGIngestionService.__new__(KGIngestionService)
        service._agent_nick = None
        service._driver = MagicMock()
        result = service.ingest_dataframe(pd.DataFrame(), "spend")
        assert result == 0
