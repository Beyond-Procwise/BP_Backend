"""Map the generated organisation and catalogue onto their uicanvas tables.

Sibling of persist.py, which does the same for documents in bp_sqldb. Same
idiom: a COLUMNS entry per table and a pure `rows_for_org` that touches no
database.

These tables carry almost no NOT NULL constraint -- `business_unit` requires
only `business_unit_id`, `cost_centre` only `cost_centre_level_id`, `item`
nothing at all. A load that put NULL in every price would be accepted in
silence. REQUIRED is the declared set that must carry a value for the row to
mean anything, and `check_required` is the only thing standing between a
mapping mistake and a table full of nulls.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from scripts.testdata.catalogue import CatalogueItem
from scripts.testdata.loader import MissingRequiredValue, _check_required
from scripts.testdata.org import BU_FUNCTIONS, BusinessUnit, CostCentre

__all__ = [
    "COLUMNS", "REQUIRED", "TABLES", "MissingRequiredValue",
    "check_required", "rows_for_org",
]

MARKER = "testdata"

# Fixed rather than wall-clock: the build must reproduce byte-for-byte from a
# seed, and now() would break that.
STAMP = datetime(2026, 7, 27, 0, 0, 0)

# The schema carries an identifier for business-unit level 1 only. The level-1
# names are a fixed tuple in org.py, so mapping them by position gives a stable
# identifier without module-level mutable state.
BU_LEVEL1_IDS: dict[str, str] = {
    name: f"BU-L1-{index + 1:03d}" for index, name in enumerate(BU_FUNCTIONS)
}

COLUMNS: dict[str, tuple[str, ...]] = {
    "business_unit": (
        "business_unit_level_1_id", "business_unit_level_1", "business_unit_level_2",
        "business_unit_level_3", "business_unit_level_4", "business_unit_level_5",
        "business_unit_id", "business_unit_head_name", "business_unit_head_email",
        "region", "bu_status", "notes", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
    "cost_centre": (
        "cost_centre_level_id", "cost_centre_level_1", "cost_centre_level_2",
        "cost_centre_level_3", "cost_centre_level_4", "cost_centre_level_5",
        "cost_centre_level_6", "business_unit_id", "finance_account_code",
        "cost_centre_manager_name", "cost_centre_manager_email", "is_active",
        "spend_threshold_limit", "currency", "po_id", "invoice_id",
        "budget_allocated_annual", "actual_spend_ytd", "forecast_spend_annual",
        "cost_centre_type", "linked_category_level_5_id", "notes",
        "created_date", "created_by", "last_modified_by", "last_modified_date",
    ),
    "item": (
        "item_id", "item_name", "category_id", "unit", "standard_price",
        "currency", "preferred_supplier_id", "manufacturer", "brand",
        "spec_sheet_url", "uom_conversion", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
}

REQUIRED: dict[str, tuple[str, ...]] = {
    "business_unit": (
        "business_unit_level_1_id", "business_unit_level_1", "business_unit_level_5",
        "business_unit_id", "business_unit_head_email", "bu_status",
    ),
    "cost_centre": (
        "cost_centre_level_id", "cost_centre_level_1", "cost_centre_level_6",
        "business_unit_id", "currency", "cost_centre_type",
        "linked_category_level_5_id",
    ),
    "item": (
        "item_id", "item_name", "category_id", "unit", "standard_price",
        "currency", "preferred_supplier_id",
    ),
}

TABLES: tuple[str, ...] = tuple(COLUMNS)


def check_required(table: str, rows: Sequence[Sequence]) -> None:
    """Raise unless every row carries a value in each of the table's required columns."""
    _check_required(table, COLUMNS[table], REQUIRED[table], rows)


def _business_unit_row(unit: BusinessUnit) -> list:
    return [
        BU_LEVEL1_IDS.get(unit.l1, "BU-L1-000"),
        unit.l1, unit.l2, unit.l3, unit.l4, unit.l5,
        unit.bu_id, unit.head_name, unit.head_email,
        unit.region, unit.status, None,
        STAMP, MARKER, MARKER, STAMP,
    ]


def _cost_centre_row(centre: CostCentre) -> list:
    return [
        centre.cc_id, *centre.levels,
        centre.bu_id, centre.finance_account_code,
        centre.manager_name, centre.manager_email, centre.is_active,
        centre.spend_threshold_limit, centre.currency,
        # Document links belong to stage S3. Inventing them here would assert a
        # relationship no document backs.
        None, None,
        centre.budget_allocated_annual, centre.actual_spend_ytd,
        centre.forecast_spend_annual, centre.cost_centre_type,
        centre.linked_category_level_5_id, None,
        STAMP, MARKER, MARKER, STAMP,
    ]


def _item_row(item: CatalogueItem) -> list:
    return [
        item.item_id, item.description, item.leaf.l5_id,
        item.unit_of_measure, item.base_price, item.currency,
        item.preferred_supplier_id,
        # Manufacturer, brand, spec sheet and UOM conversion are not modelled by
        # the generator. Inventing them would put unverifiable strings in
        # columns nothing reads.
        None, None, None, None,
        STAMP, MARKER, MARKER, STAMP,
    ]


def rows_for_org(
    units: Sequence[BusinessUnit],
    centres: Sequence[CostCentre],
    items: Sequence[CatalogueItem],
) -> dict[str, list[list]]:
    """Every table's rows, in COLUMNS order. Pure: no database contact."""
    return {
        "business_unit": [_business_unit_row(unit) for unit in units],
        "cost_centre": [_cost_centre_row(centre) for centre in centres],
        "item": [_item_row(item) for item in items],
    }
