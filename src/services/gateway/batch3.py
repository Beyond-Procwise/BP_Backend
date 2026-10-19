"""Batch 3: supply-chain, reseller and customer-pricing questions that cannot be answered yet.

Each is declared with its real parameter contract and always returns UNSUPPORTED with the specific thing it is waiting on. They
run no query and read nothing: there is no inventory, shipment, reseller, programme, tier or customer-agreement data in the
estate, and inventing an answer is not an option (rule: if no function exists, say so and log it to the function backlog).
When the blocking decision is made, the body is replaced and the function is added to an agent allowlist in a reviewed change.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Optional

from pydantic import Field

from .common import _P
from .context import CallerContext
from .envelope import GatewayFault, Raw
from .gateway import Backends


def _unsupported(reason: str):
    def _fn(ctx: CallerContext, p, backends: Optional[Backends]) -> Raw:
        raise GatewayFault("UNSUPPORTED", f"not supported yet; waiting on {reason}. Logged to the function backlog.")
    return _fn


class InventoryParams(_P):
    sku: str = Field(min_length=1, max_length=80)
    location: Optional[str] = Field(default=None, max_length=80)


class ShipmentParams(_P):
    po_id: Optional[str] = Field(default=None, max_length=40)
    shipment_id: Optional[str] = Field(default=None, max_length=80)


class LandedCostParams(_P):
    sku: str = Field(min_length=1, max_length=80)
    supplier_id: str = Field(min_length=1, max_length=80)
    quantity: Decimal = Field(gt=0)
    incoterm: Optional[str] = Field(default=None, max_length=10)


class ResellerParams(_P):
    reseller_id: str = Field(min_length=1, max_length=80)


class CustomerAgreementParams(_P):
    customer_id: str = Field(min_length=1, max_length=80)
    date: dt.date


DECLARED = {
    "get_inventory_position": (InventoryParams, _unsupported(
        "a product decision on where stock data comes from (no inventory tables exist)")),
    "get_shipment_status": (ShipmentParams, _unsupported(
        "a product decision on carrier or ERP tracking (none is connected)")),
    "get_landed_cost": (LandedCostParams, _unsupported(
        "your definition of landed cost (freight, duty, insurance, handling) and a formula in the register")),
    "get_reseller_programme_terms": (ResellerParams, _unsupported(
        "the reseller/tenant model (no reseller, programme or user-to-reseller mapping exists)")),
    "get_partner_tier": (ResellerParams, _unsupported(
        "the reseller/tenant model (no partner tier data exists)")),
    "get_customer_pricing_agreement": (CustomerAgreementParams, _unsupported(
        "the reseller/tenant model and a place to hold customer-specific price agreements")),
}
