"""Typed input/output models for the benchmark pricing engine.

The output model is the audit record: every intermediate the engine computes
is a field here, so a reviewer can reconstruct any number by hand.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BenchmarkMethod(str, Enum):
    WEIGHTED = "weighted"
    SIMPLE = "simple"
    MEDIAN = "median"


# The Excel prototype spells methods differently; accept both spellings.
_METHOD_ALIASES = {
    "weighted average": BenchmarkMethod.WEIGHTED,
    "simple average": BenchmarkMethod.SIMPLE,
    "median": BenchmarkMethod.MEDIAN,
    "weighted": BenchmarkMethod.WEIGHTED,
    "simple": BenchmarkMethod.SIMPLE,
}


class BenchmarkSettings(BaseModel):
    """All tunables — every value injectable, no magic numbers in the engine."""

    model_config = ConfigDict(frozen=True)

    min_data_points: int = Field(default=3, ge=1)
    method: BenchmarkMethod = BenchmarkMethod.WEIGHTED
    volume_elasticity: float = 0.06
    volume_min_factor: float = 0.85
    volume_max_factor: float = 1.15
    spec_factor: float = 0.03
    spec_min_factor: float = 0.90
    spec_max_factor: float = 1.20
    sla_factor: float = 0.025
    sla_min_factor: float = 0.90
    sla_max_factor: float = 1.25
    default_location_index: float = 1.0
    default_current_index: float = 1.0

    @field_validator("method", mode="before")
    @classmethod
    def _coerce_method(cls, value):
        if isinstance(value, str):
            return _METHOD_ALIASES.get(value.strip().lower(), value)
        return value


class QuoteLine(BaseModel):
    """One supplier quote line under analysis."""

    model_config = ConfigDict(frozen=True)

    deal_id: str
    item_name: str
    supplier_name: str = ""
    quote_ref: str = ""
    category: str = ""
    quantity: float
    uom: str
    currency: str
    location: str
    requested_spec_score: float = Field(ge=1, le=10)
    service_level: str = ""
    requested_sla_score: float = Field(ge=1, le=10)
    index_id: str
    quoted_unit_price: float
    delivery_cost: float = 0.0
    implementation_cost: float = 0.0
    support_cost: float = 0.0
    risk_premium: float = 0.0
    discount_rebate: float = 0.0  # positive number that SUBTRACTS from totals
    # INFORMATIONAL ONLY — echoed to the result, never used in any formula.
    supplier_risk_penalty: Optional[float] = None
    contract_risk_penalty: Optional[float] = None
    strategic_supplier_bonus: Optional[float] = None


class BenchmarkPoint(BaseModel):
    """One historical price observation (internal or external — pooled)."""

    model_config = ConfigDict(frozen=True)

    benchmark_point_id: str
    source: Literal["internal", "external"]
    item_name: str
    uom: str
    currency: str
    include: bool
    raw_unit_price: float
    source_weight: float = Field(ge=0)  # prototype range 0.5–1.5; 0 tolerated, guarded
    specification_score: float
    location_cost_index: float  # static value stored on the row (Decision 2)
    sla_score: float
    historical_quantity: float
    index_value_at_price_date: float

    @field_validator("include", mode="before")
    @classmethod
    def _coerce_include(cls, value):
        if isinstance(value, str):
            return value.strip().lower() == "yes"
        return value


class BenchmarkResult(BaseModel):
    """Full audit record for one quote line. All computed fields are None
    when the evidence gate fails (fail closed)."""

    model_config = ConfigDict(frozen=True)

    # -- inputs echoed --------------------------------------------------
    deal_id: str
    item_name: str
    supplier_name: str
    quote_ref: str
    category: str
    quantity: float
    uom: str
    currency: str
    location: str
    requested_spec_score: float
    service_level: str
    requested_sla_score: float
    index_id: str
    quoted_unit_price: float
    # -- evidence --------------------------------------------------------
    matched_point_ids: List[str]
    n_internal: int
    n_external: int
    n_total: int
    total_weight: float
    # -- raw benchmark candidates -----------------------------------------
    simple_benchmark: Optional[float] = None
    weighted_benchmark: Optional[float] = None
    median_benchmark: Optional[float] = None
    selected_benchmark: Optional[float] = None
    method_used: Optional[str] = None
    # -- weighted profiles -------------------------------------------------
    ref_quantity: Optional[float] = None
    avg_spec_score: Optional[float] = None
    avg_loc_index: Optional[float] = None
    avg_sla_score: Optional[float] = None
    avg_hist_index: Optional[float] = None
    # -- lookups (with audited fallbacks, Decision 3) ------------------------
    target_loc_index: Optional[float] = None
    current_index: Optional[float] = None
    fallbacks_used: List[str] = Field(default_factory=list)
    # -- adjustment factors ---------------------------------------------------
    volume_adjustment: Optional[float] = None
    spec_adjustment: Optional[float] = None
    location_adjustment: Optional[float] = None
    sla_adjustment: Optional[float] = None
    inflation_adjustment: Optional[float] = None
    combined_factor: Optional[float] = None
    # -- outputs -----------------------------------------------------------
    final_benchmark: Optional[float] = None
    quoted_total: Optional[float] = None
    benchmark_total: Optional[float] = None
    unit_variance_gbp: Optional[float] = None
    unit_variance_pct: Optional[float] = None  # fraction; ×100 for display
    total_cost_gap: Optional[float] = None  # positive = savings opportunity
    # -- verdict -------------------------------------------------------------
    confidence: str
    gated: bool
    # -- informational passthrough (never used in formulas) -------------------
    supplier_risk_penalty: Optional[float] = None
    contract_risk_penalty: Optional[float] = None
    strategic_supplier_bonus: Optional[float] = None
