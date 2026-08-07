"""The fact model — the atomic unit that survives the seam.

A ``CommercialFact`` records one number a document actually stated, together
with everything needed to reconstruct it later: what kind of measure it is,
what it is priced per, the currency and the rate used, and the document span it
came from. Provenance is mandatory *in the type*, so a number with no evidence
behind it cannot be built at all — not merely discouraged by convention.

Two rules shape the design and are worth stating plainly:

  * The model is a record, not a calculator. Nothing here derives a value from
    other fields. Anything computed is computed by a named deterministic engine
    a reviewer can point at, rather than by a property nobody reviews.
  * The model carries no interpretation. There is no rationale or explanation
    field anywhere in this module, because such a field is the seam through
    which a language model's reading would contaminate the fact base.

Conventions follow ``src/services/benchmark/models.py``.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.services.facts.concept_codes import is_known


class ValueBasis(str, Enum):
    """Whose baseline the value is stated against."""

    AS_SUPPLIED = "as_supplied"
    BASELINE_CORRECTED = "baseline_corrected"
    NORMALISED = "normalised"


class ValidationState(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNVERIFIED = "unverified"
    INDETERMINATE = "indeterminate"


class MeasureRole(str, Enum):
    """What kind of measure the number is.

    A number's role cannot be inferred from its field name. "Price" may mean a
    unit rate or an extended total; "volume" may mean a count or a physical
    measure. The schemas here already distinguish ``unit_price`` from
    ``line_total`` by name, and a real bug in this codebase still booked a line
    total as a unit price — correct names, wrong values. Carrying the role
    explicitly is what stops a later comparison from putting one supplier's
    unit rate against another's line total and returning a confident wrong
    answer.
    """

    UNIT_RATE = "unit_rate"
    EXTENDED_LINE = "extended_line"
    DOCUMENT_TOTAL = "document_total"
    QUANTITY = "quantity"
    TAX = "tax"
    DISCOUNT = "discount"


class ArithmeticState(str, Enum):
    """Whether ``quantity x unit_rate = extended_line`` verified the role.

    The two untestable states exist because abstaining is not the same as
    passing. Roughly a fifth of real lines have ``quantity = 1``, where a unit
    rate and a total are numerically identical and no arithmetic can separate
    them; that is exactly the blind spot that let the historical
    unit-price-as-total bug ship unnoticed. A fact whose role could not be
    verified must not look identical to one that was checked.
    """

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    UNTESTABLE_QUANTITY_ONE = "untestable_quantity_one"
    UNTESTABLE_MISSING_INPUT = "untestable_missing_input"


def _non_blank(value: str, field: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


class FactProvenance(BaseModel):
    """Where one fact's value was read from.

    ``document_id`` and ``locator`` are required and non-blank: provenance that
    cannot point at a place in a document is not provenance, and accepting a
    blank one would satisfy the mandatory-provenance rule while defeating it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    field_path: str
    locator: str
    doc_type: Optional[str] = None
    extraction_id: Optional[str] = None
    page: Optional[int] = None
    verbatim_snippet: Optional[str] = None
    extracted_at: Optional[datetime] = None
    model: Optional[str] = None
    confidence: Optional[float] = None

    @field_validator("document_id")
    @classmethod
    def _document_id_non_blank(cls, v: str) -> str:
        return _non_blank(v, "document_id")

    @field_validator("locator")
    @classmethod
    def _locator_non_blank(cls, v: str) -> str:
        return _non_blank(v, "locator")

    @field_validator("field_path")
    @classmethod
    def _field_path_non_blank(cls, v: str) -> str:
        return _non_blank(v, "field_path")


class CommercialFact(BaseModel):
    """One provenanced commercial number."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- identity -----------------------------------------------------------
    fact_id: str
    tenant_id: str = "default"
    fact_type: str
    concept_code: Optional[str] = None
    source_doc_type: Optional[str] = None
    source_doc_pk: Optional[str] = None
    document_id: Optional[str] = None
    document_version: Optional[str] = None
    line_no: Optional[int] = None

    # --- measure semantics (F6) ---------------------------------------------
    # measure_role says what KIND of number this is; basis_uom says what a unit
    # rate is per; arithmetic_state records whether the role was verified.
    measure_role: Optional[MeasureRole] = None
    basis_uom: Optional[str] = None
    # Deliberately no default. A default would let an unverified fact silently
    # claim the same standing as a checked one, which is the failure mode this
    # column exists to expose. The caller must state it, even to state None.
    arithmetic_state: Optional[ArithmeticState] = Field(...)

    # --- economics ----------------------------------------------------------
    unit_price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    extended_value: Optional[Decimal] = None
    tax_amount: Optional[Decimal] = None
    discount_amount: Optional[Decimal] = None
    uom: Optional[str] = None
    uom_normalised: Optional[str] = None
    uom_dimension: Optional[str] = None

    # --- currency -----------------------------------------------------------
    currency: Optional[str] = None
    base_currency: Optional[str] = None
    extended_value_base: Optional[Decimal] = None
    fx_rate: Optional[Decimal] = None
    fx_rate_date: Optional[datetime] = None
    fx_rate_source: Optional[str] = None

    # --- commercial identity ------------------------------------------------
    supplier_id: Optional[str] = None
    supplier_name: Optional[str] = None
    buyer_id: Optional[str] = None
    item_reference: Optional[str] = None
    item_description: Optional[str] = None
    # Nullable in this phase, and not blocked: proc.bp_category supplies an
    # L1~L2~L3 hierarchy keyed on item_description (49 rows on bp_sqldb, 0 on
    # bp_testdb). Populating these is a join the assembler could do; it is held
    # out of scope so the assembler keeps a single responsibility. Any future
    # population must fail closed rather than default.
    category_l1: Optional[str] = None
    category_l2: Optional[str] = None
    category_l3: Optional[str] = None
    category_l4: Optional[str] = None

    # --- term ---------------------------------------------------------------
    contract_id: Optional[str] = None
    term_start: Optional[datetime] = None
    term_end: Optional[datetime] = None
    term_months: Optional[int] = None
    billing_frequency: Optional[str] = None
    escalator_pct: Optional[Decimal] = None
    escalator_basis: Optional[str] = None
    escalator_cap_pct: Optional[Decimal] = None

    # --- allocation ---------------------------------------------------------
    cost_centre: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None

    # --- grouping -----------------------------------------------------------
    bundle_group_id: Optional[str] = None
    deal_id: Optional[str] = None

    # --- integrity ----------------------------------------------------------
    value_basis: ValueBasis = ValueBasis.AS_SUPPLIED
    validation_state: ValidationState = ValidationState.UNVERIFIED
    reason_codes: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None

    # --- provenance ---------------------------------------------------------
    provenance: List[FactProvenance] = Field(...)

    # --- bitemporal ---------------------------------------------------------
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    recorded_at: Optional[datetime] = None

    @field_validator("provenance")
    @classmethod
    def _provenance_must_be_non_empty(cls, v: List[FactProvenance]) -> List[FactProvenance]:
        """The load-bearing rule of this phase.

        Enforced here rather than at the call site because a call site can be
        forgotten. It is enforced a second time in the database by a deferred
        constraint trigger, because the backfill and any direct INSERT bypass
        this model entirely.
        """
        if not v:
            raise ValueError(
                "provenance must not be empty: a fact with no evidence behind it "
                "cannot be constructed"
            )
        return v

    @field_validator("concept_code")
    @classmethod
    def _concept_code_must_be_known(cls, v: Optional[str]) -> Optional[str]:
        """Reject a code no extraction schema declares.

        Silently accepting an unknown code is how a shadow vocabulary starts.
        """
        if v is None:
            return None
        if not is_known(v):
            raise ValueError(
                f"concept_code {v!r} is not a field declared by any extraction "
                "schema; add it to the schema rather than inventing a code here"
            )
        return v

    @model_validator(mode="after")
    def _measure_role_rules(self) -> "CommercialFact":
        """The cross-field rules that make a number comparable.

        A unit rate with no 'per what' is not comparable to anything, and a
        priced fact that does not say whether its role was verified is exactly
        the ambiguity this phase exists to remove.
        """
        role = self.measure_role

        if role is MeasureRole.UNIT_RATE and not (self.basis_uom or "").strip():
            raise ValueError(
                "a unit_rate fact must declare basis_uom: a rate with no unit "
                "behind it is a number, not a fact"
            )

        priced = {
            MeasureRole.UNIT_RATE,
            MeasureRole.EXTENDED_LINE,
            MeasureRole.DOCUMENT_TOTAL,
        }
        if role in priced and self.arithmetic_state is None:
            raise ValueError(
                f"a {role.value} fact must declare arithmetic_state, even to "
                "declare that its role could not be verified"
            )

        return self
