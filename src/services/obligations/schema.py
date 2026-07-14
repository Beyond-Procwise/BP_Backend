"""Schema for the contract obligation hypergraph.

The type fields are Enums on purpose. Extraction runs under Ollama's grammar-constrained
decoding, so an Enum is *enforced* at the token level rather than merely requested. With a
free-text type field the model invents a different vocabulary per chunk ("hyperedge",
"contractual obligation", "Logistics", "Quantity"...), which makes the result unqueryable.

``clause_ref`` and ``source_quote`` are separate fields, also on purpose. Given one field the
model writes a clause number into it ("27.4"), which then trivially satisfies a naive "is this
string in the document?" check — a false pass. Splitting them forces a real span.
"""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    party = "party"
    obligation = "obligation"
    trigger = "trigger"
    penalty = "penalty"
    deadline = "deadline"
    goods = "goods"
    document = "document"


class RelType(str, Enum):
    must_perform = "must_perform"
    triggered_by = "triggered_by"
    penalised_by = "penalised_by"
    entitled_to = "entitled_to"
    risk_transfer = "risk_transfer"


class ContractEntity(BaseModel):
    name: str = Field(description="Name exactly as written in the contract")
    type: EntityType


class ContractObligation(BaseModel):
    """One hyperedge: an obligation binding many parties at once."""

    name: str = Field(description="Short name for this obligation")
    type: RelType
    participants: List[str] = Field(
        description="Names of ALL entities bound by this obligation"
    )
    clause_ref: str = Field(description="Clause number, e.g. '27.4'")
    source_quote: str = Field(
        description=(
            "The COMPLETE SENTENCE from the contract stating this obligation, copied WORD "
            "FOR WORD. At least 8 words. Do NOT write a clause number here. Do NOT "
            "abbreviate with '...'."
        )
    )
