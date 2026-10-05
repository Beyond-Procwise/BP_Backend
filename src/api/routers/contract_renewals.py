"""GET /spendiq/contract-renewals — the renewals list screen.

Every active contract the expiry rule (proc.bp_rule, contract_expiry_bucket_check)
has something to say about: those inside a bucket, those already past their end
date, and those with no end date. Read-only; the evaluation is the one Today's
brief uses, so the two never disagree. All logic is in
``services/contract_expiry/renewals``.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_user
from src.services.contract_expiry import renewals

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spendiq", tags=["Renewals"])


@router.get("/contract-renewals", summary="Contracts by expiry bucket, with the rule's own edges")
def get_contract_renewals(principal: Any = Depends(require_user)) -> dict:
    try:
        return renewals.build()
    except Exception as exc:
        # An unreadable rule or database is a 503. An empty list would read as
        # "nothing is expiring", which is the one thing this screen must not say
        # when it does not know.
        logger.exception("contract-renewals failed")
        raise HTTPException(status_code=503, detail="contract renewals unavailable") from exc


@router.get("/active-contracts", summary="Live contracts the demand intake can offer as a renewal")
def get_active_contracts(principal: Any = Depends(require_user)) -> dict:
    try:
        rows = renewals.active_contracts()
    except Exception as exc:
        # The intake falls back to asking without a candidate contract; an error here
        # must not be dressed as "there are no live contracts".
        logger.exception("active-contracts failed")
        raise HTTPException(status_code=503, detail="contracts unavailable") from exc
    return {"total": len(rows), "contracts": rows}
