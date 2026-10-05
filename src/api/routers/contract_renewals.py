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
