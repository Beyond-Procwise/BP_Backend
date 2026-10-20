"""Compliance numbers the gateway used to compute for itself.

GET /compliance/po-variance   invoices billed beyond their PO, from triage's findings

Called only by the Node gateway, with its key and the identity it verified (the same
scheme as the agent-policy routes), so this router is registered outside the bearer-token
list in api/main.py.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends

from api.routers.agent_policies import gateway_principal

from src.services.compliance.po_variance import po_variance

router = APIRouter(prefix="/compliance", tags=["Compliance"],
                   dependencies=[Depends(gateway_principal)])


@router.get("/po-variance", summary="Invoices billed beyond what their PO authorised")
def get_po_variance(deal_id: Optional[str] = None) -> dict[str, Any]:
    return po_variance(deal_id)
