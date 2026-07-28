# src/api/routers/models.py
"""What models this installation offers.

Exists so the workspace can show a model control ONLY when there is a genuine
choice to make. With one model offered the response has one entry, the UI renders
nothing, and the user is never asked a question they have no basis to answer.

Model names are not secrets here — this endpoint's whole job is to name them to
the operator choosing one — so ``/models`` is exempt from the response scrubber
(see ``_OPERATOR_PATHS`` in api/main.py). Nothing else about a model leaks: the
provider-facing reference is deliberately NOT returned to the browser. The client
sends back a key; the server resolves the reference.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from repositories import model_catalogue_repo as repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("")
def list_models():
    """Offered models, standard first.

    ``selectable`` is false for a metered model whose key is not configured: it is
    shown, with its reason, rather than hidden — an operator who wonders why their
    paid model is missing deserves an answer, not an empty list.
    """
    try:
        rows = repo.list_offered()
    except Exception:  # noqa: BLE001
        # A model catalogue that cannot be read must not take the workspace down.
        # No models offered == no override control, which is the safe default.
        logger.exception("model catalogue unavailable")
        return {"models": [], "error": "The model list is unavailable."}

    return {
        "models": [
            {
                "model_key": r["model_key"],
                "display_name": r["display_name"],
                "is_default": r["is_default"],
                "selectable": r["selectable"],
                "notes": r["notes"],
                "reason": None if r["selectable"] else "Needs an API key to be configured.",
            }
            for r in rows
        ]
    }
