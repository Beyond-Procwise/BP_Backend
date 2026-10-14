"""One mapping from sell-side service errors to HTTP, shared by /catalog and /sales."""
from __future__ import annotations

from contextlib import contextmanager

from fastapi import HTTPException

from src.services.sell_side._db import NotFound, StateConflict
from src.services.sell_side.quote_render import NotCustomerReady


@contextmanager
def http_errors():
    try:
        yield
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except (StateConflict, NotCustomerReady) as exc:   # before ValueError: both subclass it
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


def money_json(obj):
    """Money leaves as a string. FastAPI's default encoder turns Decimal('16.00')
    into the float 16.0 (verified: fastapi 0.140.5), which drops the scale and
    invites float arithmetic on a price downstream."""
    from decimal import Decimal

    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    return JSONResponse(content=jsonable_encoder(obj, custom_encoder={Decimal: str}))
