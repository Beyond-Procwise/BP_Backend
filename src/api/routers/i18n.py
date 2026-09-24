"""AI translation endpoints.

GET  /i18n/languages            every language, searchable; English + the browser's languages pinned
POST /i18n/languages/resolve    a typed language name -> a registry entry or a custom code
POST /i18n/strings              UI strings: cached translations now, the rest queued (screen first)
POST /i18n/translate            dynamic content, translated on request (signed-in)
GET/PUT /i18n/preference        the signed-in user's language

Mounted behind require_user like every other router (tests/api/test_every_router_is_
authenticated.py), and every write names the principal. Translating the pre-sign-in pages
would need a public, cache-only read path; that is an auth exemption for the product owner
to approve, not something this router assumes.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

import api.auth as auth
from api.auth import require_user
import src.services.i18n as i18n
from services import output_safety as osafe
from src.services.i18n import preference
from src.services.i18n.filler import BACKGROUND, SCREEN

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/i18n", tags=["i18n"])

_MAX_KEYS = 2000
_MAX_KEY_LEN = 200
_MAX_UI_TEXT = 4000
_MAX_DYNAMIC_TEXT = 5000


class ResolveIn(BaseModel):
    text: str = Field(..., max_length=200)


class StringsIn(BaseModel):
    lang: str = Field(..., max_length=64)
    lang_name: Optional[str] = Field(None, max_length=60)
    strings: dict[str, str] = Field(default_factory=dict)
    priority: Literal["screen", "background"] = "screen"


class TranslateIn(BaseModel):
    lang: str = Field(..., max_length=64)
    lang_name: Optional[str] = Field(None, max_length=60)
    texts: list[str] = Field(..., max_length=50)


class PreferenceIn(BaseModel):
    code: str = Field(..., max_length=64)
    name: Optional[str] = Field(None, max_length=60)


def _language(code: str, name: Optional[str]):
    try:
        return i18n.get_service().language(code, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"unknown language code {code!r}") from exc


@router.get("/languages", summary="Every language, searchable, with English and the browser's languages pinned")
def languages(request: Request, q: str = Query("", max_length=60),
              limit: int = Query(1000, ge=1, le=1000)) -> dict[str, Any]:
    reg = i18n.get_registry()
    pinned = reg.pin_codes(request.headers.get("accept-language", ""))
    return {"languages": [L.to_dict() for L in reg.search(q, pinned=pinned, limit=limit)], "pinned": pinned}


@router.post("/languages/resolve", summary="Turn a typed language name into a code")
def resolve(body: ResolveIn, principal=Depends(require_user)) -> dict[str, Any]:
    try:
        return i18n.get_registry().resolve(body.text).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/strings", summary="UI strings in a language: cached ones now, the rest queued")
def strings(body: StringsIn, principal=Depends(require_user)) -> dict[str, Any]:
    if (len(body.strings) > _MAX_KEYS or any(len(k) > _MAX_KEY_LEN for k in body.strings)
            or any(len(v) > _MAX_UI_TEXT for v in body.strings.values())):
        raise HTTPException(status_code=413, detail=f"at most {_MAX_KEYS} strings, keys of {_MAX_KEY_LEN} "
                                                    f"and texts of {_MAX_UI_TEXT} characters")
    language = _language(body.lang, body.lang_name)
    # pending = still to translate (queued); failed = tried twice recently, served in English
    # until the back-off passes. The client stops polling once pending is empty.
    hits, pending, failed = i18n.get_service().status(language.code, body.strings)
    # A translation the output-safety layer would rewrite is served as English instead:
    # the screen shows the source rather than "[withheld]".
    scrubbed = osafe.scrub_payload(dict(hits), where="i18n.strings")
    withheld = sorted(k for k in hits if scrubbed.get(k) != hits[k])
    for k in withheld:
        hits.pop(k)
    if pending:
        i18n.get_filler().enqueue(language.code, {k: body.strings[k] for k in pending},
                                  SCREEN if body.priority == "screen" else BACKGROUND, body.lang_name)
    return {"lang": language.code, "dir": language.dir, "translations": hits,
            "pending": pending, "failed": failed, "queued": bool(pending),
            "complete": not pending, "withheld": withheld}


@router.post("/translate", summary="Translate dynamic content on request")
def translate(body: TranslateIn, principal=Depends(require_user)) -> dict[str, Any]:
    if any(len(t) > _MAX_DYNAMIC_TEXT for t in body.texts):
        raise HTTPException(status_code=413, detail=f"each text must be at most {_MAX_DYNAMIC_TEXT} characters")
    language = _language(body.lang, body.lang_name)
    keys = {f"t{i}": t for i, t in enumerate(body.texts)}
    result = i18n.get_service().translate(language.code, keys, lang_name=body.lang_name)
    out = [result.translations[k] for k in keys]
    failed = {int(k[1:]) for k in result.failed}
    # A translation the output-safety layer would rewrite goes back to its source text.
    scrubbed = osafe.scrub_payload(list(out), where="i18n.translate")
    for i, (before, after) in enumerate(zip(out, scrubbed)):
        if before != after:
            out[i] = body.texts[i]
            failed.add(i)
    return {"translations": out, "failed": sorted(failed)}


def _pref_value(code: str, name: Optional[str]) -> dict:
    language = _language(code, name)
    return {"code": language.code, "name": name if language.custom and name else language.label()}


@router.get("/preference", summary="The signed-in user's language")
def get_preference(principal=Depends(require_user)) -> dict[str, Any]:
    if principal is None:
        return {"language": None, "stored": False}
    try:
        return {"language": preference.get_language(principal.subject), "stored": True}
    except Exception as exc:  # the UI falls back to local storage; never a 500 here
        logger.warning("i18n: reading the language preference failed: %s", exc)
        return {"language": None, "stored": False}


@router.put("/preference", summary="Save the signed-in user's language")
def put_preference(body: PreferenceIn, principal=Depends(require_user)) -> dict[str, Any]:
    value = _pref_value(body.code, body.name)
    if principal is None:
        return {"language": value, "stored": False}
    try:
        preference.set_language(principal.subject, value)
    except Exception as exc:
        logger.warning("i18n: saving the language preference failed: %s", exc)
        return {"language": value, "stored": False}
    return {"language": value, "stored": True}
