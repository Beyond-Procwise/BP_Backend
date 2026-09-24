"""AI translation endpoints.

GET  /i18n/languages            every language, searchable; English + the browser's languages pinned
POST /i18n/languages/resolve    a typed language name -> a registry entry or a custom code
POST /i18n/strings              UI strings: cached translations now, the rest queued (screen first)
POST /i18n/translate            dynamic content, translated on request (signed-in)
GET/PUT /i18n/preference        the signed-in user's language

Registered outside the authenticated router list: the sign-in page is translated too. Only
cache READS are open. Queuing model work needs an identified caller unless auth is off.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

import api.auth as auth
import src.services.i18n as i18n
from services import output_safety as osafe
from src.services.i18n import preference
from src.services.i18n.filler import BACKGROUND, SCREEN

router = APIRouter(prefix="/i18n", tags=["i18n"])

_MAX_KEYS = 2000
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


def _may_fill(request: Request) -> bool:
    """True when this caller may queue model work."""
    if auth.auth_mode() == "off":
        return True
    if not (request.headers.get("authorization") or "").strip():
        return False
    auth.require_user(request)  # 401 on a bad token
    return True


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
def resolve(body: ResolveIn) -> dict[str, Any]:
    try:
        return i18n.get_registry().resolve(body.text).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/strings", summary="UI strings in a language: cached ones now, the rest queued")
def strings(body: StringsIn, request: Request) -> dict[str, Any]:
    if len(body.strings) > _MAX_KEYS or any(len(v) > _MAX_UI_TEXT for v in body.strings.values()):
        raise HTTPException(status_code=413, detail=f"at most {_MAX_KEYS} strings of {_MAX_UI_TEXT} characters")
    language = _language(body.lang, body.lang_name)
    hits, missing = i18n.get_service().cached(language.code, body.strings)
    # A translation the output-safety layer would rewrite is served as English instead:
    # the screen shows the source rather than "[withheld]".
    scrubbed = osafe.scrub_payload(dict(hits), where="i18n.strings")
    withheld = sorted(k for k in hits if scrubbed.get(k) != hits[k])
    for k in withheld:
        hits.pop(k)
    if missing and _may_fill(request):
        i18n.get_filler().enqueue(language.code, {k: body.strings[k] for k in missing},
                                  SCREEN if body.priority == "screen" else BACKGROUND, body.lang_name)
    return {"lang": language.code, "dir": language.dir, "translations": hits,
            "pending": missing, "complete": not missing, "withheld": withheld}


@router.post("/translate", summary="Translate dynamic content on request")
def translate(body: TranslateIn, principal=Depends(auth.require_user)) -> dict[str, Any]:
    if any(len(t) > _MAX_DYNAMIC_TEXT for t in body.texts):
        raise HTTPException(status_code=413, detail=f"each text must be at most {_MAX_DYNAMIC_TEXT} characters")
    language = _language(body.lang, body.lang_name)
    keys = {f"t{i}": t for i, t in enumerate(body.texts)}
    result = i18n.get_service().translate(language.code, keys, lang_name=body.lang_name)
    return {"translations": [result.translations[k] for k in keys],
            "failed": sorted(int(k[1:]) for k in result.failed)}


def _pref_value(code: str, name: Optional[str]) -> dict:
    language = _language(code, name)
    return {"code": language.code, "name": name if language.custom and name else language.label()}


@router.get("/preference", summary="The signed-in user's language")
def get_preference(principal=Depends(auth.require_user)) -> dict[str, Any]:
    if principal is None:
        return {"language": None, "stored": False}
    return {"language": preference.get_language(principal.subject), "stored": True}


@router.put("/preference", summary="Save the signed-in user's language")
def put_preference(body: PreferenceIn, principal=Depends(auth.require_user)) -> dict[str, Any]:
    value = _pref_value(body.code, body.name)
    if principal is None:
        return {"language": value, "stored": False}
    preference.set_language(principal.subject, value)
    return {"language": value, "stored": True}
