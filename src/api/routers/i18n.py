"""AI translation endpoints.

GET  /i18n/languages            every language, searchable; English + the browser's languages pinned
POST /i18n/languages/resolve    a typed language name -> a registry entry or a custom code
POST /i18n/strings              UI strings: cached translations now, the rest queued (screen first)
POST /i18n/translate            dynamic content, translated on request (signed-in)
GET/PUT /i18n/preference        the signed-in user's language
GET  /i18n/audit                the translation audit trail (Admin: policy action 'configure')

GET  /i18n/public/{lang}        (public_router) the sign-in screens' text, signed-out

`router` is mounted behind require_user like every other router (tests/api/test_every_
router_is_authenticated.py), and every write names the principal. `public_router` is the one
exemption, approved by the product owner on 2026-09-24: it takes no text, never calls the
model, and serves only cached translations of the keys in proc.bp_i18n_public_key.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

import api.auth as auth
from api.auth import require_user
import src.services.i18n as i18n
from services import output_safety as osafe
from src.services import rbac
from src.services.i18n import audit, preference
from src.services.i18n.filler import BACKGROUND, SCREEN
from src.services.i18n.store import source_hash

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/i18n", tags=["i18n"])
public_router = APIRouter(prefix="/i18n/public", tags=["i18n"])

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
    reg, svc = i18n.get_registry(), i18n.get_service()
    pinned = reg.pin_codes(request.headers.get("accept-language", ""))
    verdicts = svc.store.language_statuses(svc.prompt_version, svc.provider.model)
    return {"languages": [_with_quality(L, verdicts.get(L.code)) for L in reg.search(q, pinned=pinned, limit=limit)],
            "pinned": pinned}


def _with_quality(language, verdict: Optional[dict]) -> dict:
    """The picker's view of a language: supported (the model did not refuse it) and
    experimental (tier 3 in config, or the model reported low confidence)."""
    verdict = verdict or {}
    supported = verdict.get("recognized") is not False
    confidence = verdict.get("confidence")
    return {**language.to_dict(), "supported": supported, "confidence": confidence,
            "experimental": (not supported) or confidence == "low" or language.tier >= 3}


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
    quality = i18n.get_service().quality(language.code, body.lang_name)
    # supported:false -> the UI keeps English and says the language isn't supported yet;
    # experimental:true -> the picker's Experimental label and a small on-screen quality note.
    return {"lang": language.code, "dir": language.dir, "translations": hits,
            "pending": pending, "failed": failed, "queued": bool(pending),
            "complete": not pending, "withheld": withheld, **quality}


@router.post("/translate", summary="Translate dynamic content on request")
def translate(body: TranslateIn, principal=Depends(require_user)) -> dict[str, Any]:
    if any(len(t) > _MAX_DYNAMIC_TEXT for t in body.texts):
        raise HTTPException(status_code=413, detail=f"each text must be at most {_MAX_DYNAMIC_TEXT} characters")
    language = _language(body.lang, body.lang_name)
    keys = {f"t{i}": t for i, t in enumerate(body.texts)}
    svc = i18n.get_service()
    result = svc.translate(language.code, keys, lang_name=body.lang_name)
    out = [result.translations[k] for k in keys]
    failed = {int(k[1:]) for k in result.failed}
    withheld = set()
    # A translation the output-safety layer would rewrite goes back to its source text.
    scrubbed = osafe.scrub_payload(list(out), where="i18n.translate")
    for i, (before, after) in enumerate(zip(out, scrubbed)):
        if before != after:
            out[i] = body.texts[i]
            withheld.add(i)
    # Audit before showing: who asked, and exactly what they will see. A translation that
    # cannot be traced is not shown; the person gets the source text instead.
    items = [{"source": body.texts[i], "shown": out[i],
              "status": "withheld" if i in withheld else "english_fallback" if i in failed else "translated"}
             for i in range(len(out))]
    try:
        audit.record_served(lang=language.code, requested_by=getattr(principal, "subject", None),
                            model=svc.provider.model, prompt_version=svc.prompt_version, items=items)
    except audit.AuditWriteError:
        logger.error("i18n: served translation could not be audited; showing the source text")
        return {"translations": list(body.texts), "failed": list(range(len(body.texts))), "audited": False,
                "supported": result.supported, "confidence": result.confidence,
                "experimental": svc.quality(language.code, body.lang_name)["experimental"]}
    return {"translations": out, "failed": sorted(failed | withheld), "audited": True,
            "supported": result.supported, "confidence": result.confidence,
            "experimental": svc.quality(language.code, body.lang_name)["experimental"]}


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


@router.get("/audit", summary="The translation audit trail (Admin)")
def audit_trail(lang: Optional[str] = Query(None, max_length=64),
                requested_by: Optional[str] = Query(None, max_length=200),
                action_type: Optional[str] = Query(None, max_length=64),
                since: Optional[str] = Query(None, max_length=40),
                until: Optional[str] = Query(None, max_length=40),
                limit: int = Query(200, ge=1, le=1000),
                principal=Depends(require_user)) -> dict[str, Any]:
    if not rbac.may(rbac.effective_role(principal), "configure"):
        raise HTTPException(status_code=403, detail="the translation audit trail needs the configure permission")
    events = audit.read_events(lang=lang, requested_by=requested_by, action_type=action_type,
                               since=since, until=until, limit=limit)
    return {"events": events}


# ---------------------------------------------------------------------------------------
# The one signed-out path. No input text, no model, a short-lived cache in front of the DB.
# ---------------------------------------------------------------------------------------
_PUBLIC_TTL = 300.0
_public_cache: dict[str, tuple[float, dict]] = {}
_public_lock = threading.Lock()


def reset_public_cache() -> None:
    with _public_lock:
        _public_cache.clear()


def _cached(key: str, build):
    now = time.monotonic()
    with _public_lock:
        hit = _public_cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    value = build()
    with _public_lock:
        _public_cache[key] = (now + _PUBLIC_TTL, value)
    return value


def _public_available() -> list[dict]:
    svc, reg = i18n.get_service(), i18n.get_registry()
    keys = svc.store.public_keys()
    hashes = {source_hash(v) for v in keys.values()}
    counts = svc.store.languages_with(list(hashes), svc.prompt_version, svc.provider.model) if hashes else {}
    verdicts = svc.store.language_statuses(svc.prompt_version, svc.provider.model)
    ready = [reg.get(c) for c, n in counts.items()
             if n >= len(hashes) and reg.get(c) and (verdicts.get(c) or {}).get("recognized") is not False]
    langs = [reg.get("en")] + sorted(ready, key=lambda L: L.english.casefold())
    out = []
    for L in langs:
        q = _with_quality(L, verdicts.get(L.code))
        out.append({"code": L.code, "label": L.label(), "dir": L.dir, "experimental": q["experimental"]})
    return out


@public_router.get("/{lang}", summary="Sign-in screen text in a language (signed-out, cache only)")
def public_strings(lang: str) -> dict[str, Any]:
    language = i18n.get_registry().get(lang)  # registry codes only: the cache below stays bounded
    if language is None:
        raise HTTPException(status_code=400, detail=f"unknown language code {lang!r}")

    def build() -> dict:
        svc = i18n.get_service()
        keys = svc.store.public_keys()
        hits, _missing = svc.cached(language.code, keys) if keys else ({}, [])
        if language.code.split("-")[0] == "en":
            hits = {}
        scrubbed = osafe.scrub_payload(dict(hits), where="i18n.public")
        hits = {k: v for k, v in hits.items() if scrubbed.get(k) == v}
        return {"lang": language.code, "dir": language.dir, "translations": hits,
                "available": _cached("_available", _public_available)}

    return _cached(language.code, build)
