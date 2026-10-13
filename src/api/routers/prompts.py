"""The prompt registry over HTTP: read, save a version, promote, roll back.

Every route asks the gate, and the gate writes down every attempt before it
proceeds -- allowed and refused alike. For the two routes that return a
prompt's wording, that record is the log of who read which version and when.

Saving never changes what is in use. Only promote and rollback do, and both
need a named approver; promote also needs one who is not the version's author
(the registry enforces that, not this file).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from services.prompt_engine import registry, templates

router = APIRouter(prefix="/prompts", tags=["Prompts"])

_STATUS = {
    registry.PromptNotFound: 404,
    registry.VersionNotFound: 404,
    registry.PointerUnresolved: 404,
    registry.InvalidKey: 422,
    registry.KindMismatch: 409,
    registry.SelfApproval: 403,
    registry.ApproverRequired: 401,
    registry.NothingToRollBackTo: 409,
}


def _refused(exc: registry.RegistryError) -> JSONResponse:
    """The refusal, with the contract's code.

    Returned, not raised: the application's handler for HTTPException turns a
    structured ``detail`` into its text form, and the code would arrive as part
    of a sentence instead of a field.

    The code is sent in lower case. Error answers are always scrubbed by the
    output-safety layer, and it withholds a word like VERSION_NOT_FOUND as if
    it were an environment variable. In-process callers get the exception and
    its code exactly as the contract spells it.
    """
    return JSONResponse(status_code=_STATUS.get(type(exc), 400),
                        content={"code": exc.code.lower(), "message": str(exc)})


def _named(principal: Any, what: str) -> str:
    subject = (getattr(principal, "subject", None) or "").strip()
    if not subject:
        raise HTTPException(status_code=401, detail=f"{what} needs a signed-in person")
    return subject


class SaveBody(BaseModel):
    template: str
    kind: str = "prompt"
    owner: Optional[str] = None
    variables: List[Dict[str, Any]] = Field(default_factory=list)
    fragment_refs: List[Dict[str, Any]] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)
    change_note: Optional[str] = None


class PromoteBody(BaseModel):
    version: int
    note: Optional[str] = None


class RollbackBody(BaseModel):
    to_version: Optional[int] = None
    note: Optional[str] = None


@router.get("")
def list_prompts(principal=Depends(require_user)):
    """Names, kinds and version numbers. No wording."""
    gate("prompt.read", principal, agent="PromptsRouter", context={"listing": True})
    return {"prompts": registry.list_prompts()}


@router.get("/{key}")
def read_prompt(key: str, principal=Depends(require_user)):
    """The prompt as it is in use, its versions and its pointer history."""
    gate("prompt.read", principal, agent="PromptsRouter", context={"prompt_key": key})
    try:
        all_versions = registry.versions(key)
        if not all_versions:
            raise registry.PromptNotFound(f"no prompt {key!r}")
        try:
            live = registry.resolve(key)
        except registry.PointerUnresolved:
            live = None
        return {
            "prompt_key": key,
            "live": live,
            "versions": [{k: v[k] for k in ("version", "syntax", "author", "created_at",
                                            "change_note", "content_sha256")}
                         for v in all_versions],
            "pointer_history": registry.pointer_history(key),
        }
    except registry.RegistryError as exc:
        return _refused(exc)


@router.get("/{key}/versions/{version}")
def read_version(key: str, version: int, principal=Depends(require_user)):
    gate("prompt.read", principal, agent="PromptsRouter",
         context={"prompt_key": key, "version": version})
    try:
        return registry.get_version(key, version)
    except registry.RegistryError as exc:
        return _refused(exc)


@router.post("/{key}/versions", status_code=201)
def save_version(key: str, body: SaveBody, principal=Depends(require_user)):
    """Save a new version. It is not in use until someone else promotes it."""
    author = _named(principal, "saving a prompt")
    gate("prompt.author", principal, agent="PromptsRouter", context={"prompt_key": key})
    try:
        number = registry.save_version(
            key, template=body.template, author=author, kind=body.kind, owner=body.owner,
            variables=body.variables, fragment_refs=body.fragment_refs,
            settings=body.settings, change_note=body.change_note)
    except templates.TemplateInvalid as exc:
        # Each problem names a blank or a fragment, never a value.
        return JSONResponse(status_code=422, content={
            "code": exc.code.lower(), "message": "the template was not saved",
            "problems": exc.problems})
    except registry.RegistryError as exc:
        return _refused(exc)
    return {"prompt_key": key, "version": number, "in_use": False}


@router.post("/{key}/promote")
def promote(key: str, body: PromoteBody, principal=Depends(require_user)):
    approver = _named(principal, "promoting a prompt")
    gate("prompt.promote", principal, agent="PromptsRouter",
         context={"prompt_key": key, "version": body.version})
    try:
        now = registry.promote(key, body.version, approved_by=approver, note=body.note)
    except registry.RegistryError as exc:
        return _refused(exc)
    return {"prompt_key": key, "live_version": now}


@router.post("/{key}/rollback")
def rollback(key: str, body: RollbackBody, principal=Depends(require_user)):
    approver = _named(principal, "rolling back a prompt")
    gate("prompt.promote", principal, agent="PromptsRouter",
         context={"prompt_key": key, "rollback": True, "to_version": body.to_version})
    try:
        now = registry.rollback(key, approved_by=approver, note=body.note,
                                to_version=body.to_version)
    except registry.RegistryError as exc:
        return _refused(exc)
    return {"prompt_key": key, "live_version": now}
