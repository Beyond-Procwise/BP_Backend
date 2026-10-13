"""The prompt registry: versions that never change, and a pointer that moves.

A prompt is a key, a list of saved versions, and a pointer naming the version
in use. Saving never touches what is in use. Promotion and rollback each add
one row to the pointer's history; nothing is edited in place, so "what did this
say last Tuesday, and who put it there" always has an answer.

Nothing here falls back. A prompt that is not there, or a pointer that points
at nothing, raises; it never quietly becomes different wording.

The tables are in deploy/sql/2026-10-13_bp_prompt_registry.sql. The database,
not this module, is what refuses an edit to a saved version.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional

from services.db import get_conn

LIVE = "live"

_KEY = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_KINDS = ("prompt", "fragment")
_SYNTAXES = ("legacy", "v1")
# Where an imported prompt keeps the exact value the older table held, so a
# promotion can hand the older readers back precisely what they had.
LEGACY_DESC = "legacy_prompts_desc"

_VERSION_COLUMNS = (
    "prompt_key", "version", "syntax", "template", "variables", "fragment_refs",
    "settings", "change_note", "author", "created_at", "content_sha256",
)
_MOVE_COLUMNS = (
    "move_id", "prompt_key", "pointer", "from_version", "to_version", "change",
    "approved_by", "note", "moved_at",
)


class RegistryError(Exception):
    """Base for everything the registry refuses. ``code`` is the API's word for it."""

    code = "REGISTRY_ERROR"


class InvalidKey(RegistryError):
    code = "INVALID_KEY"


class KindMismatch(RegistryError):
    code = "KIND_MISMATCH"


class PromptNotFound(RegistryError):
    code = "VERSION_NOT_FOUND"


class VersionNotFound(RegistryError):
    code = "VERSION_NOT_FOUND"


class PointerUnresolved(RegistryError):
    code = "POINTER_UNRESOLVED"


class ApproverRequired(RegistryError):
    code = "APPROVER_REQUIRED"


class SelfApproval(RegistryError):
    code = "SELF_APPROVAL"


class NothingToRollBackTo(RegistryError):
    code = "NOTHING_TO_ROLL_BACK_TO"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _check_key(key: Any) -> str:
    if not isinstance(key, str) or not _KEY.match(key):
        raise InvalidKey(
            f"{key!r} is not a prompt key: lower-case letters, digits, '_', '.', '-'")
    return key


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(*, syntax: str, template: str, variables: Any, fragment_refs: Any,
                 settings: Any) -> str:
    """One hash for everything a version says. Key order does not change it."""
    body = _json({"syntax": syntax, "template": template, "variables": variables,
                  "fragment_refs": fragment_refs, "settings": settings})
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _loaded(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _version_row(row: Iterable[Any]) -> Dict[str, Any]:
    out = dict(zip(_VERSION_COLUMNS, row))
    for name in ("variables", "fragment_refs", "settings"):
        out[name] = _loaded(out[name])
    return out


def _query(sql: str, params: tuple) -> List[tuple]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
    return rows


def _kind_of(key: str) -> Optional[str]:
    rows = _query("SELECT kind FROM proc.bp_prompt_registry WHERE prompt_key = %s", (key,))
    return rows[0][0] if rows else None


def _approver(approved_by: Any) -> str:
    name = approved_by.strip() if isinstance(approved_by, str) else ""
    if not name:
        raise ApproverRequired("a pointer is only moved by a named approver")
    return name


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------
def save_version(
    key: str,
    *,
    template: str,
    author: str,
    kind: str = "prompt",
    owner: Optional[str] = None,
    syntax: str = "v1",
    variables: Optional[List[Dict[str, Any]]] = None,
    fragment_refs: Optional[List[Dict[str, Any]]] = None,
    settings: Optional[Dict[str, Any]] = None,
    change_note: Optional[str] = None,
) -> int:
    """Save a new version and return its number. It is not in use until promoted."""
    _check_key(key)
    if kind not in _KINDS:
        raise KindMismatch(f"kind must be one of {_KINDS}, got {kind!r}")
    if syntax not in _SYNTAXES:
        raise RegistryError(f"syntax must be one of {_SYNTAXES}, got {syntax!r}")
    if not isinstance(template, str):
        raise RegistryError("a template is text")
    if not isinstance(author, str) or not author.strip():
        raise RegistryError("a version is saved by a named author")

    existing = _kind_of(key)
    if existing is not None and existing != kind:
        raise KindMismatch(f"{key!r} is a {existing}; it cannot be saved as a {kind}")

    variables = list(variables or [])
    fragment_refs = list(fragment_refs or [])
    settings = dict(settings or {})
    digest = content_hash(syntax=syntax, template=template, variables=variables,
                          fragment_refs=fragment_refs, settings=settings)

    # The number is taken and the row written in one statement. Two saves at
    # once can still pick the same number; the primary key refuses the second
    # and it is tried again against the new maximum.
    sql = """
        WITH head AS (
            INSERT INTO proc.bp_prompt_registry (prompt_key, kind, owner, created_by)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (prompt_key) DO NOTHING
        )
        INSERT INTO proc.bp_prompt_version
            (prompt_key, version, syntax, template, variables, fragment_refs,
             settings, change_note, author, content_sha256)
        SELECT %s, COALESCE(MAX(version), 0) + 1, %s, %s, %s::jsonb, %s::jsonb,
               %s::jsonb, %s, %s, %s
        FROM proc.bp_prompt_version WHERE prompt_key = %s
        RETURNING version
    """
    params = (key, kind, owner, author.strip(),
              key, syntax, template, _json(variables), _json(fragment_refs),
              _json(settings), change_note, author.strip(), digest, key)
    last: Optional[Exception] = None
    for _ in range(5):
        try:
            return int(_query(sql, params)[0][0])
        except Exception as exc:  # noqa: BLE001 - only the duplicate number is retried
            if "bp_prompt_version_pkey" not in str(exc):
                raise
            last = exc
    raise RegistryError(f"could not take a version number for {key!r}") from last


def list_prompts() -> List[Dict[str, Any]]:
    """Every prompt and fragment with the version its live pointer names (or None)."""
    rows = _query(
        """
        SELECT r.prompt_key, r.kind, r.owner,
               (SELECT max(v.version) FROM proc.bp_prompt_version v
                 WHERE v.prompt_key = r.prompt_key),
               (SELECT m.to_version FROM proc.bp_prompt_pointer_move m
                 WHERE m.prompt_key = r.prompt_key AND m.pointer = %s
                 ORDER BY m.move_id DESC LIMIT 1)
          FROM proc.bp_prompt_registry r ORDER BY r.prompt_key
        """, (LIVE,))
    return [dict(zip(("prompt_key", "kind", "owner", "latest_version", "live_version"), r))
            for r in rows]


def get_version(key: str, version: int) -> Dict[str, Any]:
    _check_key(key)
    rows = _query(
        f"SELECT {', '.join(_VERSION_COLUMNS)} FROM proc.bp_prompt_version "
        "WHERE prompt_key = %s AND version = %s", (key, int(version)))
    if rows:
        return _version_row(rows[0])
    if _kind_of(key) is None:
        raise PromptNotFound(f"no prompt {key!r}")
    raise VersionNotFound(f"prompt {key!r} has no version {version}")


def versions(key: str) -> List[Dict[str, Any]]:
    _check_key(key)
    rows = _query(
        f"SELECT {', '.join(_VERSION_COLUMNS)} FROM proc.bp_prompt_version "
        "WHERE prompt_key = %s ORDER BY version", (key,))
    return [_version_row(r) for r in rows]


# ---------------------------------------------------------------------------
# the pointer
# ---------------------------------------------------------------------------
def pointer_history(key: str, pointer: str = LIVE) -> List[Dict[str, Any]]:
    """Every move of this pointer, oldest first."""
    _check_key(key)
    rows = _query(
        f"SELECT {', '.join(_MOVE_COLUMNS)} FROM proc.bp_prompt_pointer_move "
        "WHERE prompt_key = %s AND pointer = %s ORDER BY move_id", (key, pointer))
    return [dict(zip(_MOVE_COLUMNS, r)) for r in rows]


def _current(key: str, pointer: str) -> Optional[int]:
    rows = _query(
        "SELECT to_version FROM proc.bp_prompt_pointer_move "
        "WHERE prompt_key = %s AND pointer = %s ORDER BY move_id DESC LIMIT 1",
        (key, pointer))
    return int(rows[0][0]) if rows else None


def resolve(key: str, pointer: str = LIVE) -> Dict[str, Any]:
    """The version this pointer names. Raises if it names nothing."""
    _check_key(key)
    current = _current(key, pointer)
    if current is None:
        raise PointerUnresolved(f"prompt {key!r} has no {pointer!r} version")
    return get_version(key, current)


def _legacy_desc(version: Dict[str, Any]) -> Any:
    settings = version.get("settings") or {}
    if version["syntax"] == "legacy" and LEGACY_DESC in settings:
        return settings[LEGACY_DESC]
    return {"prompt_template": version["template"]}


def _move(key: str, version: Dict[str, Any], *, pointer: str, change: str,
          approved_by: str, note: Optional[str], mirror: bool = True) -> int:
    """Record the move and, for the live pointer, hand the older readers the new
    wording — in one statement, so the two can never disagree.

    proc.bp_prompt is only ever UPDATED here: a prompt the older table never
    held is not introduced to it. Its ``version`` is stepped on by one, as its
    own writers do, rather than set to the registry's number: an imported
    prompt is version 1 here whatever it had reached there, and the older
    readers must never see their number go backwards.

    ``mirror=False`` is for the import, which records what is already in use
    and must leave the older table exactly as it found it.
    """
    rows = _query(
        """
        WITH moved AS (
            INSERT INTO proc.bp_prompt_pointer_move
                (prompt_key, pointer, from_version, to_version, change, approved_by, note)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING to_version
        ), mirrored AS (
            UPDATE proc.bp_prompt
               SET prompts_desc = %s::jsonb, version = COALESCE(version, 0) + 1,
                   last_modified_by = %s, last_modified_date = now()
             WHERE %s AND prompt_name = %s AND COALESCE(prompts_status, 1) = 1
            RETURNING 1
        )
        SELECT (SELECT to_version FROM moved), (SELECT count(*) FROM mirrored)
        """,
        (key, pointer, _current(key, pointer), version["version"], change, approved_by,
         note, _json(_legacy_desc(version)), approved_by,
         mirror and pointer == LIVE, key),
    )
    return int(rows[0][0])


def _was_ever(key: str, pointer: str, version: int) -> bool:
    return bool(_query(
        "SELECT 1 FROM proc.bp_prompt_pointer_move "
        "WHERE prompt_key = %s AND pointer = %s AND to_version = %s LIMIT 1",
        (key, pointer, version)))


def promote(key: str, version: int, *, approved_by: str, note: Optional[str] = None,
            pointer: str = LIVE) -> int:
    """Put ``version`` in use. The approver is named, and is not its author."""
    approver = _approver(approved_by)
    target = get_version(key, version)
    if target["author"] == approver and not _was_ever(key, pointer, target["version"]):
        raise SelfApproval(
            f"{approver!r} wrote version {version} of {key!r}; a second person promotes it")
    return _move(key, target, pointer=pointer, change="promoted",
                 approved_by=approver, note=note)


def rollback(key: str, *, approved_by: str, note: Optional[str] = None,
             pointer: str = LIVE, to_version: Optional[int] = None) -> int:
    """Move the pointer back to a version it has held before; by default the one
    it held just before this one. Returns the version now in use.

    The version returned to was approved by a second person when it first went
    into use, so the approver here may be anyone with the right to promote.
    """
    approver = _approver(approved_by)
    history = pointer_history(key, pointer)
    if not history:
        raise NothingToRollBackTo(f"prompt {key!r} has never had a {pointer!r} version")
    target_number = to_version if to_version is not None else history[-1]["from_version"]
    if target_number is None or target_number == history[-1]["to_version"]:
        raise NothingToRollBackTo(f"prompt {key!r} has no earlier {pointer!r} version")
    if not _was_ever(key, pointer, int(target_number)):
        raise NothingToRollBackTo(
            f"version {target_number} of {key!r} was never {pointer!r}; promote it instead")
    return _move(key, get_version(key, int(target_number)), pointer=pointer,
                 change="rolled_back", approved_by=approver, note=note)
