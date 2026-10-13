"""Render: obligations in, manifest out.

``compose`` turns one saved prompt version and one request into the text an
agent is given, plus a manifest that says exactly what went into it. It reads
no clock and no database, so the same version and the same request give the
same bytes and the same hash, anywhere.

What it never does: decide policy, call a model, run a tool, fetch a document,
shorten anything or try again. Obligations are applied exactly as passed in by
the orchestrator, and one this module does not know refuses the whole render:
an obligation skipped in silence is a rule that quietly did not apply.

Nothing here logs wording or values.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from services.prompt_engine import templates

logger = logging.getLogger(__name__)

_KEY = r"[a-z0-9][a-z0-9_.-]*"
_NAME = r"[a-z][a-z0-9_]*"
_APPEND = re.compile(rf"^append:({_KEY})@([0-9]{{1,9}})$")
_PACK = re.compile(rf"^include_pack:({_KEY})$")
_REDACT = re.compile(rf"^redact:({_NAME}(?:\.{_NAME})*)$")

#: The obligation vocabulary, versioned. Adding a form is a change to this
#: number and to the policy side that emits it, together.
OBLIGATION_VOCABULARY = 1

FragmentLookup = Callable[[str, int], Optional[Dict[str, Any]]]
PackLookup = Callable[[str], Optional[int]]


class ObligationUnknown(templates.TemplateError):
    code = "OBLIGATION_UNKNOWN"


def canonical(value: Any) -> str:
    """One spelling of a JSON value: keys in order, no spare spaces."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(body: Dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def count_tokens(text: str) -> Tuple[int, str]:
    """``(count, how it was counted)``. Reported, never used to shorten."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text)), "cl100k_base"
    except Exception:  # noqa: BLE001 - no tokenizer: an estimate, and say so
        return (len(text) + 3) // 4, "chars/4"


def _parse_obligations(obligations: Any) -> Tuple[List[Tuple[str, ...]], List[str]]:
    """Every obligation, recognised, in the order given — or refuse them all."""
    parsed: List[Tuple[str, ...]] = []
    applied: List[str] = []
    for position, raw in enumerate(obligations or []):
        if isinstance(raw, str):
            for kind, pattern in (("append", _APPEND), ("pack", _PACK), ("redact", _REDACT)):
                match = pattern.match(raw)
                if match:
                    parsed.append((kind, *match.groups()))
                    applied.append(raw)
                    break
            else:
                match = None
        else:
            match = None
        if match is None:
            # The position, not the text: an obligation is policy output, and a
            # malformed one may carry anything.
            raise ObligationUnknown(
                f"obligation {position + 1} is not in vocabulary {OBLIGATION_VOCABULARY}; "
                f"nothing was rendered")
    return parsed, applied


def _mask(values: Dict[str, Any], path: str) -> None:
    """Remove the value at ``path`` in place. A path that leads nowhere masks nothing."""
    steps = path.split(".")
    if steps[0] not in values:
        return
    if len(steps) == 1:
        values[steps[0]] = templates.REDACTED
        return
    holder = values[steps[0]]
    for step in steps[1:-1]:
        if not isinstance(holder, dict) or step not in holder:
            return
        holder = holder[step]
    if isinstance(holder, dict) and steps[-1] in holder:
        holder[steps[-1]] = templates.REDACTED


def compose(version: Dict[str, Any], request: Dict[str, Any], *, fragment: FragmentLookup,
            pack: PackLookup) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(what the agent is given, the manifest)`` for one version and one request.

    ``fragment(key, version)`` returns that fragment version or None.
    ``pack(name)`` returns the version of that pack now in use, or None.
    """
    settings = version.get("settings") or {}
    values = request.get("vars") or {}
    obligations, applied = _parse_obligations(request.get("obligations"))
    used: Dict[str, int] = {}
    texts: Dict[Tuple[str, int], str] = {}

    def need(key: str, number: int) -> str:
        found = fragment(key, number)
        if found is None:
            raise templates.FragmentMissing(f"fragment '{key}@{number}' is not available")
        used[key] = number
        texts[(key, number)] = found.get("template") or ""
        return texts[(key, number)]

    if version.get("syntax") == "v1":
        templates.validate_values(version.get("variables") or [], values)
        for ref in templates.parse(version["template"]).fragment_refs:
            need(ref["key"], ref["version"])
    elif values:
        raise templates.VarUndeclared(
            f"not declared by this prompt: {', '.join(sorted(values))}")

    # Redaction first, whatever order the obligations came in: nothing may be
    # written from a value policy said to remove.
    masked = copy.deepcopy(dict(values))
    for kind, *args in obligations:
        if kind == "redact":
            _mask(masked, args[0])

    if version.get("syntax") == "v1":
        parts = [templates.fill(version["template"], masked, texts)]
    else:
        parts = [version.get("template") or ""]   # imported wording, exactly as it was

    for kind, *args in obligations:
        if kind == "append":
            parts.append(templates.fill(need(args[0], int(args[1])), masked, {}))
        elif kind == "pack":
            number = pack(args[0])
            if number is None:
                raise templates.FragmentMissing(f"pack '{args[0]}' is not in use")
            parts.append(templates.fill(need(args[0], int(number)), masked, {}))

    described = settings.get("tool_descriptions") or {}
    body = {
        "instructions": "\n\n".join(parts),
        "tool_descriptions": [{"name": name, "description": described[name]}
                              for name in (request.get("allowed_tools") or [])
                              if name in described],
        "output_schema": settings.get("output_schema"),
        "params": settings.get("params") or {},
        "tested_models": settings.get("tested_models") or [],
    }
    counted, how = count_tokens(
        body["instructions"] + "".join(t["description"] for t in body["tool_descriptions"]))
    manifest = {
        "prompt_id": version["prompt_key"],
        "version": version["version"],
        "fragments": dict(sorted(used.items())),
        "obligations_applied": applied,
        "obligation_vocabulary": OBLIGATION_VOCABULARY,
        "token_count": counted,
        "token_count_method": how,
        "hash": _hash(body),
    }
    logger.debug("composed %s v%s -> %s (%d tokens)", manifest["prompt_id"],
                 manifest["version"], manifest["hash"], counted)
    return body, manifest


# ---------------------------------------------------------------------------
# The service: resolve, compose, store
# ---------------------------------------------------------------------------
class RenderNotFound(templates.TemplateError):
    """No stored render has this hash: never rendered, or aged out."""

    code = "RENDER_NOT_FOUND"


def _registry():
    from services.prompt_engine import registry

    return registry


def _query(sql: str, params: tuple) -> List[tuple]:
    from services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() if cur.description else []
        cur.close()
    return rows


def _pack_in_use(name: str) -> Optional[int]:
    registry = _registry()
    if registry._kind_of(name) != "fragment":
        return None
    return registry._current(name, registry.LIVE)


def _store(body: Dict[str, Any], manifest: Dict[str, Any], trace: Dict[str, Any]) -> None:
    """The text once under its hash, and this use of it, in one statement.

    If this cannot be written the render fails: a prompt an agent ran under
    that nobody can produce afterwards is the thing this store exists to end.
    """
    attempt = trace.get("attempt")
    _query(
        """
        WITH kept AS (
            INSERT INTO proc.bp_prompt_render
                (render_hash, prompt_key, version, manifest, rendered)
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (render_hash) DO UPDATE SET last_used_at = now()
            RETURNING render_hash
        )
        INSERT INTO proc.bp_prompt_render_use
            (render_hash, prompt_key, version, run_id, step_id, attempt, decision_id)
        SELECT render_hash, %s, %s, %s, %s, %s, %s FROM kept
        """,
        (manifest["hash"], manifest["prompt_id"], manifest["version"],
         canonical(manifest), canonical(body),
         manifest["prompt_id"], manifest["version"], trace.get("run_id"),
         trace.get("step_id"), int(attempt) if attempt is not None else None,
         trace.get("decision_id")),
    )


def render(request: Dict[str, Any]) -> Dict[str, Any]:
    """Render one prompt for one step. The contract's one call.

    Returns the text, the settings versioned with it, and the manifest. Raises
    with the contract's code and returns no text if anything is wrong.
    ``experiment_key`` is accepted and has no effect in phase 1.
    """
    registry = _registry()
    key = request.get("prompt_id")
    if request.get("version") is not None:
        version = registry.get_version(key, int(request["version"]))
    else:
        version = registry.resolve(key, request.get("pointer") or registry.LIVE)
    body, manifest = compose(version, request, fragment=registry._fragment_version,
                             pack=_pack_in_use)
    _store(body, manifest, request.get("trace") or {})
    return {**body, "manifest": manifest}


def get_render(render_hash: str) -> Dict[str, Any]:
    """The manifest and the text stored under this hash. Restricted: the text
    carries customer values."""
    rows = _query(
        "SELECT manifest, rendered, first_rendered_at, last_used_at "
        "FROM proc.bp_prompt_render WHERE render_hash = %s", (render_hash,))
    if not rows:
        raise RenderNotFound(f"no stored render {render_hash}")
    manifest, rendered, first, last = rows[0]
    load = lambda v: json.loads(v) if isinstance(v, str) else v  # noqa: E731
    return {"manifest": load(manifest), "rendered": load(rendered),
            "first_rendered_at": first, "last_used_at": last}


def renders_for_run(run_id: str) -> List[Dict[str, Any]]:
    """Every render in a run, in the order they happened. No wording."""
    columns = ("use_id", "render_hash", "prompt_key", "version", "step_id", "attempt",
               "decision_id", "used_at")
    rows = _query(
        f"SELECT {', '.join(columns)} FROM proc.bp_prompt_render_use "
        "WHERE run_id = %s ORDER BY use_id", (run_id,))
    return [dict(zip(columns, r)) for r in rows]


def retention_days() -> int:
    """How long a stored render is kept after its last use, as policy states it.
    A policy that states none, or zero or less, refuses: nothing is swept."""
    from src.services.governed_limits import LimitUnavailable, limit

    days = limit("prompt_engine", "manifest_retention_days", cast=int)
    if days is None or days <= 0:
        raise LimitUnavailable(
            f"prompt_engine states manifest_retention_days = {days!r}; "
            f"a retention must be a positive number of days")
    return days


def sweep() -> int:
    """Delete stored renders not used within the retention window. Returns how
    many. The record of which run used which hash is kept."""
    days = retention_days()
    rows = _query(
        "WITH gone AS (DELETE FROM proc.bp_prompt_render "
        "WHERE last_used_at < now() - make_interval(days => %s) RETURNING 1) "
        "SELECT count(*) FROM gone", (days,))
    removed = int(rows[0][0]) if rows else 0
    if removed:
        logger.info("prompt render retention: removed %d stored render(s) unused for "
                    "more than %d days", removed, days)
    return removed
