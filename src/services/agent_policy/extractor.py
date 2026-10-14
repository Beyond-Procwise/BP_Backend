"""The extraction agent's two model calls: read a chunk of a policy document, and fix one policy.

The prompts are governed rows in proc.bp_prompt (agent_policy_extract, agent_policy_fix); a
missing row fails the run with "prompt unavailable" and no text held in code stands in.
The model is AgentNick only (ollama_client's default), called with think=False, the shared
load options, and a union-free JSON-schema grammar.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import ValidationError

from services.agent_policy import conditions
from services.agent_policy.extraction_schema import ChunkResult, ProposedPolicy, grammar_schema
from services.agent_policy.registry import RegistrySnapshot
from services import ollama_client
from services.ollama_client import ollama_generate

EXTRACT_PROMPT = "agent_policy_extract"
FIX_PROMPT = "agent_policy_fix"
_UNUSABLE = "the model did not return a usable answer"
MODEL_BUSY = "The model is busy at a different context size; try again when it is idle."


class PromptUnavailable(RuntimeError):
    pass


class ExtractionError(RuntimeError):
    pass


class ModelBusy(ExtractionError):
    """AgentNick is loaded at another context size: calling it would reload the shared model
    (minutes, for every session). The call is not made."""

    def __init__(self) -> None:
        super().__init__(MODEL_BUSY)


def read_ps() -> Optional[Dict[str, Any]]:
    """GET {OLLAMA_BASE_URL}/api/ps, or None when it cannot be read."""
    try:
        response = ollama_client.egress.get(f"{ollama_client.OLLAMA_BASE_URL}/api/ps",
                                            purpose=ollama_client.egress.Purpose.MODEL_INFERENCE,
                                            require_global=False, timeout=5)
        if getattr(response, "status_code", 0) != 200:
            return None
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 - unreadable means "not known": the call goes ahead as before
        return None


def model_busy(ps: Optional[Callable[[], Optional[Dict[str, Any]]]] = None) -> bool:
    """Would our call make Ollama reload AgentNick? True only when it is loaded now with a
    context_length other than load_options()'s num_ctx. Not loaded (a first load, not a
    reload), loaded at our size, or /api/ps unreadable: False, and the call goes ahead."""
    want = ollama_client.load_options(include_gpu=False)["num_ctx"]
    for m in ((ps or read_ps)() or {}).get("models") or []:
        if not isinstance(m, dict) or ollama_client.DEFAULT_MODEL not in (m.get("name"), m.get("model")):
            continue
        ctx = m.get("context_length")
        if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx != want:
            return True
    return False


def _busy_check(call: Callable[..., Optional[str]],
                busy: Optional[Callable[[], bool]]) -> Callable[[], bool]:
    if busy is not None:
        return busy
    # Only the real model call is guarded by default; a stand-in `call` reaches no model.
    return model_busy if call is ollama_generate else (lambda: False)


def registry_digest(registry: RegistrySnapshot) -> str:
    lines = ["Checkpoints:"]
    for cp in sorted(registry.checkpoints):
        status = "checked today" if registry.checkpoint_live(cp) else "not checked yet"
        lines.append(f"- {cp}: {registry.plain(cp)} ({status})")
    for cp in sorted(registry.checkpoints):
        if not registry.checkpoint_live(cp):
            continue
        lines.append("")
        lines.append(f"At {cp} ({registry.plain(cp)}):")
        lines.append("  Actions:")
        names = sorted(registry.actions.get(cp, set()))
        if not names:
            lines.append("  (none)")
        for name in names:
            lines.append(f"  - {name}: {registry.action_plain.get((cp, name)) or name}")
        lines.append("  Inputs:")
        rows = registry.inputs.get(cp, {})
        if not rows:
            lines.append("  (none)")
        for fld in sorted(rows):
            row = rows[fld]
            source = row.get("source") or "action"
            where = "from action" if source == "action" else f"from {source}"
            tail = "" if row.get("status") == "live" else " - not received yet"
            lines.append(f"  - {fld} ({row.get('type') or 'string'}, {where}): {row.get('plain') or fld}{tail}")
    return "\n".join(lines)


def load_prompt(name: str) -> str:
    from orchestration.prompt_engine import PromptEngine
    from services.db import get_conn

    for p in PromptEngine(connection_factory=get_conn).all_prompts():
        if p.get("promptName") == name and p.get("template"):
            return p["template"]
    raise PromptUnavailable(f"prompt unavailable: {name}")


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _fill(template: str, values: Mapping[str, str]) -> str:
    """One pass: a filled value is never re-scanned, so text that happens to contain
    "{sections}" cannot pull another value in. Unknown {words} are left as they are."""
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)] if m.group(1) in values else m.group(0), template)


def _taxonomy_text(taxonomy: List[Mapping[str, Any]]) -> str:
    out = []
    for a in taxonomy or []:
        tag = " (only for a policy that fits no other area)" if a.get("unassigned") else ""
        out.append(f"- {a.get('areaName')}{tag}: " + ", ".join(a.get("subAreas") or []))
    return "\n".join(out) or "(none)"


def _sections_text(sections: List[Mapping[str, Any]]) -> str:
    parts = []
    for s in sections:
        ref = s.get("reference") or "(no number)"
        parts.append(f"--- Section {ref} ---\n{s.get('text') or ''}".rstrip())
    return "\n\n".join(parts)


def _call(call: Callable[..., Optional[str]], prompt: str, model, busy: Callable[[], bool]) -> Any:
    if busy():
        raise ModelBusy()
    raw = call(prompt, format=grammar_schema(model), think=False, temperature=0, num_predict=8192,
               background=True, use_load_options=True)
    if not raw:
        raise ExtractionError(_UNUSABLE)
    try:
        return model.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise ExtractionError(_UNUSABLE) from exc


def extract_chunk(sections: List[Mapping[str, Any]], *, document: Mapping[str, Any],
                  taxonomy: List[Mapping[str, Any]], registry: RegistrySnapshot,
                  call: Callable[..., Optional[str]] = ollama_generate,
                  busy: Optional[Callable[[], bool]] = None) -> ChunkResult:
    """``document`` carries ``title`` and ``version``. Raises ModelBusy, without calling the
    model, when the call would reload it (``model_busy``)."""
    version = document.get("version")
    prompt = _fill(load_prompt(EXTRACT_PROMPT), {
        "taxonomy": _taxonomy_text(taxonomy),
        "registry": registry_digest(registry),
        "document_title": str(document.get("title") or ""),
        "document_version": "" if version is None else str(version),
        "sections": _sections_text(sections),
    })
    return _call(call, prompt, ChunkResult, _busy_check(call, busy))


def fix_policy(form: Mapping[str, Any], flipped: List[Mapping[str, Any]], *, registry: RegistrySnapshot,
               taxonomy: List[Mapping[str, Any]], settings: Mapping[str, Any],
               call: Callable[..., Optional[str]] = ollama_generate,
               busy: Optional[Callable[[], bool]] = None) -> ProposedPolicy:
    h = form.get("hidden") or {}
    source = form.get("source") or {}
    current = {
        "name": form.get("name"), "category": form.get("category"),
        "business_area": form.get("businessArea"), "sub_area": form.get("subArea"),
        "outcome": form.get("outcome"), "outcome_phrase": form.get("outcomeBecause"),
        "deciders": form.get("deciders") or [], "notify": form.get("notify") or [],
        "reference": source.get("reference"), "checkpoint": h.get("checkpoint"),
        "actions": h.get("actions"), "time_window": h.get("timeWindow"), "units": h.get("units"),
        "inputs": h.get("inputs") or [], "reason_code": h.get("reasonCode"),
        "message_for_agent": form.get("messageForAgent"), "message_for_person": form.get("messageForPerson"),
        "owner": form.get("owner"),
    }
    examples = [{"input": e.get("input") or {}, "expected": e.get("agentExpected")}
                for e in form.get("examples") or []]
    # `flipped` are what the screen sends: {"input", "expects"} (a stage-1 example's other keys
    # may ride along). Only "input" is read: the backend recomputes what the reviewer expects
    # with the same code the review screen uses (reviewer_view), so a sent "expects" is never trusted.
    view = conditions.reviewer_view({**form, "examples": [dict(f, flipped=True) for f in flipped]}, settings)
    constraints = "\n".join(
        f"- these inputs must give {row['reviewer_expects']}: {json.dumps(row['input'], sort_keys=True)}"
        for row in view) or "(none)"
    prompt = _fill(load_prompt(FIX_PROMPT), {
        "policy": json.dumps(current, sort_keys=True, default=str),
        "excerpt": str(source.get("excerpt") or ""),
        "situation": str(form.get("situation") or ""),
        "condition": json.dumps(h.get("condition"), sort_keys=True),
        "examples": json.dumps(examples, sort_keys=True),
        "constraints": constraints,
        "taxonomy": _taxonomy_text(taxonomy),
        "registry": registry_digest(registry),
    })
    return _call(call, prompt, ProposedPolicy, _busy_check(call, busy))
