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
from services.ollama_client import ollama_generate

EXTRACT_PROMPT = "agent_policy_extract"
FIX_PROMPT = "agent_policy_fix"
_UNUSABLE = "the model did not return a usable answer"


class PromptUnavailable(RuntimeError):
    pass


class ExtractionError(RuntimeError):
    pass


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


def _call(call: Callable[..., Optional[str]], prompt: str, model) -> Any:
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
                  call: Callable[..., Optional[str]] = ollama_generate) -> ChunkResult:
    """``document`` carries ``title`` and ``version``."""
    version = document.get("version")
    prompt = _fill(load_prompt(EXTRACT_PROMPT), {
        "taxonomy": _taxonomy_text(taxonomy),
        "registry": registry_digest(registry),
        "document_title": str(document.get("title") or ""),
        "document_version": "" if version is None else str(version),
        "sections": _sections_text(sections),
    })
    return _call(call, prompt, ChunkResult)


def fix_policy(form: Mapping[str, Any], flipped: List[Mapping[str, Any]], *, registry: RegistrySnapshot,
               taxonomy: List[Mapping[str, Any]], settings: Mapping[str, Any],
               call: Callable[..., Optional[str]] = ollama_generate) -> ProposedPolicy:
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
    # `flipped` are stage-1 examples ({"input", "agentExpected", "flipped": True}); what the
    # reviewer expects is computed by the same code the review screen uses, never guessed.
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
    return _call(call, prompt, ProposedPolicy)
