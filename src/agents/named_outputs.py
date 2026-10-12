"""The outputs a person named for a created agent, filled in from a run.

A created agent reasons with tools and answers in prose. When it was defined
with named outputs, the run ends with one more question to the model: fill in
each output from what the tools returned, quote the words it rests on, and say
when the evidence does not establish it. The quote is then looked for in what
the tools actually returned.

Three outcomes per output, and they mean exactly this:

``evidence_found``      a value, and the words quoted for it ARE in what the
                        tools returned or the workflow supplied.
``evidence_not_found``  a value, but the words quoted for it are not there. The
                        value is shown and flagged; it is not vouched for.
``missing``             the evidence did not establish it. No value.

``evidence_found`` says the quote is real. It does not say the value follows
from the quote: for an output that is a judgement ("recommendation"), that is
for whoever reads it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MAX_OUTPUTS = 12
# What a reasoning run already puts on its result; an output may not take one.
RESERVED = frozenset({
    "answer", "outputs", "outputs_error", "run_id", "recorded", "tools_used",
    "rounds", "trace", "prompts", "policies", "llm", "workflow_id",
})
# A quote shorter than this is not evidence of anything: "5" is in every table.
MIN_EVIDENCE_CHARS = 6
_RESULT_CHARS_EACH = 6000
_EVIDENCE_CHARS_TOTAL = 24000

_SYSTEM = """You fill in named outputs from evidence. Rules:
- Use ONLY the tool results and inputs you are shown. Never use outside knowledge.
- For each output: if the evidence establishes it, set found to true, give the value, \
and copy into evidence the exact words from the tool results or inputs that it rests on.
- If the evidence does not establish an output, set found to false and leave value and \
evidence empty. Never guess, estimate or fill a gap.
- Copy figures, names and dates exactly as given. Do not round or restate them."""


def checked_spec(raw: Any) -> List[Dict[str, str]]:
    """The list of outputs as it will be stored, or ``ValueError`` saying why not."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("outputs must be a list naming at least one output")
    if len(raw) > MAX_OUTPUTS:
        raise ValueError(f"an agent can name at most {MAX_OUTPUTS} outputs")
    spec: List[Dict[str, str]] = []
    seen = set()
    for item in raw:
        item = item if isinstance(item, dict) else {"name": item}
        name = str(item.get("name") or "").strip()
        if not NAME_RE.match(name):
            raise ValueError(
                f"output name {name!r} must be lowercase letters, digits and underscores, "
                "starting with a letter (for example overbilled_amount)"
            )
        if name in RESERVED:
            raise ValueError(f"output name {name!r} is used by the run itself; choose another")
        if name in seen:
            raise ValueError(f"output {name!r} is named twice")
        seen.add(name)
        spec.append({"name": name, "description": str(item.get("description") or "").strip()})
    return spec


def schema_for(spec: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    """The JSON schema the model must answer in. No unions: the local model
    server ignores them, and an ignored constraint is no constraint."""
    one = {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "value": {"type": "string"},
            "evidence": {"type": "string"},
        },
        "required": ["found", "value", "evidence"],
    }
    names = [s["name"] for s in spec]
    return {"type": "object", "properties": {n: dict(one) for n in names}, "required": names}


def _squash(text: str) -> str:
    return " ".join(str(text).split())


def _render(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(value)


def structured(system: str, user: str, schema: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
    """One model call answered in ``schema``. Kept as a name in this module so
    the loop's own module is only loaded when a run actually needs it."""
    from services.tool_runtime import structured as _structured

    return _structured(system, user, schema, **kw)


def fill(
    spec: Sequence[Dict[str, str]],
    *,
    task: str,
    calls: Sequence[Any],
    answer: str,
    model: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fill the named outputs from one run. Raises if the model cannot be asked."""
    shown: List[str] = []
    for call in calls:
        if not getattr(call, "ok", False):
            continue
        shown.append(
            f"Tool {call.name} asked {_render(call.arguments)} returned:\n"
            f"{_render(call.result)[:_RESULT_CHARS_EACH]}"
        )
    evidence = "\n\n".join(shown)[:_EVIDENCE_CHARS_TOTAL]
    wanted = "\n".join(
        f"- {s['name']}: {s['description'] or 'no description given'}" for s in spec
    )
    user = (
        f"Inputs the workflow supplied:\n{task}\n\n"
        f"Tool results:\n{evidence or '(no tool returned anything)'}\n\n"
        f"The agent's answer:\n{answer}\n\n"
        f"Fill in these outputs:\n{wanted}"
    )
    kwargs = {"model": model} if model else {}
    raw = structured(_SYSTEM, user, schema_for(spec), **kwargs) or {}

    haystack = _squash(f"{task}\n{evidence}")
    out: Dict[str, Dict[str, Any]] = {}
    for s in spec:
        got = raw.get(s["name"]) if isinstance(raw, dict) else None
        got = got if isinstance(got, dict) else {}
        value = str(got.get("value") or "").strip()
        if not got.get("found") or not value:
            out[s["name"]] = missing()
            continue
        quote = _squash(got.get("evidence") or "")
        real = len(quote) >= MIN_EVIDENCE_CHARS and quote in haystack
        out[s["name"]] = {
            "value": value,
            "status": "evidence_found" if real else "evidence_not_found",
            "evidence": str(got.get("evidence") or "").strip() or None,
        }
    return out


def missing() -> Dict[str, Any]:
    return {"value": None, "status": "missing", "evidence": None}
