"""quote_drafting: one instruction in, one PROPOSAL out. It never writes the quote.

The model's whole job is to read what the seller typed and say which of a closed list
of changes they asked for. It never supplies a price or a cost, and it does no
arithmetic: every figure in a proposal is worked out here, in code, from figures
already on the quote. So "a price it did not read, it does not state" is not an
instruction the model is trusted to follow -- it has no field to state one in.

What comes back is a PATCH over the quote in the quote's own field names. The seller
reads it, and it reaches the quote only through PUT /spendiq/quotes/{id} after they
press Apply. A turn that cannot be acted on answers with `refused` and no change,
which is a better turn than a guess.

The instruction is a governed row in proc.bp_prompt (`sales_quote_drafting`). A missing
row fails the turn; no text held in code stands in for it.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from src.services.ollama_client import ollama_generate

PROMPT_NAME = "sales_quote_drafting"
OPS = ("markup_all", "markup_line", "discount_all", "discount_line", "valid_until", "add_line")
LINE_TAGS = ("product", "service", "one-off")
MAX_MESSAGE = 1000

# Flat on purpose: the local model's grammar ignores unions, so every op shares one
# object and code decides which fields an op may use. No field holds a price or a cost.
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "refused": {"type": "string"},
        "intents": {"type": "array", "maxItems": 20, "items": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": list(OPS)},
                "line_id": {"type": "string"},
                "percent": {"type": "number"},
                "date": {"type": "string"},
                "description": {"type": "string"},
                "tag": {"type": "string", "enum": list(LINE_TAGS)},
                "qty": {"type": "number"},
                "unit": {"type": "string"},
            },
            "required": ["op"],
        }},
    },
    "required": ["refused", "intents"],
}


class AgentUnavailable(RuntimeError):
    """The turn could not be run (no prompt, model busy or silent). HTTP 503."""


def _d(value: Any) -> Optional[Decimal]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def _q4(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _num(x: Decimal) -> str:
    """A figure as it is written in a sentence: no trailing zeros."""
    s = format(x.normalize(), "f")
    return s if s != "-0" else "0"


def _label(line: Mapping[str, Any], n: int) -> str:
    text = str(line.get("description") or "").strip()
    return f"“{text[:60]}”" if text else f"line {n}"


class _Refuse(Exception):
    pass


_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b|per\s+cent\b|pct\b)", re.I)


def stated_percents(message: str) -> List[Decimal]:
    """Every percentage the seller actually typed. A markup or a discount is proposed only
    at one of these: asked to "set the seat price to £150", the model answered "150%
    markup" (seen against the real model, 2026-10-13), and an amount of money is not a
    percentage however confidently it is relabelled."""
    return [Decimal(m) for m in _PERCENT.findall(message or "")]


def _percent(intent: Mapping[str, Any], what: str, *, top: int,
             stated: List[Decimal]) -> Decimal:
    p = _d(intent.get("percent"))
    if p is None or p < 0 or p > top:
        raise _Refuse(f"I could not read a {what} between 0 and {top}% in that.")
    if p not in stated:
        raise _Refuse(f"I could not find a {what} percentage in what you typed, so I will "
                      f"not guess one. Say it as a percentage, for example “{what} 10%”. "
                      "A price is yours to enter on the line.")
    return p


def _target(intent: Mapping[str, Any], lines: List[Mapping[str, Any]]) -> Tuple[int, Mapping[str, Any]]:
    wanted = str(intent.get("line_id") or "")
    for n, line in enumerate(lines, start=1):
        if str(line.get("id")) == wanted:
            return n, line
    raise _Refuse("I could not tell which line that was about.")


def _work_out(intents: List[Mapping[str, Any]], quote: Mapping[str, Any],
              checks: Mapping[str, Any], message: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Intents -> (change, what it does in words, what it read). Every figure is computed
    here from the quote. Raises _Refuse with the sentence to answer with."""
    lines = [l for l in (quote.get("lines") or []) if isinstance(l, dict)]
    by_line: Dict[str, Dict[str, Any]] = {}
    change: Dict[str, Any] = {}
    did: List[str] = []
    reads: List[str] = []
    corridor = _d(((checks or {}).get("discountCorridor") or {}).get("maxPct"))
    stated = stated_percents(message)

    def patch(line: Mapping[str, Any]) -> Dict[str, Any]:
        return by_line.setdefault(str(line.get("id")), {"id": str(line.get("id"))})

    def markup(line: Mapping[str, Any], pct: Decimal) -> bool:
        cost = _d(line.get("cost"))
        if cost is None or cost <= 0:
            return False
        patch(line).update(markupPct=pct, unitPrice=_q4(cost * (Decimal(1) + pct / Decimal(100))))
        return True

    def discount(pct: Decimal) -> None:
        if corridor is not None and pct > corridor:
            raise _Refuse(f"A {_num(pct)}% discount is outside the {_num(corridor)}% corridor "
                          "for this category, so I will not propose it. It needs pricing "
                          "approval, not a quiet edit.")
        note = (f"discount corridor {_num(corridor)}%" if corridor is not None
                else "discount corridor: not set, so not tested")
        if note not in reads:
            reads.append(note)

    for intent in intents:
        op = intent.get("op")
        if op == "markup_all":
            pct = _percent(intent, "markup", top=10_000, stated=stated)
            done = [l for l in lines if markup(l, pct)]
            if not done:
                raise _Refuse("No line on this quote has a cost, so there is nothing to "
                              "mark up. Enter a cost on a line first.")
            left = len(lines) - len(done)
            did.append(f"Set a {_num(pct)}% markup on {len(done)} line"
                       f"{'s' if len(done) != 1 else ''}, re-pricing each from its own cost"
                       + (f"; {left} with no cost {'is' if left == 1 else 'are'} left as "
                          f"{'it is' if left == 1 else 'they are'}" if left else "") + ".")
            reads.append(f"the cost entered on {len(done)} line{'s' if len(done) != 1 else ''}")
        elif op == "markup_line":
            pct = _percent(intent, "markup", top=10_000, stated=stated)
            n, line = _target(intent, lines)
            if not markup(line, pct):
                raise _Refuse(f"{_label(line, n)} has no cost, so I cannot price it from a "
                              "markup. Enter its cost first.")
            did.append(f"Set a {_num(pct)}% markup on {_label(line, n)}, re-pricing it "
                       "from its cost.")
            reads.append(f"the cost entered on {_label(line, n)}")
        elif op == "discount_all":
            pct = _percent(intent, "discount", top=100, stated=stated)
            if not lines:
                raise _Refuse("This quote has no lines to discount.")
            discount(pct)
            for line in lines:
                patch(line)["discountPct"] = pct
            did.append(f"Set a {_num(pct)}% discount on all {len(lines)} line"
                       f"{'s' if len(lines) != 1 else ''}.")
        elif op == "discount_line":
            pct = _percent(intent, "discount", top=100, stated=stated)
            n, line = _target(intent, lines)
            discount(pct)
            patch(line)["discountPct"] = pct
            did.append(f"Set a {_num(pct)}% discount on {_label(line, n)}.")
        elif op == "valid_until":
            try:
                day = dt.date.fromisoformat(str(intent.get("date") or ""))
            except ValueError:
                raise _Refuse("I could not read a date in that.") from None
            if day < dt.date.today():
                raise _Refuse(f"{day:%-d %B %Y} has already passed, so the quote would be "
                              "expired the moment it was sent.")
            change["validUntil"] = day.isoformat()
            did.append(f"Make the quote valid until {day:%-d %B %Y}.")
        elif op == "add_line":
            text = str(intent.get("description") or "").strip()[:300]
            if not text:
                raise _Refuse("I could not tell what the new line should be.")
            qty = _d(intent.get("qty"))
            tag = intent.get("tag") if intent.get("tag") in LINE_TAGS else "service"
            change.setdefault("addLines", []).append({
                "tag": tag, "description": text,
                "qty": qty if qty is not None and qty > 0 else Decimal(1),
                "unit": (str(intent.get("unit") or "").strip()[:40] or "unit"),
                # No figure was read for either, so neither is given one.
                "cost": None, "markupPct": None, "unitPrice": Decimal(0), "discountPct": Decimal(0)})
            did.append(f"Add a line for “{text[:60]}”, with no price yet: that is yours to enter.")
        else:
            raise _Refuse("That is not something I can change on a quote.")
    if by_line:
        change["lines"] = [by_line[str(l.get("id"))] for l in lines if str(l.get("id")) in by_line]
    return change, did, reads


def _quote_for_model(quote: Mapping[str, Any]) -> str:
    """Just enough to tell lines apart. No cost and no price goes to the model: it does
    not need them to name an op, and a figure it never saw is one it cannot repeat."""
    rows = [{"line_id": str(l.get("id")), "tag": l.get("tag"),
             "description": str(l.get("description") or "")[:200],
             "has_cost": (_d(l.get("cost")) or Decimal(0)) > 0}
            for l in (quote.get("lines") or []) if isinstance(l, dict)]
    return json.dumps({"valid_until": str(quote.get("validUntil") or ""), "lines": rows},
                      ensure_ascii=False)


def _default_prompt() -> str:
    from src.services.agent_policy.extractor import PromptUnavailable, load_prompt
    try:
        return load_prompt(PROMPT_NAME)
    except PromptUnavailable as exc:
        raise AgentUnavailable(str(exc)) from None


def _default_busy() -> bool:
    from src.services.agent_policy.extractor import model_busy
    return model_busy()


def propose(quote: Mapping[str, Any], message: str, *, checks: Mapping[str, Any],
            session_id: Optional[str] = None,
            call: Optional[Callable[..., Optional[str]]] = None,
            prompt: Optional[Callable[[], str]] = None,
            busy: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """One turn. Returns {summary, change, reads, session_id, refused}; `change` is None
    whenever `refused` is a sentence. Raises AgentUnavailable when no turn could be run."""
    from src.services.agent_policy.extractor import _fill

    session = session_id or str(quote.get("id") or "")
    message = (message or "").strip()
    if not message:
        raise ValueError("message is needed")
    if len(message) > MAX_MESSAGE:
        raise ValueError(f"message is longer than {MAX_MESSAGE} characters")

    def answer(*, summary: str, change=None, reads=(), refused=None) -> Dict[str, Any]:
        return {"summary": summary, "change": change, "reads": list(reads),
                "session_id": session, "refused": refused}

    if (quote.get("status") or "draft") != "draft":
        sentence = ("This quote is no longer a draft, so I will not propose changes to it. "
                    "Supersede it to start the next version.")
        return answer(summary=sentence, refused=sentence)

    template = (prompt or _default_prompt)()
    # Only the real model is guarded by default; a stand-in `call` reaches no model.
    if (busy or (_default_busy if call is None else (lambda: False)))():
        raise AgentUnavailable("the model is busy with other work; try again shortly")
    raw = (call or ollama_generate)(_fill(template, {"today": dt.date.today().isoformat(),
                                "quote": _quote_for_model(quote), "message": message}),
               format=SCHEMA, think=False, temperature=0, num_predict=1024,
               use_load_options=True)
    if not raw:
        raise AgentUnavailable("the model did not answer")
    try:
        said = json.loads(raw)
        intents = said.get("intents") or []
        if not isinstance(intents, list) or not all(isinstance(i, dict) for i in intents):
            raise ValueError("intents")
    except (ValueError, AttributeError):
        raise AgentUnavailable("the model's answer could not be read") from None

    if not intents:
        # The model's own words are not repeated: a refusal is prose, and prose is where
        # an invented figure would ride out.
        sentence = ("I could not turn that into a change to this quote. I can set a markup "
                    "or a discount (on every line or on one), change how long the quote is "
                    "valid, or add a line.")
        return answer(summary=sentence, refused=sentence)
    try:
        change, did, reads = _work_out(intents, quote, checks, message)
    except _Refuse as no:
        return answer(summary=str(no), refused=str(no))
    ref = quote.get("ref") or quote.get("number") or quote.get("id")
    return answer(summary=" ".join(did), change=change,
                  reads=([f"quote {ref}"] if ref else []) + reads)
