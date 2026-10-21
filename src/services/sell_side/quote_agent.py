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

from src.services.agentnick_driver import generate as ollama_generate

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


# A quote started from a template is FILLED IN from what the seller says, so on those
# turns the model may also carry a quantity, a price, a cost, a name or a term. It is
# still not trusted with them: each one is checked against the seller's own message
# (_grounded_*) and dropped with a refusal if it is not there. The seller's words are
# the only source; the model is the clerk who files them under the right line.
FILL_PROMPT_NAME = "sales_quote_template_fill"
FILL_OPS = OPS + ("set_line", "set_customer", "set_term", "set_section", "calculate")
CALCS = ("quote_total", "margin", "with_tax", "expression")
INTO = ("qty", "unit_price", "cost")
WRITTEN_KINDS = ("scope", "assumptions", "text")
TERM_FIELDS = ("payment", "start", "lengthMonths", "renewal", "sla")
FILL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "refused": {"type": "string"},
        "intents": {"type": "array", "maxItems": 40, "items": {
            "type": "object",
            "properties": {
                **SCHEMA["properties"]["intents"]["items"]["properties"],
                "op": {"type": "string", "enum": list(FILL_OPS)},
                "unit_price": {"type": "number"},
                "cost": {"type": "number"},
                "name": {"type": "string"},
                "contact": {"type": "string"},
                "field": {"type": "string", "enum": list(TERM_FIELDS)},
                "text": {"type": "string"},
                "number": {"type": "number"},
                "section_key": {"type": "string"},
                "calc": {"type": "string", "enum": list(CALCS)},
                "expression": {"type": "string"},
                "label": {"type": "string"},
                "into": {"type": "string", "enum": list(INTO)},
            },
            "required": ["op"],
        }},
    },
    "required": ["refused", "intents"],
}


# A sum the seller asks for on their own numbers is set up by a second, very small turn
# with one job and one required answer. Asked for it as part of the fill, the real model
# named the calculation and left the sum itself out, every time (2026-10-13).
SUM_PROMPT_NAME = "sales_quote_sum"
SUM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"expression": {"type": "string"}, "label": {"type": "string"},
                   "line_id": {"type": "string"},
                   "into": {"type": "string", "enum": ["", *INTO]}},
    "required": ["expression", "label", "line_id", "into"],
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
    """A turn, or one value in it, that will not be proposed. ``note`` names the value in
    a few words, for the sentence that says what was left out of a fill."""

    def __init__(self, sentence: str, note: Optional[str] = None) -> None:
        super().__init__(sentence)
        self.note = note


_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b|per\s+cent\b|pct\b)", re.I)


def stated_percents(message: str) -> List[Decimal]:
    """Every percentage the seller actually typed. A markup or a discount is proposed only
    at one of these: asked to "set the seat price to £150", the model answered "150%
    markup" (seen against the real model, 2026-10-13), and an amount of money is not a
    percentage however confidently it is relabelled."""
    return [Decimal(m) for m in _PERCENT.findall(message or "")]


_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(k|m)?\b", re.I)
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december")


_WORDS = {w: n for n, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve".split(), start=1)}
_WORD = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.I)
# Seen against the real model, 2026-10-13: given "Monthly fee is 4,200" it filed 4,200 as
# the price AND the cost, and given no word about validity it "set" the date the quote
# already had. Neither is a figure the seller did not type, so the number check passes
# them; these say what the seller must have been talking about for each to be taken.
_COST_TALK = re.compile(r"\bcost(s|ing)?\b|\bwe pay\b|\bbuy(ing)? (it|them|in)\b|\bbought\b", re.I)
_VALIDITY_TALK = re.compile(r"\bvalid|\bexpir|\bgood for\b|\bopen for\b|\bhold(s)? (the )?(price|quote)", re.I)


def stated_numbers(message: str) -> List[Decimal]:
    """Every number the seller typed, as the figure it means: "18,000" and "18k" are both
    18000. A quantity, price or cost is filled in only if it is one of these."""
    out: List[Decimal] = []
    for digits, scale in _NUMBER.findall(message or ""):
        try:
            n = Decimal(digits.replace(",", ""))
        except InvalidOperation:
            continue
        out.append(n * {"k": 1000, "m": 1_000_000}.get(scale.lower(), 1))
        if scale:                       # "5 m" may be five metres: the bare 5 was typed too
            out.append(n)
    out.extend(Decimal(_WORDS[w.lower()]) for w in _WORD.findall(message or ""))   # "two days"
    return out


def _squash(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _grounded_number(intent: Mapping[str, Any], key: str, what: str,
                     stated: List[Decimal], *, least: Decimal = Decimal(0)) -> Optional[Decimal]:
    """None when the model gave no value for ``key``; refuses one the seller never typed."""
    if intent.get(key) is None:
        return None
    n = _d(intent.get(key))
    if n is None or n < least:
        raise _Refuse(f"I could not read {what} in that.", what)
    if n not in stated:
        raise _Refuse(f"I could not find {what} of {_num(n)} in what you typed.",
                      f"{what} of {_num(n)}")
    return n


def _grounded_text(value: Any, what: str, message: str, limit: int = 300) -> Optional[str]:
    """None when the model gave nothing; refuses words the seller did not write. What is
    kept is the seller's own spelling, cut out of their message."""
    wanted = _squash(value)
    if not wanted:
        return None
    flat = " ".join(str(message or "").split())
    at = flat.lower().find(wanted)
    if at < 0 or len(wanted) > limit:
        raise _Refuse(f"I could not find {what} “{str(value).strip()[:60]}” in what you typed.",
                      f"{what} “{str(value).strip()[:60]}”")
    return flat[at:at + len(wanted)]


def _short_field(value: Optional[str], what: str, prose: List[str]) -> Optional[str]:
    """A name or a term is a few words. Seen against the real model: a sentence of the
    seller's scope came back as the contact, and half their timeline as the service
    level -- their own words, so the wording check passed, filed in the wrong place."""
    if not value:
        return value
    words = len(value.split())
    inside = any(_squash(value) in _squash(p) for p in prose)
    # the customer's own name may well appear in the scope; nothing else short does
    if words > 8 or (inside and (words > 3 or what != "the customer")):
        raise _Refuse("", f"{what} “{value[:40]}…” (that reads as part of a section, "
                          "not as " + what + ")")
    return value


_DAY_MONTH = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(m[:3] for m in _MONTHS)
    + r")[a-z]*\.?(?:\s+(\d{4}))?", re.I)


def typed_dates(message: str, today: Optional[dt.date] = None) -> List[dt.date]:
    """Every date the seller typed as a day and a month ("1 November", "3rd of Jan 2027")
    or in full (2026-11-01). With no year, it is the next such day from today."""
    today = today or dt.date.today()
    found: List[dt.date] = []
    for iso in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", message or ""):
        try:
            found.append(dt.date.fromisoformat(iso))
        except ValueError:
            pass
    for day, month, year in _DAY_MONTH.findall(message or ""):
        month_no = [m[:3] for m in _MONTHS].index(month.lower()[:3]) + 1
        try:
            when = dt.date(int(year) if year else today.year, month_no, int(day))
            if not year and when < today:
                when = dt.date(today.year + 1, month_no, int(day))
        except ValueError:
            continue
        found.append(when)
    return found


def _grounded_date(value: Any, what: str, message: str) -> dt.date:
    """The date is read from the seller's message here, in code. The model's own date is
    only used to choose between two the seller typed; one typed date needs no choosing,
    which is as well -- the real model answered "1 November" where a date was asked for."""
    typed = typed_dates(message)
    try:
        said = dt.date.fromisoformat(str(value or ""))
    except ValueError:
        said = None
    if said in typed:
        return said
    if len(set(typed)) == 1:
        return typed[0]
    raise _Refuse(f"I could not find {what} in what you typed. Give it as a day and a "
                  "month, for example “1 November”.", what)


# --- the calculator ------------------------------------------------------------------
#
# The model never does arithmetic. Asked for a sum, it writes the sum down
# ("12000 / 340") and this works it out, in Decimal, and hands back the working beside
# the answer so the seller can check it. Only + - * / and brackets exist here; there is
# no name, no function and no power to evaluate, so there is nothing to inject.

_TOKEN = re.compile(r"\s*(?:(\d[\d,]*(?:\.\d+)?)\s?(k|m)?\s*(%)?|([-+*/()x×÷]))", re.I)
# What a sum may contain besides the seller's own numbers: 1 and 100, which are how
# "less 4%" and "per hundred" are written and are nobody's price.
_FREE = (Decimal(1), Decimal(100))


class _Sum:
    """One expression, tokenised once: `value` is its result, `literals` every number in
    it, and `text` the sum as it will be shown."""

    def __init__(self, expression: str) -> None:
        self.tokens: List[Any] = []
        self.literals: List[Decimal] = []
        text = str(expression or "").replace("£", "").replace("$", "").replace("€", "").strip()
        at = 0
        while at < len(text):
            m = _TOKEN.match(text, at)
            if m is None or m.end() == at:
                raise ValueError("unreadable")
            at = m.end()
            if m.group(4):
                self.tokens.append({"x": "*", "×": "*", "÷": "/"}.get(m.group(4).lower(), m.group(4)))
                continue
            n = Decimal(m.group(1).replace(",", "")) * {"k": 1000, "m": 1_000_000}.get(
                (m.group(2) or "").lower(), 1)
            self.literals.append(n)
            self.tokens.append(n / 100 if m.group(3) else n)
            if m.group(3):
                self.tokens[-1] = ("%", n)
        if not self.tokens or len(self.tokens) > 60:
            raise ValueError("unreadable")
        self._at = 0
        self.value = self._add()
        if self._at != len(self.tokens):
            raise ValueError("unreadable")
        self.text = " ".join(f"{_num(t[1])}%" if isinstance(t, tuple) else
                             _num(t) if isinstance(t, Decimal) else {"*": "×", "/": "÷"}.get(t, t)
                             for t in self.tokens).replace("( ", "(").replace(" )", ")")

    def _peek(self) -> Any:
        return self.tokens[self._at] if self._at < len(self.tokens) else None

    def _add(self) -> Decimal:
        left = self._mul()
        while self._peek() in ("+", "-"):
            op = self.tokens[self._at]
            self._at += 1
            right = self._mul()
            left = left + right if op == "+" else left - right
        return left

    def _mul(self) -> Decimal:
        left = self._atom()
        while self._peek() in ("*", "/"):
            op = self.tokens[self._at]
            self._at += 1
            right = self._atom()
            if op == "/" and right == 0:
                raise ValueError("division by zero")
            left = left * right if op == "*" else left / right
        return left

    def _atom(self) -> Decimal:
        t = self._peek()
        self._at += 1
        if t == "-":
            return -self._atom()
        if t == "(":
            inner = self._add()
            if self._peek() != ")":
                raise ValueError("unreadable")
            self._at += 1
            return inner
        if isinstance(t, tuple):
            return t[1] / 100
        if isinstance(t, Decimal):
            return t
        raise ValueError("unreadable")


def _money(x: Decimal) -> str:
    return f"{x.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


def _line_net(line: Mapping[str, Any]) -> Decimal:
    qty, price = _d(line.get("qty")) or Decimal(0), _d(line.get("unitPrice")) or Decimal(0)
    off = _d(line.get("discountPct")) or Decimal(0)
    return (qty * price * (Decimal(1) - off / 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def quote_figures(quote: Mapping[str, Any]) -> Dict[str, Any]:
    """The quote's own total and margin, by the builder's rules: every line in the total,
    and the margin over the costed lines only."""
    lines = [l for l in (quote.get("lines") or []) if isinstance(l, dict)]
    nets = [_line_net(l) for l in lines]
    costed = [(l, n) for l, n in zip(lines, nets) if (_d(l.get("cost")) or Decimal(0)) > 0]
    revenue = sum((n for _, n in costed), Decimal(0))
    cost = sum(((_d(l.get("qty")) or Decimal(0)) * _d(l.get("cost")) for l, _ in costed), Decimal(0))
    return {"lines": lines, "nets": nets, "total": sum(nets, Decimal(0)), "revenue": revenue,
            "cost": cost, "costed": len(costed), "uncosted": len(lines) - len(costed)}


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
              checks: Mapping[str, Any], message: str, *,
              filling: bool = False,
              ) -> Tuple[Dict[str, Any], List[str], List[str], List[Dict[str, Any]]]:
    """Intents -> (change, what it does in words, what it read). Every figure is computed
    here from the quote. Raises _Refuse with the sentence to answer with."""
    lines = [l for l in (quote.get("lines") or []) if isinstance(l, dict)]
    by_line: Dict[str, Dict[str, Any]] = {}
    change: Dict[str, Any] = {}
    did: List[str] = []
    reads: List[str] = []
    corridor = _d(((checks or {}).get("discountCorridor") or {}).get("maxPct"))
    stated = stated_percents(message)
    numbers = stated_numbers(message)
    left_out: List[str] = []
    asked: List[Mapping[str, Any]] = []
    # What the seller wrote for a section this turn is prose for the customer to read,
    # not instructions to file: a figure in it is not a quantity ("11 depots" came back
    # from the real model as 11 licences), and a phrase of it is not a contact or a term.
    prose = [str(i.get("text") or "") for i in intents if i.get("op") == "set_section"]
    outside = " ".join(str(message or "").split())
    for words in prose:
        outside = outside.replace(" ".join(words.split()), " ")
    fill_numbers, fill_percents = stated_numbers(outside), stated_percents(outside)
    calculations: List[Dict[str, Any]] = []

    def take(read: Callable[[], Any]) -> Any:
        """One value of a fill. A value that is not in the seller's words is left out and
        named, and the rest of what they gave is still filled in: refusing a whole
        paragraph of details over one misfiled number helps nobody."""
        try:
            return read()
        except _Refuse as no:
            if no.note and no.note not in left_out:
                left_out.append(no.note)
            return None

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
            if not _VALIDITY_TALK.search(message):
                continue                      # the seller said nothing about validity
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
        elif op == "set_line" and filling:
            try:
                n, line = _target(intent, lines)
            except _Refuse:
                left_out.append("a line I could not match to this quote")
                continue
            zero = lambda k: _d(intent.get(k)) == 0          # noqa: E731  "0" is "not given"
            qty = None if zero("qty") else take(
                lambda: _grounded_number(intent, "qty", "a quantity", fill_numbers))
            price = None if zero("unit_price") else take(
                lambda: _grounded_number(intent, "unit_price", "a price", fill_numbers))
            if price is not None and _DERIVE_TALK.search(message):
                left_out.append(f"a price of {_num(price)} on {_label(line, n)} (you asked "
                                "for it to be worked out, so I have not set it directly)")
                price = None
            # a cost is taken only when the seller was talking about what it costs them
            cost = (take(lambda: _grounded_number(intent, "cost", "a cost", fill_numbers))
                    if _COST_TALK.search(message) and not zero("cost") else None)
            off = None
            if intent.get("percent") is not None and not zero("percent"):
                if _d(intent.get("percent")) in fill_percents:
                    off = _percent(intent, "discount", top=100, stated=fill_percents)
                else:
                    left_out.append(f"a discount of {_num(_d(intent.get('percent')) or Decimal(0))}%")
            if off is not None:
                discount(off)
            given = {k: v for k, v in (("qty", qty), ("unitPrice", price), ("cost", cost),
                                       ("discountPct", off)) if v is not None}
            if not given:
                continue
            patch(line).update(given)
            now_cost = cost if cost is not None else _d(line.get("cost"))
            now_price = price if price is not None else _d(line.get("unitPrice"))
            if (cost is not None or price is not None) and now_cost and now_cost > 0 \
                    and now_price is not None and now_price > 0:
                # the markup the two figures imply; the screen's own formula, one decimal
                patch(line)["markupPct"] = ((now_price - now_cost) / now_cost * 100).quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP)
            said = [f"{_num(qty)} of them" if qty is not None else "",
                    f"a price of {_num(price)} each" if price is not None else "",
                    f"a cost of {_num(cost)} each" if cost is not None else "",
                    f"a {_num(off)}% discount" if off is not None else ""]
            did.append(f"Fill in {_label(line, n)}: " + ", ".join(x for x in said if x) + ".")
        elif op == "set_customer" and filling:
            name = take(lambda: _short_field(_grounded_text(
                intent.get("name"), "the customer", message), "the customer", prose))
            contact = take(lambda: _short_field(_grounded_text(
                intent.get("contact"), "the contact", message), "the contact", prose))
            if name:
                change.setdefault("customer", {})["name"] = name
                did.append(f"Address the quote to {name}.")
            if contact:
                change.setdefault("customer", {})["contact"] = contact
                did.append(f"Name {contact} as the contact.")
        elif op == "set_term" and filling:
            field = intent.get("field")
            if field not in TERM_FIELDS:
                left_out.append("a term this quote does not have")
                continue
            if field == "start":
                day = take(lambda: _grounded_date(intent.get("date") or intent.get("text"),
                                                  "a start date", message))
                if day is None:
                    continue
                value: Any = day.isoformat()
                did.append(f"Set the start date to {day:%-d %B %Y}.")
            elif field == "lengthMonths":
                months = take(lambda: _grounded_number(
                    {"n": intent.get("number") if intent.get("number") is not None
                     else intent.get("text")}, "n", "a length in months", numbers,
                    least=Decimal(1)))
                if months is None:
                    continue
                if months != months.to_integral_value() or months > 1200:
                    left_out.append("a length in months that is not a whole number")
                    continue
                value = int(months)
                did.append(f"Set the term to {value} months.")
            else:
                value = take(lambda: _short_field(_grounded_text(
                    intent.get("text"), "the " + field + " term", message),
                    "the " + field + " term", prose))
                if not value:
                    continue
                if re.search(r"\d\s*%\s*(off|discount)|\bdiscount\b", value, re.I):
                    # seen against the real model: "5% off" filed as the payment terms
                    left_out.append(f"“{value}” as a term (it reads as a discount; say "
                                    "which line it is off)")
                    continue
                did.append({"payment": f"Set payment terms to {value}.",
                            "renewal": f"Set the renewal terms to {value}.",
                            "sla": f"Set the service level to {value}."}[field])
            change.setdefault("terms", {})[field] = value
        elif op == "set_section" and filling:
            wanted = str(intent.get("section_key") or "")
            sec = next((x for x in (quote.get("sections") or [])
                        if isinstance(x, dict) and x.get("key") == wanted), None)
            if sec is None or sec.get("kind") not in WRITTEN_KINDS:
                left_out.append("some words I could not match to a section of this quote")
                continue
            words = take(lambda: _grounded_text(intent.get("text"),
                                                f"the words for “{sec.get('heading')}”",
                                                message, limit=4000))
            if not words:
                continue
            html = "<p>" + words.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</p>"
            if sec["kind"] == "scope":
                change["scopeHtml"] = html
            elif sec["kind"] == "assumptions":
                change["assumptionsHtml"] = html
            else:
                change.setdefault("sections", []).append({"key": sec["key"], "html": html})
            did.append(f"Put your words under “{sec.get('heading')}”.")
        elif op == "calculate" and filling:
            asked.append(intent)
        else:
            raise _Refuse("That is not something I can change on a quote.")

    # Sums are worked out last, on the quote as this proposal would leave it, so "what
    # is the total" after "340 seats at 200" is the total with those seats in it.
    for intent in asked:
        kind = intent.get("calc")
        if kind == "expression":
            try:
                total = _Sum(str(intent.get("expression") or ""))
            except (ValueError, InvalidOperation, ZeroDivisionError):
                left_out.append("a sum I could not read")
                continue
            strangers = [n for n in total.literals if n not in numbers and n not in _FREE]
            if strangers:
                left_out.append("a sum using " + ", ".join(_num(n) for n in strangers)
                                + ", which you did not give me")
                continue
            label = str(intent.get("label") or "").strip()[:80] or "Result"
            result = total.value
            entry: Dict[str, Any] = {"label": label, "working": f"{total.text} = {_num(_q4(result))}",
                                     "result": _q4(result)}
            into = intent.get("into")
            if into in INTO and intent.get("line_id") is not None:
                try:
                    n, line = _target(intent, lines)
                except _Refuse:
                    left_out.append("where to put the result of a sum")
                    calculations.append(entry)
                    continue
                if result < 0 or (into == "qty" and result <= 0) or \
                        (into == "cost" and not _COST_TALK.search(message)):
                    left_out.append(f"putting {_num(_q4(result))} on {_label(line, n)}")
                else:
                    field = {"qty": "qty", "unit_price": "unitPrice", "cost": "cost"}[into]
                    patch(line)[field] = _q4(result)
                    entry["into"] = {"lineId": str(line.get("id")), "field": field}
                    did.append(f"Put {_num(_q4(result))} on {_label(line, n)} as its "
                               f"{ {'qty': 'quantity', 'unitPrice': 'price', 'cost': 'cost'}[field] }.")
            calculations.append(entry)
            continue
        merged = [{**l, **by_line.get(str(l.get("id")), {})} for l in lines] \
            + list(change.get("addLines") or [])
        fig = quote_figures({"lines": merged})
        parts = " + ".join(_money(n) for n in fig["nets"]) or "0.00"
        if kind == "quote_total":
            calculations.append({"label": "Quote total, before tax",
                                 "working": f"{parts} = {_money(fig['total'])}",
                                 "result": fig["total"]})
        elif kind == "margin":
            if not fig["costed"]:
                left_out.append("a margin (no line has a cost, so there is none to work out)")
                continue
            margin = fig["revenue"] - fig["cost"]
            pct = (margin / fig["revenue"] * 100) if fig["revenue"] else Decimal(0)
            calculations.append({
                "label": f"Margin on the {fig['costed']} costed line"
                         f"{'s' if fig['costed'] != 1 else ''}"
                         + (f" ({fig['uncosted']} with no cost left out)" if fig["uncosted"] else ""),
                "working": f"{_money(fig['revenue'])} − {_money(fig['cost'])} = {_money(margin)}"
                           f", which is {_num(pct.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))}%"
                           " of that revenue",
                "result": margin.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)})
        elif kind == "with_tax":
            rate = _d(intent.get("percent"))
            if rate is None or rate not in stated or rate > 100:
                left_out.append("a total with tax (give the tax rate as a percentage; none "
                                "is set for quotes)")
                continue
            tax = (fig["total"] * rate / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            calculations.append({"label": f"Quote total with tax at {_num(rate)}%",
                                 "working": f"{_money(fig['total'])} + {_money(tax)} tax = "
                                            f"{_money(fig['total'] + tax)}",
                                 "result": fig["total"] + tax})
        else:
            left_out.append("a calculation I do not know how to do")
    if by_line:
        change["lines"] = [by_line[str(l.get("id"))] for l in lines if str(l.get("id")) in by_line]
    if left_out:
        shown = left_out[:4] + ([f"{len(left_out) - 4} more"] if len(left_out) > 4 else [])
        did.append("I left out " + "; ".join(shown) + ": I could not match "
                   + ("it" if len(left_out) == 1 else "them") + " to what you typed.")
    return change, did, reads, calculations


def _quote_for_model(quote: Mapping[str, Any]) -> str:
    """Just enough to tell lines apart. No cost and no price goes to the model: it does
    not need them to name an op, and a figure it never saw is one it cannot repeat."""
    rows = [{"line_id": str(l.get("id")), "tag": l.get("tag"),
             "description": str(l.get("description") or "")[:200],
             "has_cost": (_d(l.get("cost")) or Decimal(0)) > 0}
            for l in (quote.get("lines") or []) if isinstance(l, dict)]
    return json.dumps({"valid_until": str(quote.get("validUntil") or ""), "lines": rows},
                      ensure_ascii=False)


def _template_for_model(template: Mapping[str, Any]) -> str:
    return json.dumps({"name": template.get("name"), "what_it_is": template.get("description"),
                       "terms_it_asks_for": template.get("termFields") or []},
                      ensure_ascii=False)


# --- what is read from the seller's message in code, not by the model ------------------
#
# Tried against the real model on 2026-10-13 with the layout and the calculations all in
# its instructions, it stopped filing sections at all, offered sums nobody asked for, and
# dropped the tax rate from "with 20% VAT". These are things a pattern reads exactly, so
# a pattern reads them: the model is left with the one job it does well, filing figures
# under lines, and the one sum only it can set up (an expression).

_SECTION_LEAD = r"(?:^|(?<=[.!?:;\n])\s*)(?:and\s+)?(?:(?:the|our|for the)\s+)?"
_SECTION_JOIN = r"\s*(?::|[-–—]|\bis that\b|\bare that\b|\bis\b|\bare\b|\bwill be\b)\s*"


def section_words(message: str, sections: Any) -> List[Dict[str, Any]]:
    """The seller's words for each written section they named, as set_section intents.

    A section is named by the heading the page shows ("Deliverables: ...", "The timeline
    is ...", "Our understanding is that ...") or, for scope and assumptions, by that word.
    Its words run to the next named section or to the end, less any closing question --
    "what does that come to?" is for the calculator, not for the customer."""
    names: List[Tuple[str, str]] = []
    for sec in sections or []:
        if not isinstance(sec, dict) or sec.get("kind") not in WRITTEN_KINDS:
            continue
        for name in {str(sec.get("heading") or "").strip().lower(),
                     sec["kind"] if sec["kind"] != "text" else ""}:
            if len(name) >= 3:
                names.append((name, sec["key"]))
    if not names or not message:
        return []
    names.sort(key=lambda n: -len(n[0]))
    by_name = dict(reversed(names))
    found = list(re.finditer(_SECTION_LEAD + "(" + "|".join(re.escape(n) for n, _ in names) + ")"
                             + _SECTION_JOIN, message, re.I))
    out: List[Dict[str, Any]] = []
    for at, m in enumerate(found):
        end = found[at + 1].start() if at + 1 < len(found) else len(message)
        words = message[m.end():end].strip()
        sentences = re.split(r"(?<=[.!?])\s+", words)
        while sentences and sentences[-1].rstrip().endswith("?"):
            sentences.pop()
        words = " ".join(sentences).strip().rstrip(";,")
        if len(words) >= 3:
            out.append({"op": "set_section", "section_key": by_name[m.group(1).lower()],
                        "text": words})
    return out


_TOTAL_TALK = re.compile(r"\btotal\b|\bcomes? to\b|\badds? up\b|\bhow much\b.*\b(all|altogether|overall)\b", re.I)
_TAX_TALK = re.compile(r"\b(vat|tax)\b", re.I)
_MARGIN_TALK = re.compile(r"\bmargin\b", re.I)
_SUM_TALK = re.compile(r"\bwork(ed|ing)? (it |that |this )?out\b|\bcalculat|\bhow much\b|\bwhat(\'s| is| does| would| will)\b"
                       r"|\bper (seat|unit|user|head|day|month|year|site)\b.*\?|\btotal\b|\bcomes? to\b|=\s*\?", re.I)
# The seller wants a price DERIVED ("budget of 12,000 across 340 seats, work out the
# price per seat"). Seen against the real model: it filed the 12,000 as the unit price.
_DERIVE_TALK = re.compile(r"\bbudget\b|\bspread\b|\bsplit\b|\bdivid|\bwork(ed|ing)? (it |that |this |the \w+ )?out\b"
                          r"|\bcalculate\b", re.I)


def asked_sums(message: str) -> List[Dict[str, Any]]:
    """The sums on the quote's own figures the seller asked for, read from their words."""
    out: List[Dict[str, Any]] = []
    if _TAX_TALK.search(message or ""):
        rates = stated_percents(message)
        out.append({"op": "calculate", "calc": "with_tax",
                    "percent": rates[0] if len(set(rates)) == 1 else None})
    elif _TOTAL_TALK.search(message or ""):
        out.append({"op": "calculate", "calc": "quote_total"})
    if _MARGIN_TALK.search(message or "") and not stated_percents(message):
        out.append({"op": "calculate", "calc": "margin"})    # "40% margin" is not a question
    return out


def _after(quote: Mapping[str, Any], change: Mapping[str, Any]) -> Dict[str, Any]:
    """The quote as it would stand if the proposal were applied -- only to say what
    would still be missing. Nothing is saved."""
    patches = {str(p.get("id")): p for p in change.get("lines") or []}
    lines = [{**l, **patches.get(str(l.get("id")), {})}
             for l in (quote.get("lines") or []) if isinstance(l, dict)]
    words = {x.get("key"): x.get("html") for x in change.get("sections") or []}
    sections = [{**x, "html": words[x.get("key")]} if x.get("key") in words else x
                for x in (quote.get("sections") or []) if isinstance(x, dict)]
    return {**quote, "lines": lines + list(change.get("addLines") or []),
            "sections": sections,
            "scopeHtml": change.get("scopeHtml", quote.get("scopeHtml")),
            "customer": {**(quote.get("customer") or {}), **(change.get("customer") or {})},
            "terms": {**(quote.get("terms") or {}), **(change.get("terms") or {})}}


def _still_needed(needs: List[Mapping[str, str]]) -> str:
    labels = [n["label"] for n in needs]
    if not labels:
        return " That is everything this template asks for."
    return " Still needed: " + "; ".join(labels) + "."


def _sum_turn(quote: Mapping[str, Any], message: str, call: Callable[..., Optional[str]],
              wording: Optional[Callable[[], str]]) -> List[Dict[str, Any]]:
    """The sum the seller asked for on their own numbers, as a calculate intent -- written
    down by the model, never worked out by it. Asked only when the message talks about
    working something out and holds at least two numbers to do it with. Anything short
    of a readable answer is no sum at all; the fill carries on without it."""
    from src.services.agent_policy.extractor import _fill
    if not (_SUM_TALK.search(message) or _DERIVE_TALK.search(message)) \
            or len(set(stated_numbers(message))) < 2:
        return []
    try:
        template = (wording or (lambda: _default_prompt(SUM_PROMPT_NAME)))()
        raw = call(_fill(template, {"quote": _quote_for_model(quote), "message": message}),
                   format=SUM_SCHEMA, think=False, temperature=0, num_predict=256,
                   use_load_options=True)
        said = json.loads(raw or "")
        expression = str(said.get("expression") or "").strip()
    except (AgentUnavailable, ValueError, AttributeError):
        return []
    if not expression:
        return []
    return [{"op": "calculate", "calc": "expression", "expression": expression,
             "label": said.get("label"), "into": said.get("into") or None,
             "line_id": said.get("line_id") or None}]


def _default_prompt(name: str = PROMPT_NAME) -> str:
    from src.services.agent_policy.extractor import PromptUnavailable, load_prompt
    try:
        return load_prompt(name)
    except PromptUnavailable as exc:
        raise AgentUnavailable(str(exc)) from None


def _default_busy() -> bool:
    from src.services.agent_policy.extractor import model_busy
    return model_busy()


def propose(quote: Mapping[str, Any], message: str, *, checks: Mapping[str, Any],
            session_id: Optional[str] = None,
            template: Optional[Mapping[str, Any]] = None,
            call: Optional[Callable[..., Optional[str]]] = None,
            prompt: Optional[Callable[[], str]] = None,
            sum_prompt: Optional[Callable[[], str]] = None,
            busy: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """One turn. Returns {summary, change, reads, session_id, refused, missing}; `change`
    is None whenever `refused` is a sentence. Raises AgentUnavailable when no turn could
    be run.

    ``template`` is the template the quote was started from (templates.get), if any.
    With one, the turn fills the quote in from what the seller says and `missing` lists
    what the template still asks for; without one, `missing` is what any quote needs."""
    from src.services.sell_side import templates as quote_templates
    from src.services.agent_policy.extractor import _fill

    session = session_id or str(quote.get("id") or "")
    message = (message or "").strip()
    if not message:
        raise ValueError("message is needed")
    if len(message) > MAX_MESSAGE:
        raise ValueError(f"message is longer than {MAX_MESSAGE} characters")

    def answer(*, summary: str, change=None, reads=(), refused=None,
               calculations=()) -> Dict[str, Any]:
        return {"summary": summary, "change": change, "reads": list(reads),
                "session_id": session, "refused": refused,
                # every sum, with its working. Nothing here is on the quote; a figure
                # reaches it only as part of `change`, and only when the seller applies it.
                "calculations": list(calculations),
                "missing": quote_templates.missing(template, _after(quote, change or {}))}

    if (quote.get("status") or "draft") != "draft":
        sentence = ("This quote is no longer a draft, so I will not propose changes to it. "
                    "Supersede it to start the next version.")
        return answer(summary=sentence, refused=sentence)

    filling = template is not None
    wording = (prompt or (lambda: _default_prompt(FILL_PROMPT_NAME if filling
                                                  else PROMPT_NAME)))()
    # Only the real model is guarded by default; a stand-in `call` reaches no model.
    if (busy or (_default_busy if call is None else (lambda: False)))():
        raise AgentUnavailable("the model is busy with other work; try again shortly")
    raw = (call or ollama_generate)(
        _fill(wording, {"today": dt.date.today().isoformat(),
                        "quote": _quote_for_model(quote), "message": message,
                        "template": _template_for_model(template) if filling else ""}),
        format=FILL_SCHEMA if filling else SCHEMA, think=False, temperature=0,
        num_predict=2048 if filling else 1024, use_load_options=True)
    if not raw:
        raise AgentUnavailable("the model did not answer")
    try:
        said = json.loads(raw)
        intents = said.get("intents") or []
        if not isinstance(intents, list) or not all(isinstance(i, dict) for i in intents):
            raise ValueError("intents")
    except (ValueError, AttributeError):
        raise AgentUnavailable("the model's answer could not be read") from None

    if filling:
        # Sections and the quote's own sums are read from the message in code; from the
        # model only a sum it had to set up is kept, and only if a sum was asked for.
        intents = ([i for i in intents if i.get("op") not in ("set_section", "calculate")]
                   + section_words(message, quote.get("sections")) + asked_sums(message)
                   + _sum_turn(quote, message, call or ollama_generate, sum_prompt))
    if not intents:
        # The model's own words are not repeated: a refusal is prose, and prose is where
        # an invented figure would ride out.
        sentence = ("I could not turn that into a change to this quote. I can set a markup "
                    "or a discount (on every line or on one), change how long the quote is "
                    "valid, or add a line."
                    + (" On this template I can also fill in quantities, prices, costs, the "
                       "customer and the terms from what you tell me." if filling else ""))
        return answer(summary=sentence, refused=sentence)
    try:
        change, did, reads, sums = _work_out(intents, quote, checks, message, filling=filling)
    except _Refuse as no:
        return answer(summary=str(no), refused=str(no))
    worked = [f"{c['label']}: {c['working']}." for c in sums]
    if sums:
        worked.append("Check these figures before you rely on them: I worked them out "
                      "from what is on the quote and what you typed.")
    if not change and sums:
        # an answer to a question: nothing to apply, and not a refusal
        # (what the fill turn offered unasked alongside a question is not worth reporting)
        return answer(summary=" ".join(worked), calculations=sums,
                      reads=["what you typed", "the figures on this quote"])
    if not change:
        # `did` holds only the note of what was left out, if anything was
        sentence = " ".join(["Nothing in that gave me a value to fill in on this quote."] + did)
        return answer(summary=sentence, refused=sentence)
    ref = quote.get("ref") or quote.get("number") or quote.get("id")
    if filling:
        reads = [f"the “{template.get('name')}” template", "what you typed"] + reads
        tail = _still_needed(quote_templates.missing(template, _after(quote, change)))
    else:
        tail = ""
    return answer(summary=" ".join(did + worked) + tail, change=change, calculations=sums,
                  reads=([f"quote {ref}"] if ref else []) + reads)
