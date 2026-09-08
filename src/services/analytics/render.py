"""The analytic answer, laid out. Same answer in, same page out.

The model is not asked to format anything here. It writes at most one sentence
(the headline, once the insight writer exists) and this module decides
everything else: what is stated first, which figure sits in which column, how
wide the bar is, and which caveat the reader has to see before they trust the
table. That is the whole point of the deterministic layer — an answer's shape
must not depend on which markdown the model felt like emitting.

The output is the ``<section class="agent-answer">`` envelope the ask path
already returns, so it flows through ``_normalise_answer_html`` untouched (that
function passes a complete section straight through) and lands in the chat
bubble and the SpendIQ ask panel through the renderer they already share.

Two constraints from the far end of that pipe shape everything below:

  * ``beyond_procwise_ui/src/lib/agentAnswer.js`` rebuilds this HTML node by
    node against its own allowlist. A tag outside ``ASK_HTML_TAGS`` is
    *unwrapped*, not merely unstyled — so ``<tfoot>`` would spill the totals
    into the prose, and the totals row lives in the body instead.
  * Every attribute except ``class`` is dropped. There is no ``style`` to put a
    bar width in, so the bar is a class bucket in steps of 5, and the width
    lives in the stylesheet beside it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from html import escape
from typing import Any, Dict, List, Optional

from src.services.analytics.formatting import (
    EMPTY_AMOUNT,
    format_delta,
    format_int,
    format_money,
    format_pct,
)
from src.services.analytics.models import (
    AnalyticAnswer,
    Align,
    Column,
    ColumnType,
    Confidence,
    CurrencyBasis,
    Fact,
    FactCode,
)

# The client's own allowlist, restated. Kept in sync deliberately rather than
# imported (it is a different language in a different repository), and asserted
# in tests/services/analytics/test_render.py so a tag added here that the client
# would unwrap fails on this side first.
ASK_HTML_TAGS = frozenset({
    "SECTION", "ARTICLE", "HEADER", "DIV", "P", "H2", "H3", "UL", "OL", "LI",
    "DL", "DT", "DD", "TABLE", "THEAD", "TBODY", "TR", "TH", "TD", "STRONG",
    "EM", "BR",
})

FLAG_MARK = "⚠"

# The bar is a class, because no other attribute survives the client. Steps of
# five keep the stylesheet to twenty-one rules and are finer than the eye reads
# off a 120px bar anyway.
_BAR_STEP = 5


def render_analytic_answer(answer: AnalyticAnswer) -> str:
    """The answer as HTML: scope, headline, table, footnotes, provenance."""

    parts: List[str] = [
        '<section class="agent-answer">',
        '<article class="agent-answer__content">',
        '<div class="agent-answer__segment agent-answer__segment--analytic">',
        f'<p class="agent-answer__scope">{escape(answer.scope.line())}</p>',
        _headline(answer),
    ]
    if answer.table.rows:
        parts.append(_table(answer))
    notes = _notes(answer)
    if notes:
        parts.append(notes)
    parts.append(_provenance(answer))
    parts.extend(["</div>", "</article>", "</section>"])
    return "".join(parts)


def _headline(answer: AnalyticAnswer) -> str:
    text = escape(answer.headline.text)
    if answer.headline.confidence is not Confidence.ASSERTED:
        badge = answer.headline.confidence.value.capitalize()
        text += f' <em class="agent-answer__confidence">{escape(badge)}</em>'
    return f'<p class="agent-answer__lead">{text}</p>'


# -- the table -------------------------------------------------------------


def _cell_class(column: Column) -> str:
    edge = "right" if column.align is Align.RIGHT else "left"
    return f"agent-answer__cell agent-answer__cell--{edge}"


def _as_number(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _rendered(value: Any, column: Column) -> str:
    """One value as text, always through the shared formatter."""
    if value is None:
        return EMPTY_AMOUNT
    if column.type is ColumnType.MONEY:
        return format_money(value, column.currency, column.decimals)
    if column.type is ColumnType.PCT:
        return format_pct(value, column.decimals)
    if column.type is ColumnType.DELTA:
        return format_delta(value, column.decimals)
    if column.type is ColumnType.INT:
        return format_int(value)
    return str(value)


def _bar_bucket(value: Any, largest: Optional[Decimal]) -> Optional[int]:
    """The row's share of the largest figure, to the nearest five percent."""
    amount = _as_number(value)
    if amount is None or largest is None or largest <= 0 or amount <= 0:
        return None
    share = (amount / largest) * 100
    steps = (share / _BAR_STEP).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return min(100, int(steps) * _BAR_STEP)


def _largest(answer: AnalyticAnswer, column: Column) -> Optional[Decimal]:
    """The biggest figure in the primary column, or None if no bar belongs here.

    As billed, the rank restarts within each currency and the column holds
    amounts in different denominations. A bar across them would draw exactly
    the cross-currency comparison the answer refuses to state.
    """
    if not column.is_primary or answer.scope.currency_basis is CurrencyBasis.NATIVE:
        return None
    values = [v for v in (_as_number(row.get(column.key)) for row in answer.table.rows) if v]
    return max(values) if values else None


def _row_html(row: Dict[str, Any], columns: List[Column], answer: AnalyticAnswer,
              largest: Dict[str, Optional[Decimal]], *, flagged: bool,
              css_class: Optional[str] = None) -> str:
    cells: List[str] = []
    for column in columns:
        if column.key not in row:
            # Absent is not the same as empty: a totals row has no rank, and a
            # dash there would read as a figure we looked for and lost.
            cells.append(f'<td class="{_cell_class(column)}"></td>')
            continue
        content = escape(_rendered(row.get(column.key), column))
        if flagged and column.is_primary:
            content += f" {FLAG_MARK}"
        bucket = _bar_bucket(row.get(column.key), largest.get(column.key))
        if bucket is not None:
            content += f'<div class="agent-answer__bar agent-answer__bar--{bucket}"></div>'
        cells.append(f'<td class="{_cell_class(column)}">{content}</td>')
    opening = f'<tr class="{css_class}">' if css_class else "<tr>"
    return opening + "".join(cells) + "</tr>"


def _table(answer: AnalyticAnswer) -> str:
    columns = answer.table.columns
    head = "".join(
        f'<th class="{_cell_class(column)}">{escape(column.label)}</th>' for column in columns
    )
    largest = {column.key: _largest(answer, column) for column in columns}
    body = "".join(
        _row_html(row, columns, answer, largest, flagged=bool(_footnoted_flags(row, answer)))
        for row in answer.table.rows
    )
    if answer.table.totals:
        # In the body, not a <tfoot>: the client's allowlist has no tfoot, and
        # an unwrapped one spills the totals into the prose under the table.
        body += _row_html(answer.table.totals, columns, answer, {}, flagged=False,
                          css_class="agent-answer__row--total")
    return (
        '<table class="agent-answer__table agent-answer__table--analytic">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


# -- footnotes -------------------------------------------------------------


def _currency_mismatch_note(fact: Fact) -> str:
    return (f"{FLAG_MARK} {fact.entity} bills in {fact.display} currencies "
            f"({fact.unit}); the figure shown is the converted total.")


def _negligible_base_note(fact: Fact) -> str:
    return (f"{FLAG_MARK} {fact.entity}'s change is measured from {fact.display} in the "
            f"period before, so the percentage is arithmetic rather than a finding.")


# A flag is only worth marking on a row if the reader can find out what it
# means. Anything without a note here is not marked at all, rather than leaving
# a warning glyph on screen with nothing under the table to explain it.
_FLAG_NOTES = {
    FactCode.CURRENCY_MISMATCH: _currency_mismatch_note,
    FactCode.NEGLIGIBLE_BASE: _negligible_base_note,
}


def _footnoted_flags(row: Dict[str, Any], answer: AnalyticAnswer) -> List[Fact]:
    """The facts behind this row's flags, for the ones we can explain."""
    flags = row.get("_flags") or []
    if not flags:
        return []
    ref = row.get("entity_ref")
    return [fact for fact in answer.facts
            if fact.code.value in flags and fact.code in _FLAG_NOTES
            and (ref is None or fact.entity_ref == ref)]


def _notes(answer: AnalyticAnswer) -> str:
    """The marks explained, then everything the reader must know to trust it."""
    lines: List[str] = []
    seen: set[tuple] = set()
    for row in answer.table.rows:
        for fact in _footnoted_flags(row, answer):
            key = (fact.code, fact.entity_ref)
            if key in seen:
                continue
            seen.add(key)
            lines.append(_FLAG_NOTES[fact.code](fact))
    # Already worst-first: AnalyticAnswer sorts anomalies by severity on build.
    lines.extend(anomaly.text for anomaly in answer.anomalies)
    if not lines:
        return ""
    items = "".join(f'<li>{escape(line)}</li>' for line in lines)
    return f'<ul class="agent-answer__notes">{items}</ul>'


# -- provenance ------------------------------------------------------------


def _stamp(raw: str) -> str:
    """The refresh time as a reader reads it, or exactly as it came if we cannot."""
    try:
        parsed = datetime.fromisoformat((raw or "").replace("Z", "+00:00"))
    except ValueError:
        return raw
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return f"{parsed.day} {parsed.strftime('%b %Y')}, {parsed.strftime('%H:%M')} UTC"


def _provenance(answer: AnalyticAnswer) -> str:
    provenance = answer.provenance
    sources = " · ".join(
        f"{name} ({format_int(count)} rows)" for name, count in provenance.source_counts.items()
    )
    parts = [part for part in (sources, f"refreshed {_stamp(provenance.refreshed_at)}",
                               provenance.query_ref) if part]
    return f'<p class="agent-answer__provenance">{escape(" · ".join(parts))}</p>'
