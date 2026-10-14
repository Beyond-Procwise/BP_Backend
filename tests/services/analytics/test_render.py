"""The answer as the reader sees it, laid out here rather than by a model.

Today the ask path hands the model a block of key=value text and takes back
whatever markdown it writes: the figures, the glyphs, the precision and the
order are all the model's. This renderer takes the typed answer and lays it out
the same way every time — scope line, headline, table, footnotes, provenance —
so the shape of an answer stops depending on which sentence the model felt like
writing.

Two contracts are held here rather than assumed:

  * **The client's allowlist.** ``beyond_procwise_ui/src/lib/agentAnswer.js``
    rebuilds this HTML node by node, keeps only ``ASK_HTML_TAGS`` and drops
    every attribute except ``class``. A tag outside that set is not "styled
    differently" on screen — it is unwrapped, and its cells run together into a
    line of prose. So the tags are asserted, not trusted.
  * **Nothing here formats a figure itself.** Every money, share, count and
    delta goes through ``services/analytics/formatting``, the same one the
    dashboard's tiles are drawn with.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
import re

import pytest

from src.services.analytics.currency import NATIVE, DisplayCurrency
from src.services.analytics.models import Confidence, Headline
from src.services.analytics.next_steps import ALL_DATA, AllowAll, select_next_steps
from src.services.analytics.period import Period
from src.services.analytics.render import ASK_HTML_TAGS, render_analytic_answer
from src.services.analytics.supplier_spend import SupplierSpendRow, build_supplier_spend_ranking

RATES = {"USD": Decimal("1.0"), "GBP": Decimal("0.739722"), "INR": Decimal("94.547339")}
FETCHED = datetime(2026, 9, 7, 9, 53, tzinfo=timezone.utc)
PERIOD = Period(date(2026, 4, 1), date(2026, 9, 7), "FY26 YTD")
PRIOR = Period(date(2025, 4, 1), date(2025, 9, 7), "FY25 YTD")


def _row(name, amount, currency="GBP", supplier_id=None):
    return SupplierSpendRow(
        supplier_id=supplier_id or name.lower().replace(" ", "-"),
        supplier_name=name, currency=currency, amount=Decimal(str(amount)), invoices=1)


def _answer(rows=None, display=None, **kw):
    params = dict(
        rows=rows if rows is not None else [_row("Kestrel Supplies 8", "1200000"),
                                            _row("Blackwood Group 10", "900000")],
        display=display or DisplayCurrency(target="GBP", rates=RATES, fetched_at=FETCHED),
        period=PERIOD, population_count=2, invoice_count=100,
        answer_id="a-1", refreshed_at="2026-09-07T20:48:00Z",
        concentration_threshold_pct=Decimal("200"),  # off unless a test wants it
    )
    params.update(kw)
    return build_supplier_spend_ranking(**params)


def _tags(html):
    return {match.group(1).upper() for match in re.finditer(r"<\s*/?\s*([A-Za-z][A-Za-z0-9]*)", html)}


def _body_row(html, name):
    """The one row of the table body that names ``name`` — not the headline."""
    body = html.split("<tbody")[1].split("</tbody")[0]
    return [chunk for chunk in body.split("<tr") if name in chunk][0]


def _attributes(html):
    return {match.group(1).lower() for match in re.finditer(r"<[a-z]+\s([^>]*?)=", html)}


class TestTheEnvelopeTheClientAccepts:
    def test_the_shape_is_the_one_the_client_recognises_as_an_agent_answer(self):
        # isStructuredAgentAnswer() in agentAnswer.js: starts <section, and
        # carries class="agent-answer". Fail either and the whole answer is
        # printed as literal markup in the chat bubble.
        html = render_analytic_answer(_answer())
        assert html.startswith("<section")
        assert 'class="agent-answer"' in html
        assert html.endswith("</section>")

    def test_every_tag_emitted_is_one_the_client_allowlist_keeps(self):
        html = render_analytic_answer(_answer())
        assert _tags(html) <= ASK_HTML_TAGS

    def test_the_totals_row_is_in_the_body_because_the_client_drops_tfoot(self):
        html = render_analytic_answer(_answer())
        assert "TFOOT" not in _tags(html)
        assert "All 2 suppliers" in html
        body = html.split("<tbody")[1]
        assert "All 2 suppliers" in body

    def test_no_attribute_but_class_is_emitted_because_no_other_one_survives(self):
        html = render_analytic_answer(_answer())
        assert _attributes(html) == {"class"}


class TestTheScopeLine:
    def test_the_answer_opens_with_what_it_covers(self):
        answer = _answer()
        html = render_analytic_answer(answer)
        assert answer.scope.line() in html
        assert html.index(answer.scope.line()) < html.index(answer.headline.text)

    def test_an_empty_answer_still_says_what_it_looked_at(self):
        answer = _answer(rows=[], population_count=0)
        html = render_analytic_answer(answer)
        assert answer.scope.line() in html


class TestTheHeadline:
    def test_an_asserted_headline_carries_no_badge(self):
        html = render_analytic_answer(_answer())
        assert "Unassessed" not in html
        assert "Corroborated" not in html

    def test_a_headline_that_is_not_asserted_says_so(self):
        answer = _answer()
        answer = answer.model_copy(update={
            "headline": Headline(text="Kestrel leads.", confidence=Confidence.UNASSESSED)})
        assert "Unassessed" in render_analytic_answer(answer)


class TestTheTable:
    def test_money_is_drawn_by_the_shared_formatter_not_by_the_renderer(self):
        html = render_analytic_answer(_answer())
        assert "£1.2M" in html
        assert "1200000" not in html
        assert "1,200,000" not in html

    def test_a_share_is_a_share_and_a_rank_is_a_count(self):
        html = render_analytic_answer(_answer())
        assert "57.1%" in html  # 1.2M of 2.1M

    def test_no_table_is_drawn_when_there_is_nothing_to_rank(self):
        html = render_analytic_answer(_answer(rows=[], population_count=0))
        assert "<table" not in html

    def test_figures_are_marked_for_the_right_hand_edge_and_names_are_not(self):
        html = render_analytic_answer(_answer())
        header = html.split("<tbody")[0]
        spend = re.search(r"<th class=\"([^\"]*)\">Invoiced spend</th>", header)
        supplier = re.search(r"<th class=\"([^\"]*)\">Supplier</th>", header)
        assert spend is not None and "--right" in spend.group(1)
        assert supplier is not None and "--right" not in supplier.group(1)

    def test_a_supplier_name_cannot_carry_markup_into_the_page(self):
        html = render_analytic_answer(_answer(rows=[_row("<script>alert(1)</script> & Co", "10")]))
        assert "<script" not in html
        assert "&lt;script&gt;" in html
        assert "&amp; Co" in html


class TestTheFlaggedRows:
    def _multi_currency(self):
        return _answer(rows=[_row("Harbourline Trading 13", "1000000", "GBP"),
                             _row("Harbourline Trading 13", "50000000", "INR"),
                             _row("Blackwood Group 10", "900000")])

    def test_a_row_the_facts_flagged_is_marked_in_the_table(self):
        html = render_analytic_answer(self._multi_currency())
        row = _body_row(html, "Harbourline")
        assert "⚠" in row

    def test_the_mark_is_explained_underneath_rather_than_left_hanging(self):
        html = render_analytic_answer(self._multi_currency())
        footnotes = html.split("</table>")[1]
        assert "Harbourline Trading 13" in footnotes
        assert "GBP" in footnotes and "INR" in footnotes

    def test_an_unflagged_row_is_not_marked(self):
        html = render_analytic_answer(self._multi_currency())
        row = _body_row(html, "Blackwood")
        assert "⚠" not in row


class TestAChangeMeasuredFromNothing:
    def _answer(self):
        from src.services.analytics.period import Period
        from src.services.analytics.supplier_spend import Lens

        prior = Period(date(2025, 4, 1), date(2025, 9, 7), "FY25 YTD")
        return _answer(rows=[_row("Kestrel Supplies 8", "1200000"),
                             _row("Featherstone Ltd", "50000")],
                       prior_rows=[_row("Kestrel Supplies 8", "1000000"),
                                   _row("Featherstone Ltd", "10")],
                       prior_period=prior, lens=Lens.TREND)

    def test_the_row_is_marked(self):
        assert "⚠" in _body_row(render_analytic_answer(self._answer()), "Featherstone")

    def test_the_base_it_was_measured_from_is_under_the_table(self):
        footnotes = render_analytic_answer(self._answer()).split("</table>")[1]
        assert "Featherstone Ltd" in footnotes
        assert "£10" in footnotes


class TestTheInlineBar:
    def test_the_bar_is_measured_against_the_largest_figure_in_its_column(self):
        html = render_analytic_answer(_answer())
        buckets = re.findall(r"agent-answer__bar--(\d+)", html)
        assert buckets == ["100", "75"]  # 1.2M leads, 900K is three quarters of it

    def test_no_bar_is_drawn_across_currencies_nobody_converted(self):
        # As billed, the rank restarts per currency and the figures are in
        # different denominations. A bar comparing them would draw the very
        # cross-currency ranking the answer refuses to state.
        answer = _answer(rows=[_row("Kestrel Supplies 8", "1200000", "GBP"),
                               _row("Harbourline Trading 13", "50000000", "INR")],
                         display=DisplayCurrency(target=NATIVE, rates=RATES, fetched_at=FETCHED))
        assert "agent-answer__bar" not in render_analytic_answer(answer)


class TestTheFootnotesAndProvenance:
    def _two_anomalies(self):
        return _answer(
            rows=[_row("Kestrel Supplies 8", "1200000"),
                  _row("Tokyo Metal Works", "800000", "JPY")],
            prior_rows=[], prior_period=PRIOR)

    def test_the_worst_anomaly_is_read_first(self):
        answer = self._two_anomalies()
        codes = [a.code.value for a in answer.anomalies]
        assert codes[0] == "UNCONVERTED_CURRENCY" and "MISSING_PERIOD_DATA" in codes
        html = render_analytic_answer(answer)
        assert html.index("No exchange rate") < html.index("No comparable spend")

    def test_the_reader_is_told_what_was_read_and_when(self):
        html = render_analytic_answer(_answer())
        assert "100 invoices" in html
        assert "7 Sep 2026" in html

    def test_what_was_read_is_named_as_the_product_names_it(self):
        # Live, the first answer this layer ever served through the API came
        # back as "I couldn't retrieve that. I've raised it with the team." —
        # the output-safety gate replacing the whole thing, because the
        # provenance line named proc.bp_invoice_trgt. It was right to: a table
        # name is a description of the backend, and this line is read by a
        # buyer. The calculation's own reference stays in the payload, where
        # "how was this calculated" can reach it, and out of the prose.
        html = render_analytic_answer(_answer())
        assert "bp_invoice_trgt" not in html
        assert "supplier_spend_ranking/v1" not in html

    def test_the_whole_answer_passes_the_gate_that_stands_between_it_and_a_user(self):
        from src.services import output_safety

        assert output_safety.inspect(render_analytic_answer(_answer())) == []

    def test_a_refreshed_stamp_we_cannot_read_is_shown_as_it_came(self):
        html = render_analytic_answer(_answer(refreshed_at="just now"))
        assert "just now" in html


class TestTheNextSteps:
    def test_the_steps_are_not_drawn_here_because_a_chip_dispatches_an_action(self):
        # A step carries action_id and entity_refs, and a click must dispatch
        # them. Rendered into this HTML it would be text the client can only
        # re-ask as a question — the round trip that loses the entity, and the
        # defect the next-step engine exists to end. They travel structurally.
        answer = _answer(concentration_threshold_pct=Decimal("10"))
        steps = select_next_steps(answer, persona="default", entitlements=AllowAll(),
                                  available_data=ALL_DATA)
        answer = answer.model_copy(update={"next_steps": steps})
        assert steps, "the fixture must produce steps for this test to mean anything"
        html = render_analytic_answer(answer)
        for step in steps:
            assert step.label not in html
