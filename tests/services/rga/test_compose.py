"""The composer, and the two things it must never do.

No GPU and no Ollama: every test injects its own generator. That is not only
for speed — a suite that reaches for the card evicts the resident model out from
under whatever else is running on this box.

The tests are written around failures rather than the happy path, because the
happy path is the easy half. What matters is that a model which fabricates a
figure, returns rubbish, or is simply down produces **no report at all** rather
than a plausible one.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.services.analytics.models import Confidence
from src.services.rga import audit
from src.services.rga.compose import (
    CompositionDraft,
    CompositionError,
    PROMPT_VERSION,
    TypeSelection,
    build_prompt,
    compose_report,
    select_report_type,
    to_ast,
)
from src.services.rga.models import FormatHint, ReportAST

from tests.services.rga.conftest import make_fact, make_pack


# --------------------------------------------------------------------------
# Canned model responses
# --------------------------------------------------------------------------

# Draft shape, not AST shape: one typed array per block kind and no union
# anywhere. See composition_schema() for the measurements behind that.

GOOD = {
    "sections": [
        {"id": "exec_summary", "title": "Executive summary",
         "metrics": [{"fact_ref": "F0001", "emphasis": "primary"}],
         "narratives": [
             {"text": "Invoiced spend was {{F0001}} across {{F0002}} deals.",
              "fact_refs": ["F0001", "F0002"], "role": "statement"}]},
    ]
}

# A figure typed straight into the prose. No JSON schema can forbid this, and
# live the model does it constantly — "£5.8M across 376 deals" was its first
# unprompted answer.
FABRICATED = {
    "sections": [
        {"id": "exec_summary", "title": "Executive summary",
         "narratives": [
             {"text": "Invoiced spend was £7.2M, up 14% on the prior period.",
              "fact_refs": [], "role": "statement"}]},
    ]
}

UNKNOWN_REF = {
    "sections": [
        {"id": "s", "title": "S",
         "metrics": [{"fact_ref": "F9999", "emphasis": "primary"}]},
    ]
}

RECOMMENDS_ON_UNASSESSED = {
    "sections": [
        {"id": "s", "title": "S",
         "narratives": [
             {"text": "Prioritise savings capture, given {{F0004}}.",
              "fact_refs": ["F0004"], "role": "recommendation"}]},
    ]
}

QUOTES_WITHOUT_DECLARING = {
    "sections": [
        {"id": "s", "title": "S",
         "narratives": [{"text": "Spend was {{F0001}}.", "fact_refs": [],
                         "role": "statement"}]},
    ]
}

# Every content array absent: a section that holds nothing at all.
EMPTY = {"sections": [{"id": "s", "title": "Executive Summary"}]}


def replies(*payloads):
    """A generator that returns each canned payload in turn, recording calls."""
    seen = []

    def _generate(prompt, *, schema=None, temperature=0.0):
        seen.append({"prompt": prompt, "temperature": temperature, "schema": schema})
        index = min(len(seen) - 1, len(payloads) - 1)
        payload = payloads[index]
        return payload if isinstance(payload, str) else json.dumps(payload)

    _generate.calls = seen
    return _generate


class Recorder:
    """Stands in for the audit writer."""

    def __init__(self):
        self.rows = []

    def __call__(self, **kwargs):
        self.rows.append(kwargs)

    def of(self, action_type):
        return [r for r in self.rows if r["action_type"] == action_type]


# --------------------------------------------------------------------------


class TestTheHappyPath:
    def test_it_returns_a_validated_ast(self, pack, brief):
        ast = compose_report(pack, brief, "exec_procurement_summary",
                             generate=replies(GOOD), emit_audit=False)

        assert isinstance(ast, ReportAST)
        assert [s.id for s in ast.sections] == ["exec_summary"]
        assert ast.fact_refs() == {"F0001", "F0002"}

    def test_the_model_is_shown_rendered_figures_never_raw_amounts(self, pack, brief):
        """The same rule analytics/insight holds: a model shown raw amounts is a
        model formatting money again."""
        prompt = build_prompt(pack, brief, "exec_procurement_summary")

        assert "£5.8M" in prompt          # the rendered form
        assert "5833817.90" not in prompt  # the raw value

    def test_the_prompt_marks_which_facts_may_not_be_recommended_on(self, pack, brief):
        prompt = build_prompt(pack, brief, "exec_procurement_summary")

        assert "UNASSESSED" in prompt
        assert "never recommended on" in prompt

    def test_the_schema_handed_to_the_grammar_contains_no_union(self, pack, brief):
        """A regression guard with a measured reason behind it.

        Ollama's converter does not honour oneOf/discriminator. Handing it the
        tagged AST produced 21 finding_list blocks; removing finding_list
        produced objects with no type field at all; removing the union entirely
        produced correct output. Reintroducing a union here would not fail
        loudly — it would quietly go back to composing empty reports.
        """
        generate = replies(GOOD)
        compose_report(pack, brief, "exec_procurement_summary",
                       generate=generate, emit_audit=False)

        blob = json.dumps(generate.calls[0]["schema"])
        assert '"oneOf"' not in blob
        assert '"anyOf"' not in blob
        assert '"discriminator"' not in blob

    def test_closed_vocabularies_are_constrained_in_the_grammar(self, pack, brief):
        """Anything a schema CAN forbid is forbidden there, not caught later.

        These were bare strings first, and the model invented a role the AST
        then refused at conversion — a whole composition thrown away over a
        word the grammar could have ruled out.
        """
        generate = replies(GOOD)
        compose_report(pack, brief, "exec_procurement_summary",
                       generate=generate, emit_audit=False)
        defs = generate.calls[0]["schema"]["$defs"]

        assert defs["DraftNarrative"]["properties"]["role"]["enum"] == [
            "statement", "recommendation"]
        assert defs["DraftMetric"]["properties"]["emphasis"]["enum"] == [
            "primary", "secondary"]
        assert defs["DraftChart"]["properties"]["chart_type"]["enum"] == [
            "bar", "waterfall", "line", "donut"]

    def test_the_prompt_forbids_naming_the_period(self, pack, brief):
        """Live, "in 2026 Q1" cost a composition. The scope is on the front
        page; prose does not need to repeat it."""
        prompt = build_prompt(pack, brief, "exec_procurement_summary")

        assert "Do NOT name the period" in prompt


class TestFabricatedFigures:
    def test_a_number_typed_into_prose_is_rejected(self, pack, brief):
        """The phase 2 exit test, first half. No JSON schema can forbid a digit
        inside a string, so the type validator is what catches it."""
        with pytest.raises(CompositionError) as raised:
            compose_report(pack, brief, "exec_procurement_summary",
                           generate=replies(FABRICATED, FABRICATED),
                           emit_audit=False)

        assert "literal number" in str(raised.value)

    def test_it_is_handed_back_once_and_a_corrected_answer_is_accepted(
            self, pack, brief):
        """A rejection is a correctable mistake, not a dead end."""
        generate = replies(FABRICATED, GOOD)

        ast = compose_report(pack, brief, "exec_procurement_summary",
                             generate=generate, emit_audit=False)

        assert isinstance(ast, ReportAST)
        assert len(generate.calls) == 2
        assert "was rejected" in generate.calls[1]["prompt"]
        assert "literal number" in generate.calls[1]["prompt"]

    def test_the_retry_runs_at_a_non_zero_temperature(self, pack, brief):
        """At 0 the decoder is deterministic, so a retry reproduces the rejected
        answer byte for byte and the second attempt is pure waste."""
        generate = replies(FABRICATED, GOOD)
        compose_report(pack, brief, "exec_procurement_summary",
                       generate=generate, emit_audit=False)

        assert generate.calls[0]["temperature"] == 0.0
        assert generate.calls[1]["temperature"] > 0.0

    def test_it_gives_up_after_one_retry_rather_than_looping(self, pack, brief):
        generate = replies(FABRICATED, FABRICATED, GOOD)

        with pytest.raises(CompositionError):
            compose_report(pack, brief, "exec_procurement_summary",
                           generate=generate, emit_audit=False)

        assert len(generate.calls) == 2


class TestItFailsClosed:
    @pytest.mark.parametrize("bad,expected", [
        ("not json at all", "was not JSON"),
        ("", "returned nothing"),
        (json.dumps({"sections": [{"id": "s"}]}), "did not validate"),
    ])
    def test_an_unusable_answer_yields_no_report(self, pack, brief, bad, expected):
        with pytest.raises(CompositionError) as raised:
            compose_report(pack, brief, "exec_procurement_summary",
                           generate=replies(bad, bad), emit_audit=False)

        assert expected in str(raised.value)

    def test_a_model_that_is_down_yields_no_report(self, pack, brief):
        def explode(prompt, *, schema=None, temperature=0.0):
            raise ConnectionError("ollama is not listening")

        with pytest.raises(CompositionError, match="unreachable"):
            compose_report(pack, brief, "exec_procurement_summary",
                           generate=explode, emit_audit=False)

    def test_there_is_no_fallback_report(self, pack, brief):
        """A substituted AST would make a failed composition indistinguishable
        from a successful one."""
        with pytest.raises(CompositionError):
            compose_report(pack, brief, "exec_procurement_summary",
                           generate=replies("rubbish", "rubbish"), emit_audit=False)


class TestAnEmptyReportIsNotAReport:
    def test_untagged_blocks_are_refused_rather_than_silently_rerouted(self):
        """An empty object used to validate as a finding_list, because every
        field on that block has a default. Live, the model emitted twenty-one of
        them and the whole report released stating nothing."""
        import pytest as _pytest
        from pydantic import ValidationError

        with _pytest.raises(ValidationError):
            ReportAST.model_validate(
                {"sections": [{"id": "s", "title": "S", "blocks": [{}]}]})

    def test_a_composition_that_references_no_facts_is_rejected(self, pack, brief):
        with pytest.raises(CompositionError, match="references none of"):
            compose_report(pack, brief, "exec_procurement_summary",
                           generate=replies(EMPTY, EMPTY), emit_audit=False)

    def test_the_model_is_told_so_it_can_correct_itself(self, pack, brief):
        generate = replies(EMPTY, GOOD)

        ast = compose_report(pack, brief, "exec_procurement_summary",
                             generate=generate, emit_audit=False)

        assert ast.fact_refs() == {"F0001", "F0002"}
        assert "references none of" in generate.calls[1]["prompt"]


class TestSemanticRulesTheGrammarCannotExpress:
    def test_a_reference_to_a_fact_that_does_not_exist(self, pack, brief):
        with pytest.raises(CompositionError, match="not a fact in this pack"):
            compose_report(pack, brief, "exec_procurement_summary",
                           generate=replies(UNKNOWN_REF, UNKNOWN_REF),
                           emit_audit=False)

    def test_a_recommendation_resting_on_an_unassessed_figure(self, pack, brief):
        with pytest.raises(CompositionError, match="UNASSESSED"):
            compose_report(pack, brief, "exec_procurement_summary",
                           generate=replies(RECOMMENDS_ON_UNASSESSED,
                                            RECOMMENDS_ON_UNASSESSED),
                           emit_audit=False)

    def test_prose_quoting_a_fact_it_does_not_declare(self, pack, brief):
        """Every narrative block must declare what it relies on."""
        with pytest.raises(CompositionError, match="does not list"):
            compose_report(pack, brief, "exec_procurement_summary",
                           generate=replies(QUOTES_WITHOUT_DECLARING,
                                            QUOTES_WITHOUT_DECLARING),
                           emit_audit=False)


class TestTheAuditEvent:
    def test_a_composition_records_the_model_and_the_prompt_version(
            self, pack, brief):
        writer = Recorder()
        compose_report(pack, brief, "exec_procurement_summary",
                       generate=replies(GOOD), writer=writer)

        rows = writer.of(audit.COMPOSED)
        assert len(rows) == 1
        details = rows[0]["details"]
        assert rows[0]["status"] == "ok"
        assert rows[0]["trace_id"] == pack.pack_id
        assert details["run_id"] == pack.pack_id
        assert details["pack_hash"] == pack.hash
        assert details["prompt_version"] == PROMPT_VERSION
        assert len(details["prompt_sha256"]) == 64
        assert details["attempts"] == 1
        assert details["ast_hash"]

    def test_token_counts_are_declared_unavailable_rather_than_invented(
            self, pack, brief):
        """§7 asks for them; the shared Ollama client discards them. Saying so
        is better than a number nobody measured."""
        writer = Recorder()
        compose_report(pack, brief, "exec_procurement_summary",
                       generate=replies(GOOD), writer=writer)

        counts = writer.of(audit.COMPOSED)[0]["details"]["token_counts"]
        assert "unavailable" in counts

    def test_a_failed_composition_is_recorded_with_its_faults(self, pack, brief):
        writer = Recorder()
        with pytest.raises(CompositionError):
            compose_report(pack, brief, "exec_procurement_summary",
                           generate=replies(FABRICATED, FABRICATED), writer=writer)

        row = writer.of(audit.COMPOSED)[0]
        assert row["status"] == "rejected"
        assert row["details"]["attempts"] == 2
        assert any("literal number" in f for f in row["details"]["faults"])


class TestDraftToAst:
    def test_a_draft_becomes_the_strict_ast(self):
        draft = CompositionDraft.model_validate(GOOD)
        ast = to_ast(draft)

        kinds = [b.type for b in ast.sections[0].blocks]
        assert kinds == ["metric", "narrative"]
        assert ast.fact_refs() == {"F0001", "F0002"}

    def test_blocks_come_out_in_a_fixed_order(self):
        """Figures first, then what they mean, then the detail, then findings —
        decided here rather than by the model."""
        draft = CompositionDraft.model_validate({"sections": [{
            "id": "s", "title": "S",
            "charts": [{"chart_type": "bar",
                        "series": [{"label": "Spend", "fact_refs": ["F0001"]}]}],
            "finding_refs": ["FND1"],
            "tables": [{"columns": ["Measure"], "rows": [["F0003"]]}],
            "narratives": [{"text": "Spend was {{F0001}}.", "fact_refs": ["F0001"]}],
            "metrics": [{"fact_ref": "F0001"}],
        }]})

        kinds = [b.type for b in to_ast(draft).sections[0].blocks]
        assert kinds == ["metric", "narrative", "table", "chart", "finding_list"]

    def test_conversion_is_where_the_strict_rules_bite(self):
        """The draft happily holds a digit; the AST will not."""
        draft = CompositionDraft.model_validate(FABRICATED)

        with pytest.raises(ValidationError, match="literal number"):
            to_ast(draft)

    def test_a_section_with_no_content_produces_no_blocks(self):
        assert to_ast(CompositionDraft.model_validate(EMPTY)).sections[0].blocks == []


class TestSelectReportType:
    def test_it_picks_a_listed_type(self):
        chosen = select_report_type(
            "give me the quarterly summary for the board",
            ["exec_procurement_summary", "board_paper"],
            generate=replies({"report_type_id": "board_paper", "confidence": 0.8}))

        assert isinstance(chosen, TypeSelection)
        assert chosen.report_type_id == "board_paper"
        assert chosen.confidence == 0.8

    def test_the_grammar_forbids_an_unlisted_id(self):
        """Constrained to an enum of what exists, so a hallucinated type cannot
        be decoded in the first place."""
        generate = replies({"report_type_id": "board_paper", "confidence": 0.5})
        select_report_type("anything", ["exec_procurement_summary", "board_paper"],
                           generate=generate)

        schema = generate.calls[0]["schema"]
        assert schema["properties"]["report_type_id"] == {
            "enum": ["board_paper", "exec_procurement_summary"]}

    def test_an_id_that_is_not_available_is_refused(self):
        chosen = select_report_type(
            "something else", ["exec_procurement_summary"],
            generate=replies({"report_type_id": "invented_type", "confidence": 0.9}))

        assert chosen is None

    @pytest.mark.parametrize("bad", ["", "not json", json.dumps({"nope": 1})])
    def test_an_unusable_answer_asks_the_person_instead(self, bad):
        assert select_report_type("x", ["exec_procurement_summary"],
                                  generate=replies(bad)) is None

    def test_no_types_means_no_selection(self):
        assert select_report_type("x", [], generate=replies(GOOD)) is None
