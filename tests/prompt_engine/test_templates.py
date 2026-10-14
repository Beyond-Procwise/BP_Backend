"""Templates: blanks and fragment references, and nothing else."""
from __future__ import annotations

import pytest

from services.prompt_engine import templates as T

STR = {"name": "supplier", "type": "string"}
NUM = {"name": "rounds", "type": "number"}
BOOL = {"name": "final", "type": "boolean"}
ENUM = {"name": "region", "type": "enum", "values": ["EU", "UK"]}
OBJ = {"name": "deal", "type": "object",
       "schema": {"type": "object", "required": ["id"],
                  "properties": {"id": {"type": "string"}, "value": {"type": "number"}}}}


def _no_fragments(key, version):
    return None


# --------------------------------------------------------------------------
# Reading a template
# --------------------------------------------------------------------------
def test_blanks_and_fragment_references_are_found():
    parsed = T.parse("Dear {{supplier}}, {{> brand-voice@3}} about {{deal.id}}. {{ supplier }}")

    assert parsed.blanks == {"supplier", "deal"}
    assert parsed.fragment_refs == [{"key": "brand-voice", "version": 3}]


@pytest.mark.parametrize("bad", [
    "{{#if final}}yes{{/if}}",          # a condition
    "{{ supplier | upper }}",           # a filter
    "{{ 1 + 1 }}",                      # an expression
    "{{> brand-voice}}",                # a fragment with no version
    "{{> brand-voice@latest}}",         # a fragment that could change underneath
    "{{supplier.__class__}}",           # reaching into the object
    "{{ }}",
    "{{supplier",                       # never closed
])
def test_anything_that_is_not_a_blank_or_a_pinned_fragment_is_refused(bad):
    with pytest.raises(T.TemplateInvalid):
        T.parse(bad)


def test_literal_braces_can_be_written_and_json_examples_are_left_alone():
    text = 'Return {"price": 1} and write \\{{name}} literally.'
    assert T.render_text(text, [], {}, {}) == 'Return {"price": 1} and write {{name}} literally.'


# --------------------------------------------------------------------------
# On save
# --------------------------------------------------------------------------
def test_a_well_formed_template_has_no_problems():
    assert T.problems_on_save("Rank {{supplier}} over {{rounds}}.", [STR, NUM], _no_fragments) == []


def test_an_undeclared_blank_is_a_problem():
    said = T.problems_on_save("Rank {{supplier}} in {{region}}.", [STR], _no_fragments)
    assert said == ["blank 'region' is not declared"]


def test_a_declaration_nothing_uses_is_a_problem():
    said = T.problems_on_save("Rank {{supplier}}.", [STR, NUM], _no_fragments)
    assert said == ["'rounds' is declared but never used"]


@pytest.mark.parametrize("declaration,expected", [
    ({"name": "x", "type": "date"}, "unknown type"),
    ({"name": "x", "type": "enum"}, "needs a list of values"),
    ({"name": "X y", "type": "string"}, "not a usable name"),
    ({"type": "string"}, "not a usable name"),
])
def test_a_malformed_declaration_is_a_problem(declaration, expected):
    said = " | ".join(T.problems_on_save("{{x}}", [declaration], _no_fragments))
    assert expected in said


def test_declaring_a_name_twice_is_a_problem():
    said = T.problems_on_save("{{supplier}}", [STR, dict(STR)], _no_fragments)
    assert "declared more than once" in " ".join(said)


def test_a_fragment_that_does_not_exist_at_that_version_is_a_problem():
    said = T.problems_on_save("{{> brand-voice@3}}", [], _no_fragments)
    assert said == ["fragment 'brand-voice@3' does not exist"]


def test_a_fragments_own_blanks_must_be_declared_by_the_prompt_that_uses_it():
    def fragments(key, version):
        return {"template": "Write to {{supplier}} in a {{tone}} voice.",
                "variables": [STR, {"name": "tone", "type": "string"}]}

    said = T.problems_on_save("{{> brand-voice@3}} {{supplier}}", [STR], fragments)
    assert said == ["fragment 'brand-voice@3' needs 'tone', which is not declared"]


def test_a_fragment_cannot_pull_in_another_fragment():
    said = T.problems_on_save("{{> a@1}}", [], _no_fragments, is_fragment=True)
    assert "a fragment cannot use another fragment" in " ".join(said)


# --------------------------------------------------------------------------
# On render
# --------------------------------------------------------------------------
def test_values_fill_the_blanks_the_same_way_every_time():
    text = "{{supplier}} / {{rounds}} / {{final}} / {{region}} / {{deal.id}} / {{deal}}"
    values = {"supplier": "Acme", "rounds": 3, "final": True, "region": "EU",
              "deal": {"value": 10.5, "id": "D-1"}}
    out = T.render_text(text, [STR, NUM, BOOL, ENUM, OBJ], values, {})

    assert out == 'Acme / 3 / true / EU / D-1 / {"id":"D-1","value":10.5}'
    assert T.render_text(text, [STR, NUM, BOOL, ENUM, OBJ], dict(reversed(values.items())), {}) == out


def test_a_fragment_is_put_in_at_exactly_the_version_named():
    out = T.render_text("{{> voice@2}} Hello {{supplier}}.", [STR], {"supplier": "Acme"},
                        {("voice", 2): "Be brief with {{supplier}}."})
    assert out == "Be brief with Acme. Hello Acme."


def test_a_value_that_looks_like_a_blank_is_not_filled_in_again():
    """What a supplier wrote must never be read as part of the template."""
    out = T.render_text("{{supplier}} {{rounds}}", [STR, NUM],
                        {"supplier": "{{rounds}} {{> voice@1}}", "rounds": 3}, {})
    assert out == "{{rounds}} {{> voice@1}} 3"


def test_a_missing_value_is_refused():
    with pytest.raises(T.VarMissing) as e:
        T.render_text("{{supplier}} {{rounds}}", [STR, NUM], {"supplier": "Acme"}, {})
    assert e.value.code == "VAR_MISSING" and "rounds" in str(e.value)


def test_a_value_nobody_declared_is_refused():
    with pytest.raises(T.VarUndeclared) as e:
        T.render_text("{{supplier}}", [STR], {"supplier": "Acme", "extra": 1}, {})
    assert e.value.code == "VAR_UNDECLARED"


@pytest.mark.parametrize("declaration,value", [
    (STR, 3), (NUM, "3"), (NUM, True), (BOOL, "true"), (ENUM, "US"),
    (OBJ, "D-1"), (OBJ, {"value": 1}), (OBJ, {"id": 7}),
])
def test_a_value_of_the_wrong_type_is_refused(declaration, value):
    name = declaration["name"]
    with pytest.raises(T.VarType) as e:
        T.render_text("{{%s}}" % name, [declaration], {name: value}, {})
    assert e.value.code == "VAR_TYPE"


def test_an_optional_blank_left_out_renders_as_nothing():
    optional = {"name": "note", "type": "string", "required": False}
    assert T.render_text("Hello.{{note}}", [optional], {}, {}) == "Hello."


def test_a_fragment_that_is_not_supplied_is_refused():
    with pytest.raises(T.FragmentMissing) as e:
        T.render_text("{{> voice@2}}", [], {}, {})
    assert e.value.code == "FRAGMENT_MISSING"


def test_a_refusal_never_repeats_the_value_it_refused():
    with pytest.raises(T.VarType) as e:
        T.render_text("{{rounds}}", [NUM], {"rounds": "customer secret 4111"}, {})
    assert "4111" not in str(e.value)
