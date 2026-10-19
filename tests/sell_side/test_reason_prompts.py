"""The wording of a sales reason comes from the prompt library, never from the code."""
import pytest

from src.services.sell_side import reason_prompts as rp


class _Cur:
    def __init__(self, row):
        self._row = row

    def execute(self, sql, params=None):
        self.sql, self.params = sql, params

    def fetchone(self):
        return self._row


def _conn(monkeypatch, desc):
    cur = _Cur({"prompts_desc": desc} if desc is not None else None)
    monkeypatch.setattr(rp, "dict_cursor", lambda conn: cur)
    return object(), cur


REL = {"prompt_template": "Bought {bought_sku}; it is {relation_type} {target_sku}.",
       "phrases": {"relation_type": {"replaced_by": "replaced by"}}}
FACTS = {"bought_sku": "A", "relation_type": "replaced_by", "target_sku": "B"}


def test_the_sentence_is_the_prompts_template_filled_from_the_facts(monkeypatch):
    conn, cur = _conn(monkeypatch, REL)
    assert rp.render(conn, rp.RELATION, FACTS) == "Bought A; it is replaced by B."
    assert cur.params == (rp.RELATION,)


def test_rewording_the_prompt_rewords_the_sentence_with_no_code_change(monkeypatch):
    conn, _ = _conn(monkeypatch, {**REL, "prompt_template": "{target_sku} supersedes {bought_sku}."})
    assert rp.render(conn, rp.RELATION, FACTS) == "B supersedes A."


def test_a_missing_prompt_raises_rather_than_inventing_a_sentence(monkeypatch):
    conn, _ = _conn(monkeypatch, None)
    with pytest.raises(rp.ReasonPromptError, match="no active prompt"):
        rp.render(conn, rp.RELATION, FACTS)
    conn, _ = _conn(monkeypatch, {"templates": []})        # a row without a prompt_template
    with pytest.raises(rp.ReasonPromptError):
        rp.render(conn, rp.RELATION, FACTS)


def test_a_coded_value_with_no_wording_and_no_default_raises(monkeypatch):
    conn, _ = _conn(monkeypatch, REL)
    with pytest.raises(rp.ReasonPromptError, match="no wording for relation_type='requires'"):
        rp.render(conn, rp.RELATION, {**FACTS, "relation_type": "requires"})


def test_a_default_covers_values_nobody_listed(monkeypatch):
    desc = {"prompt_template": "{sku} reached {lifecycle_status}.",
            "phrases": {"lifecycle_status": {"end_of_sale": "end of sale", "default": "end of life"}}}
    conn, _ = _conn(monkeypatch, desc)
    assert rp.render(conn, rp.END_OF_LIFE_DATED, {"sku": "A", "lifecycle_status": "end_of_sale"}) == "A reached end of sale."
    assert rp.render(conn, rp.END_OF_LIFE_DATED, {"sku": "A", "lifecycle_status": "anything"}) == "A reached end of life."


def test_a_field_the_detector_did_not_supply_raises(monkeypatch):
    conn, _ = _conn(monkeypatch, {"prompt_template": "{sku} {colour}"})
    with pytest.raises(rp.ReasonPromptError, match="colour"):
        rp.render(conn, rp.RELATION, {"sku": "A"})
