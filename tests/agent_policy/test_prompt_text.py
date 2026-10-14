"""The governed extraction prompts carry the wording table and offer no concrete example value."""
import json
import re
from pathlib import Path

import pytest

SQL = Path(__file__).resolve().parents[2] / "deploy/sql/2026-10-10_agent_policy_prompts_wording.sql"


def _templates():
    text = SQL.read_text()
    return {name: json.loads(body)["prompt_template"]
            for body, name in re.findall(r"\$p\$(.*?)\$p\$::jsonb[^;]*?prompt_name = '(\w+)'", text, re.S)}


@pytest.mark.parametrize("name", ["agent_policy_extract", "agent_policy_fix"])
def test_wording_table_and_no_currency_example(name):
    t = _templates()[name].lower()
    for phrase in ("above", "more than", "exceeds", "over", "at least", "or more", "not less than",
                   "below", "less than", "under", "at most", "up to", "not more than", "no more than"):
        assert phrase in t
    for pair in ("gt", "gte", "lt", "lte"):
        assert pair in t
    assert not re.search(r"[$£€]\s?\d", t)


def test_extract_prompt_roles_for_deciders_notify_owner():
    t = _templates()["agent_policy_extract"]
    assert "deciders" in t and "owner" in t and "accountable" in t
