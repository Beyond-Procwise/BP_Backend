from pathlib import Path

SQL = Path("deploy/sql/2026-06-17_requirements_governance.sql").read_text().lower()


def test_prompt_seed_present():
    assert "insert into proc.bp_prompt" in SQL
    assert "requirements_elicitation" in SQL
    assert "prompt_template" in SQL          # template stored in prompts_desc jsonb
    assert "where not exists" in SQL         # idempotent (no unique constraint)
    assert "requirements_agent" in SQL       # linked agent slug


def test_policy_seed_present():
    assert "insert into proc.bp_policy" in SQL
    assert "requirement_required_fields" in SQL
    assert "policy_details" in SQL
    for f in ("title", "category", "quantity", "needed_by_date", "delivery_location"):
        assert f in SQL
