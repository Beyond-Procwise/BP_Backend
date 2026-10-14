"""The award-commitment pattern. Found live 2026-10-09: "If you agree, the contract is yours" passed every check.
The pattern under test is read from the migration that installs it, and the original from the family migration, so the
test cannot drift from what is deployed."""

import json
import re
from pathlib import Path

import pytest

SQL = Path(__file__).resolve().parents[2] / "deploy/sql"
NEW = re.findall(r"to_jsonb\(\$p\$(.*?)\$p\$::text\)", (SQL / "2026-10-09_email_award_commitment.sql").read_text(), re.S)[0]
OLD = json.loads((SQL / "2026-10-07_email_family_negotiation_counter.sql").read_text().split("$json$")[1])["rules"]["forbidden_patterns"]["award_commitment"]

COMMITMENTS = [
    "If you agree, the contract is yours.",
    "Match this and the business is yours.",
    "The order will be yours once you confirm.",
    "You've won the contract.",
    "You have secured the business for next year.",
    "The contract will be awarded to you.",
    "Consider the order placed.",
    "We'll award the work to you this week.",
    "We are awarding the contract to Acme.",
    "We will place the order today.",
    "We commit to a minimum volume of 500 units.",
    "We guarantee you the business.",
    "We will sign the contract on Friday.",
]
NOT_COMMITMENTS = [
    "We are still evaluating offers before any award.",
    "Before we can place the order, please confirm the price.",
    "Your contract is due for renewal on 31 December 2026.",
    "The contract is yours to review; please send comments by Friday.",
    "Please confirm whether you can deliver the order by Friday.",
    "We would like to discuss the volume for next year.",
]


@pytest.mark.parametrize("text", COMMITMENTS)
def test_a_commitment_to_award_is_caught(text):
    assert re.search(NEW, text, re.IGNORECASE)


@pytest.mark.parametrize("text", NOT_COMMITMENTS)
def test_ordinary_procurement_wording_is_not_a_commitment(text):
    assert not re.search(NEW, text, re.IGNORECASE)


def test_the_original_pattern_missed_most_of_them():
    missed = [t for t in COMMITMENTS if not re.search(OLD, t, re.IGNORECASE)]
    assert "If you agree, the contract is yours." in missed and len(missed) >= 8


def test_the_migration_only_replaces_the_original_pattern_and_the_rollback_only_the_new_one():
    up = (SQL / "2026-10-09_email_award_commitment.sql").read_text()
    down = (SQL / "2026-10-09_email_award_commitment_rollback.sql").read_text()
    assert f"= $p${OLD}$p$" in up and f"to_jsonb($p${NEW}$p$" in up
    assert f"= $p${NEW}$p$" in down and f"to_jsonb($p${OLD}$p$" in down
