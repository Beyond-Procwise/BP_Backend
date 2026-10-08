"""When the AI fallback's lines replace the table's, lines it dropped come back if the document's
total says they belong (Meridian V2, 2026-10-08: the fallback kept 7 of 9 lines, losing Overtime
£24,000 and Expenses £72,000, while the table extractor had read both amounts)."""
from decimal import Decimal

from src.services.extraction.completeness import restore_dropped_lines

AI = [{"item_description": "Programme Director", "line_total": 274312},
      {"item_description": "Consultant", "line_total": 1782778}]
# Table lines arrive coerced by persistence.build_line_items: money is a Decimal.
TABLE = [{"item_description": "Programme Director", "line_total": Decimal("274312")},
         {"item_description": "Overtime / out-of-hours", "line_total": Decimal("24000")},
         {"item_description": "Expenses (travel, accom., subsistence)", "line_total": Decimal("72000")},
         {"item_description": "Provisions subtotal", "line_total": Decimal("96000")}]


def test_lines_the_total_says_belong_come_back():
    lines, n = restore_dropped_lines("quote", AI, TABLE, 2153090)
    assert n == 2
    assert [l["item_description"] for l in lines][-2:] == [
        "Overtime / out-of-hours", "Expenses (travel, accom., subsistence)"]


def test_a_line_that_would_overshoot_the_total_stays_out():
    lines, _ = restore_dropped_lines("quote", AI, TABLE, 2153090)
    assert "Provisions subtotal" not in [l["item_description"] for l in lines]


def test_a_line_already_present_by_amount_is_not_added_twice():
    lines, _ = restore_dropped_lines("quote", AI, TABLE, 2153090)
    assert [float(l["line_total"]) for l in lines].count(274312.0) == 1


def test_nothing_changes_without_a_total_to_check_against():
    lines, n = restore_dropped_lines("quote", AI, TABLE, None)
    assert n == 0 and lines == AI


def test_nothing_changes_when_the_ai_lines_already_reconcile():
    lines, n = restore_dropped_lines("quote", AI, TABLE, 274312 + 1782778)
    assert n == 0 and lines == AI
