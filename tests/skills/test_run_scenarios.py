"""The scenario runner's pure helpers; the live loop itself needs a model and is not tested here."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_scenarios", ROOT / "scripts" / "skills" / "run_scenarios.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def test_swaps_apply_in_one_pass_so_a_replacement_is_never_replaced_again():
    swaps = [("Borealis", "Kestrel Supplies 8"), ("Kestrel", "Kestrel Supplies 8")]
    assert rs._swap("Borealis and Kestrel both quoted", swaps) == "Kestrel Supplies 8 and Kestrel Supplies 8 both quoted"


def test_the_longest_name_wins_when_one_swap_is_a_prefix_of_another():
    swaps = [("Acme", "Falconridge Systems 15"), ("Acme Components Ltd", "Falconridge Systems 15")]
    assert rs._swap("Acme Components Ltd sent a quote", swaps) == "Falconridge Systems 15 sent a quote"


def test_no_swaps_leaves_the_prompt_alone():
    assert rs._swap("Is Acme safe?", []) == "Is Acme safe?"
