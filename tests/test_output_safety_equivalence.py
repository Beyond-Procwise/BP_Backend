"""The name check was rewritten for speed; it must catch exactly what it caught.

``_named_in`` replaced one regex per name per string (about 260 of them) with a split of the
text into word runs and a set intersection. That is a change to a security gate, made for a
performance reason, so it is held to the only standard that makes it safe: for every string and
every vocabulary tried here it returns the SAME set of names as the original loop, which is
reproduced below, verbatim, as the reference.
"""
from __future__ import annotations

import random
import re
import string

import pytest

from services import output_safety as osafe


def reference(text: str, names) -> set:
    """The original behaviour: one \\bname\\b search per name."""
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b", text)}


TABLES = {"process_monitor", "cat_product_mapping", "bp_invoice_trgt", "bp_invoice", "bp_invoice_line_items",
          "bp_deal_overview", "supplier_master", "a_b", "x1_y2"}
ENV = {"DB_HOST", "DB_HOST_NAME", "S3_BUCKET_NAME", "AWS_REGION", "PORT_1"}
# names that are NOT pure word characters exercise the fallback path (none exist in production)
ODD = {"a-b", "x.y", "weird name"}


def cases(names):
    out = []
    for n in names:
        for pre in ("", "x", "_", "-", ".", " ", "(", "9", "é", "/"):
            for post in ("", "x", "_", "-", ".", " ", ")", "9", "é", ",", "s"):
                out.append(f"{pre}{n}{post}")
                out.append(f"see {pre}{n}{post} and {n.upper()} {n.lower()}")
    rnd = random.Random(7)
    alphabet = string.ascii_letters + string.digits + "_ -./,()é"
    pool = list(names)
    for _ in range(4000):
        words = [rnd.choice(pool) if rnd.random() < 0.3 else "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 12)))
                 for _ in range(rnd.randint(1, 6))]
        out.append(rnd.choice([" ", "", "_", "-", "."]).join(words))
    return out


@pytest.mark.parametrize("vocab", [TABLES, ENV, TABLES | ENV, TABLES | ODD, ENV | ODD, set()],
                         ids=["tables", "env", "both", "tables+odd", "env+odd", "empty"])
def test_finds_exactly_the_names_the_original_loop_found(vocab):
    for text in cases(vocab | {"plain"}):
        for variant in (text, text.lower()):
            assert set(osafe._named_in(variant, vocab)) == reference(variant, vocab), repr(variant)


def test_a_name_inside_a_longer_word_is_not_a_hit_but_a_whole_word_is():
    v = {"bp_invoice"}
    assert osafe._named_in("bp_invoice", v) == ["bp_invoice"]
    assert osafe._named_in("the bp_invoice table", v) == ["bp_invoice"]
    assert osafe._named_in("bp_invoice_trgt", v) == []        # a longer name, not this one
    assert osafe._named_in("xbp_invoice", v) == []
    assert osafe._named_in("bp_invoice9", v) == []


def test_the_whole_gate_gives_the_same_verdict_with_the_new_check(monkeypatch):
    monkeypatch.setattr(osafe, "_db_tables", set(TABLES))
    monkeypatch.setattr(osafe, "_env_keys", set(ENV))
    for text in ("Spend is read from process_monitor", "set DB_HOST first", "an ordinary sentence about invoices",
                 "the bp_invoice_line_items rows", "S3_BUCKET_NAME and AWS_REGION"):
        kinds = {v.kind for v in osafe.inspect(text)}
        ref_kinds = ({"db_table"} if reference(text.lower(), TABLES) else set()) | ({"env_var"} if reference(text, ENV) else set())
        assert ref_kinds <= kinds, text


def test_it_is_fast_enough_that_a_thousand_row_answer_does_not_stall(monkeypatch):
    import time
    names = {f"table_name_{i}" for i in range(200)}
    keys = {f"CONFIG_KEY_{i}" for i in range(60)}
    monkeypatch.setattr(osafe, "_db_tables", names)
    monkeypatch.setattr(osafe, "_env_keys", keys)
    t = time.perf_counter()
    for i in range(8000):
        osafe.inspect(f"Contract C{i:05d} Hosting 2026-10-{i % 28 + 1:02d} 24 months", prose=False)
    assert time.perf_counter() - t < 8.0    # was ~14s before; a loose bound, not a benchmark
