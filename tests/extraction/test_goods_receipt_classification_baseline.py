"""Classification must not move when the vocabulary gains a goods receipt.

`specs/2026-10-04-three-way-match-design.md` success criterion 5: *no document
that classifies correctly today classifies differently afterwards.* Task 3 of the
plan adds eleven aliases -- `delivery note`, `advice note`, `packing list`,
`packing slip`, `proof of delivery` and `pod` among them -- and those are words
that appear inside real invoices and orders.

**Read this before trusting the guard below, because the obvious version of it
checks nothing.** Measured on 2026-10-04 against the live vocabulary:

* none of the eleven proposed aliases is in the alias index -- there is no
  collision to protect against;
* each of the eleven resolves to ``None`` today, so nothing owns those words;
* injecting a ``doctype.goods_receipt`` moves **zero** of the 65 existing alias
  pages and **zero** of the five co-occurrence pages below.

The reason is `type_resolver`: aliases are matched in TITLE position, so
"goods supplied per delivery note DN-5521" in an invoice body was never able to
claim the page. A test that recorded those answers and asserted they had not
changed would therefore pass forever, whatever Task 3 does. See
`.superpowers/sdd/2026-10-04-three-way-match-plan/progress.md` for the ruling.

So the three guards here bite on three different changes, and each is named for
the one it catches:

1. :func:`test_no_alias_changes_what_it_classifies_as` -- the broad record. Red
   when any vocabulary edit changes what an existing alias resolves to.
2. :func:`test_every_proposed_alias_is_unowned` -- **the one that guards Task 3.**
   Red the moment a proposed alias is owned by ANY type other than
   ``doctype.goods_receipt``, which is the only route by which adding a goods
   receipt can break classification. Since Task 3 landed (2026-10-05) the
   eleven are owned by the goods receipt itself, and that is the one owner the
   assertion permits -- a second owner is still red.
3. the co-occurrence pages -- red if the resolver ever starts matching body text,
   at which point a receipt word inside an invoice *could* steal it and guard 2
   would not see it.

Regenerate the baseline deliberately, never to turn a red test green:

    PYTHONPATH="src:." ./venv/bin/python \
        -m tests.extraction.test_goods_receipt_classification_baseline
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, Optional

import pytest

from src.services.concepts.seed import DOCUMENT_TYPES
from src.services.concepts.vocabulary import ensure_vocabulary
from src.services.extraction.type_resolver import resolve_document_type

BASELINE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "specs"
    / "2026-10-04-classification-baseline.json"
)

#: Exactly the aliases Task 3 of the plan seeds. Kept here as literals rather than
#: imported from the seed, so that when Task 3 lands this list is what the seed is
#: checked AGAINST instead of a copy of itself.
PROPOSED_GOODS_RECEIPT_ALIASES = (
    "goods receipt", "goods received note", "grn", "delivery note",
    "despatch note", "dispatch note", "advice note", "packing list",
    "packing slip", "proof of delivery", "pod",
)

# A page whose only signal is the alias. Deliberately bare: another signal would
# muddy what the alias alone decides.
_ALIAS_PAGE = "{alias}\nReference 4500018832\nSupplier: Northwind Trading Ltd\n"

# Documents of one type that mention a word Task 3 gives to another. The type word
# sits on its own line, as a title block does: putting the number beside it
# ("TAX INVOICE INV-2024-0117") makes the segment a reference VALUE rather than a
# title -- `_is_reference_value`, one short token group carrying a digit and naming
# no type -- and the page then resolves to nothing, which no alias could steal from.
CO_OCCURRENCE = {
    "invoice_citing_a_delivery_note":
        "TAX INVOICE\nNumber INV-2024-0117\nAmount due GBP 4,210.00\n"
        "Payment terms 30 days\nGoods supplied per delivery note DN-5521\n",
    "invoice_with_a_packing_list_enclosed":
        "INVOICE\nNumber 90412\nBill to: Acme Holdings\nAmount due EUR 980.00\n"
        "Packing list enclosed for your records\n",
    "purchase_order_naming_proof_of_delivery":
        "PURCHASE ORDER\nNumber 4500018832\nShip to: Unit 7, Brentford\n"
        "Line items with quantities and a total\n"
        "Supplier must provide proof of delivery on despatch\n",
    "quote_mentioning_delivery_note_terms":
        "QUOTATION\nNumber Q-7781\nValid until 31 March 2026\n"
        "Prices exclude VAT\nA delivery note will accompany each consignment\n",
    "contract_referring_to_goods_receipt":
        "MASTER SERVICE AGREEMENT\nnumbered clauses\n"
        "Payment falls due on goods receipt by the Buyer\n",
}


def _resolved(full_text: str) -> Optional[str]:
    """The concept this page reads as, on its own evidence.

    ``declared_concept=None`` because an uploader's category label would override
    the page, and the page is what an alias can take.
    """
    return resolve_document_type(
        declared_concept=None, full_text=full_text
    ).evidence_concept


def current_answers() -> Dict[str, Optional[str]]:
    answers: Dict[str, Optional[str]] = {}
    for code, dt in sorted(DOCUMENT_TYPES.items()):
        for alias in dt.aliases:
            answers[f"alias::{code}::{alias}"] = _resolved(
                _ALIAS_PAGE.format(alias=alias)
            )
    for name, page in sorted(CO_OCCURRENCE.items()):
        answers[f"page::{name}"] = _resolved(page)
    return answers


def test_the_baseline_file_exists():
    assert BASELINE.exists(), (
        "no baseline recorded; run PYTHONPATH='src:.' ./venv/bin/python -m "
        "tests.extraction.test_goods_receipt_classification_baseline"
    )


def test_no_alias_changes_what_it_classifies_as():
    """Guard 1, broad. Red when a vocabulary edit MOVES or REMOVES a recorded page.

    An ADDITION is not a move. Criterion 5 is *no document that classifies
    correctly today classifies differently afterwards*, and a page for an alias
    that did not exist when the baseline was taken classified as nothing today --
    it cannot have moved. Comparing the union of the two key sets made every new
    alias look like a move, which would have left exactly one way to go green:
    regenerate the baseline, i.e. erase the record this file exists to keep.
    So the comparison is over the RECORDED keys, and a recorded key that
    vanishes is still red.
    """
    recorded = json.loads(BASELINE.read_text())
    current = current_answers()
    moved = {
        k: (was, current.get(k, "<<GONE>>"))
        for k, was in recorded.items()
        if k not in current or current[k] != was
    }
    assert not moved, "these recorded pages changed type:\n" + "\n".join(
        f"  {k}: {was!r} -> {now!r}" for k, (was, now) in sorted(moved.items())
    )


@pytest.mark.parametrize("alias", PROPOSED_GOODS_RECEIPT_ALIASES)
def test_every_proposed_alias_is_unowned(alias):
    """Guard 2 -- the one that actually guards Task 3.

    A goods receipt can only break classification by claiming a word another type
    already owns. `pod` is the one to watch: three letters, and a plausible token
    elsewhere. Red here means Task 3 must drop or narrow that alias, not that the
    baseline needs regenerating.
    """
    owners = tuple(ensure_vocabulary().alias_index.get(alias, ()))
    others = tuple(o for o in owners if o != "doctype.goods_receipt")
    assert others == (), (
        f"{alias!r} is owned by {others}; a goods receipt sharing it would make "
        f"the page ambiguous, and type_resolver can only answer UNRESOLVED"
    )


@pytest.mark.parametrize("name", sorted(CO_OCCURRENCE))
def test_a_receipt_word_in_the_body_does_not_claim_the_page(name):
    """Guard 3. Not a Task 3 guard -- it is currently guaranteed by the resolver
    matching titles only. It bites if that ever changes, which is exactly when
    guard 2 would stop being sufficient.
    """
    recorded = json.loads(BASELINE.read_text())
    key = f"page::{name}"
    assert current_answers()[key] == recorded[key]


if __name__ == "__main__":  # regeneration entry point
    answers = current_answers()
    BASELINE.write_text(json.dumps(answers, indent=1, sort_keys=True) + "\n")
    print(f"recorded {len(answers)} pages to {BASELINE}")
