"""A side letter, a guarantee and a renewal are read as themselves.

Found by the live run of 2026-10-11, the day the four title-only contract types
were switched on (deploy/sql/2026-10-11_contract_types_activate.sql). All three
were LABELLED correctly and then:

  * the side letter and the guarantee printed "Side Letter No. SL-2026-0302" and
    "Guarantee No. PCG-2026-0304", which no contract_id pattern knew. The only
    number the reader recognised was the framework's, so contract_id read
    FA-2026-0077, dispatch's self-parent guard cleared it, and both documents
    were held as missing_required. Safe, but every real one would be held;
  * neither got a title: their headings name no word in _TITLE_WORDS;
  * the renewal said "renews Service Agreement No. SA-2026-0310" and no
    parent_agreement_ref pattern knew "renews", so it named no parent and no
    link could be proposed.

The pages below are the parser's own text for those uploads (bp_contract_raw
1125, 1126, 1127 on bp_testdb), not retyped approximations.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_new_contract_type_reading.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.engineered.contract_header import read_header   # noqa: E402
from src.services.extraction.pattern_extractor import run_pattern_extractor  # noqa: E402

_PARTIES = ("Buyer: BrightWave Digital Ltd., 123 Innovation Park, London, UK "
            "Supplier: NexaSpark Marketing Ltd., 125 Innovation Park, London, UK ")

_SIDE_LETTER = (
    "## SIDE LETTER\n\nSide Letter No. SL-2026-0302 " + _PARTIES +
    "This Side Letter relates to and varies Framework Agreement No. FA-2026-0077 "
    "dated 5 January 2026. Effective Date: 1 May 2026 End Date: 4 January 2029\n\n"
    "1. WAIVER\n\nThe Buyer waives the minimum annual commitment in clause 6.2 of "
    "Framework Agreement No. FA-2026-0077 for the first contract year.\n"
)
_GUARANTEE = (
    "## PARENT COMPANY GUARANTEE\n\nGuarantee No. PCG-2026-0304 " + _PARTIES +
    "Guarantor: NexaSpark Holdings plc, 1 Holdings Square, London, UK This Guarantee "
    "is given in respect of Framework Agreement No. FA-2026-0077 dated 5 January 2026. "
    "Effective Date: 5 January 2026 End Date: 4 January 2029\n\n1. GUARANTEE\n\n"
    "The Guarantor guarantees to the Buyer the due performance by the Supplier of its "
    "obligations under Framework Agreement No. FA-2026-0077.\n"
)
_RENEWAL = (
    "## RENEWAL AGREEMENT\n\nRenewal Agreement No. RN-2026-0303 " + _PARTIES +
    "This Renewal Agreement renews Service Agreement No. SA-2026-0310 dated 1 March "
    "2026. Effective Date: 1 March 2027 End Date: 29 February 2028\n\n"
    "## 1. RENEWAL TERM\n\nService Agreement No. SA-2026-0310 is renewed for a further "
    "term of 12 months on its existing terms.\n"
)


class _Parsed:
    def __init__(self, text: str) -> None:
        self.full_text = text


def _value(text, field):
    """The highest-prior value the pattern layer reads for one field, or None."""
    for c in run_pattern_extractor(_Parsed(text), "contract"):
        if c.field == field:
            return c.value
    return None


# --- the document's own number -------------------------------------------------

def test_a_side_letter_reads_its_own_number_not_its_frameworks():
    assert _value(_SIDE_LETTER, "contract_id") == "SL-2026-0302"
    assert _value(_SIDE_LETTER, "framework_ref") == "FA-2026-0077"


def test_a_guarantee_reads_its_own_number_not_its_frameworks():
    assert _value(_GUARANTEE, "contract_id") == "PCG-2026-0304"
    assert _value(_GUARANTEE, "framework_ref") == "FA-2026-0077"


@pytest.mark.parametrize("caption, ref", [
    ("Side Letter No. SL-7", "SL-7"),
    ("Side Letter Number: SL-7", "SL-7"),
    ("Side Letter Ref SL-7", "SL-7"),
    ("Guarantee No. PCG-9", "PCG-9"),
    ("Guaranty No. PCG-9", "PCG-9"),
    ("Parent Company Guarantee No. PCG-9", "PCG-9"),
    ("Deed of Guarantee Number: PCG-9", "PCG-9"),
    ("| Guarantee No. | PCG-9 |", "PCG-9"),
])
def test_each_caption_shape_is_read(caption, ref):
    page = f"## HEADING\n\n{caption}\n\nIn respect of the services.\n"
    assert _value(page, "contract_id") == ref, caption


@pytest.mark.parametrize("citing", [
    "The Supplier's obligations are secured by Guarantee No. PCG-9 dated 1 May 2026.",
    "This Agreement is varied by Side Letter No. SL-7 and by nothing else.",
    "See the Parent Company Guarantee No. PCG-9 for the Guarantor's undertaking.",
])
def test_an_agreement_citing_one_keeps_its_own_number(citing):
    """A caption begins its line; a citation sits inside a sentence. Without that
    distinction a master agreement would take its guarantee's number as its own,
    and contract_id is the upsert key for proc.bp_contracts."""
    page = ("## MASTER SERVICES AGREEMENT\n\nAgreement No. MSA-4417 " + _PARTIES +
            citing + "\n")
    assert _value(page, "contract_id") == "MSA-4417", citing


@pytest.mark.parametrize("citing", [
    "The Supplier's obligations are secured by Guarantee No. PCG-9 dated 1 May 2026.",
    "This Agreement is varied by Side Letter No. SL-7 and by nothing else.",
])
def test_an_agreement_with_no_number_does_not_borrow_the_one_it_cites(citing):
    page = "## MASTER SERVICES AGREEMENT\n\n" + _PARTIES + citing + "\n"
    assert _value(page, "contract_id") is None, citing


@pytest.mark.parametrize("placeholder", ["N/A", "TBC", "NONE"])
def test_a_placeholder_is_not_a_number(placeholder):
    assert _value(f"Side Letter No. {placeholder}\n", "contract_id") is None
    assert _value(f"Guarantee No. {placeholder}\n", "contract_id") is None


# --- the agreement a renewal renews --------------------------------------------

def test_a_renewal_names_the_agreement_it_renews():
    assert _value(_RENEWAL, "parent_agreement_ref") == "SA-2026-0310"
    assert _value(_RENEWAL, "contract_id") == "RN-2026-0303"


@pytest.mark.parametrize("sentence", [
    "This Renewal Agreement renews Service Agreement No. SA-1.",
    "This Renewal Agreement renews the Master Services Agreement Number MSA-1.",
    "This is a renewal of Contract No. C-881.",
    "This is a renewal of the Agreement C-881.",
])
def test_each_renewal_phrasing_is_read(sentence):
    assert _value(sentence, "parent_agreement_ref") in ("SA-1", "MSA-1", "C-881"), sentence


@pytest.mark.parametrize("sentence", [
    "This Renewal Agreement renews the Agreement on its existing terms.",
    "This Renewal Agreement renews Service Agreement dated 1 March 2026.",
    "The Supplier renews its insurance each year under policy POL-99.",
    "On renewal of this Agreement the charges increase by CPI.",
])
def test_renewal_prose_naming_no_number_names_no_parent(sentence):
    assert _value(sentence, "parent_agreement_ref") is None, sentence


# --- the title -----------------------------------------------------------------

@pytest.mark.parametrize("page, title", [
    (_SIDE_LETTER, "Side Letter"),
    (_GUARANTEE, "Parent Company Guarantee"),
    (_RENEWAL, "Renewal Agreement"),
    ("## DEED OF GUARANTY\n\nGuaranty No. PCG-9\n", "Deed Of Guaranty"),
])
def test_the_heading_is_the_title(page, title):
    assert read_header(page).title == title


def test_a_numbered_guarantee_clause_is_still_not_a_title():
    page = ("## PARTIES\n\nTHIS AGREEMENT is made on 1 June 2026.\n\n"
            "## 3. NO GUARANTEE\n\nThe Supplier does not guarantee uninterrupted service.\n")
    assert read_header(page).title is None
