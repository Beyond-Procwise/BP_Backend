"""The supplier-research grounding guard must verify CONTENT, not just the host.

The guard used to ask one question: was the citation's hostname among the URLs the tools
returned? That is a citation-plausibility filter, not a grounding check, and three things
walked straight through it:

  1. A page that was never actually read. `seen` was populated from every web_search RESULT,
     so a host the model only saw in a result list counted as grounded. Observed live: an
     opencorporates.com fetch returned 403 and yielded empty text, yet three fields cited it
     and all three were kept. Those values came from model priors, not the web.
  2. Any path on a seen host. `opencorporates.com/totally/made/up/path` grounded a fact.
  3. Arbitrary prose. A fabricated sentence ("fined $4bn for fraud and under criminal
     investigation") was kept at 0.99 confidence, because nothing compared the claim to the
     page text.

The guard now keeps the text the model actually saw per URL and requires the claimed value to
appear in the text of the SPECIFIC url cited. These tests pin that, in both directions: the
false-positive direction (fabrications must be dropped) matters most, but a guard that drops
everything is useless, so the true-positive cases are pinned too.
"""

import pytest

from src.services.supplier_enrichment import research as R


ACME_ABOUT = (
    "About Acme Widgets Ltd. Acme Widgets Ltd is a private limited company "
    "registered in the United Kingdom. Our head office is in Manchester and we "
    "supply industrial fasteners across Europe. Visit acme.example for details."
)


def _ev(**pages):
    """Evidence map: url -> the text the model was shown for that url."""
    return dict(pages)


# --------------------------------------------------------------------------------------
# The three live breaks
# --------------------------------------------------------------------------------------

def test_citation_to_a_page_that_returned_nothing_is_dropped():
    """A 403/empty fetch grounds nothing, even though the URL was genuinely visited.

    This is break #1 and the most dangerous: the URL is real, the host is real, the model
    tried to read it — and the value is pure invention because there was no text.
    """
    evidence = _ev(**{
        "https://opencorporates.com/companies/gb/123": "",   # fetched, 403 -> empty
        "https://acme.example/about": ACME_ABOUT,
    })
    fields = {
        "city": {"value": "Manchester",
                 "source_url": "https://opencorporates.com/companies/gb/123",
                 "confidence": 0.99},
    }
    assert R._ground(fields, evidence) == {}


def test_made_up_path_on_a_visited_host_is_dropped():
    """Break #2: host-level trust let any path on that host through."""
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {
        "city": {"value": "Manchester",
                 "source_url": "https://acme.example/totally/made/up/path",
                 "confidence": 0.99},
    }
    assert R._ground(fields, evidence) == {}


def test_fabricated_claim_not_present_in_the_cited_page_is_dropped():
    """Break #3: the defamatory-sentence case, and the plain wrong-value case."""
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {
        # Not in the page at all. The page says Manchester.
        "city": {"value": "Atlantis", "source_url": "https://acme.example/about",
                 "confidence": 0.99},
        "country": {"value": "Brazil", "source_url": "https://acme.example/about",
                    "confidence": 0.99},
    }
    assert R._ground(fields, evidence) == {}


def test_uncited_field_is_still_dropped():
    """The original contract has to survive the change."""
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {"city": {"value": "Manchester", "source_url": "", "confidence": 0.9}}
    assert R._ground(fields, evidence) == {}


# --------------------------------------------------------------------------------------
# True positives — a guard that drops everything is not a guard, it is an outage
# --------------------------------------------------------------------------------------

def test_value_present_in_the_cited_page_is_kept():
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {
        "city": {"value": "Manchester", "source_url": "https://acme.example/about",
                 "confidence": 0.9},
        "country": {"value": "United Kingdom", "source_url": "https://acme.example/about",
                    "confidence": 0.8},
    }
    kept = R._ground(fields, evidence)
    assert set(kept) == {"city", "country"}


def test_matching_tolerates_case_spacing_and_punctuation():
    """Format tolerance, NOT semantic tolerance.

    "private limited company" appears in the page; the model reports it capitalised and
    parenthesised. That is the same fact written differently and must survive. What must
    NOT survive is a value that means something the page does not say - see the alias test.
    """
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {
        "legal_structure": {"value": "  Private   Limited Company  ",
                            "source_url": "https://acme.example/about", "confidence": 0.9},
    }
    assert set(R._ground(fields, evidence)) == {"legal_structure"}


def test_search_snippets_can_ground_because_the_model_did_read_them():
    """Snippets are real evidence: web_search returns title+snippet to the model.

    Excluding them would drop facts the model legitimately read. They are weak evidence,
    but they are evidence — unlike an empty fetch, which is none.
    """
    evidence = _ev(**{
        "https://dir.example/acme": "Acme Widgets Ltd — industrial fasteners, Manchester UK",
    })
    fields = {"city": {"value": "Manchester", "source_url": "https://dir.example/acme",
                       "confidence": 0.7}}
    assert set(R._ground(fields, evidence)) == {"city"}


# --------------------------------------------------------------------------------------
# Deliberate non-goals, pinned so nobody "fixes" them into holes
# --------------------------------------------------------------------------------------

def test_an_alias_the_page_does_not_use_is_dropped_not_guessed():
    """The page says "United Kingdom"; the model reports "UK".

    We do NOT expand aliases. The field drops to pending human review, which loses nothing —
    a reviewer sees it. Every alias table is a place for a wrong equivalence to hide, and a
    false positive here writes a wrong fact into a supplier record.
    """
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {"country": {"value": "UK", "source_url": "https://acme.example/about",
                          "confidence": 0.95}}
    assert R._ground(fields, evidence) == {}


def test_business_summary_is_never_treated_as_grounded_prose():
    """Generated prose cannot be substring-verified, so it is not verified at all.

    A summary is a paraphrase by construction — requiring it verbatim always fails, and
    accepting token overlap is exactly the digit-hole that lets a fabricated clause pass.
    So it is carried for the human reviewer and explicitly marked unverified; it has no
    bp_supplier column, so it can never be auto-applied either way.
    """
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {
        "business_summary": {
            "value": "Acme was fined $4bn for fraud and is under criminal investigation.",
            "source_url": "https://acme.example/about", "confidence": 0.99},
    }
    kept = R._ground(fields, evidence)
    assert "business_summary" in kept
    assert kept["business_summary"]["verified"] is False
    assert kept["business_summary"]["confidence"] == 0.0, (
        "unverified prose must not carry the model's self-reported confidence"
    )


def test_verified_fields_are_marked_verified():
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {"city": {"value": "Manchester", "source_url": "https://acme.example/about",
                       "confidence": 0.9}}
    assert R._ground(fields, evidence)["city"]["verified"] is True


def test_unverified_prose_is_never_auto_applied():
    """Belt and braces: even if a column were added for it later."""
    assert "business_summary" not in R._APPLY_COLUMNS


# --------------------------------------------------------------------------------------
# website_url is a URL, not page prose — it grounds on the host
# --------------------------------------------------------------------------------------

def test_website_url_grounds_when_its_host_is_the_page_it_was_read_from():
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {"website_url": {"value": "https://acme.example",
                              "source_url": "https://acme.example/about", "confidence": 0.9}}
    assert set(R._ground(fields, evidence)) == {"website_url"}


def test_website_url_grounds_when_the_cited_page_names_it():
    """A directory page that lists the company's site is legitimate evidence."""
    evidence = _ev(**{"https://dir.example/acme": "Acme Widgets Ltd. Website: acme.example"})
    fields = {"website_url": {"value": "https://acme.example",
                              "source_url": "https://dir.example/acme", "confidence": 0.9}}
    assert set(R._ground(fields, evidence)) == {"website_url"}


def test_website_url_for_an_unrelated_domain_is_dropped():
    """The hijack case: a real page cited for a website it never mentions."""
    evidence = _ev(**{"https://dir.example/acme": "Acme Widgets Ltd. Website: acme.example"})
    fields = {"website_url": {"value": "https://acme-widgets-official.example",
                              "source_url": "https://dir.example/acme", "confidence": 0.99}}
    assert R._ground(fields, evidence) == {}


# --------------------------------------------------------------------------------------
# URL normalisation — the model rarely echoes a URL byte-for-byte
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("cited", [
    "https://acme.example/about/",          # trailing slash
    "https://acme.example/about#team",      # fragment
    "HTTPS://ACME.EXAMPLE/about",           # scheme/host case
    "http://acme.example/about",            # scheme swap
])
def test_url_variants_still_resolve_to_the_same_evidence(cited):
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {"city": {"value": "Manchester", "source_url": cited, "confidence": 0.9}}
    assert set(R._ground(fields, evidence)) == {"city"}, f"{cited} should resolve"


def test_a_different_path_is_not_a_url_variant():
    """Normalisation must not become host-matching by the back door."""
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {"city": {"value": "Manchester",
                       "source_url": "https://acme.example/contact", "confidence": 0.9}}
    assert R._ground(fields, evidence) == {}


# --------------------------------------------------------------------------------------
# Degenerate values
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["unknown", "Unknown", "UNKNOWN", "n/a", "N/A", "none", "", "  "])
def test_sentinels_are_dropped(value):
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {"city": {"value": value, "source_url": "https://acme.example/about",
                       "confidence": 0.9}}
    assert R._ground(fields, evidence) == {}


def test_single_character_value_is_dropped_as_unverifiable():
    """A 1-char value substring-matches almost any page; the match would be meaningless."""
    evidence = _ev(**{"https://acme.example/about": ACME_ABOUT})
    fields = {"country": {"value": "U", "source_url": "https://acme.example/about",
                          "confidence": 0.9}}
    assert R._ground(fields, evidence) == {}


def test_sensitive_fields_are_dropped_even_when_present_in_the_page():
    """Presence in the source is not the test for these — they are never researched."""
    evidence = _ev(**{"https://acme.example/vat": "Acme Widgets Ltd VAT number GB123456789"})
    fields = {"vat_number": {"value": "GB123456789", "source_url": "https://acme.example/vat",
                             "confidence": 0.99}}
    assert R._ground(fields, evidence) == {}
