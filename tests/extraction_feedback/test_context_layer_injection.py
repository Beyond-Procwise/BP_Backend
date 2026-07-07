"""T4: advisory hint injection is additive and byte-identical when there are no hints."""
from src.services.extraction.context_layer import _build_prompt

_ARGS = (
    "invoice",
    "Acme Ltd\nInvoice INV-1\nSubtotal 100\nVAT @20% 20\nTotal 120",
    [("supplier_name", "text", "the supplier"), ("tax_amount", "money", "tax")],
    {},          # raw candidates
    None,        # filename hints
)


def test_no_hints_byte_identical():
    """With no vendor hints, the prompt is unchanged (None and [] both inert)."""
    base_none = _build_prompt(*_ARGS, vhints=None)
    base_empty = _build_prompt(*_ARGS, vhints=[])
    assert base_none == base_empty
    assert "VENDOR-SPECIFIC HINTS" not in base_none


def test_hint_injected_after_candidates_before_rules():
    base = _build_prompt(*_ARGS, vhints=None)
    withh = _build_prompt(*_ARGS, vhints=["capture VAT as tax_amount"])
    assert "VENDOR-SPECIFIC HINTS" in withh
    assert "capture VAT as tax_amount" in withh
    # placement: after CANDIDATE HINTS, before OUTPUT RULES
    assert withh.index("CANDIDATE HINTS") < withh.index("VENDOR-SPECIFIC HINTS") < withh.index("OUTPUT RULES:")
    # additive only: the injected text is the sole growth
    assert len(withh) > len(base)
