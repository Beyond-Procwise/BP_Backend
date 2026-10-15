from types import SimpleNamespace

from services import mail_rules

P = lambda **kw: SimpleNamespace(company_domains=("yourcompany.com",),
                                 **{"external_recipients": "known_suppliers_only", **kw})
sup = lambda a: a == "quotes@supplier.com"
LIM = 3_000_000


def test_display_name_and_case_are_normalised():
    assert mail_rules.bare_address(' Finance <FINANCE@YourCompany.com> ') == "finance@yourcompany.com"
    assert mail_rules.classify("finance@yourcompany.com", P(), is_supplier=sup) == "internal"
    assert mail_rules.classify(" Finance <FINANCE@YourCompany.com>", P(), is_supplier=sup) == "internal"


def test_classify_supplier_and_other():
    assert mail_rules.classify("Quotes <quotes@supplier.com>", P(), is_supplier=sup) == "supplier"
    assert mail_rules.classify("x@gmail.com", P(), is_supplier=sup) == "other"


def test_supplier_email_to_a_colleague_is_held():
    v = mail_rules.check_supplier_email(["quotes@supplier.com", "finance@yourcompany.com"], 10, P(),
                                        is_supplier=sup, size_limit=LIM)
    assert not v.allowed and "colleagues" in v.reason


def test_unknown_external_is_held_when_suppliers_only():
    v = mail_rules.check_supplier_email(["someone@gmail.com"], 10, P(), is_supplier=sup, size_limit=LIM)
    assert not v.allowed and v.reason == "someone@gmail.com is not a known supplier"


def test_any_external_allowed_when_policy_says_any():
    v = mail_rules.check_supplier_email(["someone@gmail.com"], 10, P(external_recipients="any"),
                                        is_supplier=sup, size_limit=LIM)
    assert v.allowed


def test_known_supplier_is_allowed():
    assert mail_rules.check_supplier_email(["quotes@supplier.com"], 10, P(), is_supplier=sup,
                                           size_limit=LIM).allowed


def test_oversize_is_held_with_the_limit():
    v = mail_rules.check_supplier_email(["quotes@supplier.com"], 3_000_001, P(), is_supplier=sup,
                                        size_limit=LIM)
    assert not v.allowed and "3 MB" in v.reason


import pytest


@pytest.mark.parametrize("external", ["known_suppliers_only", "any"])
@pytest.mark.parametrize("raw", ["a@x.com, b@y.com", "not an address", ""])
def test_an_unparseable_or_multi_address_is_refused_with_the_raw_text(external, raw):
    v = mail_rules.check_supplier_email([raw], 10, P(external_recipients=external), is_supplier=sup,
                                        size_limit=LIM)
    assert not v.allowed and v.reason == f"'{raw}' is not a single email address"
