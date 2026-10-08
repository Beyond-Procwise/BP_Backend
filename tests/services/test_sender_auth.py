"""Who an inbound reply is really from: authentication results the RECEIVER stamped, and whether the sender's domain is the supplier's.

Two traps shape this. (1) A fraudster can put a forged `Authentication-Results: ...; dmarc=pass` line inside the message, so only results
stamped by an authserv-id we trust count, and where several trusted lines disagree the WORST wins. (2) A lookalike domain passes DMARC
perfectly well, so passing authentication is not the same as being the supplier: the From domain is also compared with the domains on
the supplier master.
"""

import pytest

from src.services.draft_assurance import sender_auth as SA

TRUSTED = ["amazonses.com"]
SES_OK = ("amazonses.com; spf=pass (spfCheck: domain of acme.test designates 1.2.3.4 as permitted sender) smtp.mailfrom=alex@acme.test; "
          "dkim=pass header.i=@acme.test; dmarc=pass header.from=acme.test;")


def hdrs(*auth, frm="Alex Morgan <alex@acme.test>", **extra):
    h = {"From": (frm,), "Authentication-Results": tuple(auth)}
    h.update(extra)
    return h


# --- reading the results --------------------------------------------------------------------------------------------------------

def test_a_trusted_header_is_read_for_all_three_methods():
    r = SA.parse_results(hdrs(SES_OK), TRUSTED)
    assert (r["spf"], r["dkim"], r["dmarc"]) == ("pass", "pass", "pass") and r["trusted_headers"] == 1 and r["ignored_headers"] == 0


def test_the_authserv_id_match_ignores_case_and_header_name_case():
    r = SA.parse_results({"authentication-results": ("AmazonSES.com; dmarc=pass",)}, TRUSTED)
    assert r["dmarc"] == "pass"


def test_a_header_from_a_server_we_do_not_trust_is_ignored_even_if_it_says_pass():
    r = SA.parse_results(hdrs("evil.example; spf=pass; dkim=pass; dmarc=pass"), TRUSTED)
    assert (r["spf"], r["dkim"], r["dmarc"]) == ("missing",) * 3 and r["ignored_headers"] == 1 and r["trusted_headers"] == 0


def test_where_trusted_headers_disagree_the_worst_result_wins():
    r = SA.parse_results(hdrs("amazonses.com; dmarc=pass; spf=pass", "amazonses.com; dmarc=fail; spf=pass"), TRUSTED)
    assert r["dmarc"] == "fail" and r["trusted_headers"] == 2


def test_a_folded_multi_line_header_is_read():
    folded = "amazonses.com;\r\n\tspf=pass smtp.mailfrom=a@acme.test;\r\n dkim=fail header.i=@acme.test;\r\n dmarc=pass"
    r = SA.parse_results(hdrs(folded), TRUSTED)
    assert (r["spf"], r["dkim"], r["dmarc"]) == ("pass", "fail", "pass")


def test_no_authentication_header_at_all_is_missing_not_pass():
    r = SA.parse_results({"From": ("a@acme.test",)}, TRUSTED)
    assert (r["spf"], r["dkim"], r["dmarc"]) == ("missing",) * 3 and r["trusted_headers"] == 0


@pytest.mark.parametrize("word,expected", [("pass", "pass"), ("fail", "fail"), ("hardfail", "fail"), ("softfail", "softfail"),
                                           ("neutral", "neutral"), ("none", "none"), ("temperror", "temperror"), ("permerror", "permerror"),
                                           ("bananas", "unknown"), ("PASS", "pass")])
def test_result_words_are_normalised_and_an_unknown_word_is_never_a_pass(word, expected):
    assert SA.parse_results(hdrs(f"amazonses.com; dmarc={word}"), TRUSTED)["dmarc"] == expected


def test_a_method_the_header_does_not_mention_is_missing():
    r = SA.parse_results(hdrs("amazonses.com; dmarc=pass"), TRUSTED)
    assert (r["spf"], r["dkim"]) == ("missing", "missing")


@pytest.mark.parametrize("bad", [None, 5, [], "text", {"Authentication-Results": None}, {"Authentication-Results": 5}, {"Authentication-Results": (None, 3)}])
def test_garbage_headers_never_raise_and_never_pass(bad):
    r = SA.parse_results(bad, TRUSTED)
    assert r["dmarc"] in ("missing", "unknown") and r["spf"] != "pass"


def test_an_empty_trusted_list_trusts_nothing():
    assert SA.parse_results(hdrs(SES_OK), [])["dmarc"] == "missing"


# --- the From domain and the supplier's domains ------------------------------------------------------------------------------------

@pytest.mark.parametrize("frm,expected", [("Alex <alex@Acme.TEST>", "acme.test"), ("alex@acme.test", "acme.test"), ('"A, B" <ab@sub.acme.test>', "sub.acme.test"),
                                          ("not an address", None), ("", None)])
def test_the_from_domain(frm, expected):
    assert SA.from_domain({"From": (frm,)}, None) == expected


def test_two_from_headers_are_ambiguous_and_give_no_domain():
    assert SA.from_domain({"From": ("a@acme.test", "b@evil.example")}, None) is None


def test_the_fallback_address_is_used_when_there_is_no_from_header():
    assert SA.from_domain({}, "Alex <alex@acme.test>") == "acme.test" and SA.from_domain(None, "alex@acme.test") == "acme.test"


@pytest.mark.parametrize("domain,known,expected", [
    ("acme.test", ["acme.test"], True), ("ACME.test", ["acme.TEST"], True), ("mail.acme.test", ["acme.test"], True),
    ("evilacme.test", ["acme.test"], False), ("acme.test.evil.com", ["acme.test"], False), ("other.test", ["acme.test", "x.test"], False),
    ("acme.test", [], None), (None, ["acme.test"], None), ("acme.test", None, None)])
def test_a_senders_domain_is_the_suppliers_or_a_subdomain_of_it_and_never_just_a_suffix_of_text(domain, known, expected):
    assert SA.domain_matches(domain, known) is expected


# --- the verdict ------------------------------------------------------------------------------------------------------------------

def res(spf="pass", dkim="pass", dmarc="pass"):
    return {"spf": spf, "dkim": dkim, "dmarc": dmarc}


RULES = {"mode": "enforce", "trusted_authserv_ids": TRUSTED, "hold_on_fail": True, "hold_on_missing": False, "hold_on_domain_mismatch": False}


def verdict(results, match=True, **rules):
    return SA.decide(results, match, {**RULES, **rules})


def test_a_dmarc_pass_is_authenticated_and_not_held():
    v = verdict(res(spf="fail", dkim="none", dmarc="pass"))
    assert v["verdict"] == "authenticated" and v["hold"] is False


def test_dkim_and_spf_both_passing_is_authenticated_when_dmarc_says_nothing():
    assert verdict(res(dmarc="missing"))["verdict"] == "authenticated"
    assert verdict(res(dkim="fail", dmarc="missing"))["verdict"] == "failed"


@pytest.mark.parametrize("override", [dict(spf="fail", dmarc="none"), dict(dkim="fail", dmarc="none"), dict(dmarc="fail"), dict(spf="fail", dkim="fail", dmarc="missing")])
def test_a_hard_failure_without_a_dmarc_pass_is_failed_and_held_by_default(override):
    v = verdict(res(**override))
    assert v["verdict"] == "failed" and v["hold"] is True and "auth_failed" in v["reasons"]


def test_one_method_failing_is_not_a_failure_when_dmarc_passed():
    """DMARC passes when SPF OR DKIM passes in alignment, so a failing SPF beside a DMARC pass is an ordinary forwarded mail."""
    assert verdict(res(spf="fail", dmarc="pass"))["verdict"] == "authenticated"
    assert verdict(res(dkim="fail", dmarc="pass"))["hold"] is False


def test_failed_is_not_held_when_that_switch_is_off():
    assert verdict(res(dmarc="fail"), hold_on_fail=False)["hold"] is False


def test_nothing_stamped_is_missing_and_held_only_if_that_switch_is_on():
    miss = res(spf="missing", dkim="missing", dmarc="missing")
    assert verdict(miss)["verdict"] == "missing" and verdict(miss)["hold"] is False
    on = verdict(miss, hold_on_missing=True)
    assert on["hold"] is True and "auth_missing" in on["reasons"]


def test_soft_results_are_inconclusive_and_follow_the_missing_switch():
    soft = res(spf="softfail", dkim="none", dmarc="none")
    assert verdict(soft)["verdict"] == "inconclusive" and verdict(soft)["hold"] is False
    assert verdict(soft, hold_on_missing=True)["hold"] is True


def test_a_domain_that_is_not_the_suppliers_is_held_only_if_that_switch_is_on():
    assert verdict(res(), match=False)["hold"] is False
    v = verdict(res(), match=False, hold_on_domain_mismatch=True)
    assert v["verdict"] == "authenticated" and v["hold"] is True and v["reasons"] == ["domain_mismatch"]


def test_an_unknown_match_is_never_a_reason_to_hold():
    assert verdict(res(), match=None, hold_on_domain_mismatch=True)["hold"] is False


def test_shadow_mode_records_the_verdict_and_holds_nothing():
    v = verdict(res(dmarc="fail"), match=False, mode="shadow", hold_on_domain_mismatch=True)
    assert v["verdict"] == "failed" and v["hold"] is False and "auth_failed" in v["reasons"]


def test_both_reasons_are_reported_together():
    v = verdict(res(dmarc="fail"), match=False, hold_on_domain_mismatch=True)
    assert v["hold"] is True and v["reasons"] == ["auth_failed", "domain_mismatch"]


# --- the config -------------------------------------------------------------------------------------------------------------------

def engine(rules):
    return type("E", (), {"get_policy": staticmethod(lambda slug: None if rules is None else {"details": {"rules": rules}, "version": 1})})()


def test_good_rules_are_read():
    assert SA.load_rules(engine(RULES)) == RULES | {"trusted_authserv_ids": ["amazonses.com"]}


@pytest.mark.parametrize("bad", [None, {}, {**RULES, "mode": "maybe"}, {**RULES, "trusted_authserv_ids": []}, {**RULES, "trusted_authserv_ids": "amazonses.com"},
                                 {**RULES, "trusted_authserv_ids": [5]}, {**RULES, "hold_on_fail": "yes"}, {**{k: v for k, v in RULES.items() if k != "hold_on_missing"}}])
def test_bad_or_missing_rules_are_refused(bad):
    with pytest.raises(SA.SenderAuthRulesUnavailable):
        SA.load_rules(engine(bad))


def test_no_policy_engine_is_refused():
    with pytest.raises(SA.SenderAuthRulesUnavailable):
        SA.load_rules(None)
