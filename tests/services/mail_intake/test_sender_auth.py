"""Sender auth is read from the receiving MTA's verdict, not recomputed.

SPF checks the connecting IP, which is gone by the time a forwarded message
reaches us — recomputing it would produce an authoritative-looking number that
means nothing. What we can do honestly is record the verdict AND who asserted
it, and never treat an absent verdict as a pass.
"""
from email.message import EmailMessage

from src.services.mail_intake import sender_auth
from src.services.mail_intake.mime_walk import parse


def _bundle(auth_results=None, sender="quotes@copperleaf.test"):
    msg = EmailMessage()
    msg["From"] = f"Sales <{sender}>"
    msg["To"] = "intake@acme.procureiq.io"
    msg["Subject"] = "Quote"
    msg["Return-Path"] = f"<{sender}>"
    if auth_results:
        msg["Authentication-Results"] = auth_results
    msg.set_content("body")
    return parse(msg.as_bytes())


def test_all_pass_is_recorded_with_its_asserter():
    result = sender_auth.evaluate(
        _bundle(
            "mx.acme.test; spf=pass smtp.mailfrom=copperleaf.test; "
            "dkim=pass header.d=copperleaf.test; dmarc=pass header.from=copperleaf.test"
        ),
        trusted_asserter="mx.acme.test",
    )
    assert (result.spf, result.dkim, result.dmarc) == ("pass", "pass", "pass")
    assert result.asserted_by == "mx.acme.test"
    assert result.alignment == "aligned"


def test_no_configured_asserter_means_no_verdict_is_trusted():
    """The safe default, and the one an earlier version had backwards.

    Omitting `trusted_asserter` used to mean "trust whatever header arrived",
    so any caller that forgot it accepted forged passes. Anything upstream of
    the machine that wrote the topmost header can forge one, so not knowing who
    wrote it must read as `unknown`, never as `pass`.
    """
    result = sender_auth.evaluate(
        _bundle("anyone-at-all.test; spf=pass; dkim=pass; dmarc=pass")
    )
    assert (result.spf, result.dkim, result.dmarc) == (
        sender_auth.UNKNOWN,
        sender_auth.UNKNOWN,
        sender_auth.UNKNOWN,
    )
    # The claim is still recorded — we just do not believe it.
    assert result.asserted_by == "anyone-at-all.test"


def test_a_comment_in_the_authserv_id_does_not_distrust_a_real_mta():
    """Real MTAs write `mx.acme.test (Postfix)`. Comparing the whole string
    would downgrade a legitimate verdict for a cosmetic reason."""
    result = sender_auth.evaluate(
        _bundle("mx.acme.test (Postfix); spf=pass; dkim=pass; dmarc=pass"),
        trusted_asserter="mx.acme.test",
    )
    assert result.spf == "pass"


def test_spf_failure_is_recorded_not_raised():
    result = sender_auth.evaluate(
        _bundle("mx.acme.test; spf=fail smtp.mailfrom=copperleaf.test; dkim=pass"),
        trusted_asserter="mx.acme.test",
    )
    assert result.spf == "fail"


def test_missing_header_is_unknown_and_never_a_pass():
    result = sender_auth.evaluate(_bundle(None))
    assert result.spf == sender_auth.UNKNOWN
    assert result.dkim == sender_auth.UNKNOWN
    assert result.dmarc == sender_auth.UNKNOWN
    assert result.asserted_by is None


def test_misalignment_between_envelope_and_header_from_is_detected():
    msg = EmailMessage()
    msg["From"] = "Sales <quotes@copperleaf.test>"
    msg["Return-Path"] = "<bounce@mailer-vendor.test>"
    msg["To"] = "intake@acme.procureiq.io"
    msg["Subject"] = "Quote"
    msg["Authentication-Results"] = "mx.acme.test; spf=pass; dkim=pass; dmarc=pass"
    msg.set_content("body")
    result = sender_auth.evaluate(
        parse(msg.as_bytes()), trusted_asserter="mx.acme.test"
    )
    assert result.alignment == "misaligned"
    assert result.header_from == "quotes@copperleaf.test"
    assert result.envelope_from == "bounce@mailer-vendor.test"


def test_untrusted_asserter_downgrades_the_verdicts():
    # A header we cannot attribute to our own boundary MTA is hearsay: anyone
    # upstream can write one. Recorded, but not treated as authoritative.
    result = sender_auth.evaluate(
        _bundle("evil-relay.test; spf=pass; dkim=pass; dmarc=pass"),
        trusted_asserter="mx.acme.test",
    )
    assert result.spf == sender_auth.UNKNOWN
    assert result.asserted_by == "evil-relay.test"


def test_trusted_asserter_is_honoured():
    result = sender_auth.evaluate(
        _bundle("mx.acme.test; spf=pass; dkim=pass; dmarc=pass"),
        trusted_asserter="mx.acme.test",
    )
    assert result.spf == "pass"


def test_only_the_topmost_authentication_results_header_is_used():
    msg = EmailMessage()
    msg["From"] = "Sales <quotes@copperleaf.test>"
    msg["To"] = "intake@acme.procureiq.io"
    msg["Subject"] = "Quote"
    msg["Authentication-Results"] = "mx.acme.test; spf=fail"
    msg["Authentication-Results"] = "forged.test; spf=pass"
    msg.set_content("body")
    result = sender_auth.evaluate(
        parse(msg.as_bytes()), trusted_asserter="mx.acme.test"
    )
    # The receiving MTA prepends; the first occurrence is the one it wrote.
    # A forged header appended below must not be able to overturn it.
    assert result.spf == "fail"
    assert result.asserted_by == "mx.acme.test"
