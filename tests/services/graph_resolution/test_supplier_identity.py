from src.services import linking_engine as _le
from src.services.graph_resolution.profiles import supplier_identity as si

A = {"supplier_id": "SUP-Acme", "supplier_name": "Acme Ltd",
     "vat_number": "GB123", "registration_number": "R1",
     "duns_number": "D1", "postal_code": "SW1A 1AA", "country": "GB"}


def test_identical_records_are_clearly_separated_from_a_different_company():
    b = {"supplier_id": "SUP-Other", "supplier_name": "Globex plc",
         "vat_number": "GB999", "registration_number": "R2",
         "duns_number": "D2", "postal_code": "M1 1AA", "country": "GB"}
    r_same = si.score(A, dict(A))
    r_diff = si.score(A, b)
    assert r_same["F"] > r_diff["F"], "identical must outscore an unrelated company"
    assert r_diff["decision"] == "block_or_exception", r_diff["F"]


def test_auto_link_is_structurally_unreachable_for_this_profile():
    """The maximal case: two identical records with every signal observable,
    including bank_account_number -- the most evidence this profile can ever
    see. Measured ceiling: F=86.27 (auto_link_with_warning), never auto_link.

    This documents a consequence of the uncalibrated p0/alpha (calibrated in
    Task 5), not a defect to be tuned away here -- the suite should carry the
    limitation rather than leave it as folklore.
    """
    full = {**A, "bank_account_number": "BANK1"}
    r = si.score(full, dict(full))
    assert r["decision"] == "auto_link_with_warning", r["F"]
    assert r["F"] < _le._BAND_AUTO


def test_same_company_across_keyspaces_still_resolves():
    """The whole point: SUP-* and S#### never join on id.

    Asserts SEPARATION, not an absolute band: p0/alpha are uncalibrated until
    Task 5, and asserting a calibrated outcome from uncalibrated parameters
    tests the starting constants rather than the profile.
    """
    same = {**A, "supplier_id": "S9251"}
    other = {"supplier_id": "S9252", "supplier_name": "Globex plc",
             "vat_number": "GB999", "registration_number": "R2",
             "duns_number": "D2", "postal_code": "M1 1AA", "country": "GB"}
    r_same, r_other = si.score(A, same), si.score(A, other)
    assert r_same["F"] > r_other["F"], "agreement must outscore disagreement"
    assert r_same["decision"] != "block_or_exception", (
        f"a company agreeing on VAT, registration and DUNS must at least be "
        f"reported: {r_same['F']}"
    )


def test_different_company_is_not_linked():
    b = {"supplier_id": "SUP-Other", "supplier_name": "Globex plc",
         "vat_number": "GB999", "registration_number": "R2",
         "duns_number": "D2", "postal_code": "M1 1AA", "country": "GB"}
    r = si.score(A, b)
    assert r["decision"] in ("weak_relation", "block_or_exception"), r["F"]


def test_missing_registration_data_contributes_nothing_either_way():
    sparse = {"supplier_id": "SUP-Acme2", "supplier_name": "Acme Ltd"}
    r = si.score(A, sparse)
    statuses = {s["id"]: s["status"] for s in r["signals"]}
    assert statuses["vat"] == "MISSING"
    assert r["F"] > 0.0, "a missing signal must not zero the whole score"


def test_registration_signals_share_one_cluster():
    clusters = {s["id"]: s["cluster"] for s in si.SIGNALS}
    assert clusters["vat"] == clusters["reg_no"] == clusters["duns"]


def test_observations_name_the_fields_each_signal_read():
    obs = si.observations_for(A, dict(A))
    assert ("SUP-Acme", "vat_number") in obs["vat"]
    assert ("SUP-Acme", "supplier_name") in obs["name"]
