"""proc.bp_rule matches the seed, and the separation holds in the live database.

Needs PROCWISE_TEST_LIVE_DB=1; without it pytest uses a fake DB that cannot
answer any of these questions.
"""

import json
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
    reason="live database required (PROCWISE_TEST_LIVE_DB=1)",
)

DATABASES = ("bp_testdb", "bp_sqldb")

EXPECTED_RULES = {
    "price_variance_check": ("Price Benchmark Variance", {}),
    "volume_consolidation_check": ("Volume Consolidation", {}),
    "contract_expiry_check": ("Contract Expiry Opportunity", {"negotiation_window_days": 90}),
    "supplier_risk_check": ("Supplier Risk Alert", {}),
    "maverick_spend_check": ("Maverick Spend Detection", {}),
    "duplicate_supplier_check": ("Duplicate Supplier", {}),
    "category_overspend_check": ("Category Overspend", {}),
    "inflation_passthrough_check": ("Inflation Pass-Through", {}),
    "unused_contract_value_check": ("Unused Contract Value", {}),
    "supplier_performance_check": ("Supplier Performance Deviation", {}),
    "esg_opportunity_check": ("ESG Opportunity", {}),
    "invoice_po_variance_check": ("Invoice Overbilling", {"variance_threshold_pct": 10.0}),
    # Added by 2026-10-05_contract_expiry_bucket_rule.sql; edges set by ..._match_graph_spans.sql
    "contract_expiry_bucket_check": ("Contract Expiry Buckets", {
        "bucket_months": [1, 6, 12, 24],
        "alert_expired": True,
        "flag_missing_end_date": True,
        "lifecycle_status": "active",
        "demand_contract_key": "contract_id",
        "inactive_demand_statuses": ["draft", "closed", "cancelled", "rejected",
                                     "completed", "won", "lost"],
    }),
}


def _connect(dbname):
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=dbname,
        connect_timeout=10,
    )
    conn.set_session(readonly=True)
    return conn


@pytest.mark.parametrize("dbname", DATABASES)
def test_every_seeded_rule_is_present_and_active(dbname):
    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT detector_slug, rule_name, conditions FROM proc.bp_rule "
            "WHERE rule_status = 1"
        )
        live = {slug: (name, conds) for slug, name, conds in cur.fetchall()}

    assert set(live) == set(EXPECTED_RULES)
    for slug, (name, conditions) in EXPECTED_RULES.items():
        assert live[slug][0] == name, slug
        assert live[slug][1] == conditions, slug


@pytest.mark.parametrize("dbname", DATABASES)
def test_absent_thresholds_are_absent_not_zero(dbname):
    """The ten detectors with no default must hold '{}', never a zero.

    A zero on a minimum-value filter means "fire on everything", so a helpful
    hand filling these in would quietly turn ten detectors into noise.
    """
    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT detector_slug, conditions FROM proc.bp_rule "
            "WHERE rule_status = 1"
        )
        rows = dict(cur.fetchall())

    without_default = [s for s, (_, c) in EXPECTED_RULES.items() if not c]
    assert len(without_default) == 10
    for slug in without_default:
        assert rows[slug] == {}, f"{slug} acquired a threshold: {rows[slug]}"


@pytest.mark.parametrize("dbname", DATABASES)
def test_detector_slug_is_unique(dbname):
    """Without this the seed's ON CONFLICT has nothing to conflict on."""
    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM pg_indexes "
            "WHERE schemaname = 'proc' AND tablename = 'bp_rule' "
            "AND indexdef LIKE '%UNIQUE%detector_slug%'"
        )
        assert cur.fetchone()[0] == 1


@pytest.mark.parametrize("dbname", DATABASES)
def test_no_detector_configuration_is_left_in_the_policy_table(dbname):
    """The separation, asserted against the live row set.

    A policy answers "is this allowed". If an active policy row reappears
    carrying detector configuration, the two concerns have merged again.
    """
    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT policy_id, policy_name FROM proc.bp_policy "
            "WHERE policy_status = 1 AND policy_type = 'opportunity'"
        )
        leftovers = cur.fetchall()

    assert leftovers == [], f"detector config still active in bp_policy: {leftovers}"


@pytest.mark.parametrize("dbname", DATABASES)
def test_no_active_policy_carries_detector_configuration(dbname):
    """The second door, closed.

    ``_get_policy_engine_catalog`` enriches caller-supplied policies from every
    active bp_policy row, so a row carrying ``default_conditions`` or a
    detector slug could still reach detection by that route even though the
    registry no longer reads policies. Nothing in the table may carry those
    shapes.
    """
    detector_slugs = set(EXPECTED_RULES)

    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT policy_id, policy_name, policy_details::text "
            "FROM proc.bp_policy WHERE policy_status = 1"
        )
        rows = cur.fetchall()

    offenders = []
    for policy_id, name, details_text in rows:
        details = json.loads(details_text) if details_text else {}
        if not isinstance(details, dict):
            continue
        if "default_conditions" in details:
            offenders.append((policy_id, name, "default_conditions"))
        identifier = str(details.get("policy_identifier") or "")
        if identifier in detector_slugs:
            offenders.append((policy_id, name, f"detector slug {identifier}"))

    assert offenders == [], f"detector configuration reachable via bp_policy: {offenders}"
