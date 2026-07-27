"""Map the supplier reference data the product reads.

Sibling of persist.py (documents) and persist_org.py (organisation, catalogue),
in the same idiom: a COLUMNS entry per table and a pure `rows_for_suppliers`
that touches no database.

Scope is the supplier data that is an *input*: third-party risk profiles, ESG
figures, and contacts. Rankings, reviews, enrichment, aliases and
supplier_risk_scores are excluded -- each is a conclusion the product reaches,
and seeding it would leave the agent that produces it untestable.

ESG figures are derived from the supplier's own generated attributes rather than
drawn independently, so a supplier holding ISO 14001 does not also report the
worst renewable-energy percentage in the set.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Sequence

from scripts.testdata.loader import MissingRequiredValue, _check_required
from scripts.testdata.rng import make_rng
from scripts.testdata.suppliers import SUPPLIER_COLUMNS, Supplier

__all__ = [
    "COLUMNS", "REQUIRED", "TABLES", "MissingRequiredValue",
    "check_required", "rows_for_suppliers",
]

MARKER = "testdata"
STAMP = datetime(2026, 7, 27, 0, 0, 0)

# Third-party risk is only maintained for suppliers that matter enough to
# warrant it. Seeding all 5,000 would misrepresent how the register is used.
TPRM_TIERS = ("Strategic",)

CRITICALITY = ("Material outsourcing", "Important business service", "Standard")
RISK_BANDS = ("low", "med", "high", "crit")
LOCATION_TYPES = ("Head Office", "Regional", "Site")
LANGUAGES = ("English", "German", "French", "Hindi", "Arabic")

COLUMNS: dict[str, tuple[str, ...]] = {
    "bp_supplier_uicanvas": SUPPLIER_COLUMNS,
    "bp_tprm_supplier": (
        "tp_id", "name", "segment", "criticality", "country",
        "inherent_risk", "residual_risk", "payload", "created_at", "updated_at",
    ),
    "esg_data": (
        "supplier_id", "esg_score", "carbon_emission_tco2",
        "renewable_energy_use_perc", "waste_recycled_perc", "certifications",
        "scope_1_emissions", "scope_2_emissions", "scope_3_emissions",
        "esg_audit_date", "third_party_rating", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
    "contact": (
        "contact_id", "supplier_id", "contact_name", "contact_role",
        "contact_email", "contact_phone", "is_primary_contact", "location_type",
        "language_preference", "contact_status", "created_date", "created_by",
        "last_modified_by", "last_modified_date",
    ),
}
# bp_contact carries the identical shape; the two differ only by name.
COLUMNS["bp_contact"] = COLUMNS["contact"]

REQUIRED: dict[str, tuple[str, ...]] = {
    "bp_supplier_uicanvas": ("supplier_id", "supplier_name", "country", "default_currency"),
    "bp_tprm_supplier": (
        "tp_id", "name", "segment", "criticality", "country",
        "inherent_risk", "residual_risk", "payload",
    ),
    "esg_data": (
        "supplier_id", "esg_score", "carbon_emission_tco2",
        "renewable_energy_use_perc", "esg_audit_date",
    ),
    "contact": (
        "contact_id", "supplier_id", "contact_name", "contact_email",
        "is_primary_contact", "contact_status",
    ),
}
REQUIRED["bp_contact"] = REQUIRED["contact"]

TABLES: tuple[str, ...] = tuple(COLUMNS)

# The physical table each logical key writes to. bp_supplier_uicanvas is the
# uicanvas copy of bp_supplier, distinct from bp_sqldb's table of that name.
TARGET_TABLE: dict[str, str] = {
    "bp_supplier_uicanvas": "bp_supplier",
    "bp_tprm_supplier": "bp_tprm_supplier",
    "esg_data": "esg_data",
    "contact": "contact",
    "bp_contact": "bp_contact",
}


def check_required(table: str, rows: Sequence[Sequence]) -> None:
    _check_required(table, COLUMNS[table], REQUIRED[table], rows)


def _tprm_payload(supplier: Supplier, inherent: str, residual: str, rng) -> str:
    """The TPRM screen reads this blob, so its shape must match what live holds."""
    return json.dumps(
        {
            "id": supplier.bp_supplier_id,
            "name": supplier.name,
            "seg": supplier.tier,
            "crit": CRITICALITY[0],
            "country": supplier.country,
            "inh": inherent,
            "res": residual,
            "dom": {
                "Financial": rng.randint(1, 5),
                "Data privacy": rng.randint(1, 5),
                "Fourth-party": rng.randint(1, 5),
                "Concentration": rng.randint(1, 5),
                "ESG & conduct": rng.randint(1, 5),
                "Cyber & infosec": rng.randint(1, 5),
                "Compliance & reg": rng.randint(1, 5),
                "Operational resilience": rng.randint(1, 5),
            },
            "svc": supplier.primary_leaf.l5,
            "spend": None,
            "certs": supplier.columns["insurance_coverage_type"],
            "sanctions": "Clear",
            "owner": supplier.columns["contact_name_1"],
        },
        sort_keys=True,
    )


def _esg_row(supplier: Supplier, rng) -> list:
    # Derived from the supplier's own certifications so the figures agree with
    # the master record rather than contradicting it.
    certified = supplier.columns["esg_cert_iso14001"]
    ecovadis = supplier.columns["esg_cert_ecovadis"]
    score = rng.randint(62, 94) if certified else rng.randint(28, 70)
    renewable = rng.randint(45, 95) if certified else rng.randint(5, 55)

    scope_1 = round(rng.uniform(120, 9_000), 2)
    scope_2 = round(rng.uniform(80, 6_500), 2)
    scope_3 = round(rng.uniform(500, 48_000), 2)

    certifications = ", ".join(
        name for name, held in (
            ("ISO 14001", certified),
            ("SA8000", supplier.columns["esg_cert_sa8000"]),
            ("EcoVadis", ecovadis),
        ) if held
    ) or "None"

    return [
        supplier.uicanvas_supplier_id, score,
        round(scope_1 + scope_2 + scope_3, 2),
        renewable, rng.randint(10, 92), certifications,
        scope_1, scope_2, scope_3,
        date(2025, 1, 1) + timedelta(days=rng.randrange(0, 540)),
        rng.choice("ABCD"),
        STAMP, MARKER, MARKER, STAMP,
    ]


def _contact_rows(supplier: Supplier, rng) -> list[list]:
    """One primary contact per supplier, from the master record's own contact."""
    return [[
        f"CON-{supplier.uicanvas_supplier_id}",
        supplier.uicanvas_supplier_id,
        supplier.columns["contact_name_1"],
        supplier.columns["contact_role_1"],
        supplier.columns["contact_email_1"],
        supplier.columns["contact_phone_1"],
        True,
        rng.choice(LOCATION_TYPES),
        rng.choice(LANGUAGES),
        "Active",
        STAMP, MARKER, MARKER, STAMP,
    ]]


def rows_for_suppliers(suppliers: Sequence[Supplier]) -> dict[str, list[list]]:
    """Every table's rows, in COLUMNS order. Pure: no database contact."""
    rng = make_rng(42, "supplier_reference")

    ui_master: list[list] = []
    tprm: list[list] = []
    esg: list[list] = []
    contacts: list[list] = []

    for supplier in suppliers:
        ui_master.append([
            supplier.bp_supplier_id if column == "supplier_id"
            else supplier.columns[column]
            for column in SUPPLIER_COLUMNS
        ])

        if supplier.tier in TPRM_TIERS:
            inherent = rng.choice(RISK_BANDS)
            residual = rng.choice(RISK_BANDS)
            tprm.append([
                supplier.bp_supplier_id, supplier.name, supplier.tier,
                CRITICALITY[0], supplier.country, inherent, residual,
                _tprm_payload(supplier, inherent, residual, rng),
                STAMP, STAMP,
            ])

        esg.append(_esg_row(supplier, rng))
        contacts.extend(_contact_rows(supplier, rng))

    return {
        "bp_supplier_uicanvas": ui_master,
        "bp_tprm_supplier": tprm,
        "esg_data": esg,
        "contact": contacts,
        "bp_contact": [list(row) for row in contacts],
    }
