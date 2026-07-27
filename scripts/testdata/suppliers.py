"""5,000 suppliers, materialised under both ID conventions.

bp_sqldb names suppliers SUP-<PascalCase>; uicanvas uses SI######. Both are
reproduced and joined by a crosswalk, so the mismatch between the two databases
is testable rather than hidden behind a single invented convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Sequence

from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.rng import make_rng, weighted_apportion

SUPPLIER_COLUMNS: tuple[str, ...] = (
    "supplier_id", "supplier_name", "trading_name", "supplier_type", "legal_structure",
    "tax_id", "vat_number", "duns_number", "parent_company_id", "registered_country",
    "registration_number", "is_preferred_supplier", "risk_score", "credit_limit_amount",
    "esg_cert_iso14001", "esg_cert_sa8000", "esg_cert_ecovadis", "diversity_women_owned",
    "diversity_minority_owned", "diversity_veteran_owned", "insurance_coverage_type",
    "insurance_coverage_amount", "insurance_expiry_date", "bank_name",
    "bank_account_number", "bank_swift", "bank_iban", "default_currency", "incoterms",
    "delivery_lead_time_days", "address_line1", "address_line2", "city", "postal_code",
    "country", "website_url", "edi_enabled", "api_enabled", "ariba_integrated",
    "contact_name_1", "contact_role_1", "contact_email_1", "contact_phone_1",
    "contact_name_2", "contact_role_2", "contact_email_2", "contact_phone_2",
    "created_date", "created_by", "last_modified_by", "last_modified_date",
)

# Vocabularies taken from the existing uicanvas.proc.supplier master.
SUPPLIER_TYPES = ("Service Provider", "Wholesaler", "Manufacturer", "Retailer", "Distributor", "Consulting")
LEGAL_STRUCTURES = ("PLC", "Ltd", "LLP", "Inc", "GmbH", "LLC")
INCOTERMS = ("DAP", "DDP", "FOB", "CIF", "EXW")
INSURANCE_TYPES = ("General Liability", "Product Liability", "Professional Indemnity", "Cyber")
CONTACT_ROLES = (
    "Account Manager", "Key Account Manager", "Client Director",
    "Commercial Manager", "Sales Manager", "Customer Success Manager",
    "Business Development Manager",
)

_CONTACT_FIRST_NAMES = (
    "Alex", "Priya", "Sam", "Nadia", "Tom", "Elena", "Marcus", "Hannah",
    "Rahul", "Sofia", "Daniel", "Aisha", "Chloe", "Omar", "Grace", "Lukas",
)
_CONTACT_LAST_NAMES = (
    "Reed", "Osei", "Kaur", "Blake", "Moretti", "Whitfield", "Nowak", "Ibrahim",
    "Lindgren", "Barnes", "Ferreira", "Hughes", "Delacroix", "Yilmaz",
)

# (country, weight, currency)
COUNTRIES: tuple[tuple[str, int, str], ...] = (
    ("United Kingdom", 3100, "GBP"), ("United States", 300, "USD"),
    ("Germany", 250, "EUR"), ("Ireland", 200, "EUR"), ("France", 175, "EUR"),
    ("Netherlands", 150, "EUR"), ("India", 150, "INR"), ("Poland", 125, "PLN"),
    ("Spain", 100, "EUR"), ("Italy", 100, "EUR"),
    ("United Arab Emirates", 100, "AED"), ("China", 100, "USD"),
    ("Sweden", 50, "SEK"), ("Switzerland", 25, "CHF"), ("Singapore", 15, "SGD"),
    ("Japan", 10, "JPY"), ("Australia", 5, "AUD"), ("Canada", 45, "USD"),
)

_STEMS = (
    "Northgate", "Brightpath", "Vantage", "Ironbridge", "Clearwater", "Summit",
    "Meridian", "Kestrel", "Blackwood", "Harbourline", "Redstone", "Silverbeck",
    "Oakfield", "Copperleaf", "Windrose", "Falconridge", "Stonegate", "Thornbury",
    "Lighthouse", "Ashcroft", "Greenhollow", "Pinnacle", "Crossfell", "Marlowe",
)
_SUFFIXES = (
    "Solutions", "Systems", "Group", "Partners", "Industries", "Services",
    "Supplies", "Technologies", "Logistics", "Associates", "Works", "Trading",
)


@dataclass(frozen=True)
class Tier:
    name: str
    count: int
    spend_share: float
    min_documents: int
    max_documents: int


TIERS: tuple[Tier, ...] = (
    Tier("Strategic", 120, 0.38, 60, 140),
    Tier("Core", 700, 0.41, 12, 45),
    Tier("Tail", 3020, 0.18, 2, 9),
    Tier("One-off", 1160, 0.03, 1, 1),
)


@dataclass(frozen=True)
class Supplier:
    bp_supplier_id: str
    uicanvas_supplier_id: str
    name: str
    tier: str
    primary_leaf: TaxonomyLeaf
    secondary_leaves: tuple[TaxonomyLeaf, ...]
    country: str
    currency: str
    columns: dict[str, Any] = field(hash=False, compare=True)


CROSSWALK_DDL = """
create table if not exists proc.bp_supplier_id_crosswalk (
    bp_supplier_id        text primary key,
    uicanvas_supplier_id  text not null unique,
    legal_entity_key      text not null,
    created_date          timestamp not null default now()
);
create index if not exists ix_bp_supplier_id_crosswalk_uicanvas_supplier_id
    on proc.bp_supplier_id_crosswalk (uicanvas_supplier_id);
create index if not exists ix_bp_supplier_id_crosswalk_legal_entity_key
    on proc.bp_supplier_id_crosswalk (legal_entity_key);
"""


def _pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace(",", " ").split() if part)


def _country_for(rng) -> tuple[str, str]:
    total = sum(weight for _, weight, _ in COUNTRIES)
    pick = rng.randrange(total)
    running = 0
    for country, weight, currency in COUNTRIES:
        running += weight
        if pick < running:
            return country, currency
    return COUNTRIES[0][0], COUNTRIES[0][2]


def build_suppliers(seed: int, leaves: Sequence[TaxonomyLeaf]) -> list[Supplier]:
    """Exactly 5,000 suppliers with all 51 bp_supplier columns populated."""
    rng = make_rng(seed, "suppliers")

    per_leaf = weighted_apportion([1] * len(leaves), 5000)
    leaf_slots: list[TaxonomyLeaf] = []
    for leaf, count in zip(leaves, per_leaf):
        leaf_slots.extend([leaf] * count)

    tier_slots: list[str] = []
    for tier in TIERS:
        tier_slots.extend([tier.name] * tier.count)

    base_date = datetime(2022, 1, 1)
    suppliers: list[Supplier] = []
    used_names: set[str] = set()

    for index in range(5000):
        stem = _STEMS[rng.randrange(len(_STEMS))]
        suffix = _SUFFIXES[rng.randrange(len(_SUFFIXES))]
        name = f"{stem} {suffix}"
        disambiguator = 2
        while name in used_names:
            name = f"{stem} {suffix} {disambiguator}"
            disambiguator += 1
        used_names.add(name)

        contact_first = rng.choice(_CONTACT_FIRST_NAMES)
        contact_last = rng.choice(_CONTACT_LAST_NAMES)
        contact_name = f"{contact_first} {contact_last}"
        contact_role = rng.choice(CONTACT_ROLES)

        country, currency = _country_for(rng)
        leaf = leaf_slots[index]
        tier = tier_slots[index]

        secondary_count = rng.choice([0, 1, 1, 2])
        secondary = tuple(
            leaves[rng.randrange(len(leaves))] for _ in range(secondary_count)
        )

        bp_id = f"SUP-{_pascal(name)}"
        ui_id = f"SI{index + 1:06d}"
        created = base_date + timedelta(days=rng.randrange(0, 900))
        expiry = date(2026, 1, 1) + timedelta(days=rng.randrange(-400, 1800))

        columns: dict[str, Any] = {
            "supplier_id": bp_id,
            "supplier_name": name,
            "trading_name": f"{name} Trading Ltd.",
            "supplier_type": rng.choice(SUPPLIER_TYPES),
            "legal_structure": rng.choice(LEGAL_STRUCTURES),
            "tax_id": f"TX{rng.randrange(10_000_000, 99_999_999)}",
            "vat_number": f"VAT{rng.randrange(10_000_000, 99_999_999)}",
            "duns_number": str(rng.randrange(100_000_000, 999_999_999)),
            "parent_company_id": None,
            "registered_country": country,
            "registration_number": f"REG{rng.randrange(10_000_000, 99_999_999)}",
            "is_preferred_supplier": tier in ("Strategic", "Core"),
            "risk_score": f"{rng.uniform(5, 95):.2f}",
            "credit_limit_amount": round(rng.uniform(25_000, 5_000_000), 2),
            "esg_cert_iso14001": rng.random() < 0.42,
            "esg_cert_sa8000": rng.random() < 0.21,
            "esg_cert_ecovadis": rng.random() < 0.33,
            "diversity_women_owned": rng.random() < 0.18,
            "diversity_minority_owned": rng.random() < 0.14,
            "diversity_veteran_owned": rng.random() < 0.07,
            "insurance_coverage_type": rng.choice(INSURANCE_TYPES),
            "insurance_coverage_amount": round(rng.uniform(250_000, 10_000_000), 2),
            "insurance_expiry_date": expiry,
            "bank_name": f"{rng.choice(_STEMS)} Bank",
            "bank_account_number": str(rng.randrange(10_000_000, 99_999_999)),
            "bank_swift": f"SWIFT{rng.randrange(10_000_000, 99_999_999)}",
            "bank_iban": f"IBAN{rng.randrange(1_000_000_000, 9_999_999_999)}",
            "default_currency": currency,
            "incoterms": rng.choice(INCOTERMS),
            "delivery_lead_time_days": str(rng.randrange(1, 60)),
            "address_line1": f"{rng.randrange(1, 400)} {rng.choice(_STEMS)} Road",
            "address_line2": None,
            "city": rng.choice(("London", "Manchester", "Dublin", "Berlin", "Austin", "Pune", "Dubai")),
            "postal_code": f"{rng.choice('ABCDEFGHMNPRSW')}{rng.randrange(1, 99)} {rng.randrange(1, 9)}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}",
            "country": country,
            "website_url": f"https://{stem.lower()}{suffix.lower()}.example",
            "edi_enabled": rng.random() < 0.28,
            "api_enabled": rng.random() < 0.19,
            "ariba_integrated": rng.random() < 0.11,
            "contact_name_1": contact_name,
            "contact_role_1": contact_role,
            # Addressed to the person, not a shared sales alias: every supplier
            # showing the same mailbox reads as generated at a glance.
            "contact_email_1": (
                f"{contact_first.lower()}.{contact_last.lower()}"
                f"@{stem.lower()}{suffix.lower()}.example"
            ),
            "contact_phone_1": f"+44 20 {rng.randrange(1000, 9999)} {rng.randrange(1000, 9999)}",
            "contact_name_2": None,
            "contact_role_2": None,
            "contact_email_2": None,
            "contact_phone_2": None,
            "created_date": created,
            "created_by": "testdata",
            "last_modified_by": "testdata",
            "last_modified_date": created,
        }

        suppliers.append(
            Supplier(
                bp_supplier_id=bp_id,
                uicanvas_supplier_id=ui_id,
                name=name,
                tier=tier,
                primary_leaf=leaf,
                secondary_leaves=secondary,
                country=country,
                currency=currency,
                columns=columns,
            )
        )
    return suppliers


def build_crosswalk(suppliers: Sequence[Supplier]) -> list[tuple[str, str, str]]:
    """(bp_supplier_id, uicanvas_supplier_id, legal_entity_key) for every supplier."""
    return [
        (
            supplier.bp_supplier_id,
            supplier.uicanvas_supplier_id,
            f"{supplier.name}|{supplier.columns['vat_number']}",
        )
        for supplier in suppliers
    ]
