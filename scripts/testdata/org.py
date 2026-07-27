"""The buying organisation: entities, business units, cost centres.

business_unit is empty in the live database and cost_centre holds 500 placeholder
rows pointing at business units that do not exist. This module builds both
properly, because tests E1-E3 (roll-up integrity, budget overrun, per-cost-centre
approval thresholds) have nothing to assert against otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.rng import make_rng, weighted_apportion

GROUP_ORG_ID = "ORG-GRP"
GROUP_NAME = "Beyond Procurement Group plc"


@dataclass(frozen=True)
class Entity:
    org_id: str
    name: str
    country: str
    currency: str
    spend_share: float
    cost_centre_count: int


ENTITIES: tuple[Entity, ...] = (
    Entity("ORG-UK", "Beyond Procurement UK Ltd", "United Kingdom", "GBP", 0.42, 185),
    Entity("ORG-US", "Beyond Procurement North America Inc", "United States", "USD", 0.18, 98),
    Entity("ORG-DE", "Beyond Procurement Deutschland GmbH", "Germany", "EUR", 0.16, 92),
    Entity("ORG-IE", "Beyond Procurement Ireland Ltd", "Ireland", "EUR", 0.12, 61),
    Entity("ORG-IN", "Beyond Procurement India Pvt Ltd", "India", "INR", 0.08, 42),
    Entity("ORG-AE", "Beyond Procurement Middle East FZE", "United Arab Emirates", "AED", 0.04, 22),
)

# L1 and L2 follow the convention already present in the live cost_centre rows:
# L1 is a function, L2 is a region.
BU_FUNCTIONS = ("Operations", "Sales", "Finance", "Corporate", "Technology", "Supply Chain")
BU_REGIONS = ("Europe", "North America", "Middle East", "LATAM", "APAC")
BU_DEPARTMENTS = (
    "Procurement", "Facilities", "Engineering", "Marketing", "Legal", "People",
    "Data", "Security", "Logistics", "Customer Success",
)
CC_TYPES = ("Overhead", "Project", "Capital", "Operational")

_FIRST_NAMES = (
    "Amelia", "Noah", "Priya", "Marcus", "Sofia", "Ethan", "Yusuf", "Chloe",
    "Rahul", "Freya", "Tomas", "Aisha", "Liam", "Nadia", "Oscar", "Mei",
)
_LAST_NAMES = (
    "Hartley", "Okafor", "Sharma", "Lindqvist", "Moreau", "Kowalski", "Rivera",
    "Bennett", "Haddad", "Novak", "Fitzgerald", "Alvarez", "Devlin", "Nakamura",
)


@dataclass(frozen=True)
class BusinessUnit:
    bu_id: str
    l1: str
    l2: str
    l3: str
    l4: str
    l5: str
    org_id: str
    head_name: str
    head_email: str
    region: str
    status: str


@dataclass(frozen=True)
class CostCentre:
    cc_id: str
    levels: tuple[str, ...]
    bu_id: str
    org_id: str
    finance_account_code: str
    manager_name: str
    manager_email: str
    spend_threshold_limit: float
    currency: str
    budget_allocated_annual: float
    actual_spend_ytd: float
    forecast_spend_annual: float
    cost_centre_type: str
    linked_category_level_5_id: str
    is_active: bool


def _person(rng) -> tuple[str, str]:
    first = rng.choice(_FIRST_NAMES)
    last = rng.choice(_LAST_NAMES)
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}@beyondprocurement.example"
    return name, email


def build_business_units(seed: int) -> list[BusinessUnit]:
    """400 L5 business units under a 6/40/120/240 tree."""
    rng = make_rng(seed, "business_units")

    l2_per_l1 = weighted_apportion([1] * len(BU_FUNCTIONS), 40)
    pairs: list[tuple[str, str]] = []
    for function, count in zip(BU_FUNCTIONS, l2_per_l1):
        for index in range(count):
            pairs.append((function, BU_REGIONS[index % len(BU_REGIONS)] + f" {index // len(BU_REGIONS) + 1}"))

    l3_per_pair = weighted_apportion([1] * len(pairs), 120)
    triples: list[tuple[str, str, str]] = []
    for (function, region), count in zip(pairs, l3_per_pair):
        for index in range(count):
            triples.append((function, region, BU_DEPARTMENTS[index % len(BU_DEPARTMENTS)] + f" {index + 1}"))

    l4_per_triple = weighted_apportion([1] * len(triples), 240)
    quads: list[tuple[str, str, str, str]] = []
    for (function, region, department), count in zip(triples, l4_per_triple):
        for index in range(count):
            quads.append((function, region, department, f"{department} Group {index + 1}"))

    l5_per_quad = weighted_apportion([1] * len(quads), 400)
    units: list[BusinessUnit] = []
    entity_cycle = [entity.org_id for entity in ENTITIES]
    counter = 0
    for (function, region, department, group), count in zip(quads, l5_per_quad):
        for index in range(count):
            head_name, head_email = _person(rng)
            counter += 1
            units.append(
                BusinessUnit(
                    bu_id=f"BU-5{counter:04d}",
                    l1=function,
                    l2=region,
                    l3=department,
                    l4=group,
                    l5=f"{group} Team {index + 1}",
                    org_id=entity_cycle[counter % len(entity_cycle)],
                    head_name=head_name,
                    head_email=head_email,
                    region=region.split(" ")[0],
                    status="Active",
                )
            )
    return units


def build_cost_centres(
    seed: int, units: Sequence[BusinessUnit], leaves: Sequence[TaxonomyLeaf]
) -> list[CostCentre]:
    """500 cost centres, allocated to entities per Entity.cost_centre_count."""
    rng = make_rng(seed, "cost_centres")
    by_entity: dict[str, list[BusinessUnit]] = {entity.org_id: [] for entity in ENTITIES}
    for unit in units:
        by_entity[unit.org_id].append(unit)

    currency_by_entity = {entity.org_id: entity.currency for entity in ENTITIES}
    centres: list[CostCentre] = []
    counter = 0

    for entity in ENTITIES:
        candidates = by_entity[entity.org_id] or list(units)
        for _ in range(entity.cost_centre_count):
            counter += 1
            unit = candidates[counter % len(candidates)]
            leaf = leaves[counter % len(leaves)]
            manager_name, manager_email = _person(rng)

            budget = round(rng.uniform(80_000, 4_500_000), 2)
            # Most cost centres sit under budget; the overrun defect is planted later.
            actual = round(budget * rng.uniform(0.35, 0.94), 2)
            forecast = round(actual * rng.uniform(1.02, 1.35), 2)

            centres.append(
                CostCentre(
                    cc_id=f"CC{counter:06d}",
                    levels=(
                        unit.l1,
                        unit.l2,
                        unit.l3,
                        unit.l4,
                        unit.l5,
                        f"{unit.l5} / {rng.choice(CC_TYPES)}",
                    ),
                    bu_id=unit.bu_id,
                    org_id=entity.org_id,
                    finance_account_code=f"FAC-{rng.randint(1000, 9999)}",
                    manager_name=manager_name,
                    manager_email=manager_email,
                    spend_threshold_limit=float(rng.choice(
                        [5_000, 10_000, 25_000, 50_000, 100_000, 250_000]
                    )),
                    currency=currency_by_entity[entity.org_id],
                    budget_allocated_annual=budget,
                    actual_spend_ytd=actual,
                    forecast_spend_annual=forecast,
                    cost_centre_type=rng.choice(CC_TYPES),
                    linked_category_level_5_id=leaf.unspsc_code or leaf.l5 or "UNKNOWN",
                    is_active=rng.random() > 0.05,
                )
            )
    return centres
