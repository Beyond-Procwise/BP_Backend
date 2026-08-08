"""The concept vocabulary — derived from the extraction schemas, never hand-typed.

B3 resolution (2026-08-07). The brief asks that facts carry a stable concept
key and that we do not invent a second vocabulary shadowing an existing one.
There is no GPSS dictionary in this project, and on inspection there need not
be: ``extraction_schemas/*.yaml`` is already a versioned, declarative registry
of field definitions across the four document types, which is what a data
dictionary is — it simply is not called one.

So the vocabulary here is DERIVED by reading those schemas at import time. It
is deliberately not a hand-maintained list: a hand-typed copy is a second
vocabulary the moment either side changes, which is precisely the shadowing
failure the brief's constraint exists to prevent. If a field is added to a
schema it becomes a valid concept code automatically; if one is removed, facts
referencing it start failing validation, which is the correct signal.

The column is named ``concept_code`` and not ``gpss_code`` on purpose. Naming a
column after a standard that is not actually in use here would invite the next
reader to assume there is external authority behind the values. If GPSS is ever
adopted, ``concept_code`` is the column it maps into.

A formal external dictionary is only required for interoperability — exchanging
facts with another system, or mapping onto a customer's own taxonomy. Nothing
in Phases 1-5 requires that.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import FrozenSet

import yaml

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).resolve().parents[3] / "extraction_schemas"


def _load() -> FrozenSet[str]:
    """Collect every declared field name across the extraction schemas.

    Header fields and line-item fields alike: ``unit_price`` on a quote is the
    same concept as ``unit_price`` on an invoice, and the flat set is what makes
    that identity expressible at all.
    """
    names: set[str] = set()
    if not SCHEMA_DIR.is_dir():
        logger.warning("extraction schema directory not found at %s", SCHEMA_DIR)
        return frozenset()

    for path in sorted(SCHEMA_DIR.glob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except Exception:
            logger.exception("could not parse extraction schema %s", path)
            continue

        for field in doc.get("fields") or []:
            name = (field or {}).get("name")
            if name:
                names.add(str(name))

        # contract.yaml declares `line_items:` with no body, so this is None
        # rather than a dict. Guard for it.
        line_items = doc.get("line_items") or {}
        if isinstance(line_items, dict):
            for field in line_items.get("fields") or []:
                name = (field or {}).get("name")
                if name:
                    names.add(str(name))

    if not names:
        logger.warning("concept vocabulary loaded empty from %s", SCHEMA_DIR)
    return frozenset(names)


#: Every field name declared by any extraction schema. Parsed once at import.
CONCEPT_CODES: FrozenSet[str] = _load()


def is_known(code: str | None) -> bool:
    """True when ``code`` names a field an extraction schema actually declares."""
    if code is None:
        return False
    return code in CONCEPT_CODES
