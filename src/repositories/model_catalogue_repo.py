# src/repositories/model_catalogue_repo.py
"""Which models this installation offers — data, not code.

An agent runs on the standard model unless someone deliberately overrides it.
Which models exist, which are offered, and which one is standard are ROWS, so a
customer who wants a different model gets one row changed rather than a deploy.
Nothing here is hardcoded into the UI: the workspace renders whatever is
offered, and renders no model control at all when only the standard model is.

Deliberately small. A model row carries what is needed to call the model and to
name it to a human, and nothing else:

* ``model_key``    stable id an agent stores ("standard"). Never the vendor's name,
                   so re-pointing "standard" at a different model does not orphan
                   every agent that chose it.
* ``display_name`` what a human reads. The standard row reads "Default" — a
                   customer has no reason to care which model that is.
* ``provider``     how it is reached: ollama_local, ollama_cloud, openai, anthropic.
* ``model_ref``    what the provider is actually called with.
* ``model_status`` 1 = offered. THIS is the switch. Seeded models that are not
                   offered stay as rows so enabling one is a flag, not an insert.
* ``requires_key`` a metered provider. Never selectable while its key is unset —
                   an agent that silently starts billing is not a feature.

Only models proven present on this machine are seeded. A cloud or vendor model
is a one-row INSERT, e.g.:

    INSERT INTO proc.bp_model (model_key, display_name, provider, model_ref,
                               model_status, requires_key, notes)
    VALUES ('cloud-large', 'Large context (cloud)', 'ollama_cloud',
            '<the cloud model>', 1, TRUE, 'Charged by the provider');
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_model (
    model_id      BIGSERIAL PRIMARY KEY,
    model_key     TEXT NOT NULL UNIQUE,
    display_name  TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model_ref     TEXT NOT NULL,
    is_default    BOOLEAN NOT NULL DEFAULT FALSE,
    model_status  SMALLINT NOT NULL DEFAULT 0,
    requires_key  BOOLEAN NOT NULL DEFAULT FALSE,
    notes         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One standard model, enforced by the database rather than by hope.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_model_default
    ON proc.bp_model (is_default) WHERE is_default;

CREATE INDEX IF NOT EXISTS ix_bp_model_offered
    ON proc.bp_model (model_status, display_name);
"""

# Which env var names a provider's key. A row whose provider needs a key it does
# not have is listed but never selectable (see ``_selectable``).
_PROVIDER_KEY_ENV = {
    "ollama_cloud": "OLLAMA_CLOUD_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def ensure_schema() -> None:
    """Create the table and seed it once. Idempotent."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute("SELECT COUNT(*) FROM proc.bp_model")
        if cur.fetchone()[0] == 0:
            _seed(cur)
        conn.commit()


def _seed(cur) -> None:
    """Seed from what this machine actually runs.

    The standard row is whatever the extraction/LLM settings already point at, so
    the catalogue agrees with reality on day one instead of announcing a model
    nothing uses. The rest are seeded switched OFF: they exist so that offering
    one is a flag flip, and so nobody has to remember the vendor's exact string.
    """
    standard = (
        os.getenv("PROCWISE_EXTRACTION_MODEL")
        or os.getenv("LOCAL_PRIMARY_MODEL")
        or "BeyondProcwise/AgentNick:unified"
    ).strip().strip('"')

    rows = [
        ("standard", "Default", "ollama_local", standard, True, 1, False,
         "The model every agent uses unless it is given its own."),
    ]
    extras = [
        ("extraction", "Extraction specialist", "ollama_local",
         "BeyondProcwise/AgentNick:extract", False, 0, False,
         "Tuned for reading documents rather than reasoning about them."),
        ("general", "General purpose", "ollama_local",
         "BeyondProcwise/AgentNick:latest", False, 0, False, None),
    ]
    for key, name, provider, ref, is_default, status, needs_key, note in rows + extras:
        cur.execute(
            "INSERT INTO proc.bp_model (model_key, display_name, provider, model_ref, "
            "is_default, model_status, requires_key, notes) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (model_key) DO NOTHING",
            (key, name, provider, ref, is_default, status, needs_key, note),
        )
    logger.info("bp_model seeded; standard model is %s", standard)


def _selectable(row: Dict[str, Any]) -> bool:
    """A metered model with no key configured must not be offerable."""
    if not row.get("requires_key"):
        return True
    env = _PROVIDER_KEY_ENV.get(row.get("provider") or "")
    return bool(env and os.getenv(env))


_COLUMNS = ("model_key", "display_name", "provider", "model_ref", "is_default",
            "model_status", "requires_key", "notes")


def _row(record) -> Dict[str, Any]:
    row = dict(zip(_COLUMNS, record))
    row["selectable"] = _selectable(row)
    return row


def list_offered() -> List[Dict[str, Any]]:
    """Offered models, standard first. Rows switched off are not returned at all."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM proc.bp_model "
            "WHERE model_status = 1 ORDER BY is_default DESC, display_name"
        )
        return [_row(r) for r in cur.fetchall()]


def get(model_key: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM proc.bp_model WHERE model_key = %s",
            (model_key,),
        )
        record = cur.fetchone()
        return _row(record) if record else None


def default_model() -> Optional[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM proc.bp_model WHERE is_default LIMIT 1"
        )
        record = cur.fetchone()
        return _row(record) if record else None


def resolve_ref(model_key: Optional[str]) -> Optional[str]:
    """The provider-facing model string for a key, or None.

    Returns None for the standard key as well as for an unknown one: "standard"
    means "whatever every other agent uses", which is decided by settings at call
    time, not pinned here. Pinning it would freeze an agent onto today's standard
    model and quietly leave it behind the next time the standard moves.
    """
    if not model_key:
        return None
    row = get(model_key)
    if not row or row.get("is_default"):
        return None
    return row.get("model_ref")
