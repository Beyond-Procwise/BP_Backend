"""Assurance for supplier-facing email drafts: every figure traces to Postgres or is flagged.

This wraps the existing drafting paths; it does not replace them. Facts are resolved
before the draft is composed, and the composed text is checked after.

Nothing in this package writes to a business table, and nothing here sends mail.
"""

from .assure import SLUG, Inputs, prepare_inputs, recheck_facts  # noqa: F401
from .family import FamilyConfig, FamilyConfigUnavailable, load_family, parse_family  # noqa: F401
from .send import recheck_for_send  # noqa: F401
from . import capture  # noqa: F401
