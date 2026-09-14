"""Fact Pack builders, one per report type.

Importing this package is what puts a report type in the registry. A builder
that is written but never imported is a report type that raises
``NoBuilderRegistered`` at run time and passes every test that imports it
directly — so the import lives here rather than being left to the caller.
"""

from src.services.rga.builders import exec_procurement_summary  # noqa: F401

__all__ = ["exec_procurement_summary"]
