"""Test-data generator tests.

Integration tests here create, drop and truncate databases. They must never
touch `bp_testdb` or `uicanvas_test`: those hold the built dataset, and a test
run would silently destroy it. Every integration test targets the scratch
databases named below instead.
"""

SCRATCH_DB = "bp_testdb_it"
SCRATCH_UICANVAS_DB = "uicanvas_test_it"
