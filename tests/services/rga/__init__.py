"""Package marker.

Without it this directory is not a package, so its test modules import under
their bare basenames and collide with same-named files elsewhere under tests/ —
which aborts collection for the whole sweep. See
tests/test_module_names_are_unique.py.
"""
