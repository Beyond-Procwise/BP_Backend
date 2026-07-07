"""Extraction feedback loop v1 — propose-only per-vendor hints.

Turns recurring per-vendor extraction failures (from telemetry + discrepancies)
into human-approved, versioned extraction hints that the live extraction prompt
consumes. Propose-only: nothing changes extraction until a human approves.
"""
