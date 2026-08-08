"""Typed, provenanced commercial facts.

Everything in this package is deterministic. No module here may call an LLM:
a fact is a record of what a document said, and an inference dressed as a fact
is indistinguishable from one downstream.
"""
