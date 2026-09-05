"""Deterministic negotiation state machine and its four narrow functions.

Replaces the reasoning that lived in ``src/engines/negotiation_strategy_engine.py``
(deleted: constructed at every boot, reachable from nothing) and the leverage
reasoning that never existed in ``src/agents/negotiation_agent.py`` at all.
"""
from src.services.negotiation.leverage import BatnaAssessment, assess_batna

__all__ = ["BatnaAssessment", "assess_batna"]
