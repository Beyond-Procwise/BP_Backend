"""Report Generation Agent: the platform owns the facts, the model composes.

The pipeline is a deterministic state machine. Nothing here is an agent loop and
nothing roams over tools:

    SCOPE -> FACT_PACK -> STYLE_BRIEF -> COMPOSE -> RENDER -> POST_CHECK -> RELEASE
                                                                 |
                                                            fail -> FINDING + BLOCK

Phase 1 (this code) builds every deterministic stage and the hand-written-AST
path through them. COMPOSE is not built: it needs a model, and the integration
the brief names does not exist — see docs/rga/discovery.md §3.5.

Importing this package registers the Fact Pack builders.
"""

from src.services.rga import builders  # noqa: F401  (registration side-effect)

__all__ = ["builders"]
