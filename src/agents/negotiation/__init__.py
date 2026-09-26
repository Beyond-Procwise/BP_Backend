"""Focused modules split out of agents/negotiation_agent.py.

The agent had grown to 13,669 lines, of which the NegotiationAgent class alone
was 11,959 — too large to hold in view while changing, which is why edits to it
carried the risk they did.

What moved here first are the provably pure helpers: methods that never mention
`self`, and therefore cannot call a sibling, read an attribute, or mutate
instance state. They are leaf functions, and moving one cannot change
behaviour. The class keeps a one-line delegator for each, so every existing
call site — including the tests that invoke them unbound with None for self —
still works unchanged.
"""
