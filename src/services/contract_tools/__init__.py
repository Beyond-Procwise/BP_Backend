"""Read-only contract tools for the agent: find contracts, read their stored clauses, compare two texts.

None of them writes, and none of them reads a contract with the model: clauses come from the
grounded obligations already stored (each quote was checked against the document), and the
comparison is a plain text diff. When a store cannot be read the tool says so; it never returns
an empty list that would read as "no contracts".
"""
