"""What an answer's sources are called, and how the model is told to treat each kind.

Three kinds of information reach an answer and they are not interchangeable:

  * enforced policy: the governed rows in ``proc.bp_policy``, the rules the system applies. Stated as fact.
  * policy documentation: the policy PDFs in the vector store. Quoted and attributed to the file; never stated as
    a fact about what the system does, because a document can be out of date or differ from the engine.
  * extracted records: figures read from the customer's own documents into the ``_trgt`` tables. Stated as data.

The wording lives here once so the label on a source and the instruction about it cannot drift apart.
"""
from __future__ import annotations

from typing import Any, Mapping

ENFORCED_POLICY = "Enforced policy"
POLICY_DOCUMENTATION = "Policy documentation"
EXTRACTED_RECORDS = "Your extracted documents"

ENFORCED_POLICY_HEADING = (
    "Enforced policy from the policy engine (these are the rules the system actually applies — state them as fact "
    "and quote the values exactly as given; where several are listed, use the ones that answer the question):\n"
)

DOCUMENTATION_NOTE = (
    "Policy documentation in the snippets below is quoted from named files. Attribute it to the file "
    "(\"the <file> says …\"). It is documentation, not an enforced rule, so do not state it as a fact about what "
    "the system does."
)

PRECEDENCE_NOTE = (
    "Where a policy document and an enforced policy differ, the enforced policy applies. Say that the document "
    "differs and give the enforced rule."
)


def document_name(payload: Mapping[str, Any]) -> str:
    """The file a quoted excerpt came from, falling back to its title."""
    for key in ("filename", "doc_name", "title"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
