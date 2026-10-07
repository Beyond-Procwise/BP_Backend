"""Report data: a registry of metrics, query specs over it, and two providers behind one contract.

See docs/superpowers/specs/2026-07-13-report-builder-design.md (Stage 2). Rules that hold
everywhere in this package:

  * SQL is never built from user input. A request names registry keys; the registry maps
    each key to a fixed fragment. Values are always bound parameters.
  * No LLM produces or alters a number.
  * Live data is read under the caller's rights and scope; presentation data is never a
    fallback for live data and the two are never mixed in one report.
"""
