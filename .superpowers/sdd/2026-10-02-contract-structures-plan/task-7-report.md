# Task 7 report

Files: src/services/concepts/contract_type_map.py (brief module verbatim + a status-rule comment at the resolve_alias call), tests/services/concepts/test_contract_type_map.py.

## TDD
Pre-implementation run: collection failed ("cannot import name 'contract_type_map' from 'src.services.concepts'"; the brief predicted ModuleNotFoundError, same cause).
After implementing: 21 passed. Vocabulary source observed: `bp_concept@48+bp_document_type@20`, 20 document types (as expected). Coverage exactly 3016 / 35, unmapped {"":29,"Policy":3,"Service":2,"Indirect Procurement":1}; total 3051.

## Test changes vs brief
- `live_vocab` module fixture asserts `vocab.source.startswith("bp_concept@")`; every live test depends on it.
- Coverage test split into 4 independent tests (size, mapped, unmapped, unmapped_values) sharing one query fixture, so each is provable separately.
- Ambiguity test: one construction -- `dataclasses.replace(live_vocab, alias_index={..., "consulting": (consulting_agreement, sla)})`; the dead first build removed. Precondition assert that unmodified vocab resolves 'consulting' to exactly one owner, so the collision is the only possible cause of None. Same assertion/reasoning as brief.
- Blank-value test kept as-is.

## Break-proofs (all restored; final run 21 passed)
1. `return owners[0] if owners else None`: 1 failed, 20 passed:
   `FAILED test_an_ambiguous_value_resolves_to_nothing_rather_than_picking` -- `assert 'doctype.consulting_agreement' is None`.
2. UPDATE status='active' on doctype.policy_document in bp_concept (1 row) and bp_document_type (1 row): 5 failed, 16 passed.
   Observed coverage {'mapped': 3019, 'unmapped': 32, 'unmapped_values': {'Service': 2, '': 29, 'Indirect Procurement': 1}} (matches brief's 3,019/32).
   Failing: test_every_corpus_value_maps_as_measured[Policy-None], test_a_proposed_concept_never_resolves, test_coverage_mapped_rows_as_measured (3019 == 3016), test_coverage_unmapped_rows_as_measured (32 == 35), test_coverage_names_the_unmapped_values (missing 'Policy': 3).
   Restored both to 'proposed' (verified by query: [('c','proposed'),('d','proposed')]); re-run 21 passed.
3. coverage returns "unmapped_values": {}: 1 failed, 20 passed:
   `FAILED test_coverage_names_the_unmapped_values` -- `assert {} == {'': 29, ...}`; the mapped/unmapped tests stayed green, proving they are independent.

Nothing was written to proc.bp_contract_master (task is read-only). Step 5 psql check not run separately; the grouped counts sum to 3051 in the live test, and no write path exists in the module.
