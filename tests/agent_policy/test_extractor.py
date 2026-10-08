import json

import pytest

from services.agent_policy import extractor
from services.agent_policy.extraction_schema import ChunkResult, ProposedPolicy, grammar_schema
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS
from tests.agent_policy.test_converter import TAXONOMY, _proposal

PROMPTS = {
    "agent_policy_extract": "T={taxonomy}|R={registry}|D={document_title} v{document_version}|S={sections}",
    "agent_policy_fix": "E={excerpt}|C={constraints}|R={registry}|P={policy}",
}
SECTIONS = [{"reference": "1.1", "heading": "1.1", "text": "1.1 Refunds above $500 need approval.", "start": 0}]


@pytest.fixture(autouse=True)
def _prompts(monkeypatch):
    def load(name):
        if name not in PROMPTS:
            raise extractor.PromptUnavailable(f"prompt unavailable: {name}")
        return PROMPTS[name]
    monkeypatch.setattr(extractor, "load_prompt", load)


class Stub:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.reply


def _walk(node, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            yield path, k, v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def _assert_grammar_safe(schema):
    defs = schema.get("$defs", {})
    for path, key, value in _walk(schema):
        assert key not in ("anyOf", "oneOf", "allOf"), f"union at {path}.{key}"
        if key == "$ref":
            # refs only point into $defs, and no def refers (directly or not) to itself
            assert value.startswith("#/$defs/")
    def reach(name, seen):
        for _, key, value in _walk(defs[name]):
            if key == "$ref":
                target = value.rsplit("/", 1)[-1]
                assert target not in seen, f"recursive $ref {seen + [target]}"
                reach(target, seen + [target])
    for name in defs:
        reach(name, [name])


def _good_reply():
    return ChunkResult(policies=[_proposal()], not_enforceable=[
        {"reference": "1.3", "excerpt": "Staff should be courteous.", "reason": "a duty on people"}]).model_dump_json()


def test_extract_chunk_calls_the_model_as_required():
    stub = Stub(_good_reply())
    result = extractor.extract_chunk(SECTIONS, document={"title": "Finance Payments Policy", "version": 2},
                                     taxonomy=TAXONOMY, registry=REGISTRY, call=stub)
    assert result.policies[0].reference == "1.1" and result.not_enforceable[0].reference == "1.3"
    prompt, kw = stub.calls[0]
    assert kw["think"] is False and kw["use_load_options"] is True
    assert kw["temperature"] == 0 and kw["num_predict"] == 8192 and kw["background"] is True
    assert "model" not in kw  # the default model, AgentNick, and nothing else
    _assert_grammar_safe(kw["format"])
    assert kw["format"] == grammar_schema(ChunkResult)
    assert "D=Finance Payments Policy v2" in prompt
    assert "--- Section 1.1 ---\n1.1 Refunds above $500 need approval." in prompt
    assert "- Finance: General, Refunds and credits, Payments" in prompt
    assert "{" not in prompt.replace("{}", "")  # every placeholder filled


def test_the_raw_pydantic_schema_would_have_had_unions():
    # the guard is not vacuous: without grammar_schema the schema carries anyOf
    assert "anyOf" in json.dumps(ChunkResult.model_json_schema())
    with pytest.raises(AssertionError):
        _assert_grammar_safe(ChunkResult.model_json_schema())


@pytest.mark.parametrize("reply", [None, "", "not json", '{"policies": [{"name": "x"}], "not_enforceable": []}'])
def test_unusable_reply_raises_extraction_error(reply):
    with pytest.raises(extractor.ExtractionError, match="the model did not return a usable answer"):
        extractor.extract_chunk(SECTIONS, document={"title": "D", "version": 1}, taxonomy=TAXONOMY,
                                registry=REGISTRY, call=Stub(reply))


def test_nulls_from_the_model_still_validate():
    raw = json.loads(_good_reply())
    raw["policies"][0]["time_zone"] = None
    raw["policies"][0]["rules"][0]["value_text"] = None
    out = extractor.extract_chunk(SECTIONS, document={"title": "D", "version": 1}, taxonomy=TAXONOMY,
                                  registry=REGISTRY, call=Stub(json.dumps(raw)))
    assert out.policies[0].time_zone is None


def test_missing_prompt_row_is_prompt_unavailable(monkeypatch):
    class Engine:
        def __init__(self, connection_factory=None):
            pass

        def all_prompts(self):
            return [{"promptName": "something_else", "template": "x"}]
    import orchestration.prompt_engine as pe
    monkeypatch.undo()  # drop the autouse load_prompt stub: this test exercises the real loader
    monkeypatch.setattr(pe, "PromptEngine", Engine)
    with pytest.raises(extractor.PromptUnavailable, match="prompt unavailable: agent_policy_extract"):
        extractor.load_prompt("agent_policy_extract")


def test_registry_digest_lists_checkpoints_actions_and_inputs():
    from services.agent_policy import registry
    reg = registry.snapshot_from_rows([
        {"kind": "checkpoint", "name": "tool.call.before", "checkpoint": None, "plain": "before a tool runs",
         "status": "live"},
        {"kind": "checkpoint", "name": "message.send.before", "checkpoint": None, "plain": "before a message is sent",
         "status": "planned"},
        {"kind": "action", "name": "refund.issue", "checkpoint": "tool.call.before", "plain": "x", "status": "live"},
        {"kind": "input", "name": "args.amount", "checkpoint": "tool.call.before", "plain": "amount",
         "value_type": "number", "source": "action", "status": "live"},
        {"kind": "input", "name": "agg.r30", "checkpoint": "tool.call.before", "plain": "refunds in 30 days",
         "value_type": "number", "source": "total:r30", "status": "planned"},
    ])
    text = extractor.registry_digest(reg)
    assert "- tool.call.before: before a tool runs (checked today)" in text
    assert "- message.send.before: before a message is sent (not checked yet)" in text
    assert "  - refund.issue: x" in text
    assert "- args.amount (number, from action): amount" in text
    assert "- agg.r30 (number, from total:r30): refunds in 30 days - not received yet" in text
    assert "At message.send.before" not in text


def test_fix_policy_sends_flipped_examples_as_constraints():
    stub = Stub(_proposal(rules=[{"field": "args.amount", "op": "gte", "value_number": 500}]).model_dump_json())
    # the stage-1 example shape, as the review screen flips it: the agent said none, the
    # condition computes none, and the reviewer says that is wrong -> the outcome
    flipped = [{"input": {"tool.name": "refund.issue", "args.amount": 500}, "agentExpected": "none", "flipped": True},
               {"input": {"tool.name": "refund.issue", "args.amount": 501}, "agentExpected": "approve", "flipped": True}]
    out = extractor.fix_policy(FORM_EXAMPLE, flipped, registry=REGISTRY, taxonomy=TAXONOMY, settings=SETTINGS,
                               call=stub)
    assert isinstance(out, ProposedPolicy) and out.rules[0].op == "gte"
    prompt, kw = stub.calls[0]
    assert 'these inputs must give approve: {"args.amount": 500, "tool.name": "refund.issue"}' in prompt
    assert 'these inputs must give none: {"args.amount": 501, "tool.name": "refund.issue"}' in prompt
    assert FORM_EXAMPLE["source"]["excerpt"] in prompt
    assert kw["think"] is False and kw["use_load_options"] is True
    _assert_grammar_safe(kw["format"])


def test_fix_policy_bad_reply_raises():
    with pytest.raises(extractor.ExtractionError):
        extractor.fix_policy(FORM_EXAMPLE, [], registry=REGISTRY, taxonomy=TAXONOMY, settings=SETTINGS,
                             call=Stub("{}"))


def test_fill_is_one_pass_and_leaves_unknown_words():
    out = extractor._fill("A={a} B={b} C={c}", {"a": "{b}", "b": "x"})
    assert out == "A={b} B=x C={c}"


def test_digest_lists_one_action_per_line_with_its_purpose():
    text = extractor.registry_digest(REGISTRY)
    assert "  - refund.issue: refund.issue" in text  # fixture plain == name
    from services.agent_policy import registry
    reg = registry.snapshot_from_rows([
        {"kind": "checkpoint", "name": "tool.call.before", "checkpoint": None, "plain": "before a tool runs",
         "status": "live"},
        {"kind": "action", "name": "run_email_dispatch", "checkpoint": "tool.call.before",
         "plain": "send a drafted email to a supplier", "status": "live"}])
    assert reg.actions == {"tool.call.before": {"run_email_dispatch"}}
    assert reg.action_plain == {("tool.call.before", "run_email_dispatch"): "send a drafted email to a supplier"}
    assert "  - run_email_dispatch: send a drafted email to a supplier" in extractor.registry_digest(reg)


# ---------------------------------------------------------------- fast failure when busy

def _ps(*models):
    return lambda: {"models": [dict(m) for m in models]}


AGENTNICK = extractor.ollama_client.DEFAULT_MODEL
WANT = extractor.ollama_client.load_options(include_gpu=False)["num_ctx"]


@pytest.mark.parametrize("ps,busy", [
    (_ps({"name": AGENTNICK, "model": AGENTNICK, "context_length": WANT - 4096}), True),   # loaded, other size
    (_ps({"name": AGENTNICK, "model": AGENTNICK, "context_length": WANT}), False),         # loaded, our size
    (_ps(), False),                                                                        # not loaded: a load, not a reload
    (_ps({"name": "someone/else:7b", "model": "someone/else:7b", "context_length": 2048}), False),
    (_ps({"name": AGENTNICK, "model": AGENTNICK}), False),                                 # size not said
    (lambda: None, False),                                                                 # /api/ps unreadable
])
def test_model_busy_reads_api_ps(ps, busy):
    assert extractor.model_busy(ps) is busy


def test_a_busy_model_is_not_called_and_the_chunk_fails_with_the_message():
    stub = Stub(json.dumps({"policies": [], "not_enforceable": []}))
    busy = lambda: extractor.model_busy(_ps({"name": AGENTNICK, "context_length": WANT - 4096}))  # noqa: E731
    with pytest.raises(extractor.ModelBusy, match="The model is busy at a different context size; "
                                                  "try again when it is idle."):
        extractor.extract_chunk(SECTIONS, document={"title": "t", "version": 1}, taxonomy=TAXONOMY,
                                registry=REGISTRY, call=stub, busy=busy)
    assert stub.calls == []
    assert issubclass(extractor.ModelBusy, extractor.ExtractionError)


def test_the_real_model_call_is_guarded_by_default(monkeypatch):
    """The default call (ollama_generate) reads /api/ps first; with the model busy nothing is
    posted. egress.post is made to fail the test, so a broken guard can never reach a model."""
    def no_post(*a, **k):
        raise AssertionError("the model must not be called")
    monkeypatch.setattr(extractor.ollama_client.egress, "post", no_post)
    monkeypatch.setattr(extractor, "read_ps", _ps({"name": AGENTNICK, "context_length": WANT - 4096}))
    with pytest.raises(extractor.ModelBusy):
        extractor.extract_chunk(SECTIONS, document={"title": "t", "version": 1}, taxonomy=TAXONOMY,
                                registry=REGISTRY)
    with pytest.raises(extractor.ModelBusy):
        extractor.fix_policy(FORM_EXAMPLE, [], registry=REGISTRY, taxonomy=TAXONOMY, settings=SETTINGS)


def test_a_stand_in_call_reads_no_api_ps(monkeypatch):
    monkeypatch.setattr(extractor, "read_ps", lambda: (_ for _ in ()).throw(AssertionError("read /api/ps")))
    stub = Stub(json.dumps({"policies": [], "not_enforceable": []}))
    extractor.extract_chunk(SECTIONS, document={"title": "t", "version": 1}, taxonomy=TAXONOMY,
                            registry=REGISTRY, call=stub)
    assert len(stub.calls) == 1
