from src.services.requirement_session import RequirementSession


class _FakeRedis:
    def __init__(self):
        self.store = {}
    def set(self, k, v):
        self.store[k] = v
    def get(self, k):
        return self.store.get(k)


def _new():
    return RequirementSession(session_id="S1", requirement_id="REQ-1", created_by="alice")


def test_redis_key_and_defaults():
    s = _new()
    assert s.redis_key == "requirement_session:S1"
    assert s.status == "gathering"
    assert s.requirement == {}


def test_apply_fields_ignores_empty_values():
    s = _new()
    s.apply_fields({"title": "Laptops", "quantity": 10, "unit": None, "category": ""})
    assert s.requirement == {"title": "Laptops", "quantity": 10}


def test_add_turn_and_status_transitions():
    s = _new()
    s.add_turn("user", "I need laptops")
    assert s.turn_history[-1]["role"] == "user"
    assert s.turn_history[-1]["content"] == "I need laptops"
    s.mark_complete()
    assert s.status == "complete"
    s.mark_abandoned()
    assert s.status == "abandoned"


def test_save_load_round_trip():
    r = _FakeRedis()
    s = _new()
    s.apply_fields({"title": "Laptops"})
    s.add_turn("user", "hi")
    s.save(r)
    loaded = RequirementSession.load("S1", r)
    assert loaded is not None
    assert loaded.requirement == {"title": "Laptops"}
    assert loaded.turn_history[-1]["content"] == "hi"
    assert loaded.requirement_id == "REQ-1"


def test_none_client_degrades_gracefully():
    s = _new()
    s.save(None)  # must not raise
    assert RequirementSession.load("S1", None) is None
