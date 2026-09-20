"""The session store briefly holds real PII, so its bounds are a security property."""

from __future__ import annotations

from pii_shield.session import SessionStore


def _store(**kw):
    clock = kw.pop("clock", None)
    return SessionStore(clock=clock, **kw) if clock else SessionStore(**kw)


def test_roundtrip():
    s = SessionStore()
    sid = s.new_session()
    s.remember(sid, "Пётр Сидоров", "Иван Иванов")
    assert s.surrogate_for(sid, "Пётр Сидоров") == "Иван Иванов"
    assert s.mapping(sid) == {"Иван Иванов": "Пётр Сидоров"}


def test_sessions_are_isolated():
    s = SessionStore()
    a, b = s.new_session(), s.new_session()
    s.remember(a, "real-a", "fake")
    assert s.mapping(b) == {}
    assert s.surrogate_for(b, "real-a") is None


def test_session_ids_are_unguessable():
    s = SessionStore()
    ids = {s.new_session() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(i) >= 16 for i in ids)


def test_expiry():
    now = [1000.0]
    s = SessionStore(ttl_seconds=10, clock=lambda: now[0])
    sid = s.new_session()
    s.remember(sid, "real", "fake")
    now[0] += 11
    assert s.mapping(sid) == {}
    assert len(s) == 0


def test_entries_are_bounded_per_session():
    s = SessionStore(max_entries_per_session=2)
    sid = s.new_session()
    for i in range(5):
        s.remember(sid, f"real{i}", f"fake{i}")
    mapping = s.mapping(sid)
    assert len(mapping) == 2
    assert set(mapping.values()) == {"real3", "real4"}   # oldest evicted


def test_evicting_an_entry_clears_both_directions():
    """A stale forward entry would hand out a surrogate whose reverse map is gone."""
    s = SessionStore(max_entries_per_session=1)
    sid = s.new_session()
    s.remember(sid, "first", "f1")
    s.remember(sid, "second", "f2")
    assert s.surrogate_for(sid, "first") is None


def test_sessions_are_bounded_globally():
    s = SessionStore(max_sessions=3)
    ids = [s.new_session() for _ in range(6)]
    assert len(s) == 3
    assert s.mapping(ids[0]) == {}


def test_mapping_is_a_copy():
    s = SessionStore()
    sid = s.new_session()
    s.remember(sid, "real", "fake")
    s.mapping(sid)["fake"] = "tampered"
    assert s.mapping(sid) == {"fake": "real"}


def test_pop_consumes():
    s = SessionStore()
    sid = s.new_session()
    s.remember(sid, "real", "fake")
    assert s.pop_session(sid) == {"fake": "real"}
    assert s.mapping(sid) == {}


def test_unknown_session_is_inert():
    s = SessionStore()
    s.remember("no-such-session", "real", "fake")
    assert s.mapping("no-such-session") == {}
    assert s.pop_session("no-such-session") == {}


def test_repr_never_renders_contents():
    """A store in a traceback must not print the PII it holds."""
    s = SessionStore()
    sid = s.new_session()
    s.remember(sid, "Пётр Сидоров", "Иван Иванов")
    assert "Сидоров" not in repr(s) and "Иванов" not in repr(s)


def test_clear():
    s = SessionStore()
    s.new_session()
    s.clear()
    assert len(s) == 0
