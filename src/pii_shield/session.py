"""In-memory surrogate map, so an anonymized call can be un-anonymized on the way back.

This store is the one component that briefly holds the very data the shield exists
to remove, so its whole design is about not becoming a second copy of it:

  * memory only — never written to disk, never logged, never in a traceback;
  * TTL-bounded — a session nobody deanonymizes expires instead of accumulating;
  * size-bounded — per session and globally, so a hostile or looping caller cannot
    grow it without limit;
  * one-way by default — ``pop_session`` after deanonymizing is the intended flow.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

DEFAULT_TTL_SECONDS = 3600
DEFAULT_MAX_ENTRIES_PER_SESSION = 2048
DEFAULT_MAX_SESSIONS = 512


@dataclass
class _Session:
    created_at: float
    # surrogate -> original. Insertion-ordered; oldest evicted first when full.
    reverse: dict[str, str] = field(default_factory=dict)
    # original -> surrogate. Keeps one real value mapped to one fake value for the
    # whole session, so the model sees a consistent cast of characters instead of a
    # new name every time the same person is mentioned.
    forward: dict[str, str] = field(default_factory=dict)
    # Stand-ins that may come back in another grammatical form.
    inflectable: set[str] = field(default_factory=set)


class SessionStore:
    """Thread-safe, bounded, expiring store of surrogate mappings."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries_per_session: int = DEFAULT_MAX_ENTRIES_PER_SESSION,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock=time.monotonic,
    ) -> None:
        self._ttl = float(ttl_seconds)
        self._max_entries = int(max_entries_per_session)
        self._max_sessions = int(max_sessions)
        self._clock = clock
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}

    # -- lifecycle ----------------------------------------------------------
    def new_session(self) -> str:
        """Mint an unguessable session id and evict anything stale."""
        sid = secrets.token_urlsafe(16)
        with self._lock:
            self._expire_locked()
            self._sessions[sid] = _Session(created_at=self._clock())
            while len(self._sessions) > self._max_sessions:
                self._sessions.pop(next(iter(self._sessions)))
        return sid

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear(self) -> None:
        """Drop everything — call on profile teardown or explicit lock."""
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            self._expire_locked()
            return len(self._sessions)

    # -- mapping ------------------------------------------------------------
    def remember(
        self, session_id: str, original: str, surrogate: str, *, inflectable: bool = False
    ) -> None:
        with self._lock:
            sess = self._get_locked(session_id)
            if sess is None:
                return
            sess.reverse[surrogate] = original
            sess.forward[original] = surrogate
            if inflectable:
                sess.inflectable.add(surrogate)
            while len(sess.reverse) > self._max_entries:
                oldest = next(iter(sess.reverse))
                stale = sess.reverse.pop(oldest)
                sess.forward.pop(stale, None)
                sess.inflectable.discard(oldest)

    def surrogate_for(self, session_id: str, original: str) -> str | None:
        """Return the surrogate already assigned to *original* in this session."""
        with self._lock:
            sess = self._get_locked(session_id)
            return None if sess is None else sess.forward.get(original)

    def mapping(self, session_id: str) -> dict[str, str]:
        """Snapshot of surrogate -> original. Copy, so callers cannot mutate the store."""
        with self._lock:
            sess = self._get_locked(session_id)
            return {} if sess is None else dict(sess.reverse)

    def inflectable_pairs(self, session_id: str) -> list[tuple[str, str]]:
        """(stand-in, original) for the ones that may return in another form."""
        with self._lock:
            sess = self._get_locked(session_id)
            if sess is None:
                return []
            return [(s, sess.reverse[s]) for s in sess.inflectable if s in sess.reverse]

    def pop_session(self, session_id: str) -> dict[str, str]:
        """Take the mapping and delete it — the intended end of a request round-trip."""
        with self._lock:
            sess = self._sessions.pop(session_id, None)
            return {} if sess is None else dict(sess.reverse)

    # -- internals ----------------------------------------------------------
    def _get_locked(self, session_id: str) -> _Session | None:
        sess = self._sessions.get(session_id)
        if sess is None:
            return None
        if self._clock() - sess.created_at > self._ttl:
            self._sessions.pop(session_id, None)
            return None
        return sess

    def _expire_locked(self) -> None:
        now = self._clock()
        for sid in [s for s, v in self._sessions.items() if now - v.created_at > self._ttl]:
            self._sessions.pop(sid, None)

    def __repr__(self) -> str:  # never render contents
        with self._lock:
            return f"<SessionStore sessions={len(self._sessions)} ttl={self._ttl:g}s>"
