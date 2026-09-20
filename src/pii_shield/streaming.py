"""Deanonymization over a stream, where a surrogate can straddle a chunk boundary.

A non-streaming reply is one string, so restoring it is a plain replace. A streamed
reply is not: "Thomas Müller" may arrive as ``"Tho"`` + ``"mas Mü"`` + ``"ller"``, and a
per-chunk replace would emit all three untouched — the caller would see the fake name
and never the real one.

The fix is to withhold the shortest tail that could still be growing into a surrogate,
and emit everything before it. The scanner below is greedy and longest-match-first, so
a surrogate that is a suffix of another one cannot be split across the emit boundary.
"""

from __future__ import annotations


class StreamDeanonymizer:
    """Feed chunks in, get safely-restored text out.

    ``flush()`` must be called at the end of the stream: it releases the withheld
    tail, which would otherwise be silently dropped.
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        # Longest first: with surrogates "Thomas" and "Thomas Müller", matching the
        # short one first would leave " Müller" stranded in the output.
        self._keys = sorted(mapping, key=len, reverse=True)
        self._max_len = max((len(k) for k in mapping), default=0)
        self._pending = ""

    @property
    def pending(self) -> str:
        """Text withheld so far. Non-empty only mid-surrogate."""
        return self._pending

    def feed(self, text: str) -> str:
        if not self._mapping or not text:
            return text
        emitted, self._pending = self._consume(self._pending + text, final=False)
        return emitted

    def flush(self) -> str:
        """Release the tail. Anything still partial was never a surrogate after all."""
        if not self._pending:
            return ""
        emitted, self._pending = self._consume(self._pending, final=True)
        return emitted

    # -- internals ----------------------------------------------------------
    def _consume(self, text: str, *, final: bool) -> tuple[str, str]:
        out: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            match = self._match_at(text, i)
            if match is not None:
                out.append(self._mapping[match])
                i += len(match)
                continue
            if not final and self._could_still_grow(text, i, n):
                break
            out.append(text[i])
            i += 1
        return "".join(out), text[i:]

    def _match_at(self, text: str, i: int) -> str | None:
        for key in self._keys:
            if text.startswith(key, i):
                return key
        return None

    def _could_still_grow(self, text: str, i: int, n: int) -> bool:
        """True if the rest of the buffer is a proper prefix of some surrogate.

        Bounded by the longest surrogate, so a long tail of ordinary prose is never
        withheld — only the handful of characters that could be a partial name.
        """
        rest = n - i
        if rest >= self._max_len:
            return False
        tail = text[i:]
        return any(key.startswith(tail) for key in self._keys)
