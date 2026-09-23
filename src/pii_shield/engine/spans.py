"""Keeping track of the text already claimed by a finding.

Every layer needs the same thing: "has anything taken this stretch of text yet?"
Written the obvious way — a list of intervals and a scan over it — that question costs
a comparison against every span found so far, which is fine for a paragraph and
crippling for an agent's context. On 8 MB carrying six thousand findings it came to
22 million comparisons, about half the total running time.

The intervals never overlap each other, so they can be kept sorted by start and the
question answered by looking at the two neighbours.
"""

from __future__ import annotations

from bisect import bisect_right, insort


class SpanIndex:
    """Non-overlapping [start, end) intervals, asked about in O(log n)."""

    __slots__ = ("_spans",)

    def __init__(self) -> None:
        self._spans: list[tuple[int, int]] = []

    def overlaps(self, start: int, end: int) -> bool:
        """True if any claimed interval intersects [start, end)."""
        index = bisect_right(self._spans, (start, end))
        if index and self._spans[index - 1][1] > start:
            return True
        return index < len(self._spans) and self._spans[index][0] < end

    def add(self, start: int, end: int) -> None:
        insort(self._spans, (start, end))

    def claim(self, start: int, end: int) -> bool:
        """Take the interval if it is free. Returns whether it was taken."""
        if self.overlaps(start, end):
            return False
        self.add(start, end)
        return True

    def __len__(self) -> int:
        return len(self._spans)
