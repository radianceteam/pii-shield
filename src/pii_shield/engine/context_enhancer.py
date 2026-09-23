"""Presidio's context scoring, without the scan it does for every finding.

Presidio raises a finding's confidence when a supporting word stands near it — "тел"
beside a number, "passport" beside an identifier. To do that it has to know which
token the finding starts at, and it looks that up by walking the token list from the
beginning, once per finding. On a paragraph that is nothing. On an agent's turn it is
the single most expensive thing the shield does: 100 KB of traffic produced about a
thousand findings over twenty-five thousand tokens, and the walk alone cost 2.3 of the
4.9 seconds — more than the language model it is scoring the output of.

The token offsets are ascending, so the same question is a binary search. The answer is
identical, including the error Presidio raises when a finding cannot be placed; only
the cost changes, from a quarter of a million comparisons to a dozen.
"""

from __future__ import annotations

from bisect import bisect_right


def build_context_enhancer():
    """A drop-in replacement for Presidio's default context enhancer.

    Built lazily, because importing it imports Presidio, which the pattern-only tier
    deliberately does without.
    """
    from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer

    class IndexedLemmaContextAwareEnhancer(LemmaContextAwareEnhancer):
        """The same scoring, with the token lookup indexed instead of scanned."""

        def __init__(self) -> None:
            super().__init__()
            self._cached_key: tuple[int, int, int] | None = None
            self._cached_ends: list[int] = []

        def _ends(self, tokens, tokens_indices: list[int]) -> list[int]:
            """Where each token ends, built once per analyzed text.

            Keyed by the identity of the offsets list — one analyze() call reuses the
            same list for every finding — with the length and last offset checked, so
            a recycled id cannot hand back somebody else's index.
            """
            last = tokens_indices[-1] if tokens_indices else -1
            key = (id(tokens_indices), len(tokens_indices), last)
            if key != self._cached_key:
                self._cached_ends = [
                    index + len(token) for index, token in zip(tokens_indices, tokens, strict=False)
                ]
                self._cached_key = key
            return self._cached_ends

        def _find_index_of_match_token(self, word, start, tokens, tokens_indices) -> int:
            # Presidio takes the first token that either begins at `start` or covers
            # it. Offsets ascend and tokens are non-empty, so that is the first token
            # whose end is past `start`.
            ends = self._ends(tokens, tokens_indices)
            index = bisect_right(ends, start)
            if index >= len(ends):
                raise ValueError(
                    "Did not find word '" + word + "' "
                    "in the list of tokens although it "
                    "is expected to be found"
                )
            return index

    return IndexedLemmaContextAwareEnhancer()
