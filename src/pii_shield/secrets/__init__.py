"""Credential detection layer — see :mod:`pii_shield.secrets.patterns` for provenance."""

from __future__ import annotations

from ..engine.spans import SpanIndex
from ..types import Action, Finding
from .patterns import SECRET_PATTERNS

__all__ = ["scan", "SECRET_PATTERNS"]


def scan(text: str, *, entities: set[str] | None = None) -> list[Finding]:
    """Find credential-shaped spans in *text*.

    Overlaps are resolved by first-writer-wins in ``SECRET_PATTERNS`` order, so a
    connection string is reported once as a connection string rather than also as
    the URL credential and bearer token nested inside it.
    """
    if not text:
        return []
    taken = SpanIndex()
    found: list[Finding] = []
    for entity, pattern, score in SECRET_PATTERNS:
        if entities is not None and entity not in entities:
            continue
        for m in pattern.finditer(text):
            start, end = m.span()
            if not taken.claim(start, end):
                continue
            found.append(
                Finding(
                    entity=entity,
                    start=start,
                    end=end,
                    score=score,
                    action=Action.BLOCK,  # replaced by the caller's policy
                    recognizer=pattern.pattern[:24],
                )
            )
    found.sort(key=lambda f: f.start)
    return found
