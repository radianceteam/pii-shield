"""Credential detection layer — see :mod:`pii_shield.secrets.patterns` for provenance."""

from __future__ import annotations

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
    taken: list[tuple[int, int]] = []
    found: list[Finding] = []
    for entity, pattern, score in SECRET_PATTERNS:
        if entities is not None and entity not in entities:
            continue
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            taken.append((start, end))
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
