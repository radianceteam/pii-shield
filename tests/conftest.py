"""Shared fixtures.

Every test here runs pattern-only (no spaCy model), so the suite stays runnable in
CI and on a laptop without a 500 MB download. NER behaviour is covered separately by
tests marked ``ner``, which skip when the model is absent.
"""

from __future__ import annotations

import pytest

from pii_shield import Policy, Shield
from pii_shield.engine import NER_ENTITIES

PATTERN_ONLY_ENTITIES = [e for e in Policy().entities if e not in NER_ENTITIES]


@pytest.fixture
def pattern_policy() -> Policy:
    """ru_default() with the NER entities dropped."""
    policy = Policy.ru_default()
    policy.entities = list(PATTERN_ONLY_ENTITIES)
    return policy


@pytest.fixture
def shield(pattern_policy: Policy) -> Shield:
    return Shield(pattern_policy, use_faker=False)


class StubDetector:
    """Stands in for PresidioDetector without loading anything."""

    def __init__(self, findings=None, *, raises: Exception | None = None) -> None:
        self._findings = findings or []
        self._raises = raises

    def warm(self) -> None:
        if self._raises is not None:
            raise self._raises

    def detect(self, text, entities, threshold):
        if self._raises is not None:
            raise self._raises
        return list(self._findings)
