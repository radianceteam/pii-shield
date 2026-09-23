"""Remembering what was found in a text already read.

An agent resends its whole conversation on every turn, so by the fifth turn the
language model has read the first message five times. Detection is deterministic, so
the answer can be remembered — and the thing to prove is that it is the *same* answer,
and that it is never reused across a policy or a pipeline that would have answered
differently.
"""

from __future__ import annotations

import pytest

from pii_shield import Policy, Shield

from .conftest import PATTERN_ONLY_ENTITIES

TEXT = "Клиент: Пётр Николаевич Васильев, ИНН 7707083893, тел +7 916 123-45-67"


def cheap(**kwargs) -> Shield:
    policy = Policy.ru_default()
    policy.entities = list(PATTERN_ONLY_ENTITIES)
    return Shield(policy, use_faker=False, **kwargs)


def spans(findings) -> list[tuple[str, int, int]]:
    return [(f.entity, f.start, f.end) for f in findings]


def test_the_remembered_answer_is_the_same_answer():
    plain, cached = cheap(), cheap(detection_cache=64)
    first = spans(cached.detect(TEXT))
    assert spans(cached.detect(TEXT)) == first          # served from memory
    assert first == spans(plain.detect(TEXT))           # and equal to a fresh analysis


def test_nothing_is_remembered_unless_asked():
    shield = cheap()
    shield.detect(TEXT)
    assert len(shield._cache) == 0


def test_a_different_policy_is_a_different_question():
    """The catalogue and the thresholds change what is found, so they change the key."""
    shield = cheap(detection_cache=64)
    strict = Policy.ru_default()
    strict.entities = list(PATTERN_ONLY_ENTITIES)
    strict.default_threshold = 0.99
    shield.detect(TEXT)
    assert spans(shield.detect(TEXT, policy=strict)) != spans(shield.detect(TEXT))


def test_the_cache_is_bounded():
    shield = cheap(detection_cache=8)
    for i in range(40):
        shield.detect(f"ИНН 7707083893 — запись {i}")
    assert len(shield._cache) <= 8


def test_the_oldest_entry_goes_first():
    shield = cheap(detection_cache=2)
    shield.detect("первый ИНН 7707083893")
    shield.detect("второй ИНН 7707083893")
    shield.detect("первый ИНН 7707083893")   # touched, so it is no longer the oldest
    shield.detect("третий ИНН 7707083893")
    remembered = [shield.detect(t) for t in ("первый ИНН 7707083893", "третий ИНН 7707083893")]
    assert all(remembered)
    assert len(shield._cache) == 2


def test_an_empty_text_is_not_a_cache_entry():
    shield = cheap(detection_cache=8)
    shield.detect("")
    assert len(shield._cache) == 0


@pytest.mark.parametrize("size", [0, -1])
def test_caching_off_by_configuration(size):
    shield = cheap(detection_cache=size)
    shield.detect(TEXT)
    assert len(shield._cache) == 0
