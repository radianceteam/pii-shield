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


# --- reading one payload in several blocks ----------------------------------
LONG = "\n\n".join(
    f"[{i}] Обработчик проверяет данные. Клиент: Пётр Николаевич Васильев, ИНН 7707083893"
    for i in range(1, 700)
)


def test_parallel_finds_exactly_what_one_pass_finds():
    assert len(LONG) > 32 * 1024, "the split only happens on a payload worth splitting"
    assert spans(cheap(parallel=4).detect(LONG)) == spans(cheap().detect(LONG))


def test_offsets_survive_the_split():
    """A block's findings are reported against the whole payload, not against the block."""
    for entity, start, end in spans(cheap(parallel=4).detect(LONG)):
        assert LONG[start:end], (entity, start, end)
        if entity == "RU_INN":
            assert LONG[start:end] == "7707083893"


def test_a_small_payload_is_not_split():
    shield = cheap(parallel=4)
    assert shield._split_for_parallel(TEXT) is None
    assert shield._executor is None


def test_a_payload_without_blank_lines_is_not_split():
    shield = cheap(parallel=4)
    assert shield._split_for_parallel("x" * 40_000) is None
