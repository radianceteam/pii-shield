"""What restoration is allowed to write, and what it must leave alone.

Every case here is output this project actually produced. The first two are the ones
that matter: a restore that writes the wrong real name is worse than one that writes
nothing, because nothing is visible to the reader and wrong is not.
"""

from __future__ import annotations

import pytest

from pii_shield import Policy, Shield
from pii_shield.engine import person_patterns
from pii_shield.restore import restore
from pii_shield.streaming import StreamDeanonymizer


def _ru_available() -> bool:
    from pii_shield.engine.presidio_engine import PresidioDetector

    try:
        PresidioDetector("ru").warm()
    except Exception:
        return False
    return True


# The tests that draw a stand-in need a pipeline to find the name first; the rules
# above them are pure functions and run everywhere.
needs_ru = pytest.mark.skipif(not _ru_available(), reason="no Russian pipeline installed")

STAND_IN = "Ларионова Василиса Алексеевна"
REAL = "Петру Николаевичу Васильеву"


def test_a_restored_name_is_not_matched_again():
    """The exact pass wrote "Васильеву"; the stem of "Василиса" must not touch it."""
    text = f"Я встретился с {STAND_IN}, передал Ларионовау документы"
    out = restore(text, [(STAND_IN, REAL)], exact={STAND_IN: REAL})
    assert out.startswith(f"Я встретился с {REAL},")
    assert "Николаевичу Николаевичу" not in out
    assert "Васильеву" in out


def test_a_name_written_in_another_order_is_left_alone():
    """Word i of one name is not word i of the other, so there is nothing to map."""
    pairs = [("Гуляев Архип Юлианович", "Анна Сергеевна Кузнецова")]
    out = restore("**Архип Юлианович, пришлите смету**", pairs, exact={})
    assert out == "**Архип Юлианович, пришлите смету**"


def test_a_different_person_with_a_similar_surname_is_untouched():
    pairs = [("Ананий Юлианович Игнатьев", "Пётр Николаевич Васильев")]
    text = "встреча с Ананием Юлиановичем Игнатьевым, а Игнатов не пришёл"
    assert restore(text, pairs, exact={}) == (
        "встреча с Пётр Николаевич Васильев, а Игнатов не пришёл"
    )


def test_a_declined_fragment_still_comes_back():
    """The ordinary case: named in full once, then by the first name alone."""
    pairs = [("Иванна Олеговна Горбунова", "Анна Сергеевна Кузнецова")]
    out = restore("Иванной Олеговной Горбуновой, потом Иванне", pairs, exact={})
    assert out == "Анна Сергеевна Кузнецова, потом Анна"


def test_streaming_protects_what_it_wrote_too():
    mapping = {STAND_IN: REAL}
    stream = StreamDeanonymizer(mapping, [(STAND_IN, REAL)])
    out = "".join(stream.feed(chunk) for chunk in (f"с {STAND_IN}", ", далее")) + stream.flush()
    assert REAL in out
    assert "Николаевичу Николаевичу" not in out


# --- the other half: not choosing a stand-in that cannot come back ----------
@needs_ru
def test_the_stand_in_is_written_in_the_same_order_as_the_original():
    shield = Shield(Policy.for_language("ru"))
    for text, expected in (
        ("Позвони Петру Николаевичу Васильеву", "given_first"),
        ("ФИО: Васильев Пётр Николаевич", "family_first"),
    ):
        result = shield.anonymize(text)
        stand_in = next(iter(shield.store.mapping(result.session_id)))
        assert person_patterns.shape_of(stand_in) == expected, stand_in


@needs_ru
def test_no_part_of_a_stand_in_is_too_short_to_restore():
    """"Лука" comes back as "Луку", and four letters cannot be told from another name."""
    shield = Shield(Policy.for_language("ru"))
    for _ in range(40):
        result = shield.anonymize("Позвони Петру Николаевичу Васильеву")
        stand_in = next(iter(shield.store.mapping(result.session_id)))
        assert all(len(word) > 4 for word in stand_in.split()), stand_in


@needs_ru
def test_a_stand_in_carries_no_title():
    """Faker offers "тов. Некрасова Майя" and "Wendy King MD"; neither is a name."""
    shield = Shield(Policy.for_language("ru"))
    for _ in range(40):
        result = shield.anonymize("Позвони Петру Николаевичу Васильеву")
        stand_in = next(iter(shield.store.mapping(result.session_id)))
        assert "тов." not in stand_in.casefold()
        assert not stand_in.split()[-1].isupper()


@needs_ru
def test_the_round_trip_survives_a_declined_answer():
    """End to end, on the shape that used to fail one draw in twenty."""
    shield = Shield(Policy.for_language("ru"))
    for _ in range(25):
        result = shield.anonymize("Позвони Петру Николаевичу Васильеву")
        stand_in = next(iter(shield.store.mapping(result.session_id)))
        first = stand_in.split()[0]
        reply = f"Я встретился с {stand_in}, передал {first}у документы"
        restored = shield.deanonymize(reply, result.session_id, consume=True)
        assert "Петру Николаевичу Васильеву" in restored
        assert first not in restored
        assert stand_in not in restored


# --- which language decides that a stand-in can be declined ------------------
@needs_ru
def test_an_english_stand_in_in_russian_text_is_not_matched_loosely():
    """The loose pass matches what was written out, and Latin names do not decline."""
    policy = Policy.for_language("ru")
    policy.surrogate_language = "en"
    shield = Shield(policy)
    result = shield.anonymize("Позвони Петру Николаевичу Васильеву")
    assert shield.store.mapping(result.session_id)
    assert shield.store.inflectable_pairs(result.session_id) == []


@needs_ru
def test_the_request_decides_it_and_not_the_daemon():
    """A per-request policy carries its own languages, as the tier already does."""
    shield = Shield(Policy.for_language("en"))
    ru = Policy.for_language("ru")
    result = shield.anonymize("Позвони Петру Николаевичу Васильеву", policy=ru)
    assert shield.store.inflectable_pairs(result.session_id)
