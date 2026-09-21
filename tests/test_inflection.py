"""Restoring a stand-in that the language declined on its way back.

Reported from a live pod: anonymize turned "Позвони Петру Николаевичу Васильеву" into
a stand-in, the model replied "встреча с Федотовым Аркадием Терентьевичем", and an
exact-match restore left it alone — so the caller read an invented person as a real
one. A failure that looks like success is worse than a visible one.
"""

from __future__ import annotations

import pytest

from pii_shield import Policy, Shield, get_profile
from pii_shield.restore import restore
from pii_shield.streaming import StreamDeanonymizer

PAIRS = [("Ананий Юлианович Игнатьев", "Петру Николаевичу Васильеву")]


# --- the matcher ------------------------------------------------------------
@pytest.mark.parametrize("written,expected", [
    ("Ананий Юлианович Игнатьев", "Петру Николаевичу Васильеву"),
    ("Ананием Юлиановичем Игнатьевым", "Петру Николаевичу Васильеву"),
    ("Ананию Юлиановичу Игнатьеву", "Петру Николаевичу Васильеву"),
    ("Анания Юлиановича Игнатьева", "Петру Николаевичу Васильеву"),
])
def test_every_case_of_the_full_name_is_restored(written, expected):
    assert restore(f"встреча с {written} вчера", PAIRS) == f"встреча с {expected} вчера"


@pytest.mark.parametrize("written,expected", [
    ("Игнатьеву", "Васильеву"),
    ("Игнатьевым", "Васильеву"),
    ("Игнатьевой", "Васильеву"),      # feminine, five characters past the stem
])
def test_a_lone_surname_maps_to_the_matching_word(written, expected):
    """Word i to word i, so a single name does not drag the whole phrase in."""
    assert restore(f"о {written} говорили", PAIRS) == f"о {expected} говорили"


def test_the_whole_name_wins_over_its_parts():
    """The loose pass is only for texts where the safe one found nothing."""
    text = "встреча с Ананием Юлиановичем Игнатьевым, а Игнатов не пришёл"
    assert restore(text, PAIRS) == "встреча с Петру Николаевичу Васильеву, а Игнатов не пришёл"


@pytest.mark.parametrize("text", [
    "Игнат пришёл",                  # shorter than the stand-in less one character
    "Анна Петрова",
    "игнатьевский переулок",
])
def test_unrelated_words_are_left_alone(text):
    assert restore(text, PAIRS) == text


def test_no_pairs_is_a_no_op():
    assert restore("любой текст", []) == "любой текст"


# --- which languages, and which entities -----------------------------------
@pytest.mark.parametrize("code", ["ru", "uk", "pl", "el", "lt"])
def test_inflecting_languages_are_marked(code):
    assert get_profile(code).inflects_names


@pytest.mark.parametrize("code", ["en", "ja", "zh", "nl"])
def test_languages_that_do_not_decline_names_are_not(code):
    assert not get_profile(code).inflects_names


def test_only_free_text_names_are_matched_loosely():
    """A checksummed identifier comes back verbatim or not at all."""
    shield = Shield(Policy.pattern_only("ru"), use_faker=False)
    result = shield.anonymize("ИНН 7707083893")
    assert shield.store.inflectable_pairs(result.session_id) == []


# --- end to end -------------------------------------------------------------
def _ru_available() -> bool:
    from pii_shield.engine.presidio_engine import NerUnavailableError, PresidioDetector

    try:
        PresidioDetector("ru").warm()
    except (NerUnavailableError, Exception):
        return False
    return True


needs_ru = pytest.mark.skipif(not _ru_available(), reason="no Russian pipeline installed")


@needs_ru
def test_a_declined_reply_comes_back_real():
    shield = Shield(Policy.for_language("ru"))
    result = shield.anonymize("Позвони Петру Николаевичу Васильеву")
    pairs = shield.store.inflectable_pairs(result.session_id)
    assert pairs, "the name should have been recorded as declinable"

    surrogate = pairs[0][0]
    first, last = surrogate.split()[0], surrogate.split()[-1]
    reply = f"Я встретился с {surrogate}, передал {first}у документы"
    restored = shield.deanonymize(reply, result.session_id)
    assert surrogate not in restored
    assert first not in restored
    assert "Петру Николаевичу Васильеву" in restored
    assert last  # the surname existed to be matched


@needs_ru
def test_streaming_restores_a_declined_name(monkeypatch):
    """Agents stream, so an exact-only stream would keep the whole bug."""
    shield = Shield(Policy.for_language("ru"))
    result = shield.anonymize("Позвони Петру Николаевичу Васильеву")
    pairs = shield.store.inflectable_pairs(result.session_id)
    surrogate = pairs[0][0]
    words = surrogate.split()
    declined = f"{words[0]}ем {words[1]}ем {words[2]}ым"
    text = f"Я встретился с {declined} вчера и всё обсудил"

    for size in (1, 3, 9, 50):
        d = StreamDeanonymizer(shield.store.mapping(result.session_id), pairs)
        out = "".join(d.feed(text[i:i + size]) for i in range(0, len(text), size)) + d.flush()
        assert "Петру Николаевичу Васильеву" in out, size
        assert words[0] not in out, size
