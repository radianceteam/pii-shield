"""Surrogates must be unique (or deanonymization is ambiguous) and stable."""

from __future__ import annotations

import pytest

from pii_shield.engine.surrogates import SurrogateFactory


def test_uniqueness_beyond_pool_size():
    f = SurrogateFactory(seed="s", use_faker=False)
    made = [f.make("LOCATION", "X") for _ in range(30)]
    assert len(set(made)) == 30


def test_never_returns_the_original():
    f = SurrogateFactory(seed="s", use_faker=False)
    assert f.make("PERSON", "Иван Иванов") != "Иван Иванов"


def test_same_seed_is_reproducible():
    a = [SurrogateFactory(seed="s", use_faker=False).make("PERSON", "X") for _ in range(1)]
    b = [SurrogateFactory(seed="s", use_faker=False).make("PERSON", "X") for _ in range(1)]
    assert a == b


def test_reserve_prevents_collisions_across_calls():
    first = SurrogateFactory(seed="s", use_faker=False)
    already = first.make("PERSON", "A")
    second = SurrogateFactory(seed="s", use_faker=False)
    second.reserve([already])
    assert second.make("PERSON", "B") != already


def test_structured_entities_get_typed_tokens():
    """There is no plausible fake INN — a wrong-checksum number would be worse."""
    f = SurrogateFactory(seed="s", use_faker=False)
    assert f.make("RU_INN", "7707083893").startswith("<RU_INN_")


def test_unknown_entity_falls_back_to_a_token():
    f = SurrogateFactory(seed="s", use_faker=False)
    assert f.make("SOMETHING_NEW", "x").startswith("<SOMETHING_NEW_")


def test_pool_backend_reported():
    assert SurrogateFactory(seed="s", use_faker=False).backend == "pool"


def test_surrogate_never_reuses_a_word_of_the_original():
    """Observed live: "John Smith" drew "Margaret Smith" — the real surname survived.

    The uniqueness check passes there, because the full name differs. A shared token
    is still a partial leak, and common surnames make it likely rather than exotic.
    """
    f = SurrogateFactory(seed="s", locale="en_US", use_faker=False)
    for _ in range(20):
        candidate = f.make("PERSON", "John Smith")
        assert "smith" not in candidate.casefold()
        assert "john" not in candidate.casefold()


def test_short_words_do_not_disqualify_everything():
    """A two-letter token is too common to use as a veto; only real words count."""
    f = SurrogateFactory(seed="s", locale="ru_RU", use_faker=False)
    assert f.make("PERSON", "И О") != ""


@pytest.mark.parametrize("a,b", [
    ("Сидоров", "Сидорова"),      # masculine / feminine
    ("Сидоровым", "Сидорова"),    # instrumental / nominative
    ("Иванов", "Иванова"),
    ("Smith", "Smith"),
])
def test_same_root_catches_inflected_forms(a, b):
    assert SurrogateFactory._same_root(a, b)


@pytest.mark.parametrize("a,b", [
    ("Петру", "Юлия"), ("John", "Joan"), ("Smith", "Smyth"), ("Кузнецов", "Кузьмин"),
])
def test_same_root_leaves_unrelated_names_alone(a, b):
    assert not SurrogateFactory._same_root(a, b)


def test_inflected_surname_does_not_leak():
    """Observed live: "Petru Sidorovym" drew "Yulia Ruslanovna Sidorova".

    The tokens differ by one letter, so an exact comparison passed and the real
    surname went out in a feminine form. Russian inflects, so the guard compares stems.
    """
    f = SurrogateFactory(seed="s", locale="ru_RU", use_faker=False)
    for _ in range(20):
        assert "сидор" not in f.make("PERSON", "Петру Сидоровым").casefold()


def test_a_stand_in_carries_no_title_in_any_locale():
    """Faker decorates names differently per locale, and each way breaks the round trip.

    The stand-in went out as "Ing. Marlis Hofmann B.Eng." and the model wrote back
    "Marlis Hofmann": the exact match found nothing, and the reader was left holding an
    invented person. Measured against de, en and ru, which decorate with "Ing."/"B.Eng.",
    "Dr."/"MD" and "тов." respectively.
    """
    titles = {"md", "dds", "phd", "jr", "sr", "herr", "frau", "pan", "pani"}
    for locale in ("de_DE", "en_US", "ru_RU", "pl_PL"):
        for seed in range(60):
            name = SurrogateFactory(seed=f"s{seed}", locale=locale).make(
                "PERSON", "Пётр Николаевич Васильев"
            )
            assert "." not in name, (locale, name)
            assert not (titles & {w.casefold() for w in name.split()}), (locale, name)


def test_initials_keep_their_dots():
    """The cleaner strips abbreviations; a stand-in that *is* initials is not one."""
    name = SurrogateFactory(seed="x", locale="ru_RU").make("RU_FULL_NAME", "Васильев П.Н.")
    assert "." in name, name
