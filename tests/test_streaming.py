"""Streaming restore, chunked every way that could break it."""

from __future__ import annotations

import pytest

from pii_shield.streaming import StreamDeanonymizer

MAPPING = {"Иван Иванов": "Пётр Сидоров", "ООО Ромашка": "ЗАО Василёк"}


def _run(chunks, mapping=MAPPING) -> str:
    d = StreamDeanonymizer(mapping)
    return "".join(d.feed(c) for c in chunks) + d.flush()


def test_single_chunk():
    assert _run(["Свяжитесь с Иван Иванов сегодня"]) == "Свяжитесь с Пётр Сидоров сегодня"


def test_split_inside_a_surrogate():
    assert _run(["Свяжитесь с Ива", "н Ива", "нов сегодня"]) == "Свяжитесь с Пётр Сидоров сегодня"


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11])
def test_every_chunk_size_gives_the_same_result(size):
    """The property that matters: output must not depend on how the stream was cut."""
    text = "Письмо от Иван Иванов в ООО Ромашка отправлено"
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    assert _run(chunks) == "Письмо от Пётр Сидоров в ЗАО Василёк отправлено"


def test_character_by_character():
    text = "Иван Иванов и ООО Ромашка"
    assert _run(list(text)) == "Пётр Сидоров и ЗАО Василёк"


def test_partial_surrogate_at_end_of_stream_is_released_verbatim():
    """A tail that never completed was ordinary text and must not be swallowed."""
    assert _run(["Готово. Ива"]) == "Готово. Ива"


def test_flush_is_required_to_get_the_tail():
    d = StreamDeanonymizer(MAPPING)
    assert d.feed("текст Ива") == "текст "
    assert d.pending == "Ива"
    assert d.flush() == "Ива"
    assert d.pending == ""


def test_longest_match_wins():
    """A surrogate that is a prefix of another must not be matched first."""
    mapping = {"Иван": "Алексей", "Иван Иванов": "Пётр Сидоров"}
    assert _run(["Иван Иванов"], mapping) == "Пётр Сидоров"
    assert _run(["Иван ушёл"], mapping) == "Алексей ушёл"


def test_surrogate_that_is_a_suffix_of_another_span():
    mapping = {"AB": "x", "XAB": "y"}
    assert _run(["XAB"], mapping) == "y"
    assert _run(["X", "A", "B"], mapping) == "y"


def test_repeated_occurrences():
    assert _run(["Иван Иванов и Иван Иванов"]) == "Пётр Сидоров и Пётр Сидоров"


def test_empty_mapping_is_passthrough():
    assert _run(["любой текст"], {}) == "любой текст"


def test_empty_chunks_are_harmless():
    assert _run(["Иван", "", " Иванов"]) == "Пётр Сидоров"


def test_long_prose_is_not_withheld():
    """Only a partial surrogate may be held back, never ordinary trailing text."""
    d = StreamDeanonymizer(MAPPING)
    out = d.feed("Это обычный длинный текст без каких-либо подстановок вообще")
    assert d.pending == ""
    assert out.endswith("вообще")
