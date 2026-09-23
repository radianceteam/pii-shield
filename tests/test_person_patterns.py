"""Russian full names found without a language model.

Every case here is a shape the `ru_core_news_lg` pipeline was measured getting wrong:
a name in a form field, a table cell or a label yields nothing or — worse — yields all
of the name but its first word, so part of a real person goes to the provider while
the response looks anonymized.
"""

from __future__ import annotations

import pytest

from pii_shield import Policy, Shield
from pii_shield.engine import person_patterns

# The shapes an agent's traffic is actually made of.
FORM_SHAPES = [
    "ФИО: Пётр Николаевич Васильев",
    "Клиент: Пётр Николаевич Васильев, тел +7 916 123-45-67",
    "| Пётр Николаевич Васильев | менеджер |",
    "Пётр Николаевич Васильев согласовал бюджет.",
    "Васильев Пётр Николаевич, менеджер",
]


def spans(text: str) -> list[str]:
    return [text[f.start : f.end] for f in person_patterns.scan(text)]


@pytest.mark.parametrize("text", FORM_SHAPES)
def test_the_whole_name_is_found_in_form_shapes(text):
    assert spans(text) == [
        "Васильев Пётр Николаевич" if text.startswith("Васильев") else "Пётр Николаевич Васильев"
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Передай Петру Николаевичу Васильеву документы", "Петру Николаевичу Васильеву"),
        ("Спасибо Анне Сергеевне Кузнецовой за помощь", "Анне Сергеевне Кузнецовой"),
        ("Письмо от Ольги Ильиничны Петровой", "Ольги Ильиничны Петровой"),
        ("Договор с Петром Николаевичем Васильевым", "Петром Николаевичем Васильевым"),
    ],
)
def test_case_endings_do_not_hide_a_name(text, expected):
    """The patronymic is matched by stem: every case ending replaces its last letter."""
    assert spans(text) == [expected]


def test_a_label_in_front_is_not_swallowed():
    """Without the stop-list the surname-first shape eats the word before it."""
    assert spans("Директор Пётр Николаевич распорядился") == ["Пётр Николаевич"]
    assert spans("Клиент Васильев Пётр Николаевич") == ["Васильев Пётр Николаевич"]


def test_initials_are_found_in_both_orders():
    assert spans("Васильев П.Н. подписал акт") == ["Васильев П.Н."]
    assert spans("Акт подписан П. Н. Васильевым") == ["П. Н. Васильевым"]


@pytest.mark.parametrize(
    "text",
    [
        "Земляничная поляна рядом с домом",   # adjective ending in -ична
        "Совет директоров и Пётр решили",     # a first name with no patronymic
        "Обсудили на встрече с Иваном Петровым",
        "Москва, улица Ленина, дом 5",
    ],
)
def test_what_must_not_match(text):
    assert spans(text) == []


def test_a_name_without_a_patronymic_is_left_to_the_model():
    """This layer is a floor under the model, not a replacement for it."""
    assert person_patterns.scan("Позвони Ивану Петрову") == []


# --- through the shield -----------------------------------------------------
@pytest.mark.parametrize("text", FORM_SHAPES)
def test_the_cheap_tier_no_longer_leaks_these(text):
    """The 30 MB deployment, with no pipeline at all."""
    shield = Shield(Policy.pattern_only("ru"))
    result = shield.anonymize(text)
    for word in ("Пётр", "Николаевич", "Васильев"):
        assert word not in result.text


def test_the_cheap_tier_still_admits_it_does_not_analyze_names():
    """Finding a patronymic is not the same as finding names, and must not claim to be."""
    shield = Shield(Policy.pattern_only("ru"))
    assert shield.anonymize("ФИО: Пётр Николаевич Васильев").names_analyzed is False


def test_the_stand_in_keeps_the_shape_and_the_gender():
    shield = Shield(Policy.pattern_only("ru"))
    female = shield.anonymize("Анна Сергеевна Кузнецова").text
    assert person_patterns.looks_female(female), female
    male = shield.anonymize("Пётр Николаевич Васильев").text
    assert not person_patterns.looks_female(male), male
    form = shield.anonymize("| Васильев Пётр Николаевич |").text
    assert person_patterns.shape_of(form.strip("| ")) == "family_first", form
    initials = shield.anonymize("Васильев П.Н. подписал").text
    assert "." in initials.split()[1], initials


def test_the_name_is_restored_like_any_other():
    shield = Shield(Policy.pattern_only("ru"))
    result = shield.anonymize("ФИО: Пётр Николаевич Васильев")
    stand_in = result.text.removeprefix("ФИО: ")
    restored = shield.deanonymize(f"Ответ про {stand_in}", result.session_id)
    assert restored == "Ответ про Пётр Николаевич Васильев"


def test_a_language_with_no_name_layer_is_unaffected():
    from pii_shield.engine import name_scan

    assert name_scan("de", "Thomas Müller anrufen") == []
