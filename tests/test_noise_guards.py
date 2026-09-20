"""The allowlist and span-length guards.

These exist because an NER model scoring 0.85 on a two-letter abbreviation as an
ORGANIZATION is not a
privacy failure but a correctness one: the requirement text comes back describing a
company that never existed. Both guards are therefore about utility, and both are
deliberately barred from touching the pattern layers.
"""

from __future__ import annotations

import pytest

from pii_shield import Action, Policy, Shield
from pii_shield.types import Finding

from .conftest import StubDetector


@pytest.mark.parametrize("span,allowed", [
    ("ТЗ", True),
    ("REST API", True),          # NER returns the phrase, not the tokens
    ("PDF / JSON", True),
    ("ГОСТ", True),
    ("ООО API", False),          # a real name that merely contains a listed token
    ("Пётр Сидоров", False),
    ("ГОСТ Р", False),
])
def test_is_allowlisted(span, allowed):
    assert Policy().is_allowlisted(span) is allowed


def _shield_with_ner_span(policy, text, start, end, entity="ORGANIZATION"):
    span = Finding(entity=entity, start=start, end=end, score=0.99, action=Action.SURROGATE)
    return Shield(policy, use_faker=False, detector=StubDetector([span]))


def test_allowlisted_span_is_dropped(pattern_policy):
    pattern_policy.entities.append("ORGANIZATION")
    text = "Согласовать ТЗ с заказчиком"
    shield = _shield_with_ner_span(pattern_policy, text, 12, 14)
    assert shield.detect(text) == []


def test_non_allowlisted_span_survives(pattern_policy):
    pattern_policy.entities.append("ORGANIZATION")
    text = "Согласовать с ООО Ромашка"
    shield = _shield_with_ner_span(pattern_policy, text, 14, 25)
    assert [f.entity for f in shield.detect(text)] == ["ORGANIZATION"]


def test_short_span_is_dropped(pattern_policy):
    """Two-character NER spans are abbreviations far more often than names."""
    pattern_policy.entities.append("PERSON")
    text = "Указан ИС в документе"
    shield = _shield_with_ner_span(pattern_policy, text, 7, 9, entity="PERSON")
    assert shield.detect(text) == []


def test_min_ner_span_is_configurable(pattern_policy):
    pattern_policy.entities.append("PERSON")
    pattern_policy.min_ner_span = 1
    text = "Указан ЯЯ в документе"          # not in the allowlist
    shield = _shield_with_ner_span(pattern_policy, text, 7, 9, entity="PERSON")
    assert [f.entity for f in shield.detect(text)] == ["PERSON"]


def test_guards_never_apply_to_the_pattern_layers(shield):
    """A verified INN must survive both guards no matter how short or listed."""
    shield.policy.min_ner_span = 99
    shield.policy.allowlist = ["7707083893"]
    assert [f.entity for f in shield.detect("ИНН 7707083893")] == ["RU_INN"]


def test_secrets_are_not_allowlistable(shield):
    shield.policy.allowlist = ["ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"]
    found = shield.detect("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    assert [f.entity for f in found] == ["SECRET_API_KEY"]


def test_empty_allowlist_disables_the_guard(pattern_policy):
    pattern_policy.entities.append("ORGANIZATION")
    pattern_policy.allowlist = []
    text = "Согласовать ТЗЗ с заказчиком"
    shield = _shield_with_ner_span(pattern_policy, text, 12, 15)
    assert [f.entity for f in shield.detect(text)] == ["ORGANIZATION"]
