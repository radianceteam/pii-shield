"""Live Presidio + spaCy checks.

Skipped unless the NER extra and the Russian model are both installed, so the
default suite stays fast; CI runs this job separately.
"""

from __future__ import annotations

import pytest

from pii_shield import Policy, Shield
from pii_shield.engine.presidio_engine import NerUnavailableError, PresidioDetector


def _ru_model_available() -> bool:
    if not PresidioDetector.available():
        return False
    try:
        PresidioDetector("ru").warm()
    except NerUnavailableError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _ru_model_available(), reason="presidio + ru_core_news_lg not installed"
)


@pytest.fixture(scope="module")
def ru_shield() -> Shield:
    return Shield(Policy.ru_default(), use_faker=False)


def test_detects_a_russian_name_in_an_oblique_case(ru_shield):
    """Russian inflects names; a detector that only knows nominative is useless here."""
    found = ru_shield.detect("Согласовать с Петром Сидоровым завтра")
    assert "PERSON" in {f.entity for f in found}


def test_detects_organization_and_location(ru_shield):
    found = ru_shield.detect("ООО Ромашка находится в Казани")
    entities = {f.entity for f in found}
    assert "ORGANIZATION" in entities
    assert "LOCATION" in entities


def test_technical_prose_is_left_alone(ru_shield):
    """The utility test: a requirements sentence must survive untouched."""
    text = "Система должна отдавать отчёт в PDF по REST API согласно ГОСТ."
    assert ru_shield.anonymize(text).text == text


def test_full_round_trip(ru_shield):
    text = "Согласовать ТЗ с Петром Сидоровым, почта p.sidorov@romashka.ru, ИНН 7707083893."
    result = ru_shield.anonymize(text)
    assert "Сидоров" not in result.text
    assert "p.sidorov@romashka.ru" not in result.text
    assert "7707083893" not in result.text
    assert "ТЗ" in result.text
    assert ru_shield.deanonymize(result.text, result.session_id) == text


def test_pattern_layer_beats_ner_on_the_phone(ru_shield):
    """Presidio scores this ~0.4; the RU pattern scores 0.85 and must win the span."""
    found = ru_shield.detect("тел +7 900 123-45-67")
    assert [f.entity for f in found] == ["RU_PHONE"]
