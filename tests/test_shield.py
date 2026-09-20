"""End-to-end behaviour of the facade, including the failure modes that matter most."""

from __future__ import annotations

import pytest

from pii_shield import (
    Action,
    BlockedError,
    EntityRule,
    Policy,
    RedactionUnavailableError,
    Scope,
    SessionStore,
    Shield,
)
from pii_shield.engine.presidio_engine import NerUnavailableError
from pii_shield.types import Finding

from .conftest import StubDetector

SAMPLE = "Договор с ИНН 7707083893, звонить +7 900 123-45-67"


# --- round trip -------------------------------------------------------------
def test_anonymize_removes_the_original_values(shield):
    result = shield.anonymize(SAMPLE)
    assert "7707083893" not in result.text
    assert "+7 900 123-45-67" not in result.text
    assert result.changed


def test_deanonymize_restores_them(shield):
    result = shield.anonymize(SAMPLE)
    assert shield.deanonymize(result.text, result.session_id) == SAMPLE


def test_deanonymize_of_a_model_answer(shield):
    """The real shape: the model echoes a surrogate, we put the true value back."""
    result = shield.anonymize(SAMPLE)
    surrogate = shield.store.surrogate_for(result.session_id, "+7 900 123-45-67")
    answer = f"Позвоните по номеру {surrogate} завтра."
    restored = shield.deanonymize(answer, result.session_id)
    assert restored == "Позвоните по номеру +7 900 123-45-67 завтра."


def test_consume_empties_the_session(shield):
    result = shield.anonymize(SAMPLE)
    shield.deanonymize(result.text, result.session_id, consume=True)
    assert shield.store.mapping(result.session_id) == {}


def test_same_value_gets_the_same_surrogate_across_calls(shield):
    first = shield.anonymize("ИНН 7707083893 в договоре")
    second = shield.anonymize("Тот же ИНН 7707083893 снова", session_id=first.session_id)
    surrogate = shield.store.surrogate_for(first.session_id, "7707083893")
    assert surrogate in first.text and surrogate in second.text


def test_different_values_get_different_surrogates(shield):
    result = shield.anonymize("ИНН 7707083893 и ИНН 7736207543")
    a = shield.store.surrogate_for(result.session_id, "7707083893")
    b = shield.store.surrogate_for(result.session_id, "7736207543")
    assert a != b and a in result.text and b in result.text


def test_deanonymize_prefers_longer_surrogates():
    """A surrogate that is a prefix of another must not corrupt the substitution."""
    shield = Shield(Policy.off(), use_faker=False)
    shield.store = SessionStore()
    sid = shield.store.new_session()
    shield.store.remember(sid, "REAL_SHORT", "FAKE")
    shield.store.remember(sid, "REAL_LONG", "FAKE_EXTENDED")
    assert shield.deanonymize("FAKE_EXTENDED", sid) == "REAL_LONG"


# --- actions ----------------------------------------------------------------
def test_block_raises_and_names_the_kind_not_the_value(shield):
    with pytest.raises(BlockedError) as excinfo:
        shield.anonymize("ключ sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")
    assert "SECRET_API_KEY" in str(excinfo.value)
    assert "sk-ant" not in str(excinfo.value)


def test_mask_action(pattern_policy):
    pattern_policy.rules = [EntityRule(entity="RU_INN", action=Action.MASK)]
    shield = Shield(pattern_policy, use_faker=False)
    assert "<RU_INN>" in shield.anonymize("ИНН 7707083893").text


def test_hash_action_is_deterministic_and_irreversible(pattern_policy):
    pattern_policy.rules = [EntityRule(entity="RU_INN", action=Action.HASH)]
    shield = Shield(pattern_policy, use_faker=False)
    first = shield.anonymize("ИНН 7707083893").text
    second = shield.anonymize("ИНН 7707083893").text
    assert first == second
    assert "7707083893" not in first
    # Nothing was recorded, so there is nothing to restore.
    assert shield.store.mapping(shield.anonymize("ИНН 7707083893").session_id) == {}


def test_allow_action_leaves_text_untouched(pattern_policy):
    pattern_policy.rules = [EntityRule(entity="RU_INN", action=Action.ALLOW)]
    shield = Shield(pattern_policy, use_faker=False)
    result = shield.anonymize("ИНН 7707083893")
    assert result.text == "ИНН 7707083893"
    assert [f.entity for f in result.findings] == ["RU_INN"]  # still reported


def test_scope_off_short_circuits(pattern_policy):
    pattern_policy.scope = Scope.OFF
    shield = Shield(pattern_policy, use_faker=False)
    result = shield.anonymize(SAMPLE)
    assert result.text == SAMPLE and result.findings == []


# --- failure modes ----------------------------------------------------------
def test_fail_closed_raises_when_detection_breaks(pattern_policy):
    """A model that loads and then blows up mid-request must not yield raw text."""
    pattern_policy.entities.append("PERSON")
    shield = Shield(pattern_policy, use_faker=False, detector=StubDetector())
    shield._detector = StubDetector(raises=RuntimeError("model exploded"))
    with pytest.raises(RedactionUnavailableError):
        shield.anonymize("Иван Иванов")


def test_fail_open_returns_original_text_when_configured(pattern_policy):
    pattern_policy.entities.append("PERSON")
    pattern_policy.fail_closed = False
    shield = Shield(pattern_policy, use_faker=False, detector=StubDetector())
    shield._detector = StubDetector(raises=RuntimeError("model exploded"))
    assert shield.anonymize("Иван Иванов").text == "Иван Иванов"


def test_unexpected_error_while_warming_propagates(pattern_policy):
    """Anything other than a missing model is a real fault: fail at startup."""
    pattern_policy.entities.append("PERSON")
    with pytest.raises(RuntimeError, match="model exploded"):
        Shield(pattern_policy, use_faker=False,
               detector=StubDetector(raises=RuntimeError("model exploded")))


def test_construction_fails_when_ner_is_required_but_missing():
    """The silent-downgrade guard: asking for PERSON with no model must not start."""
    policy = Policy.ru_default()
    with pytest.raises(NerUnavailableError):
        Shield(policy, use_faker=False,
               detector=StubDetector(raises=NerUnavailableError("no model")))


def test_construction_degrades_when_fail_open(pattern_policy):
    pattern_policy.entities.append("PERSON")
    pattern_policy.fail_closed = False
    shield = Shield(
        pattern_policy, use_faker=False,
        detector=StubDetector(raises=NerUnavailableError("no model")),
    )
    assert shield.ner_ready is False


def test_ner_not_required_when_every_ner_entity_is_allowed(pattern_policy):
    """A policy that ALLOWs all NER types needs no model, and must still start."""
    pattern_policy.entities.append("PERSON")
    pattern_policy.rules.append(EntityRule(entity="PERSON", action=Action.ALLOW))
    Shield(pattern_policy, use_faker=False, detector=StubDetector(raises=NerUnavailableError("x")))


# --- merging ----------------------------------------------------------------
def test_pattern_layer_wins_over_ner_on_overlap(pattern_policy):
    """A verified INN outranks an NER guess covering the same span."""
    pattern_policy.entities.append("PERSON")
    text = "ИНН 7707083893"
    ner_span = Finding(entity="PERSON", start=4, end=14, score=0.99, action=Action.SURROGATE)
    shield = Shield(pattern_policy, use_faker=False, detector=StubDetector([ner_span]))
    assert [f.entity for f in shield.detect(text)] == ["RU_INN"]


def test_empty_text_is_inert(shield):
    result = shield.anonymize("")
    assert result.text == "" and result.findings == []


def test_text_without_pii_is_unchanged(shield):
    clean = "Система должна поддерживать экспорт отчётов в формате PDF."
    result = shield.anonymize(clean)
    assert result.text == clean and not result.changed


def test_deanonymize_with_unknown_session_is_a_noop(shield):
    assert shield.deanonymize("текст", "no-such-session") == "текст"
