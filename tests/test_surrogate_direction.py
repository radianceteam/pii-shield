"""Pseudonymizing *into* a different language than the one being read.

Detection language and surrogate language are separate axes. Reading a Russian
contract with a Russian pipeline but handing the model English stand-ins is a real
configuration: it is what you want when the downstream model is much stronger in
English, or when whoever reviews the outbound traffic does not read the source.
"""

from __future__ import annotations

import unicodedata

import pytest
from pydantic import ValidationError

from pii_shield import Policy, Shield
from pii_shield.engine.presidio_engine import NerUnavailableError, PresidioDetector

TEXT = "Согласовано с Петром Сидоровым вчера"


def _scripts(text: str) -> set[str]:
    out = set()
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for script in ("CYRILLIC", "LATIN", "CJK", "HIRAGANA", "KATAKANA", "HANGUL"):
            if name.startswith(script):
                out.add(script)
    return out


def _ru_available() -> bool:
    if not PresidioDetector.available():
        return False
    try:
        PresidioDetector("ru").warm()
    except NerUnavailableError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _ru_available(), reason="no Russian pipeline installed")


def _person_surrogate(policy: Policy) -> str:
    shield = Shield(policy)
    result = shield.anonymize(TEXT)
    person = next(f for f in result.findings if f.entity == "PERSON")
    original = TEXT[person.start : person.end]
    return shield.store.surrogate_for(result.session_id, original)


@pytest.mark.parametrize("target,script", [
    ("en", "LATIN"), ("zh", "CJK"), ("ja", None), ("ko", "HANGUL"), ("de", "LATIN"),
])
def test_surrogates_follow_the_target_language(target, script):
    policy = Policy.for_language("ru")
    policy.surrogate_language = target
    surrogate = _person_surrogate(policy)
    assert surrogate
    if script:
        assert script in _scripts(surrogate), surrogate
    assert "CYRILLIC" not in _scripts(surrogate), surrogate


def test_detection_language_is_unaffected_by_the_direction():
    """The Russian name must still be found while the stand-in comes out English."""
    policy = Policy.for_language("ru")
    policy.surrogate_language = "en"
    shield = Shield(policy)
    result = shield.anonymize(TEXT)
    assert "PERSON" in {f.entity for f in result.findings}
    assert "Сидоров" not in result.text


def test_round_trip_survives_a_language_change():
    policy = Policy.for_language("ru")
    policy.surrogate_language = "ja"
    shield = Shield(policy)
    result = shield.anonymize(TEXT)
    assert shield.deanonymize(result.text, result.session_id) == TEXT


def test_default_keeps_the_source_language():
    assert "CYRILLIC" in _scripts(_person_surrogate(Policy.for_language("ru")))


def test_exact_locale_overrides_the_language():
    """zh_TW rather than zh_CN, for callers who care about the script variant."""
    policy = Policy.for_language("ru")
    policy.surrogate_language = "en"
    policy.surrogate_locale = "zh_TW"
    assert "CJK" in _scripts(_person_surrogate(policy))


# --- validation (no pipeline needed) ----------------------------------------
def test_unknown_locale_is_rejected():
    with pytest.raises(ValidationError, match="Faker locale"):
        Policy(language="ru", surrogate_locale="xx_YY")


def test_target_language_without_a_locale_is_rejected():
    """Catalan has no Faker locale; silently emitting typed tokens would surprise."""
    with pytest.raises(ValidationError, match="no surrogate locale"):
        Policy(language="ru", surrogate_language="ca")


def test_unknown_target_language_is_rejected():
    with pytest.raises(ValidationError, match="unsupported language"):
        Policy(language="ru", surrogate_language="xx")
