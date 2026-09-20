"""Live, per-language end-to-end checks against real spaCy pipelines.

Each language skips itself when its pipeline is not installed, so the default suite
stays runnable; CI installs the models in a separate job. These are the tests that
would have caught the failures this module was built to fix — Russian surrogates in
English text, and Korean returning nothing at all because its tagset is not OntoNotes.
"""

from __future__ import annotations

import re
import unicodedata

import pytest

from pii_shield import BlockedError, Policy, Shield
from pii_shield.engine.presidio_engine import NerUnavailableError, PresidioDetector

LANGUAGES = ["ru", "en", "zh", "ja", "ko", "de"]


def _available(language: str) -> bool:
    if not PresidioDetector.available():
        return False
    try:
        PresidioDetector(language).warm()
    except NerUnavailableError:
        return False
    return True


_AVAILABLE = {lang: _available(lang) for lang in LANGUAGES}
_SHIELDS: dict[str, Shield] = {}


def shield_for(language: str) -> Shield:
    if not _AVAILABLE[language]:
        pytest.skip(f"no spaCy pipeline installed for {language!r}")
    if language not in _SHIELDS:
        _SHIELDS[language] = Shield(Policy.for_language(language))
    return _SHIELDS[language]


# --- script helpers ---------------------------------------------------------
def _scripts(text: str) -> set[str]:
    """Unicode script families present, so a surrogate can be checked for language."""
    out = set()
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for script in ("CYRILLIC", "LATIN", "CJK", "HIRAGANA", "KATAKANA", "HANGUL"):
            if name.startswith(script):
                out.add(script)
    return out


EXPECTED_SCRIPTS = {
    "ru": {"CYRILLIC"},
    "en": {"LATIN"},
    "zh": {"CJK"},
    "ja": {"CJK", "HIRAGANA", "KATAKANA"},
    "ko": {"HANGUL"},
}


# --- names are detected and replaced in the right language ------------------
NAME_CASES = {
    "ru": ("Позвонить Петру Сидорову завтра", "Сидоров"),
    "en": ("Please call John Smith tomorrow", "Smith"),
    "ja": ("田中太郎さんに連絡してください", "田中"),
    "ko": ("김철수 씨에게 연락해 주세요", "김철수"),
}


@pytest.mark.parametrize("language", sorted(NAME_CASES))
def test_person_is_detected_and_removed(language):
    shield = shield_for(language)
    text, fragment = NAME_CASES[language]
    result = shield.anonymize(text)
    assert "PERSON" in {f.entity for f in result.findings}, result.findings
    assert fragment not in result.text


@pytest.mark.parametrize("language", sorted(NAME_CASES))
def test_surrogate_is_in_the_language_of_the_text(language):
    """The defect this whole module exists for: an English name drew a Russian one."""
    shield = shield_for(language)
    text, _ = NAME_CASES[language]
    result = shield.anonymize(text)
    person = next(f for f in result.findings if f.entity == "PERSON")
    surrogate = shield.store.surrogate_for(result.session_id, text[person.start : person.end])
    assert surrogate, "no surrogate recorded"
    assert _scripts(surrogate) & EXPECTED_SCRIPTS[language], surrogate
    # And emphatically not the old hard-coded Russian, unless Russian is the language.
    if language != "ru":
        assert "CYRILLIC" not in _scripts(surrogate), surrogate


@pytest.mark.parametrize("language", sorted(NAME_CASES))
def test_round_trip_is_lossless(language):
    shield = shield_for(language)
    text, _ = NAME_CASES[language]
    result = shield.anonymize(text)
    assert shield.deanonymize(result.text, result.session_id) == text


# --- Korean needs the label override ----------------------------------------
def test_korean_finds_entities_at_all():
    """Without the KLUE label map, Presidio maps nothing and Korean silently passes."""
    shield = shield_for("ko")
    found = shield.detect("삼성전자의 김철수 씨에게 연락해 주세요")
    assert {"PERSON", "ORGANIZATION"} & {f.entity for f in found}, found


# --- national identifiers block ---------------------------------------------
BLOCKING_CASES = {
    "zh": ("身份证 11010519491231002X", "CN_RESIDENT_ID"),
    "ja": ("マイナンバー 123456789018", "JP_MY_NUMBER"),
    "en": ("SSN 521-42-8888", "US_SSN"),
    "ru": ("паспорт 4509 123456", "RU_PASSPORT"),
}


@pytest.mark.parametrize("language", sorted(BLOCKING_CASES))
def test_national_id_blocks_the_request(language):
    shield = shield_for(language)
    text, entity = BLOCKING_CASES[language]
    with pytest.raises(BlockedError) as excinfo:
        shield.anonymize(text)
    assert entity in str(excinfo.value)


def test_dashless_us_ssn_with_context_still_blocks():
    """Presidio scores this 0.4; a naive default threshold of 0.5 would miss it."""
    shield = shield_for("en")
    with pytest.raises(BlockedError):
        shield.anonymize("my social security number is 457555462")


def test_a_bare_number_without_context_is_not_an_ssn():
    """The other side of that threshold: no false block on an order reference."""
    shield = shield_for("en")
    assert shield.anonymize("order ref 457555462").text == "order ref 457555462"


# --- phones become plausible local numbers ----------------------------------
PHONE_CASES = {
    "zh": ("电话 13812345678", "CN_PHONE"),
    "ja": ("電話 090-1234-5678", "JP_PHONE"),
    "ko": ("전화 010-1234-5678", "KR_PHONE"),
    "ru": ("тел +7 900 123-45-67", "RU_PHONE"),
}


@pytest.mark.parametrize("language", sorted(PHONE_CASES))
def test_phone_is_replaced_by_a_number_not_a_token(language):
    shield = shield_for(language)
    text, entity = PHONE_CASES[language]
    result = shield.anonymize(text)
    assert entity in {f.entity for f in result.findings}
    finding = next(f for f in result.findings if f.entity == entity)
    surrogate = shield.store.surrogate_for(result.session_id, text[finding.start : finding.end])
    assert surrogate and not surrogate.startswith("<"), surrogate
    assert re.search(r"\d", surrogate), surrogate


# --- technical prose survives -----------------------------------------------
PROSE_CASES = {
    "ru": "Система должна отдавать отчёт в PDF по REST API согласно ГОСТ.",
    "en": "The system must export the report as PDF over the REST API.",
}


@pytest.mark.parametrize("language", sorted(PROSE_CASES))
def test_technical_prose_is_left_alone(language):
    shield = shield_for(language)
    text = PROSE_CASES[language]
    assert shield.anonymize(text).text == text


# --- credentials are language-independent -----------------------------------
@pytest.mark.parametrize("language", LANGUAGES)
def test_credentials_block_in_every_language(language):
    shield = shield_for(language)
    with pytest.raises(BlockedError) as excinfo:
        shield.anonymize("key ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    assert "SECRET_API_KEY" in str(excinfo.value)


# --- per-request language ---------------------------------------------------
def test_a_request_in_another_language_is_scanned_with_that_language():
    """Found on a real server: a per-request policy may name a different language.

    The sidecar accepts a policy per call, so a shield configured for Russian will be
    handed English text. Holding one detector meant that text was scanned by the
    Russian pipeline and came back clean — a silent leak of exactly the kind this
    project exists to prevent.
    """
    shield = shield_for("ru")
    if not _AVAILABLE["en"]:
        pytest.skip("no English pipeline installed")
    found = shield.detect("Wire to John Smith today", Policy(language="en"))
    assert "PERSON" in {f.entity for f in found}


def test_detectors_are_cached_per_language():
    shield = shield_for("ru")
    if not _AVAILABLE["ja"]:
        pytest.skip("no Japanese pipeline installed")
    shield.detect("田中太郎さんに連絡", Policy(language="ja"))
    assert {"ru", "ja"} <= set(shield._detectors)


def test_unavailable_language_fails_closed_rather_than_finding_nothing():
    """A language with no pipeline must raise, not silently scan with nothing."""
    shield = shield_for("ru")
    with pytest.raises(NerUnavailableError):
        shield.detect("Ein Brief an Herrn Müller", Policy(language="mk"))


# --- phone numbers outside the languages with a local pattern ----------------
def test_phone_is_found_in_a_language_without_a_local_pattern():
    """Presidio scores a parseable-but-context-free number at 0.4.

    The default threshold of 0.5 silently discarded every German, French and Japanese
    number; Russian was unaffected only because it has its own RU_PHONE pattern, which
    hid the gap.
    """
    if not _AVAILABLE.get("de", _available("de")):
        pytest.skip("no German pipeline installed")
    shield = Shield(Policy.for_language("de"))
    found = shield.detect("Rufen Sie unter +49 30 12345678 an")
    assert "PHONE_NUMBER" in {f.entity for f in found}


@pytest.mark.parametrize("text", [
    "Bestellnummer 4711 0815",
    "Version 3.8.0 vom 2026-09-20",
    "ISBN 978-3-16-148410-0",
    "Betrag 1234,56 EUR",
])
def test_lower_phone_threshold_does_not_invent_numbers(text):
    if not _AVAILABLE.get("de", _available("de")):
        pytest.skip("no German pipeline installed")
    shield = Shield(Policy.for_language("de"))
    assert "PHONE_NUMBER" not in {f.entity for f in shield.detect(text)}
