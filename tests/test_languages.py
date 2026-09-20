"""The language registry — the part that is easy to get subtly and silently wrong."""

from __future__ import annotations

import pytest

from pii_shield import PROFILES, SUPPORTED_LANGUAGES, UnsupportedLanguageError, get_profile
from pii_shield.languages import GLOBAL_NER_ENTITIES, KO_LABEL_MAP, all_national_entities


def test_twenty_four_languages():
    """spaCy 3.8 ships core pipelines for exactly these."""
    assert len(SUPPORTED_LANGUAGES) == 24
    for code in ("ru", "en", "zh", "ja", "ko", "de", "fr", "es", "it", "pl", "uk"):
        assert code in SUPPORTED_LANGUAGES


@pytest.mark.parametrize("code,expected", [
    ("en", "en_core_web_lg"),      # web, not news
    ("zh", "zh_core_web_lg"),      # the other web one — the easy bug
    ("ru", "ru_core_news_lg"),
    ("ja", "ja_core_news_lg"),
    ("ko", "ko_core_news_lg"),
    ("de", "de_core_news_lg"),
])
def test_model_names_are_not_formulaic(code, expected):
    assert get_profile(code).model_name() == expected


def test_model_size_variants():
    assert get_profile("ru").model_name("sm") == "ru_core_news_sm"


def test_unsupported_language_is_refused_not_defaulted():
    """Falling back to English would scan with the wrong model and report it clean."""
    with pytest.raises(UnsupportedLanguageError):
        get_profile("xx")


def test_korean_needs_a_label_override():
    """KLUE labels share nothing with OntoNotes; without this Korean finds nothing."""
    assert get_profile("ko").label_map == KO_LABEL_MAP
    assert get_profile("ja").label_map == {}


@pytest.mark.parametrize("code,locale", [
    ("ru", "ru_RU"), ("en", "en_US"), ("zh", "zh_CN"), ("ja", "ja_JP"), ("ko", "ko_KR"),
])
def test_surrogate_locales(code, locale):
    assert get_profile(code).faker_locale == locale


def test_languages_without_a_faker_locale_are_explicit():
    """Catalan and Norwegian Bokmål have no Faker locale; None must reach the factory."""
    assert get_profile("ca").faker_locale is None
    assert get_profile("nb").faker_locale is None


@pytest.mark.parametrize("code", ["zh", "ja", "ko"])
def test_cjk_lowers_the_minimum_span(code):
    """A full personal name is two characters; a floor of 3 would discard all of them."""
    assert get_profile(code).min_ner_span == 2


@pytest.mark.parametrize("code", ["ru", "en", "de", "fr"])
def test_space_delimited_languages_keep_the_default_floor(code):
    assert get_profile(code).min_ner_span == 3


def test_national_entities_split_between_presidio_and_local():
    """Presidio covers en/es/it/pl; ru and CJK are ours because it covers nothing there."""
    assert "US_SSN" in get_profile("en").presidio_national
    assert get_profile("en").local_national == ()
    assert get_profile("ru").presidio_national == ()
    assert "RU_INN" in get_profile("ru").local_national
    assert "CN_RESIDENT_ID" in get_profile("zh").local_national
    assert "JP_MY_NUMBER" in get_profile("ja").local_national
    assert "KR_RRN" in get_profile("ko").local_national


def test_ner_entities_include_globals_plus_national():
    profile = get_profile("en")
    assert set(GLOBAL_NER_ENTITIES) <= set(profile.ner_entities)
    assert "US_SSN" in profile.ner_entities


def test_universal_allowlist_reaches_every_language():
    for code in SUPPORTED_LANGUAGES:
        assert "API" in get_profile(code).allowlist


def test_allowlists_are_language_specific():
    assert "ТЗ" in get_profile("ru").allowlist
    assert "SSN" in get_profile("en").allowlist
    assert "株式会社" in get_profile("ja").allowlist
    assert "ТЗ" not in get_profile("en").allowlist


def test_national_catalogue_has_no_duplicates():
    catalogue = all_national_entities()
    assert len(catalogue) == len(set(catalogue))


def test_every_profile_is_self_consistent():
    for code, profile in PROFILES.items():
        assert profile.code == code
        assert profile.model_name().startswith(code + "_")
        assert profile.min_ner_span >= 1
