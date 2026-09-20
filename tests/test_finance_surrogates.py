"""A stand-in for a financial code has to be a valid financial code.

Replacing a BIC with an invented company name, or an IBAN with ``<IBAN_CODE_1>``,
does not anonymize a payment instruction — it corrupts it. The recipient's own
validation rejects the document and the failure reads as a bug rather than as policy.
"""

from __future__ import annotations

import pytest

from pii_shield import Policy, Shield
from pii_shield.engine.finance_patterns import valid_aba, valid_bic, valid_lei
from pii_shield.engine.ru_patterns import valid_inn, valid_ogrn
from pii_shield.engine.surrogates import _FAKER_AVAILABLE, SurrogateFactory

pytestmark = pytest.mark.skipif(not _FAKER_AVAILABLE, reason="faker not installed")


@pytest.mark.parametrize("locale", ["ru_RU", "en_US", "de_DE", "ja_JP"])
@pytest.mark.parametrize("entity,validator", [
    ("SWIFT_BIC", valid_bic),
    ("ABA_ROUTING", valid_aba),
    ("LEI", valid_lei),
])
def test_surrogate_passes_its_own_validator(locale, entity, validator):
    factory = SurrogateFactory(seed="s", locale=locale)
    for _ in range(10):
        assert validator(factory.make(entity, "X"))


@pytest.mark.parametrize("entity,validator", [("RU_INN", valid_inn), ("RU_OGRN", valid_ogrn)])
def test_russian_code_surrogates_satisfy_their_checksums(entity, validator):
    factory = SurrogateFactory(seed="s", locale="ru_RU")
    for _ in range(10):
        assert validator(factory.make(entity, "X"))


def test_iban_surrogate_is_not_a_token():
    factory = SurrogateFactory(seed="s", locale="de_DE")
    iban = factory.make("IBAN_CODE", "DE89370400440532013000")
    assert not iban.startswith("<")
    assert iban[:2].isalpha() and len(iban) >= 15


def test_bic_surrogate_matches_the_locale_country():
    """A Russian document should not suddenly route through a German bank code."""
    assert SurrogateFactory(seed="s", locale="ru_RU").make("SWIFT_BIC", "X")[4:6] == "RU"
    assert SurrogateFactory(seed="s", locale="de_DE").make("SWIFT_BIC", "X")[4:6] == "DE"


def test_payment_block_survives_anonymization():
    """End to end: every code in the output is still a well-formed code."""
    policy = Policy.for_language("en")
    shield = Shield(policy, detector=_NoNer())
    text = "SWIFT DEUTDEFF, ABA routing 021000021, LEI 5493001KJTIIGC8Y1R12"
    result = shield.anonymize(text)
    out = result.text
    assert "DEUTDEFF" not in out and "021000021" not in out
    assert valid_bic(out.split("SWIFT ")[1].split(",")[0])
    assert valid_aba(out.split("routing ")[1].split(",")[0])
    assert valid_lei(out.split("LEI ")[1].strip())
    assert shield.deanonymize(out, result.session_id) == text


class _NoNer:
    """Keeps this test independent of whether a spaCy pipeline is installed."""

    supported_entities: frozenset = frozenset()

    def warm(self) -> None:
        return None

    def detect(self, text, entities, threshold):
        return []


def test_card_surrogate_is_a_valid_card():
    """Same rule as the other financial codes: a malformed stand-in is corrupted data."""
    from pii_shield.engine.finance_patterns import valid_card

    factory = SurrogateFactory(seed="s", locale="ru_RU")
    for _ in range(10):
        card = factory.make("CREDIT_CARD", "4111111111111111")
        assert valid_card(card.replace(" ", "").replace("-", "")), card


@pytest.mark.parametrize("original", [
    "DE89370400440532013000", "GB33BUKB20201555555555", "FR1420041010050500013M02606",
])
def test_iban_surrogate_keeps_the_country(original):
    """A German IBAN coming back Russian has changed which country the money goes to."""
    from pii_shield.engine.finance_patterns import valid_iban

    factory = SurrogateFactory(seed="s", locale="ru_RU")
    surrogate = factory.make("IBAN_CODE", original)
    assert surrogate[:2] == original[:2]
    assert valid_iban(surrogate)


@pytest.mark.parametrize("original", ["DEUTDEFF", "SBOSUS33XXX"])
def test_bic_surrogate_keeps_the_country_and_length(original):
    factory = SurrogateFactory(seed="s", locale="ru_RU")
    surrogate = factory.make("SWIFT_BIC", original)
    assert surrogate[4:6] == original[4:6]
    assert len(surrogate) == len(original)
    assert valid_bic(surrogate)
