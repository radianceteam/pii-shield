"""Russian structured identifiers: the checksums are the whole value of this layer."""

from __future__ import annotations

import pytest

from pii_shield.engine.ru_patterns import scan, valid_inn, valid_ogrn, valid_snils


# Real, published identifiers of public companies — these are registry data, not PII.
@pytest.mark.parametrize("value", ["7707083893", "7736207543", "500100732259"])
def test_valid_inn_accepts_real_identifiers(value):
    assert valid_inn(value)


@pytest.mark.parametrize("value", ["7707083894", "770708389", "77070838931", "abcdefghij", ""])
def test_valid_inn_rejects_corrupted(value):
    assert not valid_inn(value)


def test_valid_snils_roundtrip():
    assert valid_snils("11223344595")
    assert not valid_snils("11223344596")


def test_valid_snils_rejects_unissued_low_range():
    """001-001-998 and below are never issued, and their checksum is undefined."""
    assert not valid_snils("00100199800")


def test_valid_ogrn_both_lengths():
    assert valid_ogrn("1027700132195")       # 13-digit OGRN (Sberbank)
    assert not valid_ogrn("1027700132196")
    assert not valid_ogrn("102770013219")


def test_scan_finds_inn_and_ogrn_separately():
    found = scan("ООО Ромашка, ИНН 7707083893, ОГРН 1027700132195")
    assert [f.entity for f in found] == ["RU_INN", "RU_OGRN"]


def test_ten_digit_number_with_bad_checksum_is_not_an_inn():
    """The reason this layer beats a bare digit regex."""
    assert scan("сумма 1234567890 рублей") == []


def test_passport_requires_context_word():
    assert [f.entity for f in scan("паспорт 4509 123456")] == ["RU_PASSPORT"]
    assert scan("артикул 4509 123456 на складе") == []


def test_phone_variants():
    found = scan("звоните +7 (900) 123-45-67 или 8 900 765 43 21")
    assert [f.entity for f in found] == ["RU_PHONE", "RU_PHONE"]


def test_longer_number_does_not_yield_nested_identifier():
    """A 20-digit account must not also surface as the 13-digit OGRN inside it."""
    found = scan("счёт 12345678901234567890")
    assert [f.entity for f in found] == ["RU_BANK_ACCOUNT"]


def test_entity_filter_is_honoured():
    assert scan("ИНН 7707083893", entities={"RU_PHONE"}) == []


def test_empty_text():
    assert scan("") == []
