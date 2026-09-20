"""CJK national identifiers.

The checksums are what make this layer worth having: without them, "eighteen digits"
in a Chinese document is an order number as often as an ID card.

Provenance of the fixtures: the Chinese and Japanese positives below are values
published as examples in documentation of those checksum algorithms, so they
corroborate the implementation independently. No published valid Korean RRN was used —
they are live personal data — so the Korean checksum is verified by round trip: a check
digit is computed, then validated, then corrupted and rejected.
"""

from __future__ import annotations

import pytest

from pii_shield.engine.asia_patterns import (
    _KR_WEIGHTS,
    kr_rrn_structure_ok,
    scan,
    valid_cn_resident_id,
    valid_jp_my_number,
    valid_kr_rrn_checksum,
)


# --- China ------------------------------------------------------------------
@pytest.mark.parametrize("value", ["11010519491231002X", "44030419900307721X"])
def test_valid_cn_resident_id(value):
    assert valid_cn_resident_id(value)


@pytest.mark.parametrize("value", [
    "110105194912310021",      # wrong check character
    "1101051949123100",        # too short
    "11010519491231002Y",      # invalid check character
    "11010513491231002X",      # impossible year
    "11010519491331002X",      # month 13
])
def test_invalid_cn_resident_id(value):
    assert not valid_cn_resident_id(value)


def test_cn_id_beats_a_plain_digit_run():
    assert [f.entity for f in scan("身份证 11010519491231002X")] == ["CN_RESIDENT_ID"]
    assert scan("订单号 110105194912310021") == []


def test_cn_phone():
    found = scan("电话 13812345678")
    assert [f.entity for f in found] == ["CN_PHONE"]


# --- Japan ------------------------------------------------------------------
def test_valid_jp_my_number():
    assert valid_jp_my_number("123456789018")


@pytest.mark.parametrize("value", [
    "123456789012",            # wrong check digit
    "12345678901",             # too short
    "000000000000",            # arithmetically valid, never issued
    "111111111111",
])
def test_invalid_jp_my_number(value):
    assert not valid_jp_my_number(value)


def test_jp_my_number_with_separators():
    assert [f.entity for f in scan("マイナンバー 1234-5678-9018")] == ["JP_MY_NUMBER"]


def test_jp_phone():
    assert [f.entity for f in scan("電話 090-1234-5678")] == ["JP_PHONE"]


# --- Korea ------------------------------------------------------------------
def _make_rrn(body12: str) -> str:
    total = sum(int(d) * w for d, w in zip(body12, _KR_WEIGHTS, strict=True))
    return body12 + str((11 - total % 11) % 10)


@pytest.mark.parametrize("body", ["800101112345", "990202234567", "751231398765"])
def test_kr_checksum_round_trip(body):
    rrn = _make_rrn(body)
    assert valid_kr_rrn_checksum(rrn)
    corrupted = rrn[:-1] + str((int(rrn[-1]) + 1) % 10)
    assert not valid_kr_rrn_checksum(corrupted)


def test_kr_structure_accepts_post_2020_numbers():
    """South Korea randomized digits 7-13 in 2020; the old checksum no longer holds.

    Rejecting those would miss every recently issued number, so structure alone is
    still reported — at a lower score, so a caller can exclude it by threshold.
    """
    rrn = "9902022345678"
    assert kr_rrn_structure_ok(rrn)
    found = scan(rrn)
    assert [f.entity for f in found] == ["KR_RRN"]
    assert found[0].score == pytest.approx(0.6)


def test_kr_checksum_valid_number_scores_higher():
    found = scan(_make_rrn("800101112345"))
    assert found[0].score == pytest.approx(0.95)


@pytest.mark.parametrize("value", [
    "9902029345678",           # gender digit 9 is not issued
    "9913022345678",           # month 13
])
def test_kr_structure_rejects(value):
    assert not kr_rrn_structure_ok(value)


def test_kr_phone():
    assert [f.entity for f in scan("전화 010-1234-5678")] == ["KR_PHONE"]


# --- shared behaviour -------------------------------------------------------
def test_entity_filter_is_honoured():
    assert scan("13812345678", entities={"JP_MY_NUMBER"}) == []


def test_empty_text():
    assert scan("") == []


def test_findings_carry_no_values():
    found = scan("身份证 11010519491231002X")[0]
    assert "1101051949" not in found.model_dump_json()


def test_overlapping_ids_are_claimed_once():
    """An 18-digit ID contains shorter phone-shaped runs; it must be reported once."""
    found = scan("11010519491231002X")
    assert len(found) == 1
