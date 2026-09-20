"""SWIFT/BIC and LEI.

The reason these are detected at all is not that they are personal data — a BIC
identifies a bank and an LEI a legal entity, and neither is. It is that an
unrecognized BIC does not pass through untouched: NER reads ``DEUTDEFF`` as an
organization and swaps in an invented company name, corrupting a payment instruction.
Claiming the span is what prevents that.
"""

from __future__ import annotations

import pytest

from pii_shield.engine.finance_patterns import scan, valid_bic, valid_lei


# --- BIC --------------------------------------------------------------------
@pytest.mark.parametrize("value", ["DEUTDEFF", "SABRRUMM", "SBOSUS33XXX", "CHASUS33"])
def test_valid_bic(value):
    assert valid_bic(value)


@pytest.mark.parametrize("value", [
    "ABCDEFGH",     # 'EF' is not an ISO 3166 country
    "TESTTEST",     # 'TE' is not either
    "DEUTDEF",      # seven characters
    "deutdeff",     # lowercase
    "DEUT1EFF",     # digit inside the bank code
])
def test_invalid_bic(value):
    assert not valid_bic(value)


def test_bic_needs_context_to_be_confident():
    """'SOMEUSER' is a structurally valid BIC — 'US' sits in positions 5-6.

    That is exactly why a bare match scores below the default threshold: an
    eight-letter uppercase token in technical prose must not be treated as a bank.
    """
    assert valid_bic("SOMEUSER")
    weak = scan("The SOMEUSER constant is defined in config")
    assert [f.entity for f in weak] == ["SWIFT_BIC"]
    assert weak[0].score == pytest.approx(0.4)

    strong = scan("beneficiary bank SOMEUSER")
    assert strong[0].score == pytest.approx(0.9)


@pytest.mark.parametrize("text", [
    "Wire via SWIFT DEUTDEFF today",
    "банк SABRRUMM, счёт",
    "Correspondent DEUTDEFF",
])
def test_context_words_raise_confidence(text):
    found = scan(text)
    assert found and found[0].entity == "SWIFT_BIC"
    assert found[0].score == pytest.approx(0.9)


# --- LEI --------------------------------------------------------------------
def test_valid_lei():
    """ISO 17442 is ISO 7064 MOD 97-10 — a real checksum, so no context is needed."""
    assert valid_lei("5493001KJTIIGC8Y1R12")


@pytest.mark.parametrize("value", [
    "549300VZCRLIVGZJQZ02",       # wrong check digits
    "5493001KJTIIGC8Y1R1",        # nineteen characters
    "5493001KJTIIGC8Y1RAB",       # check positions are not digits
])
def test_invalid_lei(value):
    assert not valid_lei(value)


def test_lei_scores_high_without_context():
    found = scan("LEI 5493001KJTIIGC8Y1R12")
    assert [f.entity for f in found] == ["LEI"]
    assert found[0].score == pytest.approx(0.95)


# --- shared -----------------------------------------------------------------
def test_entity_filter_is_honoured():
    assert scan("SWIFT DEUTDEFF", entities={"LEI"}) == []


def test_empty_text():
    assert scan("") == []


def test_findings_carry_no_values():
    found = scan("SWIFT DEUTDEFF")[0]
    assert "DEUTDEFF" not in found.model_dump_json()


def test_ordinary_prose_is_quiet():
    assert scan("Пожалуйста, согласуйте техническое задание до пятницы.") == []


# --- ABA routing ------------------------------------------------------------
@pytest.mark.parametrize("value", [
    "021000021",   # JPMorgan Chase, New York
    "026009593",   # Bank of America, New York
    "121000248",   # Wells Fargo
    "111000025",   # Bank of America, Texas
])
def test_valid_aba(value):
    from pii_shield.engine.finance_patterns import valid_aba

    assert valid_aba(value)


@pytest.mark.parametrize("value", [
    "123456789",   # fails the 3-7-1 check
    "021000022",   # one digit off
    "991000021",   # 99 is not a Federal Reserve routing symbol
    "02100002",    # eight digits
])
def test_invalid_aba(value):
    from pii_shield.engine.finance_patterns import valid_aba

    assert not valid_aba(value)


def test_aba_needs_context_to_be_confident():
    """Nine digits are cheap; the checksum alone passes about one in ten at random."""
    weak = scan("order id 021000021")
    assert [f.entity for f in weak] == ["ABA_ROUTING"]
    assert weak[0].score == pytest.approx(0.4)

    strong = scan("ABA routing 021000021")
    assert strong[0].score == pytest.approx(0.9)
