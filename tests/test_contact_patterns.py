"""Email and international phone numbers without Presidio.

The cheap tier knows national identifiers for four languages. For a customer writing
in anything else it protected almost nothing — while an email address and an E.164
number are fixed shapes, which is the whole definition of that tier.
"""

from __future__ import annotations

import pytest

from pii_shield import Policy, Shield
from pii_shield.engine.contact_patterns import scan


@pytest.mark.parametrize("text,expected", [
    ("Write to john.smith@acme.co.uk please", "john.smith@acme.co.uk"),
    ("a.b.c@d.example.com", "a.b.c@d.example.com"),
    ("почта@яндекс.рф", "почта@яндекс.рф"),
    ("ivan+tag@пример.рф", "ivan+tag@пример.рф"),
])
def test_email_is_matched_from_its_start(text, expected):
    """A lookbehind covering only letters let a match begin after a dot, so
    "john.smith@..." matched from "smith" and shipped the first name in plain view."""
    found = scan(text)
    assert found and text[found[0].start:found[0].end] == expected


@pytest.mark.parametrize("text", ["Version 3.8@main", "a@b.c", "цена 5@шт"])
def test_not_an_email(text):
    assert not [f for f in scan(text) if f.entity == "EMAIL_ADDRESS"]


@pytest.mark.parametrize("text", [
    "Call +1 415 555 0142", "+49 30 12345678", "+7 (900) 123-45-67",
    "+81 3-1234-5678", "+380 44 123 4567",
])
def test_international_phone(text):
    assert [f.entity for f in scan(text)] == ["PHONE_NUMBER"]


@pytest.mark.parametrize("text", ["версия +3.8", "сумма +1000", "+7 дней", "2+2=4"])
def test_not_a_phone(text):
    assert not [f for f in scan(text) if f.entity == "PHONE_NUMBER"]


def test_entity_filter_is_honoured():
    assert scan("a@example.com", entities={"PHONE_NUMBER"}) == []


def test_pattern_only_protects_a_non_russian_customer():
    shield = Shield(Policy.pattern_only("ru"), use_faker=False)
    text = "Write to john.smith@acme.co.uk or call +49 30 12345678"
    result = shield.anonymize(text)
    assert {"EMAIL_ADDRESS", "PHONE_NUMBER"} == {f.entity for f in result.findings}
    assert "john.smith@acme.co.uk" not in result.text
    assert "+49 30 12345678" not in result.text
    assert shield.deanonymize(result.text, result.session_id) == text


def test_wanting_contact_details_does_not_oblige_installing_presidio():
    policy = Policy(language="ru", entities=["EMAIL_ADDRESS", "PHONE_NUMBER"])
    assert not policy.requires_presidio()
    assert not policy.requires_ner()
