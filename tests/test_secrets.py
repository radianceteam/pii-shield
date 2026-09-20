"""Credential patterns, and the false positives that make or break them."""

from __future__ import annotations

import pytest

from pii_shield.secrets import scan

CASES = [
    ("sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", "SECRET_API_KEY"),
    ("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "SECRET_API_KEY"),
    ("AKIAIOSFODNN7EXAMPLE", "SECRET_API_KEY"),
    (
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dBjftJeZ4CVP_zzSjNLwK27uhbUJU1p1r_wW1gF",
        "SECRET_JWT",
    ),
    ("postgresql://admin:Sup3rS3cret@db.internal:5432/prod", "SECRET_CONNECTION_STRING"),
    ("Authorization: Bearer abcdef0123456789abcdef0123456789", "SECRET_AUTH_HEADER"),
    ("X-Api-Key: 8f3a9c2b1d4e5f6a", "SECRET_AUTH_HEADER"),
]


@pytest.mark.parametrize("text,entity", CASES)
def test_detects(text, entity):
    found = scan(text)
    assert [f.entity for f in found] == [entity], text


def test_private_key_block_is_one_span():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQ==\n-----END RSA PRIVATE KEY-----"
    found = scan(pem)
    assert len(found) == 1
    assert found[0].entity == "SECRET_PRIVATE_KEY"
    assert (found[0].start, found[0].end) == (0, len(pem))


@pytest.mark.parametrize("text", [
    "the bearer of bad news arrived",   # the 20-char floor exists for this
    "Иван Иванов из ООО Ромашка",
    "version sk-2",
    "",
])
def test_no_false_positives(text):
    assert scan(text) == []


def test_overlapping_patterns_report_once():
    """A DSN contains a URL credential; it must be reported as the DSN only."""
    found = scan("redis://user:pw123456@cache:6379/0")
    assert len(found) == 1
    assert found[0].entity == "SECRET_CONNECTION_STRING"


def test_entity_filter_is_honoured():
    assert scan("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", entities={"SECRET_JWT"}) == []


def test_findings_carry_offsets_not_values():
    """A Finding travels to logs; it must not carry the secret itself."""
    found = scan("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")[0]
    assert "ghp_" not in found.model_dump_json()
