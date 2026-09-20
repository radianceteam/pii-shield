"""HTTP surface of the sidecar.

The shield itself is covered elsewhere; what matters here is that the wire contract
does not leak what the in-process API is careful to withhold.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pii_shield import Policy, Shield
from pii_shieldd.app import AUTH_ENV, create_app

from .conftest import PATTERN_ONLY_ENTITIES

SAMPLE = "Договор с ИНН 7707083893, тел +7 900 123-45-67"


@pytest.fixture
def client():
    policy = Policy.ru_default()
    policy.entities = list(PATTERN_ONLY_ENTITIES)
    with TestClient(create_app(Shield(policy, use_faker=False))) as c:
        yield c


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["language"] == "ru"


def test_anonymize_then_deanonymize(client):
    resp = client.post("/v1/anonymize", json={"text": SAMPLE})
    assert resp.status_code == 200
    body = resp.json()
    assert "7707083893" not in body["text"]
    assert body["session_id"]

    back = client.post(
        "/v1/deanonymize", json={"text": body["text"], "session_id": body["session_id"]}
    )
    assert back.status_code == 200
    assert back.json()["text"] == SAMPLE


def test_consume_is_the_default(client):
    body = client.post("/v1/anonymize", json={"text": SAMPLE}).json()
    args = {"text": body["text"], "session_id": body["session_id"]}
    client.post("/v1/deanonymize", json=args)
    second = client.post("/v1/deanonymize", json=args).json()
    assert second["text"] == body["text"]  # mapping already gone


def test_session_continuation_keeps_surrogates_stable(client):
    first = client.post("/v1/anonymize", json={"text": "ИНН 7707083893"}).json()
    second = client.post(
        "/v1/anonymize", json={"text": "снова ИНН 7707083893", "session_id": first["session_id"]}
    ).json()
    assert first["text"].split("ИНН ")[1] in second["text"]


def test_blocked_uses_the_documented_envelope(client):
    """One contract for both endpoints.

    The proxy answered 400 with an OpenAI-shaped error while this endpoint answered
    422 with a different body, so a client written against the README did not catch it.
    """
    secret = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    resp = client.post("/v1/anonymize", json={"text": f"key {secret}"})
    assert resp.status_code == 400
    error = resp.json()["detail"]["error"]
    assert error["type"] == "pii_shield_blocked"
    assert error["entities"] == ["SECRET_API_KEY"]
    assert secret not in resp.text


def test_findings_carry_no_values(client):
    resp = client.post("/v1/anonymize", json={"text": SAMPLE})
    for finding in resp.json()["findings"]:
        assert set(finding) == {"entity", "start", "end", "score", "action"}


def test_per_request_policy_overrides_the_server_default(client):
    resp = client.post("/v1/anonymize", json={
        "text": "ИНН 7707083893",
        "policy": {"entities": ["RU_INN"], "default_action": "mask", "allowlist": []},
    })
    assert "<RU_INN>" in resp.json()["text"]


def test_drop_session(client):
    body = client.post("/v1/anonymize", json={"text": SAMPLE}).json()
    assert client.delete(f"/v1/session/{body['session_id']}").status_code == 204
    back = client.post(
        "/v1/deanonymize", json={"text": body["text"], "session_id": body["session_id"]}
    ).json()
    assert back["text"] == body["text"]


def test_auth_is_enforced_when_the_token_is_set(client, monkeypatch):
    monkeypatch.setenv(AUTH_ENV, "s3cret")
    assert client.post("/v1/anonymize", json={"text": SAMPLE}).status_code == 401
    ok = client.post(
        "/v1/anonymize", json={"text": SAMPLE}, headers={"Authorization": "Bearer s3cret"}
    )
    assert ok.status_code == 200


def test_health_needs_no_token(client, monkeypatch):
    monkeypatch.setenv(AUTH_ENV, "s3cret")
    assert client.get("/healthz").status_code == 200


def test_token_comparison_is_constant_time():
    """``!=`` leaks the matching-prefix length through timing; compare_digest does not."""
    from pii_shieldd.app import token_matches

    assert token_matches("s3cret", "s3cret")
    assert not token_matches("s3cres", "s3cret")
    assert not token_matches("", "s3cret")
    assert not token_matches(None, "s3cret")


def test_proxy_and_library_gates_share_the_comparison():
    import inspect

    from pii_shieldd import proxy

    assert "token_matches" in inspect.getsource(proxy.require_proxy_token)


def test_partial_policy_keeps_the_protective_defaults(client):
    """Found live: {"language": "en"} used to disarm every BLOCK rule.

    A caller sending only a language means "the usual policy, in English", not "a
    policy with no rules". Building a bare Policy from partial input turned a US SSN
    from refused into pseudonymized.
    """
    resp = client.post("/v1/anonymize", json={
        "text": "SSN 521-42-8888", "policy": {"language": "en"},
    })
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["entities"] == ["US_SSN"]


def test_explicit_empty_rules_still_disarms(client):
    """Deliberate is fine; accidental is not."""
    resp = client.post("/v1/anonymize", json={
        "text": "SSN 521-42-8888", "policy": {"language": "en", "rules": []},
    })
    assert resp.status_code == 200


def test_partial_policy_overrides_what_it_names(client):
    resp = client.post("/v1/anonymize", json={
        "text": "ИНН 7707083893", "policy": {"surrogate_language": "de"},
    })
    assert resp.status_code == 200
    assert "7707083893" not in resp.json()["text"]


# --- an unconfigured language must not answer 200 ---------------------------
def test_unknown_field_is_rejected_not_ignored(client):
    """Reported live: {"language": "en"} against a model without that field was
    silently served in the server's own language, and names came back untouched
    with a 200 — which reads as "clean"."""
    resp = client.post("/v1/anonymize", json={"text": "hi", "langauge": "en"})
    assert resp.status_code == 422


def test_language_field_is_honoured(client):
    """The same request now actually selects the language."""
    resp = client.post("/v1/anonymize", json={"text": "ИНН 7707083893", "language": "ru"})
    assert resp.status_code == 200
    assert "7707083893" not in resp.json()["text"]


def test_language_without_a_pipeline_refuses_when_names_are_wanted(pattern_policy):
    """Fails closed rather than returning the text unexamined with a 200."""
    from fastapi.testclient import TestClient

    from pii_shield import Policy, Shield
    from pii_shieldd.app import create_app

    with TestClient(create_app(Shield(Policy.pattern_only("ru"), use_faker=False))) as c:
        resp = c.post("/v1/anonymize", json={
            "text": "Please call John Smith at Microsoft",
            "policy": {"language": "mk"},          # no pipeline installed for this
        })
        assert resp.status_code == 503
        assert "John Smith" not in resp.text


def test_response_says_when_names_were_not_analyzed():
    from fastapi.testclient import TestClient

    from pii_shield import Policy, Shield
    from pii_shieldd.app import create_app

    with TestClient(create_app(Shield(Policy.pattern_only("ru"), use_faker=False))) as c:
        body = c.post("/v1/anonymize", json={"text": "ИНН 7707083893"}).json()
        assert body["names_analyzed"] is False
        assert "7707083893" not in body["text"]
