"""Redacting a credential instead of refusing the request.

The default stays refusal, and for a person pasting a key by hand that is the useful
answer. A coding agent is the case it does not fit: the agent resends its whole
conversation on every turn, so one connection string in that history refuses every
following request — and the history only grows, so the session never recovers.

The replacement is one-way on purpose. A name comes back in the answer because the
answer is about that person; a credential must not come back at all.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from pii_shield import Action, BlockedError, EntityRule, Policy, Shield
from pii_shieldd.app import create_app
from pii_shieldd.proxy import ProxyConfig

from .conftest import PATTERN_ONLY_ENTITIES

DSN = "postgres://user:s3cretpass@db.internal:5432/app"
SAMPLE = f"подключение не поднимается: {DSN}"


def policy_for(*, redact: bool) -> Policy:
    policy = Policy.ru_default()
    policy.entities = list(PATTERN_ONLY_ENTITIES)
    policy.redact_credentials = redact
    return policy


# --- the policy decision ----------------------------------------------------
def test_only_credentials_change_their_action():
    """Cards and permanent national identifiers keep refusing."""
    policy = policy_for(redact=True)
    assert policy.action_for("SECRET_CONNECTION_STRING") is Action.MASK
    assert policy.action_for("SECRET_API_KEY") is Action.MASK
    assert policy.action_for("CREDIT_CARD") is Action.BLOCK
    assert policy.action_for("RU_PASSPORT") is Action.BLOCK


def test_off_by_default():
    assert Policy.ru_default().redact_credentials is False
    assert Policy.ru_default().action_for("SECRET_CONNECTION_STRING") is Action.BLOCK


def test_blocking_entities_no_longer_advertises_a_redacted_credential():
    """What the policy reports must match what it does."""
    assert "SECRET_API_KEY" in Policy.ru_default().blocking_entities
    assert "SECRET_API_KEY" not in policy_for(redact=True).blocking_entities
    assert "CREDIT_CARD" in policy_for(redact=True).blocking_entities


def test_an_explicit_allow_is_not_turned_into_a_redaction():
    """The flag relaxes a refusal; it does not override a caller's own decision."""
    policy = policy_for(redact=True)
    policy.rules = [r for r in policy.rules if r.entity != "SECRET_JWT"]
    policy.rules.append(EntityRule(entity="SECRET_JWT", action=Action.ALLOW))
    assert policy.action_for("SECRET_JWT") is Action.ALLOW


def test_a_per_request_language_keeps_the_mode():
    """As with the tier: naming a language must not re-arm the refusal."""
    base = policy_for(redact=True)
    assert Policy.like(base, "en").redact_credentials is True
    assert Policy.like(Policy.ru_default(), "en").redact_credentials is False


# --- the shield -------------------------------------------------------------
def test_the_secret_is_replaced_and_the_call_proceeds():
    shield = Shield(policy_for(redact=True), use_faker=False)
    result = shield.anonymize(SAMPLE)
    assert "s3cretpass" not in result.text
    assert DSN not in result.text
    assert result.credentials_redacted is True


def test_without_the_flag_the_same_text_is_refused():
    shield = Shield(policy_for(redact=False), use_faker=False)
    with pytest.raises(BlockedError) as exc:
        shield.anonymize(SAMPLE)
    assert any(f.entity.startswith("SECRET_") for f in exc.value.findings)


def test_the_placeholder_is_one_way():
    """A model echoing the placeholder must not be handed the real credential."""
    shield = Shield(policy_for(redact=True), use_faker=False)
    result = shield.anonymize(SAMPLE)
    answer = f"проверьте {result.text.split('подключение не поднимается: ')[1]}"
    restored = shield.deanonymize(answer, result.session_id)
    assert "s3cretpass" not in restored
    assert DSN not in restored


def test_credentials_redacted_is_false_when_nothing_was_redacted():
    shield = Shield(policy_for(redact=True), use_faker=False)
    assert shield.anonymize("позвоните на +7 900 123-45-67").credentials_redacted is False


# --- the sidecar ------------------------------------------------------------
@pytest.fixture
def sidecar():
    app = create_app(Shield(policy_for(redact=True), use_faker=False))
    with TestClient(app) as client:
        yield client


def test_sidecar_reports_the_redaction(sidecar):
    body = sidecar.post("/v1/anonymize", json={"text": SAMPLE}).json()
    assert body["credentials_redacted"] is True
    assert "s3cretpass" not in body["text"]


def test_healthz_says_which_mode_the_deployment_runs(sidecar):
    assert sidecar.get("/healthz").json()["redact_credentials"] is True


def test_sidecar_still_refuses_by_default():
    app = create_app(Shield(policy_for(redact=False), use_faker=False))
    with TestClient(app) as client:
        response = client.post("/v1/anonymize", json={"text": SAMPLE})
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["type"] == "pii_shield_blocked"


# --- the proxy --------------------------------------------------------------
class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-test",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "перезапустите пул"}}],
        })

    @property
    def last_text(self) -> str:
        return self.requests[-1].content.decode()


@pytest.fixture
def upstream(monkeypatch) -> Upstream:
    up = Upstream()
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(up.handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("pii_shieldd.proxy.httpx.AsyncClient", factory)
    return up


def proxy_client(*, redact: bool) -> TestClient:
    app = create_app(
        Shield(policy_for(redact=redact), use_faker=False),
        proxy_config=ProxyConfig(upstream="https://upstream.test/v1"),
    )
    return TestClient(app)


def test_proxy_forwards_the_redacted_prompt_and_answers(upstream):
    """The acceptance case: an answer from the model, not a 400."""
    with proxy_client(redact=True) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "gpt-test", "messages": [{"role": "user", "content": SAMPLE}]
        })
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "перезапустите пул"
    assert "s3cretpass" not in upstream.last_text
    assert "подключение не поднимается" in json.loads(upstream.last_text)["messages"][0]["content"]


def test_proxy_refuses_without_the_flag(upstream):
    with proxy_client(redact=False) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "gpt-test", "messages": [{"role": "user", "content": SAMPLE}]
        })
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "pii_shield_blocked"
    assert error["entities"]
    assert not upstream.requests


def test_anthropic_route_redacts_too(upstream):
    with proxy_client(redact=True) as client:
        response = client.post("/v1/messages", json={
            "model": "claude-test", "max_tokens": 16,
            "messages": [{"role": "user", "content": SAMPLE}],
        })
    assert response.status_code == 200
    assert "s3cretpass" not in upstream.last_text


# --- the header that produced a 500 -----------------------------------------
@pytest.mark.parametrize(
    "route,body",
    [
        ("/v1/chat/completions",
         {"model": "m", "messages": [{"role": "user", "content": "привет"}]}),
        ("/v1/messages",
         {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "привет"}]}),
    ],
)
def test_a_non_ascii_authorization_header_is_a_clean_400(upstream, route, body):
    """It used to raise inside httpx and surface as a 500 with no explanation.

    Sent as raw bytes, the way curl does: an HTTP client will not encode a non-ASCII
    header value itself, and the server decodes what arrived as latin-1.
    """
    with proxy_client(redact=True) as client:
        response = client.post(
            route, json=body, headers={"authorization": "Bearer тест".encode()}
        )
    assert response.status_code == 400
    assert "ASCII" in json.dumps(response.json())
    assert not upstream.requests
