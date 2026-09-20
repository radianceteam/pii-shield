"""The OpenAI-compatible proxy.

Upstream is a MockTransport, so these tests assert the two things that actually
matter: what the upstream provider *received*, and what the client *got back*.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from pii_shield import Policy, Shield
from pii_shieldd.app import AUTH_ENV, create_app
from pii_shieldd.proxy import ProxyConfig

from .conftest import PATTERN_ONLY_ENTITIES

SAMPLE = "Договор с ИНН 7707083893, тел +7 900 123-45-67"


class Upstream:
    """Records what the proxy forwarded and replies with a scripted response."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.response_factory = None

    @property
    def last_json(self) -> dict:
        return json.loads(self.requests[-1].content)

    @property
    def last_text(self) -> str:
        return self.requests[-1].content.decode()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response_factory(request)


def completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-test",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content},
                 "finish_reason": "stop"}
            ],
        },
    )


def sse(chunks: list[str]) -> httpx.Response:
    body = "".join(
        "data: "
        + json.dumps({"object": "chat.completion.chunk",
                      "choices": [{"index": 0, "delta": {"content": c}}]},
                     ensure_ascii=False)
        + "\n\n"
        for c in chunks
    ) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


@pytest.fixture
def upstream(monkeypatch) -> Upstream:
    up = Upstream()
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(up.handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("pii_shieldd.proxy.httpx.AsyncClient", factory)
    return up


@pytest.fixture
def client(upstream):
    policy = Policy.ru_default()
    policy.entities = list(PATTERN_ONLY_ENTITIES)
    app = create_app(
        Shield(policy, use_faker=False),
        proxy_config=ProxyConfig(upstream="https://upstream.test/v1"),
    )
    with TestClient(app) as c:
        yield c


def _post(client, **overrides):
    body = {"model": "gpt-test", "messages": [{"role": "user", "content": SAMPLE}]}
    body.update(overrides)
    return client.post("/v1/chat/completions", json=body)


# --- the core contract ------------------------------------------------------
def test_upstream_never_sees_the_real_values(client, upstream):
    upstream.response_factory = lambda r: completion("Принято.")
    _post(client)
    assert "7707083893" not in upstream.last_text
    assert "+7 900 123-45-67" not in upstream.last_text


def test_client_gets_the_real_values_back(client, upstream):
    """The model echoes the surrogate; the client must read the original."""
    captured = {}

    def reply(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        captured["phone"] = sent.split("тел ")[1]
        return completion(f"Перезвоните на {captured['phone']}.")

    upstream.response_factory = reply
    body = _post(client).json()
    assert body["choices"][0]["message"]["content"] == "Перезвоните на +7 900 123-45-67."


def test_unrelated_fields_pass_through(client, upstream):
    upstream.response_factory = lambda r: completion("ok")
    _post(client, temperature=0.2, top_p=0.9, seed=7)
    forwarded = upstream.last_json
    assert forwarded["temperature"] == 0.2
    assert forwarded["seed"] == 7
    assert forwarded["model"] == "gpt-test"


def test_tool_results_are_anonymized(client, upstream):
    """Where PII actually arrives in an agent workload."""
    upstream.response_factory = lambda r: completion("ok")
    client.post("/v1/chat/completions", json={
        "model": "gpt-test",
        "messages": [
            {"role": "user", "content": "прочитай файл"},
            {"role": "tool", "tool_call_id": "c1", "content": "клиент: ИНН 7707083893"},
        ],
    })
    assert "7707083893" not in upstream.last_text


def test_assistant_tool_call_arguments_are_anonymized(client, upstream):
    upstream.response_factory = lambda r: completion("ok")
    client.post("/v1/chat/completions", json={
        "model": "gpt-test",
        "messages": [{
            "role": "assistant",
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "save", "arguments": '{"inn": "7707083893"}'},
            }],
        }],
    })
    assert "7707083893" not in upstream.last_text


def test_multimodal_text_parts_are_anonymized_images_untouched(client, upstream):
    upstream.response_factory = lambda r: completion("ok")
    client.post("/v1/chat/completions", json={
        "model": "gpt-test",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "ИНН 7707083893"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ]}],
    })
    parts = upstream.last_json["messages"][0]["content"]
    assert "7707083893" not in parts[0]["text"]
    assert parts[1]["image_url"]["url"] == "https://example.com/a.png"


def test_response_tool_call_arguments_are_restored(client, upstream):
    def reply(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        fake_inn = sent.split("ИНН ")[1].split(",")[0]
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "save", "arguments": json.dumps({"inn": fake_inn})}}],
        }}]})

    upstream.response_factory = reply
    body = _post(client).json()
    args = body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args)["inn"] == "7707083893"


# --- failure modes ----------------------------------------------------------
def test_blocked_request_is_never_forwarded(client, upstream):
    upstream.response_factory = lambda r: completion("ok")
    resp = _post(client, messages=[
        {"role": "user", "content": "ключ ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}
    ])
    assert resp.status_code == 400
    assert upstream.requests == []          # nothing left the process
    error = resp.json()["error"]
    assert error["type"] == "pii_shield_blocked"
    assert "SECRET_API_KEY" in error["message"]
    assert "ghp_" not in resp.text


def test_upstream_error_is_passed_through(client, upstream):
    upstream.response_factory = lambda r: httpx.Response(
        429, json={"error": {"message": "rate limited", "type": "rate_limit_error"}}
    )
    resp = _post(client)
    assert resp.status_code == 429
    assert resp.json()["error"]["type"] == "rate_limit_error"


def test_missing_messages_is_rejected(client):
    resp = client.post("/v1/chat/completions", json={"model": "gpt-test"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "missing_messages"


# --- auth -------------------------------------------------------------------
def test_authorization_is_forwarded_not_consumed(client, upstream):
    upstream.response_factory = lambda r: completion("ok")
    client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "привет"}]},
        headers={"Authorization": "Bearer sk-caller-key"},
    )
    assert upstream.requests[-1].headers["authorization"] == "Bearer sk-caller-key"


def test_proxy_auth_uses_its_own_header(client, upstream, monkeypatch):
    """Authorization belongs to the upstream provider, so the gate is a separate header."""
    upstream.response_factory = lambda r: completion("ok")
    monkeypatch.setenv(AUTH_ENV, "s3cret")
    assert _post(client).status_code == 401
    ok = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "привет"}]},
        headers={"X-Pii-Shield-Token": "s3cret", "Authorization": "Bearer sk-caller-key"},
    )
    assert ok.status_code == 200
    assert upstream.requests[-1].headers["authorization"] == "Bearer sk-caller-key"


def test_models_passthrough(client, upstream):
    upstream.response_factory = lambda r: httpx.Response(
        200, json={"object": "list", "data": [{"id": "gpt-test"}]}
    )
    body = client.get("/v1/models").json()
    assert body["data"][0]["id"] == "gpt-test"


# --- streaming --------------------------------------------------------------
def _sse_contents(raw: str) -> str:
    out = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        chunk = json.loads(payload)
        for choice in chunk.get("choices", []):
            out.append(choice.get("delta", {}).get("content") or "")
    return "".join(out)


def test_streaming_restores_across_chunk_boundaries(client, upstream):
    """The reason StreamDeanonymizer exists: the surrogate arrives in pieces."""
    def reply(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        fake_phone = sent.split("тел ")[1]
        mid = len(fake_phone) // 2
        return sse(["Перезвоните на ", fake_phone[:mid], fake_phone[mid:], " завтра."])

    upstream.response_factory = reply
    resp = _post(client, stream=True)
    assert resp.status_code == 200
    assert _sse_contents(resp.text) == "Перезвоните на +7 900 123-45-67 завтра."


def test_streaming_emits_done(client, upstream):
    upstream.response_factory = lambda r: sse(["привет"])
    assert "data: [DONE]" in _post(client, stream=True).text


def test_streaming_upstream_error_becomes_an_sse_error(client, upstream):
    upstream.response_factory = lambda r: httpx.Response(500, text="boom")
    raw = _post(client, stream=True).text
    assert "upstream_error" in raw
    assert "data: [DONE]" in raw


def test_streaming_blocked_request_is_never_forwarded(client, upstream):
    upstream.response_factory = lambda r: sse(["ok"])
    resp = _post(client, stream=True, messages=[
        {"role": "user", "content": "ключ ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}
    ])
    assert resp.status_code == 400
    assert upstream.requests == []


# --- hygiene ----------------------------------------------------------------
def test_session_is_dropped_after_a_request(client, upstream):
    upstream.response_factory = lambda r: completion("ok")
    _post(client)
    from pii_shieldd.app import get_shield

    assert len(get_shield().store) == 0


def test_session_is_dropped_after_a_stream(client, upstream):
    upstream.response_factory = lambda r: sse(["ok"])
    _ = _post(client, stream=True).text   # drain the stream so the finally block runs
    from pii_shieldd.app import get_shield

    assert len(get_shield().store) == 0


@pytest.mark.parametrize("value", ["Bearer ", "  ", "Bearer", ""])
def test_empty_authorization_is_not_forwarded(client, upstream, value):
    """An unset shell variable yields "Authorization: Bearer ", which httpx refuses.

    Forwarding it verbatim surfaced to the caller as a 502 "upstream unreachable" —
    a confusing answer to what is simply a malformed request.
    """
    upstream.response_factory = lambda r: completion("ok")
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "привет"}]},
        headers={"Authorization": value} if value else {},
    )
    assert resp.status_code == 200
    assert "authorization" not in upstream.requests[-1].headers


# --- per-request language (multi-tenant) ------------------------------------
def test_language_header_overrides_the_server_default(client, upstream):
    """A platform serving users in several languages needs this.

    The proxy otherwise reads whatever language the deployment was configured with,
    so every other user's text is scanned by the wrong model and comes back clean.
    The OpenAI body has no field for it, so it travels in a header.
    """
    upstream.response_factory = lambda r: completion("ok")
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "ИНН 7707083893"}]},
        headers={"X-Pii-Shield-Language": "ru"},
    )
    assert resp.status_code == 200
    assert "7707083893" not in upstream.last_text


def test_surrogate_language_header(client, upstream):
    upstream.response_factory = lambda r: completion("ok")
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "ИНН 7707083893"}]},
        headers={"X-Pii-Shield-Surrogate-Language": "de"},
    )
    assert resp.status_code == 200
    assert "7707083893" not in upstream.last_text


def test_unknown_language_header_is_a_clean_400(client, upstream):
    upstream.response_factory = lambda r: completion("ok")
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Pii-Shield-Language": "xx"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_language"
    assert upstream.requests == []


def test_no_header_keeps_the_server_default(client, upstream):
    upstream.response_factory = lambda r: completion("ok")
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "ИНН 7707083893"}]},
    )
    assert resp.status_code == 200
    assert "7707083893" not in upstream.last_text


def test_language_header_stays_in_the_daemons_tier(upstream):
    """A pattern-only daemon must not be upgraded into needing a language model."""
    from pii_shield import Policy, Shield
    from pii_shieldd.app import create_app

    upstream.response_factory = lambda r: completion("ok")
    app = create_app(
        Shield(Policy.pattern_only("ru"), use_faker=False),
        proxy_config=ProxyConfig(upstream="https://upstream.test/v1"),
    )
    with TestClient(app) as c:
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "ИНН 7707083893"}]},
            headers={"X-Pii-Shield-Language": "en"},
        )
    assert resp.status_code == 200, resp.text
