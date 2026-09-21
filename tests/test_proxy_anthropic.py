"""The Anthropic Messages route.

A real round trip from the client's side is indistinguishable from a pass-through —
the client sends real values and gets real values back either way. So every test here
asserts on what the *upstream* received, which is the only place the difference shows.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from pii_shield import Policy, Shield
from pii_shieldd.app import create_app
from pii_shieldd.proxy import ProxyConfig

from .conftest import PATTERN_ONLY_ENTITIES


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.response_factory = None

    @property
    def last(self) -> dict:
        return json.loads(self.requests[-1].content)

    @property
    def last_text(self) -> str:
        return self.requests[-1].content.decode()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response_factory(request)


def message(blocks) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "content": blocks, "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })


def sse(events) -> httpx.Response:
    body = "".join(
        f"event: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events
    )
    return httpx.Response(200, content=body.encode(),
                          headers={"content-type": "text/event-stream"})


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
    policy = Policy.for_language("ru")
    policy.entities = list(PATTERN_ONLY_ENTITIES)
    app = create_app(
        Shield(policy, use_faker=False),
        proxy_config=ProxyConfig(upstream="https://upstream.test/v1"),
    )
    with TestClient(app) as c:
        yield c


def post(client, **body):
    payload = {"model": "claude-opus-5", "max_tokens": 64, "messages": []}
    payload.update(body)
    return client.post("/v1/messages", json=payload)


SECRET = "ИНН 7707083893, тел +7 900 123-45-67, IBAN DE89 3704 0044 0532 0130 00"


# --- the three places identifying data lives --------------------------------
def test_system_prompt_is_cleaned(client, upstream):
    """A top-level field, not a message — and where an agent's operator details live."""
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    post(client, system=f"Ты ассистент компании. {SECRET}",
         messages=[{"role": "user", "content": "привет"}])
    assert "7707083893" not in upstream.last_text
    assert "DE89" not in upstream.last_text


def test_system_as_a_block_list_is_cleaned(client, upstream):
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    post(client, system=[{"type": "text", "text": SECRET}],
         messages=[{"role": "user", "content": "привет"}])
    assert "7707083893" not in upstream.last_text


@pytest.mark.parametrize("content", [
    SECRET,
    [{"type": "text", "text": SECRET}],
])
def test_message_content_is_cleaned(client, upstream, content):
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    post(client, messages=[{"role": "user", "content": content}])
    assert "7707083893" not in upstream.last_text


def test_tool_result_is_cleaned(client, upstream):
    """The densest personal data in an agent request: whatever the tool returned."""
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    post(client, messages=[{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": f"карточка клиента: {SECRET}"},
    ]}])
    assert "7707083893" not in upstream.last_text
    assert "+7 900 123-45-67" not in upstream.last_text


def test_tool_result_with_nested_blocks_is_cleaned(client, upstream):
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    post(client, messages=[{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1",
         "content": [{"type": "text", "text": SECRET}]},
    ]}])
    assert "7707083893" not in upstream.last_text


def test_tool_use_input_is_cleaned(client, upstream):
    """Arguments the model extracted are the data it just read, restated."""
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    post(client, messages=[{"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "save",
         "input": {"inn": "7707083893", "nested": {"phone": "+7 900 123-45-67"}}},
    ]}])
    assert "7707083893" not in upstream.last_text
    assert "+7 900 123-45-67" not in upstream.last_text


def test_stop_sequences_are_cleaned(client, upstream):
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    post(client, messages=[{"role": "user", "content": "привет"}],
         stop_sequences=["ИНН 7707083893"])
    assert "7707083893" not in upstream.last_text


def test_images_are_left_alone(client, upstream):
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    post(client, messages=[{"role": "user", "content": [
        {"type": "text", "text": SECRET},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": "AAAA"}},
    ]}])
    parts = upstream.last["messages"][0]["content"]
    assert "7707083893" not in parts[0]["text"]
    assert parts[1]["source"]["data"] == "AAAA"


def test_unrelated_fields_pass_through(client, upstream):
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    post(client, messages=[{"role": "user", "content": "привет"}],
         temperature=0.2, max_tokens=99)
    assert upstream.last["temperature"] == 0.2
    assert upstream.last["max_tokens"] == 99


# --- restoring the answer ---------------------------------------------------
def test_text_is_restored(client, upstream):
    def reply(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        return message([{"type": "text", "text": f"Записал: {sent}"}])

    upstream.response_factory = reply
    body = post(client, messages=[{"role": "user", "content": SECRET}]).json()
    assert body["content"][0]["text"] == f"Записал: {SECRET}"


def test_tool_use_input_is_restored(client, upstream):
    """Otherwise the client's tool runs against a person who does not exist."""
    def reply(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        fake_inn = sent.split("ИНН ")[1].split(",")[0]
        return message([{"type": "tool_use", "id": "t1", "name": "save",
                         "input": {"inn": fake_inn}}])

    upstream.response_factory = reply
    body = post(client, messages=[{"role": "user", "content": SECRET}]).json()
    assert body["content"][0]["input"]["inn"] == "7707083893"


# --- refusals ---------------------------------------------------------------
def test_blocked_uses_the_anthropic_envelope(client, upstream):
    """The SDK parses the response against its own schema; the OpenAI shape raises."""
    upstream.response_factory = lambda r: message([])
    resp = post(client, messages=[{"role": "user",
                                   "content": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}])
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "SECRET_API_KEY" in body["error"]["message"]
    assert upstream.requests == []
    assert "ghp_" not in resp.text


def test_a_blocked_system_prompt_is_not_forwarded(client, upstream):
    upstream.response_factory = lambda r: message([])
    resp = post(client, system="ключ ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                messages=[{"role": "user", "content": "привет"}])
    assert resp.status_code == 400
    assert upstream.requests == []


def test_missing_messages_is_rejected(client):
    resp = client.post("/v1/messages", json={"model": "m", "max_tokens": 1})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


# --- credentials ------------------------------------------------------------
def test_x_api_key_is_forwarded_untouched(client, upstream):
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    client.post("/v1/messages",
                json={"model": "m", "max_tokens": 1,
                      "messages": [{"role": "user", "content": "привет"}]},
                headers={"x-api-key": "sk-ant-caller", "anthropic-version": "2023-06-01",
                         "anthropic-beta": "tools-2024"})
    sent = upstream.requests[-1].headers
    assert sent["x-api-key"] == "sk-ant-caller"
    assert sent["anthropic-version"] == "2023-06-01"
    assert sent["anthropic-beta"] == "tools-2024"


# --- count_tokens -----------------------------------------------------------
def test_count_tokens_cleans_the_prompt(client, upstream):
    """Forwarding this untouched would send the provider exactly what is withheld
    everywhere else, and it would look like the endpoint worked."""
    upstream.response_factory = lambda r: httpx.Response(200, json={"input_tokens": 42})
    resp = client.post("/v1/messages/count_tokens", json={
        "model": "claude-opus-5",
        "system": SECRET,
        "messages": [{"role": "user", "content": SECRET}],
    })
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 42
    assert "7707083893" not in upstream.last_text


def test_count_tokens_blocks_a_credential(client, upstream):
    upstream.response_factory = lambda r: httpx.Response(200, json={"input_tokens": 1})
    resp = client.post("/v1/messages/count_tokens", json={
        "model": "m",
        "messages": [{"role": "user", "content": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}],
    })
    assert resp.status_code == 400
    assert upstream.requests == []


# --- streaming --------------------------------------------------------------
def _collect(raw: str) -> tuple[str, str]:
    """Reassemble text and tool-argument JSON from an SSE stream."""
    text, js = [], []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[len("data:"):].strip())
        except ValueError:
            continue
        delta = event.get("delta") or {}
        if isinstance(delta.get("text"), str):
            text.append(delta["text"])
        if isinstance(delta.get("partial_json"), str):
            js.append(delta["partial_json"])
    return "".join(text), "".join(js)


def _stream_of(sent_text: str, chunk: int = 3):
    """Split the echoed text into small pieces so a stand-in straddles boundaries."""
    events = [
        {"type": "message_start", "message": {"id": "m", "role": "assistant", "content": []}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
    ]
    events += [
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": sent_text[i:i + chunk]}}
        for i in range(0, len(sent_text), chunk)
    ]
    events += [
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    return events


def test_streamed_text_is_restored_across_chunks(client, upstream):
    def reply(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        return sse(_stream_of(f"Записал: {sent}"))

    upstream.response_factory = reply
    resp = post(client, stream=True, messages=[{"role": "user", "content": SECRET}])
    assert resp.status_code == 200
    text, _ = _collect(resp.text)
    assert text == f"Записал: {SECRET}"


def test_streamed_tool_arguments_are_restored(client, upstream):
    """input_json_delta arrives a fragment at a time and still has to be rejoined."""
    def reply(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        fake_inn = sent.split("ИНН ")[1].split(",")[0]
        payload = json.dumps({"inn": fake_inn}, ensure_ascii=False)
        events = [
            {"type": "message_start", "message": {"id": "m", "role": "assistant"}},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "t1", "name": "save", "input": {}}},
        ]
        events += [
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": payload[i:i + 4]}}
            for i in range(0, len(payload), 4)
        ]
        events += [{"type": "content_block_stop", "index": 0}, {"type": "message_stop"}]
        return sse(events)

    upstream.response_factory = reply
    resp = post(client, stream=True, messages=[{"role": "user", "content": SECRET}])
    _, js = _collect(resp.text)
    assert json.loads(js)["inn"] == "7707083893"


def test_event_lines_are_forwarded(client, upstream):
    """The SDK reads `event:` as well as `data:`."""
    upstream.response_factory = lambda r: sse(_stream_of("ok"))
    raw = post(client, stream=True, messages=[{"role": "user", "content": "привет"}]).text
    assert "event: message_start" in raw
    assert "event: message_stop" in raw


def test_a_tail_held_mid_name_is_released_before_the_block_closes(client, upstream):
    """Otherwise the answer loses its last characters whenever it ends on a stand-in."""
    def reply(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        fake = sent.split("ИНН ")[1].split(",")[0]
        return sse(_stream_of(fake, chunk=2))      # ends exactly on the stand-in

    upstream.response_factory = reply
    resp = post(client, stream=True, messages=[{"role": "user", "content": SECRET}])
    text, _ = _collect(resp.text)
    assert text == "7707083893"


def test_streamed_upstream_error_uses_the_anthropic_envelope(client, upstream):
    upstream.response_factory = lambda r: httpx.Response(500, text="boom")
    raw = post(client, stream=True, messages=[{"role": "user", "content": "привет"}]).text
    assert '"type": "error"' in raw


# --- several languages, as for the OpenAI route -----------------------------
@pytest.mark.parametrize("language,text,needle", [
    ("ru", "Оплата по ИНН 7707083893", "7707083893"),
    ("en", "Wire via SWIFT DEUTDEFF today", "DEUTDEFF"),
    ("de", "Rechnung IBAN DE89 3704 0044 0532 0130 00", "DE89 3704"),
    ("ja", "カード 4111 1111 1111 1111", None),           # blocked, never forwarded
])
def test_other_languages(client, upstream, language, text, needle):
    upstream.response_factory = lambda r: message([{"type": "text", "text": "ok"}])
    resp = client.post(
        "/v1/messages",
        json={"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": text}]},
        headers={"X-Pii-Shield-Language": language},
    )
    if needle is None:
        assert resp.status_code == 400
        assert upstream.requests == []
    else:
        assert resp.status_code == 200, resp.text
        assert needle not in upstream.last_text
