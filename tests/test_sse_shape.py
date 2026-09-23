"""Every event the proxy emits is a whole SSE event: a name and its data.

Anthropic's SDK dispatches on the ``event:`` line, so a data line with no name, or
under somebody else's name, is not the event it claims to be. This is not theoretical.
A Russian name is withheld until the block closes — the inflected form cannot be
recognised until its ending arrives — and the withheld tail of a tool call's arguments
used to be written out after the upstream's ``event: content_block_stop`` line, which
made the tail a stop and left the real stop unnamed. The client got empty arguments and
answered "the model's tool call could not be parsed".
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from pii_shield import Action, Finding, Policy, Shield
from pii_shieldd.app import create_app
from pii_shieldd.proxy import ProxyConfig

from .conftest import PATTERN_ONLY_ENTITIES, StubDetector

PROMPT = "Уведомить Петра Васильева о смене статуса."
NAME = "Петра Васильева"
ARGUMENT_PATH = "owners.py"


def upstream_stream(stand_in: str) -> bytes:
    """A tool call whose arguments carry the stand-in, split mid-name."""
    arguments = json.dumps(
        {"path": ARGUMENT_PATH, "content": f"# {stand_in}"}, ensure_ascii=False
    )
    cut = arguments.index(stand_in) + 3

    def event(name, data):
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    out = [
        event("message_start", {"type": "message_start", "message": {
            "id": "m1", "type": "message", "role": "assistant", "model": "m",
            "content": [], "usage": {"input_tokens": 1, "output_tokens": 1}}}),
        event("content_block_start", {"type": "content_block_start", "index": 0,
              "content_block": {"type": "tool_use", "id": "t1", "name": "Write", "input": {}}}),
    ]
    for piece in (arguments[:cut], arguments[cut:cut + 4], arguments[cut + 4:]):
        out.append(event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": piece}}))
    out.append(event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    out.append(event("message_delta", {"type": "message_delta",
                                       "delta": {"stop_reason": "tool_use"}}))
    out.append(event("message_stop", {"type": "message_stop"}))
    return "".join(out).encode()


@pytest.fixture
def client(monkeypatch):
    """A shield that replaces one Russian name, in front of an upstream that uses it.

    A person is the case that matters: the stand-in may come back inflected, so the
    restorer withholds a window of text and releases it only when the block closes.
    """
    policy = Policy.ru_default()
    policy.entities = [*PATTERN_ONLY_ENTITIES, "PERSON"]
    person = Finding(entity="PERSON", start=PROMPT.index(NAME),
                     end=PROMPT.index(NAME) + len(NAME), score=0.99, action=Action.SURROGATE)
    shield = Shield(policy, use_faker=False, detector=StubDetector([person]))

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content.decode())["messages"][0]["content"]
        assert NAME not in sent, "the real name reached the upstream"
        stand_in = sent[len("Уведомить "):].split(" о смене")[0]
        return httpx.Response(
            200, content=upstream_stream(stand_in),
            headers={"content-type": "text/event-stream"},
        )

    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr("pii_shieldd.proxy.httpx.AsyncClient", factory)
    app = create_app(shield, proxy_config=ProxyConfig(upstream="https://upstream.test/v1"))
    with TestClient(app) as c:
        yield c


def stream(client) -> str:
    return client.post("/v1/messages", json={
        "model": "m", "max_tokens": 32, "stream": True,
        "messages": [{"role": "user", "content": PROMPT}],
    }).text


def read_events(raw: str) -> list[tuple[str, dict]]:
    """Parse the stream the way a client does: a name, then its data."""
    events = []
    for block in raw.strip().split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        assert lines and lines[0].startswith("event: "), f"data with no event name: {block!r}"
        assert len(lines) == 2 and lines[1].startswith("data: "), f"malformed event: {block!r}"
        events.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return events


def deltas(events) -> list[dict]:
    return [data for name, data in events if data["type"] == "content_block_delta"]


def test_the_shield_really_withholds_a_tail_here(client):
    """Guards the premise of the rest: without an injected event there is no bug."""
    events = read_events(stream(client))
    assert len(deltas(events)) > 3, "nothing was injected — this fixture stopped testing it"


def test_every_event_carries_its_name(client):
    events = read_events(stream(client))  # asserts the shape of every one
    assert [name for name, _ in events][:2] == ["message_start", "content_block_start"]
    assert events[-1][0] == "message_stop"


def test_a_name_matches_the_type_inside_it(client):
    """A tail under the wrong name is how the tool call was lost."""
    for name, data in read_events(stream(client)):
        assert name == data["type"], f"{name} carries {data['type']}"


def test_the_arguments_arrive_whole_and_restored(client):
    arguments = "".join(
        data["delta"]["partial_json"]
        for data in deltas(read_events(stream(client)))
        if data["delta"].get("type") == "input_json_delta"
    )
    assert json.loads(arguments) == {"path": ARGUMENT_PATH, "content": f"# {NAME}"}
