"""When a stand-in cannot be put back, the answer says so.

The shield cannot restore what a model has changed past recognition: a name folded
into an identifier, or written in another alphabet. What it must never do is hand that
back as though it were real, so what could not be restored is reported.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pii_shield import Policy, Shield
from pii_shieldd.app import create_app

from .conftest import PATTERN_ONLY_ENTITIES


def cheap() -> Shield:
    policy = Policy.ru_default()
    policy.entities = list(PATTERN_ONLY_ENTITIES)
    return Shield(policy, use_faker=False)


def test_nothing_is_reported_when_everything_came_back():
    shield = cheap()
    result = shield.anonymize("ИНН 7707083893")
    mapping = shield.store.mapping(result.session_id)
    restored = shield.deanonymize(result.text, result.session_id)
    assert restored == "ИНН 7707083893"
    assert shield.stand_ins_left_in(restored, mapping) == []


def test_a_stand_in_folded_into_an_identifier_is_reported():
    """`owner = феликс_германович_хохлов` — lower case, joined, and still a stand-in."""
    mapping = {"Феликс Германович Хохлов": "Пётр Николаевич Васильев"}
    answer = "owner = феликс_германович_хохлов"
    assert Shield.stand_ins_left_in(answer, mapping) == ["Феликс Германович Хохлов"]


def test_a_declined_stand_in_is_reported():
    mapping = {"Ларионова Василиса Алексеевна": "Пётр Николаевич Васильев"}
    assert Shield.stand_ins_left_in("передал Ларионовау документы", mapping)


def test_a_short_word_does_not_raise_a_false_alarm():
    """Four letters collide with ordinary words, which is why the restore skips them."""
    mapping = {"Лев Иванов": "Пётр Васильев"}
    assert Shield.stand_ins_left_in("лев вышел на охоту", mapping) == []


def test_the_sidecar_hands_the_report_to_the_caller():
    shield = cheap()
    app = create_app(shield)
    with TestClient(app) as client:
        body = client.post("/v1/anonymize", json={"text": "ИНН 7707083893"}).json()
        stand_in = next(iter(shield.store.mapping(body["session_id"])))
        answer = client.post(
            "/v1/deanonymize",
            json={"text": f"смотри {stand_in.lower()}", "session_id": body["session_id"]},
        ).json()
    assert answer["unrestored"] == [stand_in]


@pytest.mark.parametrize("text", ["", "ничего не осталось"])
def test_an_empty_or_clean_answer_reports_nothing(text):
    assert Shield.stand_ins_left_in(text, {"Феликс Хохлов": "Пётр Васильев"}) == []


def test_the_proxy_says_so_in_a_header(monkeypatch):
    """A client reading an answer through the proxy can tell without asking."""
    import json

    import httpx

    from pii_shieldd.proxy import ProxyConfig

    shield = cheap()
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)["messages"][0]["content"]
        captured["stand_in"] = sent.split("ИНН ")[1].split()[0]
        # The model answers with the stand-in changed — lower case here, the way one
        # gets folded into an identifier. An unchanged one would simply be restored,
        # even inside a word, because that pass matches text rather than tokens.
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": f"inn_{captured['stand_in'].lower()}"}}],
        })

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("pii_shieldd.proxy.httpx.AsyncClient", factory)
    app = create_app(shield, proxy_config=ProxyConfig(upstream="https://upstream.test/v1"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "ИНН 7707083893"}]
        })
    assert response.status_code == 200
    assert response.headers.get("x-pii-shield-unrestored") == "1"
