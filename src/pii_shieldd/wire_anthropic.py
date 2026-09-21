"""Walking an Anthropic Messages payload.

The shape differs from OpenAI's in ways that matter for what gets protected, not
just in field names:

  * ``system`` is a top-level field, not a message. An agent's system prompt routinely
    carries the operator's name, address and company details, so skipping it would
    leave the most stable identifying text in the whole request untouched.
  * ``content`` is a string **or** a list of typed blocks.
  * ``tool_result`` blocks carry whatever the agent's tool returned — a database row,
    an email, a customer record. In practice this is the densest personal data in the
    request; missing it means protecting nothing.
  * ``tool_use.input`` is a JSON object whose string values are arguments the model
    extracted, which is to say the identifying data it just read.

On the way back the same two places need restoring. If the model calls a tool with a
stand-in name, the client's tool executes against a person who does not exist.
"""

from __future__ import annotations

from typing import Any

from pii_shield import Shield

# Blocks whose payload is binary or opaque. Nothing here looks inside them, and
# pretending otherwise would be worse than saying so.
_OPAQUE_BLOCK_TYPES = frozenset({"image", "document", "thinking", "redacted_thinking"})


def _map_strings(value: Any, fn) -> Any:
    """Apply *fn* to every string in a JSON-shaped structure, keeping the shape."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_map_strings(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_strings(item, fn) for key, item in value.items()}
    return value


class _Anonymizer:
    """Carries the session across every string in one request."""

    def __init__(self, shield: Shield, policy=None, session_id: str | None = None) -> None:
        self.shield = shield
        self.policy = policy
        self.session_id = session_id
        self.changed = False
        self.names_analyzed = True

    def __call__(self, text: str) -> str:
        if not text:
            return text
        result = self.shield.anonymize(text, session_id=self.session_id, policy=self.policy)
        self.session_id = result.session_id or self.session_id
        self.changed = self.changed or result.changed
        self.names_analyzed = self.names_analyzed and result.names_analyzed
        return result.text


def _walk_block(block: Any, fn) -> Any:
    """Anonymize one content block according to its type."""
    if not isinstance(block, dict):
        return block
    kind = block.get("type")
    if kind in _OPAQUE_BLOCK_TYPES:
        return block
    if kind == "text" and isinstance(block.get("text"), str):
        return {**block, "text": fn(block["text"])}
    if kind == "tool_use" and isinstance(block.get("input"), (dict, list)):
        # Arguments the model extracted from the conversation — the identifying data
        # it just read, restated in structured form.
        return {**block, "input": _map_strings(block["input"], fn)}
    if kind == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            return {**block, "content": fn(content)}
        if isinstance(content, list):
            return {**block, "content": [_walk_block(inner, fn) for inner in content]}
    return block


def _walk_content(content: Any, fn) -> Any:
    if isinstance(content, str):
        return fn(content)
    if isinstance(content, list):
        return [_walk_block(block, fn) for block in content]
    return content


def anonymize_request(
    shield: Shield, payload: dict, policy=None, session_id: str | None = None
) -> tuple[dict, str | None, bool, bool]:
    """Return the rewritten payload, the session handle, changed, names_analyzed."""
    anonymizer = _Anonymizer(shield, policy, session_id)
    out = dict(payload)

    if "system" in payload:
        out["system"] = _walk_content(payload["system"], anonymizer)

    if isinstance(payload.get("messages"), list):
        out["messages"] = [
            {**m, "content": _walk_content(m["content"], anonymizer)}
            if isinstance(m, dict) and "content" in m
            else m
            for m in payload["messages"]
        ]

    # A stop sequence carrying a name would both leak it and fail to stop: the model
    # is reading stand-ins, so it will emit the stand-in, not the original.
    if isinstance(payload.get("stop_sequences"), list):
        out["stop_sequences"] = [
            anonymizer(s) if isinstance(s, str) else s for s in payload["stop_sequences"]
        ]

    # Tool *definitions* are schema, not data, and their descriptions are written by
    # the developer rather than drawn from the conversation. Left alone deliberately.
    return out, anonymizer.session_id, anonymizer.changed, anonymizer.names_analyzed


def deanonymize_response(shield: Shield, body: Any, session_id: str) -> Any:
    """Restore real values in a non-streamed Messages response."""
    if not isinstance(body, dict) or not isinstance(body.get("content"), list):
        return body

    def restore(text: str) -> str:
        return shield.deanonymize(text, session_id)

    content = []
    for block in body["content"]:
        if not isinstance(block, dict):
            content.append(block)
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            content.append({**block, "text": restore(block["text"])})
        elif kind == "tool_use" and isinstance(block.get("input"), (dict, list)):
            content.append({**block, "input": _map_strings(block["input"], restore)})
        else:
            content.append(block)
    return {**body, "content": content}


class AnthropicStreamRestorer:
    """Restores values across Anthropic's SSE events.

    The assembly problem is the one already solved for OpenAI — a stand-in arrives
    split across chunks and has to be rejoined before it can be replaced — but the
    events and the path to the text inside them are different. Text arrives as
    ``text_delta.text``; tool arguments arrive as ``input_json_delta.partial_json``,
    a JSON fragment at a time, which is still just characters to reassemble.
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        from pii_shield.streaming import StreamDeanonymizer

        self._mapping = mapping
        self._factory = StreamDeanonymizer
        self._streams: dict[tuple, Any] = {}

    def _for(self, key: tuple):
        if key not in self._streams:
            self._streams[key] = self._factory(self._mapping)
        return self._streams[key]

    def restore_event(self, event: Any) -> Any:
        if not isinstance(event, dict):
            return event
        kind = event.get("type")
        index = event.get("index", 0)

        if kind == "content_block_delta" and isinstance(event.get("delta"), dict):
            delta = dict(event["delta"])
            if isinstance(delta.get("text"), str):
                delta["text"] = self._for((index, "text")).feed(delta["text"])
            if isinstance(delta.get("partial_json"), str):
                delta["partial_json"] = self._for((index, "json")).feed(delta["partial_json"])
            return {**event, "delta": delta}

        # A block can arrive with its opening text or arguments already populated.
        if kind == "content_block_start" and isinstance(event.get("content_block"), dict):
            block = dict(event["content_block"])
            if isinstance(block.get("text"), str) and block["text"]:
                block["text"] = self._for((index, "text")).feed(block["text"])
            if isinstance(block.get("input"), (dict, list)):
                block["input"] = _map_strings(
                    block["input"], lambda s: self._for((index, "json")).feed(s)
                )
            return {**event, "content_block": block}

        return event

    def flush_for(self, index: Any) -> tuple[str, str]:
        """Release what is still withheld for one block, at ``content_block_stop``.

        Returned separately because the tail has to go back out as the same kind of
        delta it came in as: held text belongs in ``text_delta``, a held JSON fragment
        in ``input_json_delta``.
        """
        text = self._streams[(index, "text")].flush() if (index, "text") in self._streams else ""
        js = self._streams[(index, "json")].flush() if (index, "json") in self._streams else ""
        return text, js

    def flush(self) -> str:
        return "".join(stream.flush() for stream in self._streams.values())
