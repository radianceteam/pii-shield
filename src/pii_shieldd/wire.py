"""Walking an OpenAI chat-completions payload: what gets anonymized, and what does not.

Kept apart from the proxy transport so the rules are readable on their own — this is
the part a reviewer asks about first ("does it cover tool results?").
"""

from __future__ import annotations

from typing import Any

from pii_shield import Shield
from pii_shield.streaming import StreamDeanonymizer


def anonymize_messages(
    shield: Shield,
    messages: list[Any],
    session_id: str | None = None,
    policy=None,
) -> tuple[list[Any], str | None, bool]:
    """Return rewritten messages, the session handle, and whether anything changed.

    Every role is covered, not just ``user``. In an agent workload the identifying
    data usually arrives as a ``tool`` result — a file that was read, a row that was
    queried — so filtering only user turns would miss the common case.
    """
    out: list[Any] = []
    changed = False

    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        rewritten = dict(message)

        content = message.get("content")
        if isinstance(content, str) and content:
            result = shield.anonymize(content, session_id=session_id, policy=policy)
            session_id = result.session_id or session_id
            changed = changed or result.changed
            rewritten["content"] = result.text
        elif isinstance(content, list):
            # Multimodal content: a list of parts. Only text parts are touched;
            # image and audio parts pass through untouched (this shield does not
            # look inside binaries, and pretending otherwise would be worse).
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                    result = shield.anonymize(part["text"], session_id=session_id, policy=policy)
                    session_id = result.session_id or session_id
                    changed = changed or result.changed
                    parts.append({**part, "text": result.text})
                else:
                    parts.append(part)
            rewritten["content"] = parts

        # Assistant turns replayed from history carry their tool-call arguments, and
        # those arguments are where a name the model extracted earlier lives.
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            calls = []
            for call in tool_calls:
                fn = call.get("function") if isinstance(call, dict) else None
                args = fn.get("arguments") if isinstance(fn, dict) else None
                if isinstance(args, str) and args:
                    result = shield.anonymize(args, session_id=session_id, policy=policy)
                    session_id = result.session_id or session_id
                    changed = changed or result.changed
                    calls.append({**call, "function": {**fn, "arguments": result.text}})
                else:
                    calls.append(call)
            rewritten["tool_calls"] = calls

        out.append(rewritten)

    return out, session_id, changed


def deanonymize_response(shield: Shield, body: Any, session_id: str) -> Any:
    """Restore real values in a non-streamed completion."""
    if not isinstance(body, dict):
        return body
    choices = body.get("choices")
    if not isinstance(choices, list):
        return body

    restored_choices = []
    for choice in choices:
        if not isinstance(choice, dict):
            restored_choices.append(choice)
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            message = _restore_message(shield, message, session_id)
        restored_choices.append({**choice, "message": message} if message else choice)
    return {**body, "choices": restored_choices}


def _restore_message(shield: Shield, message: dict, session_id: str) -> dict:
    out = dict(message)
    if isinstance(message.get("content"), str):
        out["content"] = shield.deanonymize(message["content"], session_id)
    # Reasoning traces echo the prompt back and leak surrogates into the UI otherwise.
    for key in ("reasoning_content", "reasoning"):
        if isinstance(message.get(key), str):
            out[key] = shield.deanonymize(message[key], session_id)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        calls = []
        for call in tool_calls:
            fn = call.get("function") if isinstance(call, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                calls.append(
                    {**call, "function": {**fn, "arguments": shield.deanonymize(args, session_id)}}
                )
            else:
                calls.append(call)
        out["tool_calls"] = calls
    return out


class StreamRestorer:
    """Per-stream restorers, one per text field that arrives in fragments.

    A single deanonymizer cannot serve the whole stream: content and each tool call's
    arguments are independent character sequences, and interleaving them through one
    buffer would splice a held-back name into the wrong field.
    """

    def __init__(
        self, mapping: dict[str, str], inflected_pairs: list[tuple[str, str]] | None = None
    ) -> None:
        self._mapping = mapping
        self._inflected = inflected_pairs or []
        self._streams: dict[tuple, StreamDeanonymizer] = {}

    def _for(self, key: tuple) -> StreamDeanonymizer:
        if key not in self._streams:
            self._streams[key] = StreamDeanonymizer(self._mapping, self._inflected)
        return self._streams[key]

    def restore_chunk(self, chunk: Any) -> Any:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("choices"), list):
            return chunk
        choices = []
        for choice in chunk["choices"]:
            if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                choices.append(choice)
                continue
            index = choice.get("index", 0)
            delta = dict(choice["delta"])
            for key in ("content", "reasoning_content", "reasoning"):
                if isinstance(delta.get(key), str):
                    delta[key] = self._for((index, key)).feed(delta[key])
            if isinstance(delta.get("tool_calls"), list):
                delta["tool_calls"] = [
                    self._restore_tool_call(index, call) for call in delta["tool_calls"]
                ]
            choices.append({**choice, "delta": delta})
        return {**chunk, "choices": choices}

    def _restore_tool_call(self, choice_index: Any, call: Any) -> Any:
        fn = call.get("function") if isinstance(call, dict) else None
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if not isinstance(args, str):
            return call
        key = (choice_index, "tool", call.get("index", 0))
        return {**call, "function": {**fn, "arguments": self._for(key).feed(args)}}

    def flush(self) -> str:
        """Any text still withheld when the stream ended.

        Returned as one string for the caller to append as a final delta — dropping it
        would truncate the answer mid-word whenever it happened to end on a partial
        surrogate.
        """
        return "".join(stream.flush() for stream in self._streams.values())
