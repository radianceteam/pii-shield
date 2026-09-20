"""Adapter: wrap any spec-editor ``LLMProvider`` so nothing identifying leaves the host.

Drop this beside ``src/providers/`` in spec-editor and wrap the provider at the point
it is constructed::

    provider = AnonymizingProvider(LiteLLMProvider(...), Shield(Policy.ru_default()))

Why the provider layer and not the agent layer: every backend in spec-editor
(``LiteLLMProvider``, ``ClaudeCodeProvider``, ``OllamaProvider``) implements the one
``LLMProvider.complete()`` contract, so wrapping it covers all of them and cannot be
bypassed by a new agent that forgets to call the filter.

Note that ``ClaudeCodeProvider`` is a cloud route despite running a local binary — the
``claude`` CLI forwards the prompt to Anthropic. Only a genuinely local backend such as
Ollama should be given ``Policy.off()``.
"""

from __future__ import annotations

from pii_shield import BlockedError, Policy, Shield

# In spec-editor these come from src.providers.base; imported lazily here so this
# example file stays importable on its own.
try:
    from src.providers.base import LLMProvider, LLMResponse, Message, ToolDef
except ImportError:  # pragma: no cover - example runs inside spec-editor
    LLMProvider = object  # type: ignore[assignment,misc]
    LLMResponse = Message = ToolDef = object  # type: ignore[assignment,misc]


class AnonymizingProvider(LLMProvider):  # type: ignore[misc,valid-type]
    """Decorator that anonymizes outbound messages and restores the answer.

    One session spans one ``complete()`` call, which is the correct granularity: the
    full history is re-sent on every call, so a per-call session still sees every
    mention of a given person and keeps one surrogate for them throughout.
    """

    def __init__(self, inner, shield: Shield | None = None) -> None:
        self._inner = inner
        self._shield = shield or Shield(Policy.ru_default())

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        session_id: str | None = None
        safe: list[Message] = []

        # Every role, not just USER: tool results carry file contents, and that is
        # where identifying data actually reaches the model in a coding agent.
        for message in messages:
            if not message.content:
                safe.append(message)
                continue
            result = self._shield.anonymize(message.content, session_id=session_id)
            session_id = result.session_id or session_id
            safe.append(message.model_copy(update={"content": result.text}))

        response = await self._inner.complete(safe, tools, temperature, max_tokens)

        if session_id and response.content:
            restored = self._shield.deanonymize(response.content, session_id)
            response = response.model_copy(update={"content": restored})

        # Tool-call arguments come back carrying surrogates too. Restoring them is
        # what stops a fake name from being written into a saved specification.
        if session_id and response.tool_calls:
            for call in response.tool_calls:
                call.arguments = {
                    k: self._shield.deanonymize(v, session_id) if isinstance(v, str) else v
                    for k, v in call.arguments.items()
                }

        if session_id:
            self._shield.store.drop(session_id)
        return response

    def supports_tools(self) -> bool:
        return self._inner.supports_tools()


__all__ = ["AnonymizingProvider", "BlockedError"]
