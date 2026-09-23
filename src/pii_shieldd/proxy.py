"""OpenAI-compatible proxy: anonymize, forward, restore — with no cooperation from the client.

Why this exists alongside ``/v1/anonymize``. That endpoint requires the consumer to
call it, and several agent frameworks cannot: OpenClaw's plugin hooks can *observe*
the model input (``llm_input``) or *block* the run (``before_agent_run``, "only pass
and block outcomes are supported"), but none of them can rewrite the prompt. What
those frameworks do support is pointing a provider at a custom ``baseUrl`` — OpenClaw
documents "Local proxies (LM Studio, vLLM, LiteLLM, etc.)" as a first-class setup.

So the shield becomes the base URL. The client sends an ordinary chat completion and
gets an ordinary one back; the substitution happens in between.

Authentication is split deliberately: ``Authorization`` is forwarded upstream
untouched, because the caller owns that credential and this proxy should never need
to hold it. The sidecar's own gate is ``X-Pii-Shield-Token``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from pii_shield import BlockedError, Policy, RedactionUnavailableError, Shield
from pii_shield.engine.presidio_engine import NerUnavailableError
from pii_shield.languages import UnsupportedLanguageError

from .app import AUTH_ENV, get_shield, token_matches
from .wire import StreamRestorer, anonymize_messages, deanonymize_response
from .wire_anthropic import AnthropicStreamRestorer
from .wire_anthropic import anonymize_request as anthropic_anonymize
from .wire_anthropic import deanonymize_response as anthropic_deanonymize

logger = logging.getLogger(__name__)

UPSTREAM_ENV = "PII_SHIELD_UPSTREAM"
UPSTREAM_KEY_ENV = "PII_SHIELD_UPSTREAM_KEY"
PROXY_TOKEN_HEADER = "x-pii-shield-token"
DEFAULT_UPSTREAM = "https://api.openai.com/v1"
SSE_DONE = "[DONE]"


class UpstreamHeaderError(ValueError):
    """A header the caller sent cannot be put on the wire.

    httpx encodes header values as ASCII, while the server decodes what arrived as
    latin-1 — so ``Authorization: Bearer тест`` reaches this process as a string of
    high code points and makes httpx raise while it builds the forwarded request.
    That surfaced to the caller as a 500 with no explanation. Clients should not send
    such a header, but naming the offending one in a 400 is both truer and actionable.
    """


@dataclass
class ProxyConfig:
    upstream: str = DEFAULT_UPSTREAM
    upstream_key: str | None = None
    timeout: float = 600.0

    @classmethod
    def from_env(cls) -> ProxyConfig:
        return cls(
            upstream=os.environ.get(UPSTREAM_ENV, DEFAULT_UPSTREAM).rstrip("/"),
            upstream_key=os.environ.get(UPSTREAM_KEY_ENV) or None,
        )


def _error(
    status: int, message: str, err_type: str, code: str, entities: list[str] | None = None
) -> JSONResponse:
    """OpenAI's error envelope, so an unmodified client renders it instead of crashing.

    ``entities`` names the *kinds* that caused a block, never the offending values, so
    a caller can tell the user what to remove without the secret travelling further.
    """
    error: dict = {"message": message, "type": err_type, "param": None, "code": code}
    if entities is not None:
        error["entities"] = entities
    return JSONResponse(status_code=status, content={"error": error})


async def require_proxy_token(
    x_pii_shield_token: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.environ.get(AUTH_ENV)
    if expected and not token_matches(x_pii_shield_token, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


ProxyAuth = Annotated[None, Depends(require_proxy_token)]
ShieldDep = Annotated[Shield, Depends(get_shield)]


def create_proxy_router(config: ProxyConfig | None = None) -> APIRouter:
    cfg = config or ProxyConfig.from_env()
    router = APIRouter(dependencies=[Depends(require_proxy_token)])

    def upstream_headers(request: Request) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        # Stripped, and dropped when it is empty: a client sending a bare
        # "Authorization: Bearer " (an unset variable in a curl command, typically)
        # produced a header value httpx refuses, which surfaced to the caller as
        # "upstream unreachable" — a confusing 502 for what is a malformed request.
        auth = (request.headers.get("authorization") or "").strip()
        if auth and auth.lower() not in ("bearer", "basic"):
            headers["authorization"] = auth
        elif cfg.upstream_key:
            headers["authorization"] = f"Bearer {cfg.upstream_key}"
        # Hop-by-hop and length headers are rebuilt by httpx; copying them corrupts
        # the forwarded request.
        # Anthropic authenticates with x-api-key rather than a bearer token, and
        # requires a version header. Both travel upstream untouched: the rule that the
        # shield never holds the caller's credential is the same either way.
        for passthrough in (
            "openai-organization", "openai-project",
            "x-api-key", "anthropic-version", "anthropic-beta",
        ):
            if passthrough in request.headers:
                headers[passthrough] = request.headers[passthrough]
        for name, value in headers.items():
            try:
                value.encode("ascii")
            except UnicodeEncodeError as exc:
                raise UpstreamHeaderError(
                    f"header {name!r} cannot be forwarded: HTTP header values must be "
                    "ASCII, and this one is not"
                ) from exc
        return headers

    def per_request_policy(request: Request, shield: Shield) -> Policy | None:
        """Build a policy from request headers, or None to use the server default.

        The OpenAI request body has no field for this and adding one would make the
        endpoint non-standard, so the language travels in headers. A multi-tenant
        caller — a platform serving users who write in different languages — needs
        this: without it one deployment can only ever read one language, and every
        other user's text is scanned by the wrong model and comes back clean.
        """
        language = request.headers.get("x-pii-shield-language")
        surrogate = request.headers.get("x-pii-shield-surrogate-language")
        if not language and not surrogate:
            return None
        policy = Policy.like(shield.policy, language)
        policy.surrogate_language = surrogate or shield.policy.surrogate_language
        policy.surrogate_locale = None if surrogate else shield.policy.surrogate_locale
        return policy

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request, shield: ShieldDep):
        try:
            payload = await request.json()
        except Exception:
            return _error(
                400, "request body is not valid JSON", "invalid_request_error", "bad_json"
            )
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            return _error(400, "missing 'messages'", "invalid_request_error", "missing_messages")

        try:
            policy = per_request_policy(request, shield)
        except (UnsupportedLanguageError, ValueError) as exc:
            return _error(400, str(exc), "invalid_request_error", "bad_language")

        # Built before anything is anonymized: a request that cannot be forwarded
        # should fail before a session map exists for it.
        try:
            headers = upstream_headers(request)
        except UpstreamHeaderError as exc:
            return _error(400, str(exc), "invalid_request_error", "bad_header")

        try:
            messages, session_id, _changed = anonymize_messages(
                shield, payload["messages"], policy=policy
            )
        except BlockedError as exc:
            kinds = sorted({f.entity for f in exc.findings})
            return _error(
                400,
                f"pii-shield blocked this request: {', '.join(kinds)}",
                "pii_shield_blocked",
                "blocked",
                entities=kinds,
            )
        except (RedactionUnavailableError, NerUnavailableError) as exc:
            # 503 and nothing forwarded: a broken detector must never degrade into
            # sending the original text upstream.
            return _error(503, f"pii-shield unavailable: {exc}", "pii_shield_error", "unavailable")

        forwarded = {**payload, "messages": messages}
        url = f"{cfg.upstream}/chat/completions"

        if payload.get("stream"):
            return StreamingResponse(
                _stream(shield, cfg, url, forwarded, headers, session_id),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            )

        try:
            async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                response = await client.post(url, json=forwarded, headers=headers)
        except httpx.HTTPError as exc:
            _drop(shield, session_id)
            return _error(502, f"upstream unreachable: {exc}", "pii_shield_error", "upstream")

        if response.status_code >= 400:
            _drop(shield, session_id)
            return JSONResponse(status_code=response.status_code, content=_json_or_text(response))

        body = _json_or_text(response)
        if session_id:
            body = deanonymize_response(shield, body, session_id)
            _drop(shield, session_id)
        return JSONResponse(status_code=response.status_code, content=body)

    @router.post("/v1/messages")
    async def anthropic_messages(request: Request, shield: ShieldDep):
        """Anthropic Messages, same treatment as chat completions.

        The body is walked differently — see wire_anthropic for why — but the contract
        is identical: nothing identifying leaves, nothing unexamined leaves either,
        and the caller's credential passes straight through.
        """
        try:
            payload = await request.json()
        except Exception:
            return _anthropic_error(400, "request body is not valid JSON", "invalid_request_error")
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            return _anthropic_error(400, "missing 'messages'", "invalid_request_error")

        try:
            policy = per_request_policy(request, shield)
        except (UnsupportedLanguageError, ValueError) as exc:
            return _anthropic_error(400, str(exc), "invalid_request_error")

        try:
            headers = upstream_headers(request)
        except UpstreamHeaderError as exc:
            return _anthropic_error(400, str(exc), "invalid_request_error")

        try:
            forwarded, session_id, _changed, _names = anthropic_anonymize(
                shield, payload, policy=policy
            )
        except BlockedError as exc:
            kinds = sorted({f.entity for f in exc.findings})
            return _anthropic_error(
                400,
                f"pii-shield blocked this request: {', '.join(kinds)}",
                "invalid_request_error",
            )
        except (RedactionUnavailableError, NerUnavailableError) as exc:
            return _anthropic_error(503, f"pii-shield unavailable: {exc}", "api_error")

        url = f"{cfg.upstream}/messages"

        if payload.get("stream"):
            return StreamingResponse(
                _anthropic_stream(shield, cfg, url, forwarded, headers, session_id),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            )

        try:
            async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                response = await client.post(url, json=forwarded, headers=headers)
        except httpx.HTTPError as exc:
            _drop(shield, session_id)
            return _anthropic_error(502, f"upstream unreachable: {exc}", "api_error")

        if response.status_code >= 400:
            _drop(shield, session_id)
            return JSONResponse(status_code=response.status_code, content=_json_or_text(response))

        body = _json_or_text(response)
        if session_id:
            body = anthropic_deanonymize(shield, body, session_id)
            _drop(shield, session_id)
        return JSONResponse(status_code=response.status_code, content=body)

    @router.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request, shield: ShieldDep):
        """Counts tokens for a full prompt — so the prompt is cleaned first.

        Forwarding this one untouched "for compatibility" would send the provider the
        very text the shield exists to withhold, and it would look like the endpoint
        worked. The count is taken on the pseudonymized text; stand-ins are chosen to
        resemble what they replace, so the number stays representative.
        """
        try:
            payload = await request.json()
        except Exception:
            return _anthropic_error(400, "request body is not valid JSON", "invalid_request_error")
        if not isinstance(payload, dict):
            return _anthropic_error(400, "body must be an object", "invalid_request_error")

        try:
            headers = upstream_headers(request)
        except UpstreamHeaderError as exc:
            return _anthropic_error(400, str(exc), "invalid_request_error")

        try:
            policy = per_request_policy(request, shield)
            forwarded, session_id, _changed, _names = anthropic_anonymize(
                shield, payload, policy=policy
            )
        except BlockedError as exc:
            kinds = sorted({f.entity for f in exc.findings})
            return _anthropic_error(
                400,
                f"pii-shield blocked this request: {', '.join(kinds)}",
                "invalid_request_error",
            )
        except (UnsupportedLanguageError, ValueError) as exc:
            return _anthropic_error(400, str(exc), "invalid_request_error")
        except (RedactionUnavailableError, NerUnavailableError) as exc:
            return _anthropic_error(503, f"pii-shield unavailable: {exc}", "api_error")

        try:
            async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                response = await client.post(
                    f"{cfg.upstream}/messages/count_tokens",
                    json=forwarded,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            return _anthropic_error(502, f"upstream unreachable: {exc}", "api_error")
        finally:
            _drop(shield, session_id)
        return JSONResponse(status_code=response.status_code, content=_json_or_text(response))

    @router.get("/v1/models")
    async def models(request: Request):
        """Plain passthrough — clients probe this before they will talk to a base URL."""
        try:
            headers = upstream_headers(request)
        except UpstreamHeaderError as exc:
            return _error(400, str(exc), "invalid_request_error", "bad_header")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{cfg.upstream}/models", headers=headers)
        except httpx.HTTPError as exc:
            return _error(502, f"upstream unreachable: {exc}", "pii_shield_error", "upstream")
        return JSONResponse(status_code=response.status_code, content=_json_or_text(response))

    return router


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"error": {"message": response.text, "type": "upstream_error", "code": "non_json"}}


def _anthropic_error(status: int, message: str, err_type: str) -> JSONResponse:
    """Anthropic's error envelope.

    The OpenAI one is a different shape, and the Anthropic SDK parses the response
    against its own schema — handed the wrong envelope it raises on the parse rather
    than surfacing the message, so the caller never learns why the request stopped.
    """
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": err_type, "message": message}},
    )


def _drop(shield: Shield, session_id: str | None) -> None:
    if session_id:
        shield.store.drop(session_id)


async def _stream(
    shield: Shield,
    cfg: ProxyConfig,
    url: str,
    payload: dict,
    headers: dict[str, str],
    session_id: str | None,
) -> AsyncIterator[bytes]:
    """Re-emit the upstream SSE stream with surrogates restored in flight."""
    mapping = shield.store.mapping(session_id) if session_id else {}
    pairs = shield.store.inflectable_pairs(session_id) if session_id else []
    restorer = StreamRestorer(mapping, pairs)
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    yield _sse({"error": {"message": raw.decode("utf-8", "replace"),
                                          "type": "upstream_error",
                                          "code": str(response.status_code)}})
                    yield f"data: {SSE_DONE}\n\n".encode()
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        # Comments and other SSE fields (": keep-alive", "event: ...")
                        # are forwarded untouched.
                        yield f"{line}\n\n".encode()
                        continue
                    data = line[len("data:") :].strip()
                    if data == SSE_DONE:
                        tail = restorer.flush()
                        if tail:
                            yield _sse(_tail_chunk(payload, tail))
                        yield f"data: {SSE_DONE}\n\n".encode()
                        return
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        yield f"data: {data}\n\n".encode()
                        continue
                    yield _sse(restorer.restore_chunk(chunk))

                # Upstream ended without [DONE]; release whatever is still held so the
                # answer is not truncated mid-name.
                tail = restorer.flush()
                if tail:
                    yield _sse(_tail_chunk(payload, tail))
    except httpx.HTTPError as exc:
        logger.warning("pii-shield proxy stream failed: %s", exc)
        yield _sse({"error": {"message": f"upstream stream failed: {exc}",
                              "type": "pii_shield_error", "code": "upstream"}})
        yield f"data: {SSE_DONE}\n\n".encode()
    finally:
        _drop(shield, session_id)


async def _anthropic_stream(
    shield: Shield,
    cfg: ProxyConfig,
    url: str,
    payload: dict,
    headers: dict[str, str],
    session_id: str | None,
) -> AsyncIterator[bytes]:
    """Re-emit Anthropic's SSE stream with stand-ins restored in flight.

    Anthropic sends ``event:`` lines alongside ``data:`` lines; the event lines are
    forwarded untouched. A tail withheld mid-name is released just before the block
    closes, as a delta of the same kind it was held from, so the client never sees a
    stand-in and never loses the last characters of an answer.
    """
    mapping = shield.store.mapping(session_id) if session_id else {}
    pairs = shield.store.inflectable_pairs(session_id) if session_id else []
    restorer = AnthropicStreamRestorer(mapping, pairs)
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    yield _sse({"type": "error", "error": {
                        "type": "api_error", "message": raw.decode("utf-8", "replace")}})
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        yield f"{line}\n".encode()
                        continue
                    data = line[len("data:") :].strip()
                    try:
                        event = json.loads(data)
                    except ValueError:
                        yield f"data: {data}\n\n".encode()
                        continue

                    if event.get("type") == "content_block_stop":
                        index = event.get("index", 0)
                        text_tail, json_tail = restorer.flush_for(index)
                        if text_tail:
                            yield _sse({"type": "content_block_delta", "index": index,
                                        "delta": {"type": "text_delta", "text": text_tail}})
                        if json_tail:
                            yield _sse({"type": "content_block_delta", "index": index,
                                        "delta": {"type": "input_json_delta",
                                                  "partial_json": json_tail}})
                        yield _sse(event)
                        continue

                    yield _sse(restorer.restore_event(event))
    except httpx.HTTPError as exc:
        logger.warning("pii-shield anthropic stream failed: %s", exc)
        yield _sse({"type": "error", "error": {
            "type": "api_error", "message": f"upstream stream failed: {exc}"}})
    finally:
        _drop(shield, session_id)


def _sse(obj: Any) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


def _tail_chunk(payload: dict, text: str) -> dict:
    return {
        "object": "chat.completion.chunk",
        "model": payload.get("model", ""),
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
