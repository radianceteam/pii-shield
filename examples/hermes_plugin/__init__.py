"""Hermes plugin: pseudonymize prompts on the way out, restore names on the way back.

Install by copying this directory to ``plugins/pii-shield/`` in a Hermes checkout.

**Which contract this uses, and why.** Hermes separates observers from middleware:
"Observer hooks report what happened. Middleware can change what happens by rewriting
a request or wrapping the actual execution callback" (``hermes_cli/middleware.py``).
The obvious-looking hooks are the wrong tool here:

  * ``pre_llm_call`` only injects context — it returns ``{"context": "..."}``, it does
    not rewrite messages;
  * ``pre_api_request`` is purely observational — ``_fire_pre_api_request_hook`` in
    ``agent/turn_api_request.py`` discards the return value entirely.

So the outbound rewrite goes through ``llm_request`` middleware, whose contract is
``{"request": {...}}`` replaces the provider kwargs, and the inbound restore through
the ``transform_llm_output`` hook ("return a replacement string, first non-None wins").

**One caveat worth knowing.** ``_run_execution_chain`` catches an exception from a
middleware callback and continues with the unmodified payload — middleware fails
*open*. Raising on a blocked credential would therefore send the original text. This
plugin instead redacts the offending span in place, so the secret cannot leave even
though the call still proceeds.

Config (``cli-config.yaml``)::

    plugins:
      pii-shield:
        language: ru
        fail_closed: true
"""

from __future__ import annotations

import logging

from pii_shield import (
    BlockedError,
    Policy,
    RedactionUnavailableError,
    Shield,
    UnknownSessionError,
)

logger = logging.getLogger(__name__)

BLOCKED_MARKER = "[pii-shield: redacted credential]"

_shield: Shield | None = None
_sessions: dict[str, str] = {}   # hermes session_id -> shield session_id


def register(ctx) -> None:
    """Entry point called by the Hermes plugin loader."""
    global _shield
    config = getattr(ctx, "config", {}) or {}
    policy = Policy.ru_default()
    policy.language = config.get("language", "ru")
    policy.fail_closed = bool(config.get("fail_closed", True))
    _shield = Shield(policy)

    ctx.register_middleware("llm_request", _on_llm_request)
    ctx.register_hook("transform_llm_output", _on_llm_output)
    logger.info("pii-shield active (language=%s, ner=%s)", policy.language, _shield.ner_ready)


def _on_llm_request(request=None, next_call=None, session_id: str = "", **_kw):
    """Rewrite outbound provider kwargs. Returns ``{"request": ...}`` or None."""
    if _shield is None or not isinstance(request, dict):
        return None
    messages = request.get("messages")
    if not isinstance(messages, list):
        return None

    shield_session = _sessions.get(session_id)
    rewritten: list = []
    changed = False

    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content:
            rewritten.append(message)
            continue
        try:
            try:
                result = _shield.anonymize(content, session_id=shield_session)
            except UnknownSessionError:
                # The conversation outlived the shield's session (an hour by default,
                # or a restart). Starting a new one costs a fresh cast of stand-ins;
                # letting the exception out would hand the original text to the
                # provider, because this middleware chain fails open.
                logger.info("pii-shield session expired, starting a new one")
                shield_session = None
                result = _shield.anonymize(content)
            shield_session = result.session_id or shield_session
            rewritten.append({**message, "content": result.text})
            changed = changed or result.changed
        except BlockedError as exc:
            # Middleware fails open, so a raise here would leak. Redact instead.
            logger.warning("pii-shield redacted a blocked payload: %s", exc)
            rewritten.append({**message, "content": _redact(content, exc)})
            changed = True
        except RedactionUnavailableError as exc:
            logger.error("pii-shield failed closed, redacting whole message: %s", exc)
            rewritten.append({**message, "content": BLOCKED_MARKER})
            changed = True

    if shield_session and session_id:
        _sessions[session_id] = shield_session
    return {"request": {**request, "messages": rewritten}} if changed else None


def _redact(content: str, exc: BlockedError) -> str:
    """Cut the offending spans out, keeping the rest of the message usable."""
    out = content
    for finding in sorted(exc.findings, key=lambda f: f.start, reverse=True):
        out = out[: finding.start] + BLOCKED_MARKER + out[finding.end :]
    return out


def _on_llm_output(text: str = "", session_id: str = "", **_kw):
    """Put the real values back. First non-None return wins, per the hook contract."""
    shield_session = _sessions.get(session_id)
    if _shield is None or not shield_session or not text:
        return None
    try:
        restored = _shield.deanonymize(text, shield_session)
    except UnknownSessionError:
        # Nothing can be put back, and saying so beats raising: the answer still
        # reaches the user, with the stand-ins visible rather than a failed turn.
        logger.warning("pii-shield cannot restore this answer: the session is gone")
        _sessions.pop(session_id, None)
        return None
    return restored if restored != text else None
