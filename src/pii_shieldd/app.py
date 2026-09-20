"""HTTP sidecar — the same shield, reachable from a non-Python consumer.

Python consumers should import :class:`pii_shield.Shield` directly and skip this
entirely: an in-process call has no serialization cost and, more importantly, no
window in which the un-anonymized payload exists on a socket. The sidecar is for
consumers that cannot import Python at all.

Defaults are deliberately paranoid, because this process handles raw PII by
construction: it binds loopback, it never logs request bodies, and it refuses to
start on a non-loopback address without an auth token.
"""

from __future__ import annotations

import os
import secrets as _secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from pii_shield import BlockedError, Policy, RedactionUnavailableError, Shield
from pii_shield.engine.presidio_engine import NerUnavailableError

AUTH_ENV = "PII_SHIELD_TOKEN"

_shield: Shield | None = None


def get_shield() -> Shield:
    if _shield is None:  # pragma: no cover - set in the lifespan
        raise HTTPException(status_code=503, detail="shield not initialized")
    return _shield


def token_matches(presented: str | None, expected: str) -> bool:
    """Constant-time secret comparison.

    ``!=`` on two strings returns as soon as a byte differs, which leaks the length of
    the matching prefix through response timing. The signal is small over a network and
    entirely real on a local socket, and the fix costs nothing.
    """
    if not presented:
        return False
    return _secrets.compare_digest(presented, expected)


async def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    """Shared-secret gate. Inactive unless the env var is set."""
    expected = os.environ.get(AUTH_ENV)
    if not expected:
        return
    if not token_matches(authorization, f"Bearer {expected}"):
        raise HTTPException(status_code=401, detail="unauthorized")


ShieldDep = Annotated[Shield, Depends(get_shield)]
AuthDep = Annotated[None, Depends(require_token)]


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------
class AnonymizeRequest(BaseModel):
    # Unknown fields are rejected rather than ignored. A request sending {"language":
    # "en"} against a model that had no such field was silently served in the server's
    # own language: names came back untouched with a 200, which reads as "clean".
    model_config = ConfigDict(extra="forbid")

    text: str
    session_id: str | None = Field(
        default=None, description="Continue an existing session to keep surrogates stable"
    )
    language: str | None = Field(
        default=None, description="Language of this text; overrides the server default"
    )
    surrogate_language: str | None = Field(
        default=None, description="Language the stand-ins are drawn from"
    )
    policy: dict | None = Field(
        default=None,
        description=(
            "Per-request policy overrides. Anything omitted keeps the language preset's "
            "value, so sending {'language': 'en'} means 'the usual policy, in English' "
            "rather than 'a policy with no rules'."
        ),
    )

    def resolved_policy(self, default: Policy) -> Policy | None:
        """Merge the request's overrides onto the preset for the requested language.

        Building a bare ``Policy(**overrides)`` here was a trap: a caller sending only
        ``{"language": "en"}`` got ``default_action=surrogate`` and **no rules at all**,
        which silently turned off every BLOCK — a US SSN came back pseudonymized
        instead of refused. Partial input must not disarm the defaults.

        Passing ``rules`` explicitly still replaces them, including with ``[]``.
        """
        overrides = dict(self.policy or {})
        if self.language is not None:
            overrides["language"] = self.language
        if self.surrogate_language is not None:
            overrides["surrogate_language"] = self.surrogate_language
        if not overrides:
            return None
        language = overrides.pop("language", default.language)
        base = Policy.for_language(language)
        if "rules" not in overrides:
            overrides["rules"] = [r.model_dump() for r in base.rules]
        merged = base.model_dump()
        merged.update(overrides)
        merged["language"] = language
        return Policy(**merged)


class FindingOut(BaseModel):
    entity: str
    start: int
    end: int
    score: float
    action: str


class AnonymizeResponse(BaseModel):
    text: str
    session_id: str
    findings: list[FindingOut]
    names_analyzed: bool = Field(
        description=(
            "False when no language model ran: people, organizations and places were "
            "not looked for. Everything with a checksum or a fixed shape still was."
        )
    )


class DeanonymizeRequest(BaseModel):
    text: str
    session_id: str
    consume: bool = Field(
        default=True, description="Drop the mapping afterwards — the intended default"
    )


class DeanonymizeResponse(BaseModel):
    text: str


class BlockedResponse(BaseModel):
    detail: str = "blocked"
    entities: list[str]


def create_app(shield: Shield | None = None, proxy_config=None) -> FastAPI:
    """Build the sidecar app.

    ``proxy_config`` enables the OpenAI-compatible proxy routes. It is opt-in rather
    than always-on: the proxy forwards credentials to a third party, and a service
    that does that should only do it when someone asked for it.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _shield
        _shield = shield or Shield(Policy.ru_default())
        yield
        _shield = None

    app = FastAPI(
        title="pii-shield",
        description="Reversible PII pseudonymization for text leaving for cloud LLMs",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz(sh: ShieldDep) -> dict:
        return {"status": "ok", "ner_ready": sh.ner_ready, "language": sh.policy.language}

    @app.post(
        "/v1/anonymize",
        response_model=AnonymizeResponse,
        responses={422: {"model": BlockedResponse}, 503: {"description": "detection unavailable"}},
        dependencies=[Depends(require_token)],
    )
    async def anonymize(req: AnonymizeRequest, sh: ShieldDep):
        try:
            result = sh.anonymize(
                req.text, session_id=req.session_id, policy=req.resolved_policy(sh.policy)
            )
        except BlockedError as exc:
            # Same envelope and status as the proxy. They used to differ — the proxy
            # answered 400 with an OpenAI-shaped error while this endpoint answered 422
            # with a different body — so a client written against the documented shape
            # simply did not catch it.
            kinds = sorted({f.entity for f in exc.findings})
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"pii-shield blocked this request: {', '.join(kinds)}",
                        "type": "pii_shield_blocked",
                        "param": None,
                        "code": "blocked",
                        "entities": kinds,
                    }
                },
            ) from exc
        except (RedactionUnavailableError, NerUnavailableError) as exc:
            # Fail closed: the caller must not fall back to sending raw text.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AnonymizeResponse(
            text=result.text,
            session_id=result.session_id,
            names_analyzed=result.names_analyzed,
            findings=[
                FindingOut(
                    entity=f.entity, start=f.start, end=f.end,
                    score=f.score, action=f.action,
                )
                for f in result.findings
            ],
        )

    @app.post(
        "/v1/deanonymize",
        response_model=DeanonymizeResponse,
        dependencies=[Depends(require_token)],
    )
    async def deanonymize(req: DeanonymizeRequest, sh: ShieldDep):
        return DeanonymizeResponse(
            text=sh.deanonymize(req.text, req.session_id, consume=req.consume)
        )

    @app.delete("/v1/session/{session_id}", status_code=204, dependencies=[Depends(require_token)])
    async def drop_session(session_id: str, sh: ShieldDep) -> Response:
        """Explicit teardown, for a caller that aborts before deanonymizing."""
        sh.store.drop(session_id)
        return Response(status_code=204)

    if proxy_config is not None:
        from .proxy import create_proxy_router

        app.include_router(create_proxy_router(proxy_config))

    return app


app = create_app()
