"""Value types shared across the shield: findings, results, errors.

Nothing here imports Presidio or any NLP stack, so a consumer can depend on the
contract (and type-check against it) without paying for the detection extras.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Action(StrEnum):
    """What to do with a detected entity.

    ``SURROGATE`` is the only reversible action: the replacement is recorded in the
    session map so the model's answer can be restored. ``HASH`` is deterministic but
    one-way — the same input always yields the same token, which keeps coreference
    ("this user") working across a conversation without ever being restorable.
    """

    ALLOW = "allow"
    MASK = "mask"
    SURROGATE = "surrogate"
    HASH = "hash"
    BLOCK = "block"


class Finding(BaseModel):
    """One detected entity. ``text`` is deliberately absent — a Finding travels to
    logs and telemetry, and carrying the raw value there would reintroduce exactly
    the leak the shield exists to prevent. Callers that need the value read it from
    the source text using the offsets."""

    entity: str = Field(description="Entity type, e.g. PERSON, RU_INN, SECRET_API_KEY")
    start: int
    end: int
    score: float = Field(ge=0.0, le=1.0)
    action: Action = Field(description="Action actually applied")
    recognizer: str = Field(default="", description="Which recognizer fired")


class ShieldError(Exception):
    """Base for every shield failure."""


class BlockedError(ShieldError):
    """Raised when a BLOCK-policy entity was found. The request must not proceed.

    Carries findings (offsets only, never values) so the caller can tell the user
    *what kind* of data blocked the call without echoing the data itself.
    """

    def __init__(self, findings: list[Finding]) -> None:
        kinds = sorted({f.entity for f in findings})
        super().__init__(f"blocked: {', '.join(kinds)}")
        self.findings = findings


class UnknownSessionError(ShieldError):
    """The session id does not exist here — invented, expired, or evicted.

    Raised rather than shrugged off, because both ways of shrugging are silent
    failures. Anonymizing into an unknown session used to substitute the text and
    record nothing, so the caller got a clean 200 and could never restore it;
    deanonymizing from one returned the text with the stand-ins still in it, which
    reads as "the model did not mention anybody" rather than "the originals are gone".
    """


class RedactionUnavailableError(ShieldError):
    """The detection stack raised, so no safe text can be produced.

    Under ``fail_closed`` (the default) this propagates and the caller must abort
    the outbound call. This is the one behaviour that makes the shield a control
    rather than a best-effort filter: a crashed analyzer must never degrade into
    "send the raw text".
    """


class AnonymizeResult(BaseModel):
    """Outcome of anonymizing one payload."""

    text: str = Field(description="Text safe to send outbound")
    session_id: str = Field(description="Handle for deanonymize(); empty if nothing reversible")
    findings: list[Finding] = Field(default_factory=list)
    names_analyzed: bool = Field(
        default=True,
        description=(
            "False when no language model ran, so people, organizations and places "
            "were not looked for. Everything with a checksum or a fixed shape still "
            "was. A caller that hides this from its users is telling them the text is "
            "clean when only half of it was examined."
        ),
    )
    credentials_redacted: bool = Field(
        default=False,
        description=(
            "True when a credential in this payload was replaced by a placeholder "
            "rather than refusing the request. The replacement is one-way: nothing is "
            "written to the session map, so the real value never returns in an answer."
        ),
    )

    @property
    def changed(self) -> bool:
        return bool(self.findings)
