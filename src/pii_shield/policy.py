"""Policy: which entities matter in which language, and what happens to each.

Policy travels with the request, not with the service. Two consumers behind one
sidecar — a Russian requirements editor and a Japanese support bot — have different
entity catalogues and different tolerances, and a policy baked into the server would
force one language's assumptions on the other.

Everything language-dependent has a ``None`` default that resolves against the
language profile. Set it explicitly to override; leave it alone to get the right
answer for whatever language the request is in.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from .languages import (
    FINANCE_ENTITIES,
    GLOBAL_NER_ENTITIES,
    LOCAL_ENTITIES,
    NER_MODEL_ENTITIES,
    PRESIDIO_ENTITIES,
    SECRET_ENTITIES,
    SUPPORTED_LANGUAGES,
    UnsupportedLanguageError,
    all_national_entities,
    get_profile,
)
from .types import Action

GLOBAL_ENTITIES = GLOBAL_NER_ENTITIES
ALL_ENTITIES = GLOBAL_ENTITIES + FINANCE_ENTITIES + all_national_entities() + SECRET_ENTITIES

# National identifiers that identify a *person* uniquely and permanently. A plausible
# fake one is worse than none: the model may reason about it or write it into output
# as if it were real, and unlike a name there is no useful ambiguity to preserve.
HIGH_RISK_NATIONAL = frozenset({
    "RU_PASSPORT", "RU_BANK_ACCOUNT", "RU_SNILS",
    "US_SSN", "US_ITIN", "US_PASSPORT", "US_DRIVER_LICENSE", "US_BANK_NUMBER",
    "UK_NHS", "ES_NIF", "ES_NIE", "IT_FISCAL_CODE", "IT_IDENTITY_CARD",
    "IT_PASSPORT", "IT_DRIVER_LICENSE", "PL_PESEL",
    "CN_RESIDENT_ID", "JP_MY_NUMBER", "KR_RRN",
})

# Punctuation stripped from each token before an allowlist comparison.
_TOKEN_TRIM = " \t\n.,;:!?()[]{}«»\"'-–—/\\、。「」『』（）：；！？"


def _validate_faker_locale(locale: str) -> None:
    """Reject a locale Faker does not know, rather than silently producing tokens.

    Skipped when Faker is absent: the pools are used then anyway, and refusing a
    perfectly reasonable policy because an optional extra is missing would be wrong.
    """
    try:
        from faker.config import AVAILABLE_LOCALES
    except ImportError:
        return
    if locale not in AVAILABLE_LOCALES:
        raise ValueError(f"unknown Faker locale {locale!r}")


class Scope(StrEnum):
    """How far the filter reaches.

    OFF      — no filtering (explicit, so it is auditable).
    DISPLAY  — scrub logs, traces and UI surfaces only; the model still sees raw text.
    FULL     — scrub what goes to the model too. The setting compliance wants.
    """

    OFF = "off"
    DISPLAY = "display"
    FULL = "full"


class EntityRule(BaseModel):
    """Per-entity override."""

    entity: str
    action: Action
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class Policy(BaseModel):
    """A complete filtering decision set for one language."""

    language: str = Field(default="ru", description=f"One of: {', '.join(SUPPORTED_LANGUAGES)}")
    entities: list[str] | None = Field(
        default=None,
        description="Entity types to detect. None resolves to the language's catalogue.",
    )
    default_action: Action = Action.SURROGATE
    default_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    rules: list[EntityRule] = Field(default_factory=list)
    scope: Scope = Scope.FULL
    secrets: bool = Field(default=True, description="Run the credential-pattern layer")
    allowlist: list[str] | None = Field(
        default=None,
        description="Spans that are never PII. None resolves to the language's list.",
    )
    min_ner_span: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Shortest NER span accepted. None resolves per language: 3 where words are "
            "space-delimited, 2 for CJK where a full personal name is two characters."
        ),
    )
    surrogate_language: str | None = Field(
        default=None,
        description=(
            "Language to pseudonymize INTO, if different from the language being read. "
            "Detection still uses `language`; only the stand-ins change. None keeps them "
            "in the source language."
        ),
    )
    surrogate_locale: str | None = Field(
        default=None,
        description="Exact Faker locale, e.g. 'zh_TW'. Overrides surrogate_language.",
    )
    fail_closed: bool = Field(
        default=True,
        description="On analyzer failure, raise instead of returning unfiltered text",
    )

    @model_validator(mode="after")
    def _resolve(self) -> Policy:
        get_profile(self.language)  # raises UnsupportedLanguageError on a typo
        if self.surrogate_language is not None:
            profile = get_profile(self.surrogate_language)
            if profile.faker_locale is None:
                raise ValueError(
                    f"language {self.surrogate_language!r} has no surrogate locale; "
                    "stand-ins would fall back to typed tokens"
                )
        if self.surrogate_locale is not None:
            _validate_faker_locale(self.surrogate_locale)
        if self.entities is None:
            self.entities = list(self.catalogue_for(self.language))
        seen = set()
        for rule in self.rules:
            if rule.entity in seen:
                raise ValueError(f"duplicate rule for entity {rule.entity!r}")
            seen.add(rule.entity)
        return self

    # -- language-resolved values ------------------------------------------
    @staticmethod
    def catalogue_for(language: str) -> tuple[str, ...]:
        """Every entity worth looking for in this language."""
        profile = get_profile(language)
        return GLOBAL_ENTITIES + FINANCE_ENTITIES + profile.national_entities + SECRET_ENTITIES

    @property
    def profile(self):
        return get_profile(self.language)

    @property
    def effective_allowlist(self) -> tuple[str, ...]:
        return tuple(self.allowlist) if self.allowlist is not None else self.profile.allowlist

    @property
    def effective_min_ner_span(self) -> int:
        return self.min_ner_span if self.min_ner_span is not None else self.profile.min_ner_span

    @property
    def effective_surrogate_locale(self) -> str | None:
        """Faker locale the stand-ins are drawn from.

        Resolution order: an explicit locale, then the profile of
        ``surrogate_language``, then the profile of the language being read. The
        direction is a separate axis from detection on purpose — a Russian contract
        read by a Russian pipeline can be handed to the model with English names, which
        is what you want when the downstream model is much stronger in English, or when
        the reviewer of the outbound traffic does not read the source language.
        """
        if self.surrogate_locale:
            return self.surrogate_locale
        if self.surrogate_language:
            return get_profile(self.surrogate_language).faker_locale
        return self.profile.faker_locale

    # -- lookups ------------------------------------------------------------
    def action_for(self, entity: str) -> Action:
        for rule in self.rules:
            if rule.entity == entity:
                return rule.action
        return self.default_action

    def threshold_for(self, entity: str) -> float:
        for rule in self.rules:
            if rule.entity == entity:
                return rule.threshold
        return self.default_threshold

    def is_allowlisted(self, span_text: str) -> bool:
        """True if the span is nothing but allowlisted terms.

        Token-wise as well as whole-span, because NER returns the phrase it found:
        "REST API" arrives as one ORGANIZATION span even though each half is listed. A
        span is spared only when *every* token is allowlisted, so "API GmbH" — a real
        company whose name contains a listed token — is still caught.

        In Chinese and Japanese there is no whitespace to split on, so this degrades
        to whole-span matching. That is why those allowlists hold legal-form words
        (株式会社, 有限公司) that a model returns on their own.
        """
        probe = span_text.strip()
        if not probe:
            return True
        allowed = {term.casefold() for term in self.effective_allowlist}
        if probe.casefold() in allowed:
            return True
        tokens = [t.strip(_TOKEN_TRIM) for t in probe.split()]
        tokens = [t for t in tokens if t]
        return bool(tokens) and all(t.casefold() in allowed for t in tokens)

    # -- what this policy actually costs to run ----------------------------
    @property
    def active_entities(self) -> frozenset[str]:
        """Entities this policy will actually change.

        ALLOW entities are excluded: detecting one and then leaving it alone produces
        the same text as not detecting it, so a policy that allows every name needs no
        language model. Span precedence does not depend on them either — the ordering
        in :meth:`Shield._ner_priority` already ranks ALLOW last.
        """
        if self.scope is Scope.OFF:
            return frozenset()
        return frozenset(
            e for e in (self.entities or []) if self.action_for(e) is not Action.ALLOW
        )

    def requires_ner(self) -> bool:
        """True if this policy cannot be honoured without a spaCy language model.

        Only the four entities the NER component produces need one. A policy built
        from checksummed identifiers and credential shapes needs no model at all, and
        demanding one made the documented pattern-only deployment impossible to build
        — the difference between a 40 MB process and a 1 GB one.
        """
        return bool(self.active_entities & NER_MODEL_ENTITIES)

    def requires_presidio(self) -> bool:
        """True if this policy needs ``presidio-analyzer`` installed.

        Presidio's pattern and checksum recognizers (email, cards, IBAN, IP, URL and
        the national ones it ships) run on a blank pipeline, so this can be true while
        :meth:`requires_ner` is false.
        """
        return bool(self.active_entities & (PRESIDIO_ENTITIES | NER_MODEL_ENTITIES))

    @property
    def local_only(self) -> bool:
        """True if every active entity is served by this project's own patterns."""
        active = self.active_entities
        return bool(active) and active <= LOCAL_ENTITIES

    @classmethod
    def pattern_only(cls, language: str = "ru") -> Policy:
        """A policy that needs neither Presidio nor a language model.

        Everything with a checksum or a fixed shape — national identifiers, banking
        codes, credentials. Names and organizations are not detected, and a caller
        showing this to end users should say so.
        """
        profile = get_profile(language)
        entities = [e for e in cls.catalogue_for(language) if e in LOCAL_ENTITIES]
        blocked = sorted(set(profile.national_entities) & HIGH_RISK_NATIONAL & LOCAL_ENTITIES)
        return cls(
            language=language,
            entities=entities,
            rules=[
                *(
                    EntityRule(entity=e, action=Action.BLOCK, threshold=0.4)
                    for e in SECRET_ENTITIES
                ),
                *(EntityRule(entity=e, action=Action.BLOCK, threshold=0.4) for e in blocked),
            ],
        )

    @property
    def blocking_entities(self) -> frozenset[str]:
        """Entities whose mere presence aborts the request."""
        out = {r.entity for r in self.rules if r.action is Action.BLOCK}
        if self.default_action is Action.BLOCK:
            out |= {e for e in (self.entities or []) if self.action_for(e) is Action.BLOCK}
        return frozenset(out)

    # -- presets ------------------------------------------------------------
    @classmethod
    def for_language(cls, language: str) -> Policy:
        """Pseudonymize identity, block credentials and permanent national IDs."""
        profile = get_profile(language)
        blocked = sorted(set(profile.national_entities) & HIGH_RISK_NATIONAL)
        return cls(
            language=language,
            rules=[
                *(
                    EntityRule(entity=e, action=Action.BLOCK, threshold=0.4)
                    for e in SECRET_ENTITIES
                ),
                # 0.4, not the 0.5 default: Presidio scores a dash-less US SSN with a
                # nearby "social security number" at exactly 0.4, which the default
                # would discard. For an entity that blocks on sight, a false block is
                # recoverable and a leaked national ID is not.
                *(EntityRule(entity=e, action=Action.BLOCK, threshold=0.4) for e in blocked),
                EntityRule(entity="CREDIT_CARD", action=Action.BLOCK, threshold=0.5),
                # 0.4: Presidio's phone recognizer scores a number that `phonenumbers`
                # parses but that has no strong context word at exactly 0.4, so the
                # default 0.5 discarded every German, French and Japanese number while
                # US ones happened to score higher. Russian numbers were unaffected only
                # because they have a local pattern. Verified not to fire on order
                # numbers, ISBNs, amounts or version strings.
                EntityRule(entity="PHONE_NUMBER", action=Action.SURROGATE, threshold=0.4),
                # Dates and URLs are load-bearing context and rarely identifying alone.
                EntityRule(entity="DATE_TIME", action=Action.ALLOW),
                EntityRule(entity="URL", action=Action.ALLOW),
                # Bank and entity identifier codes. None of them is personal data on
                # its own, but each pins a payment down to an institution and an
                # account holder, so they are pseudonymized rather than passed through.
                # The stand-ins are structurally valid — a real BIC becomes another
                # well-formed BIC, an IBAN another IBAN that passes its check digits —
                # because a malformed replacement is corrupted data, not private data.
                # Detection is context-gated (see finance_patterns), so an ordinary
                # nine-digit reference or uppercase token is reported at 0.4 and left
                # alone by the default 0.5 threshold.
                EntityRule(entity="SWIFT_BIC", action=Action.SURROGATE),
                EntityRule(entity="IBAN_CODE", action=Action.SURROGATE),
                EntityRule(entity="ABA_ROUTING", action=Action.SURROGATE),
                EntityRule(entity="LEI", action=Action.SURROGATE),
            ],
        )

    @classmethod
    def ru_default(cls) -> Policy:
        """Backwards-compatible alias for the Russian preset."""
        return cls.for_language("ru")

    @classmethod
    def strict(cls, language: str = "ru") -> Policy:
        """Nothing identifying leaves, at the cost of utility."""
        return cls(language=language, default_action=Action.BLOCK, rules=[])

    @classmethod
    def off(cls, language: str = "ru") -> Policy:
        """Explicit, auditable opt-out — e.g. a genuinely local Ollama route."""
        return cls(language=language, scope=Scope.OFF, secrets=False, entities=[])


__all__ = [
    "ALL_ENTITIES",
    "GLOBAL_ENTITIES",
    "HIGH_RISK_NATIONAL",
    "SECRET_ENTITIES",
    "EntityRule",
    "Policy",
    "Scope",
    "UnsupportedLanguageError",
]
