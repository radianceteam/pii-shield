"""pii-shield — reversible PII pseudonymization for text leaving for cloud LLMs.

    from pii_shield import Shield, Policy

    shield = Shield(Policy.for_language("ja"))
    safe = shield.anonymize("田中太郎さんに090-1234-5678で連絡してください")
    answer = call_the_model(safe.text)
    real = shield.deanonymize(answer, safe.session_id, consume=True)

Importing this module pulls in no NLP stack; the Presidio-backed detector is loaded
on first use (and eagerly validated when a Shield is constructed, so a missing model
fails at startup rather than silently passing names through).
"""

from .languages import (
    PROFILES,
    SUPPORTED_LANGUAGES,
    LanguageProfile,
    UnsupportedLanguageError,
    get_profile,
)
from .policy import ALL_ENTITIES, HIGH_RISK_NATIONAL, EntityRule, Policy, Scope
from .session import SessionStore
from .shield import Shield
from .streaming import StreamDeanonymizer
from .types import (
    Action,
    AnonymizeResult,
    BlockedError,
    Finding,
    RedactionUnavailableError,
    ShieldError,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "ALL_ENTITIES",
    "HIGH_RISK_NATIONAL",
    "PROFILES",
    "SUPPORTED_LANGUAGES",
    "LanguageProfile",
    "UnsupportedLanguageError",
    "get_profile",
    "Action",
    "AnonymizeResult",
    "BlockedError",
    "EntityRule",
    "Finding",
    "Policy",
    "RedactionUnavailableError",
    "Scope",
    "SessionStore",
    "Shield",
    "ShieldError",
    "StreamDeanonymizer",
]
