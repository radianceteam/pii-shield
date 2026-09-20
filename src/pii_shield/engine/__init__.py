"""Detection layers, merged into one span list.

Three kinds of detector run over the same text and their results are merged by
precedence rather than by score: a credential pattern and a checksum-verified national
identifier are structurally certain, while an NER span is a guess, so a Presidio PERSON
overlapping a verified 身份证 loses regardless of the score Presidio assigned it.
"""

from __future__ import annotations

from ..types import Finding
from . import asia_patterns, finance_patterns, ru_patterns
from .presidio_engine import NER_ENTITIES, NerUnavailableError, PresidioDetector
from .surrogates import SurrogateFactory

__all__ = [
    "merge_findings",
    "national_scan",
    "NATIONAL_SCANNERS",
    "NER_ENTITIES",
    "NerUnavailableError",
    "PresidioDetector",
    "SurrogateFactory",
    "asia_patterns",
    "finance_patterns",
    "ru_patterns",
]

# Language -> the pattern module responsible for its national identifiers. A language
# with no entry simply has no local pattern layer; Presidio may still cover it (en, es,
# it and pl have built-in recognizers) or nothing may, which the profile makes explicit.
NATIONAL_SCANNERS = {
    "ru": ru_patterns.scan,
    "zh": asia_patterns.scan,
    "ja": asia_patterns.scan,
    "ko": asia_patterns.scan,
}


def national_scan(language: str, text: str, entities: set[str] | None = None) -> list[Finding]:
    """Run the local national-identifier layer for *language*, if there is one."""
    scanner = NATIONAL_SCANNERS.get(language)
    return scanner(text, entities=entities) if scanner else []


def merge_findings(*layers: list[Finding]) -> list[Finding]:
    """Combine detector outputs, dropping any span that overlaps an earlier one.

    Layers are passed most-authoritative first.
    """
    kept: list[Finding] = []
    for layer in layers:
        for finding in layer:
            if any(finding.start < k.end and k.start < finding.end for k in kept):
                continue
            kept.append(finding)
    kept.sort(key=lambda f: (f.start, f.end))
    return kept
