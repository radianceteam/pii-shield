"""Email addresses and international phone numbers, without any dependency.

Presidio recognizes both, but only once it is installed. A deployment sized for an
agent rather than for a language model has no Presidio at all, and for a customer
writing in anything but Russian that left the cheap tier protecting almost nothing:
the national identifiers it does cover are Russian, Chinese, Japanese and Korean.

An email and an E.164 number are fixed shapes. They belong to the tier that is defined
by fixed shapes.

This layer runs only when Presidio is absent. Presidio's phone recognizer validates
against real numbering plans through ``phonenumbers``, which is better than a regex,
and the full tier keeps that.
"""

from __future__ import annotations

import re

from ..types import Action, Finding

# Deliberately not the full RFC 5322 grammar, which matches things no mail server
# would accept. This is the shape addresses are actually written in.
# Letters, not just ASCII: internationalized domains are ordinary for the audiences
# this is aimed at, and "почта@пример.рф" is an address like any other.
#
# The lookbehind covers the whole local-part alphabet, not only letters. Excluding
# letters alone let a match begin *after* a dot, so "john.smith@acme.co.uk" matched
# from "smith" and left "john." sitting in the outbound text — a first name shipped
# in plain view next to a replaced address.
EMAIL_RE = re.compile(
    r"(?<![\w.%+\-])"
    r"[\w.%+\-]+@[\w\-]+(?:\.[\w\-]+)*\.[^\W\d_]{2,24}"
    r"(?![\w\-])",
    re.UNICODE,
)

# E.164 with the separators people actually type. The leading + is what makes this
# safe to match without context: a bare run of digits is an order number as often as
# a telephone, but "+" followed by 8 to 15 digits is a telephone.
INTL_PHONE_RE = re.compile(
    r"(?<![0-9+])\+[1-9]\d{0,3}[\s.\-]?(?:\(?\d{1,4}\)?[\s.\-]?){1,5}\d{2,4}(?![0-9])"
)

LANGUAGE_INDEPENDENT_ENTITIES = ("EMAIL_ADDRESS", "PHONE_NUMBER")


def _digit_count(value: str) -> int:
    return sum(1 for c in value if c.isdigit())


def scan(text: str, *, entities: set[str] | None = None) -> list[Finding]:
    """Detect contact details that need no language model and no third-party package."""
    if not text:
        return []

    def wanted(entity: str) -> bool:
        return entities is None or entity in entities

    taken: list[tuple[int, int]] = []
    found: list[Finding] = []

    def claim(entity: str, start: int, end: int, score: float) -> None:
        if any(start < t_end and t_start < end for t_start, t_end in taken):
            return
        taken.append((start, end))
        found.append(
            Finding(entity=entity, start=start, end=end, score=score,
                    recognizer=f"contact:{entity.lower()}", action=Action.MASK)
        )

    if wanted("EMAIL_ADDRESS"):
        for m in EMAIL_RE.finditer(text):
            claim("EMAIL_ADDRESS", *m.span(), 0.95)

    if wanted("PHONE_NUMBER"):
        for m in INTL_PHONE_RE.finditer(text):
            # E.164 allows 15 digits including the country code; fewer than 8 is a
            # version string or a maths expression far more often than a number.
            if 8 <= _digit_count(m.group(0)) <= 15:
                claim("PHONE_NUMBER", *m.span(), 0.8)

    found.sort(key=lambda f: f.start)
    return found
