"""Russian structured identifiers: INN, SNILS, OGRN, passport, phone, bank account.

Presidio ships no recognizers for any of these. They are also the types where a
pattern layer genuinely beats NER, because most of them carry a checksum: verifying
it turns "eleven digits" into "an actual SNILS" and collapses the false-positive
rate that makes bare digit patterns unusable in a document full of identifiers,
prices and dates.

Everything here is dependency-free on purpose, so the package detects structured
Russian PII even when the NER extra is not installed.
"""

from __future__ import annotations

import re

from ..types import Action, Finding

# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------
_INN10_WEIGHTS = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_WEIGHTS_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN12_WEIGHTS_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _digit_check(digits: str, weights: tuple[int, ...]) -> int:
    return sum(int(d) * w for d, w in zip(digits, weights, strict=False)) % 11 % 10


def valid_inn(value: str) -> bool:
    """INN: 10 digits (organization) or 12 (individual), each with its own check digit."""
    if not value.isdigit():
        return False
    if len(value) == 10:
        return _digit_check(value[:9], _INN10_WEIGHTS) == int(value[9])
    if len(value) == 12:
        return (
            _digit_check(value[:10], _INN12_WEIGHTS_11) == int(value[10])
            and _digit_check(value[:11], _INN12_WEIGHTS_12) == int(value[11])
        )
    return False


def valid_snils(value: str) -> bool:
    """SNILS: 11 digits; the last two are a mod-101 check over the first nine.

    Numbers below 001-001-998 are unissued and have no valid checksum, so they are
    rejected outright rather than being accepted by the ``< 100`` branch.
    """
    if not value.isdigit() or len(value) != 11:
        return False
    if int(value[:9]) <= 1001998:
        return False
    total = sum(int(d) * (9 - i) for i, d in enumerate(value[:9]))
    if total < 100:
        check = total
    elif total in (100, 101):
        check = 0
    else:
        check = total % 101
        if check in (100, 101):
            check = 0
    return check == int(value[9:])


def valid_ogrn(value: str) -> bool:
    """OGRN (13 digits) / OGRNIP (15): last digit checks the rest mod 11 / mod 13."""
    if not value.isdigit():
        return False
    if len(value) == 13:
        return int(value[:12]) % 11 % 10 == int(value[12])
    if len(value) == 15:
        return int(value[:14]) % 13 % 10 == int(value[14])
    return False


# ---------------------------------------------------------------------------
# Surrogate construction
# ---------------------------------------------------------------------------
# A stand-in for a checksummed identifier has to satisfy that checksum. An INN field
# holding "<RU_INN_1>" — or holding digits that fail validation — is corrupted data,
# not anonymized data: the recipient's own form rejects it and the failure reads as a
# bug rather than as policy.
def make_inn(body9: str) -> str:
    """Complete a 9-digit body into a valid 10-digit organization INN."""
    return body9 + str(_digit_check(body9, _INN10_WEIGHTS))


def make_ogrn(body12: str) -> str:
    """Complete a 12-digit body into a valid 13-digit OGRN."""
    return body12 + str(int(body12) % 11 % 10)


def make_bik(body7: str) -> str:
    """BIK (bank identifier): the 04 prefix every issued code carries, plus seven digits."""
    return "04" + body7


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
# Digit runs are captured with boundaries that reject a longer number, so a 20-digit
# account is never reported as the 13-digit OGRN hiding inside it.
#
# The boundary excludes letters as well as digits. A German IBAN, DE89370400440532013000,
# carries exactly twenty digits after its country code, and a digits-only boundary read
# that as a Russian bank account — which the default policy blocks, so a perfectly
# ordinary European payment was refused outright.
_B = r"(?<![0-9A-Za-z])"
_E = r"(?![0-9A-Za-z])"

INN_RE = re.compile(rf"{_B}(\d{{12}}|\d{{10}}){_E}")
SNILS_RE = re.compile(rf"{_B}(\d{{3}})[\s-]?(\d{{3}})[\s-]?(\d{{3}})[\s-]?(\d{{2}}){_E}")
OGRN_RE = re.compile(rf"{_B}(\d{{15}}|\d{{13}}){_E}")
BANK_ACCOUNT_RE = re.compile(rf"{_B}\d{{20}}{_E}")
# BIK: nine digits, and every issued one begins with 04.
BIK_RE = re.compile(rf"{_B}04\d{{7}}{_E}")
PHONE_RE = re.compile(
    r"(?<![0-9])(?:\+7|8)[\s\-.]?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{2}[\s\-.]?\d{2}(?![0-9])"
)
# Passport has no checksum, so it is only reported near a context word. Without that
# guard every "1234 567890" in a price table or a spec becomes a passport number.
PASSPORT_RE = re.compile(rf"{_B}(\d{{4}})[\s-]?(\d{{6}}){_E}")
# Context words as they appear in Russian documents — matched literally.
_PASSPORT_CONTEXT_RE = re.compile(
    r"паспорт|серия|passport|удостоверен", re.IGNORECASE
)
_PASSPORT_CONTEXT_WINDOW = 40


def _has_context(text: str, start: int, window: int, probe: re.Pattern[str]) -> bool:
    return bool(probe.search(text[max(0, start - window) : start + window]))


def scan(text: str, *, entities: set[str] | None = None) -> list[Finding]:
    """Detect Russian structured identifiers in *text*.

    Checked types score 0.95 (a passed checksum is near-proof); unchecked ones score
    0.6 so a caller can raise the threshold to exclude them without touching the rest.
    """
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
            Finding(entity=entity, start=start, end=end, score=score, action=Action.MASK,
                    recognizer=f"ru:{entity.lower()}")
        )

    # Longest / most-specific first, so the overlap guard keeps the right one.
    if wanted("RU_BANK_ACCOUNT"):
        for m in BANK_ACCOUNT_RE.finditer(text):
            claim("RU_BANK_ACCOUNT", *m.span(), 0.6)

    if wanted("RU_OGRN"):
        for m in OGRN_RE.finditer(text):
            if valid_ogrn(m.group(1)):
                claim("RU_OGRN", *m.span(), 0.95)

    if wanted("RU_SNILS"):
        for m in SNILS_RE.finditer(text):
            if valid_snils("".join(m.groups())):
                claim("RU_SNILS", *m.span(), 0.95)

    if wanted("RU_INN"):
        for m in INN_RE.finditer(text):
            if valid_inn(m.group(1)):
                claim("RU_INN", *m.span(), 0.95)

    if wanted("RU_PASSPORT"):
        for m in PASSPORT_RE.finditer(text):
            if _has_context(text, m.start(), _PASSPORT_CONTEXT_WINDOW, _PASSPORT_CONTEXT_RE):
                claim("RU_PASSPORT", *m.span(), 0.7)

    if wanted("RU_BIK"):
        for m in BIK_RE.finditer(text):
            claim("RU_BIK", *m.span(), 0.8)

    if wanted("RU_PHONE"):
        for m in PHONE_RE.finditer(text):
            claim("RU_PHONE", *m.span(), 0.85)

    found.sort(key=lambda f: f.start)
    return found
