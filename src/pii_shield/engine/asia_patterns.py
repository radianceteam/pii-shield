"""Chinese, Japanese and Korean national identifiers and phone numbers.

Presidio ships no recognizers for any CJK locale — its country-specific set covers
only en, es, it and pl, verified by enumerating its registry. These are the same kind
of high-value, checksum-bearing types that make the Russian layer worth having: a
verified checksum turns "eighteen digits" into "an actual 身份证".

Dependency-free, so CJK structured identifiers are found even without the NER extra.
"""

from __future__ import annotations

import re

from ..types import Action, Finding

_B = r"(?<![0-9])"
_E = r"(?![0-9])"


# ---------------------------------------------------------------------------
# China — 居民身份证 (resident identity card)
# ---------------------------------------------------------------------------
_CN_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_CN_CHECK_CHARS = "10X98765432"


def valid_cn_resident_id(value: str) -> bool:
    """18 characters: 17 digits plus an ISO 7064 MOD 11-2 check character (0-9 or X)."""
    if len(value) != 18 or not value[:17].isdigit():
        return False
    if value[17] not in "0123456789Xx":
        return False
    # Digits 7-14 are the date of birth; an impossible date is not an ID.
    if not _plausible_date(value[6:10], value[10:12], value[12:14]):
        return False
    total = sum(int(d) * w for d, w in zip(value[:17], _CN_WEIGHTS, strict=True))
    return _CN_CHECK_CHARS[total % 11] == value[17].upper()


def _plausible_date(year: str, month: str, day: str) -> bool:
    try:
        y, m, d = int(year), int(month), int(day)
    except ValueError:
        return False
    return 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31


# ---------------------------------------------------------------------------
# Japan — マイナンバー (individual number)
# ---------------------------------------------------------------------------
def valid_jp_my_number(value: str) -> bool:
    """12 digits: an 11-digit body plus a mod-11 check digit.

    Per the Japanese ministerial ordinance: with P_n the n-th digit counted from the
    right of the body, Q_n is n+1 for n in 1..6 and n-5 for n in 7..11; the check
    digit is 11 - (Σ P_n·Q_n mod 11), or 0 when that result is 10 or 11.
    """
    if len(value) != 12 or not value.isdigit():
        return False
    # 000000000000 satisfies the arithmetic but is not an issued number; the same is
    # true of any single repeated digit, and those appear in test fixtures constantly.
    if len(set(value)) == 1:
        return False
    body = value[:11]
    total = 0
    for n in range(1, 12):
        p = int(body[11 - n])           # n-th digit from the right of the body
        q = n + 1 if n <= 6 else n - 5
        total += p * q
    remainder = total % 11
    check = 0 if remainder <= 1 else 11 - remainder
    return check == int(value[11])


# ---------------------------------------------------------------------------
# Korea — 주민등록번호 (resident registration number)
# ---------------------------------------------------------------------------
_KR_WEIGHTS = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)


def kr_rrn_structure_ok(digits: str) -> bool:
    """YYMMDD plus a century/gender digit of 1-8 and six more digits."""
    if len(digits) != 13 or not digits.isdigit():
        return False
    if digits[6] not in "12345678":
        return False
    month, day = int(digits[2:4]), int(digits[4:6])
    return 1 <= month <= 12 and 1 <= day <= 31


def valid_kr_rrn_checksum(digits: str) -> bool:
    """The pre-2020 check digit.

    South Korea randomized digits 7-13 in October 2020, so numbers issued since then
    do not satisfy this at all. It is therefore used to *raise* confidence, never as a
    precondition — rejecting every post-2020 number would be worse than a few false
    positives on thirteen-digit strings.
    """
    if not kr_rrn_structure_ok(digits):
        return False
    total = sum(int(d) * w for d, w in zip(digits[:12], _KR_WEIGHTS, strict=True))
    return (11 - total % 11) % 10 == int(digits[12])


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
CN_ID_RE = re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])")
CN_PHONE_RE = re.compile(rf"{_B}(?:\+?86[-\s]?)?1[3-9]\d{{9}}{_E}")

JP_MY_NUMBER_RE = re.compile(rf"{_B}\d{{4}}[-\s]?\d{{4}}[-\s]?\d{{4}}{_E}")
# Mobile (0[789]0) and Tokyo/Osaka-style landlines; the leading zero is the tell.
JP_PHONE_RE = re.compile(rf"{_B}0\d{{1,3}}[-\s]?\d{{2,4}}[-\s]?\d{{4}}{_E}")

KR_RRN_RE = re.compile(rf"{_B}(\d{{6}})[-\s]?([1-8]\d{{6}}){_E}")
KR_PHONE_RE = re.compile(rf"{_B}01[016789][-\s]?\d{{3,4}}[-\s]?\d{{4}}{_E}")

# Which entities each language's scanner is responsible for.
LANGUAGE_ENTITIES = {
    "zh": ("CN_RESIDENT_ID", "CN_PHONE"),
    "ja": ("JP_MY_NUMBER", "JP_PHONE"),
    "ko": ("KR_RRN", "KR_PHONE"),
}


def scan(text: str, *, entities: set[str] | None = None) -> list[Finding]:
    """Detect CJK structured identifiers.

    Checksum-verified types score 0.95. Phone numbers score 0.8. A Korean RRN whose
    checksum fails but whose structure holds scores 0.6, so a caller can exclude the
    post-2020 randomized numbers by raising the threshold without losing the rest.
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
                    recognizer=f"asia:{entity.lower()}")
        )

    # Longest and most-specific first, so a shorter pattern cannot claim part of an ID.
    if wanted("CN_RESIDENT_ID"):
        for m in CN_ID_RE.finditer(text):
            if valid_cn_resident_id(m.group(0)):
                claim("CN_RESIDENT_ID", *m.span(), 0.95)

    if wanted("KR_RRN"):
        for m in KR_RRN_RE.finditer(text):
            digits = m.group(1) + m.group(2)
            if valid_kr_rrn_checksum(digits):
                claim("KR_RRN", *m.span(), 0.95)
            elif kr_rrn_structure_ok(digits):
                claim("KR_RRN", *m.span(), 0.6)

    if wanted("JP_MY_NUMBER"):
        for m in JP_MY_NUMBER_RE.finditer(text):
            digits = re.sub(r"[-\s]", "", m.group(0))
            if valid_jp_my_number(digits):
                claim("JP_MY_NUMBER", *m.span(), 0.95)

    if wanted("CN_PHONE"):
        for m in CN_PHONE_RE.finditer(text):
            claim("CN_PHONE", *m.span(), 0.8)

    if wanted("KR_PHONE"):
        for m in KR_PHONE_RE.finditer(text):
            claim("KR_PHONE", *m.span(), 0.8)

    if wanted("JP_PHONE"):
        for m in JP_PHONE_RE.finditer(text):
            claim("JP_PHONE", *m.span(), 0.8)

    found.sort(key=lambda f: f.start)
    return found
