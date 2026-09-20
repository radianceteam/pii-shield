"""Financial institution and entity identifier codes: SWIFT/BIC, LEI.

These are language-independent, so they run for every language alongside the
credential layer rather than inside a national profile.

A note on what they are, because it changes how the policy should treat them: a
SWIFT/BIC identifies a *bank* and an LEI identifies a *legal entity*. Neither is
personal data on its own under GDPR or Russian law 152-FZ. They matter here for two
First, they travel in the same payment block as a name and an account number, and the
account number is personal data. Second — and this is what actually forced the
module — an unrecognized BIC does not pass through untouched: an NER model reads
``DEUTDEFF`` as an organization and replaces it with an invented company name, which
corrupts a payment instruction instead of protecting it. Naming the entity explicitly
is what stops that.
"""

from __future__ import annotations

import re

from ..types import Action, Finding

# ISO 3166-1 alpha-2. Positions 5-6 of a BIC are a country code, and checking them is
# what separates a real BIC from any other eight-character uppercase token.
ISO_3166_ALPHA2 = frozenset("""
AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN
BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ
DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA GB GD GE GF GG GH GI GL
GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM
JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME
MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP
NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD
SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO
TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS XK YE YT ZA ZM ZW
""".split())

# Bank code (4 letters), country (2), location (2), optional branch (3).
BIC_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?(?![A-Z0-9])")

# Words that turn a plausible token into a confident BIC. Without one, an ordinary
# eight-letter uppercase word can match: "SOMEUSER" has "US" in positions 5-6.
# Context words in the languages this is likely to meet, matched literally: a BIC
# in a Russian payment instruction sits next to Russian words, not English ones.
_BIC_CONTEXT_RE = re.compile(
    r"swift|bic\b|б\.?и\.?к|свифт|банк|bank|beneficiary|correspondent|получател",
    re.IGNORECASE,
)
_CONTEXT_WINDOW = 48

# ISO 17442: 18 alphanumerics plus two check digits, validated by ISO 7064 MOD 97-10.
LEI_RE = re.compile(r"(?<![A-Z0-9])[A-Z0-9]{18}[0-9]{2}(?![A-Z0-9])")

# US ABA routing transit number: nine digits with a weighted mod-10 check.
ABA_RE = re.compile(r"(?<![0-9])\d{9}(?![0-9])")
_ABA_WEIGHTS = (3, 7, 1, 3, 7, 1, 3, 7, 1)
# Federal Reserve routing symbol ranges. Nine digits are cheap to come by and the
# checksum alone passes for roughly one random number in ten, so the prefix is what
# keeps an order reference from being read as a bank.
_ABA_PREFIX_RANGES = ((1, 12), (21, 32), (61, 72), (80, 80))
_ABA_CONTEXT_RE = re.compile(
    r"aba|routing|rtn\b|ach\b|wire|transit|bank|счёт|счет|банк", re.IGNORECASE
)


def _mod97(value: str) -> int:
    """ISO 7064 MOD 97-10 over a string where A-Z expand to 10-35."""
    digits = "".join(str(int(ch, 36)) if ch.isalpha() else ch for ch in value)
    remainder = 0
    for chunk_start in range(0, len(digits), 9):
        remainder = int(str(remainder) + digits[chunk_start : chunk_start + 9]) % 97
    return remainder


def valid_bic(value: str) -> bool:
    """Structurally a BIC, with a real country code in positions 5-6."""
    if len(value) not in (8, 11) or not value.isalnum() or not value.isupper():
        return False
    if not value[:6].isalpha():
        return False
    return value[4:6] in ISO_3166_ALPHA2


def valid_lei(value: str) -> bool:
    if len(value) != 20 or not value.isalnum() or not value.isupper():
        return False
    if not value[18:].isdigit():
        return False
    return _mod97(value) == 1


def valid_aba(value: str) -> bool:
    """Nine digits, a Federal Reserve prefix, and the 3-7-1 weighted mod-10 check."""
    if len(value) != 9 or not value.isdigit():
        return False
    prefix = int(value[:2])
    if not any(low <= prefix <= high for low, high in _ABA_PREFIX_RANGES):
        return False
    total = sum(int(d) * w for d, w in zip(value, _ABA_WEIGHTS, strict=True))
    return total % 10 == 0


# ---------------------------------------------------------------------------
# IBAN
# ---------------------------------------------------------------------------
# Presidio recognizes IBANs, but only once it is installed. An IBAN carries an
# ISO 7064 MOD 97-10 checksum and a country-specific length, which is exactly the
# class the dependency-free tier is meant to cover — leaving it to Presidio meant a
# deployment too small for that dependency sent every IBAN through untouched.
IBAN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{10,30}(?![A-Z0-9])")

# Length per country, from the IBAN registry. A German IBAN is 22 characters; a
# 22-character string starting "DE" that fails this table is not one.
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22,
    "BH": 22, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22,
    "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24, "FI": 18, "FO": 18, "FR": 27,
    "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HR": 21, "HU": 28,
    "IE": 22, "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20,
    "LB": 28, "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "LY": 25, "MC": 27,
    "MD": 24, "ME": 22, "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15,
    "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "RU": 33,
    "SA": 24, "SC": 31, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25, "SV": 28,
    "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}


def valid_iban(value: str) -> bool:
    """Known country, right length for it, and the mod-97 checksum."""
    value = value.replace(" ", "").upper()
    expected = IBAN_LENGTHS.get(value[:2])
    if expected is None or len(value) != expected:
        return False
    if not value[2:4].isdigit() or not value[4:].isalnum():
        return False
    return _mod97(value[4:] + value[:4]) == 1


# ---------------------------------------------------------------------------
# Payment cards
# ---------------------------------------------------------------------------
# Presidio ships a credit-card recognizer for English, Spanish, Italian and Polish
# only. In Russian — and in every other language — a card number passed through
# untouched while the default policy claimed to block it. Detecting cards here makes
# it language-independent, and dependency-free, which the pattern-only tier needs.
CARD_RE = re.compile(r"(?<![0-9A-Za-z])(?:\d[ -]?){12,18}\d(?![0-9A-Za-z])")

# Issuer identification numbers, as (prefix, accepted lengths). Luhn alone passes one
# random number in ten; requiring a real issuer prefix is what makes this usable.
_IIN_RULES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("4", (13, 16, 19)),                                   # Visa
    *((str(p), (16,)) for p in range(51, 56)),             # Mastercard
    *((str(p), (16,)) for p in range(2221, 2721)),         # Mastercard (2-series)
    ("34", (15,)), ("37", (15,)),                          # American Express
    ("6011", (16, 19)), ("65", (16, 19)),                  # Discover
    *((str(p), (16, 19)) for p in range(644, 650)),        # Discover
    *((str(p), (16,)) for p in range(2200, 2205)),         # Mir
    ("62", (16, 17, 18, 19)),                              # UnionPay
    *((str(p), (16, 17, 18, 19)) for p in range(3528, 3590)),   # JCB
    ("36", (14, 15, 16, 17, 18, 19)),                      # Diners
    *((str(p), (14,)) for p in range(300, 306)),           # Diners
)


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def valid_card(digits: str) -> bool:
    """Luhn plus a recognized issuer prefix of the right length for that issuer."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    if len(set(digits)) == 1:
        return False
    if not any(
        digits.startswith(prefix) and len(digits) in lengths for prefix, lengths in _IIN_RULES
    ):
        return False
    return luhn_ok(digits)


def make_iban(country: str, body: str) -> str:
    """Build a valid IBAN for *country* from a supplied body of the right length.

    The country is taken from the value being replaced rather than from the locale.
    A German IBAN that comes back Russian has changed which country the money goes
    to — the stand-in stops being plausible exactly where plausibility matters.
    """
    country = country.upper()
    length = IBAN_LENGTHS.get(country)
    if length is None:
        return ""
    body = (body.upper() + "0" * length)[: length - 4]
    check = 98 - _mod97(body + country + "00")
    return f"{country}{check:02d}{body}"


def make_bic(country: str, bank: str, location: str, branch: str = "") -> str:
    """Build a BIC keeping the country of the value being replaced."""
    return f"{bank[:4].upper():X<4}{country.upper()}{location[:2].upper():X<2}{branch[:3].upper()}"


def lei_check_digits(prefix18: str) -> str:
    """Two ISO 7064 MOD 97-10 check digits for an 18-character LEI prefix.

    Needed because a surrogate LEI has to *be* a valid LEI. Replacing a real one with
    a random string would fail the recipient's own validation, turning a privacy
    measure into a data-quality bug.
    """
    remainder = _mod97(prefix18 + "00")
    return f"{(98 - remainder) % 97:02d}"


def _has_context(text: str, start: int, end: int, probe: re.Pattern[str]) -> bool:
    window = text[max(0, start - _CONTEXT_WINDOW) : end + _CONTEXT_WINDOW]
    return bool(probe.search(window))


def scan(text: str, *, entities: set[str] | None = None) -> list[Finding]:
    """Detect institution identifier codes.

    An LEI carries a checksum, so it scores 0.95 on its own. A BIC does not, so it
    scores 0.9 only next to a word like "SWIFT" or "bank" (in any of the listed
    languages), and 0.4 otherwise — below
    the default threshold, which means an eight-letter uppercase token in ordinary
    technical prose is reported but not acted on unless a caller lowers the bar.
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
                    recognizer=f"finance:{entity.lower()}")
        )

    if wanted("IBAN_CODE"):
        for m in IBAN_RE.finditer(text):
            if valid_iban(m.group(0)):
                claim("IBAN_CODE", *m.span(), 0.95)

    if wanted("CREDIT_CARD"):
        for m in CARD_RE.finditer(text):
            digits = re.sub(r"[ -]", "", m.group(0))
            if valid_card(digits):
                claim("CREDIT_CARD", *m.span(), 0.95)

    if wanted("LEI"):
        for m in LEI_RE.finditer(text):
            if valid_lei(m.group(0)):
                claim("LEI", *m.span(), 0.95)

    if wanted("SWIFT_BIC"):
        for m in BIC_RE.finditer(text):
            if not valid_bic(m.group(0)):
                continue
            score = 0.9 if _has_context(text, m.start(), m.end(), _BIC_CONTEXT_RE) else 0.4
            claim("SWIFT_BIC", *m.span(), score)

    if wanted("ABA_ROUTING"):
        for m in ABA_RE.finditer(text):
            if not valid_aba(m.group(0)):
                continue
            score = 0.9 if _has_context(text, m.start(), m.end(), _ABA_CONTEXT_RE) else 0.4
            claim("ABA_ROUTING", *m.span(), score)

    found.sort(key=lambda f: f.start)
    return found
