"""Surrogate values: replace a real entity with a plausible fake one.

Why plausible rather than ``<PERSON>``: a placeholder changes the shape of the text
the model reasons over. "Bitte <PERSON> bis <DATE_TIME> informieren" reads as a template,
and models answer templates with templates. A real-looking name keeps the sentence a
sentence, which is the whole argument for surrogates over masking.

Two properties the rest of the shield depends on:

  * **stable within a session** — one real value always maps to the same fake value,
    so a surname mentioned five times stays one person rather than becoming five;
  * **unique** — two different real values never collide onto one surrogate, because
    a collision makes deanonymization ambiguous and silently corrupts the answer.

Faker is optional. Without it the built-in pools below are used, which are smaller
but need no dependency; the uniqueness counter makes both paths exhaust-proof.
"""

from __future__ import annotations

import hashlib

try:  # pragma: no cover - exercised by both paths in tests via _FAKER_AVAILABLE
    from faker import Faker

    _FAKER_AVAILABLE = True
except ImportError:  # pragma: no cover
    Faker = None  # type: ignore[assignment]
    _FAKER_AVAILABLE = False


# Fallback pools for when Faker is not installed. Russian only, and used only for a
# Russian locale: a pool per language would be an unbounded amount of curated data,
# and a Russian name in an English sentence is worse than an honest typed token.
_FALLBACK_POOLS: dict[str, tuple[str, ...]] = {  # noqa: RUF001 - names, not prose
    "PERSON": (
        "Иван Иванов", "Пётр Петров", "Сергей Сергеев", "Анна Смирнова",
        "Мария Кузнецова", "Алексей Попов", "Ольга Соколова", "Дмитрий Новиков",
    ),
    "ORGANIZATION": (
        'ООО "Ромашка"', 'ООО "Василёк"', 'АО "Незабудка"', 'ООО "Одуванчик"',
    ),
    "LOCATION": ("Москва", "Санкт-Петербург", "Казань", "Новосибирск"),
    "EMAIL_ADDRESS": ("user@example.com", "contact@example.org", "info@example.net"),
    "PHONE_NUMBER": ("+7 900 000-00-01", "+7 900 000-00-02", "+7 900 000-00-03"),
    "RU_PHONE": ("+7 900 000-00-01", "+7 900 000-00-02", "+7 900 000-00-03"),
    "IP_ADDRESS": ("192.0.2.1", "192.0.2.2", "198.51.100.1"),
    "URL": ("https://example.com", "https://example.org"),
}

# Every national phone entity draws from Faker's locale-aware phone provider, so a
# Korean number is replaced by a Korean-shaped one rather than a bare token.
# Entities where reusing a word of the original would leak it. Structured values are
# excluded on purpose: a phone deliberately keeps its country code, a card keeps its
# issuer digit, and "sharing a token" with a number means nothing. Applying the name
# rule to them rejected every candidate and fell through to the "#1" suffix, which
# turned a phone number into something that is not one.
_FREE_TEXT_ENTITIES = frozenset({
    "PERSON", "ORGANIZATION", "LOCATION", "NRP", "EMAIL_ADDRESS", "URL",
})

_PHONE_ENTITIES = frozenset({
    "PHONE_NUMBER", "RU_PHONE", "CN_PHONE", "JP_PHONE", "KR_PHONE",
})

# Financial codes whose stand-in must itself be structurally valid. A payment
# instruction carrying a malformed BIC or an IBAN that fails its check digits is not
# anonymized data, it is corrupted data — the recipient's own validation rejects it,
# and the failure looks like a bug rather than a policy.
_FINANCE_ENTITIES = frozenset({"SWIFT_BIC", "IBAN_CODE", "ABA_ROUTING", "LEI", "CREDIT_CARD"})

def _ru_inn(f) -> str:
    from .ru_patterns import make_inn

    return make_inn(f.numerify("#########"))


def _ru_ogrn(f) -> str:
    from .ru_patterns import make_ogrn

    return make_ogrn(f.numerify("1###########"))


def _ru_bik(f) -> str:
    from .ru_patterns import make_bik

    return make_bik(f.numerify("#######"))


_RU_CODE_BUILDERS = {"RU_INN": _ru_inn, "RU_OGRN": _ru_ogrn, "RU_BIK": _ru_bik}

# Entities with no sensible human-readable stand-in get a typed token instead.
_TOKEN_ENTITIES = frozenset({
    "RU_SNILS", "RU_PASSPORT", "RU_BANK_ACCOUNT", "DATE_TIME",
})


class SurrogateFactory:
    """Mints surrogates for one session.

    ``seed`` makes a session reproducible for tests and for replaying a saved
    trajectory; it is derived from the session id by default, so two concurrent
    sessions never draw the same sequence.
    """

    def __init__(
        self, *, seed: str = "", locale: str | None = "ru_RU", use_faker: bool = True
    ) -> None:
        self._seed = seed
        self._locale = locale
        self._counters: dict[str, int] = {}
        self._issued: set[str] = set()
        self._faker = None
        # ``locale is None`` means the language has no Faker locale (Catalan, Norwegian
        # Bokmål). Falling back to the pools yields typed tokens rather than a name in
        # the wrong language, which is the lesser harm.
        if use_faker and _FAKER_AVAILABLE and locale:
            self._faker = Faker(locale)
            # int seed from the session id: same session, same cast of characters.
            self._faker.seed_instance(
                int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) if seed else 0
            )

    @property
    def backend(self) -> str:
        return "faker" if self._faker is not None else "pool"

    def reserve(self, surrogates) -> None:
        """Mark values as already handed out.

        A session spans many calls, but a factory is built per call. Re-seeding it
        from the session's existing map is what stops the second call from minting
        the same fake name again for a different person.
        """
        self._issued.update(s for s in surrogates if s)

    def make(self, entity: str, original: str) -> str:
        """Return a fresh surrogate for *entity*. Never returns a value twice."""
        for _ in range(64):
            candidate = self._draw(entity, original)
            if candidate in self._issued:
                continue
            if entity in _FREE_TEXT_ENTITIES and self._overlaps(candidate, original):
                continue
            if candidate == original:
                continue
            self._issued.add(candidate)
            return candidate
        # Exhausted. A free-text value can carry a disambiguating suffix; a structured
        # one cannot — "+49 30 12345678 #1" is not a phone number. Digits are nudged
        # instead, which keeps the shape and still yields a value not issued before.
        n = self._bump(entity)
        candidate = self._draw(entity, original)
        if entity in _FREE_TEXT_ENTITIES:
            candidate = f"{candidate} #{n}"
        else:
            candidate = self._nudge_digits(candidate, n)
        self._issued.add(candidate)
        return candidate

    @staticmethod
    def _nudge_digits(value: str, salt: int) -> str:
        """Change the trailing digits of a structured value, preserving its shape."""
        out = list(value)
        changed = 0
        for index in range(len(out) - 1, -1, -1):
            if out[index].isdigit():
                out[index] = str((int(out[index]) + salt + changed) % 10)
                changed += 1
                if changed >= 3:
                    break
        return "".join(out)

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _same_root(a: str, b: str) -> bool:
        """True if two words are the same name in different grammatical forms.

        Exact comparison is not enough in an inflected language. A live run turned
        "Petru Sidorovym" into "Yulia Ruslanovna Sidorova": the tokens differ by one
        letter, the uniqueness check passed, and the real surname went out anyway.
        Comparing a shared stem catches sidorov / sidorova / sidorovym while leaving
        genuinely unrelated names (john / joan, smith / smyth) alone.
        """
        a, b = a.casefold(), b.casefold()
        if a == b:
            return True
        if len(a) < 4 or len(b) < 4:
            return False
        stem = max(min(len(a), len(b)) - 2, 4)
        return a[:stem] == b[:stem]

    @classmethod
    def _overlaps(cls, candidate: str, original: str) -> bool:
        """True if the stand-in reuses any word of the real value.

        Drawing "Margaret Smith" for "John Smith" was observed in a live run. The
        surrogate is a different name, so the uniqueness check passes — but the real
        surname survives into the outbound text, which is a partial leak. Common
        surnames make this likely rather than exotic, so a shared word disqualifies
        the candidate outright.
        """
        if candidate == original:
            return True
        original_tokens = [t for t in original.split() if len(t) > 2]
        if not original_tokens:
            return False
        return any(
            cls._same_root(c, o) for c in candidate.split() for o in original_tokens
        )

    def _bump(self, entity: str) -> int:
        self._counters[entity] = self._counters.get(entity, 0) + 1
        return self._counters[entity]

    def _draw(self, entity: str, original: str = "") -> str:
        if entity in _TOKEN_ENTITIES:
            return f"<{entity}_{self._bump(entity)}>"
        if self._faker is not None:
            drawn = self._from_faker(entity, original)
            if drawn is not None:
                return drawn
        pool = _FALLBACK_POOLS.get(entity) if self._pools_apply() else None
        if not pool:
            return f"<{entity}_{self._bump(entity)}>"
        return pool[self._bump(entity) % len(pool) - 1]

    @staticmethod
    def _make_iban(f, original: str) -> str:
        """An IBAN for the same country as the one being replaced."""
        from .finance_patterns import IBAN_LENGTHS, make_iban

        country = original.replace(" ", "").upper()[:2]
        if country not in IBAN_LENGTHS:
            return f.iban()
        length = IBAN_LENGTHS[country]
        body = f.bothify("#" * (length - 4))
        return make_iban(country, body) or f.iban()

    @staticmethod
    def _make_bic(f, original: str) -> str:
        """A BIC for the same country as the one being replaced."""
        from .finance_patterns import ISO_3166_ALPHA2, make_bic

        value = original.strip().upper()
        country = value[4:6] if len(value) in (8, 11) else ""
        if country not in ISO_3166_ALPHA2:
            return f.swift()
        branch = f.bothify("???", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ") if len(value) == 11 else ""
        return make_bic(
            country,
            f.bothify("????", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            f.bothify("??", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            branch,
        )

    @staticmethod
    def _shape_like(original: str, generated: str) -> str:
        """Lay the generated digits into the original's punctuation.

        Faker picks a format at random, so "+7 900 123-45-67" could come back as
        "+79001234567" — same information, different shape, which reads as noise both
        to the model and to whoever reviews the outbound traffic. Keeping the
        separators of the value being replaced costs nothing and keeps the sentence
        looking like the sentence it was.
        """
        digits = [c for c in generated if c.isalnum()]
        if not original or not digits:
            return generated
        needed = sum(1 for c in original if c.isalnum())
        if not needed:
            return generated

        # An international prefix is kept as it stands. Replacing "+49 30 ..." with
        # "+73 20 ..." invents a country that does not dial, which is a different
        # kind of wrong from hiding whose number it was — the same reasoning that
        # keeps a German IBAN German.
        keep = 0
        if original.startswith("+"):
            for index, char in enumerate(original[1:4], start=1):
                if not char.isdigit():
                    break
                keep = index
        # keep == 0 means there is no dialling prefix to preserve; slicing to keep + 1
        # would still capture the first character, which corrupted every IBAN.
        prefix_digits = [c for c in original[: keep + 1] if c.isalnum()] if keep else []

        while len(digits) < needed:
            digits += digits
        tail = iter(digits[: needed - len(prefix_digits)])
        head = iter(prefix_digits)
        out = []
        taken = 0
        for char in original:
            if not char.isalnum():
                out.append(char)
                continue
            out.append(next(head) if taken < len(prefix_digits) else next(tail))
            taken += 1
        return "".join(out)

    @staticmethod
    def _make_card(f, original: str) -> str:
        """Card-shaped, same length, deliberately not Luhn-valid — see finance_patterns.

        Keeping the issuer's leading digit preserves the shape without making the
        result chargeable: the check digit is wrong by construction.
        """
        from .finance_patterns import make_unusable_card

        digits = "".join(ch for ch in original if ch.isdigit())
        length = len(digits) if 13 <= len(digits) <= 19 else 16
        prefix = digits[0] if digits else "4"
        return make_unusable_card(prefix + f.numerify("#" * (length - 1)), length)

    @staticmethod
    def _make_lei(f) -> str:
        """An LEI Faker has no provider for, built so its own checksum validates."""
        from .finance_patterns import lei_check_digits

        prefix = f.bothify("????##############", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return prefix + lei_check_digits(prefix)

    def _pools_apply(self) -> bool:
        return bool(self._locale) and self._locale.lower().startswith("ru")

    def _from_faker(self, entity: str, original: str = "") -> str | None:
        f = self._faker
        assert f is not None
        try:
            if entity == "PERSON":
                return f.name()
            if entity == "ORGANIZATION":
                return f.company()
            if entity == "LOCATION":
                return f.city()
            if entity == "EMAIL_ADDRESS":
                return f.email()
            if entity in _PHONE_ENTITIES:
                return self._shape_like(original, f.phone_number())
            if entity == "IP_ADDRESS":
                return f.ipv4()
            if entity == "URL":
                return f.url()
            if entity == "SWIFT_BIC":
                return self._make_bic(f, original)
            if entity == "IBAN_CODE":
                # Grouped in fours is how an IBAN is written; the stand-in keeps that.
                return self._shape_like(original, self._make_iban(f, original))
            if entity == "ABA_ROUTING":
                return f.aba()
            if entity == "CREDIT_CARD":
                return self._shape_like(original, self._make_card(f, original))
            if entity == "LEI":
                return self._make_lei(f)
            if entity in _RU_CODE_BUILDERS:
                return _RU_CODE_BUILDERS[entity](f)
        except Exception:
            # A locale without the provider must not take the whole request down;
            # the pool path is always available.
            return None
        return None
