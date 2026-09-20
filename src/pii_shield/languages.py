"""Per-language configuration: model, surrogate locale, noise filters, national IDs.

Language used to be a single string passed to spaCy while everything else stayed
Russian, which produced English text carrying Russian names and place names — the
data was hidden, but the sentence stopped being a sentence, which is the entire
argument for surrogates over ``<PERSON>``. A language is therefore a *profile*, and
four things vary with it:

  * **the spaCy pipeline** — and its name is not formulaic: ``en`` and ``zh`` use
    ``_core_web_``, the other 22 use ``_core_news_``;
  * **the entity labels the pipeline emits** — Korean models emit ``PS``/``OG``/``LC``
    from the KLUE tagset, which Presidio's default mapping does not know, so Korean
    findings are silently dropped without an override;
  * **the surrogate locale** — a fake name must be a name in the right language;
  * **the noise filters** — minimum span length is wrong at 3 for CJK, where a full
    personal name is two or three characters, and the allowlist of terms an NER model
    mistakes for names is different in every language.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------
# Terms that are not identifying in any language: file formats, protocols, the
# vocabulary of a technical specification. Kept separate so each language's list
# holds only what is genuinely specific to it.
UNIVERSAL_ALLOWLIST = (
    "API", "REST", "SDK", "UI", "UX", "CLI", "SLA", "KPI", "MVP", "QA", "CI", "CD",
    "PDF", "XML", "JSON", "YAML", "CSV", "HTML", "CSS", "SQL", "HTTP", "HTTPS",
    "TCP", "IP", "URL", "URI", "UUID", "CRUD", "OAuth", "JWT", "LLM", "MCP", "AI",
    "SWIFT", "BIC", "IBAN", "LEI", "ABA", "ACH", "RTN", "BBAN",
    "БИК", "СВИФТ", "ИНН", "КПП", "ОГРН", "БАНК", "СЧЁТ", "СЧЕТ",
    "GPU", "CPU", "RAM", "OS", "VPN", "DNS", "SSH", "TLS", "SSL", "FAQ", "ID",
)

# Non-Latin script below is data, not prose: these are the literal strings the
# matcher compares against Russian text, and transliterating them would stop them
# matching anything. The same applies to the Ukrainian list and to the context words
# in the pattern modules.
RU_ALLOWLIST = (
    "ТЗ", "ТУ", "ЧТЗ", "ПО", "БД", "ОС", "ИС", "АС", "ПК", "УЗ", "ЛК", "ЭП",
    "ГОСТ", "СНиП", "ФЗ", "РФ", "НДС", "ЕГАИС", "ЕСИА", "СМЭВ", "ИНН", "ОГРН", "СНИЛС",
    "Заказчик", "Исполнитель", "Пользователь", "Администратор", "Оператор",
    "Система", "Сервис", "Модуль", "Компонент", "Подсистема",
)

EN_ALLOWLIST = (
    "SSN", "EIN", "ITIN", "LLC", "Inc", "Corp", "Ltd", "PLC", "NDA", "PII", "GDPR",
    "HIPAA", "VAT", "PO", "RFC", "RFP", "SOW", "Customer", "Supplier", "User",
    "Administrator", "Operator", "System", "Service", "Module", "Component",
)

PROFILE_ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "ru": RU_ALLOWLIST,
    "en": EN_ALLOWLIST,
    "de": ("GmbH", "AG", "KG", "UG", "UStID", "DSGVO", "Kunde", "Nutzer", "System"),
    "es": ("NIF", "NIE", "IVA", "SL", "SA", "RGPD", "Cliente", "Usuario", "Sistema"),
    "fr": ("SARL", "SAS", "SA", "TVA", "RGPD", "SIRET", "Client", "Utilisateur"),
    "it": ("SpA", "Srl", "IVA", "CF", "GDPR", "Cliente", "Utente", "Sistema"),
    "pt": ("Lda", "SA", "CNPJ", "CPF", "LGPD", "Cliente", "Usuário", "Sistema"),
    "pl": ("NIP", "REGON", "PESEL", "RODO", "Klient", "Użytkownik", "System"),
    "nl": ("BV", "NV", "BTW", "AVG", "Klant", "Gebruiker", "Systeem"),
    "uk": ("ТЗ", "ЄДРПОУ", "ІПН", "Замовник", "Виконавець", "Користувач", "Система"),
    # CJK: an allowlist entry can only ever match a whole span here, because there is
    # no whitespace to tokenize on. Legal-form words are listed precisely because a
    # model returns them alone often enough to matter.
    "zh": ("有限公司", "股份有限公司", "科技", "系统", "用户", "客户", "接口", "模块"),
    "ja": ("株式会社", "有限会社", "合同会社", "システム", "ユーザー", "顧客", "モジュール"),
    "ko": ("주식회사", "유한회사", "시스템", "사용자", "고객", "모듈"),
}

# ---------------------------------------------------------------------------
# Entity-label overrides
# ---------------------------------------------------------------------------
# Korean pipelines are trained on the KLUE tagset, whose labels share nothing with
# OntoNotes. Without this, Presidio maps none of them and Korean silently finds
# nothing at all — the worst possible failure for this tool.
KO_LABEL_MAP = {
    "PS": "PERSON",
    "OG": "ORGANIZATION",
    "LC": "LOCATION",
    "DT": "DATE_TIME",
    "TI": "DATE_TIME",
}

# ---------------------------------------------------------------------------
# National identifier entities
# ---------------------------------------------------------------------------
# What Presidio itself ships, verified by enumerating its recognizer registry.
PRESIDIO_NATIONAL: dict[str, tuple[str, ...]] = {
    "en": ("US_SSN", "US_ITIN", "US_PASSPORT", "US_DRIVER_LICENSE", "US_BANK_NUMBER",
           "UK_NHS"),
    "es": ("ES_NIF", "ES_NIE"),
    "it": ("IT_FISCAL_CODE", "IT_VAT_CODE", "IT_IDENTITY_CARD", "IT_PASSPORT",
           "IT_DRIVER_LICENSE"),
    "pl": ("PL_PESEL",),
}

# What this project adds, because Presidio has nothing for these locales.
LOCAL_NATIONAL: dict[str, tuple[str, ...]] = {
    "ru": ("RU_INN", "RU_SNILS", "RU_OGRN", "RU_PASSPORT", "RU_PHONE", "RU_BANK_ACCOUNT",
           "RU_BIK"),
    "zh": ("CN_RESIDENT_ID", "CN_PHONE"),
    "ja": ("JP_MY_NUMBER", "JP_PHONE"),
    "ko": ("KR_RRN", "KR_PHONE"),
}

# Detected by the NER pipeline (or Presidio's language-agnostic recognizers) rather
# than by a national pattern. Available for every language.
GLOBAL_NER_ENTITIES = (
    "PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER",
    "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "URL", "DATE_TIME", "NRP",
)

# ---------------------------------------------------------------------------
# What each entity actually costs
# ---------------------------------------------------------------------------
# Three tiers, because they have three different dependency footprints and a
# deployment may be able to afford only the cheapest. Conflating them meant a policy
# asking for nothing but checksummed identifiers still demanded a 500 MB language
# model, which made the documented pattern-only configuration impossible to build.
#
# 1. NER_MODEL_ENTITIES — produced by the spaCy pipeline's named-entity recognizer.
#    These are the only ones that need a language model, and they are the only ones
#    whose absence must be fatal when a policy asks for them.
NER_MODEL_ENTITIES = frozenset({"PERSON", "ORGANIZATION", "LOCATION", "NRP"})

# 2. Presidio's pattern, checksum and context recognizers. They need
#    ``presidio-analyzer`` installed, but no language model: a blank spaCy pipeline
#    (a few megabytes, shipped with spaCy itself) is enough to run them.
PRESIDIO_PATTERN_ENTITIES = frozenset(
    {"EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "URL", "DATE_TIME"}
)

# 3. Everything else is this project's own regex and checksum layers, which need
#    neither Presidio nor spaCy. Computed below, once the national tables exist.

# Institution and legal-entity identifier codes. Language-independent, so they are
# available everywhere rather than living in a national profile.
FINANCE_ENTITIES = ("SWIFT_BIC", "IBAN_CODE", "ABA_ROUTING", "LEI")

# Of those, IBAN is Presidio's recognizer; the other three are this project's own.
# CREDIT_CARD is here rather than in the Presidio tier because Presidio only
# recognizes it in four languages, and a card must not depend on the language it
# was written next to.
LOCAL_FINANCE_ENTITIES = ("SWIFT_BIC", "ABA_ROUTING", "LEI", "CREDIT_CARD", "IBAN_CODE")

# Credential shapes, detected by this project's own patterns. Language-independent
# and dependency-free, which is what makes a credentials-only deployment possible.
SECRET_ENTITIES = (
    "SECRET_API_KEY",
    "SECRET_JWT",
    "SECRET_PRIVATE_KEY",
    "SECRET_AUTH_HEADER",
    "SECRET_CONNECTION_STRING",
    "SECRET_URL_CREDENTIAL",
)


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    spacy_model: str
    faker_locale: str | None = None
    label_map: dict[str, str] = field(default_factory=dict)
    min_ner_span: int = 3
    allowlist: tuple[str, ...] = ()
    presidio_national: tuple[str, ...] = ()
    local_national: tuple[str, ...] = ()

    @property
    def ner_entities(self) -> tuple[str, ...]:
        """Everything the Presidio analyzer should be asked for in this language."""
        return GLOBAL_NER_ENTITIES + self.presidio_national

    @property
    def national_entities(self) -> tuple[str, ...]:
        return self.presidio_national + self.local_national

    def model_name(self, size: str = "lg") -> str:
        return f"{self.spacy_model}_{size}"


# spaCy 3.8 ships core pipelines for these 24 languages; only en and zh use the
# ``_core_web_`` family, which is why the name cannot be derived from the code alone.
_WEB_LANGUAGES = frozenset({"en", "zh"})
_SPACY_LANGUAGES = (
    "ca", "da", "de", "el", "en", "es", "fi", "fr", "hr", "it", "ja", "ko", "lt",
    "mk", "nb", "nl", "pl", "pt", "ro", "ru", "sl", "sv", "uk", "zh",
)

# Faker locale per language. Catalan and Norwegian Bokmål have no Faker locale, so
# they fall back to typed tokens rather than to a wrong-language name.
_FAKER_LOCALES = {
    "da": "da_DK", "de": "de_DE", "el": "el_GR", "en": "en_US", "es": "es_ES",
    "fi": "fi_FI", "fr": "fr_FR", "hr": "hr_HR", "it": "it_IT", "ja": "ja_JP",
    "ko": "ko_KR", "lt": "lt_LT", "mk": "mk_MK", "nl": "nl_NL", "pl": "pl_PL",
    "pt": "pt_PT", "ro": "ro_RO", "ru": "ru_RU", "sl": "sl_SI", "sv": "sv_SE",
    "uk": "uk_UA", "zh": "zh_CN",
}

# Two or three characters is a complete personal name in Chinese, Japanese and
# Korean, so the default floor of 3 would discard exactly what must be caught.
_MIN_NER_SPAN = {"zh": 2, "ja": 2, "ko": 2}


def _build_profiles() -> dict[str, LanguageProfile]:
    profiles: dict[str, LanguageProfile] = {}
    for code in _SPACY_LANGUAGES:
        family = "core_web" if code in _WEB_LANGUAGES else "core_news"
        profiles[code] = LanguageProfile(
            code=code,
            spacy_model=f"{code}_{family}",
            faker_locale=_FAKER_LOCALES.get(code),
            label_map=dict(KO_LABEL_MAP) if code == "ko" else {},
            min_ner_span=_MIN_NER_SPAN.get(code, 3),
            allowlist=UNIVERSAL_ALLOWLIST + PROFILE_ALLOWLISTS.get(code, ()),
            presidio_national=PRESIDIO_NATIONAL.get(code, ()),
            local_national=LOCAL_NATIONAL.get(code, ()),
        )
    return profiles


PROFILES = _build_profiles()
SUPPORTED_LANGUAGES = tuple(sorted(PROFILES))


class UnsupportedLanguageError(ValueError):
    """No spaCy pipeline exists for this language code.

    Raised rather than quietly falling back to English, which would scan the text
    with the wrong model and report a clean result.
    """


def get_profile(code: str) -> LanguageProfile:
    profile = PROFILES.get(code)
    if profile is None:
        raise UnsupportedLanguageError(
            f"unsupported language {code!r}; supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    return profile


def _presidio_entities() -> frozenset[str]:
    """Tier 2, including the national recognizers Presidio ships for en/es/it/pl."""
    out = set(PRESIDIO_PATTERN_ENTITIES)
    for national in PRESIDIO_NATIONAL.values():
        out |= set(national)
    return frozenset(out)


PRESIDIO_ENTITIES = _presidio_entities()


def _local_entities() -> frozenset[str]:
    """Tier 3: detected by this project's own patterns, no third-party dependency."""
    out = set(LOCAL_FINANCE_ENTITIES) | set(SECRET_ENTITIES)
    for national in LOCAL_NATIONAL.values():
        out |= set(national)
    return frozenset(out)


LOCAL_ENTITIES = _local_entities()


def all_national_entities() -> tuple[str, ...]:
    """Every national entity across every profile — the catalogue for documentation."""
    seen: list[str] = []
    for profile in PROFILES.values():
        for entity in profile.national_entities:
            if entity not in seen:
                seen.append(entity)
    return tuple(seen)
