"""Russian full names found by pattern, where the language model does not find them.

The model reads sentences. An agent's traffic is mostly not sentences: it is tool
output, table rows, CRM dumps and form fields, and there the Russian pipeline goes
quiet. Measured on `ru_core_news_lg`:

    "ФИО: Пётр Николаевич Васильев"            -> nothing detected, the whole name left
    "Клиент: Пётр Николаевич Васильев, тел .." -> the phone replaced, the name left
    "| Пётр Николаевич Васильев | менеджер |"  -> only "Николаевич Васильев" replaced

The last one is the worst kind of miss: part of a real name goes to the provider while
the response looks anonymized.

A patronymic is a near-unique anchor for this. -ович/-евич and -овна/-евна are
productive suffixes that almost nothing else in Russian ends with, they survive every
case ending, and a name built around one is three capitalized words at most. That is
enough to find the whole name without a model — which also means the pattern-only
tier, 30 MB and no pipeline, starts finding Russian full names it could not find
before. It does not turn that tier into a name detector: a name written without a
patronymic ("Позвони Ивану Петрову") still needs the model, which is why
``names_analyzed`` stays false there.

Short patronymics — Ильич, Кузьминична — are deliberately not anchored: their suffix
is a bare -ич/-ична that ordinary words share, and the false positives would cost more
than the rare name is worth.
"""

from __future__ import annotations

import re

from ..types import Action, Finding

ENTITY = "RU_FULL_NAME"

# A capitalized Russian word, including a double-barrelled surname.
_WORD = r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"

# The patronymic is matched by stem, not by its nominative form, because every case
# ending replaces the last letter: Николаевич/Николаевича/Николаевичем, and
# Сергеевна/Сергеевны/Сергеевной. Ending the match at a word boundary keeps the stem
# from matching inside an ordinary adjective ("Земляничная").
_PATRONYMIC = r"[А-ЯЁ][а-яё]{2,}(?:вич[а-яё]{0,2}|вн[аыеоу][йю]?|ичн[аыеоу][йю]?)\b"

# Initials, as a document writes them: "П.Н." or "П. Н."
_INITIALS = r"[А-ЯЁ]\.\s?[А-ЯЁ]\."

# Words that sit next to a name and are not part of it. Without this list the
# surname-first pattern swallows the label in front of it — "Директор Пётр Николаевич"
# would be replaced whole, and the replacement would read as a different job title.
_ROLE_WORDS = frozenset({
    "фио", "имя", "клиент", "заказчик", "подрядчик", "исполнитель", "директор",
    "гендиректор", "генеральный", "главный", "старший", "младший", "заместитель",
    "зам", "начальник", "руководитель", "менеджер", "специалист", "сотрудник",
    "работник", "бухгалтер", "юрист", "инженер", "врач", "пациент", "абонент",
    "плательщик", "получатель", "отправитель", "покупатель", "продавец",
    "арендатор", "арендодатель", "владелец", "учредитель", "представитель",
    "президент", "председатель", "секретарь", "партнёр", "партнер", "контакт",
    "автор", "ответственный", "уважаемый", "уважаемая", "господин", "госпожа",
    "гражданин", "гражданка", "товарищ", "тов", "приказ", "договор", "акт",
    "справка", "заявление", "от", "кому", "для",
})

# Ordered most specific first; a later pattern never overlaps an earlier match.
_GIVEN_PATRONYMIC_FAMILY = re.compile(rf"\b{_WORD}\s+{_PATRONYMIC}\s+{_WORD}\b")
_FAMILY_GIVEN_PATRONYMIC = re.compile(rf"\b{_WORD}\s+{_WORD}\s+{_PATRONYMIC}")
_GIVEN_PATRONYMIC = re.compile(rf"\b{_WORD}\s+{_PATRONYMIC}")
_FAMILY_INITIALS = re.compile(rf"\b{_WORD}\s+{_INITIALS}")
_INITIALS_FAMILY = re.compile(rf"{_INITIALS}\s?{_WORD}\b")


def _starts_with_role_word(span: str) -> bool:
    """True if the match begins with a label rather than with a name."""
    first = span.split()[0].strip(".,;:()[]«»\"'").casefold()
    return first in _ROLE_WORDS


def scan(text: str, *, entities: set[str] | None = None) -> list[Finding]:
    """Detect Russian full names in *text*.

    A patronymic-anchored name scores 0.8: the anchor is strong evidence but, unlike a
    checksum, not proof. Initials score 0.6, so a policy can raise the threshold to
    keep the full names and drop the weaker shape without touching anything else.
    """
    if not text or (entities is not None and ENTITY not in entities):
        return []

    taken: list[tuple[int, int]] = []
    found: list[Finding] = []

    def claim(pattern: re.Pattern[str], score: float, recognizer: str) -> None:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            if _starts_with_role_word(match.group(0)):
                continue
            taken.append((start, end))
            found.append(
                Finding(
                    entity=ENTITY, start=start, end=end, score=score,
                    action=Action.SURROGATE, recognizer=recognizer,
                )
            )

    # "Пётр Николаевич Васильев" is claimed before "Клиент Пётр Николаевич" can be,
    # which is what keeps the surname inside the span and the label outside it.
    claim(_GIVEN_PATRONYMIC_FAMILY, 0.8, "ru:name_gpf")
    claim(_FAMILY_GIVEN_PATRONYMIC, 0.8, "ru:name_fgp")
    claim(_GIVEN_PATRONYMIC, 0.8, "ru:name_gp")
    claim(_FAMILY_INITIALS, 0.6, "ru:name_initials")
    claim(_INITIALS_FAMILY, 0.6, "ru:name_initials")

    found.sort(key=lambda f: f.start)
    return found


def looks_female(name: str) -> bool:
    """True if the name's patronymic is a woman's.

    Used when drawing the stand-in: replacing "Анна Сергеевна Кузнецова" with a man's
    name leaves every agreeing word in the sentence wrong.
    """
    return bool(re.search(r"(?:вн|ичн)[аыеоу][йю]?\b", name))


def shape_of(name: str) -> str:
    """How the name is written, so the stand-in can be written the same way.

    The order matters beyond looks: a stand-in written the other way round cannot be
    mapped back to the original word by word, because word *i* of one is not word *i*
    of the other. That mapping is what restores a name the model declined.
    """
    words = name.split()
    if re.search(_INITIALS, name):
        return "initials_first" if re.match(rf"^{_INITIALS}", name) else "family_initials"
    if len(words) >= 3:
        return "family_first" if re.fullmatch(_PATRONYMIC, words[-1]) else "given_first"
    if len(words) == 2:
        return "given_patronymic" if re.fullmatch(_PATRONYMIC, words[-1]) else "given_family"
    return "single"
