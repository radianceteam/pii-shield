"""Matching a stand-in back to its original when the language inflects it.

Substitution is exact: the stand-in goes out as it was generated. What comes back is
not. A model handed "Иванна Олеговна Горбунова" writes "с Иванной Олеговной
Горбуновой", and an exact-match restore leaves that untouched — so the caller reads a
person who does not exist and takes them for real. That is worse than no restore at
all, because it looks like it worked.

Matching on the stem instead of the whole word costs a little precision and buys back
every case, number and gender the language has. It is applied only to free-text names
in languages that inflect them; a checksummed identifier is returned verbatim or not
at all, and must never be matched loosely.

Two rules keep a loose match from doing damage, both learned from output this module
produced:

**What has been written is never matched again.** The exact pass replaced the stand-in
"Ларионова Василиса Алексеевна" with the real "Петру Николаевичу Васильеву"; the stem
pattern for "Василиса" then matched the real surname "Васильеву" that had just been
written and replaced it with "Николаевичу". That is not a missing restore, it is a
corrupted one: a real person's name handed back in the wrong place. Every replacement
is therefore protected, and later passes see only text nobody has written into.

**A guess with no correspondence is not made at all.** Word *i* of the stand-in maps to
word *i* of the original, which means nothing when the two are written in different
orders: "Гуляев Архип Юлианович" against "Анна Сергеевна Кузнецова" turned "Архип
Юлианович" into "Сергеевна Кузнецова". Where the patronymic does not sit at the same
index in both names, no such mapping exists and the stand-in is left standing. A reader
who sees an invented name can ask about it; a reader who sees the wrong real name
cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# How much of a word may be ending rather than stem, and how much ending a match may
# carry. Russian surnames run to four-character endings ("Горбуновыми"), which is the
# practical ceiling for this.
_STEM_TRIM = 3
# Six, because a masculine surname takes a longer ending in the feminine oblique cases
# than a rule of four allows: "Игнатьев" becomes "Игнатьевой", which is five characters
# past the stem, and stopping at four left it unrestored.
_MAX_ENDING = 6
_MIN_STEM = 4

# A patronymic, matched by stem so that every case ending is covered. Deliberately
# looser than the pattern that finds names in free text — it accepts the short forms
# (Фомич, Ильинична) — because here it only asks which word of an already-known name
# is the patronymic. A word it misreads costs a skipped restore, never a wrong one.
_PATRONYMIC = re.compile(r"(?:вич|вн|ичн|ич)[а-яё]{0,3}$", re.IGNORECASE)


@dataclass(frozen=True)
class _Segment:
    """A piece of the answer. ``written`` marks text that a replacement produced."""

    text: str
    written: bool


def _stem(word: str) -> str:
    if len(word) <= _MIN_STEM:
        return word
    return word[: max(_MIN_STEM, len(word) - _STEM_TRIM)]


def _word_pattern(word: str) -> str:
    """Stem plus an ending, but never shorter than the word less one character.

    Allowing the bare stem to match turned a genuinely different name into a
    replacement: "Иван" is the stem of the stand-in "Иванна", so an unrelated Иван
    Петров in the reply came back as somebody else. Inflection changes the ending and
    rarely shortens a word by more than one character, so requiring that much back is
    enough to tell "Иванной" from "Иван".
    """
    stem = _stem(word)
    minimum = max(0, len(word) - 1 - len(stem))
    return re.escape(stem) + rf"\w{{{minimum},{_MAX_ENDING}}}"


def _phrase_pattern(surrogate: str) -> re.Pattern[str] | None:
    words = surrogate.split()
    if not words:
        return None
    return re.compile(r"\b" + r"\s+".join(_word_pattern(w) for w in words) + r"\b", re.UNICODE)


def _patronymic_index(words: list[str]) -> int | None:
    for index, word in enumerate(words):
        if _PATRONYMIC.search(word):
            return index
    return None


def _lone_word_pattern(word: str) -> str:
    """The stand-in's own word, or that word inflected — and nothing else.

    A single word carries far less evidence than a three-part phrase, so it is matched
    against the whole word rather than a stem: either exactly as issued, or with its
    final letter replaced by an ending. Trimming three characters the way the phrase
    pattern does is what let the stand-in "Игнатьев" match "Игнатов" — a different
    person standing next to the name, replaced with a real surname that was not theirs.
    """
    return rf"(?:{re.escape(word)}|{re.escape(word[:-1])}\w{{1,{_MAX_ENDING}}})"


def _word_replacements(surrogate: str, original: str) -> list[tuple[re.Pattern[str], str]]:
    """Per-word patterns, for the parts of a name that arrive on their own.

    Word *i* maps to word *i* of the original, so a lone "Иваннау" becomes "Петру"
    rather than the entire three-part name wedged into the middle of a sentence. That
    mapping exists only when the two names have the same number of words *and* are
    written in the same order, which the position of the patronymic reveals.

    Words of four characters or fewer sit this pass out: "Иван" is a prefix of Иванов,
    Иванова and Иванченко, and the shorter the word the likelier the collision.
    """
    words, original_words = surrogate.split(), original.split()
    if len(words) < 2 or len(words) != len(original_words):
        return []
    if _patronymic_index(words) != _patronymic_index(original_words):
        return []
    out = []
    for word, original_word in zip(words, original_words, strict=True):
        if len(word) <= _MIN_STEM:
            continue
        out.append((re.compile(rf"\b{_lone_word_pattern(word)}\b", re.UNICODE), original_word))
    return out


def _replace(segments: list[_Segment], pattern: re.Pattern[str], value: str) -> list[_Segment]:
    """Apply one replacement to the parts of the answer nobody has written into."""
    out: list[_Segment] = []
    for segment in segments:
        if segment.written:
            out.append(segment)
            continue
        cursor = 0
        for match in pattern.finditer(segment.text):
            if match.start() > cursor:
                out.append(_Segment(segment.text[cursor : match.start()], False))
            out.append(_Segment(value, True))
            cursor = match.end()
        out.append(_Segment(segment.text[cursor:], False))
    return [segment for segment in out if segment.text]


def restore(
    text: str,
    pairs: list[tuple[str, str]],
    *,
    exact: dict[str, str] | None = None,
) -> str:
    """Put the originals back: exact forms first, inflected ones after.

    ``exact`` is the session's own surrogate map. Handing it to this function rather
    than substituting it beforehand is what lets the inflected pass know which text it
    must not touch.
    """
    if not text or (not pairs and not exact):
        return text

    segments = [_Segment(text, False)]

    # Longest first, so a stand-in that is a prefix of another is not half-replaced.
    for surrogate in sorted(exact or {}, key=len, reverse=True):
        segments = _replace(segments, re.compile(re.escape(surrogate)), exact[surrogate])

    for surrogate, original in sorted(pairs, key=lambda item: -len(item[0])):
        phrase = _phrase_pattern(surrogate)
        if phrase is not None:
            segments = _replace(segments, phrase, original)
        # The per-word pass runs whether or not the whole name appeared, because a
        # model naming someone in full and then by their first name alone is the
        # ordinary case, not the exception. What keeps it safe is that a lone word is
        # matched against the whole word rather than a stem.
        for pattern, value in _word_replacements(surrogate, original):
            segments = _replace(segments, pattern, value)

    return "".join(segment.text for segment in segments)
