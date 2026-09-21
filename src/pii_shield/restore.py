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
"""

from __future__ import annotations

import re

# How much of a word may be ending rather than stem, and how much ending a match may
# carry. Russian surnames run to four-character endings ("Горбуновыми"), which is the
# practical ceiling for this.
_STEM_TRIM = 3
# Six, because a masculine surname takes a longer ending in the feminine oblique cases
# than a rule of four allows: "Игнатьев" becomes "Игнатьевой", which is five characters
# past the stem, and stopping at four left it unrestored.
_MAX_ENDING = 6
_MIN_STEM = 4


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


def _word_replacements(surrogate: str, original: str) -> list[tuple[re.Pattern[str], str]]:
    """Per-word patterns, used only when the whole name is not present.

    Word *i* maps to word *i* of the original, so a lone "Иваннау" becomes "Петру"
    rather than the entire three-part name wedged into the middle of a sentence. Only
    built when the two names have the same number of words, because otherwise there is
    no such correspondence to rely on.
    """
    words, original_words = surrogate.split(), original.split()
    if len(words) < 2 or len(words) != len(original_words):
        return []
    out = []
    for word, original_word in zip(words, original_words, strict=True):
        if len(_stem(word)) < _MIN_STEM:
            continue
        out.append((re.compile(rf"\b{_word_pattern(word)}\b", re.UNICODE), original_word))
    return out


def restore(text: str, pairs: list[tuple[str, str]]) -> str:
    """Replace inflected forms of each stand-in with its original.

    The whole name is tried first, and when it is found the parts are left alone. That
    ordering matters: matching on a stem cannot tell an inflection of the stand-in from
    a different name that happens to share it — "Иванов" looks like a form of "Иванна"
    to any rule short of a morphological analyser — so the loose pass is reserved for
    texts where the safe one found nothing.
    """
    if not text or not pairs:
        return text

    # Longest stand-in first, so a name containing another name is not half-replaced.
    for surrogate, original in sorted(pairs, key=lambda item: -len(item[0])):
        phrase = _phrase_pattern(surrogate)
        if phrase is None:
            continue
        text, replaced = phrase.subn(lambda _m, v=original: v, text)
        if replaced:
            continue
        for pattern, value in _word_replacements(surrogate, original):
            text = pattern.sub(lambda _m, v=value: v, text)
    return text
