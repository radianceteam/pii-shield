"""Small repairs to code this project depends on but does not own.

Only for things that are pure, hot and cheap to prove: a function whose answer depends
on nothing but its argument, called hundreds of thousands of times per payload, doing
work that can be done once. Each patch checks that what it is replacing still looks the
way it expects and leaves it alone otherwise, so a new version of the dependency costs
speed rather than correctness.
"""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

_DISABLE_ENV = "PII_SHIELD_NO_PATCHES"
_applied: set[str] = set()


def apply_all() -> set[str]:
    """Apply every patch that fits this installation. Returns the ones applied."""
    if os.environ.get(_DISABLE_ENV, "").lower() in ("1", "true", "yes"):
        return set()
    _remember_russian_tag_conversion()
    return set(_applied)


def _remember_russian_tag_conversion() -> None:
    """Stop rebuilding a hundred-entry table for every token of Russian text.

    spaCy's Russian lemmatizer converts each morphological analysis from OpenCorpora
    notation to Universal Dependencies with ``oc2ud``, and that function builds its
    mapping table inside itself — so the table is constructed again on every call. On
    199 KB of Russian it was called 374 000 times with **22 distinct tags** between
    them, and cost 1.41× of the whole detection.

    The function is pure: a string in, a tag and a feature dictionary out. The
    dictionary is copied on the way out so that a caller free to modify it cannot
    modify what everyone else will be handed.
    """
    if "ru_oc2ud" in _applied:
        return
    try:
        from spacy.lang.ru import lemmatizer as ru_lemmatizer
    except ImportError:  # pragma: no cover - Russian support is optional
        return

    original = getattr(ru_lemmatizer, "oc2ud", None)
    if original is None or getattr(original, "_pii_shield_cached", False):
        return
    try:
        probe = original("NOUN,anim,masc sing,nomn")
        if not (isinstance(probe, tuple) and len(probe) == 2 and isinstance(probe[1], dict)):
            return
    except Exception:  # pragma: no cover - a shape we do not recognize is left alone
        logger.debug("pii-shield: oc2ud looks different than expected, leaving it", exc_info=True)
        return

    remembered = functools.lru_cache(maxsize=4096)(original)

    @functools.wraps(original)
    def cached_oc2ud(oc_tag):
        pos, features = remembered(oc_tag)
        return pos, dict(features)

    cached_oc2ud._pii_shield_cached = True
    ru_lemmatizer.oc2ud = cached_oc2ud
    _applied.add("ru_oc2ud")
