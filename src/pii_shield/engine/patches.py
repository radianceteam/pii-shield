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
    _backport_presidio_deduplication()
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


def _backport_presidio_deduplication() -> None:
    """Take Presidio's own rewrite of ``remove_duplicates`` before it is released.

    The released implementation (2.2.364, the newest on PyPI) compares every result
    with every result already kept, twice over. On 199 KB of Russian traffic that came
    to **5.5 million comparisons that removed nothing at all**, and 0.60 s — a fifth of
    the whole detection.

    Half of that work cannot ever do anything: ``result not in filtered_results`` runs
    after ``list(set(results))``, and ``__hash__`` is built from the same four fields
    as ``__eq__``, so no two elements can be equal by the time it is asked.

    The project has already fixed this on ``main``: results are grouped by entity type
    and swept with a window of those that can still overlap. What is below is that
    implementation, so the semantics are the maintainers' rather than ours, and it is
    applied only while the installed copy is still the old one — a released fix takes
    over by itself.
    """
    if "presidio_dedup" in _applied:
        return
    try:
        import inspect

        from presidio_analyzer.entity_recognizer import EntityRecognizer
    except ImportError:  # pragma: no cover - Presidio is optional
        return

    current = getattr(EntityRecognizer, "remove_duplicates", None)
    if current is None or getattr(current, "_pii_shield_backport", False):
        return
    try:
        # The marker of the quadratic version, and of nothing else. When a release
        # carries the rewrite this stops matching and the installed code is left alone.
        if "not in filtered_results" not in inspect.getsource(current):
            return
    except (OSError, TypeError):  # pragma: no cover - source not available
        return

    def remove_duplicates(results):
        from itertools import groupby

        results = list(set(results))
        results = sorted(
            results,
            key=lambda x: (str(x.entity_type), x.start, -(x.end - x.start), -x.score),
        )
        filtered_results = []

        for _, entity_results in groupby(results, key=lambda result: result.entity_type):
            active_results = []
            for result in entity_results:
                if result.score == 0:
                    continue
                # Only a result that has not ended yet can contain this one.
                active_results = [
                    kept for kept in active_results if kept.end >= result.start
                ]
                to_keep = True
                for kept in active_results:
                    if result.contained_in(kept) and result.score <= kept.score:
                        to_keep = False
                        break
                if to_keep:
                    filtered_results.append(result)
                    active_results.append(result)

        # The callers downstream expect the score-first order the old one returned.
        return sorted(filtered_results, key=lambda x: (-x.score, x.start, -(x.end - x.start)))

    remove_duplicates._pii_shield_backport = True
    EntityRecognizer.remove_duplicates = staticmethod(remove_duplicates)
    _applied.add("presidio_dedup")
