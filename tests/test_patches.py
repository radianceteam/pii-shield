"""Repairs to spaCy internals: they must be provable, and they must never be required."""

from __future__ import annotations

import pytest

from pii_shield.engine.patches import apply_all

pytest.importorskip("spacy.lang.ru.lemmatizer")


def test_the_conversion_still_answers_the_same():
    """The cache is only sound because the function is pure, so check it is."""
    from spacy.lang.ru import lemmatizer as ru

    apply_all()
    assert getattr(ru.oc2ud, "_pii_shield_cached", False)
    for tag in ("NOUN,anim,masc sing,nomn", "VERB,perf,tran sing,3per,futr,indc", "ADJF sing,nomn"):
        first_pos, first_features = ru.oc2ud(tag)
        second_pos, second_features = ru.oc2ud(tag)
        assert (first_pos, first_features) == (second_pos, second_features)


def test_a_caller_cannot_poison_what_is_remembered():
    """The features come back as a fresh dictionary each time, so mutating one is safe."""
    from spacy.lang.ru import lemmatizer as ru

    apply_all()
    tag = "NOUN,anim,masc sing,nomn"
    _, features = ru.oc2ud(tag)
    features["Case"] = "нарочно испорчено"
    _, again = ru.oc2ud(tag)
    assert "нарочно испорчено" not in again.values()


def test_applying_twice_changes_nothing():
    from spacy.lang.ru import lemmatizer as ru

    apply_all()
    once = ru.oc2ud
    apply_all()
    assert ru.oc2ud is once


def test_patches_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("PII_SHIELD_NO_PATCHES", "1")
    assert apply_all() == set()


# --- Presidio's own rewrite of remove_duplicates, taken early ----------------
presidio = pytest.importorskip("presidio_analyzer")


def original_remove_duplicates(results):
    """The released implementation, verbatim, to compare against."""
    results = list(set(results))
    results = sorted(results, key=lambda x: (-x.score, x.start, -(x.end - x.start)))
    filtered_results = []
    for result in results:
        if result.score == 0:
            continue
        to_keep = result not in filtered_results
        if to_keep:
            for filtered in filtered_results:
                if result.contained_in(filtered) and result.entity_type == filtered.entity_type:
                    to_keep = False
                    break
        if to_keep:
            filtered_results.append(result)
    return filtered_results


def as_tuples(results):
    """What the answer is, regardless of the order it comes back in.

    Order does not matter downstream: the shield sorts these spans itself, by what the
    policy does with them rather than by score.
    """
    return sorted((r.entity_type, r.start, r.end, r.score) for r in results)


def random_results(rng, count):
    from presidio_analyzer import RecognizerResult

    out = []
    for _ in range(count):
        start = rng.randrange(0, 60)
        # Short spans, so that containment and equality actually happen.
        end = start + rng.randrange(1, 12)
        entity = rng.choice(["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"])
        score = rng.choice([0.0, 0.3, 0.5, 0.5, 0.85, 1.0])
        out.append(RecognizerResult(entity_type=entity, start=start, end=end, score=score))
    return out


def test_the_backport_answers_exactly_as_the_released_code():
    import random

    from presidio_analyzer.entity_recognizer import EntityRecognizer

    apply_all()
    patched = EntityRecognizer.remove_duplicates
    assert getattr(patched, "_pii_shield_backport", False), "the backport did not apply"

    rng = random.Random(20260923)
    for case in range(300):
        results = random_results(rng, rng.randrange(0, 40))
        assert as_tuples(patched(list(results))) == as_tuples(
            original_remove_duplicates(list(results))
        ), f"case {case}"


def test_the_cases_that_matter_are_actually_exercised():
    """A test that never sees a nested span would prove nothing."""
    import random

    from presidio_analyzer.entity_recognizer import EntityRecognizer

    apply_all()
    rng = random.Random(7)
    dropped_something = False
    for _ in range(200):
        results = random_results(rng, 30)
        if len(EntityRecognizer.remove_duplicates(list(results))) < len(set(results)):
            dropped_something = True
            break
    assert dropped_something, "no case dropped a contained result"


def test_a_released_fix_is_left_alone(monkeypatch):
    """The guard is the old implementation's own line. Without it: hands off.

    Simulated by putting a function that does not carry the marker in its place, the
    way a Presidio release carrying the rewrite would.
    """
    from presidio_analyzer.entity_recognizer import EntityRecognizer

    from pii_shield.engine import patches

    def already_fixed(results):  # no marker in this source
        return list(results)

    monkeypatch.setattr(EntityRecognizer, "remove_duplicates", staticmethod(already_fixed))
    monkeypatch.setattr(patches, "_applied", set())
    assert "presidio_dedup" not in patches.apply_all()
    assert EntityRecognizer.remove_duplicates is already_fixed
