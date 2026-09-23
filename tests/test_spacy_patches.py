"""Repairs to spaCy internals: they must be provable, and they must never be required."""

from __future__ import annotations

import pytest

from pii_shield.engine.spacy_patches import apply_all

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
