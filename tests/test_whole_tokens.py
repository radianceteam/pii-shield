"""A replacement covers whole words, or it is not made.

A stand-in that begins or ends inside a word cannot survive the round trip: the model
reads a mangled token, and what comes back no longer contains the stand-in to put
right. Presidio supplies the example — its URL recognizer reads `record.get("key")` as
the address `record.ge`, because `.ge` is Georgia's domain.
"""

from __future__ import annotations

import pytest

from pii_shield import Action, EntityRule, Policy, Shield

CODE = 'key = record.get("key", "").strip()\nindex.setdefault(key, []).append(pos)\n'


def _ru_available() -> bool:
    from pii_shield.engine.presidio_engine import PresidioDetector

    try:
        PresidioDetector("ru").warm()
    except Exception:
        return False
    return True


needs_ru = pytest.mark.skipif(not _ru_available(), reason="no Russian pipeline installed")


def cheap() -> Shield:
    """This project's own contact layer, which is what sees a value glued to a word:
    Presidio's email recognizer requires a boundary and finds nothing there at all."""
    return Shield(Policy.pattern_only("ru"), use_faker=False)


def test_a_value_glued_to_a_word_is_still_replaced_and_taken_whole():
    """Dropping the fragment would leak; replacing only part of it would too."""
    shield = cheap()
    result = shield.anonymize("почтаivan.petrov@acme.ru в тексте")
    assert "acme.ru" not in result.text
    assert "petrov" not in result.text
    assert " в тексте" in result.text


def test_the_replacement_is_a_whole_token_so_it_comes_back():
    shield = cheap()
    result = shield.anonymize("почтаivan.petrov@acme.ru в тексте")
    stand_in = result.text.split(" в тексте")[0]
    restored = shield.deanonymize(f"ответ про {stand_in}", result.session_id)
    assert restored == "ответ про почтаivan.petrov@acme.ru"


@needs_ru
def test_a_policy_that_replaces_urls_does_not_cut_an_identifier_in_half():
    """The span widens to the whole word first, so no token is left half-rewritten."""
    policy = Policy.for_language("ru")
    policy.rules = [r for r in policy.rules if r.entity != "URL"]
    policy.rules.append(EntityRule(entity="URL", action=Action.SURROGATE))
    text = Shield(policy).anonymize(CODE).text
    # Whatever it decided to replace, it never left a fragment of the original behind.
    assert "record.ge" not in text
    assert "index.se" not in text or "index.setdefault" in text


@needs_ru
def test_a_fragment_the_policy_allows_is_not_reported_at_all():
    """642 of 935 findings in ordinary code were `record.ge` and its kind."""
    shield = Shield(Policy.for_language("ru"))
    result = shield.anonymize(CODE * 20)
    assert result.text == CODE * 20                     # nothing replaced
    assert not [f for f in result.findings if f.entity == "URL"]


@needs_ru
def test_a_real_url_in_prose_is_still_found():
    shield = Shield(Policy.for_language("ru"))
    result = shield.anonymize("Документация здесь: https://example.com/docs — читайте.")
    assert [f for f in result.findings if f.entity == "URL"]
