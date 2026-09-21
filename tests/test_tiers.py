"""What a policy actually costs to run, and refusing to pretend otherwise.

Reported from a live integration: a deployment sized for an agent rather than for a
language model could not build a Shield at all, even for a policy made entirely of
checksummed identifiers. The README promised that configuration worked; the code
required a spaCy pipeline unconditionally. These tests pin the three tiers so the
promise and the behaviour cannot drift apart again.
"""

from __future__ import annotations

import pytest

from pii_shield import Policy, Shield
from pii_shield.engine.presidio_engine import NerUnavailableError
from pii_shield.languages import (
    CONTACT_FALLBACK_ENTITIES,
    LOCAL_ENTITIES,
    NER_MODEL_ENTITIES,
    PRESIDIO_ENTITIES,
)

from .conftest import StubDetector


# --- the classification itself ----------------------------------------------
def test_tiers_do_not_overlap():
    """One deliberate exception: email and phone are served by both.

    Presidio validates numbers against real numbering plans and a regex cannot, so the
    full tier keeps its recognizers — but leaving them out of the cheap tier meant a
    customer writing in anything but the four languages whose national identifiers it
    knows got almost no protection at all.
    """
    assert not NER_MODEL_ENTITIES & PRESIDIO_ENTITIES
    assert PRESIDIO_ENTITIES & LOCAL_ENTITIES == CONTACT_FALLBACK_ENTITIES
    assert not NER_MODEL_ENTITIES & LOCAL_ENTITIES


def test_every_catalogued_entity_is_classified():
    from pii_shield import ALL_ENTITIES

    unclassified = set(ALL_ENTITIES) - NER_MODEL_ENTITIES - PRESIDIO_ENTITIES - LOCAL_ENTITIES
    assert not unclassified, unclassified


def test_only_four_entities_need_a_language_model():
    """Anything else demanding one is what made the pattern-only tier impossible."""
    assert NER_MODEL_ENTITIES == {"PERSON", "ORGANIZATION", "LOCATION", "NRP"}


def test_iban_is_ours_not_presidios():
    """It has a mod-97 checksum, which is the whole definition of the cheap tier."""
    assert "IBAN_CODE" in LOCAL_ENTITIES


def test_cards_are_ours_not_presidios():
    """Presidio recognizes cards in four languages only; a card must not depend on
    the language it was written next to."""
    assert "CREDIT_CARD" in LOCAL_ENTITIES


# --- the predicates ---------------------------------------------------------
def test_pattern_only_needs_nothing():
    policy = Policy.pattern_only("ru")
    assert policy.local_only
    assert not policy.requires_ner()
    assert not policy.requires_presidio()


def test_dropping_names_drops_the_model_requirement():
    base = Policy.for_language("ru")
    assert base.requires_ner()
    without = base.model_copy(update={
        "entities": [e for e in base.entities if e not in NER_MODEL_ENTITIES]
    })
    assert not without.requires_ner()
    assert without.requires_presidio()      # email, IBAN, IP still need Presidio


def test_allowed_names_do_not_require_a_model():
    """Detecting an entity and then leaving it alone produces the same text."""
    from pii_shield import Action, EntityRule

    policy = Policy(
        language="ru",
        entities=["PERSON", "RU_INN"],
        rules=[EntityRule(entity="PERSON", action=Action.ALLOW)],
    )
    assert not policy.requires_ner()


def test_scope_off_needs_nothing():
    assert not Policy.off().requires_presidio()


# --- construction -----------------------------------------------------------
def test_pattern_only_shield_builds_without_any_detector():
    shield = Shield(Policy.pattern_only("ru"), use_faker=False)
    assert shield._detectors == {}


def test_pattern_only_shield_still_detects_and_blocks():
    shield = Shield(Policy.pattern_only("ru"), use_faker=False)
    result = shield.anonymize("Клиент, ИНН 7707083893, тел +7 900 123-45-67")
    assert "7707083893" not in result.text
    assert {"RU_INN", "RU_PHONE"} <= {f.entity for f in result.findings}

    from pii_shield import BlockedError

    for blocked in ("СНИЛС 112-233-445 95", "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
        with pytest.raises(BlockedError):
            shield.anonymize(blocked)


def test_pattern_only_reports_that_names_were_not_analyzed():
    shield = Shield(Policy.pattern_only("ru"), use_faker=False)
    assert shield.anonymize("ИНН 7707083893").names_analyzed is False


def test_a_policy_asking_for_names_still_refuses_without_a_pipeline():
    """The guarantee this must not weaken."""
    policy = Policy.for_language("ru")
    assert policy.requires_ner()
    with pytest.raises(NerUnavailableError):
        Shield(policy, use_faker=False, detector=StubDetector(raises=NerUnavailableError("x")))


def test_cards_are_caught_in_any_language():
    """Russian deployments passed card numbers straight through before this."""
    shield = Shield(Policy.pattern_only("ru"), use_faker=False)
    found = shield.detect("Оплата картой 4111111111111111")
    assert [f.entity for f in found] == ["CREDIT_CARD"]


def test_pattern_only_finds_a_foreign_iban():
    """The tiny tier must catch a German IBAN without any German anything."""
    shield = Shield(Policy.pattern_only("ru"), use_faker=False)
    found = shield.detect("Rechnung: IBAN DE89370400440532013000")
    assert [f.entity for f in found] == ["IBAN_CODE"]


def test_a_german_iban_is_not_a_russian_bank_account():
    """DE89370400440532013000 carries exactly twenty digits after its country code,
    which the account pattern read as a Russian account — and then blocked."""
    shield = Shield(Policy.pattern_only("ru"), use_faker=False)
    assert "RU_BANK_ACCOUNT" not in {f.entity for f in shield.detect("IBAN DE89370400440532013000")}


# --- the tiers must promise the same refusals -------------------------------
@pytest.mark.parametrize("entity", ["CREDIT_CARD", "SECRET_API_KEY", "SECRET_JWT"])
def test_both_tiers_refuse_the_same_things(entity):
    """A tier decides what can be *detected*, not what is too dangerous to send.

    Cards were blocked by the full policy and merely pseudonymized by the cheap one,
    so the same number was refused or forwarded depending on which plan a customer
    was on.
    """
    assert entity in Policy.for_language("ru").blocking_entities
    assert entity in Policy.pattern_only("ru").blocking_entities


@pytest.mark.parametrize("factory", [Policy.for_language, Policy.pattern_only])
def test_entities_are_not_listed_twice(factory):
    """IBAN belonged to two catalogues, so every consumer counted it twice."""
    entities = factory("ru").entities
    assert len(entities) == len(set(entities))


def test_pattern_only_blocks_a_card_written_in_groups():
    from pii_shield import BlockedError

    shield = Shield(Policy.pattern_only("ru"), use_faker=False)
    with pytest.raises(BlockedError, match="CREDIT_CARD"):
        shield.anonymize("Оплата 4111 1111 1111 1111")
