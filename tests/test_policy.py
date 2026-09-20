"""Policy is the reviewable artifact — a compliance reader checks this, not the code."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pii_shield import Action, EntityRule, Policy, Scope


def test_default_action_applies_to_unlisted_entities():
    p = Policy(default_action=Action.MASK)
    assert p.action_for("PERSON") is Action.MASK


def test_rule_overrides_default():
    p = Policy(default_action=Action.MASK, rules=[EntityRule(entity="PERSON", action=Action.HASH)])
    assert p.action_for("PERSON") is Action.HASH
    assert p.action_for("LOCATION") is Action.MASK


def test_thresholds():
    p = Policy(
        default_threshold=0.5,
        rules=[EntityRule(entity="PERSON", action=Action.MASK, threshold=0.9)],
    )
    assert p.threshold_for("PERSON") == 0.9
    assert p.threshold_for("LOCATION") == 0.5


def test_duplicate_rules_rejected():
    with pytest.raises(ValidationError):
        Policy(rules=[
            EntityRule(entity="PERSON", action=Action.MASK),
            EntityRule(entity="PERSON", action=Action.HASH),
        ])


def test_ru_default_blocks_credentials_not_names():
    p = Policy.ru_default()
    assert p.action_for("SECRET_API_KEY") is Action.BLOCK
    assert p.action_for("RU_PASSPORT") is Action.BLOCK
    assert p.action_for("PERSON") is Action.SURROGATE


def test_ru_default_allows_dates_and_urls():
    """Load-bearing context in requirements work, rarely identifying alone."""
    p = Policy.ru_default()
    assert p.action_for("DATE_TIME") is Action.ALLOW
    assert p.action_for("URL") is Action.ALLOW


def test_strict_blocks_everything_listed():
    p = Policy.strict()
    assert p.action_for("PERSON") is Action.BLOCK
    assert "PERSON" in p.blocking_entities


def test_off_is_an_explicit_opt_out():
    p = Policy.off()
    assert p.scope is Scope.OFF
    assert p.entities == []
    assert p.secrets is False


def test_threshold_bounds_are_validated():
    with pytest.raises(ValidationError):
        Policy(default_threshold=1.5)
