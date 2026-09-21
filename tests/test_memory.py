"""Refusing to load a second pipeline rather than being killed for trying.

Reported from a live pod: 1600 MB limit, the Russian pipeline loaded at 998 MB, a
request naming English, OOMKilled. Failing closed is the right instinct, but being
killed is not failing closed — it is failing gone, and the agent is left with no
shield at all.
"""

from __future__ import annotations

import pytest

from pii_shield import Policy, Shield
from pii_shield import memory as mem
from pii_shield.engine.presidio_engine import NerUnavailableError

from .conftest import StubDetector


class FakeDetector(StubDetector):
    """A detector that claims a pipeline of a given size without loading one."""

    def __init__(self, model: str = "fake_lg", ner: bool = True) -> None:
        super().__init__()
        self.loaded_model = model
        self.ner_available = ner
        self.supported_entities = frozenset({"PERSON"})

    def planned_model(self) -> str:
        return self.loaded_model


def _budget(monkeypatch, *, limit: int, used: int, sizes: dict[str, int]):
    """Pretend a container budget.

    ``estimated_load_bytes`` is patched in both places: ``memory`` reads it for its own
    assessment and ``shield`` imported it directly, so patching only the module would
    leave the eviction arithmetic reading real package sizes.
    """
    from pii_shield import shield as shield_module

    monkeypatch.setattr(mem, "container_limit_bytes", lambda: limit)
    monkeypatch.setattr(mem, "current_usage_bytes", lambda: used)
    monkeypatch.setattr(mem, "estimated_load_bytes", lambda name: sizes.get(name))
    monkeypatch.setattr(shield_module, "estimated_load_bytes", lambda name: sizes.get(name))


MB = 1024 * 1024


def test_no_limit_means_no_refusal(monkeypatch):
    """A workstation has no cgroup limit; refusing there would be worse than useless."""
    monkeypatch.setattr(mem, "container_limit_bytes", lambda: None)
    assert mem.room_for("anything")[0]


def test_a_load_that_does_not_fit_is_refused(monkeypatch):
    _budget(monkeypatch, limit=1600 * MB, used=1000 * MB, sizes={"en_lg": 900 * MB})
    ok, reason = mem.room_for("en_lg")
    assert not ok
    assert "free" in reason and "needed" in reason


def test_a_load_that_fits_is_allowed(monkeypatch):
    _budget(monkeypatch, limit=4000 * MB, used=1000 * MB, sizes={"en_lg": 900 * MB})
    assert mem.room_for("en_lg")[0]


def test_the_configured_language_is_not_evicted_to_serve_another(monkeypatch):
    """Dropping it means reloading it for the next request, at forty seconds a time."""
    _budget(monkeypatch, limit=1600 * MB, used=1000 * MB,
            sizes={"ru_lg": 1000 * MB, "en_lg": 900 * MB})
    shield = Shield(Policy.for_language("ru"), use_faker=False,
                    detector=FakeDetector("ru_lg"))
    with pytest.raises(NerUnavailableError, match="not enough memory"):
        shield._make_room_for(FakeDetector("en_lg"), "en")
    assert shield._loaded_languages() == ["ru"]


def test_evict_primary_trades_the_reload_for_the_language(monkeypatch):
    _budget(monkeypatch, limit=1600 * MB, used=1000 * MB,
            sizes={"ru_lg": 1000 * MB, "en_lg": 900 * MB})
    shield = Shield(Policy.for_language("ru"), use_faker=False,
                    detector=FakeDetector("ru_lg"), evict_primary=True)
    shield._make_room_for(FakeDetector("en_lg"), "en")
    assert shield._loaded_languages() == []      # ru dropped, en about to load


def test_a_hopeless_load_evicts_nothing(monkeypatch):
    """Evicting and then refusing anyway is the worst outcome available: a working
    pipeline destroyed for a load that was never going to fit."""
    _budget(monkeypatch, limit=1600 * MB, used=1500 * MB,
            sizes={"ru_lg": 200 * MB, "huge": 5000 * MB})
    shield = Shield(Policy.for_language("ru"), use_faker=False,
                    detector=FakeDetector("ru_lg"), evict_primary=True)
    with pytest.raises(NerUnavailableError):
        shield._make_room_for(FakeDetector("huge"), "en")
    assert shield._loaded_languages() == ["ru"]


def test_the_message_says_what_to_do(monkeypatch):
    _budget(monkeypatch, limit=1600 * MB, used=1500 * MB, sizes={"huge": 5000 * MB})
    shield = Shield(Policy.for_language("ru"), use_faker=False,
                    detector=FakeDetector("ru_lg"))
    with pytest.raises(NerUnavailableError) as excinfo:
        shield._make_room_for(FakeDetector("huge"), "en")
    message = str(excinfo.value)
    assert "--pattern-only" in message
    assert "more memory" in message


def test_the_limit_can_be_stated_explicitly(monkeypatch):
    """For hosts where the cgroup files are not where they usually are."""
    monkeypatch.setenv("PII_SHIELD_MEMORY_LIMIT_BYTES", str(123 * MB))
    assert mem.container_limit_bytes() == 123 * MB
