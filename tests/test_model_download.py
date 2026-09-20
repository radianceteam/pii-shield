"""Fetching a language pipeline at run time — off by default, explicit when on.

This was disabled outright for a while, and for a good reason: Presidio fetched
models by itself, into an environment it chose, blocking startup for five minutes
with no output. It installed into a different project's virtualenv twice. The
capability is useful; letting a third party decide when and where is not.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from pii_shield import Policy, Shield
from pii_shield.engine.presidio_engine import (
    MODEL_RELEASE_URL,
    NerUnavailableError,
    PresidioDetector,
)


def test_download_is_off_by_default():
    assert PresidioDetector("sv").download is False
    assert Shield(Policy.pattern_only("ru"), use_faker=False)._download_models is False


def test_nothing_is_fetched_when_the_option_is_off(monkeypatch):
    """A missing pipeline must fail fast, not start a five-minute install."""
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a))
    detector = PresidioDetector("sv")
    assert detector.candidate_models() == []
    assert called == []


def test_candidate_models_fetches_when_enabled(monkeypatch):
    detector = PresidioDetector("sv", download=True, download_size="sm")
    monkeypatch.setattr(detector, "fetch_model", lambda *a, **k: "sv_core_news_sm")
    assert detector.candidate_models() == ["sv_core_news_sm"]


def test_the_installer_is_told_which_interpreter_to_target(monkeypatch):
    """The whole point: never let the installer resolve the environment itself."""
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(PresidioDetector, "model_installed", staticmethod(lambda name: True))
    detector = PresidioDetector("sv", download=True, download_size="sm")
    assert detector.fetch_model() == "sv_core_news_sm"
    assert sys.executable in seen["command"]
    assert any("sv_core_news_sm" in str(part) for part in seen["command"])


def test_it_falls_back_to_uv_when_pip_is_absent(monkeypatch):
    """`uv venv` creates environments without pip, which is common now."""
    attempts = []

    def fake_run(command, **kwargs):
        attempts.append(command[0])
        if command[1:3] == ["-m", "pip"]:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(PresidioDetector, "model_installed", staticmethod(lambda name: True))
    detector = PresidioDetector("sv", download=True, download_size="sm")
    assert detector.fetch_model() == "sv_core_news_sm"
    assert attempts[-1] == "uv"


def test_a_failed_download_returns_none_rather_than_half_working(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda command, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, command)),
    )
    detector = PresidioDetector("sv", download=True)
    assert detector.fetch_model() is None


def test_the_url_is_pinned_to_the_publisher():
    """Installing a model is installing code; it comes from one place."""
    assert MODEL_RELEASE_URL.startswith("https://github.com/explosion/spacy-models/releases/")


def test_release_version_follows_the_installed_spacy():
    import spacy

    major, minor = spacy.__version__.split(".")[:2]
    assert PresidioDetector("sv").model_release_version() == f"{major}.{minor}.0"


# --- the tagset guard -------------------------------------------------------
def test_a_pipeline_whose_labels_cannot_make_a_person_is_refused(monkeypatch):
    """Korean and Swedish both found names and discarded every one of them.

    Checking the loaded pipeline's own labels closes the class rather than chasing
    one tagset at a time.
    """
    detector = PresidioDetector("ru")
    monkeypatch.setattr(detector, "person_label_reachable", lambda: False)
    monkeypatch.setattr(type(detector), "analyzer", property(lambda self: object()))
    with pytest.raises(NerUnavailableError, match="maps to PERSON"):
        detector.warm(require_person=True)


@pytest.mark.parametrize("language", ["ru", "en", "de", "ko"])
def test_installed_pipelines_can_produce_a_person(language):
    detector = PresidioDetector(language)
    try:
        detector.warm(require_person=True)
    except NerUnavailableError as exc:
        if "maps to PERSON" in str(exc):
            raise
        pytest.skip(f"no pipeline installed for {language}")
