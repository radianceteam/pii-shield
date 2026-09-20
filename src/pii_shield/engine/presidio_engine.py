"""Presidio-backed named-entity detection, configured from a language profile.

Loaded lazily and kept behind a narrow surface, because it is the only expensive part
of the stack: importing Presidio drags in spaCy, and constructing the analyzer loads a
pipeline of several hundred megabytes. Nothing here is imported at package import time.

Two details that a naive integration gets wrong, and that the profile supplies:

  * the pipeline name is not formulaic (``en`` and ``zh`` use ``_core_web_``);
  * the labels a pipeline emits are not universal. Korean models use the KLUE tagset
    (``PS``, ``OG``, ``LC``), which Presidio's built-in mapping does not contain, so
    without an override Korean detection returns nothing at all while appearing to work.
"""

from __future__ import annotations

import importlib.util
import threading

from ..languages import GLOBAL_NER_ENTITIES, LanguageProfile, get_profile
from ..types import Action, Finding

# Entities answered by the NER pipeline and Presidio's own recognizers. Structured
# national identifiers and credentials are handled by the pattern layers, which are
# faster and more precise for those types.
NER_ENTITIES = frozenset(GLOBAL_NER_ENTITIES)

# Model sizes tried in order. ``lg`` first because the small pipelines are materially
# worse at person names, which is the entity that matters most here; ``sm`` last so a
# CI job or a laptop can still run with a small download.
MODEL_SIZES = ("lg", "md", "sm")


class NerUnavailableError(RuntimeError):
    """Presidio or its language model could not be loaded.

    Distinct from a detection failure: this means the shield cannot do what the policy
    asked for at all, which is a configuration problem to surface at startup rather
    than a per-request error to retry.
    """


class PresidioDetector:
    """Thin adapter over Presidio's AnalyzerEngine, one per language."""

    def __init__(self, language: str = "ru", model: str | None = None) -> None:
        self.language = language
        self.profile: LanguageProfile = get_profile(language)
        self.model = model
        self._analyzer = None
        self._loaded_model: str | None = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        """True if the NER extra is importable. Says nothing about any model."""
        try:
            import presidio_analyzer  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def supported_entities(self) -> frozenset[str]:
        """Everything this detector can be asked for, including national recognizers."""
        return frozenset(self.profile.ner_entities)

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model

    @staticmethod
    def model_installed(name: str) -> bool:
        """True if the spaCy pipeline is importable as a package.

        This check is not an optimization — it is load-bearing. Presidio's
        ``NlpEngineProvider`` reacts to a missing model by trying to *download* it,
        which shells out to the package manager, blocks for minutes, and on this
        machine installed the model into a different project's virtualenv before
        failing anyway. Candidates are therefore filtered before Presidio ever sees
        them, so a size that is not present is skipped instantly and silently.
        """
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    def candidate_models(self) -> list[str]:
        """Installed pipelines to try, largest first.

        An explicit ``model`` is returned as given: it may be a filesystem path rather
        than a package, and the caller has said what they want.
        """
        if self.model:
            return [self.model]
        return [
            name
            for name in (self.profile.model_name(size) for size in MODEL_SIZES)
            if self.model_installed(name)
        ]

    def _build(self):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        nlp_configuration: dict = {"nlp_engine_name": "spacy"}
        if self.profile.label_map:
            # Merged over Presidio's defaults rather than replacing them, so a pipeline
            # that emits both tagsets keeps the standard labels working.
            from presidio_analyzer.nlp_engine import NerModelConfiguration

            merged = dict(NerModelConfiguration().model_to_presidio_entity_mapping)
            merged.update(self.profile.label_map)
            nlp_configuration["ner_model_configuration"] = {
                "model_to_presidio_entity_mapping": merged
            }

        errors = []
        for model_name in self.candidate_models():
            try:
                provider = NlpEngineProvider(nlp_configuration={
                    **nlp_configuration,
                    "models": [{"lang_code": self.language, "model_name": model_name}],
                })
                engine = provider.create_engine()
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
                continue
            self._loaded_model = model_name
            return AnalyzerEngine(nlp_engine=engine, supported_languages=[self.language])

        tried = "; ".join(errors) or "no pipeline installed for this language"
        raise NerUnavailableError(
            f"no spaCy pipeline loadable for language {self.language!r} ({tried}). "
            f"Install one with: python -m spacy download {self.profile.model_name('lg')}"
        )

    @property
    def analyzer(self):
        if self._analyzer is None:
            with self._lock:
                if self._analyzer is None:
                    if not self.available():
                        raise NerUnavailableError(
                            "presidio-analyzer is not installed; "
                            'install the NER extra: pip install "pii-shield[ner]"'
                        )
                    self._analyzer = self._build()
        return self._analyzer

    def warm(self) -> None:
        """Force the model load now, so the first real request does not pay for it."""
        _ = self.analyzer

    def detect(self, text: str, entities: set[str], threshold: float) -> list[Finding]:
        wanted = sorted(entities & self.supported_entities)
        if not text or not wanted:
            return []
        results = self.analyzer.analyze(
            text=text, entities=wanted, language=self.language, score_threshold=threshold
        )
        return [
            Finding(
                entity=r.entity_type,
                start=r.start,
                end=r.end,
                score=float(r.score),
                action=Action.SURROGATE,  # replaced by the caller's policy
                recognizer="presidio",
            )
            for r in results
        ]
