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
import logging
import subprocess
import sys
import threading

from ..languages import (
    GLOBAL_NER_ENTITIES,
    NER_MODEL_ENTITIES,
    LanguageProfile,
    get_profile,
)
from ..types import Action, Finding

# Entities answered by the NER pipeline and Presidio's own recognizers. Structured
# national identifiers and credentials are handled by the pattern layers, which are
# faster and more precise for those types.
NER_ENTITIES = frozenset(GLOBAL_NER_ENTITIES)

# Model sizes tried in order. ``lg`` first because the small pipelines are materially
# worse at person names, which is the entity that matters most here; ``sm`` last so a
# CI job or a laptop can still run with a small download.
MODEL_SIZES = ("lg", "md", "sm")

logger = logging.getLogger(__name__)

# Where pipelines are fetched from when downloading is switched on. Pinned to the
# publisher's own release host rather than an arbitrary index: installing a model is
# installing code, and this is the one place this package will ever do that.
MODEL_RELEASE_URL = (
    "https://github.com/explosion/spacy-models/releases/download/{name}-{version}/"
    "{name}-{version}-py3-none-any.whl"
)
DOWNLOAD_TIMEOUT_SECONDS = 900


class NerUnavailableError(RuntimeError):
    """Presidio or its language model could not be loaded.

    Distinct from a detection failure: this means the shield cannot do what the policy
    asked for at all, which is a configuration problem to surface at startup rather
    than a per-request error to retry.
    """


class PresidioDetector:
    """Thin adapter over Presidio's AnalyzerEngine, one per language."""

    def __init__(
        self,
        language: str = "ru",
        model: str | None = None,
        *,
        allow_blank: bool = False,
        download: bool = False,
        download_size: str = "lg",
    ) -> None:
        self.language = language
        self.profile: LanguageProfile = get_profile(language)
        self.model = model
        # A blank spaCy pipeline carries no NER component, but Presidio's pattern,
        # checksum and context recognizers do not need one. Allowing it is what makes
        # a deployment that wants email and card numbers — but cannot afford a 500 MB
        # language model — possible at all.
        self.allow_blank = allow_blank
        # Off by default, and deliberately so: the failure this replaced was Presidio
        # fetching a model by itself, into an environment it chose, blocking startup
        # for five minutes with no output. When switched on, the fetch happens here —
        # explicitly, into this interpreter, with a timeout and a log line.
        self.download = download
        self.download_size = download_size
        self._analyzer = None
        self._loaded_model: str | None = None
        self._blank = False
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
        """Everything this detector can be asked for, including national recognizers.

        A blank pipeline drops the four model-derived entities: claiming them while
        having no NER component would report clean text that was never examined.
        """
        entities = frozenset(self.profile.ner_entities)
        return entities - NER_MODEL_ENTITIES if self._blank else entities

    @property
    def ner_available(self) -> bool:
        """False when running on a blank pipeline, i.e. names are not being detected."""
        return self._analyzer is not None and not self._blank

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

    @staticmethod
    def model_release_version() -> str:
        """The spacy-models release matching the installed spaCy, e.g. '3.8.0'."""
        import spacy

        major, minor = spacy.__version__.split(".")[:2]
        return f"{major}.{minor}.0"

    def fetch_model(self, size: str | None = None) -> str | None:
        """Install a pipeline into *this* interpreter. Returns the name, or None.

        Uses ``sys.executable -m pip`` rather than ``spacy download``: that command
        resolves the target environment itself and has been observed installing into
        a different virtualenv than the running one.
        """
        name = self.profile.model_name(size or self.download_size)
        url = MODEL_RELEASE_URL.format(name=name, version=self.model_release_version())
        logger.warning("pii-shield: downloading language pipeline %s (this takes a while)", name)

        # Both installers are told which interpreter to target. Letting either pick
        # for itself is precisely the failure this replaced. ``uv venv`` creates
        # environments without pip, which is common enough to need the second path.
        commands = (
            [sys.executable, "-m", "pip", "install", "--no-input", url],
            ["uv", "pip", "install", "--python", sys.executable, url],
        )
        last_error = "no installer available"
        for command in commands:
            try:
                subprocess.run(
                    command, check=True, capture_output=True,
                    timeout=DOWNLOAD_TIMEOUT_SECONDS,
                )
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                last_error = f"{command[0]}: {exc}"
                continue
        else:
            logger.error("pii-shield: could not download %s (%s)", name, last_error)
            return None
        # A package installed after this process started is invisible to find_spec
        # until the import caches are dropped.
        importlib.invalidate_caches()
        if not self.model_installed(name):
            logger.error("pii-shield: %s installed but is not importable", name)
            return None
        logger.warning("pii-shield: %s is ready", name)
        return name

    def candidate_models(self) -> list[str]:
        """Installed pipelines to try, largest first.

        An explicit ``model`` is returned as given: it may be a filesystem path rather
        than a package, and the caller has said what they want.
        """
        if self.model:
            return [self.model]
        installed = [
            name
            for name in (self.profile.model_name(size) for size in MODEL_SIZES)
            if self.model_installed(name)
        ]
        if installed or not self.download:
            return installed
        fetched = self.fetch_model()
        return [fetched] if fetched else []

    def planned_model(self) -> str | None:
        """The pipeline a load would choose, without loading it."""
        candidates = self.candidate_models()
        return candidates[0] if candidates else None

    def _build(self):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        from .context_enhancer import build_context_enhancer

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
            self._drop_unused_components(engine)
            return AnalyzerEngine(
                nlp_engine=engine,
                supported_languages=[self.language],
                context_aware_enhancer=build_context_enhancer(),
            )

        if self.allow_blank:
            # No language model, but Presidio is importable: run its pattern
            # recognizers on an empty pipeline. spaCy ships every language class, so
            # this costs megabytes rather than hundreds of them.
            engine = self._blank_engine()
            self._blank = True
            self._loaded_model = f"blank:{self.language}"
            return AnalyzerEngine(
                nlp_engine=engine,
                supported_languages=[self.language],
                context_aware_enhancer=build_context_enhancer(),
            )

        tried = "; ".join(errors) or "no pipeline installed for this language"
        wheel = MODEL_RELEASE_URL.format(
            name=self.profile.model_name("lg"), version=self.model_release_version()
        )
        raise NerUnavailableError(
            f"no spaCy pipeline loadable for language {self.language!r} ({tried}). "
            f"Install one with: pip install {wheel}"
        )

    # Components Presidio never reads. It asks the pipeline for tokens, lemmas and
    # entities, and the dependency parse feeds none of them, while costing roughly a
    # tenth of the running time on a large payload.
    #
    # The attribute ruler looks equally unused and is not: English lemmatization is
    # rule-based and reads the POS tags it assigns, so dropping it made spaCy warn
    # (W108) and quietly degraded the lemmas that Presidio scores context by. Russian
    # gets its tags from the statistical morphologizer and would not have noticed —
    # which is exactly how a change like this goes unnoticed in the wrong language.
    _UNUSED_COMPONENTS = ("parser", "senter")

    def _drop_unused_components(self, engine) -> None:
        """Strip pipeline components nothing downstream consumes."""
        pipelines = getattr(engine, "nlp", None)
        pipeline = pipelines.get(self.language) if isinstance(pipelines, dict) else None
        if pipeline is None:
            return
        for name in self._UNUSED_COMPONENTS:
            if name in getattr(pipeline, "pipe_names", ()):
                try:
                    pipeline.remove_pipe(name)
                except Exception:  # pragma: no cover - a pipeline that refuses keeps it
                    logger.debug("pii-shield: could not drop %r from %s", name, self.language)

    def _blank_engine(self):
        """A Presidio NLP engine backed by ``spacy.blank`` — tokenizer only, no NER."""
        import spacy
        from presidio_analyzer.nlp_engine import SpacyNlpEngine

        language = self.language

        class _BlankSpacyEngine(SpacyNlpEngine):
            def __init__(self) -> None:
                super().__init__(models=[{"lang_code": language, "model_name": "blank"}])
                self.nlp = {language: spacy.blank(language)}

            def load(self) -> None:  # nothing to load
                return None

        engine = _BlankSpacyEngine()
        engine.load()
        return engine

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

    def person_label_reachable(self) -> bool:
        """True if some label this pipeline emits maps to PERSON.

        A pipeline whose tagset Presidio does not know finds names and then discards
        every one of them, reporting clean text. Korean and Swedish both did exactly
        that. Rather than chase tagsets one language at a time, the mapping is checked
        against the labels the loaded pipeline actually declares.
        """
        if self._blank:
            return False
        try:
            from presidio_analyzer.nlp_engine import NerModelConfiguration

            mapping = dict(NerModelConfiguration().model_to_presidio_entity_mapping)
            mapping.update(self.profile.label_map)
            nlp = self.analyzer.nlp_engine.nlp[self.language]
            labels = set(nlp.get_pipe("ner").labels)
        except Exception:
            # Cannot introspect — do not turn an unknown into a refusal.
            return True
        return any(mapping.get(label) == "PERSON" for label in labels)

    def warm(self, require_person: bool = False) -> None:
        """Force the model load now, so the first real request does not pay for it.

        With *require_person*, also refuse a pipeline whose labels cannot produce a
        PERSON: loading it would look like success and detect no names at all.
        """
        _ = self.analyzer
        if require_person and not self.person_label_reachable():
            raise NerUnavailableError(
                f"the pipeline loaded for {self.language!r} ({self._loaded_model}) emits no "
                "label that maps to PERSON, so names would not be detected. Add a mapping "
                "for its tagset in pii_shield.languages._LABEL_MAPS."
            )

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
